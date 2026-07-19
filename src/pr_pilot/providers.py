from __future__ import annotations

import json
import os
import shutil
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path

from .chatgpt import run_chatgpt_agent
from .commands import run
from .config import ProviderConfig
from .errors import AgentShipError


LIMIT_MARKERS = (
    "rate limit", "usage limit", "quota exceeded", "too many requests",
    "http 429", "status 429", "limit reached", "try again after",
)


class AgentProvider(ABC):
    @abstractmethod
    def invoke(self, prompt: str, *, repo: Path, write: bool) -> str:
        raise NotImplementedError


class CodexProvider(AgentProvider):
    def __init__(self, config: ProviderConfig):
        self.config = config

    def invoke(self, prompt: str, *, repo: Path, write: bool) -> str:
        command = [
            "codex",
            "exec",
            "--sandbox",
            "workspace-write" if write else "read-only",
        ]
        if self.config.model:
            command.extend(["--model", self.config.model])
        command.append(prompt)
        return run(command, cwd=repo).stdout.strip()


class CursorProvider(AgentProvider):
    def __init__(self, config: ProviderConfig):
        self.config = config

    def invoke(self, prompt: str, *, repo: Path, write: bool) -> str:
        command = ["cursor-agent", "--print", "--output-format", "json"]
        if write:
            command.append("--force")
        if self.config.model:
            command.extend(["--model", self.config.model])
        command.append(prompt)
        output = run(command, cwd=repo).stdout.strip()
        try:
            payload = json.loads(output)
            return str(payload["result"]).strip()
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            raise AgentShipError("Cursor returned an unexpected response") from exc


class ChatGptProvider(AgentProvider):
    """chatgpt.com's backend over HTTP, reusing the Codex CLI's OAuth tokens.

    The backend has no native tools, so file work is emulated by an agent loop
    (:func:`run_chatgpt_agent`): the model inspects the repo through read ops
    and, for writing roles, edits it through write/delete ops. Serves every
    role — planner and reviewer with writes disabled, implementer with them on.
    """

    def __init__(self, config: ProviderConfig):
        self.config = config

    def invoke(self, prompt: str, *, repo: Path, write: bool) -> str:
        return run_chatgpt_agent(
            prompt, repo, allow_writes=write,
            model=self.config.model, effort=self.config.reasoning_effort,
        )


# Default provider/model slug for opencode — an OpenAI-compatible provider named
# `chatgpt-proxy` in ~/.config/opencode/opencode.json, pointed at the proxy.
_DEFAULT_OPENCODE_MODEL = "chatgpt-proxy/gpt-5.5"


def _opencode_bin() -> str:
    """Locate the opencode binary: $OPENCODE_BIN, then PATH, then the installer's
    default (~/.opencode/bin) — the last matters under systemd, whose PATH won't
    include the per-user install dir."""
    override = os.environ.get("OPENCODE_BIN")
    if override:
        return override
    found = shutil.which("opencode")
    if found:
        return found
    fallback = Path.home() / ".opencode" / "bin" / "opencode"
    return str(fallback) if fallback.exists() else "opencode"


def _opencode_text(output: str) -> str:
    """The assistant's final message from an ``opencode run --format json`` stream.

    The stream is JSONL grouped into steps (``step_start`` events). opencode
    narrates as it works — emitting ``text`` parts in the earlier tool-using
    steps — and gives the actual answer in the final step. Return that final
    step's text (joining its parts), skipping the thinking, so a designer's
    feature request or a reviewer's findings+verdict come back clean and don't
    leak into branch names, PR titles, or the repair prompt. With no step markers
    at all, every text is one step — i.e. the old join-everything behavior.
    """
    steps: list[list[str]] = []
    order: list[str] = []
    texts: dict[str, str] = {}
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "step_start":
            if order:
                steps.append([texts[pid] for pid in order])
            order, texts = [], {}
        elif kind == "text":
            part = event.get("part") or {}
            pid, text = part.get("id"), part.get("text")
            if isinstance(pid, str) and isinstance(text, str):
                if pid not in texts:
                    order.append(pid)
                texts[pid] = text
    if order:
        steps.append([texts[pid] for pid in order])
    for parts in reversed(steps):
        answer = "\n\n".join(parts).strip()
        if answer:
            return answer
    return ""


class OpencodeProvider(AgentProvider):
    """Drive the opencode CLI headless, pointed at an OpenAI-compatible endpoint.

    opencode brings its own agent loop and native tool calling, so pr-pilot just
    hands it a prompt and reads the final message. Writing roles run the default
    build agent with ``--auto`` (auto-approve edits/commands); read-only roles run
    the built-in read-only ``plan`` agent, so plan/review/designer never touch the
    tree. The model is a ``provider/model`` slug (default ``chatgpt-proxy/gpt-5.5``,
    the proxy provider defined in opencode's config).
    """

    def __init__(self, config: ProviderConfig):
        self.config = config

    def invoke(self, prompt: str, *, repo: Path, write: bool) -> str:
        command = [
            _opencode_bin(), "run", "--format", "json",
            # Pin the project dir explicitly: opencode resolves it from $PWD, which
            # subprocess cwd= does NOT set, so without this it would inspect the
            # caller's directory (e.g. pr-pilot's own) instead of the target repo.
            "--dir", str(repo),
            "--model", self.config.model or _DEFAULT_OPENCODE_MODEL,
        ]
        if write:
            command.append("--auto")          # build agent, auto-approve edits
        else:
            command.extend(["--agent", "plan"])  # read-only agent
        if self.config.reasoning_effort:
            command.extend(["--variant", self.config.reasoning_effort])
        command.append(prompt)
        output = run(command, cwd=repo).stdout
        text = _opencode_text(output)
        if not text:
            raise AgentShipError("opencode returned no assistant message")
        return text


class LimitRetryProvider(AgentProvider):
    def __init__(
        self,
        provider: AgentProvider,
        config: ProviderConfig,
        *,
        sleeper=time.sleep,
        clock=time.monotonic,
        notifier=None,
    ):
        self.provider = provider
        self.poll_seconds = config.limit_poll_seconds
        self.max_wait_seconds = config.limit_max_wait_seconds
        self.sleeper = sleeper
        self.clock = clock
        self.notifier = notifier or (lambda message: print(message, file=sys.stderr, flush=True))

    def invoke(self, prompt: str, *, repo: Path, write: bool) -> str:
        started = self.clock()
        attempt = 0
        while True:
            try:
                return self.provider.invoke(prompt, repo=repo, write=write)
            except AgentShipError as exc:
                if not self._is_limit_error(str(exc)):
                    raise
                waited = self.clock() - started
                if self.max_wait_seconds and waited + self.poll_seconds > self.max_wait_seconds:
                    raise AgentShipError(
                        f"Provider limit did not clear within {self.max_wait_seconds} seconds"
                    ) from exc
                attempt += 1
                self.notifier(
                    f"Provider limit reached; retrying in {self.poll_seconds}s "
                    f"(attempt {attempt}). Press Ctrl-C to stop."
                )
                self.sleeper(self.poll_seconds)

    @staticmethod
    def _is_limit_error(message: str) -> bool:
        lowered = message.lower()
        return any(marker in lowered for marker in LIMIT_MARKERS)


def make_provider(config: ProviderConfig) -> AgentProvider:
    if config.name == "codex":
        provider = CodexProvider(config)
    elif config.name == "cursor":
        provider = CursorProvider(config)
    elif config.name == "chatgpt":
        provider = ChatGptProvider(config)
    elif config.name == "opencode":
        provider = OpencodeProvider(config)
    else:
        raise AgentShipError(f"Unknown provider: {config.name}")
    return LimitRetryProvider(provider, config)
