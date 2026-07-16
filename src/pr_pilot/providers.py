from __future__ import annotations

import json
import sys
import time
from abc import ABC, abstractmethod
from pathlib import Path

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
    else:
        raise AgentShipError(f"Unknown provider: {config.name}")
    return LimitRetryProvider(provider, config)
