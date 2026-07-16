from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path

from .commands import run
from .config import ProviderConfig
from .errors import AgentShipError


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


def make_provider(config: ProviderConfig) -> AgentProvider:
    if config.name == "codex":
        return CodexProvider(config)
    if config.name == "cursor":
        return CursorProvider(config)
    raise AgentShipError(f"Unknown provider: {config.name}")

