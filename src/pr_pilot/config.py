from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

from .errors import AgentShipError


@dataclass(frozen=True)
class ProviderConfig:
    name: str = "codex"
    model: str | None = None
    limit_poll_seconds: int = 60
    limit_max_wait_seconds: int = 0


@dataclass(frozen=True)
class GitHubConfig:
    base_branch: str = "main"
    draft: bool = True


@dataclass(frozen=True)
class BabysitConfig:
    enabled: bool = True
    interval_seconds: int = 30
    max_cycles: int = 60
    max_fix_attempts: int = 3
    require_approval: bool = False


@dataclass(frozen=True)
class TelegramConfig:
    token_env: str = "TELEGRAM_BOT_TOKEN"
    allowed_chat_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class MemoryConfig:
    enabled: bool = True
    database: Path = Path("~/.pr-pilot/memory.db").expanduser()
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    profile_provider: str = "implementer"
    profile_model: str | None = None
    max_file_bytes: int = 524_288
    chunk_chars: int = 1_600
    chunk_overlap: int = 200
    context_results: int = 10
    context_chars: int = 12_000
    relationship_depth: int = 1
    relationship_threshold: float = 0.80


@dataclass(frozen=True)
class Config:
    repo: Path
    implementer: ProviderConfig = field(default_factory=ProviderConfig)
    reviewer: ProviderConfig = field(default_factory=ProviderConfig)
    github: GitHubConfig = field(default_factory=GitHubConfig)
    babysit: BabysitConfig = field(default_factory=BabysitConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    state_dir: Path = Path("~/.pr-pilot").expanduser()

    def with_repo(self, repo: Path) -> "Config":
        return replace(self, repo=repo.resolve())


def _provider(data: dict, default: str = "codex") -> ProviderConfig:
    return ProviderConfig(
        name=str(data.get("name", default)),
        model=data.get("model"),
        limit_poll_seconds=int(data.get("limit_poll_seconds", 60)),
        limit_max_wait_seconds=int(data.get("limit_max_wait_seconds", 0)),
    )


def load_config(path: Path | None = None, repo: Path | None = None) -> Config:
    path = path or Path("pr-pilot.toml")
    data: dict = {}
    if path.exists():
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    elif path != Path("pr-pilot.toml"):
        raise AgentShipError(f"Configuration file does not exist: {path}")

    configured_repo = repo or Path(data.get("repo", "."))
    gh = data.get("github", {})
    baby = data.get("babysit", {})
    telegram = data.get("telegram", {})
    memory = data.get("memory", {})
    config = Config(
        repo=configured_repo.resolve(),
        implementer=_provider(data.get("implementer", {})),
        reviewer=_provider(data.get("reviewer", {})),
        github=GitHubConfig(
            base_branch=str(gh.get("base_branch", "main")),
            draft=bool(gh.get("draft", True)),
        ),
        babysit=BabysitConfig(
            enabled=bool(baby.get("enabled", True)),
            interval_seconds=int(baby.get("interval_seconds", 30)),
            max_cycles=int(baby.get("max_cycles", 60)),
            max_fix_attempts=int(baby.get("max_fix_attempts", 3)),
            require_approval=bool(baby.get("require_approval", False)),
        ),
        telegram=TelegramConfig(
            token_env=str(telegram.get("token_env", "TELEGRAM_BOT_TOKEN")),
            allowed_chat_ids=tuple(int(item) for item in telegram.get("allowed_chat_ids", [])),
        ),
        memory=MemoryConfig(
            enabled=bool(memory.get("enabled", True)),
            database=Path(memory.get("database", "~/.pr-pilot/memory.db")).expanduser(),
            embedding_model=str(memory.get("embedding_model", "BAAI/bge-small-en-v1.5")),
            profile_provider=str(memory.get("profile_provider", "implementer")),
            profile_model=memory.get("profile_model"),
            max_file_bytes=int(memory.get("max_file_bytes", 524_288)),
            chunk_chars=int(memory.get("chunk_chars", 1_600)),
            chunk_overlap=int(memory.get("chunk_overlap", 200)),
            context_results=int(memory.get("context_results", 10)),
            context_chars=int(memory.get("context_chars", 12_000)),
            relationship_depth=int(memory.get("relationship_depth", 1)),
            relationship_threshold=float(memory.get("relationship_threshold", 0.80)),
        ),
        state_dir=Path(data.get("state_dir", "~/.pr-pilot")).expanduser(),
    )
    if config.implementer.name not in {"codex", "cursor"}:
        raise AgentShipError("implementer.name must be 'codex' or 'cursor'")
    if config.reviewer.name not in {"codex", "cursor"}:
        raise AgentShipError("reviewer.name must be 'codex' or 'cursor'")
    for provider in (config.implementer, config.reviewer):
        if provider.limit_poll_seconds <= 0 or provider.limit_max_wait_seconds < 0:
            raise AgentShipError(
                "provider limit_poll_seconds must be positive and limit_max_wait_seconds cannot be negative"
            )
    if config.memory.profile_provider not in {"implementer", "reviewer", "codex", "cursor"}:
        raise AgentShipError(
            "memory.profile_provider must be implementer, reviewer, codex, or cursor"
        )
    if config.memory.chunk_overlap >= config.memory.chunk_chars:
        raise AgentShipError("memory.chunk_overlap must be smaller than memory.chunk_chars")
    if config.memory.chunk_chars <= 0 or config.memory.chunk_overlap < 0:
        raise AgentShipError("memory chunk sizes must be positive")
    if config.memory.max_file_bytes <= 0 or config.memory.context_results <= 0:
        raise AgentShipError("memory file and result limits must be positive")
    if config.memory.context_chars <= 0 or config.memory.relationship_depth < 0:
        raise AgentShipError("memory context limit must be positive and depth cannot be negative")
    if not 0.0 <= config.memory.relationship_threshold <= 1.0:
        raise AgentShipError("memory.relationship_threshold must be between 0 and 1")
    if not config.repo.is_dir():
        raise AgentShipError(f"Repository directory does not exist: {config.repo}")
    if not os.access(config.repo, os.W_OK):
        raise AgentShipError(f"Repository is not writable: {config.repo}")
    return config
