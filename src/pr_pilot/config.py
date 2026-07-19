from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

from .errors import AgentShipError

# Recognized agent-provider backends (see :mod:`pr_pilot.providers`).
_PROVIDER_NAMES = frozenset({"codex", "cursor", "chatgpt", "opencode"})


@dataclass(frozen=True)
class ProviderConfig:
    name: str = "codex"
    model: str | None = None
    limit_poll_seconds: int = 60
    limit_max_wait_seconds: int = 0
    # chatgpt provider only: reasoning effort per round. Lower is faster;
    # "low" is the default for responsiveness (higher = slower, more thorough).
    reasoning_effort: str = "low"


@dataclass(frozen=True)
class GitHubConfig:
    base_branch: str = "main"
    draft: bool = True


@dataclass(frozen=True)
class WorkflowConfig:
    max_review_attempts: int = 3


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
    # Proposes the next feature in the /auto loop. Defaults to the implementer's
    # config when no [designer] table is given, so ideation runs at the same
    # model/effort without extra setup; override it to design on a different one.
    designer: ProviderConfig = field(default_factory=ProviderConfig)
    github: GitHubConfig = field(default_factory=GitHubConfig)
    workflow: WorkflowConfig = field(default_factory=WorkflowConfig)
    babysit: BabysitConfig = field(default_factory=BabysitConfig)
    telegram: TelegramConfig = field(default_factory=TelegramConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    state_dir: Path = Path("~/.pr-pilot").expanduser()
    # Named repos this instance can target (name -> path). One Telegram bot can
    # drive them all, switching with /repo; ``repo`` is the default (first). With
    # no [repos] table it is just {"main": repo}, so single-repo setups are
    # unchanged.
    repos: dict[str, Path] = field(default_factory=dict)
    # When set, the [repos] are one *group* worked together: the agent runs in
    # this workspace dir (which must contain every member repo) so it sees and
    # edits them all at once, and a PR is opened per member repo that changed.
    # Unset => each repo is an independent /repo-switchable target.
    workspace: Path | None = None

    def with_repo(self, repo: Path) -> "Config":
        return replace(self, repo=repo.resolve())


def _provider(data: dict, default: str = "codex") -> ProviderConfig:
    return ProviderConfig(
        name=str(data.get("name", default)),
        model=data.get("model"),
        limit_poll_seconds=int(data.get("limit_poll_seconds", 60)),
        limit_max_wait_seconds=int(data.get("limit_max_wait_seconds", 0)),
        reasoning_effort=str(data.get("reasoning_effort", "low")),
    )


def load_config(path: Path | None = None, repo: Path | None = None) -> Config:
    path = path or Path("pr-pilot.toml")
    data: dict = {}
    if path.exists():
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    elif path != Path("pr-pilot.toml"):
        raise AgentShipError(f"Configuration file does not exist: {path}")

    # Named repos from an optional [repos] table; the primary/default repo is the
    # --repo override, else [repo], else the first [repos] entry, else ".".
    repos_raw = data.get("repos", {})
    repos = {
        str(name): Path(str(target)).expanduser().resolve()
        for name, target in (repos_raw.items() if isinstance(repos_raw, dict) else [])
    }
    if repo is not None:
        primary = repo.resolve()
    elif "repo" in data:
        primary = Path(str(data["repo"])).expanduser().resolve()
    elif repos:
        primary = next(iter(repos.values()))
    else:
        primary = Path(".").resolve()
    if not repos:
        repos = {"main": primary}
    configured_repo = primary
    gh = data.get("github", {})
    workflow = data.get("workflow", {})
    baby = data.get("babysit", {})
    telegram = data.get("telegram", {})
    memory = data.get("memory", {})
    config = Config(
        repo=configured_repo.resolve(),
        implementer=_provider(data.get("implementer", {})),
        reviewer=_provider(data.get("reviewer", {})),
        # Inherit the implementer's table when [designer] is absent.
        designer=_provider(data.get("designer", data.get("implementer", {}))),
        github=GitHubConfig(
            base_branch=str(gh.get("base_branch", "main")),
            draft=bool(gh.get("draft", True)),
        ),
        workflow=WorkflowConfig(
            max_review_attempts=int(workflow.get("max_review_attempts", 3)),
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
        repos=repos,
        workspace=(
            Path(str(data["workspace"])).expanduser().resolve()
            if data.get("workspace") else None
        ),
    )
    for role in ("implementer", "reviewer", "designer"):
        name = getattr(config, role).name
        if name not in _PROVIDER_NAMES:
            raise AgentShipError(
                f"{role}.name must be one of {', '.join(sorted(_PROVIDER_NAMES))}"
            )
    if config.workflow.max_review_attempts < 0:
        raise AgentShipError("workflow.max_review_attempts cannot be negative")
    for provider in (config.implementer, config.reviewer, config.designer):
        if provider.limit_poll_seconds <= 0 or provider.limit_max_wait_seconds < 0:
            raise AgentShipError(
                "provider limit_poll_seconds must be positive and limit_max_wait_seconds cannot be negative"
            )
        if provider.reasoning_effort not in {"minimal", "low", "medium", "high"}:
            raise AgentShipError(
                "provider reasoning_effort must be minimal, low, medium, or high"
            )
    if config.memory.profile_provider not in {"implementer", "reviewer", *_PROVIDER_NAMES}:
        raise AgentShipError(
            "memory.profile_provider must be 'implementer', 'reviewer', or a "
            f"provider name ({', '.join(sorted(_PROVIDER_NAMES))})"
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
    for name, target in config.repos.items():
        if not target.is_dir():
            raise AgentShipError(f"Repository '{name}' directory does not exist: {target}")
        if not os.access(target, os.W_OK):
            raise AgentShipError(f"Repository '{name}' is not writable: {target}")
    if config.workspace is not None:
        if not config.workspace.is_dir():
            raise AgentShipError(f"workspace directory does not exist: {config.workspace}")
        for name, target in config.repos.items():
            if config.workspace != target and config.workspace not in target.parents:
                raise AgentShipError(
                    f"repo '{name}' ({target}) is not inside workspace {config.workspace}"
                )
    return config
