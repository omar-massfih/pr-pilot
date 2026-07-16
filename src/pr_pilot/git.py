from __future__ import annotations

import hashlib
import re
from datetime import UTC, datetime
from pathlib import Path

from .commands import run
from .errors import AgentShipError


class GitRepo:
    def __init__(self, path: Path):
        self.path = path

    def validate(self) -> None:
        run(["git", "rev-parse", "--show-toplevel"], cwd=self.path)
        if self.status():
            raise AgentShipError("Repository has uncommitted changes; start from a clean worktree")

    def status(self) -> str:
        return run(["git", "status", "--porcelain"], cwd=self.path).stdout.strip()

    def create_branch(self, feature: str, base: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", feature.lower()).strip("-")[:42] or "feature"
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        branch = f"agent/{slug}-{stamp}"
        run(["git", "switch", base], cwd=self.path)
        run(["git", "pull", "--ff-only"], cwd=self.path)
        run(["git", "switch", "-c", branch], cwd=self.path)
        return branch

    def has_changes(self) -> bool:
        return bool(self.status())

    def fingerprint(self) -> str:
        """Hash tracked diffs and untracked content to detect writes by read-only agents."""
        digest = hashlib.sha256()
        for command in (
            ["git", "status", "--porcelain=v1", "-z"],
            ["git", "diff", "--binary"],
            ["git", "diff", "--cached", "--binary"],
        ):
            digest.update(run(command, cwd=self.path).stdout.encode())
        untracked = run(
            ["git", "ls-files", "--others", "--exclude-standard", "-z"], cwd=self.path
        ).stdout.split("\0")
        for relative in sorted(item for item in untracked if item):
            path = self.path / relative
            digest.update(relative.encode())
            if path.is_file():
                digest.update(path.read_bytes())
        return digest.hexdigest()

    def commit(self, message: str) -> None:
        run(["git", "add", "-A"], cwd=self.path)
        run(["git", "commit", "-m", message], cwd=self.path)

    def push(self, branch: str) -> None:
        run(["git", "push", "--set-upstream", "origin", branch], cwd=self.path)

    def head(self) -> str:
        return run(["git", "rev-parse", "HEAD"], cwd=self.path).stdout.strip()
