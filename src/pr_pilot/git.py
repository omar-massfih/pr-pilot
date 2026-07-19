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

    def default_branch(self) -> str | None:
        """The remote's default branch (``origin/HEAD`` target), or None if unset.

        Lets one instance serve repos with different defaults (e.g. a `master`
        frontend and a `main` backend) without a per-repo base setting. `git
        clone` records origin/HEAD; None falls the caller back to config.
        """
        result = run(
            ["git", "symbolic-ref", "--quiet", "refs/remotes/origin/HEAD"],
            cwd=self.path,
            check=False,
        )
        ref = result.stdout.strip()
        prefix = "refs/remotes/origin/"
        return ref[len(prefix):] if ref.startswith(prefix) else None

    @staticmethod
    def branch_name(feature: str) -> str:
        slug = re.sub(r"[^a-z0-9]+", "-", feature.lower()).strip("-")[:42] or "feature"
        stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
        return f"agent/{slug}-{stamp}"

    def start_branch(self, branch: str, base: str) -> None:
        """Create ``branch`` off a fresh ``base`` (shared across a repo group so
        every member's PR rides the same branch name)."""
        self.checkout_base(base)
        run(["git", "switch", "-c", branch], cwd=self.path)

    def create_branch(self, feature: str, base: str) -> str:
        branch = self.branch_name(feature)
        self.start_branch(branch, base)
        return branch

    def checkout_base(self, base: str) -> None:
        run(["git", "switch", base], cwd=self.path)
        run(["git", "pull", "--ff-only"], cwd=self.path)

    def has_changes(self) -> bool:
        return bool(self.status())

    def reset_to_base(self, base: str) -> None:
        """Discard all working-tree changes and return to a clean base branch.

        A run that stops before publishing (review limit, agent error) leaves
        uncommitted edits behind, which would block every later run's clean-tree
        check. Resetting here lets an autonomous loop recover instead of wedging.
        Nothing is pushed until publishing, so only throwaway local work is lost.
        """
        run(["git", "reset", "--hard"], cwd=self.path)
        run(["git", "clean", "-fd"], cwd=self.path)
        self.checkout_base(base)

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
