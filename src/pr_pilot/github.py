from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .commands import run


@dataclass(frozen=True)
class PullRequestStatus:
    url: str
    review_decision: str
    pending_checks: tuple[str, ...]
    failed_checks: tuple[str, ...]
    feedback: tuple[str, ...]


class GitHub:
    def __init__(self, repo: Path):
        self.repo = repo

    def create_pr(self, *, title: str, body: str, base: str, draft: bool) -> str:
        command = ["gh", "pr", "create", "--title", title, "--body", body, "--base", base]
        if draft:
            command.append("--draft")
        return run(command, cwd=self.repo).stdout.strip()

    def mark_ready(self) -> None:
        run(["gh", "pr", "ready"], cwd=self.repo)

    def status(self) -> PullRequestStatus:
        fields = "url,reviewDecision,comments,reviews,statusCheckRollup"
        payload = json.loads(
            run(["gh", "pr", "view", "--json", fields], cwd=self.repo).stdout
        )
        pending: list[str] = []
        failed: list[str] = []
        for check in payload.get("statusCheckRollup") or []:
            name = check.get("name") or check.get("context") or "unnamed check"
            state = str(check.get("conclusion") or check.get("state") or "").upper()
            if state in {"FAILURE", "ERROR", "CANCELLED", "TIMED_OUT", "ACTION_REQUIRED"}:
                failed.append(name)
            elif state not in {"SUCCESS", "NEUTRAL", "SKIPPED"}:
                pending.append(name)
        feedback: list[str] = []
        for item in (payload.get("reviews") or []) + (payload.get("comments") or []):
            body = str(item.get("body") or "").strip()
            if body:
                author = (item.get("author") or {}).get("login", "reviewer")
                state = item.get("state", "COMMENT")
                feedback.append(f"[{author} / {state}] {body}")
        return PullRequestStatus(
            url=str(payload["url"]),
            review_decision=str(payload.get("reviewDecision") or ""),
            pending_checks=tuple(pending),
            failed_checks=tuple(failed),
            feedback=tuple(feedback),
        )

