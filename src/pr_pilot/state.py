from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass
class RunState:
    run_id: str
    feature: str
    repo: str
    branch: str = ""
    phase: str = "created"
    plan: str = ""
    review: str = ""
    pr_url: str = ""
    handled_feedback: list[str] = field(default_factory=list)
    fix_attempts: int = 0


class StateStore:
    def __init__(self, root: Path):
        self.root = root

    def save(self, state: RunState) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        target = self.root / f"{state.run_id}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(asdict(state), indent=2) + "\n")
        temporary.replace(target)

    def load(self, run_id: str) -> RunState:
        payload = json.loads((self.root / f"{run_id}.json").read_text())
        return RunState(**payload)

