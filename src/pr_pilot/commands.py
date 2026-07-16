from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .errors import AgentShipError


@dataclass(frozen=True)
class Result:
    stdout: str
    stderr: str
    returncode: int


def run(
    command: Sequence[str],
    *,
    cwd: Path,
    env: Mapping[str, str] | None = None,
    check: bool = True,
) -> Result:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise AgentShipError(f"Required command not found: {command[0]}") from exc
    result = Result(completed.stdout, completed.stderr, completed.returncode)
    if check and completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise AgentShipError(
            f"Command failed ({completed.returncode}): {command[0]}"
            + (f"\n{detail}" if detail else "")
        )
    return result
