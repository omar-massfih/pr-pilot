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
    timeout: float | None = None,
) -> Result:
    try:
        completed = subprocess.run(
            list(command),
            cwd=cwd,
            env=env,
            text=True,
            capture_output=True,
            check=False,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise AgentShipError(f"Required command not found: {command[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        detail = f"Command timed out after {timeout}s: {command[0]}"
        if check:
            raise AgentShipError(detail) from exc
        # A verification gate (check=False) treats a timeout as a failing run,
        # keeping any partial output the command produced before it was killed.
        return Result(exc.stdout or "", (exc.stderr or "") + "\n" + detail, 124)
    result = Result(completed.stdout, completed.stderr, completed.returncode)
    if check and completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        raise AgentShipError(
            f"Command failed ({completed.returncode}): {command[0]}"
            + (f"\n{detail}" if detail else "")
        )
    return result
