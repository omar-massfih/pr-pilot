"""Chat via chatgpt.com's backend Responses endpoint, stdlib-only.

Reuses the ChatGPT OAuth tokens the Codex CLI keeps in ``$CODEX_HOME/auth.json``
(``codex login``), so there is no API key and usage is billed to the ChatGPT
plan. The endpoint is the same private one the Codex CLI itself drives —
unofficial, so everything that touches it lives in this one module.

Two entry points:

- :func:`run_chatgpt` — one plain text turn (memory profiling, simple asks).
- :func:`run_chatgpt_agent` — an agent loop that lets the model inspect and
  (optionally) edit a repository. The backend has no native tools, so the
  model drives the repo through a fenced ``actions`` block that this module
  parses and executes, feeding results back until the model answers with no
  further actions. With ``allow_writes=False`` only read ops run (planner,
  reviewer); with ``allow_writes=True`` it can also edit files (implementer).

Failures raise :class:`AgentShipError`; HTTP status codes are kept in the
message so ``LimitRetryProvider`` recognizes 429s as limit errors.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterable, Iterator

from .errors import AgentShipError

logger = logging.getLogger(__name__)

_TOKEN_URL = "https://auth.openai.com/oauth/token"
# The Codex CLI's public OAuth client id (baked into the open-source CLI).
_CLIENT_ID = "app_EMoamEEZ73f0CkXaXp7hrann"
_RESPONSES_URL = "https://chatgpt.com/backend-api/codex/responses"
# The HTTP-SSE contract; newer codex builds use a websocket variant instead.
_OPENAI_BETA = "responses=experimental"
_INSTRUCTIONS = "You are a helpful assistant."
# Only codex-supported model ids are accepted (anything else is an HTTP 400).
DEFAULT_MODEL = "gpt-5.5"
_TIMEOUT_SECONDS = 600
_REFRESH_TIMEOUT_SECONDS = 30
# Refresh a little early so a token can't expire mid-request.
_EXPIRY_SKEW_SECONDS = 300

# Guards read-modify-write of auth.json across threads (refresh tokens are
# single-use, so a concurrent double-refresh would log one caller out).
_auth_lock = threading.Lock()


def _auth_path() -> Path:
    home = os.environ.get("CODEX_HOME")
    root = Path(home).expanduser() if home else Path.home() / ".codex"
    return root / "auth.json"


def _load_auth(path: Path) -> dict:
    try:
        auth = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise AgentShipError(
            f"No Codex auth file at {path} — run `codex login` first."
        ) from exc
    except (OSError, ValueError) as exc:
        raise AgentShipError(f"Could not read Codex auth file {path}: {exc}") from exc
    tokens = auth.get("tokens") or {}
    if not tokens.get("access_token") or not tokens.get("refresh_token"):
        raise AgentShipError(
            f"Codex auth file {path} has no ChatGPT tokens — run `codex login` "
            "(API-key-only auth cannot reach the chatgpt.com backend)."
        )
    return auth


def _jwt_claims(token: str) -> dict:
    try:
        payload = token.split(".")[1]
        padded = payload + "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(padded))
    except (IndexError, ValueError, binascii.Error):
        return {}
    return claims if isinstance(claims, dict) else {}


def _jwt_expiry(token: str) -> float:
    """The access token's ``exp`` (epoch seconds); 0.0 => treat as expired."""
    exp = _jwt_claims(token).get("exp")
    return float(exp) if isinstance(exp, (int, float)) else 0.0


def _account_id(auth: dict) -> str:
    tokens = auth.get("tokens") or {}
    account = tokens.get("account_id")
    if account:
        return str(account)
    claims = _jwt_claims(tokens.get("access_token") or "")
    account = (claims.get("https://api.openai.com/auth") or {}).get("chatgpt_account_id")
    if account:
        return str(account)
    raise AgentShipError(
        "Could not determine the ChatGPT account id from auth.json — "
        "run `codex login` to refresh it."
    )


def _store_auth(path: Path, auth: dict) -> None:
    # The tokens grant full account access: create 0600 from the start and
    # publish atomically (write-then-chmod would leave a umask window).
    tmp = path.with_suffix(".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(json.dumps(auth, indent=2))
    os.replace(tmp, path)


def _refresh(auth: dict, path: Path) -> dict:
    tokens = auth.get("tokens") or {}
    body = json.dumps(
        {
            "client_id": _CLIENT_ID,
            "grant_type": "refresh_token",
            "refresh_token": tokens.get("refresh_token"),
            "scope": "openid profile email",
        }
    ).encode()
    request = urllib.request.Request(
        _TOKEN_URL,
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=_REFRESH_TIMEOUT_SECONDS) as response:
            data = json.loads(response.read())
    except Exception as exc:  # urllib raises a zoo of errors; all mean "no token"
        raise AgentShipError(
            f"Refreshing the ChatGPT token failed ({exc}) — try `codex login`."
        ) from exc

    access = data.get("access_token")
    if not access:
        raise AgentShipError("Token endpoint returned no access_token.")
    tokens["access_token"] = access
    if data.get("refresh_token"):  # rotated — the old one is now dead
        tokens["refresh_token"] = data["refresh_token"]
    if data.get("id_token"):
        tokens["id_token"] = data["id_token"]
    auth["tokens"] = tokens
    auth["last_refresh"] = datetime.now(UTC).isoformat()
    try:
        _store_auth(path, auth)
    except OSError as exc:
        # The fresh token still works for this run; losing the rotated refresh
        # token on disk is worth a loud warning though.
        print(
            f"warning: could not write refreshed tokens to {path}: {exc}",
            file=sys.stderr,
            flush=True,
        )
    return auth


def check_auth() -> Path:
    """Validate that usable ChatGPT tokens exist on disk; return their path.

    A cheap, offline health check for ``pr-pilot doctor``: it confirms the
    auth file exists and holds ChatGPT tokens, without refreshing or calling
    the network. Raises :class:`AgentShipError` with the fix (``codex login``).
    """
    path = _auth_path()
    _load_auth(path)
    return path


def access_token(force_refresh: bool = False) -> tuple[str, str]:
    """A valid ``(access_token, account_id)`` pair, refreshed when near expiry."""
    path = _auth_path()
    with _auth_lock:
        auth = _load_auth(path)
        token = auth["tokens"]["access_token"]
        if force_refresh or _jwt_expiry(token) - time.time() < _EXPIRY_SKEW_SECONDS:
            auth = _refresh(auth, path)
            token = auth["tokens"]["access_token"]
        return token, _account_id(auth)


def build_payload(
    prompt: str, model: str, *, effort: str | None = None, cache_key: str | None = None
) -> dict:
    """The Responses-API-shaped request body. Streaming is mandatory here.

    ``effort`` sets ``reasoning.effort`` (lower = less thinking time per round,
    the main latency lever); ``cache_key`` sets ``prompt_cache_key`` so the
    stable prompt prefix is reliably cache-hit across the loop's resends. Both
    are fields codex itself sends to this endpoint.
    """
    payload: dict = {
        "model": model,
        "instructions": _INSTRUCTIONS,
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": prompt}],
            }
        ],
        "stream": True,
        "store": False,
    }
    if effort:
        payload["reasoning"] = {"effort": effort}
    if cache_key:
        payload["prompt_cache_key"] = cache_key
    return payload


def build_headers(token: str, account_id: str, session_id: str) -> dict:
    return {
        "Authorization": f"Bearer {token}",
        "chatgpt-account-id": account_id,
        "OpenAI-Beta": _OPENAI_BETA,
        "originator": "codex_cli_rs",
        "session_id": session_id,
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }


def iter_sse(lines: Iterable[bytes | str]) -> Iterator[tuple[str, dict]]:
    """Decode an SSE byte/line stream into ``(event_type, data)`` pairs.

    Frames are ``event:``/``data:`` lines terminated by a blank line; multi-line
    ``data:`` payloads are joined before JSON-decoding. ``[DONE]`` sentinels and
    non-JSON payloads are skipped. When a frame has no ``event:`` line the
    payload's own ``type`` field is used (the Responses API sets both).
    """
    event_type = ""
    data_lines: list[str] = []
    for raw in lines:
        line = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
        line = line.rstrip("\r\n")
        if line == "":
            etype, payload = event_type, "\n".join(data_lines)
            event_type, data_lines = "", []
            if not payload or payload.strip() == "[DONE]":
                continue
            try:
                data = json.loads(payload)
            except ValueError:
                continue
            if isinstance(data, dict):
                yield etype or str(data.get("type", "")), data
            continue
        if line.startswith("event:"):
            event_type = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())


def _http_error_detail(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read().decode("utf-8", errors="replace")[:500]
    except OSError:
        body = ""
    return f"HTTP {exc.code}: {body or exc.reason}"


def _open_stream(
    prompt: str, model: str, *, effort: str | None = None, cache_key: str | None = None
):
    """POST the prompt and return the live SSE response (caller closes it).

    A 401 gets one forced token refresh and retry — the JWT can be revoked
    server-side before its ``exp``. ``cache_key`` doubles as the ``session_id``
    header so all rounds of one run share a session (better prefix caching).
    """
    payload = json.dumps(
        build_payload(prompt, model, effort=effort, cache_key=cache_key)
    ).encode()
    session_id = cache_key or str(uuid.uuid4())
    for attempt in (0, 1):
        token, account = access_token(force_refresh=attempt > 0)
        request = urllib.request.Request(
            _RESPONSES_URL,
            data=payload,
            method="POST",
            headers=build_headers(token, account, session_id),
        )
        try:
            return urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS)
        except urllib.error.HTTPError as exc:
            if exc.code == 401 and attempt == 0:
                continue
            raise AgentShipError(
                f"chatgpt.com request failed — {_http_error_detail(exc)}"
            ) from exc
        except urllib.error.URLError as exc:
            raise AgentShipError(f"chatgpt.com request failed: {exc.reason}") from exc
        except TimeoutError as exc:
            raise AgentShipError(
                f"chatgpt.com did not respond within {_TIMEOUT_SECONDS}s."
            ) from exc
    raise AssertionError("unreachable")


def run_chatgpt(
    prompt: str,
    model: str | None = None,
    *,
    effort: str | None = None,
    cache_key: str | None = None,
) -> str:
    """Run one turn against the chatgpt.com backend and return the reply text."""
    response = _open_stream(
        prompt, model or DEFAULT_MODEL, effort=effort, cache_key=cache_key
    )
    parts: list[str] = []
    completed = False
    failure: str | None = None
    try:
        for event_type, data in iter_sse(response):
            if event_type == "response.output_text.delta":
                delta = data.get("delta")
                if isinstance(delta, str):
                    parts.append(delta)
            elif event_type == "response.completed":
                completed = True
            elif event_type == "response.failed":
                error = (data.get("response") or {}).get("error") or data.get("error") or {}
                failure = error.get("message") or failure or "response.failed"
            elif event_type == "error":
                failure = failure or data.get("message") or "error event"
    except TimeoutError as exc:
        raise AgentShipError(
            f"chatgpt.com stream stalled past {_TIMEOUT_SECONDS}s."
        ) from exc
    finally:
        response.close()

    if failure:
        raise AgentShipError(f"chatgpt.com reported a failure: {failure}")
    if not completed and not parts:
        raise AgentShipError("chatgpt.com stream ended without any reply text.")
    return "".join(parts).strip()


# --------------------------------------------------------------------------- #
# Agent loop — file inspection / editing over the actions protocol
# --------------------------------------------------------------------------- #

# How the model asks to touch the repo. A JSON array in a fenced block; the
# model appends one block per turn while it still needs to act, and answers
# with NO block once it is finished.
_ACTIONS_FENCE_RE = re.compile(
    r"```[ \t]*actions[ \t]*\n(.*?)```", re.DOTALL | re.IGNORECASE
)

_READ_OPS = ("list", "read", "search", "git_diff")
_WRITE_OPS = ("write", "delete")

# Bounds keep a single run — and the resent transcript — from exploding.
_MAX_ROUNDS = 40
# Wall-clock budget for one agent-loop invocation (planning, implementing, or a
# single review). The transcript is resent every round to a slow reasoning
# model, so a phase can otherwise grind for a very long time; past this it stops
# and the caller's failure path recovers. Set generously: on a large repo the
# implement phase alone can run 20-30 min of visible round-by-round progress.
_MAX_AGENT_SECONDS = 7200.0
# How many times to push an implementer that "finished" without editing files.
_MAX_WRITE_NUDGES = 2
_MAX_READ_BYTES = 12_000
_MAX_OP_OUTPUT = 8_000
_MAX_LIST_ENTRIES = 200
_MAX_TRANSCRIPT_CHARS = 120_000
# Tracked files shown to the model up front so it can target files directly
# instead of spending rounds discovering the repo layout.
_MAX_TREE_FILES = 400

_READ_PROTOCOL = """\
You are working inside a git repository. You cannot see it until you ask.
To inspect it, end your message with EXACTLY one fenced block and nothing after:
```actions
[{"op": "list", "path": "."}, {"op": "read", "path": "src/foo.py"}]
```
Read-only ops: list (directory entries), read (file text), search (ripgrep-style
regex over tracked files: {"op": "search", "query": "def foo"}), git_diff (the
current uncommitted working-tree diff, no args).
Each result comes back as an "Observation". When you have seen enough, answer in
plain text with NO actions block — that plain answer is your final response."""

_WRITE_PROTOCOL = """\
You are the implementation agent inside a git repository. You cannot see or edit
it until you ask. End your message with EXACTLY one fenced block and nothing after:
```actions
[{"op": "read", "path": "src/foo.py"},
 {"op": "write", "path": "src/foo.py", "content": "full new file contents"},
 {"op": "delete", "path": "obsolete.py"}]
```
Read-only ops: list, read, search ({"op":"search","query":"regex"}), git_diff.
Write ops: write (creates or fully overwrites a file with "content" — always send
the complete file, never a patch), delete (remove a file). Paths are relative to
the repo root and may not escape it.
Make the smallest coherent change, update or add tests, and inspect before you
edit. Each result comes back as an "Observation". When every intended change is
written to disk, answer in plain text with NO actions block, summarizing what you
changed — that plain answer ends the run."""


def _repo_tree(repo: Path, limit: int = _MAX_TREE_FILES) -> str:
    """A capped list of tracked files, to seed the model with the repo layout."""
    proc = subprocess.run(
        ["git", "ls-files"], cwd=repo, text=True, capture_output=True
    )
    if proc.returncode != 0:
        return ""
    files = proc.stdout.splitlines()
    tree = "\n".join(files[:limit])
    if len(files) > limit:
        tree += f"\n… ({len(files) - limit} more files — use list/search to see them)"
    return tree


def _parse_actions(text: str) -> tuple[str, list[dict] | None]:
    """Split a reply into ``(prose, actions)``.

    ``actions is None`` means the model emitted no fenced block — it is done and
    ``prose`` is its final answer. An empty list means a block was present but
    unusable (malformed / no valid ops); the caller nudges and retries.
    """
    match = _ACTIONS_FENCE_RE.search(text)
    if match is None:
        return text.strip(), None
    prose = (text[: match.start()] + text[match.end() :]).strip()
    try:
        data = json.loads(match.group(1).strip())
    except ValueError:
        return prose, []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return prose, []
    actions = [item for item in data if isinstance(item, dict) and item.get("op")]
    return prose, actions


def _safe_target(repo: Path, rel: str) -> Path:
    """Resolve ``rel`` under ``repo``, rejecting traversal and the git dir."""
    if not isinstance(rel, str) or not rel.strip():
        raise AgentShipError("path is required")
    target = (repo / rel).resolve()
    root = repo.resolve()
    if target != root and root not in target.parents:
        raise AgentShipError(f"path escapes the repository: {rel}")
    if ".git" in target.relative_to(root).parts:
        raise AgentShipError("the .git directory is off limits")
    return target


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n… [truncated, {len(text) - limit} more chars]"


def _do_list(repo: Path, rel: str) -> str:
    target = _safe_target(repo, rel or ".")
    if not target.is_dir():
        raise AgentShipError(f"not a directory: {rel}")
    entries = []
    for child in sorted(target.iterdir())[:_MAX_LIST_ENTRIES]:
        if child.name == ".git":
            continue
        entries.append(child.name + ("/" if child.is_dir() else ""))
    return "\n".join(entries) or "(empty)"


def _do_read(repo: Path, rel: str) -> str:
    target = _safe_target(repo, rel)
    if not target.is_file():
        raise AgentShipError(f"not a file: {rel}")
    return _clip(target.read_text(encoding="utf-8", errors="replace"), _MAX_READ_BYTES)


def _do_search(repo: Path, query: str) -> str:
    if not isinstance(query, str) or not query:
        raise AgentShipError("search needs a 'query'")
    # git grep stays inside tracked files and honors .gitignore; -n adds line
    # numbers. A non-match exits 1, which is a normal "nothing found" here.
    proc = subprocess.run(
        ["git", "grep", "-n", "-I", "-E", query],
        cwd=repo,
        text=True,
        capture_output=True,
    )
    if proc.returncode not in (0, 1):
        raise AgentShipError((proc.stderr or "git grep failed").strip())
    return _clip(proc.stdout, _MAX_OP_OUTPUT) if proc.stdout else "(no matches)"


def _do_git_diff(repo: Path) -> str:
    proc = subprocess.run(
        ["git", "diff"], cwd=repo, text=True, capture_output=True
    )
    if proc.returncode != 0:
        raise AgentShipError((proc.stderr or "git diff failed").strip())
    return _clip(proc.stdout, _MAX_OP_OUTPUT) if proc.stdout.strip() else "(no changes yet)"


def _do_write(repo: Path, rel: str, content: object) -> str:
    if not isinstance(content, str):
        raise AgentShipError("write needs string 'content'")
    target = _safe_target(repo, rel)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return f"wrote {rel} ({len(content)} bytes)"


def _do_delete(repo: Path, rel: str) -> str:
    target = _safe_target(repo, rel)
    if not target.is_file():
        raise AgentShipError(f"not a file: {rel}")
    target.unlink()
    return f"deleted {rel}"


def _execute_action(action: dict, repo: Path, allow_writes: bool) -> str:
    op = str(action.get("op"))
    if op in _WRITE_OPS and not allow_writes:
        return f"error: '{op}' is not permitted in this read-only role"
    try:
        if op == "list":
            return _do_list(repo, str(action.get("path", ".")))
        if op == "read":
            return _do_read(repo, str(action.get("path", "")))
        if op == "search":
            return _do_search(repo, action.get("query", ""))
        if op == "git_diff":
            return _do_git_diff(repo)
        if op == "write":
            return _do_write(repo, str(action.get("path", "")), action.get("content"))
        if op == "delete":
            return _do_delete(repo, str(action.get("path", "")))
        return f"error: unknown op '{op}'"
    except AgentShipError as exc:
        return f"error: {exc}"


def _compose_transcript(
    prefix: str, rounds: list[str], limit: int = _MAX_TRANSCRIPT_CHARS
) -> str:
    """``prefix`` + the most recent rounds that fit, dropping the oldest.

    The prefix (protocol + repo tree + task) is always kept, and so are the
    newest rounds — the model never loses the files it just read, which is what
    stops it re-reading them. Only stale middle rounds are dropped.
    """
    budget = limit - len(prefix)
    if budget <= 0:
        return prefix
    kept: list[str] = []
    total = 0
    dropped = False
    for rnd in reversed(rounds):
        if kept and total + len(rnd) > budget:
            dropped = True
            break
        kept.append(rnd)
        total += len(rnd)
    kept.reverse()
    marker = "\n\n[… older observations dropped …]" if dropped else ""
    return prefix + marker + "".join(kept)


def run_chatgpt_agent(
    prompt: str,
    repo: Path,
    *,
    allow_writes: bool,
    model: str | None = None,
    effort: str | None = None,
    max_rounds: int = _MAX_ROUNDS,
    max_seconds: float = _MAX_AGENT_SECONDS,
) -> str:
    """Drive the model through inspect/edit rounds; return its final answer.

    The conversation is stateless on the backend (``store=false``), so each
    round resends a transcript composed of a stable prefix (protocol + repo
    tree + task) plus the most recent rounds that fit in ``_MAX_TRANSCRIPT_CHARS``
    (see :func:`_compose_transcript`). The loop ends when the model replies
    without an actions block, or when ``max_rounds`` is hit — for a writing role
    the caller then checks the working tree, so a truncated run still leaves real
    edits behind. A per-invocation wall-clock budget (``max_seconds``) stops a
    runaway phase; every round is logged at INFO. ``effort`` sets the model's
    reasoning effort (lower = faster); one ``cache_key`` is reused for every
    round so the prefix stays cache-hot.
    """
    repo = Path(repo)
    role = "write" if allow_writes else "read"
    protocol = _WRITE_PROTOCOL if allow_writes else _READ_PROTOCOL
    tree = _repo_tree(repo)
    layout = f"\n\n--- REPOSITORY FILES ---\n{tree}" if tree else ""
    prefix = f"{protocol}{layout}\n\n--- TASK ---\n{prompt}"
    rounds: list[str] = []
    last_prose = ""
    wrote = False  # did any write/delete actually land?
    nudges = 0
    cache_key = str(uuid.uuid4())
    started = time.monotonic()
    logger.info(
        "agent loop start: role=%s max_rounds=%d budget=%.0fs effort=%s",
        role, max_rounds, max_seconds, effort or "default",
    )
    for round_num in range(1, max_rounds + 1):
        elapsed = time.monotonic() - started
        if elapsed > max_seconds:
            raise AgentShipError(
                f"agent loop ({role}) exceeded its {max_seconds:.0f}s budget "
                f"after {round_num - 1} rounds"
            )
        transcript = _compose_transcript(prefix, rounds)
        reply = run_chatgpt(transcript, model=model, effort=effort, cache_key=cache_key)
        prose, actions = _parse_actions(reply)
        last_prose = prose or last_prose
        if actions is None:  # no fenced block — the model thinks it is done
            # An implementer that "finished" without editing anything has done
            # nothing (the run would fail on an empty worktree). Push back a
            # couple of times before accepting it, unless it explicitly says no
            # change is needed.
            if allow_writes and not wrote and nudges < _MAX_WRITE_NUDGES \
                    and prose.strip().upper() != "NO_CHANGE":
                nudges += 1
                logger.info("implementer finished without writing; nudge %d/%d", nudges, _MAX_WRITE_NUDGES)
                rounds.append(
                    f"\n\n--- ASSISTANT ---\n{reply}"
                    + "\n\n--- OBSERVATION ---\n"
                    + "You have not created or edited any files, so nothing is "
                    "implemented yet. Emit write actions to make the change now. "
                    "Only if no code change is genuinely required, reply with "
                    "exactly NO_CHANGE."
                )
                continue
            logger.info(
                "agent loop done: role=%s rounds=%d elapsed=%.0fs wrote=%s",
                role, round_num, elapsed, wrote,
            )
            return prose
        ops = ", ".join(
            f"{a.get('op')} {a.get('path') or ''}".strip() for a in actions
        ) if actions else "(empty block)"
        logger.info("agent round %d/%d (%s): %s", round_num, max_rounds, role, ops[:300])
        if not actions:
            rounds.append(
                f"\n\n--- ASSISTANT ---\n{reply}"
                + "\n\n--- OBSERVATION ---\n"
                + "error: the actions block was empty or malformed JSON. Send a "
                "valid JSON array, or answer in plain text with no block if done."
            )
            continue
        observations = []
        for action in actions:
            result = _execute_action(action, repo, allow_writes)
            if action.get("op") in _WRITE_OPS and not result.startswith("error"):
                wrote = True
            label = action.get("path") or action.get("op")
            observations.append(f"[{action.get('op')} {label}]\n{result}")
        rounds.append(
            f"\n\n--- ASSISTANT ---\n{reply}"
            + "\n\n--- OBSERVATION ---\n"
            + "\n\n".join(observations)
        )
    logger.info("agent loop hit max_rounds=%d (role=%s)", max_rounds, role)
    return last_prose or "Reached the maximum number of agent rounds."
