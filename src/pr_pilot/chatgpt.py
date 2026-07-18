"""Chat via chatgpt.com's backend Responses endpoint, stdlib-only.

Reuses the ChatGPT OAuth tokens the Codex CLI keeps in ``$CODEX_HOME/auth.json``
(``codex login``), so there is no API key and usage is billed to the ChatGPT
plan. The endpoint is the same private one the Codex CLI itself drives —
unofficial, so everything that touches it lives in this one module. Unlike the
codex/cursor CLI providers this is text-only: it returns a reply and cannot
edit files, which is why only read-only roles (reviewer, memory profiling) may
use it.

Failures raise :class:`AgentShipError`; HTTP status codes are kept in the
message so ``LimitRetryProvider`` recognizes 429s as limit errors.
"""

from __future__ import annotations

import base64
import binascii
import json
import os
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


def build_payload(prompt: str, model: str) -> dict:
    """The Responses-API-shaped request body. Streaming is mandatory here."""
    return {
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


def _open_stream(prompt: str, model: str):
    """POST the prompt and return the live SSE response (caller closes it).

    A 401 gets one forced token refresh and retry — the JWT can be revoked
    server-side before its ``exp``.
    """
    payload = json.dumps(build_payload(prompt, model)).encode()
    for attempt in (0, 1):
        token, account = access_token(force_refresh=attempt > 0)
        request = urllib.request.Request(
            _RESPONSES_URL,
            data=payload,
            method="POST",
            headers=build_headers(token, account, str(uuid.uuid4())),
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


def run_chatgpt(prompt: str, model: str | None = None) -> str:
    """Run one turn against the chatgpt.com backend and return the reply text."""
    response = _open_stream(prompt, model or DEFAULT_MODEL)
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
