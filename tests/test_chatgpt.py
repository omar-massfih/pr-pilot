from __future__ import annotations

import base64
import io
import json
import stat
import tempfile
import time
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

from pr_pilot import chatgpt
from pr_pilot.errors import AgentShipError


def _jwt(claims: dict) -> str:
    payload = base64.urlsafe_b64encode(json.dumps(claims).encode()).rstrip(b"=")
    return f"header.{payload.decode()}.signature"


def _fresh_token() -> str:
    return _jwt({"exp": time.time() + 3600})


class FakeResponse:
    """Stands in for urlopen's return: iterable SSE lines + read() for JSON."""

    def __init__(self, lines=None, body=b"{}"):
        self._lines = lines or []
        self._body = body
        self.closed = False

    def __iter__(self):
        return iter(self._lines)

    def read(self):
        return self._body

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _sse(*events: tuple[str, dict]) -> list[bytes]:
    lines: list[bytes] = []
    for etype, data in events:
        lines += [
            f"event: {etype}\n".encode(),
            f"data: {json.dumps(data)}\n".encode(),
            b"\n",
        ]
    return lines


def _reply_events(*deltas: str) -> list[bytes]:
    events = [("response.output_text.delta", {"delta": d}) for d in deltas]
    events.append(("response.completed", {"response": {}}))
    return _sse(*events)


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        chatgpt._RESPONSES_URL, code, "nope", None, io.BytesIO(b"denied")
    )


class AuthHome:
    """Context manager: a temporary CODEX_HOME holding a fake auth.json."""

    def __init__(self, access: str, refresh: str = "rt-old"):
        self.access = access
        self.refresh = refresh

    def __enter__(self) -> Path:
        self._dir = tempfile.TemporaryDirectory()
        self._env = patch.dict("os.environ", {"CODEX_HOME": self._dir.name})
        self._env.start()
        path = Path(self._dir.name) / "auth.json"
        path.write_text(
            json.dumps(
                {
                    "tokens": {
                        "access_token": self.access,
                        "refresh_token": self.refresh,
                        "account_id": "acct-1",
                    },
                    "last_refresh": "2026-01-01",
                }
            )
        )
        return path

    def __exit__(self, *exc):
        self._env.stop()
        self._dir.cleanup()


class RequestBuildingTests(unittest.TestCase):
    def test_payload_shape(self):
        payload = chatgpt.build_payload("hello", "gpt-5.5")
        self.assertEqual(payload["model"], "gpt-5.5")
        self.assertTrue(payload["stream"])
        self.assertFalse(payload["store"])
        [item] = payload["input"]
        self.assertEqual(item["content"], [{"type": "input_text", "text": "hello"}])

    def test_headers_shape(self):
        headers = chatgpt.build_headers("tok", "acct", "sess")
        self.assertEqual(headers["Authorization"], "Bearer tok")
        self.assertEqual(headers["chatgpt-account-id"], "acct")
        self.assertEqual(headers["originator"], "codex_cli_rs")
        self.assertEqual(headers["Accept"], "text/event-stream")


class SseTests(unittest.TestCase):
    def test_decodes_frames_and_skips_noise(self):
        lines = [
            b"event: response.output_text.delta\r\n",
            b'data: {"delta": "Hi"}\r\n',
            b"\r\n",
            b"data: [DONE]\n",
            b"\n",
            b"data: not json\n",
            b"\n",
        ]
        self.assertEqual(
            list(chatgpt.iter_sse(lines)),
            [("response.output_text.delta", {"delta": "Hi"})],
        )

    def test_falls_back_to_payload_type_field(self):
        lines = ['data: {"type": "response.completed"}\n', "\n"]
        self.assertEqual(
            list(chatgpt.iter_sse(lines)),
            [("response.completed", {"type": "response.completed"})],
        )


class AuthTests(unittest.TestCase):
    def test_fresh_token_needs_no_network(self):
        token = _fresh_token()
        with AuthHome(token):
            def no_network(*args, **kwargs):
                raise AssertionError("fresh token must not hit the network")

            with patch("pr_pilot.chatgpt.urllib.request.urlopen", no_network):
                self.assertEqual(chatgpt.access_token(), (token, "acct-1"))

    def test_expired_token_is_refreshed_and_persisted(self):
        new_token = _fresh_token()
        with AuthHome(_jwt({"exp": time.time() - 10})) as path:
            seen = {}

            def fake_urlopen(request, timeout=None):
                seen["url"] = request.full_url
                seen["body"] = json.loads(request.data)
                return FakeResponse(
                    body=json.dumps(
                        {"access_token": new_token, "refresh_token": "rt-new"}
                    ).encode()
                )

            with patch("pr_pilot.chatgpt.urllib.request.urlopen", fake_urlopen):
                self.assertEqual(chatgpt.access_token(), (new_token, "acct-1"))

            self.assertEqual(seen["url"], chatgpt._TOKEN_URL)
            self.assertEqual(seen["body"]["refresh_token"], "rt-old")
            stored = json.loads(path.read_text())
            self.assertEqual(stored["tokens"]["access_token"], new_token)
            self.assertEqual(stored["tokens"]["refresh_token"], "rt-new")
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)

    def test_refresh_failure_raises(self):
        with AuthHome(_jwt({"exp": 0})):
            def failing(*args, **kwargs):
                raise urllib.error.URLError("connection refused")

            with patch("pr_pilot.chatgpt.urllib.request.urlopen", failing):
                with self.assertRaisesRegex(AgentShipError, "codex login"):
                    chatgpt.access_token()

    def test_missing_auth_file_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.dict("os.environ", {"CODEX_HOME": directory}):
                with self.assertRaisesRegex(AgentShipError, "codex login"):
                    chatgpt.access_token()


class RunChatGptTests(unittest.TestCase):
    def test_concatenates_deltas(self):
        response = FakeResponse(_reply_events("Hel", "lo", " world"))
        with AuthHome(_fresh_token()):
            with patch(
                "pr_pilot.chatgpt.urllib.request.urlopen",
                lambda req, timeout=None: response,
            ):
                self.assertEqual(chatgpt.run_chatgpt("hi"), "Hello world")
        self.assertTrue(response.closed)

    def test_failure_event_raises(self):
        events = _sse(
            ("response.failed", {"response": {"error": {"message": "usage limit hit"}}})
        )
        with AuthHome(_fresh_token()):
            with patch(
                "pr_pilot.chatgpt.urllib.request.urlopen",
                lambda req, timeout=None: FakeResponse(events),
            ):
                with self.assertRaisesRegex(AgentShipError, "usage limit hit"):
                    chatgpt.run_chatgpt("hi")

    def test_http_429_keeps_status_for_limit_retry(self):
        # LimitRetryProvider matches "http 429" in the message, so the status
        # code must survive into the raised error.
        def raising(request, timeout=None):
            if request.full_url == chatgpt._TOKEN_URL:
                return FakeResponse(
                    body=json.dumps({"access_token": _fresh_token()}).encode()
                )
            raise _http_error(429)

        with AuthHome(_fresh_token()):
            with patch("pr_pilot.chatgpt.urllib.request.urlopen", raising):
                with self.assertRaisesRegex(AgentShipError, "HTTP 429"):
                    chatgpt.run_chatgpt("hi")

    def test_http_401_refreshes_once_and_retries(self):
        calls = {"responses": 0, "token": 0}

        def fake_urlopen(request, timeout=None):
            if request.full_url == chatgpt._TOKEN_URL:
                calls["token"] += 1
                return FakeResponse(
                    body=json.dumps({"access_token": _fresh_token()}).encode()
                )
            calls["responses"] += 1
            if calls["responses"] == 1:
                raise _http_error(401)
            return FakeResponse(_reply_events("recovered"))

        with AuthHome(_fresh_token()):
            with patch("pr_pilot.chatgpt.urllib.request.urlopen", fake_urlopen):
                self.assertEqual(chatgpt.run_chatgpt("hi"), "recovered")
        self.assertEqual(calls, {"responses": 2, "token": 1})

    def test_empty_stream_raises(self):
        with AuthHome(_fresh_token()):
            with patch(
                "pr_pilot.chatgpt.urllib.request.urlopen",
                lambda req, timeout=None: FakeResponse([]),
            ):
                with self.assertRaisesRegex(AgentShipError, "without any reply"):
                    chatgpt.run_chatgpt("hi")


if __name__ == "__main__":
    unittest.main()
