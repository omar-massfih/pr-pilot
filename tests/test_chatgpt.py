from __future__ import annotations

import base64
import io
import json
import stat
import subprocess
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


def _actions_block(actions: list[dict]) -> str:
    return "```actions\n" + json.dumps(actions) + "\n```"


class ParseActionsTests(unittest.TestCase):
    def test_no_block_means_done(self):
        prose, actions = chatgpt._parse_actions("All finished.\nVERDICT: APPROVE")
        self.assertIsNone(actions)
        self.assertEqual(prose, "All finished.\nVERDICT: APPROVE")

    def test_block_is_extracted_and_stripped_from_prose(self):
        text = "Let me look.\n" + _actions_block([{"op": "read", "path": "a.py"}])
        prose, actions = chatgpt._parse_actions(text)
        self.assertEqual(prose, "Let me look.")
        self.assertEqual(actions, [{"op": "read", "path": "a.py"}])

    def test_malformed_json_yields_empty_list_not_none(self):
        prose, actions = chatgpt._parse_actions("oops ```actions\n{not json\n```")
        self.assertEqual(actions, [])

    def test_single_object_is_wrapped(self):
        _, actions = chatgpt._parse_actions(_actions_block([{"op": "list", "path": "."}])
                                            .replace("[", "").replace("]", ""))
        self.assertEqual(actions, [{"op": "list", "path": "."}])


class SafeTargetTests(unittest.TestCase):
    def test_rejects_parent_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AgentShipError, "escapes"):
                chatgpt._safe_target(Path(directory), "../secret")

    def test_rejects_git_dir(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(AgentShipError, "\\.git"):
                chatgpt._safe_target(Path(directory), ".git/config")

    def test_accepts_nested_repo_path(self):
        with tempfile.TemporaryDirectory() as directory:
            target = chatgpt._safe_target(Path(directory), "src/pkg/mod.py")
            self.assertTrue(str(target).endswith("src/pkg/mod.py"))


class ExecuteActionTests(unittest.TestCase):
    def test_write_then_read_roundtrips(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            out = chatgpt._execute_action(
                {"op": "write", "path": "pkg/new.py", "content": "x = 1\n"},
                repo, allow_writes=True,
            )
            self.assertIn("wrote", out)
            self.assertEqual((repo / "pkg/new.py").read_text(), "x = 1\n")
            self.assertEqual(
                chatgpt._execute_action({"op": "read", "path": "pkg/new.py"}, repo, True),
                "x = 1\n",
            )

    def test_write_blocked_in_read_only_role(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            out = chatgpt._execute_action(
                {"op": "write", "path": "f.py", "content": "bad"}, repo, allow_writes=False
            )
            self.assertIn("not permitted", out)
            self.assertFalse((repo / "f.py").exists())

    def test_delete_removes_file(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "gone.py").write_text("bye")
            chatgpt._execute_action({"op": "delete", "path": "gone.py"}, repo, True)
            self.assertFalse((repo / "gone.py").exists())

    def test_traversal_write_is_reported_not_executed(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory) / "repo"
            repo.mkdir()
            out = chatgpt._execute_action(
                {"op": "write", "path": "../escape.py", "content": "pwn"}, repo, True
            )
            self.assertIn("error", out)
            self.assertFalse((Path(directory) / "escape.py").exists())

    def test_unknown_op_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            out = chatgpt._execute_action({"op": "nope"}, Path(directory), True)
            self.assertIn("unknown op", out)


class AgentLoopTests(unittest.TestCase):
    def test_loop_inspects_then_finishes(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "foo.py").write_text("print('hi')\n")
            replies = iter([
                "Looking.\n" + _actions_block([{"op": "read", "path": "foo.py"}]),
                "Done reviewing.\nVERDICT: APPROVE",
            ])
            with patch("pr_pilot.chatgpt.run_chatgpt", lambda prompt, model=None: next(replies)):
                result = chatgpt.run_chatgpt_agent("review", repo, allow_writes=False)
            self.assertEqual(result, "Done reviewing.\nVERDICT: APPROVE")

    def test_loop_writes_a_file_then_summarizes(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            replies = iter([
                "Implementing.\n" + _actions_block(
                    [{"op": "write", "path": "feature.py", "content": "VALUE = 42\n"}]
                ),
                "Added feature.py with VALUE.",
            ])
            with patch("pr_pilot.chatgpt.run_chatgpt", lambda prompt, model=None: next(replies)):
                result = chatgpt.run_chatgpt_agent("implement", repo, allow_writes=True)
            self.assertEqual((repo / "feature.py").read_text(), "VALUE = 42\n")
            self.assertIn("feature.py", result)

    def test_loop_stops_at_max_rounds(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            # Always asks for another action — never finishes on its own.
            never_done = "working\n" + _actions_block([{"op": "list", "path": "."}])
            with patch("pr_pilot.chatgpt.run_chatgpt", lambda prompt, model=None: never_done):
                result = chatgpt.run_chatgpt_agent(
                    "loop", repo, allow_writes=False, max_rounds=3
                )
            self.assertIsInstance(result, str)

    def test_implementer_is_nudged_to_write_when_it_finishes_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            replies = iter([
                "I reviewed it and here's what I'd change.",  # done, but wrote nothing
                "Now doing it.\n" + _actions_block(
                    [{"op": "write", "path": "f.py", "content": "x = 1\n"}]
                ),
                "Done, added f.py.",
            ])
            with patch("pr_pilot.chatgpt.run_chatgpt", lambda prompt, model=None: next(replies)):
                result = chatgpt.run_chatgpt_agent("implement", repo, allow_writes=True)
            self.assertEqual((repo / "f.py").read_text(), "x = 1\n")
            self.assertIn("f.py", result)

    def test_implementer_may_declare_no_change_without_nudging(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            with patch("pr_pilot.chatgpt.run_chatgpt", lambda prompt, model=None: "NO_CHANGE"):
                result = chatgpt.run_chatgpt_agent("implement", repo, allow_writes=True)
            self.assertEqual(result, "NO_CHANGE")

    def test_read_role_is_not_nudged_to_write(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            calls = {"n": 0}

            def once(prompt, model=None):
                calls["n"] += 1
                return "Here is my review.\nVERDICT: APPROVE"

            with patch("pr_pilot.chatgpt.run_chatgpt", once):
                result = chatgpt.run_chatgpt_agent("review", repo, allow_writes=False)
            self.assertEqual(calls["n"], 1)  # accepted immediately, no nagging
            self.assertIn("VERDICT", result)

    def test_first_prompt_seeds_the_repo_file_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            for args in (["init"], ["config", "user.email", "t@t.co"],
                         ["config", "user.name", "t"]):
                subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
            (repo / "alpha.py").write_text("x = 1\n")
            subprocess.run(["git", "add", "-A"], cwd=repo, check=True, capture_output=True)
            subprocess.run(["git", "commit", "-m", "i"], cwd=repo, check=True, capture_output=True)
            seen = []

            def fake(prompt, model=None):
                seen.append(prompt)
                return "done"

            with patch("pr_pilot.chatgpt.run_chatgpt", fake):
                chatgpt.run_chatgpt_agent("task", repo, allow_writes=False)
            self.assertIn("REPOSITORY FILES", seen[0])
            self.assertIn("alpha.py", seen[0])

    def test_repo_tree_empty_outside_a_git_repo(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(chatgpt._repo_tree(Path(directory)), "")

    def test_loop_stops_when_the_time_budget_is_exceeded(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            forever = "working\n" + _actions_block([{"op": "list", "path": "."}])
            with patch("pr_pilot.chatgpt.run_chatgpt", lambda prompt, model=None: forever):
                with self.assertRaisesRegex(AgentShipError, "budget"):
                    # Zero budget: the deadline check trips before any round runs.
                    chatgpt.run_chatgpt_agent(
                        "task", repo, allow_writes=True, max_seconds=0
                    )

    def test_observations_are_fed_back_into_the_transcript(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = Path(directory)
            (repo / "readme.txt").write_text("SENTINEL")
            seen = []

            replies = iter([
                "look\n" + _actions_block([{"op": "read", "path": "readme.txt"}]),
                "done",
            ])

            def fake(prompt, model=None):
                seen.append(prompt)
                return next(replies)

            with patch("pr_pilot.chatgpt.run_chatgpt", fake):
                chatgpt.run_chatgpt_agent("task", repo, allow_writes=False)
            # The second prompt must carry the file contents observed in round 1.
            self.assertIn("SENTINEL", seen[1])


if __name__ == "__main__":
    unittest.main()
