from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pr_pilot.commands import Result
from pr_pilot.config import ProviderConfig
from pr_pilot.errors import AgentShipError
from pr_pilot.providers import (
    AgentProvider,
    ChatGptProvider,
    CodexProvider,
    CursorProvider,
    LimitRetryProvider,
    OpencodeProvider,
    _opencode_text,
    make_provider,
)


class SequenceProvider(AgentProvider):
    def __init__(self, results):
        self.results = iter(results)
        self.calls = 0

    def invoke(self, prompt, *, repo, write):
        self.calls += 1
        result = next(self.results)
        if isinstance(result, Exception):
            raise result
        return result


class ProviderTests(unittest.TestCase):
    def test_codex_uses_read_only_for_non_writing_tasks(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "pr_pilot.providers.run", return_value=Result("plan\n", "", 0)
        ) as mocked:
            output = CodexProvider(ProviderConfig("codex")).invoke(
                "plan", repo=Path(directory), write=False
            )
            command = mocked.call_args.args[0]
            self.assertEqual(output, "plan")
            self.assertIn("read-only", command)

    def test_cursor_force_is_only_used_for_writes(self):
        payload = json.dumps({"type": "result", "result": "done"})
        with tempfile.TemporaryDirectory() as directory, patch(
            "pr_pilot.providers.run", return_value=Result(payload, "", 0)
        ) as mocked:
            provider = CursorProvider(ProviderConfig("cursor"))
            provider.invoke("review", repo=Path(directory), write=False)
            self.assertNotIn("--force", mocked.call_args.args[0])
            provider.invoke("fix", repo=Path(directory), write=True)
            self.assertIn("--force", mocked.call_args.args[0])

    def test_chatgpt_read_role_runs_the_agent_loop_without_writes(self):
        with patch(
            "pr_pilot.providers.run_chatgpt_agent", return_value="looks good"
        ) as mocked:
            output = ChatGptProvider(ProviderConfig("chatgpt", model="gpt-5.5")).invoke(
                "review this", repo=Path("."), write=False
            )
        self.assertEqual(output, "looks good")
        mocked.assert_called_once_with(
            "review this", Path("."), allow_writes=False, model="gpt-5.5", effort="low"
        )

    def test_chatgpt_write_role_enables_writes(self):
        with patch(
            "pr_pilot.providers.run_chatgpt_agent", return_value="shipped"
        ) as mocked:
            ChatGptProvider(ProviderConfig("chatgpt")).invoke(
                "implement", repo=Path("/repo"), write=True
            )
        self.assertTrue(mocked.call_args.kwargs["allow_writes"])

    def test_chatgpt_forwards_reasoning_effort(self):
        with patch(
            "pr_pilot.providers.run_chatgpt_agent", return_value="ok"
        ) as mocked:
            ChatGptProvider(ProviderConfig("chatgpt", reasoning_effort="medium")).invoke(
                "review", repo=Path("."), write=False
            )
        self.assertEqual(mocked.call_args.kwargs["effort"], "medium")

    def test_opencode_read_role_uses_read_only_plan_agent(self):
        stream = json.dumps({"type": "text", "part": {"id": "p1", "text": "the plan"}})
        with tempfile.TemporaryDirectory() as directory, patch(
            "pr_pilot.providers.run", return_value=Result(stream, "", 0)
        ) as mocked, patch("pr_pilot.providers._opencode_bin", return_value="opencode"):
            output = OpencodeProvider(ProviderConfig("opencode")).invoke(
                "plan it", repo=Path(directory), write=False
            )
        command = mocked.call_args.args[0]
        self.assertEqual(output, "the plan")
        self.assertIn("--agent", command)
        self.assertIn("plan", command)
        self.assertNotIn("--auto", command)
        self.assertIn("chatgpt-proxy/gpt-5.5", command)  # default model slug
        # The target dir is pinned explicitly (opencode reads $PWD, not cwd).
        self.assertEqual(command[command.index("--dir") + 1], directory)

    def test_opencode_write_role_uses_auto_not_plan_agent(self):
        stream = json.dumps({"type": "text", "part": {"id": "p1", "text": "done"}})
        with tempfile.TemporaryDirectory() as directory, patch(
            "pr_pilot.providers.run", return_value=Result(stream, "", 0)
        ) as mocked, patch("pr_pilot.providers._opencode_bin", return_value="opencode"):
            OpencodeProvider(
                ProviderConfig("opencode", reasoning_effort="high")
            ).invoke("implement", repo=Path(directory), write=True)
        command = mocked.call_args.args[0]
        self.assertIn("--auto", command)
        self.assertNotIn("plan", command)
        self.assertEqual(command[command.index("--variant") + 1], "high")

    def test_opencode_raises_when_no_message(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "pr_pilot.providers.run", return_value=Result("", "", 0)
        ), patch("pr_pilot.providers._opencode_bin", return_value="opencode"):
            with self.assertRaises(AgentShipError):
                OpencodeProvider(ProviderConfig("opencode")).invoke(
                    "x", repo=Path(directory), write=False
                )

    def test_opencode_text_returns_only_the_final_step(self):
        # Narration in earlier (tool-using) steps must be dropped; only the final
        # step's text is the answer, so it doesn't pollute branch names / titles.
        stream = "\n".join([
            json.dumps({"type": "step_start", "part": {"type": "step-start"}}),
            json.dumps({"type": "text", "part": {"id": "n1", "text": "I'll inspect the repo."}}),
            json.dumps({"type": "step_start", "part": {"type": "step-start"}}),
            json.dumps({"type": "text", "part": {"id": "n2", "text": "Reading docs and tests."}}),
            json.dumps({"type": "step_start", "part": {"type": "step-start"}}),
            json.dumps({"type": "text", "part": {"id": "a", "text": "Add a client-side notes search."}}),
        ])
        self.assertEqual(_opencode_text(stream), "Add a client-side notes search.")

    def test_opencode_text_joins_parts_and_preserves_last_line(self):
        stream = "\n".join([
            json.dumps({"type": "step_start", "part": {"id": "s", "type": "step-start"}}),
            json.dumps({"type": "text", "part": {"id": "p1", "text": "Findings: ok."}}),
            json.dumps({"type": "text", "part": {"id": "p2", "text": "VERDICT: APPROVE"}}),
            "not json, ignored",
        ])
        text = _opencode_text(stream)
        assert text.splitlines()[-1] == "VERDICT: APPROVE"
        self.assertIn("Findings: ok.", text)

    def test_make_provider_wires_opencode(self):
        self.assertIsInstance(make_provider(ProviderConfig("opencode")), LimitRetryProvider)

    def test_make_provider_wires_chatgpt(self):
        provider = make_provider(ProviderConfig("chatgpt"))
        self.assertIsInstance(provider, LimitRetryProvider)
        self.assertIsInstance(provider.provider, ChatGptProvider)

    def test_limit_errors_poll_until_provider_continues(self):
        inner = SequenceProvider(
            [AgentShipError("HTTP 429 rate limit exceeded"), AgentShipError("usage limit"), "done"]
        )
        sleeps = []
        notices = []
        provider = LimitRetryProvider(
            inner,
            ProviderConfig("codex", limit_poll_seconds=7),
            sleeper=sleeps.append,
            clock=lambda: 0,
            notifier=notices.append,
        )

        result = provider.invoke("work", repo=Path("."), write=False)

        self.assertEqual(result, "done")
        self.assertEqual(inner.calls, 3)
        self.assertEqual(sleeps, [7, 7])
        self.assertEqual(len(notices), 2)

    def test_non_limit_errors_are_not_retried(self):
        error = AgentShipError("authentication failed")
        provider = LimitRetryProvider(
            SequenceProvider([error]), ProviderConfig("codex"), sleeper=lambda _: None
        )
        with self.assertRaisesRegex(AgentShipError, "authentication failed"):
            provider.invoke("work", repo=Path("."), write=False)

    def test_configured_limit_wait_can_expire(self):
        times = iter([0, 6])
        provider = LimitRetryProvider(
            SequenceProvider([AgentShipError("rate limit")]),
            ProviderConfig("codex", limit_poll_seconds=5, limit_max_wait_seconds=10),
            sleeper=lambda _: None,
            clock=lambda: next(times),
            notifier=lambda _: None,
        )
        with self.assertRaisesRegex(AgentShipError, "did not clear"):
            provider.invoke("work", repo=Path("."), write=False)


if __name__ == "__main__":
    unittest.main()
