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
