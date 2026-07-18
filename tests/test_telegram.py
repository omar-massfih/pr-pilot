from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pr_pilot.config import Config, TelegramConfig
from pr_pilot.telegram import TelegramBot

CHAT = 123
STRANGER = 999


class FakeState:
    def __init__(self, feature):
        self.pr_url = f"https://example.test/pr/{feature[:8]}"
        self.phase = "pr_open"


class FakeWorkflow:
    def __init__(self, suggestions):
        self.suggestions = list(suggestions)
        self.recommended = 0
        self.runs = []
        self.raise_on_run = False

    def recommend_feature(self):
        if self.recommended < len(self.suggestions):
            value = self.suggestions[self.recommended]
            self.recommended += 1
            return value
        return None

    def run(self, feature, *, watch=None):
        if self.raise_on_run:
            raise RuntimeError("boom")
        self.runs.append((feature, watch))
        return FakeState(feature)


def _msg(text, chat_id=CHAT):
    return {"message": {"chat": {"id": chat_id}, "text": text}}


class TelegramLoopTests(unittest.TestCase):
    def _bot(self, suggestions):
        with tempfile.TemporaryDirectory() as directory:
            config = Config(
                repo=Path(directory),
                telegram=TelegramConfig(allowed_chat_ids=(CHAT,)),
            )
        workflow = FakeWorkflow(suggestions)
        self.sent: list[str] = []

        def transport(method, values):
            if method == "sendMessage":
                self.sent.append(values["text"])
            return {"ok": True, "result": {}}

        with patch.dict("os.environ", {"TELEGRAM_BOT_TOKEN": "tok"}):
            bot = TelegramBot(config, lambda: workflow, transport=transport)
        return bot, workflow

    def test_auto_suggests_and_sets_pending(self):
        bot, workflow = self._bot(["Add a --json flag"])
        bot._handle(_msg("/auto"))
        self.assertEqual(bot.pending, "Add a --json flag")
        self.assertTrue(any("Add a --json flag" in text for text in self.sent))

    def test_yes_builds_then_suggests_the_next(self):
        bot, workflow = self._bot(["First feature", "Second feature"])
        bot._handle(_msg("/auto"))
        self.assertEqual(bot.pending, "First feature")
        bot._handle(_msg("/yes"))
        # First feature was built (watch disabled to stay responsive)...
        self.assertEqual(workflow.runs, [("First feature", False)])
        # ...and the loop continued by proposing the next one.
        self.assertEqual(bot.pending, "Second feature")
        self.assertTrue(any("Finished" in text for text in self.sent))
        self.assertTrue(any("Second feature" in text for text in self.sent))

    def test_no_skips_to_a_different_suggestion(self):
        bot, workflow = self._bot(["Alpha", "Beta"])
        bot._handle(_msg("/auto"))
        self.assertEqual(bot.pending, "Alpha")
        bot._handle(_msg("/no"))
        self.assertEqual(bot.pending, "Beta")
        self.assertEqual(workflow.runs, [])  # nothing built on a skip

    def test_stop_clears_pending(self):
        bot, _ = self._bot(["Alpha"])
        bot._handle(_msg("/auto"))
        bot._handle(_msg("/stop"))
        self.assertIsNone(bot.pending)

    def test_yes_without_pending_is_a_no_op(self):
        bot, workflow = self._bot([])
        bot._handle(_msg("/yes"))
        self.assertEqual(workflow.runs, [])
        self.assertTrue(any("Nothing to confirm" in text for text in self.sent))

    def test_no_more_suggestions_reports_and_clears(self):
        bot, _ = self._bot([])
        bot._handle(_msg("/auto"))
        self.assertIsNone(bot.pending)
        self.assertTrue(any("No new feature" in text for text in self.sent))

    def test_explicit_feature_builds_immediately(self):
        bot, workflow = self._bot([])
        bot._handle(_msg("/feature Add retry to the client"))
        self.assertEqual(workflow.runs, [("Add retry to the client", False)])

    def test_feature_without_description_is_rejected(self):
        bot, workflow = self._bot([])
        bot._handle(_msg("/feature"))
        self.assertEqual(workflow.runs, [])
        self.assertTrue(any("Expected: /feature" in text for text in self.sent))

    def test_unauthorized_chat_is_ignored(self):
        bot, workflow = self._bot(["Alpha"])
        bot._handle(_msg("/auto", chat_id=STRANGER))
        self.assertEqual(self.sent, [])
        self.assertIsNone(bot.pending)

    def test_build_failure_is_reported_and_loop_continues(self):
        bot, workflow = self._bot(["First", "Second"])
        workflow.raise_on_run = True
        bot._handle(_msg("/auto"))
        bot._handle(_msg("/yes"))
        self.assertTrue(any("Run stopped" in text for text in self.sent))
        # Even after a failed build, the loop offers the next suggestion.
        self.assertEqual(bot.pending, "Second")


if __name__ == "__main__":
    unittest.main()
