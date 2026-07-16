from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pr_pilot.commands import Result
from pr_pilot.config import ProviderConfig
from pr_pilot.providers import CodexProvider, CursorProvider


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


if __name__ == "__main__":
    unittest.main()
