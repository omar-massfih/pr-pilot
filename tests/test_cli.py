from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pr_pilot.cli import doctor
from pr_pilot.config import Config, MemoryConfig, ProviderConfig
from pr_pilot.errors import AgentShipError


def _config(root: Path, reviewer: str) -> Config:
    return Config(
        repo=root,
        implementer=ProviderConfig("codex"),
        reviewer=ProviderConfig(reviewer),
        memory=MemoryConfig(enabled=False),
    )


def _which(*known: str):
    return lambda command: f"/usr/bin/{command}" if command in known else None


class DoctorTests(unittest.TestCase):
    def test_chatgpt_reviewer_needs_no_binary_but_needs_auth(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            auth = root / "auth.json"
            auth.write_text(
                json.dumps({"tokens": {"access_token": "a", "refresh_token": "r"}})
            )
            with patch("pr_pilot.cli.shutil.which", _which("git", "gh", "codex")):
                with patch.dict("os.environ", {"CODEX_HOME": directory}):
                    self.assertEqual(doctor(_config(root, "chatgpt")), 0)

    def test_chatgpt_reviewer_without_auth_fails_with_fix(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("pr_pilot.cli.shutil.which", _which("git", "gh", "codex")):
                with patch.dict("os.environ", {"CODEX_HOME": str(root / "empty")}):
                    with self.assertRaisesRegex(AgentShipError, "codex login"):
                        doctor(_config(root, "chatgpt"))

    def test_cursor_reviewer_still_requires_its_binary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("pr_pilot.cli.shutil.which", _which("git", "gh", "codex")):
                with self.assertRaisesRegex(AgentShipError, "cursor-agent"):
                    doctor(_config(root, "cursor"))

    def test_missing_codex_is_reported(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("pr_pilot.cli.shutil.which", _which("git", "gh")):
                with self.assertRaisesRegex(AgentShipError, "codex"):
                    doctor(_config(root, "codex"))


if __name__ == "__main__":
    unittest.main()
