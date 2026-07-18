from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pr_pilot.config import load_config
from pr_pilot.errors import AgentShipError


class ConfigTests(unittest.TestCase):
    def test_loads_provider_and_babysit_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_file = root / "config.toml"
            config_file.write_text(
                f'''repo = "{root}"
[implementer]
name = "cursor"
limit_poll_seconds = 15
limit_max_wait_seconds = 300
[reviewer]
name = "codex"
[workflow]
max_review_attempts = 5
[babysit]
enabled = false
max_fix_attempts = 7
[telegram]
allowed_chat_ids = [42]
'''
            )
            config = load_config(config_file)
            self.assertEqual(config.implementer.name, "cursor")
            self.assertEqual(config.implementer.limit_poll_seconds, 15)
            self.assertEqual(config.implementer.limit_max_wait_seconds, 300)
            self.assertEqual(config.reviewer.name, "codex")
            self.assertEqual(config.workflow.max_review_attempts, 5)
            self.assertFalse(config.babysit.enabled)
            self.assertEqual(config.babysit.max_fix_attempts, 7)
            self.assertEqual(config.telegram.allowed_chat_ids, (42,))
            self.assertEqual(config.memory.embedding_model, "BAAI/bge-small-en-v1.5")

    def test_rejects_unknown_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_file = root / "config.toml"
            config_file.write_text(f'repo = "{root}"\n[implementer]\nname = "other"\n')
            with self.assertRaises(AgentShipError):
                load_config(config_file)

    def test_chatgpt_reviews_but_does_not_implement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_file = root / "config.toml"
            config_file.write_text(f'repo = "{root}"\n[reviewer]\nname = "chatgpt"\n')
            self.assertEqual(load_config(config_file).reviewer.name, "chatgpt")
            # chatgpt is text-only, so the writing role must reject it.
            config_file.write_text(f'repo = "{root}"\n[implementer]\nname = "chatgpt"\n')
            with self.assertRaisesRegex(AgentShipError, "cannot edit files"):
                load_config(config_file)

    def test_chatgpt_profile_provider_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_file = root / "config.toml"
            config_file.write_text(
                f'repo = "{root}"\n[memory]\nprofile_provider = "chatgpt"\n'
            )
            self.assertEqual(load_config(config_file).memory.profile_provider, "chatgpt")

    def test_rejects_negative_review_attempt_limit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_file = root / "config.toml"
            config_file.write_text(
                f'repo = "{root}"\n[workflow]\nmax_review_attempts = -1\n'
            )
            with self.assertRaisesRegex(AgentShipError, "max_review_attempts"):
                load_config(config_file)


if __name__ == "__main__":
    unittest.main()
