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
[reviewer]
name = "codex"
[babysit]
enabled = false
max_fix_attempts = 7
[telegram]
allowed_chat_ids = [42]
'''
            )
            config = load_config(config_file)
            self.assertEqual(config.implementer.name, "cursor")
            self.assertEqual(config.reviewer.name, "codex")
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


if __name__ == "__main__":
    unittest.main()
