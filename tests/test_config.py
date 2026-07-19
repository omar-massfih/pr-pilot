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

    def test_designer_inherits_implementer_when_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_file = root / "config.toml"
            config_file.write_text(
                f'''repo = "{root}"
[implementer]
name = "chatgpt"
reasoning_effort = "high"
'''
            )
            config = load_config(config_file)
            # No [designer] table => it mirrors the implementer's provider config.
            self.assertEqual(config.designer.name, "chatgpt")
            self.assertEqual(config.designer.reasoning_effort, "high")

    def test_designer_can_override_the_implementer(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_file = root / "config.toml"
            config_file.write_text(
                f'''repo = "{root}"
[implementer]
name = "chatgpt"
[designer]
name = "codex"
reasoning_effort = "medium"
'''
            )
            config = load_config(config_file)
            self.assertEqual(config.designer.name, "codex")
            self.assertEqual(config.designer.reasoning_effort, "medium")
            self.assertEqual(config.implementer.name, "chatgpt")

    def test_rejects_unknown_designer_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_file = root / "config.toml"
            config_file.write_text(
                f'repo = "{root}"\n[designer]\nname = "other"\n'
            )
            with self.assertRaises(AgentShipError):
                load_config(config_file)

    def test_repos_table_parsed_with_first_as_default(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "frontend").mkdir()
            (root / "backend").mkdir()
            config_file = root / "config.toml"
            config_file.write_text(
                f'''[repos]
frontend = "{root / "frontend"}"
backend = "{root / "backend"}"
'''
            )
            config = load_config(config_file)
            self.assertEqual(set(config.repos), {"frontend", "backend"})
            self.assertEqual(config.repos["backend"], (root / "backend").resolve())
            # First entry is the default single repo.
            self.assertEqual(config.repo, (root / "frontend").resolve())

    def test_single_repo_is_exposed_as_main(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_file = root / "config.toml"
            config_file.write_text(f'repo = "{root}"\n')
            config = load_config(config_file)
            self.assertEqual(config.repos, {"main": root.resolve()})

    def test_missing_named_repo_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_file = root / "config.toml"
            config_file.write_text(
                f'[repos]\ngood = "{root}"\nbad = "{root / "nope"}"\n'
            )
            with self.assertRaises(AgentShipError):
                load_config(config_file)

    def test_rejects_unknown_provider(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_file = root / "config.toml"
            config_file.write_text(f'repo = "{root}"\n[implementer]\nname = "other"\n')
            with self.assertRaises(AgentShipError):
                load_config(config_file)

    def test_chatgpt_is_accepted_for_every_role(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_file = root / "config.toml"
            config_file.write_text(
                f'repo = "{root}"\n'
                '[implementer]\nname = "chatgpt"\n'
                '[reviewer]\nname = "chatgpt"\n'
            )
            config = load_config(config_file)
            self.assertEqual(config.implementer.name, "chatgpt")
            self.assertEqual(config.reviewer.name, "chatgpt")

    def test_reasoning_effort_defaults_low_and_parses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_file = root / "config.toml"
            config_file.write_text(f'repo = "{root}"\n')
            self.assertEqual(load_config(config_file).implementer.reasoning_effort, "low")
            config_file.write_text(
                f'repo = "{root}"\n[implementer]\n'
                'name = "chatgpt"\nreasoning_effort = "medium"\n'
            )
            self.assertEqual(
                load_config(config_file).implementer.reasoning_effort, "medium"
            )

    def test_invalid_reasoning_effort_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_file = root / "config.toml"
            config_file.write_text(
                f'repo = "{root}"\n[implementer]\nreasoning_effort = "turbo"\n'
            )
            with self.assertRaisesRegex(AgentShipError, "reasoning_effort"):
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
