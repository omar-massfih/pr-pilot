from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pr_pilot.config import BabysitConfig, Config, GitHubConfig
from pr_pilot.github import PullRequestStatus
from pr_pilot.state import RunState
from pr_pilot.workflow import Workflow


class FakeProvider:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def invoke(self, prompt, *, repo, write):
        self.calls.append((prompt, write))
        return next(self.responses)


class FakeRepo:
    def __init__(self):
        self.changed = False
        self.commits = []
        self.pushes = []

    def validate(self):
        pass

    def fingerprint(self):
        return "unchanged"

    def create_branch(self, feature, base):
        return "agent/test-branch"

    def has_changes(self):
        return self.changed

    def commit(self, message):
        self.commits.append(message)
        self.changed = False

    def push(self, branch):
        self.pushes.append(branch)


class FakeGitHub:
    def __init__(self, statuses=None):
        self.statuses = iter(statuses or [])

    def create_pr(self, **kwargs):
        return "https://github.test/pull/1"

    def status(self):
        return next(self.statuses)


class WorkflowTests(unittest.TestCase):
    def test_full_run_uses_separate_review_and_opens_pr(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Config(
                repo=Path(directory),
                github=GitHubConfig(draft=True),
                babysit=BabysitConfig(enabled=False),
                state_dir=Path(directory) / "state",
            )
            implementer = FakeProvider(["the plan", "implemented"])
            reviewer = FakeProvider(["No findings\nVERDICT: APPROVE"])
            workflow = Workflow(config, implementer=implementer, reviewer=reviewer)
            workflow.repo = FakeRepo()
            workflow.repo.changed = True
            workflow.github = FakeGitHub()

            state = workflow.run("Add a thing", watch=False)

            self.assertEqual(state.phase, "pr_open")
            self.assertEqual(state.pr_url, "https://github.test/pull/1")
            self.assertEqual([call[1] for call in implementer.calls], [False, True])
            self.assertEqual([call[1] for call in reviewer.calls], [False])
            self.assertEqual(len(workflow.repo.commits), 1)

    def test_babysitter_fixes_failure_then_completes(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Config(
                repo=Path(directory),
                babysit=BabysitConfig(interval_seconds=0, max_cycles=2),
                state_dir=Path(directory) / "state",
            )
            implementer = FakeProvider(["fixed"])
            workflow = Workflow(config, implementer=implementer, reviewer=FakeProvider([]), sleeper=lambda _: None)
            repo = FakeRepo()
            repo.changed = True
            workflow.repo = repo
            workflow.github = FakeGitHub(
                [
                    PullRequestStatus("url", "", (), ("tests",), ()),
                    PullRequestStatus("url", "APPROVED", (), (), ()),
                ]
            )
            state = RunState("run", "feature", str(config.repo), branch="agent/test", pr_url="url")

            result = workflow.babysit(state)

            self.assertEqual(result.phase, "complete")
            self.assertEqual(result.fix_attempts, 1)
            self.assertEqual(len(repo.commits), 1)
            self.assertEqual(repo.pushes, ["agent/test"])


if __name__ == "__main__":
    unittest.main()
