from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pr_pilot.config import (
    BabysitConfig,
    Config,
    GitHubConfig,
    MemoryConfig,
    WorkflowConfig,
)
from pr_pilot.errors import AgentShipError
from pr_pilot.github import PullRequestStatus
from pr_pilot.memory import Project
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
        self.checked_out_bases = []
        self.detected_branch = None  # None => fall back to config base_branch

    def validate(self):
        pass

    def default_branch(self):
        return self.detected_branch

    def fingerprint(self):
        return "unchanged"

    def create_branch(self, feature, base):
        return "agent/test-branch"

    def start_branch(self, branch, base):
        self.checked_out_bases.append(base)
        self.branch = branch

    def checkout_base(self, base):
        self.checked_out_bases.append(base)

    def reset_to_base(self, base):
        self.checked_out_bases.append(base)

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
        self.created = []

    def create_pr(self, **kwargs):
        self.created.append(kwargs)
        return "https://github.test/pull/1"

    def status(self):
        return next(self.statuses)


class FakeMemoryDB:
    def __init__(self, project):
        self.project = project

    def project_for_path(self, path):
        return self.project


class FakeMemory:
    def __init__(self, project):
        self.db = FakeMemoryDB(project)
        self.recorded = 0

    def index_related(self, project):
        return []

    def context(self, query, project):
        return "[orders] API.md:1-2\nUse versioned invoice events."

    def record_run(self, state):
        self.recorded += 1


class WorkflowTests(unittest.TestCase):
    def test_recommends_next_feature_from_repository_and_run_history(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = Config(
                repo=root,
                memory=MemoryConfig(enabled=False),
                state_dir=root / "state",
            )
            designer = FakeProvider(["Add a dry-run mode for publishing changes."])
            workflow = Workflow(
                config,
                implementer=FakeProvider([]),
                reviewer=FakeProvider([]),
                designer=designer,
            )
            repo = FakeRepo()
            workflow.repo = repo
            workflow.store.save(RunState("previous", "Add JSON output", str(root)))

            feature = workflow.recommend_feature()

            self.assertEqual(feature, "Add a dry-run mode for publishing changes.")
            self.assertEqual(repo.checked_out_bases, ["main"])
            # Recommendation runs read-only through the designer, and past runs
            # are fed in so it doesn't repeat them.
            self.assertEqual(designer.calls[0][1], False)
            self.assertIn("Add JSON output", designer.calls[0][0])

    def test_detected_default_branch_overrides_config_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = Config(
                repo=root,
                memory=MemoryConfig(enabled=False),
                state_dir=root / "state",
            )
            workflow = Workflow(
                config,
                implementer=FakeProvider([]),
                reviewer=FakeProvider([]),
                designer=FakeProvider(["a feature"]),
            )
            repo = FakeRepo()
            repo.detected_branch = "master"  # e.g. a master-default frontend
            workflow.repo = repo

            workflow.recommend_feature()

            # The repo's real default wins over the configured "main".
            self.assertEqual(repo.checked_out_bases, ["master"])

    def test_recommendation_can_stop_the_autonomous_loop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = Config(
                repo=root,
                memory=MemoryConfig(enabled=False),
                state_dir=root / "state",
            )
            workflow = Workflow(
                config,
                implementer=FakeProvider([]),
                reviewer=FakeProvider([]),
                designer=FakeProvider(["NO_FEATURE"]),
            )
            workflow.repo = FakeRepo()

            self.assertIsNone(workflow.recommend_feature())

    def test_full_run_uses_separate_review_and_opens_pr(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Config(
                repo=Path(directory),
                github=GitHubConfig(draft=True),
                babysit=BabysitConfig(enabled=False),
                memory=MemoryConfig(enabled=False),
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

    def test_group_opens_a_pr_per_changed_member(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fe, be = root / "frontend", root / "backend"
            config = Config(
                repo=fe,
                repos={"frontend": fe, "backend": be},
                workspace=root,
                github=GitHubConfig(draft=True),
                babysit=BabysitConfig(enabled=False),
                memory=MemoryConfig(enabled=False),
                state_dir=root / "state",
            )
            implementer = FakeProvider(["the plan", "implemented"])
            reviewer = FakeProvider(["ok\nVERDICT: APPROVE"])
            workflow = Workflow(config, implementer=implementer, reviewer=reviewer)
            fe_repo, be_repo = FakeRepo(), FakeRepo()
            fe_repo.changed = True  # only the frontend was edited
            fe_gh, be_gh = FakeGitHub(), FakeGitHub()
            workflow.members = {"frontend": fe_repo, "backend": be_repo}
            workflow.member_github = {"frontend": fe_gh, "backend": be_gh}

            state = workflow.run("Add full-stack search", watch=False)

            # A PR only for the repo that changed; the untouched one is reset.
            self.assertEqual(len(fe_gh.created), 1)
            self.assertEqual(len(be_gh.created), 0)
            self.assertEqual(len(fe_repo.commits), 1)
            self.assertEqual(len(be_repo.commits), 0)
            self.assertIn("frontend:", state.pr_url)
            # The agent ran once in the shared workspace per phase (plan, implement).
            self.assertEqual([call[1] for call in implementer.calls], [False, True])

    def test_review_and_repair_loops_until_approval_before_opening_pr(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Config(
                repo=Path(directory),
                workflow=WorkflowConfig(max_review_attempts=2),
                babysit=BabysitConfig(enabled=False),
                memory=MemoryConfig(enabled=False),
                state_dir=Path(directory) / "state",
            )
            implementer = FakeProvider(["the plan", "implemented", "repair 1", "repair 2"])
            reviewer = FakeProvider(
                [
                    "First finding\nVERDICT: CHANGES_REQUESTED",
                    "Second finding\nVERDICT: CHANGES_REQUESTED",
                    "No findings\nVERDICT: APPROVE",
                ]
            )
            workflow = Workflow(config, implementer=implementer, reviewer=reviewer)
            workflow.repo = FakeRepo()
            workflow.repo.changed = True
            github = FakeGitHub()
            workflow.github = github

            state = workflow.run("Add a loop", watch=False)

            self.assertEqual(state.phase, "pr_open")
            self.assertEqual(state.review_attempts, 2)
            self.assertEqual([call[1] for call in implementer.calls], [False, True, True, True])
            self.assertEqual(len(reviewer.calls), 3)
            self.assertEqual(len(github.created), 1)

    def test_review_loop_stops_before_pr_on_invalid_follow_up_verdict(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Config(
                repo=Path(directory),
                workflow=WorkflowConfig(max_review_attempts=2),
                babysit=BabysitConfig(enabled=False),
                memory=MemoryConfig(enabled=False),
                state_dir=Path(directory) / "state",
            )
            workflow = Workflow(
                config,
                implementer=FakeProvider(["the plan", "implemented", "repaired"]),
                reviewer=FakeProvider(
                    [
                        "Finding\nVERDICT: CHANGES_REQUESTED",
                        "Review completed, but I cannot give VERDICT: APPROVE",
                    ]
                ),
            )
            workflow.repo = FakeRepo()
            workflow.repo.changed = True
            github = FakeGitHub()
            workflow.github = github

            with self.assertRaisesRegex(AgentShipError, "required verdict"):
                workflow.run("Add a loop", watch=False)

            self.assertEqual(github.created, [])

    def test_babysitter_fixes_failure_then_completes(self):
        with tempfile.TemporaryDirectory() as directory:
            config = Config(
                repo=Path(directory),
                babysit=BabysitConfig(interval_seconds=0, max_cycles=2),
                memory=MemoryConfig(enabled=False),
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

    def test_memory_context_is_injected_into_plan_and_review(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = Config(
                repo=root,
                github=GitHubConfig(draft=True),
                babysit=BabysitConfig(enabled=False),
                memory=MemoryConfig(enabled=True, database=root / "memory.db"),
                state_dir=root / "state",
            )
            project = Project("id", "active", root, "", "", "", "")
            memory = FakeMemory(project)
            implementer = FakeProvider(["the plan", "implemented"])
            reviewer = FakeProvider(["VERDICT: APPROVE"])
            workflow = Workflow(
                config, implementer=implementer, reviewer=reviewer, memory=memory
            )
            workflow.repo = FakeRepo()
            workflow.repo.changed = True
            workflow.github = FakeGitHub()

            workflow.run("Add invoices", watch=False)

            self.assertIn("UNTRUSTED PROJECT MEMORY", implementer.calls[0][0])
            self.assertIn("versioned invoice events", reviewer.calls[0][0])
            self.assertEqual(memory.recorded, 3)


if __name__ == "__main__":
    unittest.main()
