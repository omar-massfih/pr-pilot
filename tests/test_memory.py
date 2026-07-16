from __future__ import annotations

import json
import sqlite3
import struct
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pr_pilot.config import Config, MemoryConfig
from pr_pilot.errors import AgentShipError
from pr_pilot.memory import MemoryDB, MemoryService
from pr_pilot.state import RunState


class FakeProfiler:
    def __init__(self, domain: str = "billing"):
        self.domain = domain
        self.calls = 0
        self.snapshot_paths = []
        self.snapshot_readmes = []

    def invoke(self, prompt, *, repo, write):
        self.calls += 1
        self.snapshot_paths.append(repo)
        readme = repo / "README.md"
        self.snapshot_readmes.append(readme.read_text() if readme.exists() else "")
        return json.dumps(
            {
                "summary": f"A Python {self.domain} service",
                "tags": [
                    {"name": "lang:python", "confidence": 1, "evidence": "app.py"},
                    {"name": f"domain:{self.domain}", "confidence": 0.9, "evidence": "README.md"},
                ],
                "artifacts": [{"name": repo.name, "kind": "service"}],
                "dependencies": [],
                "relationships": [],
            }
        )


class FakeEmbedder:
    error = None
    model_name = "fake"

    def embed(self, texts):
        vectors = []
        for text in texts:
            lower = text.lower()
            values = [
                float("billing" in lower), float("python" in lower),
                float("search" in lower), 0.1,
            ]
            length = sum(value * value for value in values) ** 0.5
            vectors.append(struct.pack("4f", *(value / length for value in values)))
        return vectors


class DegradedEmbedder:
    error = "model unavailable"
    model_name = "missing"

    def embed(self, texts):
        return None


class RetryProfiler(FakeProfiler):
    def invoke(self, prompt, *, repo, write):
        self.calls += 1
        if self.calls == 1:
            return "not json"
        self.calls -= 1
        return super().invoke(prompt, repo=repo, write=write)


class FailingProfiler:
    def __init__(self):
        self.snapshot_path = None

    def invoke(self, prompt, *, repo, write):
        self.snapshot_path = repo
        raise RuntimeError("profiler failed")


def make_repo(root: Path, name: str) -> Path:
    repo = root / name
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=repo, check=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, check=True)
    (repo / "README.md").write_text("# Billing\n\nHandles invoice search and payment records.\n")
    (repo / "app.py").write_text("def search_invoice(invoice_id):\n    return invoice_id\n")
    (repo / ".env").write_text("SECRET=do-not-index\n")
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, check=True, capture_output=True)
    return repo


class MemoryTests(unittest.TestCase):
    def service(self, root: Path, repo: Path, profiler=None):
        config = Config(
            repo=repo,
            memory=MemoryConfig(database=root / "memory.db", relationship_threshold=0.8),
            state_dir=root / "state",
        )
        service = MemoryService(config, profiler=profiler or FakeProfiler())
        service.embedder = FakeEmbedder()
        self.addCleanup(service.db.close)
        return service

    def test_indexes_tracked_text_searches_and_excludes_secrets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = make_repo(root, "payments")
            service = self.service(root, repo)
            project = service.add_project(repo)

            result = service.index_project(project.id)

            self.assertEqual(result["changed_files"], 2)
            paths = {
                row["path"] for row in service.db.connection.execute("SELECT path FROM documents")
            }
            self.assertNotIn(".env", paths)
            self.assertEqual({"README.md", "app.py"}, paths)
            hits = service.search("invoice search", project_ref=project.id)
            self.assertTrue(hits)
            self.assertEqual(hits[0].project, "payments")
            tags = service.graph(project.id)["nodes"][0]["tags"]
            self.assertIn("domain:billing", tags)

    def test_incremental_index_and_run_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = make_repo(root, "payments")
            profiler = FakeProfiler()
            service = self.service(root, repo, profiler)
            project = service.add_project(repo)
            service.index_project(project.id)

            second = service.index_project(project.id)
            state = RunState("run-1", "Add refunds", str(repo), plan="Implement refund ledger")
            service.record_run(state)
            hits = service.search("refund ledger", project_ref=project.id)

            self.assertEqual(second["changed_files"], 0)
            self.assertEqual(profiler.calls, 1)
            self.assertTrue(any(hit.source_type == "run" for hit in hits))

    def test_generated_relationship_and_manual_block(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_repo = make_repo(root, "payments")
            second_repo = make_repo(root, "orders")
            service = self.service(root, first_repo)
            first = service.add_project(first_repo)
            second = service.add_project(second_repo)
            service.index_project(first.id)
            service.index_project(second.id)

            graph = service.graph()
            related = [edge for edge in graph["edges"] if edge["relation_type"] == "related_to"]
            self.assertEqual(len(related), 1)
            service.remove_link(first.id, "related_to", second.id)
            service._rebuild_generated_relationships()
            self.assertFalse(service.graph()["edges"])

    def test_keyword_search_survives_embedding_failure_and_profiler_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = make_repo(root, "payments")
            profiler = RetryProfiler()
            service = self.service(root, repo, profiler)
            service.embedder = DegradedEmbedder()
            project = service.add_project(repo)

            result = service.index_project(project.id)
            hits = service.search("invoice", project_ref=project.id)

            self.assertFalse(result["semantic"])
            self.assertTrue(hits)
            self.assertEqual(profiler.calls, 2)

    def test_manual_tag_survives_reprofiling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = make_repo(root, "payments")
            service = self.service(root, repo)
            project = service.add_project(repo)
            service.add_tag(project.id, "domain:finance")
            service.index_project(project.id, force=True)

            tags = service.graph(project.id)["nodes"][0]["tags"]
            self.assertIn("domain:finance", tags)

    def test_migrates_existing_projects_to_head_ref(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.db"
            connection = sqlite3.connect(path)
            connection.executescript(
                """
                CREATE TABLE schema_meta(version INTEGER NOT NULL);
                INSERT INTO schema_meta VALUES(1);
                CREATE TABLE projects(
                    id TEXT PRIMARY KEY, name TEXT NOT NULL, path TEXT NOT NULL UNIQUE,
                    remote_url TEXT NOT NULL DEFAULT '', description TEXT NOT NULL DEFAULT '',
                    indexed_commit TEXT NOT NULL DEFAULT '', profile_commit TEXT NOT NULL DEFAULT '',
                    profile_json TEXT NOT NULL DEFAULT '{}', created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                INSERT INTO projects(id,name,path,created_at,updated_at)
                VALUES('id','legacy','/tmp/legacy','now','now');
                """
            )
            connection.commit()
            connection.close()

            database = MemoryDB(path)
            self.addCleanup(database.close)

            row = database.connection.execute(
                "SELECT index_ref FROM projects WHERE id='id'"
            ).fetchone()
            version = database.connection.execute("SELECT version FROM schema_meta").fetchone()[0]
            self.assertEqual(row["index_ref"], "HEAD")
            self.assertEqual(version, 2)

    def test_default_ref_uses_isolated_snapshot_and_preserves_worktree(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = make_repo(root, "payments")
            main_commit = subprocess.run(
                ["git", "rev-parse", "main"], cwd=repo, text=True,
                capture_output=True, check=True,
            ).stdout.strip()
            subprocess.run(
                ["git", "update-ref", "refs/remotes/origin/main", main_commit],
                cwd=repo, check=True,
            )
            subprocess.run(
                ["git", "symbolic-ref", "refs/remotes/origin/HEAD", "refs/remotes/origin/main"],
                cwd=repo, check=True,
            )
            subprocess.run(["git", "switch", "-c", "feature"], cwd=repo, check=True, capture_output=True)
            (repo / "README.md").write_text("UNCOMMITTED FEATURE CONTENT\n")
            profiler = FakeProfiler()
            service = self.service(root, repo, profiler)

            project = service.add_project(repo, index_ref="default")
            result = service.index_project(project.id)

            self.assertEqual(project.index_ref, "origin/main")
            self.assertEqual(result["commit"], main_commit)
            self.assertEqual(
                subprocess.run(
                    ["git", "branch", "--show-current"], cwd=repo, text=True,
                    capture_output=True, check=True,
                ).stdout.strip(),
                "feature",
            )
            self.assertEqual((repo / "README.md").read_text(), "UNCOMMITTED FEATURE CONTENT\n")
            self.assertNotIn("UNCOMMITTED", profiler.snapshot_readmes[0])
            self.assertFalse(profiler.snapshot_paths[0].exists())

    def test_fetch_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = make_repo(root, "payments")
            service = self.service(root, repo)
            project = service.add_project(repo)
            with patch.object(service, "_fetch") as fetch:
                service.index_project(project.id)
                fetch.assert_not_called()
                service.index_project(project.id, fetch=True)
                fetch.assert_called_once_with(repo.resolve())

    def test_ref_updates_validate_and_reregister_preserves_ref(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = make_repo(root, "payments")
            service = self.service(root, repo)
            project = service.add_project(repo, index_ref="main")

            same = service.add_project(repo)
            with self.assertRaises(AgentShipError):
                service.set_project_ref(project.id, "missing-ref")

            self.assertEqual(same.index_ref, "main")
            updated = service.set_project_ref(project.id, "HEAD")
            self.assertEqual(updated.index_ref, "HEAD")

    def test_snapshot_is_cleaned_when_profiler_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = make_repo(root, "payments")
            profiler = FailingProfiler()
            service = self.service(root, repo, profiler)
            project = service.add_project(repo)

            with self.assertRaisesRegex(RuntimeError, "profiler failed"):
                service.index_project(project.id)

            self.assertIsNotNone(profiler.snapshot_path)
            self.assertFalse(profiler.snapshot_path.exists())


if __name__ == "__main__":
    unittest.main()
