from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from dataclasses import replace
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .config import load_config
from .errors import AgentShipError
from .memory import MemoryService, RELATION_TYPES
from .state import StateStore
from .telegram import TelegramBot
from .workflow import Workflow


_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"
# On-disk log ceiling: one active file plus its rotations. ~2 MB × (3 + 1) ≈ 8 MB
# total, so a long-running Telegram/auto service can't fill the disk.
_LOG_MAX_BYTES = 2_000_000
_LOG_BACKUPS = 3


def _setup_file_logging(state_dir: Path) -> None:
    """Persist the INFO log to a size-capped rotating file beside the run state.

    stderr already carries INFO (for `journalctl` / an attached terminal), but
    that stream is gone once the process exits — so a run's rounds, nudges, and
    outcomes can't be inspected after the fact. This adds a bounded on-disk copy
    (``state_dir/pr-pilot.log``, rotated) that survives. Best-effort: a log we
    can't open must never take the CLI down.
    """
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            state_dir / "pr-pilot.log",
            maxBytes=_LOG_MAX_BYTES,
            backupCount=_LOG_BACKUPS,
            encoding="utf-8",
        )
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logging.getLogger().addHandler(handler)
    except OSError as exc:
        logging.getLogger(__name__).warning(
            "could not open log file under %s: %s", state_dir, exc
        )


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="pr-pilot", description="Ship features with coding agents")
    root.add_argument("--config", type=Path, default=Path("pr-pilot.toml"))
    root.add_argument("--repo", type=Path)
    commands = root.add_subparsers(dest="command", required=True)

    run_command = commands.add_parser("run", help="Implement a feature and open a PR")
    run_command.add_argument("feature", help="Feature request text")
    run_command.add_argument("--provider", choices=("codex", "cursor"))
    run_command.add_argument("--reviewer", choices=("codex", "cursor"))
    run_command.add_argument("--no-watch", action="store_true")

    auto = commands.add_parser(
        "auto", help="Recommend, implement, and ship features continuously"
    )
    auto.add_argument("--provider", choices=("codex", "cursor"))
    auto.add_argument("--reviewer", choices=("codex", "cursor"))
    auto.add_argument(
        "--max-features",
        type=int,
        default=0,
        help="Stop after this many features; 0 continues until no feature is recommended",
    )

    watch = commands.add_parser("watch", help="Resume babysitting a previous run")
    watch.add_argument("run_id")

    commands.add_parser("telegram", help="Run Telegram long-polling intake")
    commands.add_parser("doctor", help="Check local command dependencies")

    project = commands.add_parser("project", help="Manage remembered projects")
    project_commands = project.add_subparsers(dest="project_command", required=True)
    project_add = project_commands.add_parser("add", help="Register and index a Git repository")
    project_add.add_argument("path", nargs="?", type=Path)
    project_add.add_argument("--ref", help="Git ref to index, or 'default' (new projects: HEAD)")
    project_add.add_argument("--fetch", action="store_true", help="Fetch origin before indexing")
    project_commands.add_parser("list", help="List registered projects")
    project_show = project_commands.add_parser("show", help="Show one project and its graph")
    project_show.add_argument("project")
    project_remove = project_commands.add_parser("remove", help="Forget a project and its index")
    project_remove.add_argument("project")
    project_ref = project_commands.add_parser("ref", help="Set a project's durable index ref")
    project_ref_commands = project_ref.add_subparsers(dest="ref_command", required=True)
    project_ref_set = project_ref_commands.add_parser("set")
    project_ref_set.add_argument("project")
    project_ref_set.add_argument("ref")
    tag = project_commands.add_parser("tag", help="Override generated project tags")
    tag_commands = tag.add_subparsers(dest="tag_command", required=True)
    for action in ("add", "remove"):
        tag_action = tag_commands.add_parser(action)
        tag_action.add_argument("project")
        tag_action.add_argument("tag")
    link = project_commands.add_parser("link", help="Override generated project relationships")
    link_commands = link.add_subparsers(dest="link_command", required=True)
    for action in ("add", "remove"):
        link_action = link_commands.add_parser(action)
        link_action.add_argument("source")
        link_action.add_argument("type", choices=sorted(RELATION_TYPES))
        link_action.add_argument("target")

    memory = commands.add_parser("memory", help="Index and search local project memory")
    memory_commands = memory.add_subparsers(dest="memory_command", required=True)
    memory_index = memory_commands.add_parser("index", help="Refresh project indexes")
    memory_index.add_argument("project", nargs="?")
    memory_index.add_argument("--all", action="store_true")
    memory_index.add_argument("--force", action="store_true")
    memory_index.add_argument("--fetch", action="store_true", help="Fetch origin before indexing")
    memory_search = memory_commands.add_parser("search", help="Search project memory")
    memory_search.add_argument("query")
    memory_search.add_argument("--project")
    memory_search.add_argument("--tag")
    memory_search.add_argument("--limit", type=int, default=10)
    memory_search.add_argument("--json", action="store_true")
    memory_graph = memory_commands.add_parser("graph", help="Show the relationship graph")
    memory_graph.add_argument("project", nargs="?")
    memory_graph.add_argument("--depth", type=int, default=1)
    memory_graph.add_argument("--json", action="store_true")
    memory_commands.add_parser("stats", help="Show memory database statistics")
    memory_commands.add_parser("setup", help="Download and verify the local embedding model")
    return root


def main(argv: list[str] | None = None) -> int:
    # INFO to stderr so a long-running `telegram`/`auto` service shows activity
    # in journalctl (commands, run phases, outcomes). Harmless for one-shots.
    logging.basicConfig(level=logging.INFO, format=_LOG_FORMAT)
    args = parser().parse_args(argv)
    try:
        config = load_config(args.config, args.repo)
        # Now that state_dir is known, mirror the INFO log to a bounded file there.
        _setup_file_logging(config.state_dir)
        if args.command == "doctor":
            return doctor(config)
        if args.command == "project":
            return project_command(config, args)
        if args.command == "memory":
            return memory_command(config, args)
        if args.command in {"run", "auto"}:
            if args.provider:
                config = replace(
                    config, implementer=replace(config.implementer, name=args.provider)
                )
            if args.reviewer:
                config = replace(
                    config, reviewer=replace(config.reviewer, name=args.reviewer)
                )
        if args.command == "run":
            state = Workflow(config).run(args.feature, watch=not args.no_watch)
            print(f"Run: {state.run_id}\nPR: {state.pr_url}\nState: {state.phase}")
            return 0
        if args.command == "auto":
            if args.max_features < 0:
                raise AgentShipError("--max-features cannot be negative")
            workflow = Workflow(config)
            completed = 0
            while not args.max_features or completed < args.max_features:
                feature = workflow.recommend_feature()
                if feature is None:
                    print("No additional scoped feature was recommended; stopping.")
                    return 0
                print(f"Recommended feature: {feature}", flush=True)
                state = workflow.run(feature, watch=True)
                completed += 1
                print(
                    f"Run: {state.run_id}\nPR: {state.pr_url}\nState: {state.phase}",
                    flush=True,
                )
            return 0
        if args.command == "watch":
            state = StateStore(config.state_dir / "runs").load(args.run_id)
            if Path(state.repo).resolve() != config.repo:
                config = config.with_repo(Path(state.repo))
            result = Workflow(config).babysit(state)
            print(f"PR: {result.pr_url}\nState: {result.phase}")
            return 0
        if args.command == "telegram":
            TelegramBot(config).serve_forever()
            return 0
    except KeyboardInterrupt:
        print("Stopped.", file=sys.stderr)
        return 130
    except AgentShipError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


def doctor(config) -> int:
    # chatgpt is not a binary — it is checked via its auth file below.
    binaries = {"codex": "codex", "cursor": "cursor-agent"}
    required = ["git", "gh"]
    for provider in (config.implementer, config.reviewer):
        binary = binaries.get(provider.name)
        if binary:
            required.append(binary)
    missing = sorted({command for command in required if shutil.which(command) is None})
    if missing:
        raise AgentShipError("Missing commands: " + ", ".join(missing))
    if "chatgpt" in {
        config.implementer.name,
        config.reviewer.name,
        config.memory.profile_provider,
    }:
        from .chatgpt import check_auth

        check_auth()
    if config.memory.enabled:
        try:
            import fastembed  # noqa: F401
            import numpy  # noqa: F401
        except ImportError as exc:
            raise AgentShipError(f"Missing memory dependency: {exc.name}") from exc
    print("Dependencies found: " + ", ".join(sorted(set(required))))
    return 0


def project_command(config, args) -> int:
    service = MemoryService(config)
    if args.project_command == "add":
        project = service.add_project(
            args.path or config.repo, index_ref=args.ref, fetch=args.fetch
        )
        result = service.index_project(project.id)
        print(json.dumps({"id": project.id, **result}, indent=2))
    elif args.project_command == "list":
        for project in service.db.list_projects():
            print(f"{project.id}\t{project.name}\t{project.index_ref}\t{project.path}")
    elif args.project_command == "show":
        project = service.db.resolve_project(args.project)
        payload = service.graph(project.id, depth=1)
        payload["project"] = {
            "id": project.id, "name": project.name, "path": str(project.path),
            "remote_url": project.remote_url, "description": project.description,
            "index_ref": project.index_ref, "indexed_commit": project.indexed_commit,
        }
        print(json.dumps(payload, indent=2))
    elif args.project_command == "remove":
        service.remove_project(args.project)
        print(f"Removed project: {args.project}")
    elif args.project_command == "ref":
        project = service.set_project_ref(args.project, args.ref)
        print(f"Project index ref: {project.name} -> {project.index_ref}")
    elif args.project_command == "tag":
        action = service.add_tag if args.tag_command == "add" else service.remove_tag
        action(args.project, args.tag)
        print(f"Tag {args.tag_command}: {args.tag}")
    elif args.project_command == "link":
        action = service.add_link if args.link_command == "add" else service.remove_link
        action(args.source, args.type, args.target)
        print(f"Relationship {args.link_command}: {args.source} {args.type} {args.target}")
    return 0


def memory_command(config, args) -> int:
    service = MemoryService(config)
    if args.memory_command == "index":
        if args.all:
            projects = service.db.list_projects()
        elif args.project:
            projects = [service.db.resolve_project(args.project)]
        else:
            project = service.db.project_for_path(config.repo)
            if not project:
                raise AgentShipError("Current repository is not registered; run `pr-pilot project add`")
            projects = [project]
        results = [
            service.index_project(project.id, force=args.force, fetch=args.fetch)
            for project in projects
        ]
        print(json.dumps(results, indent=2))
    elif args.memory_command == "search":
        results = service.search(
            args.query, project_ref=args.project, tag=args.tag, limit=max(1, args.limit)
        )
        if args.json:
            print(json.dumps([result.__dict__ for result in results], indent=2))
        else:
            for result in results:
                print(
                    f"[{result.project}] {result.path}:{result.start_line}-{result.end_line} "
                    f"score={result.score:.4f}\n{result.content}\n"
                )
    elif args.memory_command == "graph":
        graph = service.graph(args.project, args.depth)
        if args.json:
            print(json.dumps(graph, indent=2))
        else:
            names = {node["id"]: node["name"] for node in graph["nodes"]}
            for node in graph["nodes"]:
                print(f"{node['name']} [{', '.join(node['tags'])}]")
            for edge in graph["edges"]:
                print(
                    f"{names[edge['source_id']]} --{edge['relation_type']}--> "
                    f"{names[edge['target_id']]} ({edge['confidence']:.2f})"
                )
    elif args.memory_command == "stats":
        print(json.dumps(service.stats(), indent=2))
    elif args.memory_command == "setup":
        vectors = service.embedder.embed(["PR Pilot local memory setup"])
        if not vectors:
            raise AgentShipError("Embedding model setup failed: " + str(service.embedder.error))
        print(f"Embedding model ready: {config.memory.embedding_model}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
