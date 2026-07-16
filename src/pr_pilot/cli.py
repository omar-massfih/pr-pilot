from __future__ import annotations

import argparse
import shutil
import sys
from dataclasses import replace
from pathlib import Path

from .config import ProviderConfig, load_config
from .errors import AgentShipError
from .state import StateStore
from .telegram import TelegramBot
from .workflow import Workflow


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

    watch = commands.add_parser("watch", help="Resume babysitting a previous run")
    watch.add_argument("run_id")

    commands.add_parser("telegram", help="Run Telegram long-polling intake")
    commands.add_parser("doctor", help="Check local command dependencies")
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        config = load_config(args.config, args.repo)
        if args.command == "doctor":
            return doctor(config)
        if args.command == "run":
            if args.provider:
                config = replace(config, implementer=ProviderConfig(args.provider))
            if args.reviewer:
                config = replace(config, reviewer=ProviderConfig(args.reviewer))
            state = Workflow(config).run(args.feature, watch=not args.no_watch)
            print(f"Run: {state.run_id}\nPR: {state.pr_url}\nState: {state.phase}")
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
    required = ["git", "gh", "codex" if config.implementer.name == "codex" else "cursor-agent"]
    reviewer = "codex" if config.reviewer.name == "codex" else "cursor-agent"
    required.append(reviewer)
    missing = sorted({command for command in required if shutil.which(command) is None})
    if missing:
        raise AgentShipError("Missing commands: " + ", ".join(missing))
    print("Dependencies found: " + ", ".join(sorted(set(required))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
