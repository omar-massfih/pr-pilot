# PR Pilot

PR Pilot turns a feature request from a CLI or Telegram into a reviewed GitHub pull request. It
uses either Codex CLI or Cursor Agent CLI for planning and implementation, starts a separate agent
session for code review, and watches the PR for CI failures and reviewer feedback.

The MVP deliberately keeps Git, GitHub, state, and policy in this program. Coding agents only plan,
edit, test, and review. That makes the workflow provider-neutral and prevents an agent from pushing
or opening PRs on its own.

## Workflow

```text
CLI / Telegram
      |
      v
plan (read-only agent) -> implement (write agent) -> independent review
                                                      |
                                                      v
                                          commit -> push -> draft PR
                                                      |
                                                      v
                              poll checks/reviews -> fix -> push -> repeat
```

Each run is stored as JSON under `~/.pr-pilot/runs`, so PR monitoring can be resumed after a
restart. The target repository must begin with a clean worktree. PR Pilot creates a branch named
`agent/<feature>-<timestamp>` and never merges or closes the PR.

## Prerequisites

- Python 3.11+
- Git and an `origin` remote
- [GitHub CLI](https://cli.github.com/) authenticated with `gh auth login`
- At least one authenticated coding CLI:
  - [Codex CLI](https://developers.openai.com/codex/cli/) (`codex`)
  - [Cursor Agent CLI](https://docs.cursor.com/en/cli/overview) (`cursor-agent`)

Codex runs planning/review in its read-only sandbox and implementation in `workspace-write`.
Cursor uses print mode without `--force` for planning/review, and adds `--force` only for work that
must edit files. Cursor CLI is currently documented as beta, so pin and test CLI upgrades in your
deployment.

## Install and configure

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
cp pr-pilot.toml.example pr-pilot.toml
```

Edit `pr-pilot.toml`, especially `repo`, provider choices, base branch, and Telegram allowlist.
The local config is gitignored because it contains machine-specific details.

Validate installed commands:

```bash
pr-pilot doctor
```

## CLI intake

Run the full workflow and babysit checks until they settle:

```bash
pr-pilot run "Add CSV export to the invoices page"
```

Override providers for a run, or return immediately after opening the PR:

```bash
pr-pilot run --provider codex --reviewer cursor "Add CSV export"
pr-pilot run --no-watch "Add CSV export"
```

Resume PR monitoring using the run ID printed by the first command:

```bash
pr-pilot watch 20260716123000123456
```

## Telegram intake

Create a bot with BotFather, put only its token in the environment, and add your numeric Telegram
chat ID to `telegram.allowed_chat_ids`. Messages from all other chats are ignored.

```bash
export TELEGRAM_BOT_TOKEN="..."
pr-pilot telegram
```

Send:

```text
/feature Add CSV export to the invoices page
```

The Telegram MVP processes one request at a time. For production, run the workflow jobs in isolated
containers or per-run Git worktrees and put intake onto a durable queue.

## Safety and operating boundaries

- Run this only against repositories you trust. Coding agents can execute repository commands.
- Start with draft PRs and branch protection enabled.
- Use separate bot/service credentials with least-privilege repository access.
- The program stops before opening a PR if the reviewer still requests changes after one repair pass.
- PR babysitting has bounded polling and repair attempts. It does not merge, approve, or bypass CI.
- A comment can trigger an agent assessment, but the prompt tells it to ignore non-actionable text.
- Run one PR Pilot process per target worktree. The MVP does not lock concurrent runs.

## Tests

```bash
python -m unittest discover -s tests -v
```
