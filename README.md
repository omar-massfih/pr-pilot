# PR Pilot

PR Pilot turns a feature request from a CLI or Telegram into a reviewed GitHub pull request. It
uses either Codex CLI or Cursor Agent CLI for planning and implementation, starts a separate agent
session for code review, watches the PR for CI failures and reviewer feedback, and remembers
architecture and decisions across related projects.

The MVP deliberately keeps Git, GitHub, state, and policy in this program. Coding agents only plan,
edit, test, and review. That makes the workflow provider-neutral and prevents an agent from pushing
or opening PRs on its own.

## Workflow

```text
CLI / Telegram
      |
      v
plan (read-only agent) -> implement (write agent) -> independent review
                              ^                       |          |
                              |                       | changes  | approved
                              +-------- repair <------+          v
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
- Enough disk space for the local FastEmbed model downloaded during memory setup

Codex runs planning/review in its read-only sandbox and implementation in `workspace-write`.
Cursor uses print mode without `--force` for planning/review, and adds `--force` only for work that
must edit files. Cursor CLI is currently documented as beta, so pin and test CLI upgrades in your
deployment.

Provider quota and rate-limit responses are retried automatically. The default polls every 60
seconds and waits indefinitely until Codex/Cursor accepts the invocation; `Ctrl-C` always stops it.
Configure `limit_poll_seconds` and `limit_max_wait_seconds` under `[implementer]` or `[reviewer]`.
A maximum of `0` means no deadline. Authentication, permission, and other errors fail immediately.

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
pr-pilot memory setup
```

## Project memory and local RAG

Register each repository once. Registration indexes committed, tracked text at `HEAD`, asks the
configured read-only coding agent to create a project profile, and relates the project to the other
registered projects:

```bash
pr-pilot --repo /work/payments project add --ref default --fetch
pr-pilot --repo /work/orders project add --ref default --fetch
pr-pilot project list
pr-pilot project show payments
```

Memory is stored locally in `~/.pr-pilot/memory.db`. Keyword retrieval uses SQLite FTS5 and semantic
retrieval uses the local `BAAI/bge-small-en-v1.5` FastEmbed model. Repository content is not sent to
an embedding API. Agent-assisted project profiling does use the configured Codex or Cursor provider.

Projects index their configured Git ref. `HEAD` is the backward-compatible default; `--ref default`
resolves the cached `origin/HEAD` (normally `origin/main`). Profiling runs from a temporary detached
clone of the resolved commit, so feature branches and uncommitted files in your checkout are neither
indexed nor shown to the profiler. Automatic workflow refreshes stay offline; use `--fetch` when you
want PR Pilot to update `origin` first.

Refresh and search memory:

```bash
pr-pilot memory index --all --fetch
pr-pilot memory search "how are invoice events versioned?"
pr-pilot memory search "refund ledger" --project payments --tag domain:billing
pr-pilot memory graph payments --depth 2
pr-pilot memory stats
```

Change an existing project's durable ref without re-registering it:

```bash
pr-pilot project ref set payments default
pr-pilot memory index payments --fetch
```

Generated tags use namespaced values such as `lang:python`, `framework:django`, `domain:billing`,
and `kind:service`. Relationships are evidence-backed `depends_on`, `integrates_with`, `replaces`,
or `related_to` edges. Correct generated metadata with durable overrides:

```bash
pr-pilot project tag add payments domain:finance
pr-pilot project tag remove payments domain:legacy
pr-pilot project link add orders depends_on payments
pr-pilot project link remove orders related_to website
```

Before a feature run, PR Pilot incrementally refreshes a registered active project and its one-hop
neighbors. It injects bounded, source-labeled memory into planning and review. Unregistered projects
continue through the original workflow without memory.

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
- Indexing excludes untracked files, common credentials, keys, binaries, generated/vendor trees,
  lockfiles, and tracked files larger than 512 KiB.
- Retrieved cross-project content is marked untrusted and cannot authorize edits outside the active
  repository.
- Automatic project profiling sends repository context to the selected Codex/Cursor provider;
  embeddings and search remain local.
- Start with draft PRs and branch protection enabled.
- Use separate bot/service credentials with least-privilege repository access.
- Before opening a PR, review and repair repeat up to `workflow.max_review_attempts` times.
- PR babysitting has bounded polling and repair attempts. It does not merge, approve, or bypass CI.
- A comment can trigger an agent assessment, but the prompt tells it to ignore non-actionable text.
- Run one PR Pilot process per target worktree. The MVP does not lock concurrent runs.

## Tests

```bash
python -m unittest discover -s tests -v
```
