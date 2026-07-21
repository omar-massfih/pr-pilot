from __future__ import annotations

import json
import logging
import os
import queue
import threading
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import replace

from .config import Config
from .errors import AgentShipError
from .workflow import Workflow

logger = logging.getLogger(__name__)

HELP = (
    "PR Pilot commands:\n"
    "/auto — suggest the next feature and wait for your go-ahead\n"
    "/yes — build the suggested feature, then suggest the next one\n"
    "/no — skip it and suggest a different feature\n"
    "/edit <description> — revise the suggestion before building "
    "(or just send the revised text)\n"
    "/repo [name] — list repos, or switch which one you're targeting\n"
    "/stop — end the auto loop\n"
    "/cancel — stop the run currently in progress\n"
    "/feature <description> — build a specific feature now"
)


class TelegramBot:
    """A long-polling bot that ships features, one confirmation at a time.

    Beyond the direct ``/feature <description>`` command, ``/auto`` starts an
    interactive loop: the bot recommends a feature, waits for ``/yes`` (build
    it, then suggest the next) or ``/no`` (suggest a different one), so the
    autonomous ``auto`` flow runs with a human gate on every change. State is
    just the pending suggestion held in memory; a restart resets it and the
    user runs ``/auto`` again.
    """

    def __init__(
        self,
        config: Config,
        workflow_factory: Callable[[], Workflow] | None = None,
        *,
        transport: Callable[[str, dict], dict] | None = None,
    ):
        self.config = config
        self.token = os.environ.get(config.telegram.token_env, "")
        if not self.token:
            raise AgentShipError(f"Missing Telegram token environment variable: {config.telegram.token_env}")
        if not config.telegram.allowed_chat_ids:
            raise AgentShipError("telegram.allowed_chat_ids must contain at least one trusted chat ID")
        self.allowed = set(config.telegram.allowed_chat_ids)
        # Targets a command acts on. In group mode the members are worked together
        # as one target (the workspace); otherwise each repo is its own target,
        # switchable with /repo. Each target gets its own lazily-built Workflow.
        if config.workspace is not None:
            self.repos = {config.workspace.name or "group": config.workspace}
        else:
            self.repos = dict(config.repos) or {"main": config.repo}
        self.active = next(iter(self.repos))
        self.workflow_factory = workflow_factory or (
            lambda repo_path: Workflow(replace(config, repo=repo_path))
        )
        self._transport = transport
        self._workflows: dict[str, Workflow] = {}
        self.pending: str | None = None
        # A plan awaiting /yes, kept alongside self.pending when plan_approval is
        # on, so an approved feature is built from the plan already shown (and not
        # planned a second time).
        self._pending_plan: str | None = None
        # Long builds run on one background worker so the poll loop keeps handling
        # /stop, /cancel, and status while a run is in flight. Tests that call
        # _handle directly never start the worker, so work runs inline for them.
        self._jobs: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._cancel = threading.Event()
        self._busy = False
        self._progress_chat: int | None = None
        # True while an /auto suggestion loop is active, so /yes and /no continue
        # proposing; a gated /feature sets it False so approval builds just that.
        self._auto_loop = False
        # Resume from the last confirmed update so a restart neither reprocesses
        # old commands (a re-run /feature could rebuild) nor drops new ones.
        self.offset = self._load_offset()

    @property
    def _offset_path(self):
        return self.config.state_dir / "telegram_offset"

    def _load_offset(self) -> int:
        try:
            return int(self._offset_path.read_text().strip())
        except (OSError, ValueError):
            return 0

    def _save_offset(self) -> None:
        try:
            self.config.state_dir.mkdir(parents=True, exist_ok=True)
            self._offset_path.write_text(str(self.offset))
        except OSError:
            logger.warning("could not persist telegram offset", exc_info=True)

    def serve_forever(self) -> None:
        logger.info(
            "telegram bot polling; repo=%s allowed_chats=%d offset=%d",
            self.config.repo, len(self.allowed), self.offset,
        )
        self._start_worker()
        while True:
            updates = self._call("getUpdates", {"timeout": 30, "offset": self.offset})["result"]
            for update in updates:
                self.offset = max(self.offset, int(update["update_id"]) + 1)
                # Persist before handling so a message that crashes the handler
                # is skipped on restart instead of wedging in a crash loop.
                self._save_offset()
                self._handle(update)

    def _start_worker(self) -> None:
        if self._worker and self._worker.is_alive():
            return
        self._worker = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker.start()

    def _worker_loop(self) -> None:
        while True:
            fn, args = self._jobs.get()
            self._cancel.clear()
            self._busy = True
            try:
                fn(*args)
            except Exception:
                logger.exception("background job failed")
            finally:
                self._busy = False
                self._jobs.task_done()

    def _dispatch(self, fn: Callable, *args) -> None:
        """Run heavy (workflow-invoking) work on the worker when it's running so
        polling stays responsive; otherwise inline — the path tests take when they
        call _handle directly without serve_forever."""
        if self._worker and self._worker.is_alive():
            self._jobs.put((fn, args))
        else:
            fn(*args)

    def _on_phase(self, phase: str, state) -> None:
        """Progress hook the workflow fires at each phase boundary. pr_open and
        complete are already announced by _build/babysit, so they're skipped."""
        chat = self._progress_chat
        if chat is None:
            return
        messages = {
            "planning": "📋 Planning…",
            "implementing": "🔧 Implementing…",
            "verifying": "✅ Running tests/lint/build…",
            "reviewing": f"🔍 Reviewing (attempt {state.review_attempts + 1})…",
            "repairing": "🛠 Addressing findings…",
            "publishing": "📤 Opening the pull request…",
            "babysitting": "👀 Watching CI…",
        }
        text = messages.get(phase)
        if text:
            self.send(chat, f"[{self.active}] {text}")

    def _workflow_instance(self) -> Workflow:
        # One workflow per repo, shared across that repo's recommend/run so they
        # use the same providers and memory; built lazily so construction errors
        # surface on use.
        if self.active not in self._workflows:
            wf = self.workflow_factory(self.repos[self.active])
            # The real Workflow exposes a progress hook and a cooperative cancel
            # probe; injected fakes may not, so wire them only when present.
            if hasattr(wf, "on_phase"):
                wf.on_phase = self._on_phase
            if hasattr(wf, "cancel_check"):
                wf.cancel_check = self._cancel.is_set
            self._workflows[self.active] = wf
        return self._workflows[self.active]

    def _handle(self, update: dict) -> None:
        message = update.get("message") or {}
        chat_id = int((message.get("chat") or {}).get("id", 0))
        text = str(message.get("text") or "").strip()
        if chat_id not in self.allowed:
            return
        if chat_id == 0:
            return
        command = text.split(maxsplit=1)[0].lower() if text else ""
        logger.info("command %r from chat %s", command or "(empty)", chat_id)
        if command in ("/start", "/help", ""):
            self.send(chat_id, HELP)
        elif command == "/auto":
            self._dispatch(self._suggest, chat_id)
        elif command in ("/yes", "/y", "/approve"):
            self._dispatch(self._approve, chat_id)
        elif command in ("/no", "/n", "/skip"):
            self._dispatch(self._skip, chat_id)
        elif command == "/repo":
            self._switch_repo(chat_id, text.removeprefix("/repo").strip())
        elif command == "/stop":
            self.pending = None
            self._pending_plan = None
            self.send(chat_id, "Auto loop stopped. Send /auto to start again.")
        elif command == "/cancel":
            self._cancel_run(chat_id)
        elif command == "/feature":
            feature = text.removeprefix("/feature").strip()
            if not feature:
                self.send(chat_id, "Expected: /feature <description>")
            elif self.config.workflow.plan_approval:
                # Gate: plan first and wait for /yes instead of building now.
                self.pending = None
                self._pending_plan = None
                self._dispatch(self._plan_for_approval, chat_id, feature)
            else:
                self.pending = None
                self._pending_plan = None
                self._dispatch(self._build, chat_id, feature)
        elif command == "/edit":
            self._edit(chat_id, text.removeprefix("/edit").strip())
        elif self.pending and text and not text.startswith("/"):
            # A plain-text reply while a suggestion is on the table is a revision
            # of it, not an unknown command — treat it as /edit.
            self._edit(chat_id, text)
        else:
            self.send(chat_id, HELP)

    def _recover(self) -> None:
        """Best-effort return to a clean base after a failed run.

        Without this a stopped run leaves a dirty worktree that blocks every
        later suggestion/build, wedging the loop until someone cleans up by hand.
        """
        try:
            self._workflow_instance().reset_worktree()
            logger.info("worktree reset to a clean base after a failed run")
        except Exception:
            logger.exception("worktree reset failed")

    def _switch_repo(self, chat_id: int, name: str) -> None:
        """List repos, or switch which one commands target (clears any pending)."""
        if not name:
            listing = "\n".join(
                f"{'▶' if n == self.active else ' '} {n}" for n in self.repos
            )
            self.send(chat_id, f"Repos (▶ active):\n{listing}\n\nSwitch with /repo <name>.")
            return
        match = name if name in self.repos else next(
            (n for n in self.repos if n.lower() == name.lower()), None
        )
        if match is None:
            self.send(chat_id, f"Unknown repo '{name}'. Known: {', '.join(self.repos)}.")
            return
        self.active = match
        self.pending = None
        self._pending_plan = None
        self.send(chat_id, f"Now targeting '{match}'. Send /auto for a suggestion.")

    def _cancel_run(self, chat_id: int) -> None:
        if self._busy:
            self._cancel.set()
            self.send(chat_id, "Cancelling the current run at the next step…")
        else:
            self.send(chat_id, "Nothing is running right now.")

    def _plan_for_approval(self, chat_id: int, feature: str) -> None:
        """Plan a /feature request and wait for /yes, when plan_approval is on."""
        self._auto_loop = False
        self.send(chat_id, f"[{self.active}] Planning '{feature[:60]}'…")
        try:
            plan = self._workflow_instance().preview_plan(feature)
        except Exception as exc:
            logger.warning("preview_plan failed: %s", exc)
            self._recover()
            self.send(chat_id, f"Could not plan that feature: {exc}")
            return
        self.pending = feature
        self._pending_plan = plan
        self.send(
            chat_id,
            f"📋 [{self.active}] Plan for:\n\n{feature}\n\n{plan}\n\n"
            "/yes to build it · /edit to revise · /no to discard.",
        )

    def _suggest(self, chat_id: int) -> None:
        self._auto_loop = True
        self._pending_plan = None
        self.send(chat_id, f"[{self.active}] Looking for the next feature to suggest…")
        wf = self._workflow_instance()
        feature = None
        error: Exception | None = None
        # A prior failure may have left the tree dirty; recover once and retry.
        for attempt in (0, 1):
            try:
                feature = wf.recommend_feature()
                error = None
                break
            except Exception as exc:
                error = exc
                logger.warning("recommend_feature failed (attempt %d): %s", attempt, exc)
                if attempt == 0:
                    self._recover()
        if error is not None:
            self.pending = None
            self.send(chat_id, f"Could not suggest a feature: {error}")
            return
        if not feature:
            self.pending = None
            self.send(
                chat_id,
                "No new feature to suggest right now. Try /auto again later, "
                "or /feature <description>.",
            )
            return
        self.pending = feature
        logger.info("suggested feature: %s", feature)
        self.send(
            chat_id,
            f"💡 [{self.active}] Suggested feature:\n\n{feature}\n\n"
            "/yes to build it · /no for a different one · /stop to end.",
        )

    def _approve(self, chat_id: int) -> None:
        if not self.pending:
            self.send(chat_id, "Nothing to confirm. Send /auto to get a suggestion.")
            return
        feature = self.pending
        plan = self._pending_plan
        auto = self._auto_loop
        self.pending = None
        self._pending_plan = None
        self._build(chat_id, feature, plan=plan)
        # In the /auto loop, continue by proposing the next feature; a gated
        # /feature approval builds just the one and stops.
        if auto:
            self._suggest(chat_id)

    def _skip(self, chat_id: int) -> None:
        if not self.pending:
            self.send(chat_id, "Nothing to skip. Send /auto to get a suggestion.")
            return
        self.pending = None
        self._pending_plan = None
        if self._auto_loop:
            self._suggest(chat_id)
        else:
            self.send(chat_id, "Discarded. Send /feature <description> or /auto.")

    def _edit(self, chat_id: int, feature: str) -> None:
        """Replace the pending suggestion with the user's revision.

        Editing only revises what is on the table and echoes it back for
        confirmation — it does not build. ``/yes`` remains the single build
        trigger, so an edited feature is still gated by an explicit approval.
        """
        if not self.pending:
            self.send(
                chat_id,
                "Nothing to edit yet. Send /auto for a suggestion, or "
                "/feature <description> to build one directly.",
            )
            return
        if not feature:
            self.send(chat_id, "Expected: /edit <revised feature>")
            return
        self.pending = feature
        self._pending_plan = None  # the revised text invalidates any planned preview
        logger.info("edited suggested feature: %s", feature)
        self.send(
            chat_id,
            f"✏️ [{self.active}] Updated suggestion:\n\n{feature}\n\n"
            "/yes to build it · /no for a different one · /stop to end.",
        )

    def _build(self, chat_id: int, feature: str, plan: str | None = None) -> None:
        self._progress_chat = chat_id
        self.send(chat_id, f"Accepted [{self.active}]. Planning, implementation, verification, and CI babysitting have started — this can take a while. Send /cancel to stop.")
        logger.info("building feature on %s: %s", self.active, feature)
        try:
            # watch=True: after opening the PR, babysit CI and address failures
            # before the loop moves on to the next suggestion. A pre-approved plan
            # (from the /feature gate) is reused so it isn't planned twice.
            kwargs: dict = {"watch": True}
            if plan is not None:
                kwargs["plan"] = plan
            state = self._workflow_instance().run(feature, **kwargs)
            logger.info("build finished: phase=%s pr=%s", state.phase, state.pr_url)
            self.send(chat_id, f"✅ Finished: {state.pr_url}\nState: {state.phase}")
        except Exception as exc:
            logger.warning("run stopped: %s", exc)
            # Clear the failed run's partial edits so the loop can continue.
            self._recover()
            self.send(chat_id, f"Run stopped: {exc}")
        finally:
            self._progress_chat = None

    def send(self, chat_id: int, text: str) -> None:
        self._call("sendMessage", {"chat_id": chat_id, "text": text[:4096]})

    def _call(self, method: str, values: dict) -> dict:
        if self._transport is not None:
            return self._transport(method, values)
        data = urllib.parse.urlencode(values).encode()
        request = urllib.request.Request(
            f"https://api.telegram.org/bot{self.token}/{method}", data=data
        )
        try:
            with urllib.request.urlopen(request, timeout=40) as response:
                payload = json.load(response)
        except OSError as exc:
            code = getattr(exc, "code", None)
            suffix = f" (HTTP {code})" if code else ""
            raise AgentShipError(f"Telegram request failed{suffix}") from exc
        if not payload.get("ok"):
            raise AgentShipError(f"Telegram API error: {payload}")
        return payload
