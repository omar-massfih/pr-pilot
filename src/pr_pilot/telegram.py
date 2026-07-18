from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from collections.abc import Callable

from .config import Config
from .errors import AgentShipError
from .workflow import Workflow

HELP = (
    "PR Pilot commands:\n"
    "/auto — suggest the next feature and wait for your go-ahead\n"
    "/yes — build the suggested feature, then suggest the next one\n"
    "/no — skip it and suggest a different feature\n"
    "/stop — end the auto loop\n"
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
        self.workflow_factory = workflow_factory or (lambda: Workflow(config))
        self._transport = transport
        self._workflow: Workflow | None = None
        self.pending: str | None = None
        self.offset = 0

    def serve_forever(self) -> None:
        while True:
            updates = self._call("getUpdates", {"timeout": 30, "offset": self.offset})["result"]
            for update in updates:
                self.offset = max(self.offset, int(update["update_id"]) + 1)
                self._handle(update)

    def _workflow_instance(self) -> Workflow:
        # One workflow per session so recommend/run share the same providers
        # and memory; created lazily so construction errors surface on use.
        if self._workflow is None:
            self._workflow = self.workflow_factory()
        return self._workflow

    def _handle(self, update: dict) -> None:
        message = update.get("message") or {}
        chat_id = int((message.get("chat") or {}).get("id", 0))
        text = str(message.get("text") or "").strip()
        if chat_id not in self.allowed:
            return
        command = text.split(maxsplit=1)[0].lower() if text else ""
        if command in ("/start", "/help", ""):
            self.send(chat_id, HELP)
        elif command == "/auto":
            self._suggest(chat_id)
        elif command in ("/yes", "/y", "/approve"):
            self._approve(chat_id)
        elif command in ("/no", "/n", "/skip"):
            self._skip(chat_id)
        elif command == "/stop":
            self.pending = None
            self.send(chat_id, "Auto loop stopped. Send /auto to start again.")
        elif command == "/feature":
            feature = text.removeprefix("/feature").strip()
            if not feature:
                self.send(chat_id, "Expected: /feature <description>")
            else:
                self.pending = None
                self._build(chat_id, feature)
        else:
            self.send(chat_id, HELP)

    def _suggest(self, chat_id: int) -> None:
        self.send(chat_id, "Looking for the next feature to suggest…")
        try:
            feature = self._workflow_instance().recommend_feature()
        except Exception as exc:
            self.pending = None
            self.send(chat_id, f"Could not suggest a feature: {exc}")
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
        self.send(
            chat_id,
            f"💡 Suggested feature:\n\n{feature}\n\n"
            "/yes to build it · /no for a different one · /stop to end.",
        )

    def _approve(self, chat_id: int) -> None:
        if not self.pending:
            self.send(chat_id, "Nothing to confirm. Send /auto to get a suggestion.")
            return
        feature = self.pending
        self.pending = None
        self._build(chat_id, feature)
        # Continue the loop: propose the next feature for confirmation.
        self._suggest(chat_id)

    def _skip(self, chat_id: int) -> None:
        if not self.pending:
            self.send(chat_id, "Nothing to skip. Send /auto to get a suggestion.")
            return
        self.pending = None
        self._suggest(chat_id)

    def _build(self, chat_id: int, feature: str) -> None:
        self.send(chat_id, "Accepted. Planning, implementation, and CI babysitting have started — this can take a while.")
        try:
            # watch=True: after opening the PR, babysit CI and address failures
            # before the loop moves on to the next suggestion.
            state = self._workflow_instance().run(feature, watch=True)
            self.send(chat_id, f"✅ Finished: {state.pr_url}\nState: {state.phase}")
        except Exception as exc:
            self.send(chat_id, f"Run stopped: {exc}")

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
