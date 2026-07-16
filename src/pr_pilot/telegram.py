from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from collections.abc import Callable

from .config import Config
from .errors import AgentShipError
from .workflow import Workflow


class TelegramBot:
    def __init__(self, config: Config, workflow_factory: Callable[[], Workflow] | None = None):
        self.config = config
        self.token = os.environ.get(config.telegram.token_env, "")
        if not self.token:
            raise AgentShipError(f"Missing Telegram token environment variable: {config.telegram.token_env}")
        if not config.telegram.allowed_chat_ids:
            raise AgentShipError("telegram.allowed_chat_ids must contain at least one trusted chat ID")
        self.allowed = set(config.telegram.allowed_chat_ids)
        self.workflow_factory = workflow_factory or (lambda: Workflow(config))
        self.offset = 0

    def serve_forever(self) -> None:
        while True:
            updates = self._call("getUpdates", {"timeout": 30, "offset": self.offset})["result"]
            for update in updates:
                self.offset = max(self.offset, int(update["update_id"]) + 1)
                self._handle(update)

    def _handle(self, update: dict) -> None:
        message = update.get("message") or {}
        chat_id = int((message.get("chat") or {}).get("id", 0))
        text = str(message.get("text") or "").strip()
        if chat_id not in self.allowed:
            return
        if text == "/start":
            self.send(chat_id, "Send /feature followed by the feature request.")
            return
        if not text.startswith("/feature "):
            self.send(chat_id, "Expected: /feature <description>")
            return
        feature = text.removeprefix("/feature ").strip()
        self.send(chat_id, "Accepted. Planning and implementation have started.")
        try:
            state = self.workflow_factory().run(feature)
            self.send(chat_id, f"Finished: {state.pr_url}\nState: {state.phase}")
        except Exception as exc:
            self.send(chat_id, f"Run stopped: {exc}")

    def send(self, chat_id: int, text: str) -> None:
        self._call("sendMessage", {"chat_id": chat_id, "text": text[:4096]})

    def _call(self, method: str, values: dict) -> dict:
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
