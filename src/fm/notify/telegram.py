"""Telegram channel (DESIGN section 11): proposals with inline Approve/Reject buttons over the Bot API, long polling.

Setup is a bot from @BotFather whose token sits in ``.env`` as ``TELEGRAM_BOT_TOKEN``. ``fm notify setup`` captures
``TELEGRAM_CHAT_ID`` with :meth:`TelegramChannel.wait_for_chat`: it prints a one-time code (and a ``t.me`` link that
sends it) and saves the private chat that sends that code, so a stranger who happens to message the public bot first
is not captured. From then on every press is checked against that one chat: a callback whose user or chat differs is
dropped unanswered, before its data is even decoded.

Proposals go out as ``sendMessage`` with an inline keyboard whose ``callback_data`` is
``<decision>:<proposal id>:<nonce>`` (:func:`fm.notify.base.encode_callback`). Presses arrive through ``getUpdates``
held open for up to :data:`DEFAULT_POLL_TIMEOUT_S` (outbound only: no webhook, no port), acknowledged by the offset of
the next call. A decision is confirmed by answering the press (a toast) and editing the message to show the outcome,
which also removes its buttons; a press handled too late to answer (the PC was asleep) gets a reply message instead,
so the phone still hears about it.

The token is part of every request URL, so it is registered with :func:`fm.notify.base.hide_in_logs` and errors name
the method and status only.
"""

from __future__ import annotations

import logging
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self

import httpx

from fm.config import Config
from fm.notify.base import (
    BUTTONS,
    DecisionResult,
    Message,
    NotifyError,
    ProposalNotice,
    Reply,
    decode_callback,
    hide_in_logs,
    redact,
    split_text,
)

logger = logging.getLogger(__name__)

BOT_API_ROOT = "https://api.telegram.org"
DEFAULT_POLL_TIMEOUT_S = 25
"""Seconds ``getUpdates`` waits server-side for a press before answering empty (Telegram allows up to 50)."""
UPDATE_BATCH = 100
MESSAGE_LIMIT = 4096
"""Characters Telegram accepts in one message."""
TOAST_LIMIT = 200
"""Characters ``answerCallbackQuery`` shows."""
TOKEN_ENV = "TELEGRAM_BOT_TOKEN"
CHAT_ID_ENV = "TELEGRAM_CHAT_ID"
TOKEN_PATTERN = re.compile(r"^[0-9]+:[A-Za-z0-9_-]+$")
"""``<bot id>:<secret>``, the shape @BotFather hands out."""


@dataclass(frozen=True, slots=True)
class ChatCapture:
    """The private chat that sent the setup code."""

    chat_id: int
    name: str | None


class TelegramChannel:
    """Bot API client bound to one private chat. ``chat_id=None`` is only for setup, before the chat is known.

    ``client`` may be an injected ``httpx.Client`` (tests); it is left open by :meth:`close`.
    """

    name = "telegram"

    def __init__(
        self,
        token: str,
        chat_id: int | None,
        *,
        client: httpx.Client | None = None,
        api_root: str = BOT_API_ROOT,
        poll_timeout_s: int = DEFAULT_POLL_TIMEOUT_S,
    ) -> None:
        token = token.strip()
        if TOKEN_PATTERN.match(token) is None:
            raise NotifyError(f"{TOKEN_ENV} does not look like a bot token from @BotFather (<digits>:<letters>)")
        hide_in_logs(token)
        self._token = token
        self.chat_id = chat_id
        self._api_root = api_root.rstrip("/")
        self.poll_timeout_s = max(0, poll_timeout_s)
        self._client = client
        self._owns_client = client is None
        self._offset: int | None = None

    @classmethod
    def from_config(cls, config: Config, *, client: httpx.Client | None = None) -> Self:
        """The channel ``.env`` sets up; ``NotifyError`` names what is missing."""
        token = config.secrets.telegram_bot_token
        chat_id = config.secrets.telegram_chat_id
        missing = [name for name, value in ((TOKEN_ENV, token), (CHAT_ID_ENV, chat_id)) if value is None]
        if token is None or chat_id is None:
            raise NotifyError(f"telegram is not set up: {', '.join(missing)} missing from .env (run fm notify setup)")
        return cls(token.get_secret_value(), chat_id, client=client)

    # --- channel interface ------------------------------------------------------------------------------------------

    def send(self, message: Message) -> None:
        """Push ``message`` as plain text; a link goes at the end, where Telegram makes it tappable. (A URL button
        would let one malformed link get the whole alert refused.)"""
        text = "\n".join([message.text, message.link]) if message.link else message.text
        for piece in split_text(text):
            self.call("sendMessage", chat_id=self._chat(), text=piece, disable_notification=message.priority == "low")

    def send_proposal(self, notice: ProposalNotice) -> None:
        row = [{"text": label, "callback_data": notice.callback(decision)} for label, decision in BUTTONS]
        keyboard = {"inline_keyboard": [row]}
        pieces = split_text(notice.message.text)
        for piece in pieces[:-1]:
            self.call("sendMessage", chat_id=self._chat(), text=piece)
        self.call("sendMessage", chat_id=self._chat(), text=pieces[-1], reply_markup=keyboard)

    def poll(self, *, wait: bool = True) -> list[Reply]:
        """One ``getUpdates`` (held open up to ``poll_timeout_s`` when ``wait``); the presses from the accepted chat,
        oldest first."""
        chat_id = self._chat()
        params: dict[str, Any] = {
            "timeout": self.poll_timeout_s if wait else 0,
            "limit": UPDATE_BATCH,
            "allowed_updates": ["callback_query"],
        }
        if self._offset is not None:
            params["offset"] = self._offset
        replies: list[Reply] = []
        for update in self._updates(params):
            reply = self._reply(update, chat_id)
            if reply is not None:
                replies.append(reply)
        return replies

    def confirm(self, reply: Reply, result: DecisionResult) -> None:
        """Answer the press with a toast and edit the proposal message to show the outcome (dropping its buttons). When
        the press is too old to answer, send a reply message instead, so the phone is notified."""
        answered = True
        try:
            self._answer(reply, result.describe())
        except NotifyError as exc:
            answered = False
            logger.info("could not answer the press on proposal #%d: %s", reply.proposal_id, exc)
        message_id = reply.ref.get("message_id")
        if isinstance(message_id, int):
            original = str(reply.ref.get("text") or "")
            edited = f"{original}\n\n{_outcome(result)}".strip()[:MESSAGE_LIMIT]
            try:
                # Editing the text without a reply_markup removes the inline keyboard.
                self.call("editMessageText", chat_id=self._chat(), message_id=message_id, text=edited)
            except NotifyError as exc:
                logger.warning("could not update the message of proposal #%d: %s", reply.proposal_id, exc)
        if not answered:
            params: dict[str, Any] = {"chat_id": self._chat(), "text": result.describe()[:MESSAGE_LIMIT]}
            if isinstance(message_id, int):
                params["reply_parameters"] = {"message_id": message_id, "allow_sending_without_reply": True}
            self.call("sendMessage", **params)

    def dismiss(self, reply: Reply, reason: str) -> None:
        """Answer an ignored press from our own chat with a toast, so the button stops spinning."""
        self._answer(reply, reason)

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- setup ------------------------------------------------------------------------------------------------------

    def username(self) -> str:
        """The bot's username from ``getMe``, which also proves the token works."""
        result = self.call("getMe")
        name = result.get("username") if isinstance(result, dict) else None
        if not isinstance(name, str) or not name:
            raise NotifyError("telegram getMe: the response has no bot username")
        return name

    def wait_for_chat(
        self,
        code: str,
        *,
        timeout_s: float,
        sleep: Callable[[float], object] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> ChatCapture | None:
        """Wait up to ``timeout_s`` for a private-chat message containing ``code`` and return that chat. Every message
        seen is acknowledged, so the bot never sees them again. ``None`` when the code did not arrive in time."""
        deadline = monotonic() + timeout_s
        while (remaining := deadline - monotonic()) > 0:
            params: dict[str, Any] = {
                "timeout": max(0, min(self.poll_timeout_s, int(remaining))),
                "limit": UPDATE_BATCH,
                "allowed_updates": ["message"],
            }
            if self._offset is not None:
                params["offset"] = self._offset
            updates = self._updates(params)
            for update in updates:
                capture = _capture(update, code)
                if capture is not None:
                    self._acknowledge()
                    return capture
            if not updates:
                sleep(1.0)  # never spin when the server answers at once
        self._acknowledge()
        return None

    # --- Bot API ----------------------------------------------------------------------------------------------------

    def call(self, method: str, **params: Any) -> Any:
        """POST one Bot API method and return its ``result``. Raises :class:`NotifyError`, never echoing the token."""
        try:
            response = self.client.post(f"{self._api_root}/bot{self._token}/{method}", json=params)
        except httpx.HTTPError as exc:
            raise NotifyError(f"telegram {method}: {type(exc).__name__}: {redact(str(exc))}") from None
        try:
            body = response.json()
        except ValueError:
            body = None
        if not isinstance(body, dict):
            raise NotifyError(f"telegram {method}: HTTP {response.status_code}: not a Bot API response")
        if response.status_code != 200 or body.get("ok") is not True:
            description = redact(str(body.get("description") or "no description"))
            raise NotifyError(f"telegram {method}: HTTP {response.status_code}: {description}")
        return body.get("result")

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            # The read timeout must outlast the long poll, which Telegram holds open for poll_timeout_s.
            timeout = httpx.Timeout(30.0, connect=10.0, read=float(self.poll_timeout_s + 15))
            self._client = httpx.Client(timeout=timeout, follow_redirects=False)
        return self._client

    def _updates(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        result = self.call("getUpdates", **params)
        updates = [update for update in result if isinstance(update, dict)] if isinstance(result, list) else []
        ids = [update["update_id"] for update in updates if isinstance(update.get("update_id"), int)]
        if ids:
            self._offset = max(ids) + 1  # acknowledged by the next getUpdates
        return updates

    def _acknowledge(self) -> None:
        """Confirm every update seen so far, so neither setup nor ``fm bot`` sees them again."""
        if self._offset is not None:
            self.call("getUpdates", offset=self._offset, limit=1, timeout=0)

    def _answer(self, reply: Reply, text: str) -> None:
        query_id = reply.ref.get("callback_query_id")
        if not isinstance(query_id, str):
            raise NotifyError("telegram answerCallbackQuery: the reply has no callback query id")
        self.call("answerCallbackQuery", callback_query_id=query_id, text=text[:TOAST_LIMIT])

    def _reply(self, update: dict[str, Any], chat_id: int) -> Reply | None:
        query = update.get("callback_query")
        if not isinstance(query, dict):
            return None
        sender = _object(query.get("from"))
        message = _object(query.get("message"))
        chat = _object(message.get("chat"))
        if sender.get("id") != chat_id or chat.get("id") != chat_id:
            logger.warning("ignored a Telegram button press from another chat (user %s)", sender.get("id"))
            return None
        query_id = query.get("id")
        callback = decode_callback(str(query.get("data") or ""))
        if callback is None or not isinstance(query_id, str):
            logger.warning("ignored a Telegram button press that is not a decision")
            return None
        ref: dict[str, str | int] = {"callback_query_id": query_id, "text": str(message.get("text") or "")}
        message_id = message.get("message_id")
        if isinstance(message_id, int):
            ref["message_id"] = message_id
        return Reply(
            channel=self.name,
            proposal_id=callback.proposal_id,
            decision=callback.decision,
            nonce=callback.nonce,
            sender=str(chat_id),
            ref=ref,
        )

    def _chat(self) -> int:
        if self.chat_id is None:
            raise NotifyError(f"{CHAT_ID_ENV} is not set; run fm notify setup")
        return self.chat_id


def _object(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _capture(update: dict[str, Any], code: str) -> ChatCapture | None:
    message = _object(update.get("message"))
    chat = _object(message.get("chat"))
    chat_id = chat.get("id")
    if chat.get("type") != "private" or not isinstance(chat_id, int):
        return None
    if code not in str(message.get("text") or ""):
        return None
    sender = _object(message.get("from"))
    name = " ".join(str(sender[key]) for key in ("first_name", "last_name") if sender.get(key)) or None
    return ChatCapture(chat_id=chat_id, name=name)


def _outcome(result: DecisionResult) -> str:
    if result.ok:
        return f"{result.done.capitalize()} via Telegram."
    return f"Not {result.done}: {result.detail}"
