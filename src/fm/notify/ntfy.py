"""ntfy channel (DESIGN section 11): pushes to a secret topic, Approve/Reject as ``http`` action buttons, no account.

``fm notify setup`` generates two long random topic names and keeps them in ``.env``: ``NTFY_TOPIC`` is what the phone
subscribes to, ``NTFY_REPLY_TOPIC`` is what the PC listens on. A proposal is published with two ``http`` actions that
POST ``<decision>:<proposal id>:<nonce>`` (:func:`fm.notify.base.encode_callback`) to the reply topic, so a press needs
only the phone's outbound connection and the PC needs no port: :meth:`NtfyChannel.poll` subscribes to the reply topic
(a JSON stream held open until a press or the server's keepalive; ``poll=1`` when not waiting) and the bot checks the
nonce before deciding anything. The id of the last reply seen is kept in ``<config dir>/notify/ntfy-since``, so a
``fm bot`` restarted within :data:`RESUME_WINDOW_S` resumes after it instead of replaying the server's cache (the
replayed nonces would be refused anyway, but each would be logged as a bad press).

Topic names are the secret, so they are registered with :func:`fm.notify.base.hide_in_logs` and never appear in
errors; a post to the reply topic without a live nonce is ignored. After every decision a confirmation push goes to the
main topic: the iOS app does not clear a notification after a button tap even with ``clear``, so the confirmation is
how the phone learns the press landed.
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Self

import httpx

from fm import paths
from fm.config import Config
from fm.notify.base import (
    BUTTONS,
    Decision,
    DecisionResult,
    Message,
    NotifyError,
    Priority,
    ProposalNotice,
    Reply,
    decode_callback,
    hide_in_logs,
    redact,
    split_text,
)
from fm.notify.messages import decision_message

logger = logging.getLogger(__name__)

DEFAULT_SERVER = "https://ntfy.sh"
TOPIC_PATTERN = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
TOPIC_PREFIX = "espn-fantasy-"
TOPIC_BYTES = 18
"""Random bytes in a generated topic name: 144 bits, 24 characters after the prefix."""
SINCE_ALL = "all"
"""Where a subscription starts without a saved resume point: every reply the server still caches."""
SINCE_FILE = Path("notify") / "ntfy-since"
"""The resume point under the config dir: the id of the last reply-topic message seen."""
RESUME_WINDOW_S = 6 * 3600.0
"""A saved resume point older than this is not trusted (ntfy.sh caches messages for 12 hours, and a message that left
the cache cannot be resumed after), so the subscription starts from the whole cache instead: replays only cost log
lines, while a resume point the server no longer knows could cost a press."""
MESSAGE_ID_PATTERN = re.compile(r"^[A-Za-z0-9]{1,32}$")
PRIORITIES: Mapping[Priority, int] = {"low": 2, "default": 3, "high": 4}
"""ntfy priorities run from 1 (min) to 5 (max)."""
TOPIC_ENV = "NTFY_TOPIC"
REPLY_TOPIC_ENV = "NTFY_REPLY_TOPIC"
REQUEST_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
STREAM_READ_TIMEOUT_S = 60.0
"""Longest silence on the subscription stream before it is closed; ntfy.sh sends a keepalive every 45 s."""
STREAM_WINDOW_S = 55.0
"""Longest one waiting :meth:`NtfyChannel.poll` listens, so the listener regains control even under junk traffic."""


def generate_topic() -> str:
    """A fresh secret topic name: ``espn-fantasy-`` plus 24 random URL-safe characters."""
    return TOPIC_PREFIX + secrets.token_urlsafe(TOPIC_BYTES)


class NtfyChannel:
    """Publishes to ``topic`` and listens on ``reply_topic`` at ``server``, resuming after the message id saved in
    ``since_file`` when one is given (:meth:`from_config` uses :data:`SINCE_FILE`).

    ``client`` may be an injected ``httpx.Client`` (tests); it is left open by :meth:`close`.
    """

    name = "ntfy"

    def __init__(
        self,
        topic: str,
        reply_topic: str,
        *,
        server: str = DEFAULT_SERVER,
        client: httpx.Client | None = None,
        since_file: Path | None = None,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        for env, value in ((TOPIC_ENV, topic), (REPLY_TOPIC_ENV, reply_topic)):
            if TOPIC_PATTERN.match(value) is None:
                raise NotifyError(f"{env} is not a valid ntfy topic name (1-64 letters, digits, - or _)")
        if topic == reply_topic:
            raise NotifyError(f"{TOPIC_ENV} and {REPLY_TOPIC_ENV} must differ: the phone must not see the replies")
        hide_in_logs(topic, reply_topic)
        self._topic = topic
        self._reply_topic = reply_topic
        self._server = server.rstrip("/")
        self._client = client
        self._owns_client = client is None
        self._monotonic = monotonic
        self._since_file = since_file
        self._since = _saved_since(since_file)

    @classmethod
    def from_config(cls, config: Config, *, client: httpx.Client | None = None, server: str = DEFAULT_SERVER) -> Self:
        """The channel ``.env`` sets up; ``NotifyError`` names what is missing."""
        topic = config.secrets.ntfy_topic
        reply_topic = config.secrets.ntfy_reply_topic
        missing = [name for name, value in ((TOPIC_ENV, topic), (REPLY_TOPIC_ENV, reply_topic)) if value is None]
        if topic is None or reply_topic is None:
            raise NotifyError(f"ntfy is not set up: {', '.join(missing)} missing from .env (run fm notify setup)")
        since_file = paths.config_dir() / SINCE_FILE
        return cls(
            topic.get_secret_value(),
            reply_topic.get_secret_value(),
            server=server,
            client=client,
            since_file=since_file,
        )

    # --- channel interface ------------------------------------------------------------------------------------------

    def send(self, message: Message) -> None:
        body = message.body.strip()
        if not body:
            self.publish(message.title, priority=message.priority, tags=message.tags, click=message.link)
            return
        pieces = split_text(body)
        titles = _titles(message.title, len(pieces))
        for index, (piece, title) in enumerate(zip(pieces, titles, strict=True), start=1):
            link = message.link if index == len(pieces) else None
            self.publish(piece, title=title, priority=message.priority, tags=message.tags, click=link)

    def send_proposal(self, notice: ProposalNotice) -> None:
        message = notice.message
        actions = [self._action(label, notice, decision) for label, decision in BUTTONS]
        pieces = split_text(message.body.strip() or message.title)
        titles = _titles(message.title, len(pieces))
        for piece, title in zip(pieces[:-1], titles[:-1], strict=True):
            self.publish(piece, title=title, priority=message.priority)
        self.publish(pieces[-1], title=titles[-1], priority=message.priority, tags=message.tags, actions=actions)

    def poll(self, *, wait: bool = True) -> list[Reply]:
        """Presses posted to the reply topic since the last poll.

        Waiting, this subscribes to the topic's JSON stream and returns at the first press, at the server's keepalive,
        or after :data:`STREAM_WINDOW_S`, one request per call; without waiting it asks once with ``poll=1``.
        """
        url = f"{self._server}/{self._reply_topic}/json"
        if not wait:
            try:
                response = self.client.get(url, params={"poll": "1", "since": self._since}, timeout=REQUEST_TIMEOUT)
            except httpx.HTTPError as exc:
                raise NotifyError(f"ntfy poll: {type(exc).__name__}: {redact(str(exc))}") from None
            if response.status_code != 200:
                raise NotifyError(f"ntfy poll: HTTP {response.status_code}")
            return [reply for line in response.text.splitlines() if (reply := self._read(line)) is not None]
        return self._listen(url)

    def confirm(self, reply: Reply, result: DecisionResult) -> None:
        """The confirmation push, on the topic the phone shows."""
        self.send(decision_message(result))

    def dismiss(self, reply: Reply, reason: str) -> None:
        """Nothing: anyone who learned the reply topic can post to it, so an ignored press gets no push (no spam)."""

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # --- HTTP -------------------------------------------------------------------------------------------------------

    def publish(
        self,
        message: str,
        *,
        title: str | None = None,
        priority: Priority = "default",
        tags: tuple[str, ...] = (),
        click: str | None = None,
        actions: list[dict[str, Any]] | None = None,
    ) -> None:
        """Publish one message to the main topic, as JSON (``POST /`` with the topic in the body)."""
        payload: dict[str, Any] = {"topic": self._topic, "message": message, "priority": PRIORITIES[priority]}
        if title:
            payload["title"] = title
        if tags:
            payload["tags"] = list(tags)
        if click:
            payload["click"] = click
        if actions:
            payload["actions"] = actions
        try:
            response = self.client.post(f"{self._server}/", json=payload, timeout=REQUEST_TIMEOUT)
        except httpx.HTTPError as exc:
            raise NotifyError(f"ntfy publish: {type(exc).__name__}: {redact(str(exc))}") from None
        if response.status_code != 200:
            raise NotifyError(f"ntfy publish: HTTP {response.status_code}")

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=REQUEST_TIMEOUT, follow_redirects=False)
        return self._client

    def _action(self, label: str, notice: ProposalNotice, decision: Decision) -> dict[str, Any]:
        return {
            "action": "http",
            "label": label,
            "url": f"{self._server}/{self._reply_topic}",
            "method": "POST",
            "body": notice.callback(decision),
            "clear": True,
        }

    def _listen(self, url: str) -> list[Reply]:
        started = self._monotonic()
        timeout = httpx.Timeout(30.0, connect=10.0, read=STREAM_READ_TIMEOUT_S)
        try:
            with self.client.stream("GET", url, params={"since": self._since}, timeout=timeout) as response:
                if response.status_code != 200:
                    raise NotifyError(f"ntfy subscribe: HTTP {response.status_code}")
                for line in response.iter_lines():
                    if _event(line).get("event") == "keepalive":
                        break
                    reply = self._read(line)
                    if reply is not None:
                        return [reply]  # the next poll resumes after it
                    if self._monotonic() - started >= STREAM_WINDOW_S:
                        break
        except httpx.ReadTimeout:
            pass  # a quiet topic: nothing to report
        except httpx.HTTPError as exc:
            raise NotifyError(f"ntfy subscribe: {type(exc).__name__}: {redact(str(exc))}") from None
        return []

    def _read(self, line: str) -> Reply | None:
        """The press in one stream line, advancing the resume point past every message seen."""
        event = _event(line)
        if event.get("event") != "message":
            return None
        message_id = event.get("id")
        if isinstance(message_id, str) and MESSAGE_ID_PATTERN.match(message_id):
            self._advance(message_id)
        callback = decode_callback(str(event.get("message") or ""))
        if callback is None:
            logger.warning("ignored a post on the ntfy reply topic that is not a decision")
            return None
        return Reply(
            channel=self.name,
            proposal_id=callback.proposal_id,
            decision=callback.decision,
            nonce=callback.nonce,
            sender=str(message_id or "?"),
            ref={"message_id": str(message_id or "")},
        )

    def _advance(self, message_id: str) -> None:
        """Move the resume point past ``message_id``, and save it so a restart resumes there too."""
        self._since = message_id
        if self._since_file is None:
            return
        try:
            self._since_file.parent.mkdir(parents=True, exist_ok=True)
            self._since_file.write_text(message_id, encoding="utf-8")
        except OSError as exc:  # only costs a replay after a restart, which the nonces make harmless
            logger.warning("could not save the ntfy resume point: %s", exc)


def _saved_since(since_file: Path | None) -> str:
    """The saved resume point when it is recent and well formed, else :data:`SINCE_ALL`."""
    if since_file is None:
        return SINCE_ALL
    try:
        fresh = time.time() - since_file.stat().st_mtime < RESUME_WINDOW_S
        saved = since_file.read_text(encoding="utf-8").strip()
    except OSError:
        return SINCE_ALL
    return saved if fresh and MESSAGE_ID_PATTERN.match(saved) else SINCE_ALL


def _event(line: str) -> dict[str, Any]:
    line = line.strip()
    if not line:
        return {}
    try:
        data = json.loads(line)
    except ValueError:
        return {}
    return data if isinstance(data, dict) else {}


def _titles(title: str, count: int) -> list[str]:
    """One title per piece of a split message, numbered ``(1/3)`` when there is more than one."""
    return [title] if count == 1 else [f"{title} ({index}/{count})" for index in range(1, count + 1)]
