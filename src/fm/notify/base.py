"""The phone channel interface (DESIGN section 11): what every adapter sends, and the presses that come back.

A channel carries three kinds of traffic:

- a :class:`Message`: alerts, reports and confirmations, as plain pushes;
- a :class:`ProposalNotice`: a proposal with Approve and Reject buttons;
- the :class:`Reply` a button press produces, which :mod:`fm.notify.bot` turns into ``fm.proposals.approve`` or
  ``reject``.

Neither adapter opens an inbound port: Telegram is long polling, and ntfy is a reply topic the PC subscribes to over an
outbound connection.

Buttons never carry a proposal's execution token: ``approve`` mints that token and it authorizes a write, so it stays
on this machine (the interface note in :mod:`fm.proposals.queue`). A button carries the single-use approval nonce that
:mod:`fm.notify.nonces` issued for its one notification. Both adapters encode a press as
``<decision>:<proposal id>:<nonce>`` (:func:`encode_callback`), which fits Telegram's 64-byte ``callback_data`` and is
the body an ntfy ``http`` action posts to the reply topic.

The bot token and the ntfy topic names are secrets that travel in request URLs, and httpx logs every request URL at
INFO, so adapters pass them to :func:`hide_in_logs` before their first request and scrub them from error messages.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol, cast

from fm.notify.nonces import NONCE_LENGTH

type Priority = Literal["low", "default", "high"]
type Decision = Literal["approve", "reject"]

DECISIONS: tuple[Decision, ...] = ("approve", "reject")
PAST_TENSE: Mapping[Decision, str] = {"approve": "approved", "reject": "rejected"}
APPROVE_LABEL = "Approve"
REJECT_LABEL = "Reject"
BUTTONS: tuple[tuple[str, Decision], ...] = ((APPROVE_LABEL, "approve"), (REJECT_LABEL, "reject"))
"""The buttons on a proposal push, left to right: label and the decision it sends."""
TEXT_LIMIT = 3800
"""Most UTF-8 bytes in one push: Telegram allows 4096 characters per message, ntfy 4096 bytes per message body."""
REDACTED = "***"

_CALLBACK = re.compile(
    rf"^(?P<decision>approve|reject):(?P<proposal_id>[1-9][0-9]{{0,11}}):(?P<nonce>[A-Za-z0-9_-]{{{NONCE_LENGTH}}})$"
)


class NotifyError(Exception):
    """A channel is not set up, or a request to it failed. The message never contains a bot token or a topic name."""


@dataclass(frozen=True, slots=True)
class Message:
    """A plain push. ``tags`` are ntfy emoji tags (Telegram ignores them); ``link`` is the page that fixes the problem
    (an ESPN deep link): tapping the ntfy notification opens it, and Telegram shows it as a link under the text."""

    title: str
    body: str = ""
    priority: Priority = "default"
    tags: tuple[str, ...] = ()
    link: str | None = None

    @property
    def text(self) -> str:
        """Title and body as one block, for a channel without a separate title."""
        body = self.body.strip()
        return f"{self.title}\n{body}" if body else self.title


@dataclass(frozen=True, slots=True)
class ProposalNotice:
    """A proposal waiting for a decision, and the nonce its Approve and Reject buttons carry."""

    proposal_id: int
    nonce: str
    message: Message

    def callback(self, decision: Decision) -> str:
        """What the ``decision`` button sends back."""
        return encode_callback(decision, self.proposal_id, self.nonce)


@dataclass(frozen=True, slots=True)
class Reply:
    """A button press as received, before its nonce is checked.

    ``sender`` says where it came from, for logs (the chat id, or the reply topic's message id); ``ref`` is what the
    channel needs to answer on the pressed message (Telegram's callback query and message ids).
    """

    channel: str
    proposal_id: int
    decision: Decision
    nonce: str
    sender: str
    ref: Mapping[str, str | int] = field(default_factory=dict[str, str | int])


@dataclass(frozen=True, slots=True)
class DecisionResult:
    """What a press did: the decision recorded, or why ``fm.proposals`` refused it (expired, already decided)."""

    proposal_id: int
    decision: Decision
    ok: bool
    detail: str

    @property
    def done(self) -> str:
        """The decision in the past tense: ``approved`` or ``rejected``."""
        return PAST_TENSE[self.decision]

    @property
    def headline(self) -> str:
        """``#12 approved``, ``#12 rejected``, or ``#12 not approved`` when it was refused."""
        return f"#{self.proposal_id} {'' if self.ok else 'not '}{self.done}"

    def describe(self) -> str:
        return f"{self.headline}: {self.detail}"


@dataclass(frozen=True, slots=True)
class Callback:
    """A decoded button press."""

    decision: Decision
    proposal_id: int
    nonce: str


class NotifyChannel(Protocol):
    """One phone channel: :class:`fm.notify.telegram.TelegramChannel` or :class:`fm.notify.ntfy.NtfyChannel`."""

    @property
    def name(self) -> str:
        """``telegram`` or ``ntfy``; recorded as the proposal's ``decided_by``."""
        ...

    def send(self, message: Message) -> None:
        """Push a plain message, split into several when longer than :data:`TEXT_LIMIT`."""
        ...

    def send_proposal(self, notice: ProposalNotice) -> None:
        """Push a proposal with Approve and Reject buttons carrying ``notice.callback(...)``."""
        ...

    def poll(self, *, wait: bool = True) -> list[Reply]:
        """Presses that arrived since the last poll, from the accepted chat or reply topic only. ``wait`` lets the
        server hold the request open until a press arrives (bounded); without it only what is waiting returns."""
        ...

    def confirm(self, reply: Reply, result: DecisionResult) -> None:
        """Tell the phone what a press did, where it was pressed."""
        ...

    def dismiss(self, reply: Reply, reason: str) -> None:
        """Answer a press that was ignored (a used or unknown nonce), where the channel can do so privately."""
        ...

    def close(self) -> None: ...


def encode_callback(decision: Decision, proposal_id: int, nonce: str) -> str:
    """``approve:12:<nonce>``: what a button carries, on both channels (at most 45 bytes)."""
    text = f"{decision}:{proposal_id}:{nonce}"
    if _CALLBACK.match(text) is None:
        raise ValueError(f"cannot encode a {decision!r} button for proposal {proposal_id!r} with that nonce")
    return text


def decode_callback(text: str) -> Callback | None:
    """The press :func:`encode_callback` encoded, or ``None`` for anything else (never an exception)."""
    match = _CALLBACK.match(text.strip())
    if match is None:
        return None
    return Callback(cast(Decision, match["decision"]), int(match["proposal_id"]), match["nonce"])


def split_text(text: str, limit: int = TEXT_LIMIT) -> list[str]:
    """Split ``text`` into pieces of at most ``limit`` UTF-8 bytes, cutting at line breaks where possible and never
    inside a character. Always at least one piece."""
    if limit < 4:
        raise ValueError("limit must be at least 4 bytes, the longest UTF-8 character")
    pieces: list[str] = []
    rest = text.rstrip()
    while len(rest.encode()) > limit:
        head = rest.encode()[:limit].decode(errors="ignore")
        cut = head.rfind("\n")
        if cut <= 0:
            cut = len(head)
        piece = rest[:cut].rstrip()
        if piece:
            pieces.append(piece)
        rest = rest[cut:].lstrip("\n")
    if rest or not pieces:
        pieces.append(rest)
    return pieces


class RedactSecretsFilter(logging.Filter):
    """Replaces registered secrets with ``***`` in log records, such as httpx's ``HTTP Request:`` line, which carries
    the full URL (``/bot<token>/getUpdates``, ``/<reply topic>/json``). Records are rewritten before any handler sees
    them."""

    def __init__(self) -> None:
        super().__init__()
        self._secrets: set[str] = set()

    def add(self, secret: str) -> None:
        if secret:
            self._secrets.add(secret)

    def redact(self, text: str) -> str:
        for secret in sorted(self._secrets, key=len, reverse=True):
            text = text.replace(secret, REDACTED)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        redacted = self.redact(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()  # already formatted; a ``%`` in a URL must not be interpolated again
        return True


_REDACT = RedactSecretsFilter()
"""One shared instance, so ``addFilter`` (which skips a filter already present) installs it once per process."""


def hide_in_logs(*secrets: str) -> None:
    """Keep ``secrets`` out of every line logged through httpx's logger, and out of :func:`redact` output."""
    for secret in secrets:
        _REDACT.add(secret)
    logging.getLogger("httpx").addFilter(_REDACT)


def redact(text: str) -> str:
    """``text`` with every secret passed to :func:`hide_in_logs` replaced by ``***``."""
    return _REDACT.redact(text)
