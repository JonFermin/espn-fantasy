"""Phone channels (DESIGN section 11): approvals with buttons, alerts and reports over Telegram or ntfy.

Usage::

    from fm.notify import notify_proposal, open_channel, send_alert

    channel = open_channel(config)                      # [notify].channel with its .env secrets
    notify_proposal(channel, store, row.row_id)         # Approve/Reject buttons carrying a single-use nonce
    send_alert(channel, "Lineup not set", "start X over Y", link=espn_team_url)

``fm bot`` (:func:`listen`) turns a press into ``fm.proposals.approve`` / ``reject`` with ``decided_by`` set to the
channel; ``fm notify setup`` captures the Telegram chat id or generates the ntfy topics. Nothing here writes to ESPN,
and no channel ever carries a proposal's execution token: buttons carry nonces from :mod:`fm.notify.nonces`.

- :mod:`fm.notify.base`: the :class:`NotifyChannel` interface, message types, the button encoding, log redaction.
- :mod:`fm.notify.messages`: what the phone shows (proposals in player names, confirmations, alerts, reports).
- :mod:`fm.notify.telegram` / :mod:`fm.notify.ntfy`: the two adapters, neither needing an inbound port.
- :mod:`fm.notify.nonces`: single-use button tokens. :mod:`fm.notify.send`: pushing. :mod:`fm.notify.bot`: listening.
"""

from __future__ import annotations

from fm.notify.base import (
    APPROVE_LABEL,
    BUTTONS,
    DECISIONS,
    PAST_TENSE,
    REJECT_LABEL,
    TEXT_LIMIT,
    Callback,
    Decision,
    DecisionResult,
    Message,
    NotifyChannel,
    NotifyError,
    Priority,
    ProposalNotice,
    Reply,
    decode_callback,
    encode_callback,
    hide_in_logs,
    split_text,
)
from fm.notify.bot import INVALID_PRESS, handle_reply, listen
from fm.notify.messages import alert, decision_message, describe_payload, proposal_message, report
from fm.notify.nonces import DEFAULT_TTL, LATE_GRACE, IssuedNonce
from fm.notify.ntfy import NtfyChannel, generate_topic
from fm.notify.send import notify_proposal, open_channel, send_alert, send_report
from fm.notify.telegram import ChatCapture, TelegramChannel

__all__ = [
    "APPROVE_LABEL",
    "BUTTONS",
    "DECISIONS",
    "DEFAULT_TTL",
    "INVALID_PRESS",
    "LATE_GRACE",
    "PAST_TENSE",
    "REJECT_LABEL",
    "TEXT_LIMIT",
    "Callback",
    "ChatCapture",
    "Decision",
    "DecisionResult",
    "IssuedNonce",
    "Message",
    "NotifyChannel",
    "NotifyError",
    "NtfyChannel",
    "Priority",
    "ProposalNotice",
    "Reply",
    "TelegramChannel",
    "alert",
    "decision_message",
    "decode_callback",
    "describe_payload",
    "encode_callback",
    "generate_topic",
    "handle_reply",
    "hide_in_logs",
    "listen",
    "notify_proposal",
    "open_channel",
    "proposal_message",
    "report",
    "send_alert",
    "send_report",
    "split_text",
]
