"""Sending to the phone (DESIGN sections 11 and 12): the configured channel, proposals with buttons, alerts, reports.

:func:`notify_proposal` is what a producer calls for a proposal that needs a decision: it re-reads the proposal,
issues a fresh single-use nonce (:mod:`fm.notify.nonces`), renders the proposal (:mod:`fm.notify.messages`) and pushes
it with Approve/Reject buttons. A push that fails discards its nonce at once, so only buttons that reached the phone
are live. :func:`send_alert` and :func:`send_report` push the plain messages. Nothing here writes to ESPN.
"""

from __future__ import annotations

from datetime import datetime, tzinfo

import httpx

from fm.config import Config
from fm.notify import nonces
from fm.notify.base import Message, NotifyChannel, NotifyError, ProposalNotice
from fm.notify.messages import alert, proposal_message, report
from fm.notify.ntfy import NtfyChannel
from fm.notify.telegram import TelegramChannel
from fm.proposals.policy import as_utc
from fm.store import Store


def open_channel(config: Config, *, client: httpx.Client | None = None) -> NotifyChannel:
    """The channel ``[notify].channel`` names, built from its ``.env`` secrets; ``NotifyError`` says what is missing."""
    if config.notify.channel == "telegram":
        return TelegramChannel.from_config(config, client=client)
    return NtfyChannel.from_config(config, client=client)


def notify_proposal(
    channel: NotifyChannel,
    store: Store,
    proposal_id: int,
    *,
    now: datetime | None = None,
    tz: tzinfo | None = None,
) -> ProposalNotice:
    """Push a proposal that is still waiting for a decision, with Approve and Reject buttons.

    ``NotifyError`` when the proposal is unknown, already decided or past its deadline, or when the push fails. The
    buttons stay live until a day after the deadline (:data:`fm.notify.nonces.LATE_GRACE`), so a late press reaches
    ``fm.proposals`` and is refused as expired rather than ignored; without a deadline they last a week. ``tz`` is the
    zone the deadline is shown in.
    """
    at = as_utc(now)
    row = store.proposals.get(proposal_id)
    if row is None:
        raise NotifyError(f"no proposal #{proposal_id}")
    if row.status != "proposed":
        raise NotifyError(f"proposal #{proposal_id} is {row.status}; only a proposed one can be sent for a decision")
    if row.deadline is not None and row.deadline <= at:
        raise NotifyError(f"proposal #{proposal_id}: its deadline {row.deadline:%Y-%m-%d %H:%M} UTC has passed")
    league = store.leagues.get(row.league_id)
    if league is None:
        raise NotifyError(f"proposal #{proposal_id} belongs to unknown league {row.league_id}")
    message = proposal_message(store, league, row, now=at, tz=tz)
    nonces.sweep(now=at)
    issued = nonces.issue(row.row_id, deadline=row.deadline, now=at)
    notice = ProposalNotice(row.row_id, issued.nonce, message)
    try:
        channel.send_proposal(notice)
    except Exception:
        nonces.discard(issued.nonce)
        raise
    return notice


def send_alert(channel: NotifyChannel, title: str, body: str = "", *, link: str | None = None) -> Message:
    """Push an :func:`fm.notify.messages.alert` and return it."""
    message = alert(title, body, link=link)
    channel.send(message)
    return message


def send_report(channel: NotifyChannel, title: str, body: str) -> Message:
    """Push a :func:`fm.notify.messages.report`, split across pushes when long, and return it."""
    message = report(title, body)
    channel.send(message)
    return message
