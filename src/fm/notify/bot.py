"""The listener behind ``fm bot`` (DESIGN section 13): button presses become ``fm.proposals`` decisions.

:func:`handle_reply` is the trust boundary. A press counts only when :func:`fm.notify.nonces.consume` accepts its nonce
for that proposal (issued here, unused, not expired); anything else is logged, answered with "no longer valid" where
the channel can do that privately (a Telegram toast), and never decided. An accepted press calls
``fm.proposals.approve`` or ``reject`` with ``decided_by`` set to the channel's name. Those refuse a proposal that is
no longer open and expire one whose deadline has passed on the way, so an old button can never approve an expired
proposal; the refusal is what the phone hears back. After a decision the proposal's other nonces are revoked, so the
buttons of a second notification die with the first. An unexpected failure puts the nonce back before it propagates:
the press did not land, so pressing again must work.

:func:`listen` polls until stopped. Both channels wait server-side (Telegram's long poll, the ntfy subscription), so a
press lands within seconds; a channel or database error is logged and retried with backoff instead of ending the
listener. Recording a decision is all it does: executing an approved proposal belongs to the executor.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from collections.abc import Callable
from datetime import datetime

from fm.notify import nonces
from fm.notify.base import DecisionResult, NotifyChannel, NotifyError, Reply
from fm.notify.messages import describe_payload
from fm.proposals import ProposalError, approve, kind_spec, parse_payload, pause_state, reject
from fm.proposals.policy import as_utc
from fm.store import ProposalRow, Store

logger = logging.getLogger(__name__)

INVALID_PRESS = "This button is no longer valid."
DEFAULT_INTERVAL_S = 1.0
"""Pause between polls; the polls themselves wait server-side for a press."""
FIRST_BACKOFF_S = 2.0
MAX_BACKOFF_S = 60.0


def handle_reply(
    store: Store, channel: NotifyChannel, reply: Reply, *, now: datetime | None = None
) -> DecisionResult | None:
    """Record one press. Returns what happened, or ``None`` when it was ignored (unknown, used or expired nonce)."""
    at = as_utc(now)
    issued = nonces.consume(reply.nonce, reply.proposal_id, now=at)
    if issued is None:
        logger.warning(
            "ignored %s for proposal #%d via %s (from %s): unknown, used or expired button",
            reply.decision,
            reply.proposal_id,
            channel.name,
            reply.sender,
        )
        try:
            channel.dismiss(reply, INVALID_PRESS)
        except NotifyError as exc:
            logger.warning("could not answer the ignored press: %s", exc)
        return None
    decide = approve if reply.decision == "approve" else reject
    try:
        row = decide(store, reply.proposal_id, decided_by=channel.name, now=at)
    except ProposalError as exc:
        result = DecisionResult(reply.proposal_id, reply.decision, ok=False, detail=str(exc))
        logger.warning("%s of proposal #%d via %s refused: %s", reply.decision, reply.proposal_id, channel.name, exc)
    except Exception:
        nonces.restore(issued)  # the decision did not happen, so the button must still work
        raise
    else:
        nonces.revoke(row.row_id)
        result = DecisionResult(row.row_id, reply.decision, ok=True, detail=_detail(store, row))
        logger.info("proposal #%d %s via %s", row.row_id, result.done, channel.name)
    try:
        channel.confirm(reply, result)
    except NotifyError as exc:
        logger.warning("proposal #%d: decision recorded, but the confirmation failed: %s", reply.proposal_id, exc)
    return result


def listen(
    store: Store,
    channel: NotifyChannel,
    *,
    once: bool = False,
    stop: Callable[[], bool] | None = None,
    interval_s: float = DEFAULT_INTERVAL_S,
    sleep: Callable[[float], object] = time.sleep,
    clock: Callable[[], datetime] | None = None,
    on_result: Callable[[DecisionResult], object] | None = None,
) -> int:
    """Poll ``channel`` and record every press until ``stop()`` is true, or forever. ``once`` handles only the presses
    already waiting and returns. Returns how many decisions were recorded or refused (ignored presses do not count).

    A channel error (``NotifyError``) is logged and retried with backoff. A database error drops only that press, whose
    nonce :func:`handle_reply` put back, so pressing again records it. With ``once`` both are raised instead.
    """
    nonces.sweep(now=None if clock is None else clock())
    handled = 0
    backoff = FIRST_BACKOFF_S
    while True:
        try:
            replies = channel.poll(wait=not once)
        except NotifyError as exc:
            if once:
                raise
            logger.warning("%s: %s; retrying in %.0fs", channel.name, exc, backoff)
            replies, pause, backoff = [], backoff, min(backoff * 2, MAX_BACKOFF_S)
        else:
            pause, backoff = interval_s, FIRST_BACKOFF_S
        for reply in replies:
            try:
                result = handle_reply(store, channel, reply, now=None if clock is None else clock())
            except sqlite3.Error as exc:
                if once:
                    raise
                logger.error("could not record the press on proposal #%d (press again): %s", reply.proposal_id, exc)
                continue
            if result is not None:
                handled += 1
                if on_result is not None:
                    on_result(result)
        if once or (stop is not None and stop()):
            return handled
        sleep(pause)


def _detail(store: Store, row: ProposalRow) -> str:
    """``nfl lineup change: Player A: BE -> QB``, plus a note when ``fm pause`` holds an approved move back."""
    league = store.leagues.get(row.league_id)
    try:
        payload = parse_payload(row)
        what = payload.summary() if league is None else describe_payload(store, league, payload)
        text = f"{kind_spec(row.kind).label}: {what}"
    except (ProposalError, ValueError):
        text = row.kind
    if league is not None:
        text = f"{league.key} {text}"
    paused = pause_state()
    if paused is not None and row.status == "approved":
        text += f" ({paused.describe()}: nothing runs until fm resume)"
    return text
