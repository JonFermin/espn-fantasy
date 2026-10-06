"""Proposal lifecycle (DESIGN section 11): proposed -> approved | rejected | expired -> executing -> verified | failed.

Every function takes the open :class:`fm.store.Store` and an optional aware ``now`` (tests pin the clock) and writes
inside one transaction, so two processes (the tick and the bot) cannot both move a proposal.

- :func:`propose` first expires the league's proposals whose deadline has passed (a stale one is neither handed back
  as a duplicate nor holds a transaction slot), then runs :func:`fm.proposals.policy.evaluate` and stores the row with
  the setting it was made under (``proposals.policy``). It raises ``PolicyError`` when blocked. A ``dedupe_key``
  matching an open proposal in the same league returns that proposal instead of a duplicate.
- :func:`approve` issues the single-use execution token; :func:`reject` works on proposed and approved rows (a pending
  approval can still be withdrawn). Both first expire a row whose deadline has passed, so an approval is void after
  the lock: an expired proposal cannot be approved.
- :func:`expire_due` is the tick's sweep; :func:`auto_approve_due` approves ``auto`` proposals still unanswered inside
  :data:`AUTO_LEAD` of their deadline (T-15), which is the only way ``auto`` ever acts, and does nothing while paused.
- :func:`begin_execution` is the executor's entry: approved, before the deadline, not paused, and the token matches and
  is unused, all checked and consumed atomically, so a second attempt with the same token fails.
- :func:`finish_execution` records the verified or failed outcome.

Human decisions (``approve`` / ``reject``) are allowed while paused; automatic ones and execution are not.

Interface note for the phone channels (ROADMAP #18)
---------------------------------------------------
The only token on a row is ``execution_token``, minted by :func:`approve` and consumed by :func:`begin_execution`. It
authorizes a write, so it never leaves this machine, and nothing exists on a proposal before approval. The ntfy
Approve/Reject buttons (DESIGN section 11) therefore carry an *approval nonce*, not this token: #18 mints a fresh
single-use nonce for every notification it sends, keeps the nonce -> proposal map in its own store (its scope; a
proposal can be notified more than once, so a column on the row would not fit), and calls :func:`approve` or
:func:`reject` only for a reply whose nonce it issued and has not seen before. No schema change is needed: unknown or
reused nonces are ignored, and :func:`approve` still refuses an expired proposal.
"""

from __future__ import annotations

import secrets
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any, Literal

from fm.config import Approval, Config
from fm.proposals.pause import pause_state
from fm.proposals.payloads import Payload
from fm.proposals.policy import ProposalError, ProposalKind, as_utc, evaluate, kind_spec
from fm.store import OPEN_PROPOSAL_STATUSES, LeagueRow, ProposalRow, ProposalStatus, Store

AUTO_LEAD = timedelta(minutes=15)
"""How long before its deadline an unanswered ``auto`` proposal is approved on its own."""
DECIDED_BY_AUTO = "auto"
DECIDED_BY_EXPIRY = "expiry"
TOKEN_BYTES = 32

type Outcome = Literal["verified", "failed"]

TRANSITIONS: Mapping[ProposalStatus, frozenset[ProposalStatus]] = {
    "proposed": frozenset({"approved", "rejected", "expired"}),
    "approved": frozenset({"executing", "rejected", "expired"}),
    "executing": frozenset({"verified", "failed"}),
    "rejected": frozenset(),
    "expired": frozenset(),
    "verified": frozenset(),
    "failed": frozenset(),
}
"""Legal status changes. Terminal statuses have no exits; a new attempt is a new proposal."""


class LifecycleError(ProposalError):
    """An illegal status change, an unknown proposal, or an execution token that does not match or was used."""


class PausedError(ProposalError):
    """``fm pause`` is in force, so nothing automatic runs and nothing executes."""


def propose(
    store: Store,
    config: Config,
    league: LeagueRow,
    kind: ProposalKind | str,
    payload: Payload,
    *,
    created_by: str,
    scoring_period_id: int | None = None,
    engine_numbers: Mapping[str, Any] | None = None,
    rationale: str | None = None,
    deadline: datetime | None = None,
    dedupe_key: str | None = None,
    max_setting: Approval | None = None,
    now: datetime | None = None,
) -> ProposalRow:
    """Store a proposal once it clears policy; raise ``PolicyError`` listing every reason it does not.

    ``created_by`` names the producer (``decide.lineup``, ``advisor.strategist``, ``mcp``). ``deadline`` is when the
    move stops making sense; it is required for a kind whose setting is ``auto``. With a ``dedupe_key``, an open
    proposal in the league carrying the same key is returned instead of storing a second one. ``max_setting`` caps the
    setting the proposal runs under (``fm.proposals.policy.evaluate``): ``approve`` keeps it from ever being
    auto-approved. The league's proposals whose deadline has already passed are expired first, so one of them is never
    the duplicate handed back and no longer counts toward the weekly cap.
    """
    at = as_utc(now)
    spec = kind_spec(kind)
    expire_due(store, now=at, league_id=league.row_id)
    with store.db.transaction():
        if dedupe_key is not None:
            for existing in store.proposals.open(league.row_id):
                if existing.dedupe_key == dedupe_key:
                    return existing
        verdict = evaluate(
            store,
            config,
            league,
            spec.kind,
            payload,
            scoring_period_id=scoring_period_id,
            deadline=deadline,
            max_setting=max_setting,
            now=at,
        )
        verdict.raise_if_blocked()
        row = ProposalRow(
            league_id=league.row_id,
            kind=spec.kind.value,
            status="proposed",
            policy=verdict.setting,
            scoring_period_id=scoring_period_id,
            payload=payload.model_dump(mode="json"),
            engine_numbers=dict(engine_numbers or {}),
            rationale=rationale,
            deadline=None if deadline is None else as_utc(deadline),
            created_by=created_by,
            created_at=at,
            dedupe_key=dedupe_key,
        )
        return store.proposals.insert(row)


def get_proposal(store: Store, proposal_id: int) -> ProposalRow:
    """The proposal, or ``LifecycleError`` when there is none with that id."""
    row = store.proposals.get(proposal_id)
    if row is None:
        raise LifecycleError(f"no proposal #{proposal_id}")
    return row


def approve(store: Store, proposal_id: int, *, decided_by: str, now: datetime | None = None) -> ProposalRow:
    """``proposed -> approved`` and issue the execution token. ``decided_by`` is the channel or person (``cli``,
    ``telegram``, ``auto``). A proposal past its deadline is expired instead and the call fails."""
    at = as_utc(now)
    _expire_if_due(store, proposal_id, at)
    with store.db.transaction():
        row = get_proposal(store, proposal_id)
        _check(row, "approved", "approve")
        return store.proposals.update(
            row.model_copy(
                update={
                    "status": "approved",
                    "decided_by": decided_by,
                    "decided_at": at,
                    "execution_token": secrets.token_urlsafe(TOKEN_BYTES),
                }
            )
        )


def reject(store: Store, proposal_id: int, *, decided_by: str, now: datetime | None = None) -> ProposalRow:
    """``proposed | approved -> rejected``. Rejecting an approved proposal withdraws the approval before it runs."""
    at = as_utc(now)
    _expire_if_due(store, proposal_id, at)
    with store.db.transaction():
        row = get_proposal(store, proposal_id)
        _check(row, "rejected", "reject")
        return store.proposals.update(
            row.model_copy(update={"status": "rejected", "decided_by": decided_by, "decided_at": at})
        )


def expire_due(store: Store, *, now: datetime | None = None, league_id: int | None = None) -> list[ProposalRow]:
    """Expire every proposed or approved proposal whose deadline is at or before ``now``; returns the rows expired."""
    at = as_utc(now)
    expired: list[ProposalRow] = []
    with store.db.transaction():
        for row in store.proposals.find(league_id=league_id, statuses=("proposed", "approved")):
            if _is_due(row, at):
                expired.append(_expire(store, row, at))
    return expired


def auto_approve_due(
    store: Store, *, now: datetime | None = None, lead: timedelta = AUTO_LEAD, league_id: int | None = None
) -> list[ProposalRow]:
    """Approve, as ``auto``, every unanswered ``auto`` proposal within ``lead`` of a deadline that has not passed.

    Run :func:`expire_due` first so a proposal past its deadline expires rather than fires. Returns the rows approved;
    nothing happens while paused.
    """
    at = as_utc(now)
    if pause_state() is not None:
        return []
    approved: list[ProposalRow] = []
    with store.db.transaction():
        for row in store.proposals.find(league_id=league_id, statuses=("proposed",)):
            if row.policy != "auto" or row.deadline is None:
                continue
            if row.deadline - lead <= at < row.deadline:
                approved.append(approve(store, row.row_id, decided_by=DECIDED_BY_AUTO, now=at))
    return approved


def begin_execution(store: Store, proposal_id: int, token: str, *, now: datetime | None = None) -> ProposalRow:
    """``approved -> executing``, consuming the single-use execution token.

    Refused (``PausedError``) while paused, and (``LifecycleError``) when the proposal is not approved, its deadline
    has passed (it is expired on the spot), or ``token`` is not its unused execution token. All of it happens in one
    transaction, so a second caller holding the same token loses.
    """
    at = as_utc(now)
    paused = pause_state()
    if paused is not None:
        raise PausedError(f"cannot execute proposal #{proposal_id}: {paused.describe()}; run fm resume")
    _expire_if_due(store, proposal_id, at)
    with store.db.transaction():
        row = get_proposal(store, proposal_id)
        _check(row, "executing", "execute")
        if not store.proposals.consume_execution_token(row.row_id, token, at):
            raise LifecycleError(f"proposal #{proposal_id}: execution token does not match or was already used")
        consumed = get_proposal(store, proposal_id)
        return store.proposals.update(consumed.model_copy(update={"status": "executing"}))


def finish_execution(store: Store, proposal_id: int, outcome: Outcome) -> ProposalRow:
    """``executing -> verified | failed``: the executor's verdict after the API re-read."""
    with store.db.transaction():
        row = get_proposal(store, proposal_id)
        _check(row, outcome, f"mark {outcome}")
        return store.proposals.update(row.model_copy(update={"status": outcome}))


def _expire_if_due(store: Store, proposal_id: int, at: datetime) -> None:
    """Lazy expiry on the way to a decision, committed on its own so it survives the decision being refused."""
    with store.db.transaction():
        row = get_proposal(store, proposal_id)
        if _is_due(row, at):
            _expire(store, row, at)


def _is_due(row: ProposalRow, at: datetime) -> bool:
    return row.status in ("proposed", "approved") and row.deadline is not None and row.deadline <= at


def _expire(store: Store, row: ProposalRow, at: datetime) -> ProposalRow:
    return store.proposals.update(
        row.model_copy(update={"status": "expired", "decided_by": DECIDED_BY_EXPIRY, "decided_at": at})
    )


def _check(row: ProposalRow, target: ProposalStatus, verb: str) -> None:
    if target not in TRANSITIONS[row.status]:
        detail = f"is {row.status}"
        if row.status == "expired" and row.deadline is not None:
            detail = f"expired at {row.deadline:%Y-%m-%d %H:%M} UTC"
        elif row.decided_by is not None and row.status in ("approved", "rejected"):
            detail = f"was {row.status} by {row.decided_by}"
        raise LifecycleError(f"cannot {verb} proposal #{row.row_id}: it {detail}")


def is_open(row: ProposalRow) -> bool:
    """Still in flight: proposed, approved or executing."""
    return row.status in OPEN_PROPOSAL_STATUSES
