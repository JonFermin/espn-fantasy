"""Open trade offers and what to do about them (DESIGN section 9.4, ROADMAP #44).

This module reads and judges; it has no write path. It imports nothing from ``fm.executor`` or ``fm.browser`` and
sends nothing to ESPN (CLAUDE.md: workers propose, the executor acts; trades are approval-only). Its only output is
proposals, stored through :func:`fm.proposals.propose`, whose trade kinds have no policy field and run as ``approve``
whatever the config says.

**Which offers are open.** Offers come from ``mTransactions2`` (types ``TRADE_PROPOSAL``), not
``mPendingTransactions``. ESPN leaves an expired offer ``PENDING`` and closes it with a separate ``CANCEL`` record that
still says ``isPending``, so "open" means pending, not expired (offers expire 48 hours after ``proposedDate``) and not
named by a ``CANCEL`` record (:func:`fm.espn.models.open_only`; docs/espn-api.md section 1 #2). :func:`open_offers`
turns the open trade proposals that involve our team into :class:`OfferView` values: *incoming* when another team
proposed it, *outgoing* when we did. An offer this tool cannot value or answer (more than two teams, FAAB or draft-pick
items) is kept with its reasons in :attr:`OfferView.unsupported` rather than dropped, so it is still shown and still
blocks a duplicate.

**Incoming offers become proposals.** :func:`propose_incoming` evaluates each supported incoming offer with
:func:`fm.decide.trades.evaluate_trade` (our change in rest-of-season value and title odds, legality, P(accept) is
beside the point here) and drafts exactly one answer through :func:`fm.proposals.propose`:

- :data:`fm.decide.trades.ACCEPT` becomes a ``trade_accept`` proposal, unless accepting needs a drop on our side (a
  :class:`fm.proposals.TradeResponsePayload` carries none), which is reported in ``blocked``;
- every other recommendation (:data:`fm.decide.trades.DECLINE`, :data:`fm.decide.trades.COUNTER`) becomes a
  ``trade_decline`` proposal (a counter is made by hand; the rationale says the deal is worth one).

Each proposal carries the offer's expiry as its deadline, so it expires with the offer, and the offer's id as its
dedupe key (``offer:<id>``), so a second tick returns the open answer instead of drafting another. An offer that
already has an answer in any state but expired (rejected by us, executing, verified, failed) is left alone: a rejected
recommendation is not drafted again every tick. Values use the engine's ``p_active``, never a Claude-only signal
(:func:`fm.decide.trades.engine_p_active` reads ``inputs["news"]["before"]``), so a news item alone never moves a
trade. Outgoing offers are ours already (:class:`fm.proposals.TradePayload` is the payload :meth:`OfferView.payload`
gives them); they are listed in the decision, and :func:`duplicate_blockers` is what the propose flow uses to refuse
a second offer to the same team or one that reuses a player already in an open offer.

**Registered for the tick** as ``("nfl", "offers")`` and ``("nba", "offers")`` in :mod:`fm.decide.registry`. The
decision reads the offers from ``client`` (an :class:`fm.espn.client.EspnClient`, read-only) when it is given one,
else from the ``mTransactions2`` pages the sync captured (:func:`fm.decide.faab.load_bid_history`), whose age the
warnings say; the executor re-reads the live league before anything is sent either way.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any, Final

from fm.config import Config
from fm.decide.faab import load_bid_history
from fm.decide.registry import register
from fm.decide.trades import (
    ACCEPT,
    COUNTER,
    MarketLike,
    TradeContext,
    TradeError,
    TradeEvaluation,
    TradeSpec,
    engine_numbers,
    evaluate_trade,
    load_trade_context,
    rationale,
)
from fm.espn.client import EspnClient
from fm.espn.models import ITEM_TRADE, MatchupsView, Transaction, open_only
from fm.model.projections import BlendWeights
from fm.proposals import PolicyError, ProposalKind, TradePayload, TradeResponsePayload, propose
from fm.proposals.policy import as_utc
from fm.sports.base import ScheduleLike
from fm.store import LeagueRow, ProposalRow, Store

OFFERS_KIND: Final = "offers"
"""The decision kind this module registers for both sports (:mod:`fm.decide.registry`)."""
OFFERS_CREATED_BY: Final = "decide.offers"
"""``created_by`` of the proposals :func:`propose_incoming` stores."""
TRADE_PROPOSAL_TYPE: Final = "TRADE_PROPOSAL"
"""The ``mTransactions2`` type of an offer (and of the record that cancels or expires one)."""
ANSWER_KINDS: Final = (ProposalKind.TRADE_ACCEPT.value, ProposalKind.TRADE_DECLINE.value)
"""The proposal kinds that answer an incoming offer."""


class OffersError(ValueError):
    """The league's offers cannot be judged now (it is not synced, or the context cannot be built). The message says
    what to fix."""


class Direction(StrEnum):
    """Whose offer it is."""

    INCOMING = "incoming"  # another team proposed it to us
    OUTGOING = "outgoing"  # we proposed it


# --- reading the open offers ------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OfferView:
    """One open trade offer that involves our team, from our side: ``give`` are our players it takes, ``get`` the ones
    it brings (ESPN ids), ``other_team_id`` the team on the other side. ``unsupported`` lists why this tool cannot
    value or answer it (empty: it can)."""

    offer_id: str
    direction: Direction
    other_team_id: int
    give: tuple[int, ...]
    get: tuple[int, ...]
    proposer_team_id: int | None = None
    proposed: datetime | None = None
    expires: datetime | None = None
    unsupported: tuple[str, ...] = ()

    @property
    def supported(self) -> bool:
        return not self.unsupported

    @property
    def players(self) -> tuple[int, ...]:
        return (*self.give, *self.get)

    @property
    def dedupe_key(self) -> str:
        """The key the answer to this offer is stored under."""
        return f"offer:{self.offer_id}"

    def spec(self) -> TradeSpec:
        """The deal for :func:`fm.decide.trades.evaluate_trade`. Raises :class:`OffersError` when it is unsupported."""
        if self.unsupported:
            raise OffersError(f"offer {self.offer_id} cannot be evaluated: {'; '.join(self.unsupported)}")
        try:
            return TradeSpec(self.other_team_id, self.give, self.get)
        except TradeError as exc:
            raise OffersError(f"offer {self.offer_id}: {exc}") from exc

    def payload(self) -> TradePayload | TradeResponsePayload:
        """The proposal payload for this offer: a :class:`fm.proposals.TradeResponsePayload` naming the offer for an
        incoming one, the :class:`fm.proposals.TradePayload` it was made from for an outgoing one."""
        spec = self.spec()
        if self.direction is Direction.OUTGOING:
            return TradePayload(other_team_id=spec.other_team_id, give_espn_ids=spec.give, get_espn_ids=spec.get)
        return TradeResponsePayload(
            other_team_id=spec.other_team_id,
            give_espn_ids=spec.give,
            get_espn_ids=spec.get,
            espn_transaction_id=self.offer_id,
        )

    def describe(self) -> str:
        arrow = "to" if self.direction is Direction.OUTGOING else "from"
        give = ", ".join(map(str, self.give)) or "nothing"
        get = ", ".join(map(str, self.get)) or "nothing"
        return f"offer {self.offer_id} {arrow} team {self.other_team_id}: give {give}; get {get}"


def view_of(transaction: Transaction, team_id: int) -> OfferView | None:
    """``transaction`` as an offer involving ``team_id``, or ``None`` when it is not a trade proposal or does not touch
    that team. It does not say whether the offer is still open: see :func:`open_offers`."""
    if transaction.type != TRADE_PROPOSAL_TYPE:
        return None
    give: list[int] = []
    get: list[int] = []
    others: list[int] = []
    unsupported: list[str] = []
    for item in transaction.items:
        if item.type != ITEM_TRADE:
            reason = f"it carries {item.type} items, which this tool does not value or answer"
            if reason not in unsupported:
                unsupported.append(reason)
            continue
        if item.from_team_id == team_id and item.to_team_id is not None:
            give.append(item.player_id)
            others.append(item.to_team_id)
        elif item.to_team_id == team_id and item.from_team_id is not None:
            get.append(item.player_id)
            others.append(item.from_team_id)
        else:
            unsupported.append(f"player {item.player_id} moves between teams {item.from_team_id} and {item.to_team_id}")
    if not give and not get:
        return None  # nothing of ours is involved
    distinct = sorted(set(others))
    if len(distinct) != 1:
        unsupported.append(f"it involves teams {distinct}, and this tool answers two-team deals only")
    direction = Direction.OUTGOING if transaction.team_id == team_id else Direction.INCOMING
    return OfferView(
        offer_id=transaction.id,
        direction=direction,
        other_team_id=distinct[0] if distinct else 0,
        give=tuple(give),
        get=tuple(get),
        proposer_team_id=transaction.team_id,
        proposed=transaction.proposed_date,
        expires=transaction.expiration_date,
        unsupported=tuple(unsupported),
    )


def open_offers(transactions: Iterable[Transaction], team_id: int, now: datetime) -> tuple[OfferView, ...]:
    """The trade offers still open at ``now`` that involve ``team_id``, oldest first. Pass every ``mTransactions2``
    record (the ``CANCEL`` records that close an offer must be among them: :func:`fm.espn.models.open_only`)."""
    records = tuple(transactions)
    found = (view_of(transaction, team_id) for transaction in open_only(records, as_utc(now)))
    return tuple(view for view in found if view is not None)


def incoming(offers: Iterable[OfferView]) -> tuple[OfferView, ...]:
    return tuple(offer for offer in offers if offer.direction is Direction.INCOMING)


def outgoing(offers: Iterable[OfferView]) -> tuple[OfferView, ...]:
    return tuple(offer for offer in offers if offer.direction is Direction.OUTGOING)


def duplicate_blockers(
    offers: Iterable[OfferView], other_team_id: int, give: Iterable[int], get: Iterable[int]
) -> list[str]:
    """Why a new offer to ``other_team_id`` for these players must not be sent while ``offers`` are open: one open
    offer of ours per team (DESIGN 9.4), and no player in two offers at once (ESPN reserves a player named in a pending
    trade). Every reason is listed."""
    wanted = {*give, *get}
    reasons: list[str] = []
    for offer in offers:
        if offer.direction is Direction.OUTGOING and offer.other_team_id == other_team_id:
            reasons.append(f"an offer of ours to team {other_team_id} is already open ({offer.offer_id})")
        shared = sorted(wanted & set(offer.players))
        if shared:
            reasons.append(f"players {shared} are already in the open offer {offer.offer_id}")
    return reasons


# --- answering the incoming ones --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OfferOutcome:
    """One incoming offer's fate: the ``evaluation``, the stored (or already open) ``proposal`` answering it, or the
    ``blocked`` reason there is none. ``existing`` means the proposal was already there."""

    offer: OfferView
    evaluation: TradeEvaluation | None = None
    proposal: ProposalRow | None = None
    blocked: str | None = None
    existing: bool = False


@dataclass(frozen=True, slots=True)
class OffersDecision:
    """What :func:`decide_offers` returns. ``proposals`` and ``blocked`` are what the tick reads."""

    league: LeagueRow
    answers: tuple[OfferOutcome, ...] = ()
    outgoing: tuple[OfferView, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def proposals(self) -> tuple[ProposalRow, ...]:
        return tuple(outcome.proposal for outcome in self.answers if outcome.proposal is not None)

    @property
    def blocked(self) -> tuple[tuple[OfferView, str], ...]:
        return tuple((outcome.offer, outcome.blocked) for outcome in self.answers if outcome.blocked is not None)


def answer_kind(evaluation: TradeEvaluation) -> ProposalKind | str:
    """The proposal kind that answers an evaluated offer, or the reason there is none. Only :data:`ACCEPT` accepts, and
    not when accepting needs a drop on our side (the payload cannot carry one); everything else declines."""
    if evaluation.recommendation == ACCEPT:
        if evaluation.legality.drops_ours:
            return (
                "accepting needs us to drop a player to make room, which a trade response cannot carry; "
                "answer it by hand"
            )
        return ProposalKind.TRADE_ACCEPT
    return ProposalKind.TRADE_DECLINE


def _already_answered(store: Store, league: LeagueRow, offer: OfferView) -> ProposalRow | None:
    """The offer's answer in any state but expired (an expired one is stale: the offer it answered ran out)."""
    for row in store.proposals.find(league_id=league.row_id, kinds=ANSWER_KINDS):
        if row.dedupe_key == offer.dedupe_key and row.status != "expired":
            return row
    return None


def _settled(store: Store, league: LeagueRow, offer: OfferView) -> OfferOutcome:
    """The outcome of an offer that needs no evaluation: unsupported, or answered already (as in
    :func:`propose_incoming`)."""
    if not offer.supported:
        return OfferOutcome(offer, blocked="; ".join(offer.unsupported))
    return OfferOutcome(offer, proposal=_already_answered(store, league, offer), existing=True)


def _answer_numbers(ctx: TradeContext, evaluation: TradeEvaluation, offer: OfferView) -> dict[str, Any]:
    return {
        **engine_numbers(ctx, evaluation),
        "offer": {
            "id": offer.offer_id,
            "from_team_id": offer.proposer_team_id,
            "expires": None if offer.expires is None else offer.expires.isoformat(),
        },
        "recommendation": evaluation.recommendation,
    }


def _answer_rationale(ctx: TradeContext, evaluation: TradeEvaluation, offer: OfferView) -> str:
    deal = rationale(ctx, evaluation).replace("Give ", f"Offer {offer.offer_id}: give ", 1)
    verdict = f"Recommendation: {evaluation.recommendation} ({evaluation.reason})."
    if evaluation.recommendation == COUNTER:
        verdict += " Declining is proposed; the deal is worth a counter-offer made by hand."
    return f"{deal} {verdict}"


def propose_incoming(
    store: Store,
    config: Config,
    ctx: TradeContext,
    offers: Iterable[OfferView],
    *,
    runs: int = 10_000,
    now: datetime | None = None,
) -> list[OfferOutcome]:
    """Evaluate each open incoming offer of ``offers`` and draft its answer (the module docs give the rules). A policy
    refusal (the trade deadline has passed, an untouchable, a pause) is reported in ``blocked``, never raised, and an
    offer that is unsupported or no longer matches the synced rosters (run ``fm sync``) is reported the same way."""
    at = as_utc(now if now is not None else ctx.now)
    outcomes: list[OfferOutcome] = []
    for offer in incoming(offers):
        if not offer.supported:
            outcomes.append(OfferOutcome(offer, blocked="; ".join(offer.unsupported)))
            continue
        prior = _already_answered(store, ctx.league, offer)
        if prior is not None:
            outcomes.append(OfferOutcome(offer, proposal=prior, existing=True))
            continue
        try:
            evaluation = evaluate_trade(ctx, offer.spec(), runs=runs)
        except TradeError as exc:
            outcomes.append(OfferOutcome(offer, blocked=f"{exc}; run fm sync"))
            continue
        kind = answer_kind(evaluation)
        if isinstance(kind, str) and not isinstance(kind, ProposalKind):
            outcomes.append(OfferOutcome(offer, evaluation, blocked=kind))
            continue
        payload = offer.payload()
        assert isinstance(payload, TradeResponsePayload)
        try:
            row = propose(
                store,
                config,
                ctx.league,
                kind,
                payload,
                created_by=OFFERS_CREATED_BY,
                scoring_period_id=ctx.lock_period,
                engine_numbers=_answer_numbers(ctx, evaluation, offer),
                rationale=_answer_rationale(ctx, evaluation, offer),
                deadline=offer.expires,
                dedupe_key=offer.dedupe_key,
                now=at,
            )
        except PolicyError as exc:
            outcomes.append(OfferOutcome(offer, evaluation, blocked=str(exc)))
            continue
        outcomes.append(OfferOutcome(offer, evaluation, row))
    return outcomes


# --- the decision -----------------------------------------------------------------------------------------------------


def _league_row(store: Store, league: str | LeagueRow) -> LeagueRow:
    if isinstance(league, LeagueRow):
        return league
    row = store.leagues.by_key(league)
    if row is None:
        raise OffersError(f"league {league!r} is not in the store; run fm sync")
    return row


def read_offers(
    store: Store, league: LeagueRow, now: datetime, client: EspnClient | None, warnings: list[str]
) -> tuple[OfferView, ...]:
    """The league's open offers: read live through ``client`` (reads only), else from the ``mTransactions2`` pages the
    sync captured, with a warning that they may be stale."""
    if client is not None:
        return open_offers(client.pending_offers(now=now), league.team_id, now)
    history, notes = load_bid_history(store, league)
    warnings.extend(notes)
    if not history:
        warnings.append(
            f"{league.key}: no captured transactions to read offers from (no ESPN session, no sync capture)"
        )
    else:
        warnings.append(f"{league.key}: offers read from the sync's captured transactions, which may be stale")
    return open_offers(history, league.team_id, now)


def decide_offers(
    store: Store,
    config: Config,
    league: str | LeagueRow,
    *,
    now: datetime | None = None,
    schedule: ScheduleLike | None = None,
    client: EspnClient | None = None,
    offers: Sequence[OfferView] | None = None,
    matchups: MatchupsView | None = None,
    market: MarketLike | None = None,
    weights: BlendWeights | None = None,
    runs: int = 10_000,
    store_proposals: bool = True,
) -> OffersDecision:
    """Answer the league's open incoming trade offers with proposals (the module docs give the rules).

    ``offers`` overrides reading (tests, a caller that already has them); otherwise they come from ``client`` or the
    sync's captures. With no incoming offer nothing else is loaded. ``matchups`` and ``market`` are
    :func:`fm.decide.trades.load_trade_context`'s; without them offers are judged on rest-of-season value and our own
    ranks, and the warnings say so. With ``store_proposals`` false nothing is stored (offers are only evaluated).
    Raises :class:`OffersError` for a league that is not synced.
    """
    at = as_utc(now)
    row = _league_row(store, league)
    warnings: list[str] = []
    found = tuple(offers) if offers is not None else read_offers(store, row, at, client, warnings)
    waiting = incoming(found)
    if not waiting:
        return OffersDecision(row, outgoing=outgoing(found), warnings=tuple(warnings))
    if store_proposals and not any(o.supported and _already_answered(store, row, o) is None for o in waiting):
        settled = [_settled(store, row, offer) for offer in waiting]  # every offer has its answer: skip the model
        return OffersDecision(row, tuple(settled), outgoing(found), tuple(dict.fromkeys(warnings)))
    try:
        ctx = load_trade_context(
            store, row, now=at, config=config, schedule=schedule, matchups=matchups, market=market, weights=weights
        )
    except TradeError as exc:
        raise OffersError(str(exc)) from exc
    warnings.extend(ctx.warnings)
    if not store_proposals:
        answers = [_evaluated_only(ctx, offer, runs) for offer in waiting]
    else:
        answers = propose_incoming(store, config, ctx, waiting, runs=runs, now=at)
    return OffersDecision(row, tuple(answers), outgoing(found), tuple(dict.fromkeys(warnings)))


def _evaluated_only(ctx: TradeContext, offer: OfferView, runs: int) -> OfferOutcome:
    if not offer.supported:
        return OfferOutcome(offer, blocked="; ".join(offer.unsupported))
    try:
        return OfferOutcome(offer, evaluate_trade(ctx, offer.spec(), runs=runs))
    except TradeError as exc:
        return OfferOutcome(offer, blocked=f"{exc}; run fm sync")


for _sport in ("nfl", "nba"):
    register(_sport, OFFERS_KIND, decide_offers)
