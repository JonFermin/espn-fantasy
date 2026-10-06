"""``propose_trade``, ``respond_trade`` and ``cancel_trade``: a trade offer, our answer to an incoming one, and the
withdrawal of our own open offer (DESIGN 6.3, docs/espn-api.md section 4 "propose_trade", "respond_trade", "cancel").

**Trades are approval-only, and that is not configurable** (CLAUDE.md). The trade kinds have no policy field
(:data:`fm.proposals.policy.KINDS`), ``effective_setting`` is ``approve`` for all of them, and these flows refuse a
proposal that was made under ``auto`` or approved by ``auto`` even if a row said so: every precondition list starts
there. An offer is a message to another manager, so nothing here is retried and nothing is sent without the executor's
single-use token.

:class:`ProposeTrade` serves ``trade_propose`` (a :class:`fm.proposals.TradePayload`: our players to give, theirs to
get), :class:`RespondTrade` serves ``trade_accept`` and ``trade_decline`` (a :class:`fm.proposals.TradeResponsePayload`
naming the incoming offer and what it exchanges) and :class:`CancelTrade` serves ``trade_cancel`` (a
:class:`fm.proposals.TransactionCancelPayload`: the id of our open offer). The executor (:func:`fm.executor.execute`)
drives them and owns the write-safety rules.

Preconditions, read through the API (``mRoster``, ``mSettings``, ``proTeamSchedules_wl``, ``mPendingTransactions`` and
``mTransactions2``) and listed all at once:

- **Open offers.** ``EspnClient.pending_offers`` keeps only the open ones: ESPN leaves an expired offer ``PENDING`` and
  closes it with a separate ``CANCEL`` record that still says ``isPending`` (docs/espn-api.md section 1 #2). A proposal
  never duplicates one: no second open offer of ours to the same team, and no player already in an open offer
  (:func:`fm.decide.offers.duplicate_blockers`). A response or a cancel needs its offer open, the right way round
  (incoming to answer, ours to cancel) and, for a response, still exchanging exactly what the proposal evaluated.
- **The deadline** is the league's own, ``LeagueSettings.trade.deadline``: a proposal or an acceptance after it is
  refused, a decline or a cancel is not.
- **Locks, for every player on both sides** of a proposal or an acceptance: ESPN's ``tradeLocked`` and ``rosterLocked``
  flags, and the league's roster lock (``rosterLocktimeType``: each player's own game in the real NFL league, the day's
  first tip in the NBA league) applied to the pro schedule. An unmapped lock type refuses rather than guesses. ESPN has
  accepted a trade that included a locked player (DESIGN 6.3), so this tool checks both sides itself.
- **Rosters.** Every player is where the deal says. Our roster must not end over the league's size: a trade payload
  carries no drop, so a deal that needs one is refused with the reason (the other team's overflow is only noted: its
  manager drops when he answers).

API mode sends the web client's envelope, built by :mod:`fm.browser.transactions` and held body for body to the
captures (``ffl/write_TRADE_PROPOSAL_1.json``, observed; ``{ffl,fba}/write_TRADE_PROPOSAL_derived.json``, derived from
the saved client code): ``TRADE`` items, ours first, ``expirationDate`` as an ISO string with milliseconds (now plus
:data:`EXPIRY_DAYS`, the builder's default) and ``comment: ""``. The envelope's ``expiration_date`` field is annotated
``int`` for ESPN's records, which carry epoch milliseconds, so this module adds the string to the serialized body
rather than loosen that type. Responses and cancels have no capture (they rest on the saved client code); they carry
``relatedTransactionId`` and, for a decline, ``comment``.

The proposal also has a UI walk (:meth:`ProposeTrade.run_ui`), the one page the trade-review capture drove: the
builder (``/{sport}/team/trade?...&fromTeamId=``), ``Trade <Player>`` checkboxes, ``Continue``, the expiry select and
``Send Trade Proposal``, which is the one state-changing click (:meth:`fm.browser.flows.UiDriver.confirm`). **It is not
listed in** :attr:`ProposeTrade.modes` **yet**: the weekly UI drill (``fm.browser.drills``) walks every registered UI
flow up to its confirm, and its safety layers know only the lineup, add and claim controls, so a flow whose final click
is ``Send Trade Proposal`` must not register a UI mode until the drill has a planner for it and treats that button as a
final save. Until then the walk is exercised directly (``tests/executor/test_trade_flow.py`` runs it through a
subclass that lists both modes) and the executor runs the proposal in API mode only. No page for responding to or
withdrawing an offer has been observed, so those flows are API only by design.

Verification re-reads the offers: a proposal is verified by an open offer of ours to that team with the same players
and an expiry ahead; an accept by the trade having executed (the record says so, or our roster shows it: a league that
holds trades for review leaves the offer open, which is *not* verified); a decline or a cancel by the offer no longer
being open and not having executed.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Final

from fm.browser import selectors
from fm.browser.flows import (
    Flow,
    FlowContext,
    Mode,
    ModeUnavailableError,
    Preconditions,
    UiDriver,
    Verification,
    WriteRequest,
    WriteResponse,
    register_flow,
)
from fm.browser.flows.add_drop import LeagueRead, check_saved, cutoff_for, open_page, require_period, stamp
from fm.browser.transactions import Envelope, ExecutionType, TransactionType, trade_item, transaction_request
from fm.browser.transactions import ui_may_follow as transactions_ui_may_follow
from fm.decide.offers import Direction, OfferView, duplicate_blockers, open_offers, view_of
from fm.espn.client import EspnSchemaError
from fm.espn.models import Transaction
from fm.proposals import DECIDED_BY_AUTO, Payload, ProposalKind, TradePayload, TradeResponsePayload
from fm.proposals import TransactionCancelPayload as CancelPayload

EXPIRY_DAYS: Final = 2
"""How long an offer we send stays open: the trade builder's default (``2 Days``), which matches the 48 hours real
offers expire after."""
UI_EXPIRY_VALUE: Final = str(EXPIRY_DAYS)
"""The builder's expiry select value for :data:`EXPIRY_DAYS` (``1`` to ``7``, shown as ``2 Days``)."""
OFFER_TYPES: Final = ("TRADE_PROPOSAL",)


# --- what a precondition read finds -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Piece:
    """One player in a deal as the rosters show him."""

    espn_id: int
    name: str
    team_id: int
    slot: int
    trade_locked: bool
    roster_locked: bool
    cutoff: datetime | None

    @property
    def label(self) -> str:
        return f"{self.name} ({self.espn_id})" if self.name else f"player {self.espn_id}"

    def to_json(self) -> dict[str, Any]:
        return {
            "espn_id": self.espn_id,
            "name": self.name,
            "team_id": self.team_id,
            "slot": self.slot,
            "trade_locked": self.trade_locked,
            "roster_locked": self.roster_locked,
            "cutoff": None if self.cutoff is None else self.cutoff.isoformat(),
        }


@dataclass(frozen=True, slots=True)
class _Reading:
    """One precondition read: the league, ESPN's open offers and waiver claims, and our view of the offers."""

    read: LeagueRead
    pending: tuple[Transaction, ...]
    views: tuple[OfferView, ...]

    @property
    def latest(self) -> int:
        return self.read.latest


def _read[P: Payload](ctx: FlowContext[P]) -> _Reading:
    """Rosters of ESPN's current period (a trade is made now, in the period that is current), settings, the pro
    schedule and the open offers. Raises ``EspnClientError`` (a refusal that spends no token) when a view cannot be
    read."""
    view = ctx.reader.rosters(None).data
    settings = ctx.reader.settings().data
    latest: int | None = None
    if view.status is not None and view.status.latest_scoring_period:
        latest = view.status.latest_scoring_period
    elif settings.current_scoring_period:
        latest = settings.current_scoring_period
    if latest is None:
        raise EspnSchemaError("neither mRoster's status nor mSettings says which scoring period is current")
    require_period(view, latest)
    schedule = ctx.reader.pro_schedule().data
    read = LeagueRead(settings=settings, rosters=view, schedule=schedule, latest=latest, target=latest)
    pending = ctx.reader.pending_offers(now=ctx.now)
    return _Reading(read, pending, open_offers(pending, ctx.team_id, ctx.now))


def _approval_failures[P: Payload](ctx: FlowContext[P]) -> list[str]:
    """Trades are approval-only (CLAUDE.md): refuse a proposal made or approved under ``auto``."""
    failures: list[str] = []
    if ctx.proposal.policy != "approve":
        failures.append(
            f"trades are approval-only, but this proposal ran under policy {ctx.proposal.policy!r}; "
            "nothing is sent without a person's approval"
        )
    if ctx.proposal.decided_by == DECIDED_BY_AUTO:
        failures.append("trades are approval-only, but this proposal was approved by auto, not by a person")
    return failures


def _deadline_failure(reading: _Reading, now: datetime) -> str | None:
    """The league's trade deadline (``tradeSettings.deadlineDate``, a setting, never a constant) has passed."""
    trade = reading.read.settings.trade
    if trade.is_open(now) or trade.deadline is None:
        return None
    return f"the league's trade deadline {stamp(trade.deadline)} has passed"


def _pieces[P: Payload](
    ctx: FlowContext[P], reading: _Reading, team_id: int, ids: tuple[int, ...], whose: str
) -> tuple[list[_Piece], list[str]]:
    """The players ``ids`` as team ``team_id``'s roster shows them, and a failure for each who is not on it."""
    read = reading.read
    roster = read.our_roster(team_id)
    found: list[_Piece] = []
    failures: list[str] = []
    if roster is None:
        return found, [f"ESPN's rosters for scoring period {read.target} have no team {team_id} ({whose})"]
    for espn_id in ids:
        try:
            entry = roster.entry(espn_id)
        except KeyError:
            failures.append(f"player {espn_id} is not on team {team_id} ({whose}) in scoring period {read.target}")
            continue
        pool = entry.player_pool_entry
        found.append(
            _Piece(
                espn_id=espn_id,
                name=pool.player.full_name,
                team_id=team_id,
                slot=entry.lineup_slot_id,
                trade_locked=pool.trade_locked,
                roster_locked=pool.roster_locked,
                cutoff=cutoff_for(read, ctx.game, pool.player.pro_team_id),
            )
        )
    return found, failures


def _lock_failures(pieces: list[_Piece], reading: _Reading, now: datetime) -> list[str]:
    """Every player in the deal, both sides, who ESPN or the league's roster lock closes to trading now."""
    read = reading.read
    failures: list[str] = []
    if pieces and read.unmapped_lock is not None:
        failures.append(read.unmapped_lock)
    for piece in pieces:
        if piece.trade_locked:
            failures.append(f"ESPN marks {piece.label} trade-locked")
        if piece.cutoff is not None and now >= piece.cutoff:
            failures.append(f"trading {piece.label} closed at {stamp(piece.cutoff)}: {read.lock_rule}")
        elif piece.roster_locked:
            failures.append(f"ESPN marks {piece.label} roster-locked: his game has started")
    return failures


def _room(reading: _Reading, team_id: int, leaving: list[_Piece], arriving: int) -> tuple[int, int, int]:
    """``(players now, players after, the league's size)``, IR not counted (it is extra, as in ``add_drop``)."""
    read = reading.read
    roster = read.our_roster(team_id)
    held = 0 if roster is None else read.roster_count(roster)
    ir = read.ir_slots()
    out = sum(1 for piece in leaving if piece.slot not in ir)
    return held, held - out + arriving, read.settings.roster_size


def _overflow(held: int, after: int, size: int) -> bool:
    """The deal puts the roster over the league's size (a roster already over it is not the deal's doing)."""
    return after > max(size, held)


def _json(reading: _Reading, payload_facts: Mapping[str, Any]) -> dict[str, Any]:
    deadline = reading.read.settings.trade.deadline
    return {
        "scoring_period_id": reading.latest,
        "latest_scoring_period": reading.latest,
        "roster_lock_type": reading.read.settings.roster_lock_type.value,
        "trade_deadline": None if deadline is None else deadline.isoformat(),
        "open_offers": [view.describe() for view in reading.views],
        **payload_facts,
    }


def _same_deal(view: OfferView, payload: TradePayload) -> bool:
    return (
        view.other_team_id == payload.other_team_id
        and set(view.give) == set(payload.give_espn_ids)
        and set(view.get) == set(payload.get_espn_ids)
    )


def _record_of[P: Payload](ctx: FlowContext[P], offer_id: str) -> Transaction | None:
    """The offer's own record in ``mTransactions2`` whatever its status (an executed or closed offer is not open)."""
    for record in ctx.reader.transactions(None, types=OFFER_TYPES).data.transactions:
        if record.id == offer_id and not record.is_cancellation:
            return record
    return None


# --- the proposal -----------------------------------------------------------------------------------------------------


class ProposeTrade(Flow[TradePayload]):
    """An offer to another team (``trade_propose``) in both sports. API mode; the builder walk waits for the drill
    (the module docs say why)."""

    name = "propose_trade"
    kinds = (ProposalKind.TRADE_PROPOSE,)
    payload_type = TradePayload
    modes = (Mode.API,)

    # --- preconditions ------------------------------------------------------------------------------------------------

    def check(self, ctx: FlowContext[TradePayload]) -> Preconditions:
        payload = ctx.payload
        failures = _approval_failures(ctx)
        reading = _read(ctx)
        other = payload.other_team_id
        if other == ctx.team_id:
            failures.append(f"team {other} is our own team: a trade is with another team")
        deadline = _deadline_failure(reading, ctx.now)
        if deadline is not None:
            failures.append(deadline)
        gives, missing_ours = _pieces(ctx, reading, ctx.team_id, payload.give_espn_ids, "ours")
        gets, missing_theirs = _pieces(ctx, reading, other, payload.get_espn_ids, "theirs")
        failures.extend((*missing_ours, *missing_theirs))
        failures.extend(_lock_failures([*gives, *gets], reading, ctx.now))
        failures.extend(duplicate_blockers(reading.views, other, payload.give_espn_ids, payload.get_espn_ids))
        held, after, size = _room(reading, ctx.team_id, gives, len(payload.get_espn_ids))
        notes: list[str] = []
        if _overflow(held, after, size):
            failures.append(
                f"the trade would leave us {after} players and the league allows {size}: a trade proposal here "
                "carries no drop, so make room first or change the deal"
            )
        held_theirs, after_theirs, _ = _room(reading, other, gets, len(payload.give_espn_ids))
        if _overflow(held_theirs, after_theirs, size):
            notes.append(f"team {other} would hold {after_theirs} of {size} players and must drop one to accept")
        facts = {
            "other_team_id": other,
            "give": [piece.to_json() for piece in gives],
            "get": [piece.to_json() for piece in gets],
            "roster_count": held,
            "roster_after": after,
            "roster_size": size,
            "notes": notes,
        }
        return Preconditions(tuple(failures), _json(reading, facts))

    # --- API mode -----------------------------------------------------------------------------------------------------

    def build_request(self, ctx: FlowContext[TradePayload], pre: Preconditions) -> WriteRequest:
        payload = ctx.payload
        items = (
            *(trade_item(espn_id, ctx.team_id, payload.other_team_id) for espn_id in payload.give_espn_ids),
            *(trade_item(espn_id, payload.other_team_id, ctx.team_id) for espn_id in payload.get_espn_ids),
        )
        envelope = Envelope(
            team_id=ctx.team_id,
            type=TransactionType.TRADE_PROPOSAL,
            member_id=ctx.member_id,
            scoring_period_id=int(pre.observed["latest_scoring_period"]),
            items=items,
        )
        request = transaction_request(ctx.league, envelope)
        body = dict(request.body)
        body["expirationDate"] = _iso_ms(ctx.now + timedelta(days=EXPIRY_DAYS))
        body["comment"] = ""
        return WriteRequest(url=request.url, body=body, headers=request.headers, method=request.method)

    def ui_may_follow(self, response: WriteResponse) -> bool:
        return transactions_ui_may_follow(response)

    # --- UI mode ------------------------------------------------------------------------------------------------------

    def run_ui(self, ctx: FlowContext[TradePayload], ui: UiDriver, pre: Preconditions) -> None:
        names = [str(piece["name"]) for piece in (*pre.observed.get("give", ()), *pre.observed.get("get", ()))]
        if not names or not all(names):
            raise ModeUnavailableError("the preconditions found no player names to pick in the trade builder")
        again = self.check(ctx)
        if not again.ok:
            raise ModeUnavailableError(
                "the league changed since the preconditions were read: " + "; ".join(again.failures)
            )
        page = ui.page
        league = ctx.league
        url = selectors.trade_builder_url(
            ctx.game, league.espn_league_id, league.season, ctx.payload.other_team_id, ctx.team_id
        )
        open_page(page, url, selectors.TRADE_HEADING, "the trade builder")
        ui.screenshot("trade-builder")
        for name in names:
            box = selectors.trade_player_checkbox(page, name)
            found = box.count()
            if found != 1:
                why = "no checkbox" if not found else f"{found} checkboxes"
                raise ModeUnavailableError(f"the trade builder shows {why} named 'Trade {name}'")
            box.check()
        selectors.TRADE_CONTINUE.locate(page).click()
        send = selectors.TRADE_SEND.locate(page)
        send.wait_for(state="visible")
        expiry = selectors.TRADE_EXPIRY.locate(page)
        if not expiry.count():
            raise ModeUnavailableError("the trade review shows no expiry select")
        expiry.select_option(UI_EXPIRY_VALUE)
        ui.screenshot("trade-review")
        ui.confirm(send, what=f"send the trade proposal to team {ctx.payload.other_team_id}: {', '.join(names)}")
        check_saved(page)

    # --- verification -------------------------------------------------------------------------------------------------

    def verify(self, ctx: FlowContext[TradePayload], pre: Preconditions) -> Verification:
        payload = ctx.payload
        reading = _read(ctx)
        outgoing = [view for view in reading.views if view.direction is Direction.OUTGOING]
        expected = {
            "other_team_id": payload.other_team_id,
            "give": sorted(payload.give_espn_ids),
            "get": sorted(payload.get_espn_ids),
        }
        observed = {"open_offers": [view.describe() for view in reading.views]}
        for view in outgoing:
            if _same_deal(view, payload) and view.expires is not None and view.expires > ctx.now:
                return Verification(True, f"offer {view.offer_id} is open", expected, observed)
        return Verification(
            False, f"no open offer of ours to team {payload.other_team_id} for this deal", expected, observed
        )


# --- accept and decline -----------------------------------------------------------------------------------------------


class RespondTrade(Flow[TradeResponsePayload]):
    """Our answer to an incoming offer: ``trade_accept`` or ``trade_decline``, API mode only (no page for it has been
    observed)."""

    name = "respond_trade"
    kinds = (ProposalKind.TRADE_ACCEPT, ProposalKind.TRADE_DECLINE)
    payload_type = TradeResponsePayload
    modes = (Mode.API,)

    def check(self, ctx: FlowContext[TradeResponsePayload]) -> Preconditions:
        payload = ctx.payload
        accepting = ctx.kind is ProposalKind.TRADE_ACCEPT
        failures = _approval_failures(ctx)
        reading = _read(ctx)
        wanted = payload.espn_transaction_id
        transaction = next((record for record in reading.pending if record.id == wanted), None)
        view = None if transaction is None else view_of(transaction, ctx.team_id)
        facts: dict[str, Any] = {"offer_id": wanted, "accepting": accepting, "other_team_id": payload.other_team_id}
        if transaction is None:
            failures.append(
                f"transaction {wanted} is not an open offer (answered, cancelled or expired already); run fm sync"
            )
        elif view is None:
            failures.append(f"transaction {wanted} is a {transaction.type} that does not involve our team")
        else:
            facts["offer"] = view.describe()
            if view.direction is not Direction.INCOMING:
                failures.append(f"offer {wanted} is ours: withdraw it with a cancel, it cannot be answered")
            if not _same_deal(view, payload):
                failures.append(
                    f"offer {wanted} no longer exchanges what was evaluated ({view.describe()}); run fm sync and "
                    "decide again"
                )
            if accepting and not view.supported:
                failures.append(f"offer {wanted} cannot be accepted by this tool: {'; '.join(view.unsupported)}")
        if accepting:
            failures.extend(self._accept_failures(ctx, reading, facts))
        return Preconditions(tuple(failures), _json(reading, facts))

    @staticmethod
    def _accept_failures(ctx: FlowContext[TradeResponsePayload], reading: _Reading, facts: dict[str, Any]) -> list[str]:
        """What only an acceptance needs: the league's deadline, every player on both sides unlocked and where the offer
        says, and room on our roster (a decline needs none of it)."""
        payload = ctx.payload
        failures: list[str] = []
        deadline = _deadline_failure(reading, ctx.now)
        if deadline is not None:
            failures.append(deadline)
        gives, missing_ours = _pieces(ctx, reading, ctx.team_id, payload.give_espn_ids, "ours")
        gets, missing_theirs = _pieces(ctx, reading, payload.other_team_id, payload.get_espn_ids, "theirs")
        failures.extend((*missing_ours, *missing_theirs))
        failures.extend(_lock_failures([*gives, *gets], reading, ctx.now))
        held, after, size = _room(reading, ctx.team_id, gives, len(payload.get_espn_ids))
        if _overflow(held, after, size):
            failures.append(
                f"accepting would leave us {after} players and the league allows {size}: a trade response here "
                "carries no drop, so make room first or answer it by hand"
            )
        facts.update(
            give=[piece.to_json() for piece in gives],
            get=[piece.to_json() for piece in gets],
            roster_count=held,
            roster_after=after,
            roster_size=size,
        )
        return failures

    def build_request(self, ctx: FlowContext[TradeResponsePayload], pre: Preconditions) -> WriteRequest:
        accepting = ctx.kind is ProposalKind.TRADE_ACCEPT
        envelope = Envelope(
            team_id=ctx.team_id,
            type=TransactionType.TRADE_ACCEPT if accepting else TransactionType.TRADE_DECLINE,
            member_id=ctx.member_id,
            scoring_period_id=int(pre.observed["latest_scoring_period"]),
            comment=None if accepting else "",
            related_transaction_id=ctx.payload.espn_transaction_id,
        )
        return transaction_request(ctx.league, envelope)

    def ui_may_follow(self, response: WriteResponse) -> bool:
        return False  # there is no UI mode to follow with

    def verify(self, ctx: FlowContext[TradeResponsePayload], pre: Preconditions) -> Verification:
        payload = ctx.payload
        accepting = ctx.kind is ProposalKind.TRADE_ACCEPT
        wanted = payload.espn_transaction_id
        reading = _read(ctx)
        still_open = any(record.id == wanted for record in reading.pending)
        record = _record_of(ctx, wanted)
        status = None if record is None else record.status
        expected = {"offer_id": wanted, "answer": "accept" if accepting else "decline"}
        observed: dict[str, Any] = {"open": still_open, "status": status}
        if accepting:
            ours = reading.read.our_roster(ctx.team_id)
            held = set() if ours is None else set(ours.player_ids)
            moved = set(payload.get_espn_ids) <= held and not set(payload.give_espn_ids) & held
            observed["players_moved"] = moved
            if moved or (record is not None and record.executed):
                return Verification(True, f"the trade of offer {wanted} went through", expected, observed)
            detail = (
                f"offer {wanted} is still open: ESPN may be holding the accepted trade (league review)"
                if still_open
                else f"offer {wanted} is closed (status {status}) but the rosters do not show the trade"
            )
            return Verification(False, detail, expected, observed)
        if still_open:
            return Verification(False, f"offer {wanted} is still open", expected, observed)
        if record is not None and record.executed:
            return Verification(False, f"offer {wanted} was executed, not declined", expected, observed)
        return Verification(True, f"offer {wanted} is no longer open (status {status})", expected, observed)


# --- the cancel -------------------------------------------------------------------------------------------------------


class CancelTrade(Flow[CancelPayload]):
    """Withdraw our open offer (``trade_cancel``) in both sports; API mode only, as documented."""

    name = "cancel_trade"
    kinds = (ProposalKind.TRADE_CANCEL,)
    payload_type = CancelPayload
    modes = (Mode.API,)

    def check(self, ctx: FlowContext[CancelPayload]) -> Preconditions:
        wanted = ctx.payload.espn_transaction_id
        failures = _approval_failures(ctx)
        reading = _read(ctx)
        transaction = next((record for record in reading.pending if record.id == wanted), None)
        view = None if transaction is None else view_of(transaction, ctx.team_id)
        if transaction is None:
            failures.append(
                f"transaction {wanted} is not an open offer (answered, cancelled or expired already); run fm sync"
            )
        elif transaction.type != "TRADE_PROPOSAL":
            failures.append(f"transaction {wanted} is a {transaction.type}, not a trade offer")
        elif transaction.team_id != ctx.team_id or (view is not None and view.direction is not Direction.OUTGOING):
            failures.append(f"offer {wanted} is team {transaction.team_id}'s, not team {ctx.team_id}'s (ours)")
        facts = {"offer_id": wanted, "offer": None if view is None else view.describe()}
        return Preconditions(tuple(failures), _json(reading, facts))

    def build_request(self, ctx: FlowContext[CancelPayload], pre: Preconditions) -> WriteRequest:
        envelope = Envelope(
            team_id=ctx.team_id,
            type=TransactionType.TRADE_PROPOSAL,
            member_id=ctx.member_id,
            scoring_period_id=int(pre.observed["latest_scoring_period"]),
            execution_type=ExecutionType.CANCEL,
            related_transaction_id=ctx.payload.espn_transaction_id,
        )
        return transaction_request(ctx.league, envelope)

    def ui_may_follow(self, response: WriteResponse) -> bool:
        return False

    def verify(self, ctx: FlowContext[CancelPayload], pre: Preconditions) -> Verification:
        wanted = ctx.payload.espn_transaction_id
        reading = _read(ctx)
        still_open = any(record.id == wanted for record in reading.pending)
        record = _record_of(ctx, wanted)
        expected = {"open": False, "offer_id": wanted}
        observed = {"open": still_open, "status": None if record is None else record.status}
        if still_open:
            return Verification(False, f"offer {wanted} is still open", expected, observed)
        if record is not None and record.executed:
            return Verification(False, f"offer {wanted} was accepted before it was cancelled", expected, observed)
        return Verification(True, f"offer {wanted} is no longer open", expected, observed)


def _iso_ms(at: datetime) -> str:
    """``2026-10-08T18:10:21.078Z``: how the web client writes ``expirationDate`` on the wire."""
    return at.astimezone(UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")


PROPOSE_TRADE = register_flow(ProposeTrade())
"""The process-wide ``propose_trade`` flow."""
RESPOND_TRADE = register_flow(RespondTrade())
"""The process-wide ``respond_trade`` flow (``trade_accept`` and ``trade_decline``)."""
CANCEL_TRADE = register_flow(CancelTrade())
"""The process-wide ``cancel_trade`` flow."""
