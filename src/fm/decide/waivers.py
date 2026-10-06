"""NFL waivers and free agents (DESIGN section 9.2, ROADMAP #21): rank (add, drop) pairs by the change in our roster's
rest-of-season value and propose the best as waiver claims or free-agent adds.

**The wire** (:class:`Wire`) is the free-agent pool as the league's latest ``fm sync`` read it: the ``kona_player_info``
pages the sync captured and indexed in ``raw_snapshots`` (:func:`load_wire`), less everyone on a roster in the latest
roster snapshot. Each pool entry says whether the player is a free agent (``FREEAGENT``: an immediate add, an
:class:`fm.proposals.AddDropPayload`) or on waivers (``WAIVERS``: a claim, a :class:`fm.proposals.WaiverPayload`) and,
on waivers, when a claim on him is processed: ``waiverProcessDate``, the league's waiver run for him as ESPN schedules
it from the league's settings (``waiverProcessHour`` alone does not give the run's time; docs/espn-api.md section 1 #8).

**Ranking** (:func:`rank_moves`). Each candidate add (the wire's best players by rest-of-season value whose pool status
is known and whose kind the league's policy does not turn off) is paired with every droppable player, and with no drop
when the roster has an open spot. A pair scores :meth:`fm.model.valuation.RosterValuer.gain`: the change in the
roster's rest-of-season start value (playoff weeks weighted up, holes in later weeks filled at replacement level, the
best player actually on the wire), with the drop gone from the period the move executes in and the add counting from
the first period he can play for us. Equal gains prefer keeping everyone, then dropping the least valuable player. A
pair is left out when it would break the league's position limits, or when its drop is locked when the move executes
(his game this period has started; a claim processed in a later period is not affected).

Never dropped: an untouchable (:func:`fm.proposals.find_untouchables`; :func:`fm.proposals.evaluate` refuses one again
at propose time), a player in the IR slot (dropping him frees no roster spot), and a player nothing projects (his value
is unknown, not zero). The league's open add/drop and waiver proposals count as moves already made: the valuation
applies them from the periods they count in, and new moves leave their players alone, so two proposals never claim or
drop the same player; the same move proposed again is deduplicated by :func:`fm.proposals.propose`.

**Timing** is read from the league, never assumed. A claim's deadline is its waiver run (``waiverProcessDate``), and it
counts from the period that run falls in (the next one when the add's game that period starts before the run). A
free-agent add executes and counts in this period; its deadline is the earliest of the add's and the drop's roster
lock this period (``SportPlugin.transaction_cutoff`` under the league's ``roster_lock_type``) and the period's end. A
free agent whose game this period has started (``lineupLocked`` in the pool, or his kickoff passed since the sync) is
not proposed: the roster lock closes his add until the period turns, by when the real league has him on waivers, so
the move would hold a transaction slot the executor cannot spend; a fresh sync shows him as a claim or still locked. A
league whose roster lock type the plugins refuse to read (ESPN's weekly types parse as ``UNKNOWN``) gets no move at
all: when its adds and drops close is unknown, and the executor refuses to guess (ROADMAP #27). The timing needs the
pro schedule; without one a claim counts from the next period and an add has no deadline.

**FAAB** (:mod:`fm.decide.faab`). In a league that bids, a claim bids what the league's winning bids say its gain
is worth (:class:`fm.decide.faab.BidModel`, fitted from the ``mTransactions2`` history the sync captured), shrunk
toward the heuristic (:func:`fm.decide.faab.heuristic_bid`: the share of the budget left that its gain is of a quarter
of the roster's remaining value) and falling back to it entirely when the history is too thin; the decision's warnings
and the proposal's engine numbers say which. A bid is at least ESPN's minimum bid and never more than the budget left
or :func:`fm.proposals.faab_bid_cap` (the policy's ``max_faab_pct_per_bid`` of the synced season budget, the cap
:func:`fm.proposals.evaluate` enforces again). The budget left is the season budget less the spend the sync read and
less the bids pledged by the league's unsettled waiver claims (:func:`_pending`: the open proposals, and the claims the
executor submitted that stay pending on ESPN until their waiver run is synced), so successive runs together never
overbid; a plan's later claims bid from what its earlier bids leave. Leagues without FAAB claim by waiver
priority, with no bid (both real leagues).

**Proposing** (:func:`decide_waivers`, registered as ``("nfl", "waivers")`` in :mod:`fm.decide.registry`). The plan
takes the best pair worth at least ``min_gain`` with a transaction slot left in the week it executes in, applies it,
ranks again, and repeats until no pair fits. A move spends one of the policy's ``max_transactions_per_week`` in the
week of its ``scoring_period``: an add this week, a claim the week its waiver run falls in (an NFL league's Wednesday
run is the next week's), and a week's slots are the cap less what the proposals already in it hold, which is how
:func:`fm.proposals.evaluate` counts them too (the week is the matchup period; periods no matchup lists share the
trailing seven days). Each move goes through :func:`fm.proposals.propose`, which runs every guardrail again; a refusal
is reported, not raised. Nothing here writes to ESPN: the executor acts on approved proposals (CLAUDE.md).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Final

from pydantic import ValidationError

from fm import paths
from fm.config import Config, Policy
from fm.decide.faab import (
    DEFAULT_BID_SHARE,
    BidModel,
    bid_strength,
    fit_bid_model,
    heuristic_bid,
    load_bid_history,
    modeled_bid,
)
from fm.decide.registry import register
from fm.espn.client import View
from fm.espn.models import POOL_FREE_AGENT, POOL_WAIVERS, PlayersView, PoolEntry, Transaction
from fm.espn.settings import LeagueSettings, LockType
from fm.model.projections import ESPN, BlendWeights
from fm.model.valuation import (
    DEFAULT_PLAYOFF_WEIGHT,
    LeagueValuation,
    PlayerOutlook,
    Replacement,
    RosterValuer,
    league_settings,
    load_valuation,
)
from fm.proposals import (
    ACQUISITION_KINDS,
    AddDropPayload,
    PolicyError,
    ProposalKind,
    WaiverPayload,
    acquisitions_this_week,
    effective_setting,
    expire_due,
    faab_bid_cap,
    find_untouchables,
    parse_payload,
    propose,
)
from fm.proposals.policy import ROLLING_WEEK, as_utc
from fm.sports.base import ScheduleLike, fantasy_day, game_for, last_start, period_turn, plugin_for
from fm.store import OPEN_PROPOSAL_STATUSES, LeagueRow, ProposalRow, RosterEntryRow, Store

WAIVERS_KIND: Final = "waivers"
"""The decision kind this module registers for ``nfl``."""
WAIVERS_CREATED_BY: Final = "decide.waivers"
"""``proposals.created_by`` of the proposals it stores."""
DEFAULT_MIN_GAIN: Final = 5.0
"""Rest-of-season points (weighted) a move must add to be proposed: about half a point a week over the rest of a
season, below which a move is projection noise that still costs a transaction (and, on waivers, claim priority)."""
DEFAULT_CANDIDATES: Final = 25
"""Wire players, best rest-of-season value first, considered as adds."""
POOL_KIND: Final = View.PLAYER_INFO.value
"""``raw_snapshots.kind`` of the free-agent pool pages the sync captures."""


class WaiverError(ValueError):
    """The waiver decision cannot run for a league: unknown to the store, not in ``config.toml``, or not NFL."""


# --- the wire ---------------------------------------------------------------------------------------------------------


class WireStatus(StrEnum):
    """How an unrostered player is acquired: added at once, or claimed at the waiver run."""

    FREE_AGENT = POOL_FREE_AGENT
    WAIVERS = POOL_WAIVERS


@dataclass(frozen=True, slots=True)
class WireEntry:
    """One player on the wire: his status, when a claim on him is processed (``waiverProcessDate``, on waivers only)
    and whether his game this period has started (``lineupLocked``)."""

    espn_id: int
    status: WireStatus
    clears_at: datetime | None = None
    locked: bool = False


@dataclass(frozen=True, slots=True)
class Wire:
    """The league's wire: who is available and how, as of ``as_of`` (when the pool was read)."""

    entries: Mapping[int, WireEntry]
    as_of: datetime | None = None
    warnings: tuple[str, ...] = ()

    @classmethod
    def from_pool(
        cls, players: Iterable[PoolEntry], *, rostered: Iterable[int] = (), as_of: datetime | None = None
    ) -> Wire:
        """The wire from ``kona_player_info`` pool entries, less ``rostered`` players and any entry ESPN puts on a
        team. An entry neither ``FREEAGENT`` nor ``WAIVERS`` is left off with a warning."""
        taken = frozenset(rostered)
        entries: dict[int, WireEntry] = {}
        other = 0
        for entry in players:
            if entry.id in taken or entry.rostered_team_id is not None:
                continue
            if entry.status == POOL_WAIVERS:
                status, clears = WireStatus.WAIVERS, entry.waiver_process_date
            elif entry.status == POOL_FREE_AGENT:
                status, clears = WireStatus.FREE_AGENT, None
            else:
                other += 1
                continue
            entries[entry.id] = WireEntry(entry.id, status, clears, entry.lineup_locked)
        warnings = (
            (f"{other} pool players were neither free agents nor on waivers; left off the wire",) if other else ()
        )
        return cls(MappingProxyType(entries), as_of, warnings)

    @property
    def ids(self) -> frozenset[int]:
        return frozenset(self.entries)

    def get(self, espn_id: int) -> WireEntry | None:
        return self.entries.get(espn_id)

    def __contains__(self, espn_id: object) -> bool:
        return espn_id in self.entries

    def __len__(self) -> int:
        return len(self.entries)


def _sync_started(store: Store, league: LeagueRow) -> datetime | None:
    """When the league's latest sync began: its settings capture, the first read every sync makes."""
    row = store.settings.get(league.row_id)
    if row is None or row.raw_snapshot_id is None:
        return None
    snapshot = store.raw_snapshots.get(row.raw_snapshot_id)
    return None if snapshot is None else snapshot.fetched_at


def load_wire(store: Store, league: LeagueRow, *, scoring_period: int | None = None) -> Wire:
    """The wire as the league's latest ``fm sync`` read it, from the pool pages it captured.

    The pages are the ``kona_player_info`` captures indexed in ``raw_snapshots`` for the league and period (default:
    the latest roster snapshot's) since that sync's settings read, read back from under ``fm.paths.cache_dir()``; a
    player on two pages takes the later one. Players rostered in the period's snapshot are left out. With no page, or a
    page gone from the cache (it is deletable), the wire is empty or partial, with a warning naming the fix.
    """
    period = scoring_period if scoring_period is not None else store.rosters.latest_period(league.row_id)
    if period is None:
        return Wire(MappingProxyType({}), None, (f"{league.key}: no roster snapshot; run fm sync",))
    pages = store.raw_snapshots.find(ESPN, POOL_KIND, league_id=league.row_id, scoring_period_id=period)
    started = _sync_started(store, league)
    if started is not None:
        pages = [page for page in pages if page.fetched_at >= started]
    if not pages:
        missing = f"{league.key}: no free-agent pool captured for scoring period {period}; run fm sync"
        return Wire(MappingProxyType({}), None, (missing,))
    root = paths.cache_dir()
    entries: dict[int, PoolEntry] = {}
    warnings: list[str] = []
    for page in pages:
        try:
            view = PlayersView.model_validate_json((root / page.path).read_bytes())
        except FileNotFoundError:
            warnings.append(f"{league.key}: pool page {page.path} is gone from the cache; run fm sync")
            continue
        except (OSError, ValidationError) as exc:
            warnings.append(f"{league.key}: pool page {page.path} could not be read ({type(exc).__name__})")
            continue
        entries.update((entry.id, entry) for entry in view.players)
    rostered = store.rosters.rostered_ids(league.row_id, period)
    wire = Wire.from_pool(entries.values(), rostered=rostered, as_of=max(page.fetched_at for page in pages))
    return replace(wire, warnings=(*warnings, *wire.warnings))


# --- FAAB -------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Bidding:
    """What a league's FAAB bids are drawn from: the budget left, the policy's per-bid cap and ESPN's minimum bid.

    ``pledged`` is what the budget left already excludes because open proposals hold it (for the proposal's numbers);
    ``model`` is the league's fitted winning-bid model, or the reason there is none (:attr:`modeled`): a claim then
    bids the heuristic's amount.
    """

    budget_left: int
    cap: int
    minimum_bid: int = 0
    share: float = DEFAULT_BID_SHARE
    pledged: int = 0
    model: BidModel | None = None

    @classmethod
    def for_league(
        cls,
        settings: LeagueSettings,
        policy: Policy,
        *,
        spent: int,
        pledged: int = 0,
        model: BidModel | None = None,
        share: float = DEFAULT_BID_SHARE,
    ) -> Bidding | None:
        """The league's bidding with ``spent`` dollars of its season budget gone and ``pledged`` held by open claims;
        ``None`` when it does not bid. The cap is :func:`fm.proposals.faab_bid_cap`, the one
        :func:`fm.proposals.evaluate` enforces."""
        cap = faab_bid_cap(policy, settings)
        budget = settings.acquisition.budget
        if cap is None or budget is None:
            return None
        return cls(max(0, budget - spent - pledged), cap, settings.acquisition.minimum_bid, share, pledged, model)

    @property
    def possible(self) -> bool:
        """False when the cap or the budget left is below ESPN's minimum bid, so no claim can be made."""
        return min(self.cap, self.budget_left) >= self.minimum_bid

    @property
    def modeled(self) -> bool:
        """True when bids come from the fitted model rather than the heuristic alone."""
        return self.model is not None and self.model.fitted

    def bid(self, gain: float, roster_value: float) -> int | None:
        """The bid for a claim worth ``gain`` to a roster worth ``roster_value``: the model's, shrunk toward the
        heuristic (:func:`fm.decide.faab.modeled_bid`), or the heuristic's alone when no model is fitted."""
        if self.model is not None and self.model.fitted:
            if not (math.isfinite(gain) and math.isfinite(roster_value)):
                raise ValueError(f"gain and roster_value must be finite, got {gain!r} and {roster_value!r}")
            return modeled_bid(
                self.model,
                bid_strength(gain, roster_value, self.share),
                budget_left=self.budget_left,
                cap=self.cap,
                minimum_bid=self.minimum_bid,
            )
        return heuristic_bid(
            gain,
            roster_value=roster_value,
            budget_left=self.budget_left,
            cap=self.cap,
            minimum_bid=self.minimum_bid,
            share=self.share,
        )

    def numbers(self) -> dict[str, Any]:
        """The bidding's side of a proposal's engine numbers: the source of the bid and what it drew on."""
        numbers: dict[str, Any] = {
            "cap": self.cap,
            "budget_left": self.budget_left,
            "pledged": self.pledged,
            "minimum_bid": self.minimum_bid,
            "share": self.share,
            "source": "model" if self.modeled else "heuristic",
        }
        if self.model is not None:
            numbers["history"] = self.model.numbers()
        return numbers


# --- moves ------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class WaiverMove:
    """One ranked (add, drop) pair and the proposal it becomes.

    ``scoring_period`` is the period the move executes in (the drop is gone from then on, and the weekly transaction
    cap counts it there) and ``start`` the first period the add counts in: the same, or for a claim the next one when
    his game in the period of the run is over by then. ``gain`` is the change in rest-of-season value (weighted
    points) over both. ``deadline`` is
    when the proposal stops making sense: the waiver run for a claim, the earliest roster lock or the period's end for
    an add. The values and VORs are each player's own (the add's from ``start``, the drop's from ``scoring_period``);
    ``roster_value`` (the roster's rest-of-season value from ``start``) and ``bidding`` are what a FAAB bid was drawn
    from.
    """

    kind: ProposalKind
    add: PlayerOutlook
    drop: PlayerOutlook | None
    gain: float
    start: int
    scoring_period: int
    payload: WaiverPayload | AddDropPayload
    deadline: datetime | None = None
    clears_at: datetime | None = None
    add_value: float = 0.0
    drop_value: float | None = None
    add_vor: float | None = None
    drop_vor: float | None = None
    roster_value: float | None = None
    bidding: Bidding | None = None

    @property
    def bid(self) -> int | None:
        return self.payload.bid

    @property
    def players(self) -> tuple[int, ...]:
        """The ESPN ids the move touches: the add, then the drop."""
        return (self.add.espn_id,) if self.drop is None else (self.add.espn_id, self.drop.espn_id)

    def dedupe_key(self, league: LeagueRow) -> str:
        """One proposal per move per league and season: a re-run finds it instead of storing it twice (the bid may
        have changed since; the move has not)."""
        drop = self.drop.espn_id if self.drop is not None else 0
        return f"{WAIVERS_KIND}:{league.key}:{league.season}:{self.kind.value}:{self.add.espn_id}:{drop}"

    def engine_numbers(self) -> dict[str, Any]:
        """The numbers behind the move, as the proposal stores them."""
        numbers: dict[str, Any] = {
            "model": "ros_start_value",
            "gain": round(self.gain, 3),
            "start_period": self.start,
            "scoring_period": self.scoring_period,
            "add": _numbers(self.add, self.add_value, self.add_vor),
            "drop": None if self.drop is None else _numbers(self.drop, self.drop_value or 0.0, self.drop_vor),
        }
        if self.clears_at is not None:
            numbers["clears_at"] = self.clears_at.isoformat()
        if self.bidding is not None:
            numbers["bid"] = {
                "amount": self.bid,
                **self.bidding.numbers(),
                "roster_value": None if self.roster_value is None else round(self.roster_value, 3),
            }
        return numbers

    def rationale(self) -> str:
        """One templated line for the proposal (the explain worker, ROADMAP #36, may write a longer one)."""
        claim = self.kind is ProposalKind.WAIVER
        text = f"{'Claim' if claim else 'Add'} {_label(self.add)} {'off waivers' if claim else 'from free agency'}"
        if self.drop is not None:
            text += f", dropping {_label(self.drop)}"
        text += f": +{self.gain:.1f} rest-of-season points from period {self.start}"
        if self.bid is not None and self.bidding is not None:
            text += f"; bid ${self.bid} (cap ${self.bidding.cap}, ${self.bidding.budget_left} left)"
        return text + "."


def _numbers(outlook: PlayerOutlook, value: float, vor: float | None) -> dict[str, Any]:
    return {
        "espn_id": outlook.espn_id,
        "name": outlook.name,
        "position": outlook.position,
        "ros": round(value, 3),
        "vor": None if vor is None else round(vor, 3),
        "per_game": round(outlook.per_game, 3),
        "games": outlook.games,
        "health": round(outlook.health, 3),
        "p_active": outlook.p_active,
        "basis": outlook.basis,
    }


def _label(outlook: PlayerOutlook) -> str:
    return f"{outlook.name} ({outlook.position})" if outlook.position else outlook.name


# --- timing -----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Timing:
    """When a move on one wire player happens: the period it counts from, the period it executes in, the instant it
    takes effect, its deadline, and the waiver run for a claim."""

    kind: ProposalKind
    start: int
    executes_in: int
    at: datetime
    deadline: datetime | None
    clears_at: datetime | None = None


class _Clock:
    """Kickoffs, roster locks and period ends from the pro schedule and the league's roster lock type."""

    def __init__(self, settings: LeagueSettings, schedule: ScheduleLike | None, current: int) -> None:
        self.settings = settings
        self.schedule = schedule
        self.current = current
        self.plugin = plugin_for(settings.game)
        # Why no move can be timed: the roster lock type is one the plugins (and the executor) refuse to read.
        self.unmapped: str | None = None
        if settings.roster_lock_type is LockType.UNKNOWN:
            raw = settings.roster_lock_type_raw or LockType.UNKNOWN.value
            self.unmapped = (
                f"the league's roster lock type {raw} is not mapped, so when adds and drops close is unknown"
            )

    def period_at(self, at: datetime) -> int | None:
        return None if self.schedule is None else self.plugin.scoring_period_at(at, self.schedule)

    def kickoff(self, team_id: int | None, period: int) -> datetime | None:
        if self.schedule is None or team_id is None:
            return None
        game = game_for(self.schedule, team_id, period)
        return None if game is None else game.date

    def cutoff(self, team_id: int | None, period: int) -> datetime | None:
        """When moves of the team's players close in the period under the league's roster lock; ``None`` without a
        schedule or a team. Never asked under an unmapped lock type (:attr:`unmapped` turns every candidate away
        first), and would raise rather than guess."""
        if self.schedule is None or team_id is None:
            return None
        return self.plugin.transaction_cutoff(team_id, period, self.schedule, lock_type=self.settings.roster_lock_type)

    def period_end(self, period: int) -> datetime | None:
        """When ESPN's calendar leaves the period: 03:00 ET after the fantasy day of its last game."""
        if self.schedule is None:
            return None
        last = last_start(self.schedule, period)
        return None if last is None else period_turn(fantasy_day(last) + timedelta(days=1))


STARTED_REASON: Final = (
    "his game this period has started: locked until the period turns, or on waivers by now; run fm sync"
)
"""Why a free agent whose game this period has begun is not proposed (the module docs)."""


def _timing(entry: WireEntry, add: PlayerOutlook, clock: _Clock, now: datetime) -> _Timing | str:
    """The move's timing, or why the player cannot be acquired now."""
    if clock.unmapped is not None:
        return clock.unmapped
    current = clock.current
    if entry.status is WireStatus.WAIVERS:
        clears = entry.clears_at
        if clears is None:
            return "on waivers with no waiver process date"
        if clears <= now:
            return "the waiver run passed after the last sync; run fm sync"
        lands = max(current, clock.period_at(clears) or current + 1)
        kickoff = clock.kickoff(add.pro_team_id, lands)
        start = lands + 1 if kickoff is not None and kickoff <= clears else lands
        return _Timing(ProposalKind.WAIVER, start, lands, clears, clears, clears)
    cutoff = clock.cutoff(add.pro_team_id, current)
    if entry.locked or (cutoff is not None and cutoff <= now):
        return STARTED_REASON
    ends = (moment for moment in (cutoff, clock.period_end(current)) if moment is not None and moment > now)
    return _Timing(ProposalKind.ADD_DROP, current, current, now, min(ends, default=None))


def _drop_locked(entry: RosterEntryRow, drop: PlayerOutlook, timing: _Timing, clock: _Clock) -> bool:
    """True when the drop cannot happen when the move executes: in this period, once his game has started."""
    if timing.executes_in > clock.current:
        return False
    if entry.lineup_locked:
        return True
    cutoff = clock.cutoff(drop.pro_team_id, clock.current)
    return cutoff is not None and cutoff <= timing.at


def _deadline(timing: _Timing, drop: PlayerOutlook | None, clock: _Clock, now: datetime) -> datetime | None:
    """A claim's deadline is its waiver run; an add's is also held to the drop's roster lock this period."""
    if timing.kind is ProposalKind.WAIVER or drop is None:
        return timing.deadline
    cutoff = clock.cutoff(drop.pro_team_id, clock.current)
    return min((moment for moment in (timing.deadline, cutoff) if moment is not None and moment > now), default=None)


# --- ranking and planning ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class _Board:
    """Everything ranking reads besides the valuer, fixed for one decision."""

    valuation: LeagueValuation
    wire: Wire
    clock: _Clock
    now: datetime
    entries: Mapping[int, RosterEntryRow]
    protected: frozenset[int]
    kinds: frozenset[ProposalKind]
    bidding: Bidding | None
    candidates: int


def _board(
    valuation: LeagueValuation,
    wire: Wire,
    *,
    now: datetime,
    schedule: ScheduleLike | None,
    protected: Iterable[int],
    kinds: Iterable[ProposalKind],
    bidding: Bidding | None,
    candidates: int,
) -> _Board:
    if candidates < 0:
        raise ValueError(f"candidates must be >= 0, got {candidates!r}")
    return _Board(
        valuation=valuation,
        wire=wire,
        clock=_Clock(valuation.settings, schedule, valuation.scoring_period),
        now=as_utc(now),
        entries=MappingProxyType({entry.espn_id: entry for entry in valuation.team}),
        protected=frozenset(protected),
        kinds=frozenset(kinds),
        bidding=bidding,
        candidates=candidates,
    )


def _droppable(board: _Board, valuer: RosterValuer) -> list[int]:
    """Our players a move may drop: not untouchable, outside the IR slot, and projected by something."""
    ir_slot = board.valuation.settings.ids.ir_slot
    return sorted(
        espn_id
        for espn_id in valuer.roster
        if espn_id in board.entries
        and espn_id not in board.protected
        and board.entries[espn_id].lineup_slot_id != ir_slot
        and valuer.outlooks[espn_id].has_projection
    )


def _open_spots(board: _Board, valuer: RosterValuer) -> int:
    """Roster spots free outside IR: the league's active and bench slots less our players outside the IR slot."""
    ir_slot = board.valuation.settings.ids.ir_slot
    held = sum(
        1
        for espn_id in valuer.roster
        if espn_id not in board.entries or board.entries[espn_id].lineup_slot_id != ir_slot
    )
    return board.valuation.settings.roster_size - held


def _within_limits(board: _Board, valuer: RosterValuer, add: int, drop: int | None) -> bool:
    """The league's position limit for the add's position still holds after the move."""
    players = board.valuation.players
    position = players[add].default_position_id if add in players else None
    limit = board.valuation.settings.position_limit(position) if position is not None else None
    if limit is None:
        return True
    after = (valuer.roster - {drop}) | {add}
    return (
        sum(1 for espn_id in after if espn_id in players and players[espn_id].default_position_id == position) <= limit
    )


def _candidates(
    board: _Board, valuer: RosterValuer, exclude: frozenset[int], skipped: Counter[str]
) -> list[tuple[int, _Timing]]:
    """The wire's best players by rest-of-season value that a move could bring in now, with their timing."""
    ranked = sorted(
        (espn_id for espn_id in valuer.wire if espn_id not in exclude and valuer.ros(espn_id) > 0),
        key=lambda espn_id: (-valuer.ros(espn_id), espn_id),
    )
    chosen: list[tuple[int, _Timing]] = []
    for espn_id in ranked:
        if len(chosen) >= board.candidates:
            break
        entry = board.wire.get(espn_id)
        if entry is None:
            skipped["not in the free-agent pool the last sync read"] += 1
            continue
        timing = _timing(entry, valuer.outlooks[espn_id], board.clock, board.now)
        if isinstance(timing, str):
            skipped[timing] += 1
        elif timing.kind not in board.kinds:
            skipped[f"{timing.kind.value} is off in the league's policy"] += 1
        elif timing.kind is ProposalKind.WAIVER and board.bidding is not None and not board.bidding.possible:
            skipped["no FAAB bid fits the cap and the budget left"] += 1
        else:
            chosen.append((espn_id, timing))
    return chosen


def _rank(board: _Board, valuer: RosterValuer, exclude: frozenset[int]) -> tuple[list[WaiverMove], Counter[str]]:
    """Every allowed (add, drop) pair for the valuer's roster, best first, and why candidates were skipped."""
    skipped: Counter[str] = Counter()
    drops: list[int | None] = [espn_id for espn_id in _droppable(board, valuer) if espn_id not in exclude]
    if _open_spots(board, valuer) > 0:
        drops.append(None)
    moves: list[WaiverMove] = []
    for add, timing in _candidates(board, valuer, exclude, skipped):
        for drop in drops:
            outlook = valuer.outlooks[drop] if drop is not None else None
            if outlook is not None and _drop_locked(board.entries[outlook.espn_id], outlook, timing, board.clock):
                continue
            if not _within_limits(board, valuer, add, drop):
                continue
            gain = valuer.gain(add, drop, start=timing.start, drop_start=timing.executes_in)
            moves.append(_move(board, valuer, add, outlook, timing, gain))
    moves.sort(key=_order)
    return moves, skipped


def _order(move: WaiverMove) -> tuple[float, int, bool, float, int]:
    """Best gain first; among equal gains (to a millionth of a point), keeping everyone when a spot is open, then
    dropping the least valuable player, then ESPN ids."""
    drop = move.drop
    return (
        round(-move.gain, 6),
        move.add.espn_id,
        drop is not None,
        move.drop_value if move.drop_value is not None else 0.0,
        drop.espn_id if drop is not None else 0,
    )


def _move(
    board: _Board, valuer: RosterValuer, add: int, drop: PlayerOutlook | None, timing: _Timing, gain: float
) -> WaiverMove:
    drop_id = None if drop is None else drop.espn_id
    payload: WaiverPayload | AddDropPayload
    bidding = board.bidding if timing.kind is ProposalKind.WAIVER else None
    roster_value = valuer.value(start=timing.start) if bidding is not None else None
    if timing.kind is ProposalKind.WAIVER:
        bid = bidding.bid(gain, roster_value or 0.0) if bidding is not None else None
        payload = WaiverPayload(add_espn_id=add, drop_espn_id=drop_id, bid_amount=bid)
    else:
        payload = AddDropPayload(add_espn_id=add, drop_espn_id=drop_id)
    return WaiverMove(
        kind=timing.kind,
        add=valuer.outlooks[add],
        drop=drop,
        gain=gain,
        start=timing.start,
        scoring_period=timing.executes_in,
        payload=payload,
        deadline=_deadline(timing, drop, board.clock, board.now),
        clears_at=timing.clears_at,
        add_value=valuer.ros(add, start=timing.start),
        drop_value=None if drop_id is None else valuer.ros(drop_id, start=timing.executes_in),
        add_vor=valuer.vor(add),
        drop_vor=None if drop_id is None else valuer.vor(drop_id),
        roster_value=roster_value,
        bidding=bidding,
    )


class _Budget:
    """The transaction slots left per week for a plan to spend: ``slots_left`` (the policy's weekly cap less what the
    week's proposals already hold) is asked once per week, the first time a move in it comes up, and each move chosen
    spends one. A week is the matchup period the move's scoring period falls in, the one
    :func:`fm.proposals.acquisitions_this_week` counts; periods no matchup lists share one week, as its trailing
    seven days do."""

    def __init__(self, settings: LeagueSettings, slots_left: Callable[[int], int]) -> None:
        self.settings = settings
        self.slots_left = slots_left
        self._left: dict[int | None, int] = {}

    def _week(self, period: int) -> int | None:
        return self.settings.schedule.matchup_period_for(period)

    def left(self, period: int) -> int:
        """Slots left in the week ``period`` falls in."""
        week = self._week(period)
        if week not in self._left:
            self._left[week] = self.slots_left(period)
        return self._left[week]

    def spend(self, period: int) -> None:
        self._left[self._week(period)] = self.left(period) - 1


def _plan(
    board: _Board,
    valuer: RosterValuer,
    *,
    min_gain: float,
    max_moves: int | None,
    exclude: frozenset[int],
    budget: _Budget | None = None,
) -> tuple[list[WaiverMove], list[WaiverMove], Counter[str]]:
    """(the moves chosen, the ranking of the roster as it stands, why candidates were skipped in that ranking).

    ``max_moves`` caps the plan's length (``None``: no cap; the wire runs out) and ``budget`` the moves per week: a
    pair whose week has no slot left is passed over for the next best that fits.
    """
    if not math.isfinite(min_gain):
        raise ValueError(f"min_gain must be finite, got {min_gain!r}")
    if max_moves is not None and max_moves < 0:
        raise ValueError(f"max_moves must be >= 0, got {max_moves!r}")
    used = set(exclude)
    first, skipped = _rank(board, valuer, frozenset(used))
    moves = first
    chosen: list[WaiverMove] = []
    while max_moves is None or len(chosen) < max_moves:
        best = next(
            (
                move
                for move in moves
                if move.gain >= min_gain and (budget is None or budget.left(move.scoring_period) > 0)
            ),
            None,
        )
        if best is None:
            break
        chosen.append(best)
        if budget is not None:
            budget.spend(best.scoring_period)
        used.update(best.players)
        drop = best.drop.espn_id if best.drop is not None else None
        valuer = valuer.after(best.add.espn_id, drop, start=best.start, drop_start=best.scoring_period)
        if best.bid and board.bidding is not None:  # later claims bid from what this one leaves
            left = replace(board.bidding, budget_left=max(0, board.bidding.budget_left - best.bid))
            board = replace(board, bidding=left)
        if max_moves is None or len(chosen) < max_moves:
            moves, _ = _rank(board, valuer, frozenset(used))
    return chosen, first, skipped


def rank_moves(
    valuation: LeagueValuation,
    wire: Wire,
    *,
    now: datetime,
    schedule: ScheduleLike | None = None,
    protected: Iterable[int] = (),
    kinds: Iterable[ProposalKind] = ACQUISITION_KINDS,
    bidding: Bidding | None = None,
    candidates: int = DEFAULT_CANDIDATES,
    exclude: Iterable[int] = (),
) -> tuple[WaiverMove, ...]:
    """Every allowed (add, drop) pair for our roster, best gain first, whatever the gain. Among equal gains a pair
    keeping everyone (with a spot open) comes first, then the one dropping the least valuable player.

    ``protected`` players are never dropped (the untouchables); ``kinds`` limits the moves to claims, adds or both;
    ``bidding`` prices claims in a FAAB league; ``exclude`` keeps players out of every pair; ``schedule`` is the pro
    schedule behind the timing (see the module docs).
    """
    board = _board(
        valuation,
        wire,
        now=now,
        schedule=schedule,
        protected=protected,
        kinds=kinds,
        bidding=bidding,
        candidates=candidates,
    )
    moves, _ = _rank(board, valuation.valuer(), frozenset(exclude))
    return tuple(moves)


def plan_moves(
    valuation: LeagueValuation,
    wire: Wire,
    *,
    now: datetime,
    schedule: ScheduleLike | None = None,
    protected: Iterable[int] = (),
    kinds: Iterable[ProposalKind] = ACQUISITION_KINDS,
    bidding: Bidding | None = None,
    candidates: int = DEFAULT_CANDIDATES,
    min_gain: float = DEFAULT_MIN_GAIN,
    max_moves: int = 1,
    exclude: Iterable[int] = (),
) -> tuple[WaiverMove, ...]:
    """Up to ``max_moves`` moves chosen one at a time: the best pair worth at least ``min_gain``, applied to the roster
    before ranking again, so later moves see the earlier ones and no player is in two. A later claim bids from the
    FAAB budget the earlier bids leave."""
    board = _board(
        valuation,
        wire,
        now=now,
        schedule=schedule,
        protected=protected,
        kinds=kinds,
        bidding=bidding,
        candidates=candidates,
    )
    chosen, _, _ = _plan(board, valuation.valuer(), min_gain=min_gain, max_moves=max_moves, exclude=frozenset(exclude))
    return tuple(chosen)


# --- the decision -----------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class WaiverDecision:
    """One run of the waiver decision for a league: the valuation and wire it read, the replacement levels, every pair
    ranked for the roster as it stands, the moves planned, the proposals stored (or found again, deduplicated), the
    moves policy refused with the reason, and every warning."""

    league: LeagueRow
    valuation: LeagueValuation
    wire: Wire
    replacement: Mapping[int, Replacement]
    ranked: tuple[WaiverMove, ...]
    moves: tuple[WaiverMove, ...]
    proposals: tuple[ProposalRow, ...] = ()
    blocked: tuple[tuple[WaiverMove, str], ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def scoring_period(self) -> int:
        return self.valuation.scoring_period


def _league_row(store: Store, league: str | LeagueRow) -> LeagueRow:
    if isinstance(league, LeagueRow):
        return league
    row = store.leagues.by_key(league)
    if row is None:
        raise WaiverError(f"no league {league!r} in the store; run fm sync")
    return row


type PendingMove = tuple[int | None, int | None, int]
"""``(add, drop, start)``: a move already proposed, changing the roster from period ``start`` on (a move whose drop goes
before its add counts is two of them)."""


def _in_flight(store: Store, league: LeagueRow, now: datetime) -> list[ProposalRow]:
    """The league's acquisition proposals that have not settled on ESPN: the open ones, and the waiver claims the executor
    submitted (``verified``) whose waiver run has not been read back yet.

    A ``verified`` claim stays pending on ESPN until its run, so its bid is still pledged. The run is the proposal's
    ``deadline`` (:func:`decide_waivers` sets a claim's to its waiver run; a claim stored without one is held for
    :data:`fm.proposals.policy.ROLLING_WEEK` from its creation). The claim counts until a sync after the run: the
    team's ``as_of`` (the sync that read the FAAB spend) past the run means the spend and the roster have it.
    """
    kinds = [kind.value for kind in ACQUISITION_KINDS]
    team = store.teams.get(league.row_id, league.team_id)
    synced_at = team.as_of if team is not None else None
    rows: list[ProposalRow] = []
    for row in store.proposals.find(league_id=league.row_id, statuses=(*OPEN_PROPOSAL_STATUSES, "verified"), kinds=kinds):
        if row.status == "verified":
            if row.kind != ProposalKind.WAIVER.value:
                continue
            run = row.deadline if row.deadline is not None else row.created_at + ROLLING_WEEK
            if synced_at is not None and synced_at > run:
                continue
        elif row.status != "executing" and row.deadline is not None and row.deadline <= now:
            continue
        rows.append(row)
    return rows


def _pending(
    store: Store, league: LeagueRow, valuation: LeagueValuation, now: datetime
) -> tuple[tuple[PendingMove, ...], frozenset[int], int]:
    """The moves the league's unsettled add/drop and waiver proposals would make, for the valuer, every player they
    touch, which new moves leave alone, and the FAAB dollars their waiver bids pledge (``payload.bid``), which new bids
    must leave. Unsettled (:func:`_in_flight`) are the open proposals and the claims the executor already submitted
    whose waiver run has not been synced yet: ESPN holds those bids until the run. A proposal past its deadline is not
    pending (it expires unexecuted) and pledges nothing; an add the valuation cannot value (not on the wire it read)
    stays out of the valuer, its players still left alone and its bid still pledged."""
    moves: list[PendingMove] = []
    players: set[int] = set()
    pledged = 0
    for row in _in_flight(store, league, now):
        payload = parse_payload(row)
        if not isinstance(payload, WaiverPayload | AddDropPayload):
            continue
        add, drop = payload.add_espn_id, payload.drop_espn_id
        players.update(espn_id for espn_id in (add, drop) if espn_id is not None)
        if isinstance(payload, WaiverPayload):
            pledged += payload.bid or 0
        if add is not None and add not in valuation.wire:
            continue
        executes = row.scoring_period_id if row.scoring_period_id is not None else valuation.scoring_period
        start = row.engine_numbers.get("start_period")
        start = start if isinstance(start, int) else executes
        dropped = drop if drop in valuation.roster else None
        moves.extend([(add, dropped, start)] if start == executes else [(None, dropped, executes), (add, None, start)])
    return tuple(moves), frozenset(players), pledged


def _bid_model(
    store: Store,
    league: LeagueRow,
    settings: LeagueSettings,
    history: Iterable[Transaction] | None,
    warnings: list[str],
) -> BidModel | None:
    """The league's winning-bid model, or ``None`` when it does not bid. A model that could not be fitted is returned
    all the same (it says why) and a warning names the reason: claims then bid the heuristic."""
    budget = settings.acquisition.budget
    if not settings.acquisition.uses_faab or budget is None:
        return None
    if history is None:
        history, problems = load_bid_history(store, league)
        warnings.extend(problems)
    model = fit_bid_model(history, budget=budget)
    if not model.fitted:
        warnings.append(f"{league.key}: no FAAB bid model ({model.reason}); bids use the heuristic")
    return model


def decide_waivers(
    store: Store,
    config: Config,
    league: str | LeagueRow,
    *,
    now: datetime | None = None,
    schedule: ScheduleLike | None = None,
    wire: Wire | None = None,
    weights: BlendWeights | None = None,
    playoff_weight: float = DEFAULT_PLAYOFF_WEIGHT,
    min_gain: float = DEFAULT_MIN_GAIN,
    max_moves: int | None = None,
    candidates: int = DEFAULT_CANDIDATES,
    store_proposals: bool = True,
    history: Iterable[Transaction] | None = None,
) -> WaiverDecision:
    """Rank the league's (add, drop) pairs and propose the best moves (the module docs give the rules).

    ``league`` is the config key or the store's row: an NFL points league in ``config.toml`` that ``fm sync`` has
    synced. ``wire`` defaults to :func:`load_wire`; ``schedule`` is the season's pro schedule
    (``EspnClient.pro_schedule``), which times the moves and zeroes byes; ``weights`` are the blend weights (default
    ``data/blend_weights.toml``). Each move needs a transaction slot in the week it executes in (the policy's
    ``max_transactions_per_week`` less what that week's proposals hold; the module docs), and ``max_moves`` caps the
    plan's length on top of that. The league's open add/drop and waiver proposals count as moves already made: the
    valuation applies them from the period each counts in, and new moves leave their players alone. With
    ``store_proposals`` false nothing is written (the decision only ranks and plans); otherwise the league's proposals
    past their deadline are expired first, as :func:`fm.proposals.propose` would. Raises :class:`WaiverError` for a
    league it cannot decide for and :class:`fm.model.valuation.ValuationError` for one that is not synced.

    In a league that bids, ``history`` (default: :func:`fm.decide.faab.load_bid_history`, the ``mTransactions2`` pages
    the sync captured) fits the winning-bid model the claims bid from; too little of it leaves the heuristic, with a
    warning saying so. The bids of the league's open waiver proposals and of the claims already submitted whose run is not synced yet come
    off the budget left (the module docs).
    """
    at = as_utc(now)
    row = _league_row(store, league)
    if row.sport != "nfl":
        raise WaiverError(f"league {row.key!r} is {row.sport}; the waiver decision is for NFL leagues")
    try:
        policy = config.league(row.key).policy
    except KeyError as exc:
        raise WaiverError(exc.args[0]) from None
    if store_proposals:
        expire_due(store, now=at, league_id=row.row_id)
    settings = league_settings(store, row)
    loaded = wire if wire is not None else load_wire(store, row)
    valuation = load_valuation(
        store,
        row,
        now=at,
        settings=settings,
        wire=loaded.ids,
        schedule=schedule,
        weights=weights,
        playoff_weight=playoff_weight,
    )
    warnings = [*loaded.warnings, *valuation.warnings]
    protected = find_untouchables(store, row.sport, policy, sorted(entry.espn_id for entry in valuation.team))
    kinds = [kind for kind in ACQUISITION_KINDS if effective_setting(kind, policy) != "off"]
    team = store.teams.get(row.row_id, row.team_id)
    if team is None and settings.acquisition.uses_faab:
        warnings.append(f"{row.key}: team {row.team_id} has no synced FAAB spending; the whole budget is assumed left")
    pending, committed, pledged = _pending(store, row, valuation, at)
    model = _bid_model(store, row, settings, history, warnings)
    bidding = Bidding.for_league(
        settings, policy, spent=team.acquisition_budget_spent if team is not None else 0, pledged=pledged, model=model
    )
    if bidding is not None and pledged:
        warnings.append(f"{row.key}: ${pledged} of the FAAB budget is pledged by open or submitted waiver claims")

    def slots_left(period: int) -> int:
        held = acquisitions_this_week(store, row, settings, scoring_period_id=period, now=at)
        return max(0, policy.max_transactions_per_week - held)

    if committed:
        warnings.append(f"{row.key}: {len(committed)} players in open proposals were left out of new moves")
    board = _board(
        valuation,
        loaded,
        now=at,
        schedule=schedule,
        protected=protected,
        kinds=kinds,
        bidding=bidding,
        candidates=candidates,
    )
    valuer = valuation.valuer(pending=pending)
    moves, ranked, skipped = _plan(
        board,
        valuer,
        min_gain=min_gain,
        max_moves=max_moves,
        exclude=committed,
        budget=_Budget(settings, slots_left),
    )
    warnings.extend(f"{row.key}: {count} wire candidates skipped: {why}" for why, count in sorted(skipped.items()))

    proposals: list[ProposalRow] = []
    blocked: list[tuple[WaiverMove, str]] = []
    if store_proposals:
        for move in moves:
            try:
                stored = propose(
                    store,
                    config,
                    row,
                    move.kind,
                    move.payload,
                    created_by=WAIVERS_CREATED_BY,
                    scoring_period_id=move.scoring_period,
                    engine_numbers=_proposal_numbers(move, valuation, valuer),
                    rationale=move.rationale(),
                    deadline=move.deadline,
                    dedupe_key=move.dedupe_key(row),
                    now=at,
                )
            except PolicyError as exc:
                blocked.append((move, str(exc)))
            else:
                proposals.append(stored)
    return WaiverDecision(
        league=row,
        valuation=valuation,
        wire=loaded,
        replacement=valuer.replacement,
        ranked=tuple(ranked),
        moves=tuple(moves),
        proposals=tuple(proposals),
        blocked=tuple(blocked),
        warnings=tuple(warnings),
    )


def _proposal_numbers(move: WaiverMove, valuation: LeagueValuation, valuer: RosterValuer) -> dict[str, Any]:
    """The move's numbers plus what they were computed against: the horizon and the replacement levels."""
    horizon = valuation.horizon
    return {
        **move.engine_numbers(),
        "horizon": {
            "periods": [horizon.periods[0], horizon.periods[-1]] if horizon.periods else [],
            "playoff_periods": sorted(horizon.playoff_periods),
            "playoff_weight": horizon.playoff_weight,
        },
        "replacement": {
            level.label: {"espn_id": level.espn_id, "name": level.name, "ros": round(level.value, 3)}
            for level in valuer.replacement.values()
        },
        "as_of": valuation.as_of.isoformat(),
    }


register("nfl", WAIVERS_KIND, decide_waivers)
