"""NBA daily lineups across the matchup week (DESIGN sections 8.2, 9.1 and 9.3, ROADMAP #31).

An NBA scoring period is a day, a matchup is a week of them, and the lineup is set each day. :func:`plan_week` plans
every remaining day of the matchup at once. Each player has a per-game value from an empty slot (league points under the
league's scoring items, or in a category league the weighted
:meth:`fm.model.categories.CategoryModel.contribution`) times his chance ``p_active`` of playing, and each day he plays
if his team has a game. The week is one small integer program (``scipy.optimize.milp``, HiGHS) over *player, day, slot*
choices, solved lexicographically like :mod:`fm.decide.lineup`'s assignment:

1. **Never start a zero.** Fill as many slot-days as possible with players who play (his team has a game and
   ``p_active`` is above 0). A starter who will not play is benched whenever a player who plays can take his slot;
   with nobody to put there he stays (:attr:`DayPlan.idle_starters`).
2. **The objective.** The most expected value over the week.
3. **Stability.** Among equally good plans, the fewest moves from the lineup as it stands.

Over the week these are exactly the per-day lineup assignments, except where a **games-played limit** couples the days:
``rosterSettings.lineupSlotStatLimits`` caps the games a slot may be started for (``LeagueSettings.games_played_limit``;
the season total for all of the slot's places, so 246 for three UTIL places). The weekly plan spends what is left of
each cap (``games_used`` says what is spent; ESPN's team views carry no per-slot counter, so the store cannot) on the
best days, and a slot with nothing left holds nobody. The cap is read as binding within this matchup week and is not
saved for later weeks. A league with caps and no ``games_used`` is warned about.

**Open slot-days.** A day's slot is open when nobody on the roster who plays can fill it (:attr:`DayPlan.open_slots`).
They are what a streamer fills (:mod:`fm.decide.streaming`).

**Availability.** ``p_active`` is the availability row's (the NBA official injury report's designation where it lists
the player, ESPN's otherwise: :func:`fm.model.availability.assess`), never ESPN's raw ``injuryStatus``: a player
ESPN still shows OUT whom the official report clears as Available plays. The row also zeroes the player who is
inactive, has no team or no game; whether he has a game comes from the pro schedule in hand.

**Locks** are hard constraints: a locked player (ESPN's ``lineupLocked`` at the last sync, or his lock time under
``LeagueSettings.lineup_lock_type`` has passed) keeps his slot and the slot stays his; a player on IR stays there. An
``UNKNOWN`` lock type is refused (:class:`DailyLineupError`), never guessed.

**Late swaps** (DESIGN 8.2). Among lineups worth the same, the target day's slot arrangement is chosen to value the
late-swap pivots (:func:`fm.model.availability.plan_pivots`): a questionable starter is slotted where a later bench
player eligible for the slot can cover him (:func:`prefer_pivots`).

**Category leagues** (:func:`swing_outlook`). P(win category) is approximated by ``Phi(delta_mu / sigma)`` over the
rest of the matchup, in the category model's score units, and a game is valued by its swing:
``dP/dscore = phi(delta_mu / sigma) / sigma`` per category times what he adds to the category, so a category already
decided counts for little and one near 50% for most (:func:`swing_weights`). The opponent's side is his stored
roster played at its best; without one the weights are flat (equal G-scores). ``game_sd`` is a model parameter, not
a league setting.

**Proposals** (:func:`propose_daily_lineup`, registered as ``("nba", "lineup_daily")``). :func:`plan_daily_lineup` plans
the week for the lineup as the last sync read it, which future days carry forward, and drafts the target day's moves
(and ``days_ahead`` more days': a later day's draft is stale as soon as news breaks, so the default is today only)
as at most two proposals a day, as for the NFL: ``bench_inactive`` keeps every starter who plays in the lineup and
``lineup`` is the week's best day. Each move names the slot its player is in now, so the executor refuses a stale
draft. A draft's deadline is the earliest lock among the players it moves. Nothing here writes to ESPN (CLAUDE.md).
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import combinations
from statistics import NormalDist
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from fm.config import Config
from fm.decide import registry as _registry
from fm.decide.lineup import active_slot_counts
from fm.espn.calendar import matchup_period_of, matchup_scoring_periods
from fm.espn.ids import Game, IdMaps, ids_for
from fm.espn.settings import LeagueSettings, LockType
from fm.model.availability import (
    LineupPlayer,
    OfficialReport,
    assess,
    plan_pivots,
)
from fm.model.categories import CategoryModel, fit_categories
from fm.model.projections import BlendWeights
from fm.model.scoring import Scorer
from fm.model.value_nba import blend_day, per_game_lines
from fm.proposals import LineupMove, LineupPayload, PolicyError, ProposalKind, propose
from fm.proposals.policy import stored_settings
from fm.sports.base import (
    ScheduleLike,
    SportPlugin,
    fantasy_day,
    first_start,
    game_for,
    last_start,
    period_turn,
    plugin_for,
)
from fm.sports.nba import NBA, NbaPlugin
from fm.store import AvailabilityRow, LeagueRow, PlayerRow, ProposalRow, RosterEntryRow, Store

if TYPE_CHECKING:
    from fm.sources.nba_injuries import OfficialInjuryReport

DAILY_LINEUP_KIND: Final = "lineup_daily"
"""The kind :func:`propose_daily_lineup` is registered under for NBA in :mod:`fm.decide.registry`."""
DAILY_LINEUP_CREATED_BY: Final = "decide.lineup_daily"
"""``created_by`` of the proposals this module stores."""
DEFAULT_GAME_SD: Final = 1.0
"""A player's game-to-game spread in a category in score units, absent a better estimate: about one pool standard
deviation (a single game varies about as much as players differ). A model parameter."""
PIVOT_MIN_GAIN: Final = 0.1
"""Expected value a different slot arrangement must add through late-swap pivots to be worth its moves."""
SPREAD_FLOOR: Final = 1e-9

_TOLERANCE: Final = 1e-6
"""Relative slack when a stage's optimum is held while the next is optimized."""


class DailyLineupError(ValueError):
    """The daily lineup cannot be planned as asked: the league is not synced for the period or not NBA, the matchup
    cannot be placed on ESPN's calendar, or its lineup lock type is unknown. Nothing is proposed."""


# --- the planner's inputs ---------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlayerDay:
    """One player's one day: whether his team has a game, his chance ``p_active`` of playing it and ``value``, the
    expected worth of that game from an empty slot (``p_active`` times his per-game value; 0 when he does not play).
    ``locked`` says he has locked by this day's plan time, ``lock_at`` when he locks (``None``: never, no game)."""

    has_game: bool = False
    p_active: float = 0.0
    value: float = 0.0
    locked: bool = False
    lock_at: datetime | None = None
    availability: AvailabilityRow | None = None
    designation: str | None = None
    reason: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_active <= 1.0:
            raise ValueError(f"p_active must be within [0, 1], got {self.p_active!r}")
        if not math.isfinite(self.value):
            raise ValueError(f"value must be finite, got {self.value!r}")
        if self.lock_at is not None and (self.lock_at.tzinfo is None or self.lock_at.utcoffset() is None):
            raise ValueError("lock_at must be an aware datetime")

    @property
    def plays(self) -> bool:
        """He has a game and may be active: a starter who does not play is a zero."""
        return self.has_game and self.p_active > 0.0


_NO_GAME: Final = PlayerDay()


@dataclass(frozen=True, slots=True)
class DayPlayer:
    """A player over the planning days. ``slot_id`` is where he sits now, the base every day's moves are measured from
    (future days carry the lineup forward); ``eligible`` the active slots he may fill; ``per_game`` the value of one
    game from an empty slot (what ``days`` scales by ``p_active``)."""

    espn_id: int
    slot_id: int
    eligible: frozenset[int] = frozenset()
    per_game: float = 0.0
    days: Mapping[int, PlayerDay] = field(default_factory=lambda: MappingProxyType({}))
    position: str | None = None
    name: str = ""
    pro_team_id: int | None = None

    def day(self, period: int) -> PlayerDay:
        return self.days.get(period, _NO_GAME)

    @property
    def label(self) -> str:
        return self.name or f"ESPN {self.espn_id}"


# --- the plan ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DayPlan:
    """One day's lineup: ``slots`` places every player (ESPN id -> slot id), ``moves`` take the base lineup there (both
    sides of every swap), ``value`` is the starters' expected value, ``open_slots`` the active slots nobody who plays
    can fill (one slot id per opening) and ``idle_starters`` the starters who will not play and have no replacement."""

    period: int
    slots: Mapping[int, int]
    starters: tuple[int, ...]
    moves: tuple[LineupMove, ...]
    value: float
    open_slots: tuple[int, ...] = ()
    idle_starters: tuple[int, ...] = ()
    games: Mapping[int, int] = field(default_factory=lambda: MappingProxyType({}))
    """Starts that day by slot id: what a games-played limit counts."""

    def payload(self) -> LineupPayload | None:
        return LineupPayload(moves=self.moves) if self.moves else None


@dataclass(frozen=True, slots=True)
class WeekPlan:
    """The lineups of every planned day."""

    days: tuple[DayPlan, ...]

    @property
    def total(self) -> float:
        return math.fsum(day.value for day in self.days)

    def day(self, period: int) -> DayPlan:
        for plan in self.days:
            if plan.period == period:
                return plan
        raise KeyError(f"day {period} is not in the plan; planned days: {[plan.period for plan in self.days]}")

    @property
    def open_slot_days(self) -> int:
        """Open slot-days: active slots nobody who plays can fill, summed over the days."""
        return sum(len(day.open_slots) for day in self.days)

    def open_slots_by_day(self) -> dict[int, tuple[int, ...]]:
        return {day.period: day.open_slots for day in self.days if day.open_slots}

    def starts(self, espn_id: int) -> int:
        """Days the player holds an active slot."""
        return sum(espn_id in day.starters for day in self.days)


@dataclass(frozen=True, slots=True)
class _Setup:
    """The slot arithmetic of a plan: ids, counts and what each day's locks hold."""

    ids: IdMaps
    counts: Mapping[int, int]
    players: tuple[DayPlayer, ...]

    def is_active(self, slot_id: int) -> bool:
        return slot_id not in (self.ids.bench_slot, self.ids.ir_slot)

    def is_free(self, player: DayPlayer, period: int) -> bool:
        """He may be moved on ``period``: not on IR, not locked."""
        return player.slot_id != self.ids.ir_slot and not player.day(period).locked


def _setup(players: Iterable[DayPlayer], slot_counts: Mapping[int, int], sport: Game | str) -> _Setup:
    ids = ids_for(sport)
    roster = tuple(sorted(players, key=lambda player: player.espn_id))
    seen: set[int] = set()
    for player in roster:
        if player.espn_id in seen:
            raise ValueError(f"player {player.espn_id} is listed twice")
        seen.add(player.espn_id)
    counts: dict[int, int] = {}
    for slot_id, count in slot_counts.items():
        if slot_id in (ids.bench_slot, ids.ir_slot):
            raise ValueError(f"slot_counts lists active slots only; got {ids.slot_label(slot_id)} ({slot_id})")
        if count < 0:
            raise ValueError(f"slot {ids.slot_label(slot_id)} ({slot_id}) has a negative count {count}")
        if count:
            counts[slot_id] = count
    return _Setup(ids, MappingProxyType(counts), roster)


@dataclass(frozen=True, slots=True)
class _Choice:
    """One 0/1 variable of the program: ``player`` starts on ``period`` in ``slot``."""

    period: int
    player: DayPlayer
    slot: int


def _held(setup: _Setup, period: int) -> Counter[int]:
    """Per active slot, the players who cannot move out of it on ``period`` (locked starters)."""
    return Counter(
        player.slot_id
        for player in setup.players
        if setup.is_active(player.slot_id) and not setup.is_free(player, period)
    )


def _solve(
    count: int, objective: np.ndarray, rows: list[np.ndarray], bounds: list[float]
) -> tuple[np.ndarray, float] | None:
    """Maximize ``objective . x`` over binary ``x`` subject to ``rows . x <= bounds``; ``None`` when infeasible."""
    upper: Any = np.array(bounds)
    constraints = LinearConstraint(np.array(rows), ub=upper) if rows else None
    result = milp(
        c=-objective,
        constraints=constraints,
        integrality=np.ones(count),
        bounds=Bounds(0, 1),
        options={"mip_rel_gap": 0.0},
    )
    if result.status != 0 or result.x is None:
        return None
    return result.x, float(-result.fun)


def plan_week(
    players: Iterable[DayPlayer],
    slot_counts: Mapping[int, int],
    days: Sequence[int],
    *,
    sport: Game | str = Game.FBA,
    slot_limits: Mapping[int, int] | None = None,
    keep_starters: bool = False,
) -> WeekPlan | None:
    """The best legal lineup for each of ``days`` (see the module docstring for the order of priorities).

    ``slot_counts`` maps each active slot to its count (:func:`active_slot_counts`); ``slot_limits`` maps a slot to the
    games its places may still be started for over these days (a games-played limit less what is spent). With
    ``keep_starters`` every starter who plays stays in the lineup on every day (the ``bench_inactive`` lineup: it only
    benches starters who will not play and fills empty slots); ``None`` is returned when no such plan is legal. Raises
    ``ValueError`` for a player listed twice or a bench or IR slot in ``slot_counts``.
    """
    setup = _setup(players, slot_counts, sport)
    limits = dict(slot_limits or {})
    periods = tuple(dict.fromkeys(days))
    choices: list[_Choice] = []
    for period in periods:
        held = _held(setup, period)
        open_slots = {slot: setup.counts[slot] - held[slot] for slot in setup.counts if setup.counts[slot] > held[slot]}
        for player in setup.players:
            if not setup.is_free(player, period) or not player.day(period).plays:
                continue
            choices.extend(_Choice(period, player, slot) for slot in sorted(player.eligible & open_slots.keys()))
    n = len(choices)
    held_games: Counter[int] = Counter()  # games locked starters use of each slot's limit
    for period in periods:
        for player in setup.players:
            if setup.is_active(player.slot_id) and not setup.is_free(player, period) and player.day(period).plays:
                held_games[player.slot_id] += 1
    assignment: dict[tuple[int, int], int] = {}
    if n:
        rows: list[np.ndarray] = []
        bounds: list[float] = []

        def row(members: Iterable[int]) -> np.ndarray:
            vector = np.zeros(n)
            vector[list(members)] = 1.0
            return vector

        by_player_day: dict[tuple[int, int], list[int]] = {}
        by_slot_day: dict[tuple[int, int], list[int]] = {}
        by_slot: dict[int, list[int]] = {}
        for index, choice in enumerate(choices):
            by_player_day.setdefault((choice.period, choice.player.espn_id), []).append(index)
            by_slot_day.setdefault((choice.period, choice.slot), []).append(index)
            by_slot.setdefault(choice.slot, []).append(index)
        for members in by_player_day.values():
            rows.append(row(members))
            bounds.append(1.0)
        held_by_day = {period: _held(setup, period) for period in periods}
        for (period, slot), members in by_slot_day.items():
            rows.append(row(members))
            bounds.append(float(setup.counts[slot] - held_by_day[period][slot]))
        for slot, members in by_slot.items():
            if slot in limits:
                rows.append(row(members))
                bounds.append(float(max(0, limits[slot] - held_games[slot])))
        if keep_starters:
            for members in by_player_day.values():
                if setup.is_active(choices[members[0]].player.slot_id):
                    rows.append(-row(members))
                    bounds.append(-1.0)
        values = np.array([choice.player.day(choice.period).value for choice in choices])
        starts = np.ones(n)
        stays = np.array(
            [
                float(choice.slot == choice.player.slot_id) + float(setup.is_active(choice.player.slot_id))
                for choice in choices
            ]
        )
        first = _solve(n, starts, rows, bounds)
        if first is None:
            return None
        most = round(first[1])
        rows.append(-starts)
        bounds.append(-float(most))
        second = _solve(n, values, rows, bounds)
        if second is None:
            return None
        rows.append(-values)
        bounds.append(-(second[1] - _TOLERANCE * (1.0 + abs(second[1]))))
        third = _solve(n, stays, rows, bounds)
        if third is None:
            return None
        for index in np.flatnonzero(third[0] > 0.5).tolist():
            choice = choices[index]
            assignment[(choice.period, choice.player.espn_id)] = choice.slot
    plans = [
        _day_plan(setup, period, _resolve(setup, period, {p: s for (d, p), s in assignment.items() if d == period}))
        for period in periods
    ]
    return WeekPlan(tuple(plans))


def _resolve(setup: _Setup, period: int, assigned: Mapping[int, int]) -> dict[int, int]:
    """Every player's slot on ``period`` given who the program started: locked and IR players stay, an assigned player
    takes his slot, a starter nobody replaces stays when he cannot play and the room is there, everyone else sits."""
    slots: dict[int, int] = {}
    used: Counter[int] = Counter()
    for player in setup.players:
        if not setup.is_free(player, period):
            slots[player.espn_id] = player.slot_id
            if setup.is_active(player.slot_id):
                used[player.slot_id] += 1
        elif player.espn_id in assigned:
            slots[player.espn_id] = assigned[player.espn_id]
            used[assigned[player.espn_id]] += 1
    for player in setup.players:
        if player.espn_id in slots:
            continue
        stays = (
            setup.is_active(player.slot_id)
            and not player.day(period).plays
            and used[player.slot_id] < setup.counts.get(player.slot_id, 0)
        )
        slots[player.espn_id] = player.slot_id if stays else setup.ids.bench_slot
        if stays:
            used[player.slot_id] += 1
    return slots


def _move_order(setup: _Setup, move: LineupMove) -> tuple[int, int]:
    """Players leaving the lineup first, then shifts between slots, then players entering it."""
    leaving = setup.is_active(move.from_slot_id) and not setup.is_active(move.to_slot_id)
    entering = not setup.is_active(move.from_slot_id) and setup.is_active(move.to_slot_id)
    return (0 if leaving else 2 if entering else 1, move.espn_id)


def _day_plan(setup: _Setup, period: int, slots: Mapping[int, int]) -> DayPlan:
    """The :class:`DayPlan` for a full slot map."""
    moves = sorted(
        (
            LineupMove(espn_id=player.espn_id, from_slot_id=player.slot_id, to_slot_id=slots[player.espn_id])
            for player in setup.players
            if slots[player.espn_id] != player.slot_id
        ),
        key=lambda move: _move_order(setup, move),
    )
    starters = [player for player in setup.players if setup.is_active(slots[player.espn_id])]
    held = _held(setup, period)
    games: Counter[int] = Counter()
    free_playing: Counter[int] = Counter()
    for player in starters:
        if player.day(period).plays:
            games[slots[player.espn_id]] += 1
            if setup.is_free(player, period):
                free_playing[slots[player.espn_id]] += 1
    open_slots = tuple(
        slot
        for slot, count in sorted(setup.counts.items())
        for _ in range(max(0, count - held[slot] - free_playing[slot]))
    )
    return DayPlan(
        period=period,
        slots=MappingProxyType(dict(slots)),
        starters=tuple(player.espn_id for player in starters),
        moves=tuple(moves),
        value=math.fsum(player.day(period).value for player in starters if player.day(period).plays),
        open_slots=open_slots,
        idle_starters=tuple(player.espn_id for player in starters if not player.day(period).plays),
        games=MappingProxyType(dict(games)),
    )


def hold_week(
    players: Iterable[DayPlayer], slot_counts: Mapping[int, int], days: Sequence[int], *, sport: Game | str = Game.FBA
) -> WeekPlan:
    """The lineup as it stands, carried through ``days`` and scored like a plan (no moves)."""
    setup = _setup(players, slot_counts, sport)
    return WeekPlan(
        tuple(
            _day_plan(setup, period, {player.espn_id: player.slot_id for player in setup.players})
            for period in dict.fromkeys(days)
        )
    )


# --- late-swap pivots -------------------------------------------------------------------------------------------------


def _pivot_value(setup: _Setup, period: int, slots: Mapping[int, int], lead: timedelta | None) -> float:
    lineup: list[tuple[int, LineupPlayer]] = []
    bench: list[LineupPlayer] = []
    for player in setup.players:
        day = player.day(period)
        if day.availability is None or not day.plays or player.slot_id == setup.ids.ir_slot:
            continue
        late = LineupPlayer(day.availability, player.per_game, player.eligible, day.lock_at)
        slot = slots[player.espn_id]
        if setup.is_active(slot):
            lineup.append((slot, late))
        else:
            bench.append(late)
    return plan_pivots(lineup, bench, lead=lead).expected


def prefer_pivots(
    players: Iterable[DayPlayer],
    slot_counts: Mapping[int, int],
    plan: DayPlan,
    *,
    sport: Game | str = Game.FBA,
    lead: timedelta | None = None,
    min_gain: float = PIVOT_MIN_GAIN,
) -> DayPlan:
    """``plan`` with its starters moved between slots where that makes the day worth more counting late swaps
    (:func:`fm.model.availability.plan_pivots`): the FLEX/UTIL trick, a questionable starter whose game is late sits in
    a slot a later bench player can also fill. The starters and so the day's base value do not change; a rearrangement
    must add at least ``min_gain`` of expected value to be worth its moves. Only players who are free to move are
    swapped."""
    setup = _setup(players, slot_counts, sport)
    period = plan.period
    slots = dict(plan.slots)
    current = _pivot_value(setup, period, slots, lead)
    by_id = {player.espn_id: player for player in setup.players}
    for _ in range(len(setup.players)):
        best: tuple[float, dict[int, int]] | None = None
        movable = [by_id[i] for i in plan.starters if setup.is_free(by_id[i], period) and by_id[i].day(period).plays]
        for first, second in combinations(movable, 2):
            one, two = slots[first.espn_id], slots[second.espn_id]
            if one == two or one not in second.eligible or two not in first.eligible:
                continue
            trial = {**slots, first.espn_id: two, second.espn_id: one}
            value = _pivot_value(setup, period, trial, lead)
            if value > current + min_gain and (best is None or value > best[0]):
                best = (value, trial)
        if best is None:
            break
        current, slots = best
    if slots == dict(plan.slots):
        return plan
    return _day_plan(setup, period, slots)


# --- category matchups ------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SwingOutlook:
    """Where each category stands over the rest of the matchup, in the category model's score units: the expected
    ``margins`` (ours minus theirs), ``sds`` of that difference, ``win_probability`` ``Phi(margin / sd)`` and the swing
    ``weights`` ``phi(margin / sd) / sd`` (``dP/dscore``): near 50% a category is worth the most, decided it is worth
    almost nothing."""

    margins: Mapping[str, float]
    sds: Mapping[str, float]
    win_probability: Mapping[str, float]
    weights: Mapping[str, float]

    @property
    def expected_wins(self) -> float:
        return math.fsum(self.win_probability.values())


def swing_outlook(
    ours: Mapping[str, float],
    theirs: Mapping[str, float],
    *,
    games_ours: float,
    games_theirs: float,
    game_sd: Mapping[str, float] | float = DEFAULT_GAME_SD,
    margin_so_far: Mapping[str, float] | None = None,
) -> SwingOutlook:
    """The outlook from each side's expected category scores over the rest of the matchup.

    ``games_*`` count the started games behind them; every game's category score varies independently by ``game_sd``
    (one number for all categories or one per category), so the difference's spread is ``game_sd * sqrt(games_ours +
    games_theirs)``. ``margin_so_far`` adds what the matchup has already banked (ours minus theirs, same units).
    """
    normal = NormalDist()
    margins: dict[str, float] = {}
    sds: dict[str, float] = {}
    chances: dict[str, float] = {}
    weights: dict[str, float] = {}
    for category, mine in ours.items():
        margin = mine - theirs.get(category, 0.0) + (margin_so_far or {}).get(category, 0.0)
        sd_game = game_sd if isinstance(game_sd, int | float) else game_sd.get(category, DEFAULT_GAME_SD)
        sd = max(SPREAD_FLOOR, sd_game * math.sqrt(max(games_ours + games_theirs, 1.0)))
        margins[category], sds[category] = margin, sd
        chances[category] = normal.cdf(margin / sd)
        weights[category] = normal.pdf(margin / sd) / sd
    return SwingOutlook(
        MappingProxyType(margins), MappingProxyType(sds), MappingProxyType(chances), MappingProxyType(weights)
    )


def swing_weights(outlook: SwingOutlook | None, categories: Iterable[str]) -> dict[str, float] | None:
    """The per-category weights for :meth:`fm.model.categories.CategoryModel.contribution`: the outlook's swing weights,
    or ``None`` (every category weighs 1) without one."""
    if outlook is None:
        return None
    return {category: outlook.weights.get(category, 0.0) for category in categories}


def week_category_scores(
    plan: WeekPlan, players: Iterable[DayPlayer], vectors: Mapping[int, Mapping[str, float]]
) -> tuple[dict[str, float], int]:
    """A plan's expected category scores over its days (each started game's ``p_active`` times the player's per-game
    category contributions in ``vectors``, from :meth:`fm.model.categories.CategoryModel.contributions`) and its
    started games that count."""
    by_id = {player.espn_id: player for player in players}
    totals: dict[str, float] = {}
    games = 0
    for day in plan.days:
        for espn_id in day.starters:
            player = by_id.get(espn_id)
            if player is None or not player.day(day.period).plays or espn_id not in vectors:
                continue
            p_active = player.day(day.period).p_active
            games += 1
            for category, score in vectors[espn_id].items():
                totals[category] = totals.get(category, 0.0) + p_active * score
    return totals, games


# --- reading the store ------------------------------------------------------------------------------------------------


def slot_limits_left(
    settings: LeagueSettings, games_used: Mapping[int, int] | None = None
) -> tuple[dict[int, int], tuple[str, ...]]:
    """What is left of each slot's games-played limit (``lineupSlotStatLimits``) and the warnings: a league with limits
    and no ``games_used`` has them assumed unspent."""
    limits = {slot: cap for slot in settings.slot_stat_limits if (cap := settings.games_played_limit(slot)) is not None}
    if not limits:
        return {}, ()
    used = games_used or {}
    left = {slot: max(0, cap - used.get(slot, 0)) for slot, cap in limits.items()}
    warnings = (
        ()
        if games_used is not None
        else (
            "the league caps games played per slot but the games used so far are unknown "
            "(pass games_used by slot id); assuming none are spent",
        )
    )
    return left, warnings


def matchup_window(
    settings: LeagueSettings, period: int, *, horizon: int | None = None
) -> tuple[int | None, tuple[int, ...]]:
    """The matchup period of ``period`` and its remaining days from ``period`` through the league's last scoring
    period (``horizon`` caps the count). Raises :class:`DailyLineupError` when ESPN's calendar cannot place the day."""
    span = matchup_scoring_periods(settings, period)
    if span is None:
        raise DailyLineupError(
            f"league {settings.league_id} matchup days for scoring period {period} are unknown: "
            f"no ESPN calendar for {settings.game.value} {settings.season} (fm.espn.calendar) or the day is in no "
            "listed matchup"
        )
    last = settings.final_scoring_period
    remaining = tuple(day for day in span if day >= period and (last is None or day <= last))
    if horizon is not None:
        remaining = remaining[: max(1, horizon)]
    return matchup_period_of(settings, period), remaining


def _require_aware(now: datetime) -> None:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError(f"now must be an aware datetime; got a naive {now.isoformat()}")


def _freshest(
    store: Store,
    player: PlayerRow,
    *,
    season: int,
    period: int,
    now: datetime,
    schedule: ScheduleLike,
    official: OfficialReport | OfficialInjuryReport | None,
) -> AvailabilityRow:
    """The stored availability row when it is no older than the player's synced designation, else a fresh one."""
    stored = store.availability.get(player.sport, player.espn_id, season, period)
    if stored is not None and stored.as_of >= player.as_of:
        return stored
    return assess(player, season=season, scoring_period=period, as_of=now, schedule=schedule, official=official)


@dataclass(frozen=True)
class DayModel:
    """Builds :class:`DayPlayer` rows for one league's planning days: the per-game value of each player (``per_game``),
    his game and ``p_active`` each day, and his lock each day."""

    store: Store
    league: LeagueRow
    settings: LeagueSettings
    schedule: ScheduleLike
    now: datetime
    days: tuple[int, ...]
    per_game: Mapping[int, float]
    plugin: SportPlugin = NBA
    official: OfficialReport | OfficialInjuryReport | None = None
    respect_locks: bool = True
    _locks: Mapping[int, tuple[Mapping[int, datetime], datetime | None]] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _require_aware(self.now)
        tables: dict[int, tuple[Mapping[int, datetime], datetime | None]] = {}
        lock_type = self.settings.lineup_lock_type
        for day in self.days:
            try:
                locks = self.plugin.locks(day, self.schedule, lock_type=lock_type)
            except ValueError as exc:
                raise DailyLineupError(str(exc)) from exc
            by_team: dict[int, datetime] = {}
            for lock in locks:
                by_team.setdefault(lock.team_id, lock.at)
            everyone = None if lock_type is LockType.INDIVIDUAL_GAME else first_start(self.schedule, day)
            tables[day] = (by_team, everyone)
        object.__setattr__(self, "_locks", MappingProxyType(tables))

    def lock_at(self, player: PlayerRow, day: int) -> datetime | None:
        by_team, everyone = self._locks[day]
        team = by_team.get(player.pro_team_id) if player.pro_team_id is not None else None
        return team if team is not None else everyone

    def day_player(self, row: PlayerRow, slot_id: int, *, flagged_locked: bool = False) -> DayPlayer | None:
        """The player as the planner sees him, or ``None`` when he has no per-game value (no projection line)."""
        per_game = self.per_game.get(row.espn_id)
        if per_game is None:
            return None
        listed = frozenset(row.eligible_slot_ids)
        if not listed and row.position is not None:
            try:
                listed = self.plugin.eligible_slots(row.position, include_reserve=False)
            except KeyError:
                listed = frozenset()
        eligible = listed & frozenset(active_slot_counts(self.settings))
        days: dict[int, PlayerDay] = {}
        for day in self.days:
            lock_at = self.lock_at(row, day)
            has_game = row.pro_team_id is not None and game_for(self.schedule, row.pro_team_id, day) is not None
            if not has_game:
                days[day] = PlayerDay(lock_at=lock_at)
                continue
            availability = _freshest(
                self.store,
                row,
                season=self.league.season,
                period=day,
                now=self.now,
                schedule=self.schedule,
                official=self.official,
            )
            p_active = availability.p_active
            locked = self.respect_locks and (
                (flagged_locked and day == self.days[0]) or (lock_at is not None and self.now >= lock_at)
            )
            days[day] = PlayerDay(
                has_game=True,
                p_active=p_active,
                value=p_active * per_game,
                locked=locked,
                lock_at=lock_at,
                availability=availability,
                designation=availability.designation,
                reason=availability.inputs.get("reason"),
            )
        return DayPlayer(
            espn_id=row.espn_id,
            slot_id=slot_id,
            eligible=eligible,
            per_game=per_game,
            days=MappingProxyType(days),
            position=row.position,
            name=row.full_name,
            pro_team_id=row.pro_team_id,
        )

    def roster(
        self, entries: Iterable[RosterEntryRow], rows: Mapping[int, PlayerRow]
    ) -> tuple[list[DayPlayer], list[str]]:
        """The roster entries as players; entries without a ``players`` row or a value are warned about and held where
        they are as locked non-players."""
        players: list[DayPlayer] = []
        warnings: list[str] = []
        for entry in entries:
            row = rows.get(entry.espn_id)
            if row is None:
                warnings.append(f"player {entry.espn_id} has no players row; left in slot {entry.lineup_slot_id}")
                players.append(self._frozen(entry))
                continue
            player = self.day_player(row, entry.lineup_slot_id, flagged_locked=entry.lineup_locked)
            if player is None:
                warnings.append(f"no projection for {row.full_name} ({row.espn_id}); counted as 0 and left in place")
                player = self._frozen(entry, row)
            players.append(player)
        return players, warnings

    def _frozen(self, entry: RosterEntryRow, row: PlayerRow | None = None) -> DayPlayer:
        """A player the plan cannot value: kept where he is, locked every day."""
        days = {day: PlayerDay(locked=True) for day in self.days}
        return DayPlayer(
            espn_id=entry.espn_id,
            slot_id=entry.lineup_slot_id,
            days=MappingProxyType(days),
            name=row.full_name if row is not None else "",
        )


@dataclass(frozen=True, slots=True)
class GameValues:
    """The per-game value of each player in the unit the lineups maximize: league points, or a category league's swing
    weighted category score, and the :class:`SwingOutlook` behind the weights."""

    per_game: Mapping[int, float]
    unit: str
    outlook: SwingOutlook | None = None
    model: CategoryModel | None = None
    vectors: Mapping[int, Mapping[str, float]] = field(default_factory=lambda: MappingProxyType({}))
    warnings: tuple[str, ...] = ()


def game_values(
    settings: LeagueSettings,
    lines: Mapping[int, Mapping[str, float]],
    positions: Mapping[int, str | None],
    *,
    model: CategoryModel | None = None,
    outlook: SwingOutlook | None = None,
) -> GameValues:
    """One game's value from an empty slot for each player with a per-game line. A points league scores the line with
    the league's items; a category league needs a fitted ``model`` (:func:`fit_categories` on the pool's lines when not
    given) and weights each category by the ``outlook``'s swing (flat without one)."""
    if settings.game is not Game.FBA:
        raise DailyLineupError(f"league {settings.league_id} is {settings.game.value}, not an NBA (fba) league")
    if settings.is_points:
        scorer = Scorer(settings)
        return GameValues(
            MappingProxyType({i: scorer.points(line, position=positions.get(i)) for i, line in lines.items()}),
            "points",
        )
    fitted = model if model is not None else fit_categories(lines, settings)
    weights = swing_weights(outlook, fitted.categories)
    return GameValues(
        MappingProxyType({i: fitted.contribution(line, weights) for i, line in lines.items()}),
        "category score",
        outlook,
        fitted,
        MappingProxyType({i: MappingProxyType(fitted.contributions(line)) for i, line in lines.items()}),
        fitted.warnings,
    )


# --- the decision -----------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class DailyInputs:
    """Our roster over the planning days as :class:`DayPlayer` rows, read from the store by :func:`daily_inputs`."""

    team_id: int
    season: int
    period: int
    days: tuple[int, ...]
    matchup_period: int | None
    players: tuple[DayPlayer, ...]
    slot_counts: Mapping[int, int]
    slot_limits: Mapping[int, int]
    values: GameValues
    model: DayModel
    roster_as_of: datetime
    warnings: tuple[str, ...] = ()


def _lines_for(
    store: Store,
    league: LeagueRow,
    period: int,
    lines: Mapping[int, Mapping[str, float]] | None,
    weights: BlendWeights | None,
) -> tuple[Mapping[int, Mapping[str, float]], tuple[str, ...]]:
    """The day's per-game lines: given, else the blend the tick stored, else a fresh blend of the sources."""
    if lines is not None:
        return lines, ()
    stored = per_game_lines(store, league.season, period)
    if stored:
        return stored, ()
    blended = blend_day(store, league.season, period, weights=weights if weights is not None else BlendWeights.load())
    return per_game_lines(store, league.season, period), blended.warnings


def daily_inputs(
    store: Store,
    league: LeagueRow,
    settings: LeagueSettings,
    *,
    schedule: ScheduleLike,
    period: int,
    now: datetime,
    lines: Mapping[int, Mapping[str, float]] | None = None,
    weights: BlendWeights | None = None,
    horizon: int | None = None,
    games_used: Mapping[int, int] | None = None,
    opponent_team_id: int | None = None,
    model: CategoryModel | None = None,
    official: OfficialReport | OfficialInjuryReport | None = None,
    game_sd: Mapping[str, float] | float = DEFAULT_GAME_SD,
    margin_so_far: Mapping[str, float] | None = None,
) -> DailyInputs:
    """Our roster for the rest of the matchup as :class:`DayPlayer` rows.

    Reads the roster snapshot of ``period``, the ``players`` rows, the day's blended per-game lines (``lines`` or
    :func:`_lines_for`), availability (:func:`_freshest`) and locks (from ``schedule`` at ``now`` and ESPN's
    ``lineupLocked``). In a category league the opponent's stored roster (``opponent_team_id``) sets the swing weights
    (:func:`swing_outlook`). Raises :class:`DailyLineupError` when the period's roster snapshot is missing, the
    settings are not an NBA league's, the matchup cannot be placed on the calendar or the lock type is unknown.
    """
    _require_aware(now)
    if settings.game is not Game.FBA or league.sport != "nba":
        raise DailyLineupError(f"league {league.key!r} is {league.sport}; the daily lineup is for NBA leagues")
    entries = store.rosters.team(league.row_id, period, league.team_id)
    if not entries:
        raise DailyLineupError(
            f"no roster for team {league.team_id} of league {league.key!r} in scoring period {period}; run fm sync"
        )
    matchup, days = matchup_window(settings, period, horizon=horizon)
    if not days:
        raise DailyLineupError(f"league {league.key!r} has no scoring period left from {period}: the season is over")
    day_lines, warnings = _lines_for(store, league, period, lines, weights)
    roster_ids = {entry.espn_id for entry in entries}
    rows = {row.espn_id: row for row in store.players.many(league.sport, roster_ids)}
    all_warnings = list(warnings)
    flat = game_values(settings, day_lines, _positions(store, league, day_lines, rows), model=model)
    values = flat
    plugin = plugin_for(league.sport)
    day_model = DayModel(store, league, settings, schedule, now, days, flat.per_game, plugin, official)
    players, notes = day_model.roster(entries, rows)
    all_warnings.extend(notes)
    if settings.is_categories and flat.model is not None:
        outlook = None
        if opponent_team_id is not None:
            outlook, opponent_notes = _opponent_outlook(
                store, league, settings, flat, day_model, players, period, opponent_team_id, game_sd, margin_so_far
            )
            all_warnings.extend(opponent_notes)
        else:
            all_warnings.append("no opponent: category swing weights are flat (equal scores)")
        values = game_values(
            settings, day_lines, _positions(store, league, day_lines, rows), model=flat.model, outlook=outlook
        )
        day_model = DayModel(store, league, settings, schedule, now, days, values.per_game, plugin, official)
        players, _ = day_model.roster(entries, rows)
    all_warnings.extend(values.warnings)
    limits, limit_notes = slot_limits_left(settings, games_used)
    all_warnings.extend(limit_notes)
    return DailyInputs(
        team_id=league.team_id,
        season=league.season,
        period=period,
        days=days,
        matchup_period=matchup,
        players=tuple(players),
        slot_counts=MappingProxyType(active_slot_counts(settings)),
        slot_limits=MappingProxyType(limits),
        values=values,
        model=day_model,
        roster_as_of=min(entry.as_of for entry in entries),
        warnings=tuple(dict.fromkeys(all_warnings)),
    )


def _positions(
    store: Store, league: LeagueRow, lines: Mapping[int, Mapping[str, float]], rows: Mapping[int, PlayerRow]
) -> dict[int, str | None]:
    """Positions of everyone with a line: the roster's rows plus the store's for the rest."""
    known = {espn_id: row.position for espn_id, row in rows.items()}
    missing = [espn_id for espn_id in lines if espn_id not in known]
    if missing:
        known.update({row.espn_id: row.position for row in store.players.many(league.sport, missing)})
    return known


def _opponent_outlook(
    store: Store,
    league: LeagueRow,
    settings: LeagueSettings,
    flat: GameValues,
    day_model: DayModel,
    ours: Sequence[DayPlayer],
    period: int,
    opponent_team_id: int,
    game_sd: Mapping[str, float] | float,
    margin_so_far: Mapping[str, float] | None,
) -> tuple[SwingOutlook | None, list[str]]:
    entries = store.rosters.team(league.row_id, period, opponent_team_id)
    if not entries:
        return None, [f"opponent: no roster for team {opponent_team_id} in period {period}; flat category weights"]
    rows = {row.espn_id: row for row in store.players.many(league.sport, {entry.espn_id for entry in entries})}
    unlocked = DayModel(
        day_model.store,
        league,
        settings,
        day_model.schedule,
        day_model.now,
        day_model.days,
        flat.per_game,
        day_model.plugin,
        day_model.official,
        respect_locks=False,
    )
    theirs, notes = unlocked.roster(entries, rows)
    counts = active_slot_counts(settings)
    mine = plan_week(ours, counts, day_model.days)
    other = plan_week(theirs, counts, day_model.days)
    if mine is None or other is None:  # unreachable without keep_starters; stay safe
        return None, ["opponent: no lineup could be planned; flat category weights"]
    our_scores, our_games = week_category_scores(mine, ours, flat.vectors)
    their_scores, their_games = week_category_scores(other, theirs, flat.vectors)
    outlook = swing_outlook(
        our_scores,
        their_scores,
        games_ours=our_games,
        games_theirs=their_games,
        game_sd=game_sd,
        margin_so_far=margin_so_far,
    )
    return outlook, [f"opponent: {note}" for note in notes]


@dataclass(frozen=True, slots=True)
class DailyDraft:
    """A lineup proposal for one day, ready for :func:`fm.proposals.propose`."""

    kind: ProposalKind
    payload: LineupPayload
    scoring_period_id: int
    deadline: datetime | None
    engine_numbers: Mapping[str, Any]
    rationale: str
    dedupe_key: str


@dataclass(frozen=True, slots=True)
class DailyLineupDecision:
    """What :func:`plan_daily_lineup` found: the inputs, the lineups as they stand (``current``), the best plan
    (``best``), the ``bench_inactive`` plan that keeps every starter who plays (``rescue``, ``None`` when no such plan
    is legal) and the drafts for the days proposed."""

    league_id: int
    inputs: DailyInputs
    current: WeekPlan
    best: WeekPlan
    rescue: WeekPlan | None
    drafts: tuple[DailyDraft, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class DailyLineupProposals:
    """What :func:`propose_daily_lineup` stored (``proposals``; an open proposal with the same dedupe key is handed
    back) and the policy refusals it caught (``blocked``, one message per draft)."""

    decision: DailyLineupDecision
    proposals: tuple[ProposalRow, ...] = ()
    blocked: tuple[str, ...] = ()

    @property
    def warnings(self) -> tuple[str, ...]:
        return self.decision.warnings


def _digest(parts: Iterable[object]) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16]


def _deadline(moved: Iterable[DayPlayer], period: int, schedule: ScheduleLike, plugin: NbaPlugin) -> datetime | None:
    """The earliest lock among the moved players; with none, when ESPN's calendar leaves the day."""
    locks = [lock for player in moved if (lock := player.day(period).lock_at) is not None]
    if locks:
        return min(locks)
    day = plugin.period_day(period, schedule)
    if day is not None:
        return period_turn(day + timedelta(days=1))
    last = last_start(schedule, period)
    return None if last is None else period_turn(fantasy_day(last) + timedelta(days=1))


def _label(player: DayPlayer, period: int) -> str:
    day = player.day(period)
    if not day.has_game:
        return f"{player.label} (no game)"
    if day.reason == "designation" or day.p_active == 0.0:
        return f"{player.label} ({day.designation or 'out'})"
    return player.label


def _rationale(kind: ProposalKind, period: int, plan: DayPlan, before: DayPlan, by_id: Mapping[int, DayPlayer]) -> str:
    entering = [
        by_id[i].label for i in plan.starters if i not in before.starters and i in {m.espn_id for m in plan.moves}
    ]
    leaving = [by_id[i] for i in before.starters if i not in plan.starters and i in {m.espn_id for m in plan.moves}]
    parts: list[str] = []
    if kind is ProposalKind.BENCH_INACTIVE:
        if leaving:
            parts.append("Bench " + ", ".join(_label(player, period) for player in leaving))
        if entering:
            parts.append("start " + ", ".join(entering))
    else:
        if entering:
            parts.append("Start " + ", ".join(entering))
        if leaving:
            parts.append("bench " + ", ".join(player.label for player in leaving))
    text = "; ".join(parts) or "Rearrange the starters"
    text += f" on day {period}: {plan.value:.1f} expected ({plan.value - before.value:+.1f})"
    if plan.open_slots:
        text += f", {len(plan.open_slots)} slot(s) left open"
    return text + "."


def _move_numbers(move: LineupMove, player: DayPlayer, period: int, ids: IdMaps) -> dict[str, Any]:
    day = player.day(period)
    return {
        "espn_id": move.espn_id,
        "name": player.name,
        "position": player.position,
        "from_slot_id": move.from_slot_id,
        "to_slot_id": move.to_slot_id,
        "from": ids.slot_label(move.from_slot_id),
        "to": ids.slot_label(move.to_slot_id),
        "per_game": player.per_game,
        "p_active": day.p_active,
        "expected": day.value,
        "has_game": day.has_game,
        "designation": day.designation,
        "reason": day.reason,
        "lock_at": None if day.lock_at is None else day.lock_at.isoformat(),
    }


def _draft(
    kind: ProposalKind,
    plan: DayPlan,
    before: DayPlan,
    inputs: DailyInputs,
    *,
    schedule: ScheduleLike,
    basis: str,
) -> DailyDraft:
    by_id = {player.espn_id: player for player in inputs.players}
    ids = ids_for(Game.FBA)
    moved = [by_id[move.espn_id] for move in plan.moves]
    outlook = inputs.values.outlook
    numbers: dict[str, Any] = {
        "unit": inputs.values.unit,
        "day": plan.period,
        "expected_value": plan.value,
        "current_expected_value": before.value,
        "gain": plan.value - before.value,
        "open_slots": list(plan.open_slots),
        "idle_starters": list(plan.idle_starters),
        "basis": basis,
        "roster_as_of": inputs.roster_as_of.isoformat(),
        "moves": [_move_numbers(move, by_id[move.espn_id], plan.period, ids) for move in plan.moves],
        "swing_outlook": None
        if outlook is None
        else {
            "expected_wins": outlook.expected_wins,
            "win_probability": dict(outlook.win_probability),
            "weights": dict(outlook.weights),
        },
    }
    moves_key = _digest(f"{m.espn_id}:{m.from_slot_id}>{m.to_slot_id}" for m in plan.moves)
    return DailyDraft(
        kind=kind,
        payload=LineupPayload(moves=plan.moves),
        scoring_period_id=plan.period,
        deadline=_deadline(moved, plan.period, schedule, NBA),
        engine_numbers=numbers,
        rationale=_rationale(kind, plan.period, plan, before, by_id),
        dedupe_key=f"{kind.value}:{inputs.season}:{plan.period}:{moves_key}",
    )


def plan_daily_lineup(
    store: Store,
    league: LeagueRow,
    *,
    schedule: ScheduleLike,
    now: datetime,
    settings: LeagueSettings | None = None,
    period: int | None = None,
    days_ahead: int = 0,
    horizon: int | None = None,
    lines: Mapping[int, Mapping[str, float]] | None = None,
    weights: BlendWeights | None = None,
    games_used: Mapping[int, int] | None = None,
    opponent_team_id: int | None = None,
    model: CategoryModel | None = None,
    official: OfficialReport | OfficialInjuryReport | None = None,
    game_sd: Mapping[str, float] | float = DEFAULT_GAME_SD,
    margin_so_far: Mapping[str, float] | None = None,
    pivot_min_gain: float = PIVOT_MIN_GAIN,
) -> DailyLineupDecision:
    """Plan our lineups for the rest of the matchup week from the store and draft the target day's proposals (and
    ``days_ahead`` more), writing nothing (the module docs give the rules). ``settings`` default to the synced ones and
    ``period`` to the day current at ``now`` on the pro schedule; ``horizon`` caps the days planned. Raises
    :class:`DailyLineupError` as :func:`daily_inputs` does, and when the league is not synced or the season is over."""
    _require_aware(now)
    synced = settings if settings is not None else stored_settings(store, league)
    if synced is None:
        raise DailyLineupError(f"league {league.key!r} has no synced settings; run fm sync")
    target = period if period is not None else NBA.scoring_period_at(now, schedule)
    if target is None:
        raise DailyLineupError(f"the pro schedule has no scoring period left at {now.isoformat()}: the season is over")
    inputs = daily_inputs(
        store,
        league,
        synced,
        schedule=schedule,
        period=target,
        now=now,
        lines=lines,
        weights=weights,
        horizon=horizon,
        games_used=games_used,
        opponent_team_id=opponent_team_id,
        model=model,
        official=official,
        game_sd=game_sd,
        margin_so_far=margin_so_far,
    )
    best = plan_week(inputs.players, inputs.slot_counts, inputs.days, slot_limits=inputs.slot_limits)
    if best is None:  # no constraint forces a plan to exist only with keep_starters
        raise DailyLineupError(f"league {league.key!r}: no legal lineup could be planned")
    first = inputs.days[0]
    best = WeekPlan(
        (
            prefer_pivots(inputs.players, inputs.slot_counts, best.day(first), min_gain=pivot_min_gain),
            *best.days[1:],
        )
    )
    rescue = plan_week(
        inputs.players, inputs.slot_counts, inputs.days, slot_limits=inputs.slot_limits, keep_starters=True
    )
    current = hold_week(inputs.players, inputs.slot_counts, inputs.days)
    basis = _digest(sorted((player.espn_id, player.slot_id) for player in inputs.players))
    drafts: list[DailyDraft] = []
    for day in inputs.days[: days_ahead + 1]:
        before = current.day(day)
        rescue_day = rescue.day(day) if rescue is not None else None
        if rescue_day is not None and rescue_day.moves:
            drafts.append(
                _draft(ProposalKind.BENCH_INACTIVE, rescue_day, before, inputs, schedule=schedule, basis=basis)
            )
        best_day = best.day(day)
        if best_day.moves and (rescue_day is None or dict(best_day.slots) != dict(rescue_day.slots)):
            drafts.append(_draft(ProposalKind.LINEUP, best_day, before, inputs, schedule=schedule, basis=basis))
    return DailyLineupDecision(
        league_id=league.row_id,
        inputs=inputs,
        current=current,
        best=best,
        rescue=rescue,
        drafts=tuple(drafts),
        warnings=inputs.warnings,
    )


def propose_daily_lineup(
    store: Store,
    config: Config,
    league: LeagueRow,
    *,
    schedule: ScheduleLike,
    now: datetime,
    settings: LeagueSettings | None = None,
    period: int | None = None,
    days_ahead: int = 0,
    horizon: int | None = None,
    lines: Mapping[int, Mapping[str, float]] | None = None,
    weights: BlendWeights | None = None,
    games_used: Mapping[int, int] | None = None,
    opponent_team_id: int | None = None,
    model: CategoryModel | None = None,
    official: OfficialReport | OfficialInjuryReport | None = None,
    game_sd: Mapping[str, float] | float = DEFAULT_GAME_SD,
    margin_so_far: Mapping[str, float] | None = None,
    pivot_min_gain: float = PIVOT_MIN_GAIN,
) -> DailyLineupProposals:
    """The NBA ``lineup_daily`` decision: :func:`plan_daily_lineup`, then each draft through
    :func:`fm.proposals.propose` (policy, guardrails, dedupe). A draft policy refuses is reported in ``blocked``, not
    raised; :class:`DailyLineupError` still is. Takes :func:`plan_daily_lineup`'s arguments plus the config."""
    decision = plan_daily_lineup(
        store,
        league,
        schedule=schedule,
        now=now,
        settings=settings,
        period=period,
        days_ahead=days_ahead,
        horizon=horizon,
        lines=lines,
        weights=weights,
        games_used=games_used,
        opponent_team_id=opponent_team_id,
        model=model,
        official=official,
        game_sd=game_sd,
        margin_so_far=margin_so_far,
        pivot_min_gain=pivot_min_gain,
    )
    stored: list[ProposalRow] = []
    blocked: list[str] = []
    for draft in decision.drafts:
        try:
            stored.append(
                propose(
                    store,
                    config,
                    league,
                    draft.kind,
                    draft.payload,
                    created_by=DAILY_LINEUP_CREATED_BY,
                    scoring_period_id=draft.scoring_period_id,
                    engine_numbers=draft.engine_numbers,
                    rationale=draft.rationale,
                    deadline=draft.deadline,
                    dedupe_key=draft.dedupe_key,
                    now=now,
                )
            )
        except PolicyError as exc:
            blocked.append(str(exc))
    return DailyLineupProposals(decision=decision, proposals=tuple(stored), blocked=tuple(blocked))


_registry.register("nba", DAILY_LINEUP_KIND, propose_daily_lineup)
