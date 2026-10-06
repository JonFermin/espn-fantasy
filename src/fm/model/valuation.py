"""Valuation for points leagues: rest-of-season start value, replacement level and value over replacement (DESIGN
section 8.3, ROADMAP #21).

Everything here is in the league's own points (:class:`fm.model.scoring.Scorer` over its scoring items, so PPR, half
PPR and custom scoring all go through one path) and in expected points: a projection times the chance the player
plays.

**Weekly expected points** (:class:`PlayerOutlook`). For the current scoring period it is the blended projection (ESPN
and Sleeper, :func:`fm.model.projections.blend`) scored for the league, times ``p_active`` from the designation model
(:func:`fm.model.availability.assess` with the pro schedule and no news signals, so a Claude-only signal never moves a
drop; CLAUDE.md). Later periods come from ESPN's season projection, the only line that reaches past this week: its
points per game, times the player's games that week (none on a bye, from the pro schedule), times ``health``, the share
of his team's remaining games ESPN projects him to play (its games-played stat ``GP`` over the games left), which is
how an injury's expected length enters.

**Which season line is rest-of-season.** ESPN sends ``ffl`` players two season projections: ``102026`` (split 0,
"Season", what :meth:`fm.espn.models.Player.projection` reads for period 0 and :mod:`fm.jobs.sync` stores as scoring
period 0) and ``122026`` (split 2, labelled "Rest Of Season"). The labels mislead, and the real week-4 2026 captures
settle it (docs/espn-api.md section 1 #10): ``102026``'s ``GP`` is exactly each healthy player's remaining games, 13
(NFL weeks 5 to 18 less his bye) for Colston Loveland, Chase Brown, Rashee Rice and Jalen Coker, whose OUT week 4 was
already played, and falls short only for the injured (DeVonta Smith 12, A.J. Brown 10 on IR), while ``122026`` gives
Smith 16 games with 13 left. So ``102026`` is the rest-of-season projection; ``122026`` is not, and nothing reads it
(``tests/model/test_valuation.py`` pins the evidence). The line spans the rest of the NFL regular season, week 18
included, past the fantasy season's last week, which is why only its rate per game is used and the weeks are counted
from the schedule.

**Start value** (:func:`start_value`). A week is worth the best lineup a roster can field in the league's active slots:
a max-weight assignment over slot eligibility (``scipy.optimize.linear_sum_assignment``, DESIGN section 9.1). In a later
week, a slot no rostered player can fill with a positive projection (a bye, an injury, a thin position) is filled from
the wire at replacement level, because a manager streams that slot rather than leave it empty; this week a hole stays
a hole, since the wire's players may have played already or sit on waivers past their games. A rostered starter is
never displaced by a wire player in that baseline: that would assume a move nobody has made. Lineup locks are not
modelled (this week's lineup is valued as if nobody had played yet): they move one week of the rest of the season, and
the lineup optimizer (ROADMAP #20) is what honours them.

**Replacement level** (:class:`Replacement`): for each active slot, the best player actually on the league's wire for
that slot by rest-of-season value (DESIGN section 8.3), from the players the league's latest sync saw unrostered.
:meth:`RosterValuer.vor` is a player's value over the replacement at his own position.

**Rest-of-season value** sums a week's value over the periods left in the fantasy season (:class:`Horizon`), with the
league's playoff periods (read from its settings) weighted by ``playoff_weight``, a model parameter with the default
:data:`DEFAULT_PLAYOFF_WEIGHT`. :meth:`RosterValuer.gain` is the change in a roster's rest-of-season value from an add,
a drop or both: what the waiver decision (:mod:`fm.decide.waivers`) ranks.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Final

import numpy as np
from scipy.optimize import linear_sum_assignment

from fm.espn.settings import LeagueSettings
from fm.model.availability import assess
from fm.model.projections import ESPN, BlendWeights, blend, position_for
from fm.model.scoring import Scorer
from fm.sports.base import FREE_AGENT_TEAM, ScheduleLike, plugin_for
from fm.store import LeagueRow, PlayerRow, RosterEntryRow, Store

DEFAULT_PLAYOFF_WEIGHT: Final = 1.5
"""How much a fantasy playoff week counts against a regular-season week. Playoff weeks decide the title but only count
for a team that gets there, so they weigh more without dwarfing the weeks that get it there; a starting point for the
backtest (ROADMAP #33) to tune."""
DEFAULT_FILL_DEPTH: Final = 1
"""Wire players kept per slot beyond the slot's count in the fill pool, so a week with two holes a position's best
wire player could fill still finds two distinct players."""
GAMES_STAT: Final = "GP"
"""The games-played stat (``ffl`` 210, ``fba`` 42): in ESPN's season projection, the games it projects."""
POINT_UNIT: Final = 1000
"""The lineup solver works in thousandths of a point, so its lexicographic objective is exact integer arithmetic."""

# ``PlayerOutlook.basis``: where a player's weeks after the current one come from.
BASIS_SEASON: Final = "season"
BASIS_SEASON_WITHOUT_GAMES: Final = "season_without_games"
BASIS_THIS_WEEK: Final = "this_week"
BASIS_NONE: Final = "none"


class ValuationError(ValueError):
    """A league cannot be valued as asked: not synced, not a points league, or no current scoring period."""


def _require_aware(at: datetime, what: str = "now") -> None:
    if at.tzinfo is None or at.utcoffset() is None:
        raise ValueError(f"{what} must be an aware datetime, got a naive {at.isoformat()}")


# --- the rest of the season -------------------------------------------------------------------------------------------


def last_scoring_period(settings: LeagueSettings) -> int | None:
    """The fantasy season's last scoring period: the last one a matchup lists (when matchups are listed in scoring
    periods) or ESPN's ``finalScoringPeriod``, whichever is earlier; ``None`` when neither is known."""
    known: list[int] = []
    if settings.schedule.lists_scoring_periods:
        listed = [period for periods in settings.schedule.matchup_periods.values() for period in periods]
        if listed:
            known.append(max(listed))
    if settings.final_scoring_period is not None:
        known.append(settings.final_scoring_period)
    return min(known) if known else None


@dataclass(frozen=True, slots=True)
class Horizon:
    """The scoring periods left in a fantasy season, from ``current`` to the last, with each one's weight in a
    rest-of-season value: 1 for a regular-season period, ``playoff_weight`` for a playoff one. ``playoffs_known`` is
    false when the league's matchups are not listed in scoring periods, so no period could be weighted up."""

    current: int
    periods: tuple[int, ...]
    weights: Mapping[int, float]
    playoff_periods: frozenset[int] = frozenset()
    playoffs_known: bool = True
    playoff_weight: float = DEFAULT_PLAYOFF_WEIGHT

    @classmethod
    def for_league(
        cls,
        settings: LeagueSettings,
        *,
        current: int | None = None,
        playoff_weight: float = DEFAULT_PLAYOFF_WEIGHT,
    ) -> Horizon:
        """The league's remaining periods from ``current`` (default: ESPN's current scoring period), playoff periods
        from its settings. Raises :class:`ValuationError` when the current or last period is unknown and
        ``ValueError`` for a negative or non-finite ``playoff_weight``. Past the last period the horizon is empty."""
        if not math.isfinite(playoff_weight) or playoff_weight < 0:
            raise ValueError(f"playoff_weight must be a finite number >= 0, got {playoff_weight!r}")
        start = current if current is not None else settings.current_scoring_period
        if start is None:
            raise ValuationError(f"league {settings.league_id}: the current scoring period is unknown; run fm sync")
        end = last_scoring_period(settings)
        if end is None:
            raise ValuationError(f"league {settings.league_id}: the season's last scoring period is unknown")
        listed = settings.schedule.playoff_scoring_periods
        playoffs = frozenset(listed or ())
        periods = tuple(range(start, end + 1))
        weights = {period: playoff_weight if period in playoffs else 1.0 for period in periods}
        return cls(start, periods, MappingProxyType(weights), playoffs, listed is not None, playoff_weight)

    def weight(self, period: int) -> float:
        """A period's weight; 0 outside the horizon."""
        return self.weights.get(period, 0.0)

    @property
    def future(self) -> tuple[int, ...]:
        """The periods after the current one."""
        return tuple(period for period in self.periods if period > self.current)

    def since(self, start: int | None = None) -> tuple[int, ...]:
        """The periods from ``start`` on (all of them when ``start`` is ``None``)."""
        return self.periods if start is None else tuple(period for period in self.periods if period >= start)


# --- one player's weeks -----------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlayerOutlook:
    """One player's expected points in each period of a :class:`Horizon`, in one league's scoring.

    ``slots`` are the active lineup slots he may fill. ``per_game`` is his rest-of-season points per game, ``games`` the
    games ESPN's season line projects (``None`` when it has no ``GP``), ``health`` the share of his team's remaining
    games that is, and ``basis`` where the later weeks came from (:data:`BASIS_SEASON` and friends). ``p_active``
    applies to the current period only.
    """

    espn_id: int
    name: str
    position: str | None
    pro_team_id: int | None
    slots: frozenset[int]
    weekly: Mapping[int, float]
    per_game: float = 0.0
    games: float | None = None
    health: float = 1.0
    p_active: float = 1.0
    basis: str = BASIS_NONE

    def expected(self, period: int) -> float:
        """Expected points in a period; 0 outside the horizon."""
        return self.weekly.get(period, 0.0)

    def value(self, horizon: Horizon, *, start: int | None = None) -> float:
        """Rest-of-season value: weighted expected points over the horizon's periods from ``start`` on."""
        return math.fsum(horizon.weight(period) * self.expected(period) for period in horizon.since(start))

    @property
    def has_projection(self) -> bool:
        """False when nothing projects him at all: his value is unknown, not zero."""
        return self.basis != BASIS_NONE


class _Games:
    """Games per pro team and period from a pro schedule, cached. The schedule covers the periods from its first to its
    last with games; inside that span a period without games has none (a bye week, an All-Star break day), and a
    period outside it, or every period without a schedule, counts as one game: the weekly sport's usual week."""

    def __init__(self, schedule: ScheduleLike | None) -> None:
        self.schedule = schedule
        periods = tuple(schedule.scoring_periods) if schedule is not None else ()
        self.first = min(periods) if periods else None
        self.last = max(periods) if periods else None
        self._counts: dict[tuple[int, int], int] = {}

    def covers(self, period: int) -> bool:
        return self.first is not None and self.last is not None and self.first <= period <= self.last

    def count(self, team_id: int, period: int) -> int:
        if self.schedule is None or not self.covers(period):
            return 1
        key = (team_id, period)
        if key not in self._counts:
            self._counts[key] = len(self.schedule.games_for(team_id, period))
        return self._counts[key]

    def remaining(self, team_id: int, horizon: Horizon) -> int:
        """The games ESPN's season line spans: every scheduled game after the current period (past the fantasy season
        too: the line runs to the end of the NFL regular season), plus one per later horizon period the schedule does
        not cover."""
        scheduled = 0
        if self.first is not None and self.last is not None:
            span = range(max(horizon.current + 1, self.first), self.last + 1)
            scheduled = sum(self.count(team_id, period) for period in span)
        return scheduled + sum(1 for period in horizon.future if not self.covers(period))


def _rate(
    season_line: Mapping[str, float] | None,
    this_week: float | None,
    remaining: int,
    scorer: Scorer,
    position: str | None,
) -> tuple[float, float | None, float, str]:
    """(points per game, ESPN's projected games, health, basis) for the weeks after the current one."""
    if season_line is not None:
        total = scorer.points(season_line, position=position)
        played = season_line.get(GAMES_STAT)
        if played is None:
            # No games count: spread the line over the scheduled games it spans.
            return (total / remaining if remaining > 0 else 0.0), None, 1.0, BASIS_SEASON_WITHOUT_GAMES
        if played <= 0:
            return 0.0, float(played), 0.0, BASIS_SEASON
        health = min(1.0, played / remaining) if remaining > 0 else 1.0
        return total / played, float(played), health, BASIS_SEASON
    if this_week is not None and this_week > 0:
        return this_week, None, 1.0, BASIS_THIS_WEEK
    # Nothing beyond an empty week (a bye, an injury): the rest of his season is unknown, not zero.
    return 0.0, None, 0.0, BASIS_NONE


def _outlook(
    player: PlayerRow,
    *,
    horizon: Horizon,
    scorer: Scorer,
    slots: frozenset[int],
    current_line: Mapping[str, float] | None,
    season_line: Mapping[str, float] | None,
    p_active: float,
    games: _Games,
) -> PlayerOutlook:
    if not 0.0 <= p_active <= 1.0:
        raise ValueError(f"p_active must be within [0, 1], got {p_active!r}")
    position = position_for(player.sport, player.espn_id, {player.espn_id: player.position})
    team = player.pro_team_id if player.pro_team_id not in (None, FREE_AGENT_TEAM) else None
    this_week = scorer.points(current_line, position=position) if current_line is not None else None
    remaining = games.remaining(team, horizon) if team is not None else 0
    per_game, played, health, basis = _rate(season_line, this_week, remaining, scorer, position)
    weekly: dict[int, float] = {}
    for period in horizon.periods:
        if team is None or not player.active:
            weekly[period] = 0.0
        elif period == horizon.current:
            base = this_week if this_week is not None else per_game * games.count(team, period)
            weekly[period] = p_active * base
        else:
            weekly[period] = per_game * health * games.count(team, period)
    return PlayerOutlook(
        espn_id=player.espn_id,
        name=player.full_name,
        position=position,
        pro_team_id=team,
        slots=slots,
        weekly=MappingProxyType(weekly),
        per_game=per_game,
        games=played,
        health=health,
        p_active=p_active,
        basis=basis,
    )


def project_outlook(
    player: PlayerRow,
    *,
    horizon: Horizon,
    scorer: Scorer,
    slots: frozenset[int],
    current_line: Mapping[str, float] | None = None,
    season_line: Mapping[str, float] | None = None,
    p_active: float = 1.0,
    schedule: ScheduleLike | None = None,
) -> PlayerOutlook:
    """A player's expected points per period of ``horizon`` in the league ``scorer`` scores.

    The current period is ``p_active`` times ``current_line`` (this week's projection, blended or single-source). Each
    later period is ESPN's ``season_line`` per game (its points over its ``GP``) times ``health`` (``GP`` over the
    games his team has left after the current period, at most 1) times his team's games that period: none on a bye,
    from ``schedule``; a period it does not cover, or every period without one, counts one game. A season line without
    ``GP`` is spread over the remaining games instead; without a season line, a positive projection this week stands in
    for every week, and with neither his later weeks are unknown (:data:`BASIS_NONE`, never a reason to drop him). A
    player with no pro team, or not on an active pro roster, is worth 0 every week. Raises ``ValueError`` for a
    ``p_active`` outside ``[0, 1]`` and :class:`fm.model.scoring.ScoringError` for a bad line.
    """
    return _outlook(
        player,
        horizon=horizon,
        scorer=scorer,
        slots=slots,
        current_line=current_line,
        season_line=season_line,
        p_active=p_active,
        games=_Games(schedule),
    )


# --- lineup slots -----------------------------------------------------------------------------------------------------


def slot_instances(settings: LeagueSettings) -> tuple[int, ...]:
    """The league's active lineup slots, one entry per slot by slot id (QB, RB, RB, WR, WR, TE, D/ST, K, FLEX is
    ``(0, 2, 2, 4, 4, 6, 16, 17, 23)``): what a week's start value fills. Bench and IR do not score."""
    return tuple(slot.slot_id for slot in settings.active_slots for _ in range(slot.count))


def eligible_active_slots(player: PlayerRow, settings: LeagueSettings) -> frozenset[int]:
    """The league's active slots a player may fill: ESPN's ``eligibleSlots`` for him when stored, else the sport's rule
    for his position (:meth:`fm.sports.base.SportPlugin.eligible_slots`); empty for an unknown position."""
    active = {slot.slot_id for slot in settings.active_slots}
    if player.eligible_slot_ids:
        return frozenset(active.intersection(player.eligible_slot_ids))
    position = position_for(player.sport, player.espn_id, {player.espn_id: player.position})
    if position is None:
        return frozenset()
    try:
        allowed = plugin_for(settings.game).eligible_slots(position, include_reserve=False)
    except KeyError:
        return frozenset()
    return frozenset(active & allowed)


# --- one week's lineup ------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StartValue:
    """What a roster fields in one period: ``points`` from rostered starters, ``filled`` from wire players in slots no
    rostered player could fill, and who went where (``(slot id, ESPN id)`` pairs)."""

    points: float
    filled: float
    starters: tuple[tuple[int, int], ...] = ()
    fills: tuple[tuple[int, int], ...] = ()

    @property
    def total(self) -> float:
        return self.points + self.filled


def _units(value: float) -> int:
    return round(value * POINT_UNIT)


def _assign(
    roster: np.ndarray, roster_masks: np.ndarray, fill: np.ndarray, fill_masks: np.ndarray
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Assign rostered players (values ``roster`` in point units, eligibility ``roster_masks``) and fill players to
    slot columns: the rostered players' total is maximised first, then the fill's, as one exact assignment (every
    rostered value is scaled past the whole fill's sum). Returns the (row, column) pairs used by each kind."""
    if roster.size + fill.size == 0:
        return [], []
    scale = float(fill.sum()) + 1.0
    matrix = np.vstack((np.where(roster_masks, roster[:, None] * scale, 0.0), np.where(fill_masks, fill[:, None], 0.0)))
    rows, columns = linear_sum_assignment(matrix, maximize=True)
    count = roster.size
    starters: list[tuple[int, int]] = []
    fills: list[tuple[int, int]] = []
    for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
        if matrix[row, column] <= 0:
            continue  # an empty slot, or a player put where he cannot play (worth nothing)
        if row < count:
            starters.append((row, column))
        else:
            fills.append((row - count, column))
    return starters, fills


def _mask(outlook: PlayerOutlook, slots: Sequence[int]) -> np.ndarray:
    return np.fromiter((slot in outlook.slots for slot in slots), dtype=bool, count=len(slots))


def _stack(masks: Sequence[np.ndarray], width: int) -> np.ndarray:
    return np.vstack(masks) if masks else np.zeros((0, width), dtype=bool)


def start_value(
    players: Iterable[PlayerOutlook],
    slots: Sequence[int],
    period: int,
    *,
    fill: Iterable[PlayerOutlook] = (),
) -> StartValue:
    """The best lineup ``players`` field in ``slots`` (slot instances, :func:`slot_instances`) in ``period``.

    Only positive expected points start; a slot left empty takes the best eligible ``fill`` player (the wire at
    replacement level) not on the roster, each at most once. Rostered players always come first: a fill player never
    displaces one, whatever their projections.
    """
    roster = [outlook for outlook in players if _units(outlook.expected(period)) > 0]
    taken = {outlook.espn_id for outlook in roster}
    extra = [outlook for outlook in fill if outlook.espn_id not in taken and _units(outlook.expected(period)) > 0]
    width = len(slots)
    if not width:
        return StartValue(0.0, 0.0)
    starters, fills = _assign(
        np.array([_units(outlook.expected(period)) for outlook in roster], dtype=float),
        _stack([_mask(outlook, slots) for outlook in roster], width),
        np.array([_units(outlook.expected(period)) for outlook in extra], dtype=float),
        _stack([_mask(outlook, slots) for outlook in extra], width),
    )
    return StartValue(
        points=math.fsum(roster[row].expected(period) for row, _ in starters),
        filled=math.fsum(extra[row].expected(period) for row, _ in fills),
        starters=tuple(sorted((slots[column], roster[row].espn_id) for row, column in starters)),
        fills=tuple(sorted((slots[column], extra[row].espn_id) for row, column in fills)),
    )


# --- replacement level and roster value -------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Replacement:
    """Replacement level at one active slot: the best wire player eligible for it by rest-of-season ``value`` (0, with
    no player, when nobody on the wire can fill it)."""

    slot_id: int
    label: str
    espn_id: int | None
    name: str | None
    value: float


class RosterValuer:
    """Rest-of-season value of one team's roster and how a move changes it.

    ``outlooks`` must cover ``roster``, ``wire`` and every pending add (all ESPN ids). ``wire`` less the roster is the
    league's wire, the source of the replacement levels and of the fill pool: for each active slot, its
    ``count + depth`` best eligible wire players by rest-of-season value. ``slots`` are the active slot instances;
    ``labels`` name them in :attr:`replacement`. ``pending`` are moves already planned, ``(add, drop, start)`` each,
    which change the roster from period ``start`` on (a claim processed next week leaves this week's lineup alone):
    :meth:`roster_at` is the roster in a period, :attr:`roster` the one once every move has gone through. Week values
    are cached, so ranking every (add, drop) pair stays cheap.
    """

    def __init__(
        self,
        outlooks: Mapping[int, PlayerOutlook],
        *,
        roster: Iterable[int],
        wire: Iterable[int],
        slots: Sequence[int],
        horizon: Horizon,
        labels: Mapping[int, str] | None = None,
        depth: int = DEFAULT_FILL_DEPTH,
        pending: Iterable[tuple[int | None, int | None, int]] = (),
    ) -> None:
        if depth < 0:
            raise ValueError(f"depth must be >= 0, got {depth!r}")
        self.outlooks: Mapping[int, PlayerOutlook] = MappingProxyType(dict(outlooks))
        self.base = frozenset(roster)
        self.pending = tuple(pending)
        self.roster = _apply(self.base, self.pending)
        added = {add for add, _, _ in self.pending if add is not None}
        self.wire = frozenset(wire) - self.roster - added
        missing = sorted((self.base | self.roster | self.wire) - self.outlooks.keys())
        if missing:
            raise ValueError(f"no outlook for ESPN ids {missing}")
        self.slots = tuple(slots)
        self.horizon = horizon
        self.depth = depth
        self.labels: Mapping[int, str] = MappingProxyType(dict(labels or {}))
        known = self.base | self.roster | self.wire
        self._ros = {espn_id: self.outlooks[espn_id].value(horizon) for espn_id in known}
        self._counts = Counter(self.slots)
        self._ranked = {slot: self._best_for(slot) for slot in self._counts}
        self.replacement: Mapping[int, Replacement] = MappingProxyType(
            {slot: self._replacement(slot) for slot in self._counts}
        )
        self._masks: dict[int, np.ndarray] = {}
        self._pools: dict[frozenset[int], tuple[int, ...]] = {}
        self._weeks: dict[tuple[frozenset[int], tuple[int, ...], int], float] = {}

    def roster_at(self, period: int) -> frozenset[int]:
        """The roster in ``period``: the base roster with the pending moves that count by then."""
        return _apply(self.base, self.pending, period) if self.pending else self.base

    def _best_for(self, slot: int) -> tuple[int, ...]:
        eligible = (espn_id for espn_id in self.wire if slot in self.outlooks[espn_id].slots)
        return tuple(sorted(eligible, key=lambda espn_id: (-self._ros[espn_id], espn_id)))

    def _replacement(self, slot: int) -> Replacement:
        label = self.labels.get(slot, f"SLOT_{slot}")
        best = self._ranked[slot]
        if not best:
            return Replacement(slot, label, None, None, 0.0)
        outlook = self.outlooks[best[0]]
        return Replacement(slot, label, outlook.espn_id, outlook.name, self._ros[best[0]])

    def ros(self, espn_id: int, *, start: int | None = None) -> float:
        """A player's own rest-of-season value (from ``start``, default the whole horizon)."""
        if start is None and espn_id in self._ros:
            return self._ros[espn_id]
        return self.outlooks[espn_id].value(self.horizon, start=start)

    def vor(self, espn_id: int) -> float | None:
        """Value over replacement: his rest-of-season value less the replacement level at his own position, the lowest
        among the slots he may fill (a running back's RB slot, not FLEX). ``None`` when he fits no active slot."""
        levels = [self.replacement[slot].value for slot in self.outlooks[espn_id].slots if slot in self.replacement]
        return self.ros(espn_id) - min(levels) if levels else None

    def fill_pool(self, exclude: Iterable[int] = ()) -> tuple[int, ...]:
        """The wire players that fill empty slots, less ``exclude`` (a player a move would claim)."""
        excluded = frozenset(exclude)
        cached = self._pools.get(excluded)
        if cached is not None:
            return cached
        chosen: dict[int, None] = {}
        for slot, count in self._counts.items():
            picked = [espn_id for espn_id in self._ranked[slot] if espn_id not in excluded][: count + self.depth]
            chosen.update(dict.fromkeys(picked))
        pool = tuple(sorted(chosen))
        self._pools[excluded] = pool
        return pool

    def _mask_for(self, espn_id: int) -> np.ndarray:
        mask = self._masks.get(espn_id)
        if mask is None:
            mask = self._masks[espn_id] = _mask(self.outlooks[espn_id], self.slots)
        return mask

    def week(self, roster: frozenset[int], period: int, pool: tuple[int, ...]) -> float:
        """The start value (rostered points plus fill) of ``roster`` in ``period`` with ``pool`` filling holes after the
        current period (this week nothing fills a hole; see the module docs)."""
        if period == self.horizon.current:
            pool = ()
        key = (roster, pool, period)
        cached = self._weeks.get(key)
        if cached is not None:
            return cached
        members = [espn_id for espn_id in sorted(roster) if _units(self.outlooks[espn_id].expected(period)) > 0]
        extra = [
            espn_id for espn_id in pool if espn_id not in roster and _units(self.outlooks[espn_id].expected(period)) > 0
        ]
        width = len(self.slots)
        starters, fills = _assign(
            np.array([_units(self.outlooks[espn_id].expected(period)) for espn_id in members], dtype=float),
            _stack([self._mask_for(espn_id) for espn_id in members], width),
            np.array([_units(self.outlooks[espn_id].expected(period)) for espn_id in extra], dtype=float),
            _stack([self._mask_for(espn_id) for espn_id in extra], width),
        )
        value = math.fsum(self.outlooks[members[row]].expected(period) for row, _ in starters)
        value += math.fsum(self.outlooks[extra[row]].expected(period) for row, _ in fills)
        self._weeks[key] = value
        return value

    def lineup(self, period: int, roster: Iterable[int] | None = None) -> StartValue:
        """The lineup behind a week's value, with who starts and who fills in (for output)."""
        members = self.roster_at(period) if roster is None else frozenset(roster)
        filling = self.fill_pool(members) if period != self.horizon.current else ()
        pool = [self.outlooks[espn_id] for espn_id in filling]
        return start_value((self.outlooks[espn_id] for espn_id in sorted(members)), self.slots, period, fill=pool)

    def value(
        self, roster: Iterable[int] | None = None, *, start: int | None = None, exclude: Iterable[int] = ()
    ) -> float:
        """Rest-of-season value from ``start``: each week's start value, weighted, of ``roster`` (default: the team's
        roster in each week, pending moves included); ``exclude`` keeps players out of the fill pool."""
        fixed = None if roster is None else frozenset(roster)
        pool = self.fill_pool(exclude)
        weeks = (
            self.horizon.weight(period) * self.week(self.roster_at(period) if fixed is None else fixed, period, pool)
            for period in self.horizon.since(start)
        )
        return math.fsum(weeks)

    def gain(
        self,
        add: int | None = None,
        drop: int | None = None,
        *,
        start: int | None = None,
        drop_start: int | None = None,
    ) -> float:
        """The change in the roster's rest-of-season value from adding ``add`` (a wire player) from period ``start`` on
        and dropping ``drop`` (a rostered one) from period ``drop_start`` on (default: ``start``; both default to the
        whole horizon). A drop can take effect before the add counts: a free agent whose game has started plays for us
        from next week, but the player dropped for him is gone at once. The claimed player leaves the fill pool on
        both sides, so a wire player is worth his edge over the next-best one, not over himself. Raises
        ``ValueError`` for an add not on the wire or a drop not on the roster."""
        if add is not None and add not in self.wire:
            raise ValueError(f"ESPN {add} is not on the wire")
        if drop is not None and drop not in self.roster:
            raise ValueError(f"ESPN {drop} is not on the roster")
        if add is None and drop is None:
            return 0.0
        first = self._first(start)
        gone = self._first(start if drop_start is None else drop_start)
        pool = self.fill_pool(() if add is None else (add,))
        changes: list[float] = []
        for period in self.horizon.since(min(first, gone)):
            before = self.roster_at(period)
            after = _apply(before, ((None, drop, gone), (add, None, first)), period)
            change = self.week(after, period, pool) - self.week(before, period, pool)
            changes.append(self.horizon.weight(period) * change)
        return math.fsum(changes)

    def after(
        self,
        add: int | None = None,
        drop: int | None = None,
        *,
        start: int | None = None,
        drop_start: int | None = None,
    ) -> RosterValuer:
        """The valuer once a move is planned: ``add`` leaves the wire and joins the roster from period ``start``, and
        ``drop`` leaves it from ``drop_start`` (default: ``start``; both default to the current period)."""
        first = self._first(start)
        gone = self._first(start if drop_start is None else drop_start)
        planned = ((add, drop, first),) if first == gone else ((None, drop, gone), (add, None, first))
        return RosterValuer(
            self.outlooks,
            roster=self.base,
            wire=self.wire,
            slots=self.slots,
            horizon=self.horizon,
            labels=self.labels,
            depth=self.depth,
            pending=(*self.pending, *planned),
        )

    def _first(self, start: int | None) -> int:
        return self.horizon.current if start is None else start


def _apply(
    roster: frozenset[int], moves: Iterable[tuple[int | None, int | None, int]], period: int | None = None
) -> frozenset[int]:
    """``roster`` with ``moves`` (``(add, drop, start)``) applied in order: those counting by ``period``, or all."""
    members = roster
    for add, drop, start in moves:
        if period is not None and start > period:
            continue
        if drop is not None:
            members = members - {drop}
        if add is not None:
            members = members | {add}
    return members


# --- a league from the store ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class LeagueValuation:
    """Everything one league's valuation reads from the store, for our team: the settings, the horizon from the latest
    roster snapshot's period, an outlook for every player on our roster and on the wire, and the warnings collected
    on the way. :meth:`valuer` values rosters over it."""

    league: LeagueRow
    settings: LeagueSettings
    horizon: Horizon
    outlooks: Mapping[int, PlayerOutlook]
    players: Mapping[int, PlayerRow]
    team: tuple[RosterEntryRow, ...]
    """Our team's roster entries for the period (slot and lock flag included)."""
    rostered: frozenset[int]
    """Every player on any roster in the league for the period."""
    wire: frozenset[int]
    as_of: datetime
    warnings: tuple[str, ...] = ()

    @property
    def scoring_period(self) -> int:
        return self.horizon.current

    @property
    def roster(self) -> frozenset[int]:
        """Our rostered players the valuation could value (a player missing from ``players`` is left out, warned)."""
        return frozenset(entry.espn_id for entry in self.team if entry.espn_id in self.outlooks)

    def valuer(
        self, *, depth: int = DEFAULT_FILL_DEPTH, pending: Iterable[tuple[int | None, int | None, int]] = ()
    ) -> RosterValuer:
        """A :class:`RosterValuer` for our roster and the wire, with ``pending`` moves (``(add, drop, start)``)."""
        labels = {slot.slot_id: slot.label for slot in self.settings.active_slots}
        return RosterValuer(
            self.outlooks,
            roster=self.roster,
            wire=self.wire,
            slots=slot_instances(self.settings),
            horizon=self.horizon,
            labels=labels,
            depth=depth,
            pending=pending,
        )


def league_settings(store: Store, league: LeagueRow) -> LeagueSettings:
    """The league's synced ESPN settings; :class:`ValuationError` before the first ``fm sync``."""
    row = store.settings.get(league.row_id)
    if row is None:
        raise ValuationError(f"league {league.key!r} has no synced settings; run fm sync")
    return LeagueSettings.model_validate(row.settings)


def load_valuation(
    store: Store,
    league: LeagueRow,
    *,
    now: datetime,
    settings: LeagueSettings | None = None,
    wire: Iterable[int] | None = None,
    schedule: ScheduleLike | None = None,
    weights: BlendWeights | None = None,
    playoff_weight: float = DEFAULT_PLAYOFF_WEIGHT,
    include_rostered: bool = False,
) -> LeagueValuation:
    """Outlooks for our roster and the wire of a points league, from what ``fm sync`` stored.

    The period is the latest roster snapshot's. ``wire`` is the ESPN ids on the league's wire (the waiver module passes
    the free-agent pool the sync read); by default it is every player with an ESPN projection for the period who is on
    no roster, which is the pool the sync stored lines for. With ``include_rostered`` every player on any roster in the
    league is valued as well (``LeagueValuation.rostered`` says who they are), for league-wide rankings, the season
    simulator and trade evaluation (ROADMAP #35, #32, #38); our roster, the wire and the replacement levels are the
    same either way. This week's lines are the stored sources blended with ``weights`` (default
    ``data/blend_weights.toml``); later weeks come from ESPN's season lines (period 0). ``p_active`` is
    :func:`fm.model.availability.assess` with ``schedule``, which must cover the period when given, and no news
    signals. Players without a ``players`` row are skipped with a warning. Raises :class:`ValuationError` for a league
    that is not synced, has no roster for our team, or is not a points league.
    """
    _require_aware(now)
    resolved = settings if settings is not None else league_settings(store, league)
    if not resolved.is_points:
        raise ValuationError(
            f"league {league.key!r} is a category league; points valuation does not apply (ROADMAP #24 values them)"
        )
    period = store.rosters.latest_period(league.row_id)
    if period is None:
        raise ValuationError(f"league {league.key!r} has no roster snapshot; run fm sync")
    horizon = Horizon.for_league(resolved, current=period, playoff_weight=playoff_weight)
    sport, season = league.sport, league.season
    entries = store.rosters.league(league.row_id, period)
    rostered = frozenset(entry.espn_id for entry in entries)
    team = tuple(entry for entry in entries if entry.team_id == league.team_id)
    if not team:
        raise ValuationError(
            f"league {league.key!r}: team {league.team_id} has no roster in scoring period {period}; check team_id"
        )
    period_rows = store.projections.for_period(sport, season, period)
    pool = frozenset(wire) if wire is not None else frozenset(row.espn_id for row in period_rows if row.source == ESPN)
    valued = rostered if include_rostered else frozenset(entry.espn_id for entry in team)
    wanted = valued | (pool - rostered)
    players = {player.espn_id: player for player in store.players.many(sport, wanted)}
    warnings: list[str] = []
    missing = sorted(wanted - players.keys())
    if missing:
        warnings.append(f"{league.key}: {len(missing)} players have no players row and were not valued: {missing[:5]}")

    blended = blend(
        (row for row in period_rows if row.espn_id in players),
        weights=weights if weights is not None else BlendWeights.load(),
        positions={espn_id: player.position for espn_id, player in players.items()},
    )
    warnings.extend(blended.warnings)
    current = {row.espn_id: row.stats for row in blended.rows}
    season_lines = {
        row.espn_id: row.stats
        for row in store.projections.for_period(sport, season, 0, source=ESPN)
        if row.espn_id in players
    }
    scorer = Scorer(resolved)
    games = _Games(schedule)
    outlooks: dict[int, PlayerOutlook] = {}
    for espn_id, player in players.items():
        try:
            availability = assess(player, season=season, scoring_period=period, as_of=now, schedule=schedule)
        except ValueError as exc:
            raise ValuationError(f"league {league.key!r}: {exc}") from exc
        outlooks[espn_id] = _outlook(
            player,
            horizon=horizon,
            scorer=scorer,
            slots=eligible_active_slots(player, resolved),
            current_line=current.get(espn_id),
            season_line=season_lines.get(espn_id),
            p_active=availability.p_active,
            games=games,
        )
    if not horizon.playoffs_known:
        warnings.append(f"{league.key}: playoff periods are not listed in scoring periods; none weighted up")
    return LeagueValuation(
        league=league,
        settings=resolved,
        horizon=horizon,
        outlooks=MappingProxyType(outlooks),
        players=MappingProxyType(players),
        team=team,
        rostered=rostered,
        wire=frozenset(pool - rostered) & frozenset(outlooks),
        as_of=now,
        warnings=tuple(warnings),
    )
