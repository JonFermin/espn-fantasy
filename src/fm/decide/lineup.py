"""Lineup decisions: one scoring period's lineup as an assignment problem over slot eligibility (DESIGN section 9.1).

A lineup puts every rostered player in one slot for one scoring period. Each active slot the league uses (QB, RB, WR,
TE, FLEX, D/ST, K, ...: ``LeagueSettings.active_slots``, never a constant) holds up to its count, and a player may only
move into a slot he is eligible for: ESPN's own ``eligibleSlots`` for him where the sync stored them, the sport
plugin's position table otherwise (:meth:`fm.sports.base.SportPlugin.eligible_slots`). Everyone else sits on the bench;
players on IR stay there and nobody is moved onto it (that frees a roster spot, an add/drop matter).
:func:`optimize_lineup` solves the assignment of players to slot openings with ``scipy.optimize.linear_sum_assignment``,
lexicographically:

1. **Never start a zero** (DESIGN section 2). Fill as many active slots as possible with players who will play. A
   player plays when his pro team has a game in the period and ``p_active`` > 0; an OUT, IR or suspended player, a bye
   and a team without a game never do. So an OUT, bye or no-game starter is benched whenever any legal arrangement
   puts a player who plays in his place, and an empty slot counts as a zero starter. When no alternative exists the
   starter stays where he is (:attr:`LineupPlan.idle_starters`): moving him would change nothing.
2. **The objective.** Expected points, ``p_active`` x the projection's league points
   (:func:`fm.model.availability.expected_points`). With a :class:`MatchupOutlook` for the opponent, a lineup's P(win)
   is a normal approximation (:func:`win_probability`). When the matchup is lopsided, the expected-points lineup winning
   with probability at least :data:`LOPSIDED_AT` or at most one minus it, the objective becomes P(win): a heavy
   favorite gives up a little expected value for a steadier lineup and an underdog takes on variance. P(win) is not
   linear in the lineup, so it is maximized through the linear objective ``expected + w * variance``, whose weight
   ``w = -(mean - opponent_mean) / (2 * spread^2)`` is P(win)'s own trade-off between the two at the current lineup.
   The weight is re-derived from each lineup it produces until it settles, a few multiples of it are tried as well,
   and the lineup with the best P(win) wins; the expected-points lineup is always a candidate, so the switch never
   lowers P(win).
3. **Stability.** Among equally good lineups, the one that moves the fewest players.

A player's variance is that of ``B * X``, ``B`` ~ Bernoulli(``p_active``) and ``X`` his projection with the
uncertainty model's standard deviation (:meth:`fm.model.projections.BlendWeights.projection_sd`), players independent:
``p * sd^2 + p * (1 - p) * points^2``, and 0 for a player who does not play.

**Locks** are hard constraints: a locked player stays in his slot, and that slot stays his. A player is locked once
ESPN's own ``lineupLocked`` said so at the last sync or his lock time has passed. Lock times come from
``plugin.locks(period, schedule, lock_type=settings.lineup_lock_type)``: per game in both real leagues
(``INDIVIDUAL_GAME``, where a player without a game never locks) or everyone at the period's first game
(``FIRSTGAME_SCORINGPERIOD``); an ``UNKNOWN`` lock type is refused (:class:`LineupError`), never guessed.

**Proposals** (DESIGN section 11). :func:`plan_lineup` reads one team's roster, players, projections and availability
from the store (:func:`lineup_inputs`) and drafts at most two proposals, both relative to the lineup as the last sync
read it:

- ``bench_inactive`` (policy default ``auto`` at T-15): :func:`bench_inactive_lineup`, the best lineup that keeps
  every starter who plays in the lineup, so it only benches zero starters (and fills empty slots), shifting starters
  between slots where that makes room for a replacement;
- ``lineup`` (default ``approve``): :func:`optimize_lineup`, when it goes further than the first.

The two are alternatives: the ``lineup`` draft fills as many slots with players who play as the ``bench_inactive`` one
does, and once either has executed the other no longer matches the roster. Each move names the slot its player is in
now (``from_slot_id``), so the executor refuses a stale draft rather than apply it to a changed lineup, and the next
run plans again from the new roster. Both drafts of one run carry the same ``engine_numbers["basis"]``. A draft's
deadline is the earliest lock among the players it moves; its ``dedupe_key`` names its kind, period and moves.
:func:`propose_lineup` hands the drafts to :func:`fm.proposals.propose` and is the NFL ``lineup`` decision in
:mod:`fm.decide.registry`. Nothing here writes to ESPN (CLAUDE.md: workers propose, the executor acts).

Projections are stat lines (CLAUDE.md): the stored ESPN and Sleeper lines for the period are blended in memory
(:func:`fm.model.projections.blend`) and scored with the league's own items (:class:`fm.model.scoring.Scorer`), so a
category league is refused here (ROADMAP #31 plans those). ``p_active`` is the stored availability row when it is no
older than the player's synced designation (the full model's, ROADMAP #22), else the designation model's
(:func:`fm.model.availability.assess`); an OUT, IR or suspended designation, or a player off an active pro roster,
is always 0. The opponent's outlook (:func:`team_outlook`) assumes he starts his own expected-points lineup and uses
projections for games already played; the season simulator (ROADMAP #32) can pass a better one.
"""

from __future__ import annotations

import hashlib
import math
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum
from statistics import NormalDist
from types import MappingProxyType
from typing import Any, Final

import numpy as np
from scipy.optimize import linear_sum_assignment

from fm.config import Config
from fm.decide import registry as _registry
from fm.espn.ids import Game, IdMaps, InjuryStatus, ids_for
from fm.espn.settings import LeagueSettings, LockType
from fm.model.availability import INACTIVE_DESIGNATIONS, assess, designation
from fm.model.projections import BlendWeights, blend, position_for
from fm.model.scoring import Scorer
from fm.proposals import LineupMove, LineupPayload, PolicyError, ProposalKind, propose
from fm.proposals.policy import stored_settings
from fm.sports.base import ScheduleLike, SportPlugin, fantasy_day, first_start, last_start, period_turn, plugin_for
from fm.store import AvailabilityRow, LeagueRow, PlayerRow, ProposalRow, Store

DECISION_KIND: Final = "lineup"
"""The kind :func:`propose_lineup` is registered under for NFL in :mod:`fm.decide.registry`."""
CREATED_BY: Final = "decide.lineup"
"""``created_by`` of the proposals this module stores."""
LOPSIDED_AT: Final = 0.75
"""P(win) of the expected-points lineup at or above which (or at or below one minus which) the matchup counts as
lopsided and the objective becomes P(win). A model parameter, not a league setting."""

_KEEP: Final = 1e-6
"""Tie-break bonus, in points, for each player left in his slot: lineups closer than this are equally good."""
_MAX_ROUNDS: Final = 8
"""Re-derivations of the variance weight in the P(win) search; it usually settles in two or three."""
_WEIGHT_SCALES: Final = (0.5, 2.0, 4.0)
"""Multiples of the first variance weight the P(win) search also tries."""
_DESIGNATION_LABELS: Mapping[str, str] = MappingProxyType(
    {
        InjuryStatus.OUT.value: "OUT",
        InjuryStatus.INJURY_RESERVE.value: "IR",
        InjuryStatus.SUSPENSION.value: "suspended",
    }
)


class LineupError(ValueError):
    """The lineup cannot be planned as asked: the league is not synced for the period, it is not a points league, or
    its lock type is unknown. Nothing is proposed."""


class Objective(StrEnum):
    """What a lineup maximizes once it fills every slot it can with a player who plays."""

    EXPECTED_POINTS = "expected_points"
    WIN_PROBABILITY = "win_probability"


# --- the optimizer's inputs -------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LineupCandidate:
    """One rostered player as the optimizer sees him for one scoring period.

    ``slot_id`` is where he sits now (ESPN's ``lineupSlotId``) and ``eligible`` the active slots he may move into (bench
    and IR are implied); staying where he is is always allowed. ``points`` and ``sd`` are his projection in league
    points if he plays, ``p_active`` the probability he does. ``lock_at`` is when he locks this period (``None``: never,
    like a bye under per-game locks) and ``locked`` whether he already has.
    """

    espn_id: int
    slot_id: int
    eligible: frozenset[int] = frozenset()
    points: float = 0.0
    sd: float = 0.0
    p_active: float = 1.0
    has_game: bool = True
    locked: bool = False
    lock_at: datetime | None = None
    position: str | None = None
    name: str = ""
    designation: str | None = None

    def __post_init__(self) -> None:
        if not 0.0 <= self.p_active <= 1.0:
            raise ValueError(f"player {self.espn_id}: p_active must be within [0, 1], got {self.p_active!r}")
        if not (math.isfinite(self.points) and math.isfinite(self.sd)) or self.sd < 0:
            raise ValueError(
                f"player {self.espn_id}: points and sd must be finite and sd >= 0, got {self.points!r} and {self.sd!r}"
            )
        if self.lock_at is not None and (self.lock_at.tzinfo is None or self.lock_at.utcoffset() is None):
            raise ValueError(f"player {self.espn_id}: lock_at must be an aware datetime")

    @property
    def plays(self) -> bool:
        """His team has a game and he may be active: a starter who does not play is a zero."""
        return self.has_game and self.p_active > 0.0

    @property
    def expected(self) -> float:
        """Expected league points in the lineup: ``p_active * points``, 0 when he does not play."""
        return self.p_active * self.points if self.plays else 0.0

    @property
    def variance(self) -> float:
        """Variance of his score in the lineup: ``p * sd^2 + p * (1 - p) * points^2``, 0 when he does not play."""
        if not self.plays:
            return 0.0
        p = self.p_active
        return p * self.sd**2 + p * (1.0 - p) * self.points**2

    @property
    def label(self) -> str:
        return self.name or f"ESPN {self.espn_id}"


@dataclass(frozen=True, slots=True)
class MatchupOutlook:
    """The opponent's score for the period as a normal distribution, in league points."""

    opponent_mean: float
    opponent_sd: float = 0.0

    def __post_init__(self) -> None:
        if not (math.isfinite(self.opponent_mean) and math.isfinite(self.opponent_sd)) or self.opponent_sd < 0:
            raise ValueError(
                f"opponent mean and sd must be finite and sd >= 0, got {self.opponent_mean!r} and {self.opponent_sd!r}"
            )


def win_probability(mean: float, sd: float, outlook: MatchupOutlook) -> float:
    """P(our score beats the opponent's) when ours is N(``mean``, ``sd``^2) and his the outlook's, independently:
    ``Phi((mean - opponent_mean) / sqrt(sd^2 + opponent_sd^2))``. With no uncertainty on either side it is 1, 0, or
    0.5 for a dead heat."""
    spread = math.hypot(sd, outlook.opponent_sd)
    margin = mean - outlook.opponent_mean
    if spread == 0.0:
        return 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5
    return NormalDist().cdf(margin / spread)


# --- plans ------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LineupPlan:
    """A lineup for one scoring period and what it is worth.

    ``slots`` places every player (ESPN id -> slot id) and ``moves`` take the current lineup there, listing every player
    whose slot changes, so both sides of each swap are in. ``expected`` and ``sd`` describe the starters' total;
    ``win_probability`` is set when a :class:`MatchupOutlook` was given. ``objective`` is what chose the lineup
    (``None`` for a lineup scored as it stands) and ``variance_weight`` the weight on variance it was found with (0 for
    expected points). ``open_slots`` lists each active opening left empty (a slot id per opening) and ``idle_starters``
    the starters who will not play, both because no legal alternative exists (or a lock holds them).
    """

    slots: Mapping[int, int]
    starters: tuple[int, ...]
    moves: tuple[LineupMove, ...]
    expected: float
    sd: float
    objective: Objective | None = None
    win_probability: float | None = None
    variance_weight: float = 0.0
    open_slots: tuple[int, ...] = ()
    idle_starters: tuple[int, ...] = ()

    def payload(self) -> LineupPayload | None:
        """The moves as a :class:`fm.proposals.LineupPayload`, or ``None`` when the lineup stays as it is."""
        return LineupPayload(moves=self.moves) if self.moves else None


@dataclass(frozen=True)
class _Problem:
    """The assignment behind a plan: who may move (``free``: neither locked nor on IR), how many openings each active
    slot has left for them (``capacity``), and where each of them may go (``eligible``: openings only)."""

    players: tuple[LineupCandidate, ...]
    slot_counts: Mapping[int, int]
    bench: int
    ir: int
    free: tuple[LineupCandidate, ...]
    capacity: Mapping[int, int]
    eligible: Mapping[int, frozenset[int]]

    def is_active(self, slot_id: int) -> bool:
        return slot_id not in (self.bench, self.ir)

    def is_starter(self, player: LineupCandidate) -> bool:
        return self.is_active(player.slot_id)


def _problem(players: Iterable[LineupCandidate], slot_counts: Mapping[int, int], sport: Game | str) -> _Problem:
    ids = ids_for(sport)
    roster = tuple(players)
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

    def stays(player: LineupCandidate) -> bool:
        return player.locked or player.slot_id == ids.ir_slot

    held = Counter(player.slot_id for player in roster if stays(player))
    capacity = {slot_id: max(0, count - held[slot_id]) for slot_id, count in counts.items()}
    free = tuple(player for player in roster if not stays(player))
    eligible = {
        player.espn_id: frozenset(slot for slot in player.eligible | {player.slot_id} if capacity.get(slot, 0) > 0)
        for player in free
    }
    return _Problem(
        players=roster,
        slot_counts=MappingProxyType(counts),
        bench=ids.bench_slot,
        ir=ids.ir_slot,
        free=free,
        capacity=MappingProxyType(capacity),
        eligible=MappingProxyType(eligible),
    )


def _assign(problem: _Problem, values: Mapping[int, float], *, keep_starters: bool) -> dict[int, int]:
    """The best slot for every free player: openings filled by players who play first, then the most ``values``, then
    the fewest moves. With ``keep_starters`` a starter who plays is never benched (the ``bench_inactive`` lineup)."""
    if not problem.free:
        return {}
    columns = [slot_id for slot_id, count in sorted(problem.capacity.items()) for _ in range(count)]
    openings = len(columns)
    columns.extend([problem.bench] * len(problem.free))
    # Weights dominate lexicographically: ``fill`` beats any difference in values and stay bonuses, ``stay`` beats any
    # number of fills.
    fill = 1.0 + math.fsum(abs(values[player.espn_id]) for player in problem.free) + _KEEP * (len(problem.free) + 1)
    stay = (openings + 2) * fill
    weights = np.full((len(problem.free), len(columns)), -np.inf)
    for row, player in enumerate(problem.free):
        value = values[player.espn_id] + (fill if player.plays else 0.0)
        for column, slot_id in enumerate(columns[:openings]):
            if slot_id in problem.eligible[player.espn_id]:
                weights[row, column] = value + (_KEEP if slot_id == player.slot_id else 0.0)
        benched = _KEEP if player.slot_id == problem.bench else 0.0
        if keep_starters and player.plays and problem.is_starter(player):
            benched -= stay
        weights[row, openings:] = benched
    rows, chosen = linear_sum_assignment(weights, maximize=True)
    return {
        problem.free[row].espn_id: columns[column] for row, column in zip(rows.tolist(), chosen.tolist(), strict=True)
    }


def _slots(problem: _Problem, assigned: Mapping[int, int]) -> dict[int, int]:
    return {player.espn_id: assigned.get(player.espn_id, player.slot_id) for player in problem.players}


def _totals(problem: _Problem, slots: Mapping[int, int]) -> tuple[float, float]:
    """Expected points and variance of the starters' total."""
    starters = [player for player in problem.players if problem.is_active(slots[player.espn_id])]
    return math.fsum(player.expected for player in starters), math.fsum(player.variance for player in starters)


def _values(problem: _Problem, variance_weight: float) -> dict[int, float]:
    return {player.espn_id: player.expected + variance_weight * player.variance for player in problem.free}


def _variance_weight(mean: float, variance: float, outlook: MatchupOutlook) -> float:
    """P(win)'s trade-off between variance and mean at a lineup: ``-(mean - opponent_mean) / (2 * spread^2)``,
    negative for a favorite (variance costs) and positive for an underdog (variance helps)."""
    spread_squared = variance + outlook.opponent_sd**2
    if spread_squared <= 0.0:
        return 0.0
    return -(mean - outlook.opponent_mean) / (2.0 * spread_squared)


def _move_order(problem: _Problem, move: LineupMove) -> tuple[int, int]:
    """Players leaving the lineup first, then shifts between slots, then players entering it."""
    leaving = problem.is_active(move.from_slot_id) and not problem.is_active(move.to_slot_id)
    entering = not problem.is_active(move.from_slot_id) and problem.is_active(move.to_slot_id)
    return (0 if leaving else 2 if entering else 1, move.espn_id)


def _plan(
    problem: _Problem,
    slots: Mapping[int, int],
    *,
    objective: Objective | None,
    outlook: MatchupOutlook | None,
    variance_weight: float = 0.0,
) -> LineupPlan:
    mean, variance = _totals(problem, slots)
    sd = math.sqrt(variance)
    moves = sorted(
        (
            LineupMove(espn_id=player.espn_id, from_slot_id=player.slot_id, to_slot_id=slots[player.espn_id])
            for player in problem.players
            if slots[player.espn_id] != player.slot_id
        ),
        key=lambda move: _move_order(problem, move),
    )
    held = Counter(slots.values())
    return LineupPlan(
        slots=MappingProxyType(dict(slots)),
        starters=tuple(player.espn_id for player in problem.players if problem.is_active(slots[player.espn_id])),
        moves=tuple(moves),
        expected=mean,
        sd=sd,
        objective=objective,
        win_probability=None if outlook is None else win_probability(mean, sd, outlook),
        variance_weight=variance_weight,
        open_slots=tuple(
            slot_id for slot_id, count in sorted(problem.slot_counts.items()) for _ in range(count - held[slot_id])
        ),
        idle_starters=tuple(
            player.espn_id
            for player in problem.players
            if problem.is_active(slots[player.espn_id]) and not player.plays
        ),
    )


def _check_legal(problem: _Problem, slots: Mapping[int, int]) -> None:
    """Refuse a plan that breaks a rule (a bug, never an input): it would be proposed to a real league."""
    for player in problem.players:
        slot_id = slots[player.espn_id]
        if slot_id == player.slot_id:
            continue
        if player.locked or player.slot_id == problem.ir or slot_id == problem.ir:
            raise LineupError(f"refusing a plan that moves {player.label} from {player.slot_id} to {slot_id}")
        if problem.is_active(slot_id) and slot_id not in problem.eligible[player.espn_id]:
            raise LineupError(f"refusing a plan that puts {player.label} in slot {slot_id}, which he cannot fill")
    used = Counter(slots[player.espn_id] for player in problem.free if problem.is_active(slots[player.espn_id]))
    for slot_id, count in used.items():
        if count > problem.capacity.get(slot_id, 0):
            raise LineupError(f"refusing a plan that overfills slot {slot_id}")


def _choose(
    problem: _Problem,
    slots: Mapping[int, int],
    outlook: MatchupOutlook | None,
    objective: Objective | None,
    lopsided_at: float,
) -> Objective:
    if objective is not None:
        return objective
    if outlook is None:
        return Objective.EXPECTED_POINTS
    mean, variance = _totals(problem, slots)
    chance = win_probability(mean, math.sqrt(variance), outlook)
    lopsided = chance >= lopsided_at or chance <= 1.0 - lopsided_at
    return Objective.WIN_PROBABILITY if lopsided else Objective.EXPECTED_POINTS


def _search_win(
    problem: _Problem, outlook: MatchupOutlook, start: dict[int, int], *, keep_starters: bool
) -> tuple[dict[int, int], float]:
    """The lineup with the best P(win) among those ``expected + w * variance`` picks for the weights tried (see the
    module docstring), and its weight; ties go to more expected points, then fewer moves, then the start."""

    def key(slots: Mapping[int, int]) -> tuple[float, float, int]:
        mean, variance = _totals(problem, slots)
        moved = sum(1 for player in problem.players if slots[player.espn_id] != player.slot_id)
        return (round(win_probability(mean, math.sqrt(variance), outlook), 12), round(mean, 9), -moved)

    candidates: list[tuple[tuple[float, float, int], float, dict[int, int]]] = [(key(start), 0.0, start)]

    def solve(weight: float) -> dict[int, int]:
        slots = _slots(problem, _assign(problem, _values(problem, weight), keep_starters=keep_starters))
        candidates.append((key(slots), weight, slots))
        return slots

    first = _variance_weight(*_totals(problem, start), outlook)
    if first == 0.0:
        return start, 0.0
    weight = first
    for _ in range(_MAX_ROUNDS):
        following = _variance_weight(*_totals(problem, solve(weight)), outlook)
        if math.isclose(following, weight, rel_tol=1e-9, abs_tol=1e-15):
            break
        weight = following
    for scale in _WEIGHT_SCALES:
        solve(first * scale)
    best = max(candidates, key=lambda candidate: candidate[0])  # the first of equals: the start wins a tie
    return best[2], best[1]


def _optimize(
    problem: _Problem,
    *,
    outlook: MatchupOutlook | None,
    objective: Objective | None,
    lopsided_at: float,
    keep_starters: bool,
) -> LineupPlan:
    if not 0.5 < lopsided_at <= 1.0:
        raise ValueError(f"lopsided_at must be in (0.5, 1], got {lopsided_at!r}")
    if objective is Objective.WIN_PROBABILITY and outlook is None:
        raise ValueError("the win-probability objective needs a MatchupOutlook for the opponent")
    expected = _slots(problem, _assign(problem, _values(problem, 0.0), keep_starters=keep_starters))
    chosen = _choose(problem, expected, outlook, objective, lopsided_at)
    slots, weight = expected, 0.0
    if chosen is Objective.WIN_PROBABILITY and outlook is not None:
        slots, weight = _search_win(problem, outlook, expected, keep_starters=keep_starters)
    _check_legal(problem, slots)
    return _plan(problem, slots, objective=chosen, outlook=outlook, variance_weight=weight)


def optimize_lineup(
    players: Iterable[LineupCandidate],
    slot_counts: Mapping[int, int],
    *,
    sport: Game | str,
    outlook: MatchupOutlook | None = None,
    objective: Objective | None = None,
    lopsided_at: float = LOPSIDED_AT,
) -> LineupPlan:
    """The best legal lineup for one scoring period (see the module docstring for the order of priorities).

    ``slot_counts`` maps each active slot the league uses to its count (:func:`active_slot_counts`); bench and IR come
    from ``sport``'s id maps. ``objective`` forces one objective; by default it is expected points, switching to P(win)
    when an ``outlook`` says the matchup is lopsided (``lopsided_at``). Raises ``ValueError`` for a player listed twice,
    a bench or IR slot in ``slot_counts``, or the P(win) objective without an outlook.
    """
    problem = _problem(players, slot_counts, sport)
    return _optimize(problem, outlook=outlook, objective=objective, lopsided_at=lopsided_at, keep_starters=False)


def bench_inactive_lineup(
    players: Iterable[LineupCandidate],
    slot_counts: Mapping[int, int],
    *,
    sport: Game | str,
    outlook: MatchupOutlook | None = None,
    objective: Objective | None = None,
    lopsided_at: float = LOPSIDED_AT,
) -> LineupPlan:
    """The best lineup that keeps every starter who plays in the lineup: it benches only starters who will not play
    (an OUT, bye or no-game starter) and fills empty slots, shifting starters between slots where that makes room for
    a replacement. It fills as many slots with players who play as :func:`optimize_lineup` does. Same arguments."""
    problem = _problem(players, slot_counts, sport)
    return _optimize(problem, outlook=outlook, objective=objective, lopsided_at=lopsided_at, keep_starters=True)


def score_lineup(
    players: Iterable[LineupCandidate],
    slot_counts: Mapping[int, int],
    *,
    sport: Game | str,
    outlook: MatchupOutlook | None = None,
) -> LineupPlan:
    """The lineup as it stands, scored like a plan (no moves, ``objective`` ``None``)."""
    problem = _problem(players, slot_counts, sport)
    return _plan(
        problem, {player.espn_id: player.slot_id for player in problem.players}, objective=None, outlook=outlook
    )


# --- reading a team's lineup from the store ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LineupInputs:
    """One team's players for one scoring period as the optimizer sees them, read from the store by
    :func:`lineup_inputs`, with the league's active slot counts and what could not be read cleanly."""

    team_id: int
    season: int
    scoring_period_id: int
    players: tuple[LineupCandidate, ...]
    slot_counts: Mapping[int, int]
    roster_as_of: datetime
    warnings: tuple[str, ...] = ()


def active_slot_counts(settings: LeagueSettings) -> dict[int, int]:
    """The league's active slots and their counts (bench and IR excluded), from its settings."""
    return {slot.slot_id: slot.count for slot in settings.active_slots}


def _require_aware(now: datetime) -> None:
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError(f"now must be an aware datetime; got a naive {now.isoformat()}")


def _lock_times(
    plugin: SportPlugin, settings: LeagueSettings, schedule: ScheduleLike, period: int
) -> tuple[dict[int, datetime], datetime | None]:
    """Pro team id -> when its players lock in the period, plus the lock for a team the table lacks: never under
    per-game locks (no game), the period's first start under a first-game lock (everyone locks then)."""
    try:
        locks = plugin.locks(period, schedule, lock_type=settings.lineup_lock_type)
    except ValueError as exc:
        raise LineupError(str(exc)) from exc
    by_team: dict[int, datetime] = {}
    for lock in locks:  # by lock time, so a team's first lock wins
        by_team.setdefault(lock.team_id, lock.at)
    everyone = None if settings.lineup_lock_type is LockType.INDIVIDUAL_GAME else first_start(schedule, period)
    return by_team, everyone


def _availability(
    store: Store, player: PlayerRow, *, season: int, period: int, now: datetime, schedule: ScheduleLike
) -> AvailabilityRow:
    """The stored availability row when it is no older than the player's synced designation, else a fresh one."""
    stored = store.availability.get(player.sport, player.espn_id, season, period)
    if stored is not None and stored.as_of >= player.as_of:
        return stored
    return assess(player, season=season, scoring_period=period, as_of=now, schedule=schedule)


def lineup_inputs(
    store: Store,
    league: LeagueRow,
    settings: LeagueSettings,
    *,
    schedule: ScheduleLike,
    period: int,
    now: datetime,
    team_id: int | None = None,
    weights: BlendWeights | None = None,
) -> LineupInputs:
    """One team's roster for ``period`` as :class:`LineupCandidate` rows (our team unless ``team_id`` names another).

    Reads the roster snapshot, the ``players`` rows, every source's projected stat lines for the period (blended with
    ``weights``, the committed ``blend_weights.toml`` by default, and scored with the league's items) and availability
    (see the module docstring); locks come from ``schedule`` at ``now`` and ESPN's ``lineupLocked``. A player without a
    projection counts as 0 points and one without a ``players`` row is left where he is; both are warned about. Raises
    :class:`LineupError` when the period's roster snapshot is missing, the settings belong to another game or score
    categories, or the lineup lock type is unknown.
    """
    _require_aware(now)
    sport = league.sport
    if settings.game is not Game.coerce(sport):
        raise LineupError(f"league {league.key!r} is {sport} but its settings are for {settings.game.value}")
    if not settings.is_points:
        raise LineupError(
            f"league {league.key!r} scores categories ({settings.scoring_type.value}); this optimizer maximizes points"
        )
    plugin = plugin_for(sport)
    team = league.team_id if team_id is None else team_id
    entries = store.rosters.team(league.row_id, period, team)
    if not entries:
        raise LineupError(f"no roster for team {team} of league {league.key!r} in scoring period {period}; run fm sync")
    by_team, everyone = _lock_times(plugin, settings, schedule, period)
    roster_ids = {entry.espn_id for entry in entries}
    rows = {row.espn_id: row for row in store.players.many(sport, roster_ids)}
    positions = {espn_id: row.position for espn_id, row in rows.items()}
    blend_weights = weights if weights is not None else BlendWeights.load()
    projected = [row for row in store.projections.for_period(sport, league.season, period) if row.espn_id in roster_ids]
    blended = blend(projected, weights=blend_weights, positions=positions)
    lines = {row.espn_id: row.stats for row in blended.rows}
    warnings = list(blended.warnings)
    scorer = Scorer(settings)
    active = frozenset(active_slot_counts(settings))
    players: list[LineupCandidate] = []
    for entry in entries:
        row = rows.get(entry.espn_id)
        if row is None:
            warnings.append(f"player {entry.espn_id} has no players row; left in slot {entry.lineup_slot_id}")
            players.append(
                LineupCandidate(espn_id=entry.espn_id, slot_id=entry.lineup_slot_id, p_active=0.0, locked=True)
            )
            continue
        position = position_for(sport, row.espn_id, positions)
        line = lines.get(row.espn_id)
        if line is None:
            warnings.append(f"no projection for {row.full_name} ({row.espn_id}) in period {period}; counted as 0")
        points = 0.0 if line is None else scorer.points(line, position=position)
        availability = _availability(store, row, season=league.season, period=period, now=now, schedule=schedule)
        status = designation(row.injury_status, sport)
        playing = availability.p_active if row.active and status not in INACTIVE_DESIGNATIONS else 0.0
        listed = frozenset(row.eligible_slot_ids)
        if not listed and position is not None:
            try:
                listed = plugin.eligible_slots(position, include_reserve=False)
            except KeyError:
                warnings.append(f"{row.full_name} ({row.espn_id}) has an unknown position {position!r}")
        team_lock = by_team.get(row.pro_team_id) if row.pro_team_id is not None else None
        lock_at = team_lock if team_lock is not None else everyone
        players.append(
            LineupCandidate(
                espn_id=row.espn_id,
                slot_id=entry.lineup_slot_id,
                eligible=listed & active,
                points=points,
                sd=blend_weights.projection_sd(points, sport=sport, position=position),
                p_active=playing,
                has_game=availability.has_game,
                locked=entry.lineup_locked or (lock_at is not None and now >= lock_at),
                lock_at=lock_at,
                position=position,
                name=row.full_name,
                designation=status.value if row.injury_status else None,
            )
        )
    return LineupInputs(
        team_id=team,
        season=league.season,
        scoring_period_id=period,
        players=tuple(players),
        slot_counts=MappingProxyType(active_slot_counts(settings)),
        roster_as_of=min(entry.as_of for entry in entries),
        warnings=tuple(warnings),
    )


def team_outlook(inputs: LineupInputs, *, sport: Game | str) -> MatchupOutlook:
    """A team's score for the period if it starts its expected-points lineup (locked players where they are): the
    opponent's side of a :class:`MatchupOutlook` (DESIGN section 8.4: each team plays its expected-optimal lineup)."""
    plan = optimize_lineup(inputs.players, inputs.slot_counts, sport=sport, objective=Objective.EXPECTED_POINTS)
    return MatchupOutlook(opponent_mean=plan.expected, opponent_sd=plan.sd)


# --- the decision -----------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class LineupDraft:
    """A lineup proposal ready for :func:`fm.proposals.propose`."""

    kind: ProposalKind
    payload: LineupPayload
    scoring_period_id: int
    deadline: datetime | None
    engine_numbers: Mapping[str, Any]
    rationale: str
    dedupe_key: str


@dataclass(frozen=True, slots=True)
class LineupDecision:
    """What :func:`plan_lineup` found for the league (``league_id`` is the store's row id): the inputs, the lineup as
    it stands (``current``), the best lineup (``best``), the ``bench_inactive`` lineup (``rescue``), the opponent's
    outlook if any, and the drafts (none when the lineup is already right)."""

    league_id: int
    inputs: LineupInputs
    current: LineupPlan
    best: LineupPlan
    rescue: LineupPlan
    outlook: MatchupOutlook | None
    drafts: tuple[LineupDraft, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class LineupProposals:
    """What :func:`propose_lineup` stored (``proposals``; an existing open proposal with the same dedupe key is handed
    back instead of a duplicate) and the policy refusals it caught (``blocked``, one message per draft)."""

    decision: LineupDecision
    proposals: tuple[ProposalRow, ...] = ()
    blocked: tuple[str, ...] = ()


def _digest(parts: Iterable[object]) -> str:
    return hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:16]


def _why(player: LineupCandidate) -> str:
    """Why a starter will not play, for the rationale."""
    if not player.has_game:
        return "no game"
    return _DESIGNATION_LABELS.get(player.designation or "", "inactive")


def _deadline(moved: Iterable[LineupCandidate], schedule: ScheduleLike, period: int) -> datetime | None:
    """The earliest lock among the moved players; with none, when ESPN's calendar leaves the period."""
    locks = [player.lock_at for player in moved if player.lock_at is not None]
    if locks:
        return min(locks)
    last = last_start(schedule, period)
    return None if last is None else period_turn(fantasy_day(last) + timedelta(days=1))


def _rationale(kind: ProposalKind, plan: LineupPlan, current: LineupPlan, moved: Mapping[int, LineupCandidate]) -> str:
    starters, before = set(plan.starters), set(current.starters)
    entering = [moved[espn_id].label for espn_id in moved if espn_id in starters and espn_id not in before]
    leaving = [moved[espn_id] for espn_id in moved if espn_id in before and espn_id not in starters]
    if kind is ProposalKind.BENCH_INACTIVE:
        parts = [f"Bench {', '.join(f'{player.label} ({_why(player)})' for player in leaving)}"] if leaving else []
        if entering:
            parts.append(f"start {', '.join(entering)}")
        text = "; ".join(parts) or "Fill the empty slots"
    else:
        parts = [f"Start {', '.join(entering)}"] if entering else []
        if leaving:
            parts.append(f"bench {', '.join(player.label for player in leaving)}")
        text = "; ".join(parts) or "Rearrange the starters"
    text += f": {plan.expected:.1f} expected points ({plan.expected - current.expected:+.1f})"
    if plan.win_probability is not None and current.win_probability is not None:
        text += f", win probability {current.win_probability:.0%} -> {plan.win_probability:.0%}"
        if plan.objective is Objective.WIN_PROBABILITY:
            text += " (lopsided matchup: maximizing win probability)"
    return text + "."


def _move_numbers(move: LineupMove, player: LineupCandidate, ids: IdMaps) -> dict[str, Any]:
    """One move with the numbers behind it, as JSON for ``proposals.engine_numbers``."""
    return {
        "espn_id": move.espn_id,
        "name": player.name,
        "position": player.position,
        "from_slot_id": move.from_slot_id,
        "to_slot_id": move.to_slot_id,
        "from": ids.slot_label(move.from_slot_id),
        "to": ids.slot_label(move.to_slot_id),
        "points": player.points,
        "p_active": player.p_active,
        "expected": player.expected,
        "sd": player.sd,
        "has_game": player.has_game,
        "designation": player.designation,
        "lock_at": None if player.lock_at is None else player.lock_at.isoformat(),
    }


def _draft(
    kind: ProposalKind,
    plan: LineupPlan,
    *,
    current: LineupPlan,
    inputs: LineupInputs,
    outlook: MatchupOutlook | None,
    schedule: ScheduleLike,
    ids: IdMaps,
    basis: str,
) -> LineupDraft:
    by_id = {player.espn_id: player for player in inputs.players}
    moved = {move.espn_id: by_id[move.espn_id] for move in plan.moves}
    numbers: dict[str, Any] = {
        "objective": None if plan.objective is None else plan.objective.value,
        "expected_points": plan.expected,
        "current_expected_points": current.expected,
        "gain": plan.expected - current.expected,
        "sd": plan.sd,
        "win_probability": plan.win_probability,
        "current_win_probability": current.win_probability,
        "variance_weight": plan.variance_weight,
        "opponent": None if outlook is None else {"mean": outlook.opponent_mean, "sd": outlook.opponent_sd},
        "idle_starters": list(plan.idle_starters),
        "basis": basis,
        "roster_as_of": inputs.roster_as_of.isoformat(),
        "moves": [_move_numbers(move, moved[move.espn_id], ids) for move in plan.moves],
    }
    moves_key = _digest(f"{move.espn_id}:{move.from_slot_id}>{move.to_slot_id}" for move in plan.moves)
    return LineupDraft(
        kind=kind,
        payload=LineupPayload(moves=plan.moves),
        scoring_period_id=inputs.scoring_period_id,
        deadline=_deadline(moved.values(), schedule, inputs.scoring_period_id),
        engine_numbers=numbers,
        rationale=_rationale(kind, plan, current, moved),
        dedupe_key=f"{kind.value}:{inputs.season}:{inputs.scoring_period_id}:{moves_key}",
    )


def plan_lineup(
    store: Store,
    league: LeagueRow,
    *,
    schedule: ScheduleLike,
    now: datetime,
    settings: LeagueSettings | None = None,
    period: int | None = None,
    opponent_team_id: int | None = None,
    outlook: MatchupOutlook | None = None,
    objective: Objective | None = None,
    weights: BlendWeights | None = None,
    lopsided_at: float = LOPSIDED_AT,
) -> LineupDecision:
    """Plan our lineup for a scoring period from the store and draft its proposals, writing nothing.

    ``settings`` default to the synced ones and ``period`` to the one current at ``now`` on the pro schedule
    (:meth:`fm.sports.base.SportPlugin.scoring_period_at`). The P(win) objective needs the opponent: an ``outlook``, or
    an ``opponent_team_id`` whose stored roster gives one (:func:`team_outlook`; a missing roster is a warning and the
    plan stays on expected points). Raises :class:`LineupError` when the league is not synced for the period, is not a
    points league, the season is over, or its lineup lock type is unknown.
    """
    _require_aware(now)
    sport = league.sport
    synced = settings if settings is not None else stored_settings(store, league)
    if synced is None:
        raise LineupError(f"league {league.key!r} has no synced settings; run fm sync")
    target = period if period is not None else plugin_for(sport).scoring_period_at(now, schedule)
    if target is None:
        raise LineupError(f"the pro schedule has no scoring period left at {now.isoformat()}: the season is over")
    blend_weights = weights if weights is not None else BlendWeights.load()
    inputs = lineup_inputs(store, league, synced, schedule=schedule, period=target, now=now, weights=blend_weights)
    warnings = list(inputs.warnings)
    if outlook is None and opponent_team_id is not None:
        try:
            opponent = lineup_inputs(
                store,
                league,
                synced,
                schedule=schedule,
                period=target,
                now=now,
                team_id=opponent_team_id,
                weights=blend_weights,
            )
        except LineupError as exc:
            warnings.append(f"opponent: {exc}; planning for expected points")
        else:
            outlook = team_outlook(opponent, sport=sport)
            warnings.extend(f"opponent: {warning}" for warning in opponent.warnings)
    best = optimize_lineup(
        inputs.players, inputs.slot_counts, sport=sport, outlook=outlook, objective=objective, lopsided_at=lopsided_at
    )
    rescue = bench_inactive_lineup(
        inputs.players, inputs.slot_counts, sport=sport, outlook=outlook, objective=best.objective
    )
    current = score_lineup(inputs.players, inputs.slot_counts, sport=sport, outlook=outlook)
    basis = _digest(sorted((player.espn_id, player.slot_id) for player in inputs.players))

    def draft(kind: ProposalKind, plan: LineupPlan) -> LineupDraft:
        return _draft(
            kind,
            plan,
            current=current,
            inputs=inputs,
            outlook=outlook,
            schedule=schedule,
            ids=ids_for(sport),
            basis=basis,
        )

    drafts: list[LineupDraft] = []
    if rescue.moves:
        drafts.append(draft(ProposalKind.BENCH_INACTIVE, rescue))
    if best.moves and dict(best.slots) != dict(rescue.slots):
        drafts.append(draft(ProposalKind.LINEUP, best))
    return LineupDecision(
        league_id=league.row_id,
        inputs=inputs,
        current=current,
        best=best,
        rescue=rescue,
        outlook=outlook,
        drafts=tuple(drafts),
        warnings=tuple(warnings),
    )


def propose_lineup(
    store: Store,
    config: Config,
    league: LeagueRow,
    *,
    schedule: ScheduleLike,
    now: datetime,
    settings: LeagueSettings | None = None,
    period: int | None = None,
    opponent_team_id: int | None = None,
    outlook: MatchupOutlook | None = None,
    objective: Objective | None = None,
    weights: BlendWeights | None = None,
    lopsided_at: float = LOPSIDED_AT,
) -> LineupProposals:
    """The NFL ``lineup`` decision: :func:`plan_lineup`, then each draft through :func:`fm.proposals.propose` (policy,
    guardrails, dedupe). A draft policy refuses (its kind ``off``, ``fm pause``) is reported in ``blocked``, not raised;
    :class:`LineupError` still is. Takes :func:`plan_lineup`'s arguments plus the config the policy reads."""
    decision = plan_lineup(
        store,
        league,
        schedule=schedule,
        now=now,
        settings=settings,
        period=period,
        opponent_team_id=opponent_team_id,
        outlook=outlook,
        objective=objective,
        weights=weights,
        lopsided_at=lopsided_at,
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
                    created_by=CREATED_BY,
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
    return LineupProposals(decision=decision, proposals=tuple(stored), blocked=tuple(blocked))


_registry.register("nfl", DECISION_KIND, propose_lineup)
