"""NBA streaming: the add/drop sequence that fills the matchup week's open slot-days (DESIGN section 9.3, ROADMAP #31).

A matchup week has more lineup slots than most rosters have players with games on a given night.
:mod:`fm.decide.lineup_daily` finds the **open slot-days** (a slot nobody on the roster who plays can fill) and this
module spends the week's acquisitions on free agents whose games land on them.

**The value of a game** is the daily lineup's: league points times ``p_active`` in a points league, and in a category
league the swing-weighted :meth:`fm.model.categories.CategoryModel.contribution`
(:func:`fm.decide.lineup_daily.game_values`): P(win category) is approximated by ``Phi(delta_mu / sigma)`` over the
rest of the matchup, so a category near 50% weighs ``dP/dscore`` the most. A streamer therefore ranks by the sum, over
his games, of ``dP/dstat x production`` where he fills a slot (:func:`rank_streamers`: his marginal
:func:`fm.model.value_nba.marginal_lineup_value` over the remaining days, with the open slot-days he fills and each
category's share, ``swing``). The lines are :func:`fm.model.value_nba.blend_day`'s.

**The sequence** (:func:`plan_streams`) is a daily dynamic program. The state after each day is the roster (with the
adds and drops so far) and the acquisitions used; a day either does nothing or adds one candidate
(``max_adds_per_day``) for a droppable player (or for no one while the roster has room), and the roster's value that
day is the daily lineup's (:func:`fm.model.value_nba.daily_lineup_value`, per day, with that day's ``p_active``).
Every acquisition costs ``min_gain`` of value, so a move must pay for the transaction slot it uses and the noise in
its projection. Per (day, acquisitions used) only the ``beam`` best rosters survive, and the surviving plans are re-
valued exactly with :func:`fm.decide.lineup_daily.plan_week` (locks, games-played limits), the best of them winning
over doing nothing.

**Limits** are read from the league. The acquisitions a matchup allows are
``AcquisitionSettings.matchup_limit_for(days)`` (ESPN encodes "3 per weekly matchup" as 3/7 per day; never the raw
number) and the policy's ``max_transactions_per_week``, less the add/drop proposals the matchup already holds
(:func:`fm.proposals.acquisitions_this_week`, the count :func:`fm.proposals.evaluate` enforces; ``acquisitions_used``
overrides it for what ESPN's own counter says). The season limit is not tracked: ESPN's counter is not in the store.
Roster spots: a roster with no open spot drops someone for each add, and a league's position limits are kept.

**Protected players** are never dropped: the policy's untouchables, players on IR, players in other open proposals
(add/drop and waiver claims, and the lineup proposals that move them), a player the plan cannot value (no ``players``
row or no projection line: his value is unknown, not zero), the league's ``core`` (the best ``core_size`` players by
per-game value; default the league's active slots, so only bench-quality players are droppable), and any ``protected``
ESPN ids given. A player whose lock has passed cannot be dropped.

**A drop costs what the player is worth beyond this matchup.** The matchup's lineups value a bench player who is OUT
this week, or has one game, at about 0, so on the week alone any small gain would drop him. The rest of his season is
priced the way the add's is: the add stays on the roster after the week, so a drop loses nothing when the add is at
least as good per game (:attr:`fm.decide.lineup_daily.DayPlayer.per_game`, the rate the lineups scale by games and
``p_active``), and a drop whose per-game value is *higher* than the add's gives up that difference for every game the
team has left, which no matchup's streaming gain is taken to cover: such a pair is never planned
(:func:`_additions`). The rule holds in the season's last matchup too, where it is stricter than needed; a stream that
keeps the better player is just not made.

**Timing.** Adds, drops and trades lock by the league's ``roster_lock_type`` (ESPN's ``rosterLocktimeType``, a setting
of its own beside the lineup lock). The real NBA league's ``FIRSTGAME_SCORINGPERIOD`` closes them at the day's *first*
tip, so a streamer must be added before it to play that day; a move's deadline is ``plugin.transaction_cutoff(...)``
for the add's and the drop's team, the earlier of the two (``INDIVIDUAL_GAME`` locks per team, and a team without a
game does not close). A cutoff that has passed rules the day out and the plan starts the next day. An ``UNKNOWN``
roster lock type (ESPN's weekly types parse so) is refused (:class:`StreamingError`), never guessed. The add lands on
the bench; :func:`fm.decide.lineup_daily.propose_daily_lineup` moves him into the lineup on its next run.

**Proposals** (:func:`propose_streaming`, registered as ``("nba", "streaming")``): only the target day's move is
proposed, as an ``add_drop`` through :func:`fm.proposals.propose` (policy and guardrails again, a refusal is reported,
not raised); later days' moves are the plan's outlook and are planned again that day. Nothing here writes to ESPN
(CLAUDE.md: workers propose, the executor acts).
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final

from fm.config import Config, Policy
from fm.decide import registry as _registry
from fm.decide.lineup_daily import (
    DEFAULT_GAME_SD,
    DailyInputs,
    DailyLineupError,
    DayPlayer,
    PlayerDay,
    WeekPlan,
    daily_inputs,
    plan_week,
)
from fm.decide.waivers import Wire, WireStatus, load_wire
from fm.espn.calendar import matchup_scoring_periods
from fm.espn.settings import LeagueSettings, LockType
from fm.model.categories import CategoryModel
from fm.model.projections import BlendWeights
from fm.model.value_nba import daily_lineup_value, marginal_lineup_value, team_games
from fm.proposals import (
    ACQUISITION_KINDS,
    AddDropPayload,
    PolicyError,
    ProposalKind,
    acquisitions_this_week,
    expire_due,
    find_untouchables,
    parse_payload,
    propose,
)
from fm.proposals.payloads import LineupPayload, WaiverPayload
from fm.proposals.policy import as_utc, stored_settings
from fm.sports.base import ScheduleLike, period_turn
from fm.sports.nba import NBA
from fm.store import LeagueRow, PlayerRow, ProposalRow, Store

if TYPE_CHECKING:
    from fm.model.availability import OfficialReport
    from fm.sources.nba_injuries import OfficialInjuryReport

IR_SLOT: Final = NBA.ids.ir_slot
STREAMING_KIND: Final = "streaming"
"""The kind :func:`propose_streaming` is registered under for NBA in :mod:`fm.decide.registry`."""
STREAMING_CREATED_BY: Final = "decide.streaming"
"""``created_by`` of the proposals this module stores."""
DEFAULT_MIN_GAIN_POINTS: Final = 2.0
"""League points a streaming move must add over the rest of the matchup to be worth its transaction slot (a points
league). A model parameter: about half a good bench game, below which a move is projection noise."""
DEFAULT_MIN_GAIN_CATEGORIES: Final = 0.03
"""Expected category wins a move must add to be worth its slot (a category league)."""
DEFAULT_SCREEN: Final = 25
"""Free agents (best games-times-value first) ranked by their marginal lineup value."""
DEFAULT_STREAMERS: Final = 8
"""Ranked streamers the sequence considers."""
DEFAULT_BEAM: Final = 20
"""Rosters kept per (day, acquisitions used)."""
DEFAULT_FINALISTS: Final = 5
"""Plans re-valued exactly against doing nothing."""


class StreamingError(ValueError):
    """Streaming cannot be planned for the league: not NBA, not synced, or its roster lock type is unknown. Nothing is
    proposed."""


# --- ranking streamers ------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StreamerScore:
    """A free agent's worth over the rest of the matchup: ``value`` is his marginal lineup value with nobody dropped
    (:func:`fm.model.value_nba.marginal_lineup_value`), ``games`` the games he plays (his team has one and he may be
    active), ``open_slot_games`` those that fall on an open slot-day he is eligible for and ``swing`` a category
    league's share of ``value`` by category: swing weight times his production times the open slot-days he fills."""

    espn_id: int
    name: str
    value: float
    games: int
    open_slot_games: int
    swing: Mapping[str, float] = MappingProxyType({})


def rank_streamers(
    inputs: DailyInputs,
    rows: Mapping[int, PlayerRow],
    candidates: Mapping[int, DayPlayer],
    schedule: ScheduleLike,
    settings: LeagueSettings,
    *,
    baseline: WeekPlan | None = None,
) -> list[StreamerScore]:
    """The candidates by marginal lineup value over the matchup's remaining days, best first (ties by ESPN id).

    ``rows`` are the roster's and the candidates' ``players`` rows. A candidate who adds nothing is kept (value 0) so
    the caller sees why. ``baseline`` is the roster's own plan (open slot-days); by default it is planned here.
    """
    roster = [rows[player.espn_id] for player in inputs.players if player.slot_id != IR_SLOT and player.espn_id in rows]
    days = inputs.days
    base = baseline if baseline is not None else plan_week(inputs.players, inputs.slot_counts, days)
    if base is None:
        raise StreamingError("the roster's own lineup could not be planned")
    weights = inputs.values.outlook.weights if inputs.values.outlook is not None else None
    vectors = inputs.values.vectors
    scores: list[StreamerScore] = []
    for espn_id, player in candidates.items():
        row = rows.get(espn_id)
        if row is None:
            continue
        value = 0.0
        games = open_games = 0
        swing: dict[str, float] = {}
        for day in days:
            today = player.day(day)
            if not today.plays:
                continue
            games += 1
            per_game = {p.espn_id: p.day(day).value for p in inputs.players} | {espn_id: today.value}
            value += marginal_lineup_value(roster, per_game, settings, schedule, [day], add=row)
            if player.eligible & set(base.day(day).open_slots):
                open_games += 1
                if vectors.get(espn_id) is not None:
                    for category, score in vectors[espn_id].items():
                        share = (weights or {}).get(category, 1.0) * today.p_active * score
                        swing[category] = swing.get(category, 0.0) + share
        scores.append(StreamerScore(espn_id, player.label, value, games, open_games, MappingProxyType(swing)))
    return sorted(scores, key=lambda score: (-score.value, score.espn_id))


# --- the sequence -----------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StreamMove:
    """One add on one day: the free agent in, the player out (``None`` while the roster has a spot) and what the
    plan's value gained from it, ``gain`` (the plan with this move less the plan without it)."""

    day: int
    add: int
    drop: int | None
    gain: float = 0.0


@dataclass(frozen=True, slots=True)
class StreamPlan:
    """A sequence of moves and what it is worth: ``value`` over the matchup's remaining days (the lineups' expected
    value in the league's unit, exact: :func:`fm.decide.lineup_daily.plan_week`) against ``baseline``, doing nothing."""

    moves: tuple[StreamMove, ...]
    value: float
    baseline: float

    @property
    def gain(self) -> float:
        return self.value - self.baseline

    def on(self, day: int) -> tuple[StreamMove, ...]:
        return tuple(move for move in self.moves if move.day == day)


@dataclass(frozen=True, slots=True)
class _Problem:
    """What the dynamic program searches: the roster's and candidates' players and rows, who may be dropped, the days
    and what each day's moves must satisfy."""

    settings: LeagueSettings
    schedule: ScheduleLike
    days: tuple[int, ...]
    rows: Mapping[int, PlayerRow]
    players: Mapping[int, DayPlayer]
    roster: frozenset[int]
    droppable: frozenset[int]
    candidates: tuple[int, ...]
    capacity: int
    budget: int
    max_per_day: int
    min_gain: float
    allowed: Mapping[tuple[int, int, int | None], bool]
    """(day, add, drop) -> a move that is allowed that day (cutoffs and locks); absent means allowed."""


def _within_limits(problem: _Problem, roster: Collection[int]) -> bool:
    counts: dict[int, int] = {}
    for espn_id in roster:
        row = problem.rows.get(espn_id)  # a roster player without a players row is held in place, outside the limits
        position = row.default_position_id if row is not None else None
        if position is not None:
            counts[position] = counts.get(position, 0) + 1
    for position, count in counts.items():
        limit = problem.settings.position_limit(position)
        if limit is not None and count > limit:
            return False
    return True


class _Values:
    """Day values of a roster through :func:`fm.model.value_nba.daily_lineup_value`, cached by (roster, day)."""

    def __init__(self, problem: _Problem) -> None:
        self.problem = problem
        self._cache: dict[tuple[frozenset[int], int], float] = {}

    def day(self, roster: frozenset[int], day: int) -> float:
        key = (roster, day)
        cached = self._cache.get(key)
        if cached is None:
            problem = self.problem
            valued = sorted(espn_id for espn_id in roster if espn_id in problem.rows)
            per_game = {espn_id: problem.players[espn_id].day(day).value for espn_id in valued}
            lineup = daily_lineup_value(
                [problem.rows[espn_id] for espn_id in valued],
                per_game,
                problem.settings,
                problem.schedule,
                [day],
            )
            cached = lineup.total
            self._cache[key] = cached
        return cached


type _State = tuple[frozenset[int], int]
"""(roster, acquisitions used): what the future of a plan depends on."""


@dataclass(frozen=True, slots=True)
class _Path:
    value: float
    moves: tuple[tuple[int, int, int | None], ...] = ()
    gone: frozenset[int] = frozenset()


def _search(
    problem: _Problem, beam: int, finalists: int
) -> list[tuple[tuple[tuple[int, int, int | None], ...], float]]:
    """The dynamic program: the best plans (moves, fast value net of the transaction costs), best first."""
    values = _Values(problem)
    states: dict[_State, _Path] = {(problem.roster, 0): _Path(0.0)}
    for day in problem.days:
        following: dict[_State, _Path] = {}
        for (roster, used), path in states.items():
            options: list[tuple[frozenset[int], int, tuple[tuple[int, int, int | None], ...], frozenset[int]]] = [
                (roster, used, path.moves, path.gone)
            ]
            if used < problem.budget:
                options.extend(_additions(problem, roster, used, path, day))
            for new_roster, new_used, moves, gone in options:
                gained = path.value + values.day(new_roster, day) - problem.min_gain * (new_used - used)
                key = (new_roster, new_used)
                kept = following.get(key)
                if kept is None or gained > kept.value:
                    following[key] = _Path(gained, moves, gone)
        states = _prune(following, beam)
    ranked = sorted(states.values(), key=lambda path: (-path.value, len(path.moves)))
    return [(path.moves, path.value) for path in ranked[:finalists]]


def _additions(
    problem: _Problem, roster: frozenset[int], used: int, path: _Path, day: int
) -> Iterable[tuple[frozenset[int], int, tuple[tuple[int, int, int | None], ...], frozenset[int]]]:
    """Every way to add up to ``max_per_day`` candidates on ``day`` from ``roster``."""
    streamers = {espn_id for espn_id in roster if espn_id not in problem.roster}
    frontier = [(roster, used, path.moves, path.gone, 0)]
    while frontier:
        current, spent, moves, gone, today = frontier.pop()
        if today >= problem.max_per_day or spent >= problem.budget:
            continue
        for add in problem.candidates:
            if add in current or add in gone or not problem.players[add].day(day).plays:
                continue
            drops: list[int | None] = []
            if len(current) < problem.capacity:
                drops.append(None)
            drops.extend(
                sorted(
                    espn_id
                    for espn_id in current
                    if (espn_id in problem.droppable or espn_id in streamers)
                    and espn_id not in {m[1] for m in moves if m[0] == day}
                    and problem.players[espn_id].per_game <= problem.players[add].per_game  # no rate lost for good
                )
            )
            for drop in drops:
                if not problem.allowed.get((day, add, drop), True):
                    continue
                new_roster = current - {drop} | {add} if drop is not None else current | {add}
                if len(new_roster) > problem.capacity or not _within_limits(problem, new_roster):
                    continue
                new_gone = gone | {add} | ({drop} if drop is not None else set())
                step = (*moves, (day, add, drop))
                yield new_roster, spent + 1, step, new_gone
                frontier.append((new_roster, spent + 1, step, new_gone, today + 1))


def _prune(states: dict[_State, _Path], beam: int) -> dict[_State, _Path]:
    """Keep the ``beam`` best rosters at each number of acquisitions used."""
    by_used: dict[int, list[tuple[_State, _Path]]] = {}
    for state, path in states.items():
        by_used.setdefault(state[1], []).append((state, path))
    kept: dict[_State, _Path] = {}
    for entries in by_used.values():
        entries.sort(key=lambda entry: (-entry[1].value, len(entry[1].moves), sorted(entry[0][0])))
        kept.update(entries[:beam])
    return kept


# --- the decision -----------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Budget:
    """The acquisitions a matchup has left: ``limit`` it allows (ESPN's matchup limit for its days and the policy's
    ``max_transactions_per_week``, whichever is smaller), ``used`` so far and ``left``."""

    limit: int | None
    used: int
    espn_limit: int | None = None
    policy_limit: int | None = None

    @property
    def left(self) -> int:
        return 10**6 if self.limit is None else max(0, self.limit - self.used)


@dataclass(frozen=True, slots=True)
class StreamDraft:
    """An ``add_drop`` proposal for the target day, ready for :func:`fm.proposals.propose`."""

    kind: ProposalKind
    payload: AddDropPayload
    scoring_period_id: int
    deadline: datetime | None
    engine_numbers: Mapping[str, Any]
    rationale: str
    dedupe_key: str


@dataclass(frozen=True, slots=True)
class StreamingDecision:
    """What :func:`plan_streaming` found: the inputs, the free agents ranked, the budget, the plan (``None`` when doing
    nothing is best) and the drafts for the target day."""

    league_id: int
    inputs: DailyInputs
    ranked: tuple[StreamerScore, ...]
    budget: Budget
    plan: StreamPlan | None
    baseline: WeekPlan
    drafts: tuple[StreamDraft, ...]
    protected: frozenset[int] = frozenset()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class StreamingProposals:
    """What :func:`propose_streaming` stored and the policy refusals it caught (``blocked``, one message per draft)."""

    decision: StreamingDecision
    proposals: tuple[ProposalRow, ...] = ()
    blocked: tuple[str, ...] = ()

    @property
    def warnings(self) -> tuple[str, ...]:
        return self.decision.warnings


def acquisition_budget(
    store: Store,
    league: LeagueRow,
    settings: LeagueSettings,
    policy: Policy,
    *,
    period: int,
    now: datetime,
    used: int | None = None,
) -> Budget:
    """The acquisitions the matchup of ``period`` has left (see the module docstring). ``used`` replaces the count of
    the league's add/drop and waiver proposals in the matchup."""
    span = matchup_scoring_periods(settings, period)
    espn_limit = settings.acquisition.matchup_limit_for(len(span)) if span else settings.acquisition.matchup_limit
    held = (
        used
        if used is not None
        else acquisitions_this_week(store, league, settings, scoring_period_id=period, now=as_utc(now))
    )
    limits = [limit for limit in (espn_limit, policy.max_transactions_per_week) if limit is not None]
    return Budget(min(limits) if limits else None, held, espn_limit, policy.max_transactions_per_week)


def _committed(store: Store, league: LeagueRow, now: datetime) -> frozenset[int]:
    """Players the league's open proposals touch: new moves leave them alone. That is every player of an add/drop or
    waiver proposal, and every player an open ``bench_inactive`` or ``lineup`` proposal moves (dropping him would leave
    that proposal moving someone who is gone)."""
    kinds = {kind.value for kind in ACQUISITION_KINDS} | {ProposalKind.BENCH_INACTIVE.value, ProposalKind.LINEUP.value}
    players: set[int] = set()
    for row in store.proposals.open(league.row_id):
        if row.kind not in kinds or (row.status != "executing" and row.deadline is not None and row.deadline <= now):
            continue
        payload = parse_payload(row)
        if isinstance(payload, AddDropPayload | WaiverPayload):
            players.update(i for i in (payload.add_espn_id, payload.drop_espn_id) if i is not None)
        elif isinstance(payload, LineupPayload):
            players.update(move.espn_id for move in payload.moves)
    return frozenset(players)


def core_players(per_game: Mapping[int, float], count: int) -> frozenset[int]:
    """The ``count`` players with the highest per-game value (ties by ESPN id)."""
    best = sorted(per_game, key=lambda espn_id: (-per_game[espn_id], espn_id))
    return frozenset(best[: max(0, count)])


def _cutoff(settings: LeagueSettings, schedule: ScheduleLike, row: PlayerRow | None, day: int) -> datetime | None:
    """When adds and drops of ``row`` close on ``day`` under the league's roster lock; ``None`` when they do not (a
    team without a game under per-game locks)."""
    if row is None or row.pro_team_id is None:
        return None
    return NBA.transaction_cutoff(row.pro_team_id, day, schedule, lock_type=settings.roster_lock_type)


def _allowed_moves(
    settings: LeagueSettings,
    schedule: ScheduleLike,
    rows: Mapping[int, PlayerRow],
    wire: Wire,
    candidates: Iterable[int],
    droppable: Iterable[int],
    *,
    target: int,
    now: datetime,
) -> dict[tuple[int, int, int | None], bool]:
    """Which (day, add, drop) moves cannot be made: on the target day, a cutoff that has passed (the add's or the
    drop's) or an add whose game has begun (``lineupLocked`` in the pool). Later days are planned ahead and checked
    when they come."""
    blocked: dict[tuple[int, int, int | None], bool] = {}
    drops: list[int | None] = [None, *droppable]
    for add in candidates:
        entry = wire.get(add)
        add_cutoff = _cutoff(settings, schedule, rows.get(add), target)
        add_closed = (entry is not None and entry.locked) or (add_cutoff is not None and now >= add_cutoff)
        for drop in drops:
            drop_cutoff = _cutoff(settings, schedule, rows.get(drop), target) if drop is not None else None
            if add_closed or (drop_cutoff is not None and now >= drop_cutoff):
                blocked[(target, add, drop)] = False
    return blocked


def _transaction_deadline(
    settings: LeagueSettings, schedule: ScheduleLike, add: PlayerRow | None, drop: PlayerRow | None, day: int
) -> datetime | None:
    """The earlier of the add's and the drop's cutoff on ``day``; with neither, when ESPN's calendar leaves the day."""
    cutoffs = [at for row in (add, drop) if (at := _cutoff(settings, schedule, row, day)) is not None]
    if cutoffs:
        return min(cutoffs)
    date = NBA.period_day(day, schedule)
    return None if date is None else period_turn(date + timedelta(days=1))


def _shift(player: DayPlayer, day: int, *, joins: bool) -> DayPlayer:
    """``player`` as the exact planner sees him when a move on ``day`` brings him in (his earlier days have no game) or
    sends him away (his days from ``day`` on have none)."""
    gone = PlayerDay()
    days = {
        period: (gone if (period < day if joins else period >= day) else value) for period, value in player.days.items()
    }
    return replace(player, days=MappingProxyType(days))


def _exact(
    inputs: DailyInputs,
    candidates: Mapping[int, DayPlayer],
    moves: Sequence[tuple[int, int, int | None]],
    slot_limits: Mapping[int, int],
) -> float | None:
    """A plan's value by :func:`plan_week` with the roster changing on the days of its moves."""
    roster = {player.espn_id: player for player in inputs.players}
    extra: dict[int, DayPlayer] = {}
    for day, add, drop in moves:
        if drop is not None:
            holder = extra if drop in extra else roster
            holder[drop] = _shift(holder[drop], day, joins=False)
        extra[add] = replace(_shift(candidates[add], day, joins=True), slot_id=NBA.ids.bench_slot)
    plan = plan_week([*roster.values(), *extra.values()], inputs.slot_counts, inputs.days, slot_limits=slot_limits)
    return None if plan is None else plan.total


def plan_streams(
    inputs: DailyInputs,
    rows: Mapping[int, PlayerRow],
    candidates: Mapping[int, DayPlayer],
    settings: LeagueSettings,
    schedule: ScheduleLike,
    *,
    budget: int,
    droppable: Iterable[int],
    allowed: Mapping[tuple[int, int, int | None], bool] | None = None,
    min_gain: float,
    max_adds_per_day: int = 1,
    beam: int = DEFAULT_BEAM,
    finalists: int = DEFAULT_FINALISTS,
) -> StreamPlan | None:
    """The best sequence of at most ``budget`` adds over ``inputs.days`` from ``candidates`` (see the module docstring),
    or ``None`` when no sequence beats doing nothing by ``min_gain`` a move. ``droppable`` are the players who may be
    dropped; ``allowed`` marks the (day, add, drop) moves that cannot be made."""
    base_players = [player for player in inputs.players if player.slot_id != IR_SLOT]
    everyone = {**{p.espn_id: p for p in inputs.players}, **candidates}
    problem = _Problem(
        settings=settings,
        schedule=schedule,
        days=inputs.days,
        rows=rows,
        players=everyone,
        roster=frozenset(player.espn_id for player in base_players),
        droppable=frozenset(droppable),
        candidates=tuple(sorted(candidates)),
        capacity=settings.roster_size,
        budget=budget,
        max_per_day=max(1, max_adds_per_day),
        min_gain=min_gain,
        allowed=MappingProxyType(dict(allowed or {})),
    )
    if budget <= 0 or not candidates:
        return None
    found = _search(problem, beam, finalists)
    baseline = _exact(inputs, candidates, (), inputs.slot_limits)
    if baseline is None:
        return None
    best: StreamPlan | None = None
    for moves, _ in found:
        if not moves:
            continue
        value = _exact(inputs, candidates, moves, inputs.slot_limits)
        if value is None or value - baseline < min_gain * len(moves):
            continue
        if best is None or value > best.value:
            best = StreamPlan(_gains(inputs, candidates, moves, baseline, value), value, baseline)
    return best


def _gains(
    inputs: DailyInputs,
    candidates: Mapping[int, DayPlayer],
    moves: Sequence[tuple[int, int, int | None]],
    baseline: float,
    total: float,
) -> tuple[StreamMove, ...]:
    """Each move's gain: the plan less the plan without that move (and without the later moves that depend on its
    add), the last move taking what is left so the gains add up to the plan's."""
    result: list[StreamMove] = []
    previous = baseline
    for index, (day, add, drop) in enumerate(moves):
        value = total if index == len(moves) - 1 else _exact(inputs, candidates, moves[: index + 1], inputs.slot_limits)
        value = previous if value is None else value
        result.append(StreamMove(day, add, drop, value - previous))
        previous = value
    return tuple(result)


def plan_streaming(
    store: Store,
    league: LeagueRow,
    *,
    schedule: ScheduleLike,
    now: datetime,
    policy: Policy | None = None,
    settings: LeagueSettings | None = None,
    period: int | None = None,
    wire: Wire | None = None,
    lines: Mapping[int, Mapping[str, float]] | None = None,
    weights: BlendWeights | None = None,
    games_used: Mapping[int, int] | None = None,
    opponent_team_id: int | None = None,
    model: CategoryModel | None = None,
    official: OfficialReport | OfficialInjuryReport | None = None,
    game_sd: Mapping[str, float] | float = DEFAULT_GAME_SD,
    margin_so_far: Mapping[str, float] | None = None,
    acquisitions_used: int | None = None,
    protected: Iterable[int] = (),
    core_size: int | None = None,
    min_gain: float | None = None,
    max_adds_per_day: int = 1,
    screen: int = DEFAULT_SCREEN,
    candidates: int = DEFAULT_STREAMERS,
    beam: int = DEFAULT_BEAM,
    finalists: int = DEFAULT_FINALISTS,
) -> StreamingDecision:
    """Rank the league's free agents and plan the matchup's add/drop sequence, writing nothing (the module docstring
    gives the rules). ``policy`` is the league's (untouchables, ``max_transactions_per_week``; default the
    :class:`fm.config.Policy` defaults); ``wire`` defaults to :func:`fm.decide.waivers.load_wire`. Raises
    :class:`StreamingError` for a league that is not NBA, is not synced, or whose roster lock type is unknown, and
    :class:`fm.decide.lineup_daily.DailyLineupError` as the daily lineup does."""
    at = as_utc(now)
    if league.sport != "nba":
        raise StreamingError(f"league {league.key!r} is {league.sport}; streaming is for NBA leagues")
    synced = settings if settings is not None else stored_settings(store, league)
    if synced is None:
        raise StreamingError(f"league {league.key!r} has no synced settings; run fm sync")
    if synced.roster_lock_type is LockType.UNKNOWN:
        raise StreamingError(
            f"league {league.key!r}: its roster lock type is UNKNOWN ({synced.roster_lock_type_raw!r}), so when adds "
            "and drops close is unknown; refusing to guess"
        )
    target = period if period is not None else NBA.scoring_period_at(at, schedule)
    if target is None:
        raise StreamingError(f"the pro schedule has no scoring period left at {at.isoformat()}: the season is over")
    rules = policy if policy is not None else Policy()
    try:
        inputs = daily_inputs(
            store,
            league,
            synced,
            schedule=schedule,
            period=target,
            now=at,
            lines=lines,
            weights=weights,
            games_used=games_used,
            opponent_team_id=opponent_team_id,
            model=model,
            official=official,
            game_sd=game_sd,
            margin_so_far=margin_so_far,
        )
    except DailyLineupError as exc:
        raise StreamingError(str(exc)) from exc
    warnings = list(inputs.warnings)
    loaded = wire if wire is not None else load_wire(store, league, scoring_period=target)
    warnings.extend(loaded.warnings)
    roster_ids = {player.espn_id for player in inputs.players}
    rows = {row.espn_id: row for row in store.players.many(league.sport, roster_ids)}
    committed = _committed(store, league, at)
    wanted = [
        espn_id
        for espn_id, entry in loaded.entries.items()
        if entry.status is WireStatus.FREE_AGENT and espn_id not in roster_ids and espn_id not in committed
    ]
    pool_rows = {row.espn_id: row for row in store.players.many(league.sport, wanted)}
    per_game_value = inputs.values.per_game
    usable = [i for i in wanted if i in pool_rows and i in per_game_value and pool_rows[i].pro_team_id is not None]
    if len(usable) < len(wanted):
        warnings.append(f"{len(wanted) - len(usable)} free agents have no players row or projection line; left out")
    usable.sort(
        key=lambda i: (-per_game_value[i] * team_games(schedule, pool_rows[i].pro_team_id, inputs.days), i)
    )
    pool: dict[int, DayPlayer] = {}
    for espn_id in usable[: max(0, screen)]:
        player = inputs.model.day_player(pool_rows[espn_id], NBA.ids.bench_slot)
        if player is not None:
            rows[espn_id] = pool_rows[espn_id]
            pool[espn_id] = player
    budget = acquisition_budget(store, league, synced, rules, period=target, now=at, used=acquisitions_used)
    baseline = plan_week(inputs.players, inputs.slot_counts, inputs.days, slot_limits=inputs.slot_limits)
    if baseline is None:
        raise StreamingError(f"league {league.key!r}: no legal lineup could be planned")
    screened = dict(
        sorted(
            pool.items(),
            key=lambda item: (-sum(item[1].day(day).value for day in inputs.days), item[0]),
        )[: max(0, screen)]
    )
    ranked = rank_streamers(inputs, rows, screened, schedule, synced, baseline=baseline)
    top = {score.espn_id: screened[score.espn_id] for score in ranked[: max(0, candidates)] if score.value > 0}
    untouchable = find_untouchables(store, league.sport, rules, sorted(roster_ids))
    unvalued = sorted(
        player.espn_id
        for player in inputs.players
        if player.slot_id != IR_SLOT and (player.espn_id not in per_game_value or player.espn_id not in rows)
    )
    if unvalued:
        labels = {player.espn_id: player.label for player in inputs.players}
        names = ", ".join(f"{labels[i]} ({i})" for i in unvalued)
        warnings.append(f"{names}: no players row or projection line, so his value is unknown; never dropped")
    per_game = {
        player.espn_id: player.per_game
        for player in inputs.players
        if player.slot_id != IR_SLOT and player.espn_id not in unvalued
    }
    core = core_players(per_game, core_size if core_size is not None else synced.active_slot_count)
    barred = frozenset(untouchable) | core | committed | frozenset(protected) | frozenset(unvalued)
    droppable = sorted(
        player.espn_id
        for player in inputs.players
        if player.slot_id != IR_SLOT and player.espn_id not in barred and player.espn_id in rows
    )
    gain_floor = (
        min_gain
        if min_gain is not None
        else DEFAULT_MIN_GAIN_CATEGORIES
        if synced.is_categories
        else DEFAULT_MIN_GAIN_POINTS
    )
    allowed = _allowed_moves(synced, schedule, rows, loaded, top, droppable, target=target, now=at)
    plan = (
        plan_streams(
            inputs,
            rows,
            top,
            synced,
            schedule,
            budget=min(budget.left, len(inputs.days) * max(1, max_adds_per_day)),
            droppable=droppable,
            allowed=allowed,
            min_gain=gain_floor,
            max_adds_per_day=max_adds_per_day,
            beam=beam,
            finalists=finalists,
        )
        if top
        else None
    )
    drafts = _drafts(plan, ranked, inputs, rows, synced, schedule, budget, target)
    return StreamingDecision(
        league_id=league.row_id,
        inputs=inputs,
        ranked=tuple(ranked),
        budget=budget,
        plan=plan,
        baseline=baseline,
        drafts=drafts,
        protected=barred,
        warnings=tuple(dict.fromkeys(warnings)),
    )


def _name(rows: Mapping[int, PlayerRow], espn_id: int | None) -> str:
    return "no one" if espn_id is None else rows[espn_id].full_name if espn_id in rows else f"ESPN {espn_id}"


def _drafts(
    plan: StreamPlan | None,
    ranked: Sequence[StreamerScore],
    inputs: DailyInputs,
    rows: Mapping[int, PlayerRow],
    settings: LeagueSettings,
    schedule: ScheduleLike,
    budget: Budget,
    target: int,
) -> tuple[StreamDraft, ...]:
    if plan is None:
        return ()
    scores = {score.espn_id: score for score in ranked}
    drafts: list[StreamDraft] = []
    for move in plan.on(target):
        add, drop = rows.get(move.add), rows.get(move.drop) if move.drop is not None else None
        score = scores.get(move.add)
        numbers: dict[str, Any] = {
            "unit": inputs.values.unit,
            "day": move.day,
            "gain": move.gain,
            "plan_gain": plan.gain,
            "plan_value": plan.value,
            "baseline_value": plan.baseline,
            "add": {"espn_id": move.add, "name": _name(rows, move.add)},
            "drop": None if move.drop is None else {"espn_id": move.drop, "name": _name(rows, move.drop)},
            "streamer": None
            if score is None
            else {
                "value": score.value,
                "games": score.games,
                "open_slot_games": score.open_slot_games,
                "swing": dict(score.swing),
            },
            "later_moves": [
                {"day": later.day, "add": later.add, "drop": later.drop, "gain": later.gain}
                for later in plan.moves
                if later.day != target
            ],
            "acquisitions": {"limit": budget.limit, "used": budget.used, "left": budget.left},
            "days": list(inputs.days),
            "roster_as_of": inputs.roster_as_of.isoformat(),
        }
        text = f"Add {_name(rows, move.add)}"
        text += f", drop {_name(rows, move.drop)}" if move.drop is not None else " (a roster spot is open)"
        extra = f" over {score.games} games, {score.open_slot_games} on open slots" if score is not None else ""
        text += f": +{move.gain:.1f} {inputs.values.unit}{extra} through day {inputs.days[-1]}."
        drafts.append(
            StreamDraft(
                kind=ProposalKind.ADD_DROP,
                payload=AddDropPayload(add_espn_id=move.add, drop_espn_id=move.drop),
                scoring_period_id=move.day,
                deadline=_transaction_deadline(settings, schedule, add, drop, move.day),
                engine_numbers=numbers,
                rationale=text,
                dedupe_key=f"add_drop:{inputs.season}:{move.day}:{move.add}:{move.drop}",
            )
        )
    return tuple(drafts)


def propose_streaming(
    store: Store,
    config: Config,
    league: LeagueRow,
    *,
    schedule: ScheduleLike,
    now: datetime,
    settings: LeagueSettings | None = None,
    period: int | None = None,
    wire: Wire | None = None,
    lines: Mapping[int, Mapping[str, float]] | None = None,
    weights: BlendWeights | None = None,
    games_used: Mapping[int, int] | None = None,
    opponent_team_id: int | None = None,
    model: CategoryModel | None = None,
    official: OfficialReport | OfficialInjuryReport | None = None,
    game_sd: Mapping[str, float] | float = DEFAULT_GAME_SD,
    margin_so_far: Mapping[str, float] | None = None,
    acquisitions_used: int | None = None,
    protected: Iterable[int] = (),
    core_size: int | None = None,
    min_gain: float | None = None,
    max_adds_per_day: int = 1,
) -> StreamingProposals:
    """The NBA ``streaming`` decision: :func:`plan_streaming` with the league's policy, then the target day's draft
    through :func:`fm.proposals.propose` (policy, guardrails, dedupe). A draft policy refuses is reported in
    ``blocked``, not raised; :class:`StreamingError` still is. The league's proposals past their deadline are expired
    first, as :func:`fm.proposals.propose` would."""
    at = as_utc(now)
    try:
        policy = config.league(league.key).policy
    except KeyError as exc:
        raise StreamingError(exc.args[0]) from None
    expire_due(store, now=at, league_id=league.row_id)
    decision = plan_streaming(
        store,
        league,
        schedule=schedule,
        now=at,
        policy=policy,
        settings=settings,
        period=period,
        wire=wire,
        lines=lines,
        weights=weights,
        games_used=games_used,
        opponent_team_id=opponent_team_id,
        model=model,
        official=official,
        game_sd=game_sd,
        margin_so_far=margin_so_far,
        acquisitions_used=acquisitions_used,
        protected=protected,
        core_size=core_size,
        min_gain=min_gain,
        max_adds_per_day=max_adds_per_day,
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
                    created_by=STREAMING_CREATED_BY,
                    scoring_period_id=draft.scoring_period_id,
                    engine_numbers=draft.engine_numbers,
                    rationale=draft.rationale,
                    deadline=draft.deadline,
                    dedupe_key=draft.dedupe_key,
                    now=at,
                )
            )
        except PolicyError as exc:
            blocked.append(str(exc))
    return StreamingProposals(decision=decision, proposals=tuple(stored), blocked=tuple(blocked))


_registry.register("nba", STREAMING_KIND, propose_streaming)
