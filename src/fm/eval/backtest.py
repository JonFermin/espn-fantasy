"""Backtest harness: replay past weeks and score projection sources and lineups against what happened (DESIGN section
15, ROADMAP #33).

A backtest answers three questions about a league's history, for every projection source and for the manager's own
lineups:

1. **Projection MAE by position.** How far, in the league's points, was each source's projected line from the
   player's actual line, per position (:class:`PositionError`). A projection with no non-zero stat is a bye, not a
   miss, and is skipped (as :func:`fm.model.projections.fit_position_sd` does); a player a source projected to play
   who has no actual line scored 0, so an injury surprise counts against the source that missed it.
2. **Lineup efficiency.** The points a lineup actually scored divided by the hindsight-optimal lineup's points from
   the same roster (:class:`LineupReport`). For a source the lineup is the one :func:`fm.decide.lineup.optimize_lineup`
   builds from that source's projections alone, so the number is the value of the projections and of nothing else;
   for the manager (``baseline``) it is the lineup the roster snapshot recorded: **the number to beat** (DESIGN
   section 15, "Baseline first").
3. **Start/sit regret.** The points a lineup left on the bench against the hindsight optimum, per week and in total,
   with the number of starters the optimum would have swapped (:attr:`LineupReport.swaps`).

Everything is computed from stat lines and the league's own settings: points come from :class:`fm.model.scoring.Scorer`
(per-position overrides included), and both the projected and the hindsight lineups come from the same assignment
solver the live lineup module uses, over the slots and eligibility the league actually has
(:func:`fm.model.valuation.eligible_active_slots`, :func:`fm.decide.lineup.active_slot_counts`). A player on IR in a
week's lineup stays there in every lineup of that week.

**Pluggable sources.** A source is just a name and its :class:`fm.store.ProjectionRow` s for the replayed weeks
(``BacktestData.projections``). :meth:`BacktestData.with_source` adds one, :func:`with_blend` adds the blend of the
others under chosen weights (:class:`fm.model.projections.BlendWeights`), and :func:`run_backtest` evaluates whichever
names it is given. The weight tuning (ROADMAP #39) fits weights by calling it on held-out periods
(:meth:`BacktestData.restrict_to`) and the in-house baselines (ROADMAP #42, #43) register themselves as a source and
compare "blend with" against "blend without" with ``restrict_to_common=True``, so both are scored on the same
player-weeks.

**Data.** :func:`load_fixture` reads the scrubbed replay format described in ``tests/fixtures/backtest/README.md``;
:func:`load_store_data` reads the same from the state database (roster snapshots for the lineup, stored projections,
ESPN's actual lines), as deep as the sync job has captured history. Neither touches the network.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, ValidationError

from fm.config import Sport
from fm.decide.lineup import LineupCandidate, active_slot_counts, optimize_lineup
from fm.espn.ids import Game, ids_for
from fm.espn.settings import LeagueSettings, SettingsParseError, load_league_settings
from fm.model.projections import BLEND, ESPN, BlendWeights, blend, position_for
from fm.model.scoring import Scorer
from fm.model.valuation import eligible_active_slots
from fm.store import LeagueRow, PlayerRow, ProjectionRow, Store

FIXTURE_FORMAT: Final = 1
"""``format`` of the replay file :func:`load_fixture` reads."""
FIXTURE_FILE: Final = "backtest.json"
BASELINE: Final = "actual"
"""Label of the manager's own recorded lineups in a :class:`BacktestReport` (not a projection source name)."""
ALL_POSITIONS: Final = "ALL"
UNKNOWN_POSITION: Final = "?"
EPSILON: Final = 1e-9
"""Point differences below this are ties (a lineup that scores the optimum within it left nothing on the bench)."""

_GAME_SPORT: Final[Mapping[Game, Sport]] = MappingProxyType({Game.FFL: "nfl", Game.FBA: "nba"})


class BacktestError(ValueError):
    """The replay data cannot be backtested as given: an unreadable fixture, a category league, a lineup naming a
    player or slot the data does not know."""


# --- the replay data --------------------------------------------------------------------------------------------------

type StatLines = Mapping[int, Mapping[str, float]]


@dataclass(frozen=True, slots=True)
class BacktestWeek:
    """One replayed scoring period.

    ``lineup`` is the manager's roster as recorded then: ESPN id to the lineup slot he held (an active slot, bench or
    IR), so its keys are the roster the hindsight lineup draws from. ``actuals`` is each player's actual stat line;
    a player without one did not play and scored 0.
    """

    period: int
    lineup: Mapping[int, int]
    actuals: StatLines

    @property
    def roster(self) -> tuple[int, ...]:
        return tuple(self.lineup)


@dataclass(frozen=True)
class BacktestData:
    """A league's replayed weeks: its settings, the players involved (positions and slot eligibility), the weeks, and
    each projection source's rows (``projections``: source name to projected :class:`fm.store.ProjectionRow` s, which
    may cover players the manager never rostered, so the position MAEs have more to measure).

    Raises :class:`BacktestError` for a league that is not a points league, a sport that does not match the settings,
    repeated or out-of-order periods, a lineup or row naming a player without a ``players`` entry, a slot the league
    lacks, or a projection row of another sport, season or kind.
    """

    sport: Sport
    season: int
    settings: LeagueSettings
    players: Mapping[int, PlayerRow]
    weeks: tuple[BacktestWeek, ...]
    projections: Mapping[str, tuple[ProjectionRow, ...]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if _GAME_SPORT[self.settings.game] != self.sport:
            raise BacktestError(f"league settings are {self.settings.game.value}, not {self.sport}")
        if not self.settings.is_points:
            raise BacktestError("backtests score points leagues; category leagues are not supported yet")
        ids = ids_for(self.settings.game)
        slots = {slot.slot_id for slot in self.settings.active_slots} | {ids.bench_slot, ids.ir_slot}
        periods = [week.period for week in self.weeks]
        if periods != sorted(set(periods)):
            raise BacktestError(f"weeks must be unique and in period order, got {periods}")
        for week in self.weeks:
            for espn_id, slot_id in week.lineup.items():
                if espn_id not in self.players:
                    raise BacktestError(f"period {week.period}: lineup player {espn_id} is not in the players table")
                if slot_id not in slots:
                    raise BacktestError(
                        f"period {week.period}: player {espn_id} is in slot {slot_id}, not a league slot"
                    )
            for espn_id in week.actuals:
                if espn_id not in self.players:
                    raise BacktestError(f"period {week.period}: actual line for unknown player {espn_id}")
        for source, rows in self.projections.items():
            for row in rows:
                if (row.sport, row.season, row.kind) != (self.sport, self.season, "projected"):
                    raise BacktestError(
                        f"source {source!r}: row for {row.espn_id} period {row.scoring_period_id} is "
                        f"{row.sport} {row.season} {row.kind}, not {self.sport} {self.season} projected"
                    )
                if row.espn_id not in self.players:
                    raise BacktestError(f"source {source!r}: projection for unknown player {row.espn_id}")

    @property
    def periods(self) -> tuple[int, ...]:
        return tuple(week.period for week in self.weeks)

    @property
    def sources(self) -> tuple[str, ...]:
        """Projection source names, in insertion order."""
        return tuple(self.projections)

    @property
    def positions(self) -> dict[int, str | None]:
        """ESPN id to position label, for scoring overrides and the MAE grouping."""
        return {espn_id: player.position for espn_id, player in self.players.items()}

    def with_source(self, name: str, rows: Iterable[ProjectionRow]) -> BacktestData:
        """A copy with ``rows`` registered as source ``name`` (replacing a source of that name). Rows are projected
        stat lines for this sport and season, whatever source name they carry."""
        relabelled = tuple(row.model_copy(update={"source": name}) for row in rows)
        return BacktestData(
            self.sport,
            self.season,
            self.settings,
            self.players,
            self.weeks,
            MappingProxyType({**self.projections, name: relabelled}),
        )

    def without_source(self, name: str) -> BacktestData:
        """A copy without source ``name`` (a name it does not have changes nothing)."""
        kept = {source: rows for source, rows in self.projections.items() if source != name}
        return BacktestData(self.sport, self.season, self.settings, self.players, self.weeks, MappingProxyType(kept))

    def restrict_to(self, periods: Iterable[int]) -> BacktestData:
        """A copy with only these periods: the held-out weeks of a fit, or the weeks a weight was fit on."""
        keep = set(periods)
        weeks = tuple(week for week in self.weeks if week.period in keep)
        rows = {
            source: tuple(row for row in source_rows if row.scoring_period_id in keep)
            for source, source_rows in self.projections.items()
        }
        return BacktestData(self.sport, self.season, self.settings, self.players, weeks, MappingProxyType(rows))


def blend_source(
    data: BacktestData, weights: BlendWeights, *, sources: Iterable[str] | None = None, name: str = BLEND
) -> tuple[ProjectionRow, ...]:
    """The blend of ``sources`` (default: every source of ``data``) under ``weights`` as rows labelled ``name``.

    This is :func:`fm.model.projections.blend` on the replayed rows, so a source the weights file does not name is left
    out, and positions come from ``data``. Raises :class:`BacktestError` for a source ``data`` does not have.
    """
    names = tuple(data.projections) if sources is None else tuple(sources)
    unknown = [source for source in names if source not in data.projections]
    if unknown:
        raise BacktestError(f"no such projection source: {', '.join(unknown)} (have {', '.join(data.projections)})")
    blended = blend(
        (row for source in names for row in data.projections[source]), weights=weights, positions=data.positions
    )
    return tuple(row.model_copy(update={"source": name}) for row in blended.rows)


def with_blend(
    data: BacktestData, weights: BlendWeights, *, sources: Iterable[str] | None = None, name: str = BLEND
) -> BacktestData:
    """``data`` plus the blend of ``sources`` under ``weights`` as source ``name`` (see :func:`blend_source`)."""
    return data.with_source(name, blend_source(data, weights, sources=sources, name=name))


# --- metrics ----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PositionError:
    """Projection error at one position, in league points: ``mae`` is the mean absolute error, ``bias`` the mean of
    projected minus actual (positive: the source runs high), ``rmse`` the root mean square error, over ``samples``
    player-weeks."""

    position: str
    samples: int
    mae: float
    bias: float
    rmse: float


@dataclass(frozen=True, slots=True)
class LineupWeek:
    """One week's lineup against the hindsight optimum from the same roster.

    ``points`` is what the lineup's starters actually scored, ``optimal`` what the best lineup in hindsight scored,
    ``starters`` and ``optimal_starters`` who they were. ``swaps`` is how many starters the optimum would have
    replaced (0 when the lineup scored the optimum, even through a different tie).
    """

    period: int
    points: float
    optimal: float
    starters: tuple[int, ...]
    optimal_starters: tuple[int, ...]
    swaps: int

    @property
    def regret(self) -> float:
        """Points left on the bench: ``optimal - points``."""
        return self.optimal - self.points

    @property
    def efficiency(self) -> float | None:
        """``points / optimal`` for the week; ``None`` when the optimum scored nothing."""
        return self.points / self.optimal if self.optimal > EPSILON else None


@dataclass(frozen=True, slots=True)
class LineupReport:
    """A lineup policy over every replayed week: ``label`` is a source name or :data:`BASELINE`."""

    label: str
    weeks: tuple[LineupWeek, ...]

    @property
    def points(self) -> float:
        return math.fsum(week.points for week in self.weeks)

    @property
    def optimal(self) -> float:
        return math.fsum(week.optimal for week in self.weeks)

    @property
    def efficiency(self) -> float | None:
        """Total actual points over total hindsight-optimal points (the points-weighted efficiency); ``None`` when
        the optimum scored nothing."""
        return self.points / self.optimal if self.optimal > EPSILON else None

    @property
    def mean_weekly_efficiency(self) -> float | None:
        """The plain mean of the weekly efficiencies over the weeks that have one."""
        weekly = [value for week in self.weeks if (value := week.efficiency) is not None]
        return math.fsum(weekly) / len(weekly) if weekly else None

    @property
    def regret(self) -> float:
        """Total start/sit regret: points left on the bench over all weeks."""
        return self.optimal - self.points

    @property
    def regret_per_week(self) -> float:
        return self.regret / len(self.weeks) if self.weeks else 0.0

    @property
    def swaps(self) -> int:
        """Starters the hindsight optimum would have replaced, summed over the weeks."""
        return sum(week.swaps for week in self.weeks)

    @property
    def worst_week(self) -> LineupWeek | None:
        """The week with the most regret (the earliest on a tie), or ``None`` without weeks."""
        return max(self.weeks, key=lambda week: week.regret, default=None)


@dataclass(frozen=True, slots=True)
class SourceReport:
    """One projection source: its error by position (``overall`` over every position) and the lineups its projections
    alone would have set."""

    source: str
    by_position: Mapping[str, PositionError]
    overall: PositionError
    lineups: LineupReport

    @property
    def mae(self) -> float:
        return self.overall.mae


@dataclass(frozen=True, slots=True)
class BacktestReport:
    """What :func:`run_backtest` measured. ``baseline`` is the manager's own lineups (:data:`BASELINE`); ``sources``
    holds one :class:`SourceReport` per evaluated source, in the order given."""

    sport: Sport
    season: int
    periods: tuple[int, ...]
    baseline: LineupReport
    sources: Mapping[str, SourceReport]
    restricted_to_common: bool = False
    warnings: tuple[str, ...] = ()

    def source(self, name: str) -> SourceReport:
        try:
            return self.sources[name]
        except KeyError:
            raise KeyError(f"source {name!r} was not evaluated (have {', '.join(self.sources)})") from None

    @property
    def positions(self) -> tuple[str, ...]:
        """Every position any source has an error for, in ESPN position-id order."""
        found = {position for report in self.sources.values() for position in report.by_position}
        return _ordered_positions(found, self.sport)


def _ordered_positions(positions: Iterable[str], sport: Sport) -> tuple[str, ...]:
    ids = ids_for(sport)

    def key(label: str) -> tuple[int, str]:
        try:
            return (ids.position_id(label), label)
        except KeyError:
            return (10_000, label)

    return tuple(sorted(set(positions), key=key))


def _is_zero_line(line: Mapping[str, float]) -> bool:
    return not any(value != 0 for value in line.values())


@dataclass(frozen=True)
class _Replay:
    """What every metric shares: the scorer, positions and each week's actual points."""

    data: BacktestData
    scorer: Scorer
    positions: Mapping[int, str | None]
    actual: Mapping[int, Mapping[int, float]]

    @classmethod
    def build(cls, data: BacktestData, scorer: Scorer | None) -> _Replay:
        scoring = scorer if scorer is not None else Scorer(data.settings)
        positions = data.positions
        actual = {
            week.period: {
                espn_id: scoring.points(line, position=position_for(data.sport, espn_id, positions))
                for espn_id, line in week.actuals.items()
            }
            for week in data.weeks
        }
        return cls(data, scoring, positions, actual)

    def position(self, espn_id: int) -> str | None:
        return position_for(self.data.sport, espn_id, self.positions)

    def projected_points(self, row: ProjectionRow) -> float:
        return self.scorer.points(row.stats, position=self.position(row.espn_id))


def _source_rows(data: BacktestData, source: str, periods: set[int]) -> dict[tuple[int, int], ProjectionRow]:
    """A source's row per (period, player) over the replayed periods; the later of two rows for one key wins."""
    return {
        (row.scoring_period_id, row.espn_id): row
        for row in data.projections[source]
        if row.scoring_period_id in periods
    }


def _projected_keys(rows: Mapping[tuple[int, int], ProjectionRow]) -> set[tuple[int, int]]:
    """The (period, player) keys a source projects to play: a row with at least one non-zero stat."""
    return {key for key, row in rows.items() if not _is_zero_line(row.stats)}


def _position_errors(
    replay: _Replay, rows: Mapping[tuple[int, int], ProjectionRow], keys: set[tuple[int, int]], sport: Sport
) -> tuple[dict[str, PositionError], PositionError]:
    residuals: dict[str, list[float]] = defaultdict(list)
    for key in sorted(keys):
        period, espn_id = key
        projected = replay.projected_points(rows[key])
        observed = replay.actual[period].get(espn_id, 0.0)
        residuals[replay.position(espn_id) or UNKNOWN_POSITION].append(projected - observed)

    def summarize(position: str, errors: Sequence[float]) -> PositionError:
        count = len(errors)
        return PositionError(
            position=position,
            samples=count,
            mae=math.fsum(abs(error) for error in errors) / count,
            bias=math.fsum(errors) / count,
            rmse=math.sqrt(math.fsum(error * error for error in errors) / count),
        )

    by_position = {
        position: summarize(position, residuals[position]) for position in _ordered_positions(residuals, sport)
    }
    everything = [error for errors in residuals.values() for error in errors]
    overall = summarize(ALL_POSITIONS, everything) if everything else PositionError(ALL_POSITIONS, 0, 0.0, 0.0, 0.0)
    return by_position, overall


def _candidates(
    replay: _Replay,
    week: BacktestWeek,
    points: Mapping[int, float],
    plays: set[int],
) -> list[LineupCandidate]:
    """The week's roster as optimizer inputs: ``points`` each player would score, ``plays`` who is expected to play.
    Everyone starts on the bench except players on IR, who the solver leaves where they are."""
    ids = ids_for(replay.data.sport)
    return [
        LineupCandidate(
            espn_id=espn_id,
            slot_id=ids.ir_slot if slot_id == ids.ir_slot else ids.bench_slot,
            eligible=eligible_active_slots(replay.data.players[espn_id], replay.data.settings),
            points=points.get(espn_id, 0.0),
            has_game=espn_id in plays,
            position=replay.position(espn_id),
        )
        for espn_id, slot_id in week.lineup.items()
    ]


def _lineup_week(
    replay: _Replay, week: BacktestWeek, starters: Iterable[int], optimal_points: float, optimal: set[int]
) -> LineupWeek:
    chosen = tuple(starters)
    actual = replay.actual[week.period]
    earned = math.fsum(actual.get(espn_id, 0.0) for espn_id in chosen)
    swaps = 0 if optimal_points - earned <= EPSILON else len(set(chosen) - optimal)
    return LineupWeek(week.period, earned, optimal_points, chosen, tuple(sorted(optimal)), swaps)


def _optimal(replay: _Replay, week: BacktestWeek) -> tuple[float, set[int]]:
    """The hindsight-optimal lineup: every player scores what he actually scored."""
    actual = replay.actual[week.period]
    plan = optimize_lineup(
        _candidates(replay, week, actual, set(week.lineup)),
        active_slot_counts(replay.data.settings),
        sport=replay.data.sport,
    )
    return math.fsum(actual.get(espn_id, 0.0) for espn_id in plan.starters), set(plan.starters)


def _baseline_report(replay: _Replay, optimal: Mapping[int, tuple[float, set[int]]]) -> LineupReport:
    ids = ids_for(replay.data.sport)
    weeks = []
    for week in replay.data.weeks:
        started = [espn_id for espn_id, slot_id in week.lineup.items() if slot_id not in (ids.bench_slot, ids.ir_slot)]
        best, best_starters = optimal[week.period]
        weeks.append(_lineup_week(replay, week, started, best, best_starters))
    return LineupReport(BASELINE, tuple(weeks))


def _source_lineups(
    replay: _Replay,
    source: str,
    rows: Mapping[tuple[int, int], ProjectionRow],
    optimal: Mapping[int, tuple[float, set[int]]],
) -> LineupReport:
    """The lineups a source's projections alone set: it starts whoever it projects highest, a player it has no
    non-zero line for counts as not playing (so a bye is benched while anyone else can fill the slot)."""
    plays = _projected_keys(rows)
    weeks = []
    for week in replay.data.weeks:
        projected = {
            espn_id: replay.projected_points(rows[(week.period, espn_id)])
            for espn_id in week.lineup
            if (week.period, espn_id) in rows
        }
        playing = {espn_id for espn_id in week.lineup if (week.period, espn_id) in plays}
        plan = optimize_lineup(
            _candidates(replay, week, projected, playing),
            active_slot_counts(replay.data.settings),
            sport=replay.data.sport,
        )
        best, best_starters = optimal[week.period]
        weeks.append(_lineup_week(replay, week, plan.starters, best, best_starters))
    return LineupReport(source, tuple(weeks))


def run_backtest(
    data: BacktestData,
    *,
    sources: Iterable[str] | None = None,
    restrict_to_common: bool = False,
    scorer: Scorer | None = None,
) -> BacktestReport:
    """Replay ``data``'s weeks and measure the manager's lineups and each of ``sources`` (default: every source).

    With ``restrict_to_common`` every source's position errors are computed over the player-weeks all the evaluated
    sources project (a non-zero line), so a source that covers more players is not compared on a different sample;
    its lineups are unaffected. ``scorer`` defaults to the league's :class:`fm.model.scoring.Scorer`. Raises
    :class:`BacktestError` for a source ``data`` lacks.
    """
    names = tuple(data.projections) if sources is None else tuple(sources)
    unknown = [name for name in names if name not in data.projections]
    if unknown:
        raise BacktestError(f"no such projection source: {', '.join(unknown)} (have {', '.join(data.projections)})")
    replay = _Replay.build(data, scorer)
    periods = set(data.periods)
    optimal = {week.period: _optimal(replay, week) for week in data.weeks}
    indexed = {name: _source_rows(data, name, periods) for name in names}
    shared: set[tuple[int, int]] | None = None
    if restrict_to_common and names:
        shared = set.intersection(*(_projected_keys(rows) for rows in indexed.values()))
    reports: dict[str, SourceReport] = {}
    warnings: list[str] = []
    for name, rows in indexed.items():
        keys = _projected_keys(rows) if shared is None else shared
        by_position, overall = _position_errors(replay, rows, keys, data.sport)
        if overall.samples == 0:
            warnings.append(f"source {name!r} projects no player-week with an actual to compare")
        reports[name] = SourceReport(name, by_position, overall, _source_lineups(replay, name, rows, optimal))
    return BacktestReport(
        sport=data.sport,
        season=data.season,
        periods=data.periods,
        baseline=_baseline_report(replay, optimal),
        sources=MappingProxyType(reports),
        restricted_to_common=shared is not None,
        warnings=tuple(warnings),
    )


# --- replay files -----------------------------------------------------------------------------------------------------


class _Model(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class _FixturePlayer(_Model):
    espn_id: int
    name: str
    position: str
    eligible_slots: list[int] = []


class _FixtureWeek(_Model):
    period: int
    lineup: dict[int, int]
    projections: dict[str, dict[int, dict[str, float]]] = {}
    actuals: dict[int, dict[str, float]] = {}


class _FixtureFile(_Model):
    format: int
    sport: Sport
    season: int
    settings: str
    as_of: datetime
    players: list[_FixturePlayer]
    weeks: list[_FixtureWeek]


def load_fixture(directory: Path | str, *, settings: LeagueSettings | None = None) -> BacktestData:
    """Read ``<directory>/backtest.json`` (format in ``tests/fixtures/backtest/README.md``) and the league settings
    file it names. ``settings`` overrides that file. Raises :class:`BacktestError` for a missing, malformed or
    inconsistent fixture."""
    root = Path(directory)
    path = root / FIXTURE_FILE
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise BacktestError(f"cannot read backtest fixture {path}: {exc.strerror or exc}") from exc
    try:
        parsed = _FixtureFile.model_validate_json(raw)
    except ValidationError as exc:
        raise BacktestError(f"{path}: {exc}") from exc
    if parsed.format != FIXTURE_FORMAT:
        raise BacktestError(f"{path}: format {parsed.format} is not supported (expected {FIXTURE_FORMAT})")
    if settings is None:
        try:
            settings = load_league_settings(root / parsed.settings)
        except (OSError, SettingsParseError, ValueError) as exc:
            raise BacktestError(f"{root / parsed.settings}: {exc}") from exc
    ids = ids_for(parsed.sport)
    players: dict[int, PlayerRow] = {}
    for player in parsed.players:
        if player.espn_id in players:
            raise BacktestError(f"{path}: player {player.espn_id} is listed twice")
        try:
            position_id = ids.position_id(player.position)
        except KeyError as exc:
            raise BacktestError(f"{path}: {exc.args[0]}") from exc
        players[player.espn_id] = PlayerRow(
            sport=parsed.sport,
            espn_id=player.espn_id,
            full_name=player.name,
            default_position_id=position_id,
            position=player.position,
            eligible_slot_ids=list(player.eligible_slots),
            as_of=parsed.as_of,
        )
    weeks = tuple(
        BacktestWeek(
            week.period,
            MappingProxyType(dict(week.lineup)),
            MappingProxyType({espn_id: MappingProxyType(dict(line)) for espn_id, line in week.actuals.items()}),
        )
        for week in parsed.weeks
    )
    by_source: dict[str, list[ProjectionRow]] = {}
    for week in parsed.weeks:
        for source, lines in week.projections.items():
            by_source.setdefault(source, []).extend(
                ProjectionRow(
                    sport=parsed.sport,
                    espn_id=espn_id,
                    source=source,
                    kind="projected",
                    season=parsed.season,
                    scoring_period_id=week.period,
                    stats=dict(line),
                    as_of=parsed.as_of,
                )
                for espn_id, line in lines.items()
            )
    return BacktestData(
        sport=parsed.sport,
        season=parsed.season,
        settings=settings,
        players=MappingProxyType(players),
        weeks=weeks,
        projections=MappingProxyType({source: tuple(rows) for source, rows in by_source.items()}),
    )


def load_store_data(
    store: Store,
    league: LeagueRow,
    settings: LeagueSettings,
    *,
    periods: Iterable[int] | None = None,
    actual_source: str = ESPN,
) -> BacktestData:
    """Replay the periods the state database holds for ``league``: the manager's roster snapshot, every source's
    stored projected lines (stored ``blend`` rows are left out: blend them fresh with :func:`with_blend`) and
    ``actual_source``'s actual lines. By default the periods are those with both a roster snapshot of our team and at
    least one actual line, so the weeks that have not been played are skipped. History is only as deep as the sync job
    captured: ESPN overwrites a past week's projection with the result. Pool players without a ``players`` row are
    dropped. Raises :class:`BacktestError` when no period qualifies or a rostered player is missing from the
    ``players`` table."""
    league_id = league.row_id
    sport = league.sport
    cursor = store.db.all(
        "SELECT DISTINCT scoring_period_id FROM roster_snapshots WHERE league_id = ? AND team_id = ? "
        "ORDER BY scoring_period_id",
        (league_id, league.team_id),
    )
    candidates = sorted(row["scoring_period_id"] for row in cursor)
    if periods is not None:
        wanted = set(periods)
        candidates = [period for period in candidates if period in wanted]
    weeks: list[BacktestWeek] = []
    by_source: dict[str, list[ProjectionRow]] = defaultdict(list)
    seen: set[int] = set()
    for period in candidates:
        actual_rows = [
            row
            for row in store.projections.for_period(sport, league.season, period, kind="actual")
            if row.source == actual_source
        ]
        if not actual_rows:
            continue
        roster = store.rosters.team(league_id, period, league.team_id)
        weeks.append(
            BacktestWeek(
                period,
                MappingProxyType({entry.espn_id: entry.lineup_slot_id for entry in roster}),
                MappingProxyType({row.espn_id: MappingProxyType(dict(row.stats)) for row in actual_rows}),
            )
        )
        seen.update(entry.espn_id for entry in roster)
        seen.update(row.espn_id for row in actual_rows)
        for row in store.projections.for_period(sport, league.season, period, kind="projected"):
            if row.source != BLEND:
                by_source[row.source].append(row)
                seen.add(row.espn_id)
    if not weeks:
        raise BacktestError(
            f"league {league.key!r} has no played period with a roster snapshot and {actual_source} actuals in the "
            "store; use --fixtures, or let fm sync run through a week first"
        )
    rows = {player.espn_id: player for player in store.players.many(sport, seen)}
    unknown = sorted(espn_id for week in weeks for espn_id in week.lineup if espn_id not in rows)
    if unknown:
        raise BacktestError(f"rostered players missing from the players table: {', '.join(map(str, unknown[:10]))}")
    weeks = [
        BacktestWeek(
            week.period, week.lineup, MappingProxyType({i: line for i, line in week.actuals.items() if i in rows})
        )
        for week in weeks
    ]
    return BacktestData(
        sport=sport,
        season=league.season,
        settings=settings,
        players=MappingProxyType(rows),
        weeks=tuple(weeks),
        projections=MappingProxyType(
            {
                source: tuple(row for row in source_rows if row.espn_id in rows)
                for source, source_rows in by_source.items()
            }
        ),
    )
