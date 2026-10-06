"""NBA valuation: per-game projections from the ESPN + DARKO blend, points over the schedule, and the daily lineup value
of a roster (DESIGN sections 8.1, 8.3 and 9.3; ROADMAP #24).

**Per-game lines.** NBA projections are per game. ESPN publishes no daily projections (``kona_game_state`` says
``hasGameStatProjections: false``), so its projection for a day is its season projection's per-game rate
(docs/espn-api.md section 1 #12): :func:`per_game_line` divides the totals by ``GP``, keeps the rates (``FG%``,
``APG``, ...; :data:`fm.model.categories.RATE_CATEGORIES`) and sets ``GP`` to 1. DARKO projects a per-game line from
per-100 rates, pace and minutes (:meth:`fm.sources.darko.DarkoProjection.per_game`); :func:`darko_line` keys it by ESPN
abbreviation (:data:`DARKO_STATS`) and :func:`darko_rows` joins it to ESPN ids through the NBA crosswalk ``fm sync``
saves (:meth:`fm.model.ids_nba.NbaCrosswalk.espn_id`). DARKO zeroes the minutes of a player it does not expect to
play; such a row is left out rather than blended as a line of zeros, since availability is ``p_active``'s business
(:mod:`fm.model.availability`), not the stat line's.

**Sources.** Importing this module attaches :class:`DarkoLoader` to the ``darko`` source ROADMAP #15 registered without
one. ESPN's NBA source stays the ``stored`` one the sync job writes, and what it stores is the season line, so a day's
blend goes through :func:`blend_day`: :func:`fm.model.projections.blend_period` over the same sources, with ESPN read
as the per-game rate of its stored season line (:class:`EspnDayLoader`, :func:`day_sources`). A blended row for day
``d`` is a per-game line, what the player produces if he plays; the blend weights' NBA uncertainty covers one game.
The line does not change from day to day, so one day's blend values any span of days.

**Points** (:func:`scheduled_points`). In a points league a player's worth over a span of days is his per-game points
under the league's scoring items times the games his team plays in the span (DESIGN 8.3), times ``p_active`` when
given.

**Daily lineup value** (:func:`daily_lineup_value`). Only players in active slots score, so a roster's worth over a
span is the sum, day by day, of its most valuable lineup among the players whose team plays: an assignment over the
league's slots and each player's ESPN ``eligibleSlots`` (the NBA plugin's table for a player without them) solved with
``scipy.optimize.linear_sum_assignment`` (DESIGN 9.1). A crowded day benches the weakest player the slots allow; a thin
one leaves open slots (DESIGN 9.3's open slot-days), as does a player ruled out, and an open slot is where a streamer
adds his whole game. :func:`marginal_lineup_value` is the difference an add (and a drop) makes. A game's value is its
expected worth measured from an empty slot: league points per game times ``p_active``, or in a category league
:meth:`fm.model.categories.CategoryModel.contribution`. Locks, games-played caps and the acquisition limit belong to
the daily lineup and streaming decisions (ROADMAP #31).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from types import MappingProxyType
from typing import Final, Unpack

import numpy as np
from scipy.optimize import linear_sum_assignment

from fm.espn.ids import FBA, Game
from fm.espn.settings import LeagueSettings
from fm.model.categories import GAMES_PLAYED, RATE_CATEGORIES
from fm.model.ids_nba import NBA_SPORT, NbaCrosswalk
from fm.model.projections import (
    BLEND,
    DARKO,
    ESPN,
    BlendWeights,
    Converted,
    PeriodBlend,
    ProjectionSourceRegistry,
    blend_period,
    load_stored,
    source_registry,
)
from fm.model.scoring import Scorer, derive_stats
from fm.sources.base import Fetched, FetchOptions, SourceError
from fm.sources.darko import DarkoProjection, DarkoSource
from fm.sports.base import FREE_AGENT_TEAM, ScheduleLike, SportPlugin
from fm.sports.nba import NBA
from fm.store import PlayerRow, ProjectionRow, Store

SEASON_PERIOD: Final = 0
"""The scoring period of a season line in ``projections``."""
TALENT_DATASET: Final = "talent"
"""``Fetched.dataset`` of DARKO's talent sheet (:meth:`fm.sources.darko.DarkoSource.projections`)."""
DARKO_LABEL: Final = "DARKO per-game projections (talent sheet, joined through the NBA crosswalk)"
ESPN_DAY_LABEL: Final = "ESPN per-game rate of its stored season projection"

DARKO_STATS: Mapping[str, str] = MappingProxyType(
    {
        "min": "MIN",
        "pts": "PTS",
        "oreb": "OREB",
        "dreb": "DREB",
        "reb": "REB",
        "ast": "AST",
        "stl": "STL",
        "blk": "BLK",
        "tov": "TO",
        "pf": "PF",
        "fga": "FGA",
        "fgm": "FGM",
        "fg3a": "3PA",
        "fg3m": "3PM",
        "fta": "FTA",
        "ftm": "FTM",
    }
)
"""DARKO's per-game keys (:meth:`fm.sources.darko.DarkoProjection.per_game`) to ESPN ``fba`` abbreviations."""

for _abbreviation in DARKO_STATS.values():  # a typo fails at import rather than in a blend
    FBA.stat_id(_abbreviation)


# --- per-game lines ---------------------------------------------------------------------------------------------------


def per_game_line(line: Mapping[str, float]) -> dict[str, float] | None:
    """A line over several games (ESPN's season projection) as one game's: totals divided by its ``GP``, rates as they
    are, ``GP`` 1. ``None`` when the line has no games. Raises :class:`fm.model.scoring.ScoringError` for a
    non-numeric or non-finite stat."""
    clean = derive_stats(line, Game.FBA, ())  # derives nothing; validates every value
    games = clean.get(GAMES_PLAYED, 0.0)
    if games <= 0:
        return None
    per_game = {stat: value if stat in RATE_CATEGORIES else value / games for stat, value in clean.items()}
    per_game[GAMES_PLAYED] = 1.0
    return per_game


def scale_line(line: Mapping[str, float], games: float) -> dict[str, float]:
    """A per-game line over ``games`` games: totals multiplied, rates as they are, ``GP`` = ``games``. Raises
    ``ValueError`` for a negative or non-finite count."""
    if isinstance(games, bool) or not math.isfinite(games) or games < 0:
        raise ValueError(f"games should be a finite number >= 0, got {games!r}")
    scaled = {stat: value if stat in RATE_CATEGORIES else value * games for stat, value in line.items()}
    scaled[GAMES_PLAYED] = float(games)
    return scaled


def darko_line(projection: DarkoProjection) -> dict[str, float]:
    """DARKO's per-game line keyed by ESPN abbreviation, with ``GP`` 1. Raises ``KeyError`` for a per-game key
    :data:`DARKO_STATS` does not map, since a stat dropped quietly would score wrong."""
    line = {DARKO_STATS[key]: value for key, value in projection.per_game().items()}
    line[GAMES_PLAYED] = 1.0
    return line


def darko_rows(
    projections: Iterable[DarkoProjection],
    crosswalk: NbaCrosswalk,
    *,
    season: int,
    scoring_period: int,
    as_of: datetime,
) -> Converted:
    """DARKO's projections as ``darko`` rows for one day, keyed by ESPN id through the NBA crosswalk.

    Left out: players DARKO projects for no minutes or marks unavailable (counted in a warning), and players the
    crosswalk does not map (listed in ``unmapped`` by nba.com id). The crosswalk holds the players ``fm sync`` has seen,
    so most of DARKO's sheet is unmapped by design; a warning comes only when nobody maps. When two rows name one
    player the later wins.
    """
    rows: dict[int, ProjectionRow] = {}
    unmapped: list[str] = []
    resting = 0
    for projection in projections:
        if not projection.available or projection.minutes <= 0:
            resting += 1
            continue
        espn_id = crosswalk.espn_id(projection.nba_id)
        if espn_id is None:
            unmapped.append(str(projection.nba_id))
            continue
        rows[espn_id] = ProjectionRow(
            sport=NBA_SPORT,
            espn_id=espn_id,
            source=DARKO,
            kind="projected",
            season=season,
            scoring_period_id=scoring_period,
            stats=darko_line(projection),
            as_of=as_of,
        )
    warnings: list[str] = []
    if unmapped and not rows:
        warnings.append(
            f"darko: the NBA crosswalk maps none of the {len(unmapped)} players DARKO projects; run fm sync to save it"
        )
    if resting:
        warnings.append(f"darko: left out {resting} players projected for no minutes or marked unavailable")
    return Converted(tuple(rows.values()), tuple(unmapped), tuple(warnings))


def espn_day_rows(rows: Iterable[ProjectionRow], *, scoring_period: int) -> Converted:
    """ESPN's NBA season projections (period 0) as its projection for one day: each line's per-game rate
    (:func:`per_game_line`) under ``scoring_period``. Other rows are ignored; a season line without games is left out
    and counted in a warning."""
    converted: list[ProjectionRow] = []
    no_games = 0
    for row in rows:
        if row.sport != NBA_SPORT or row.kind != "projected" or row.scoring_period_id != SEASON_PERIOD:
            continue
        line = per_game_line(row.stats)
        if line is None:
            no_games += 1
            continue
        converted.append(row.model_copy(update={"scoring_period_id": scoring_period, "stats": line}))
    warnings: list[str] = []
    if no_games:
        warnings.append(f"espn: {no_games} season projections have no games (GP 0 or missing); no per-game line")
    return Converted(tuple(converted), (), tuple(warnings))


class EspnDayLoader:
    """ESPN's NBA projection for a day, as :func:`blend_day` loads it: the per-game rate of the season projection the
    sync job stored (:func:`espn_day_rows`). Nothing stored is an empty ``degraded`` result naming the fix."""

    def __call__(
        self, store: Store, season: int, scoring_period: int, options: FetchOptions
    ) -> Fetched[tuple[ProjectionRow, ...]]:
        stored = load_stored(store, NBA_SPORT, ESPN, season, SEASON_PERIOD)
        converted = espn_day_rows(stored.data, scoring_period=scoring_period)
        return Fetched(
            converted.rows,
            stored.as_of,
            ESPN,
            stored.dataset,
            f"{NBA_SPORT}_{season}_{scoring_period}",
            cached=True,
            degraded=stored.degraded or not converted.rows,
            warnings=(*stored.warnings, *converted.warnings),
        )


class DarkoLoader:
    """The ``darko`` source's loader: DARKO's talent sheet through the adapter, joined to ESPN ids.

    ``source`` defaults to a fresh :class:`fm.sources.darko.DarkoSource` over the cache dir (closed after each call),
    which serves its cached sheet while it is fresh; ``crosswalk`` defaults to the NBA crosswalk saved in the store. A
    sheet that cannot be fetched or parsed with no copy to fall back on is an empty ``degraded`` result with the reason
    in its warnings, so the blend runs on ESPN alone.
    """

    def __init__(self, source: DarkoSource | None = None, *, crosswalk: NbaCrosswalk | None = None) -> None:
        self.source = source
        self.crosswalk = crosswalk

    def __call__(
        self, store: Store, season: int, scoring_period: int, options: FetchOptions
    ) -> Fetched[tuple[ProjectionRow, ...]]:
        adapter = self.source if self.source is not None else DarkoSource()
        try:
            fetched = adapter.projections(**options)
        except SourceError as exc:
            reason = f"darko: unavailable, blending without it ({exc})"
            return Fetched((), adapter.clock(), DARKO, TALENT_DATASET, "current", degraded=True, warnings=(reason,))
        finally:
            if self.source is None:
                adapter.close()
        walk = self.crosswalk if self.crosswalk is not None else NbaCrosswalk.from_store(store)
        converted = darko_rows(fetched.data, walk, season=season, scoring_period=scoring_period, as_of=fetched.as_of)
        return Fetched(
            converted.rows,
            fetched.as_of,
            fetched.source,
            fetched.dataset,
            fetched.key,
            cached=fetched.cached,
            stale=fetched.stale,
            degraded=fetched.degraded,
            warnings=(*fetched.warnings, *converted.warnings),
            raw_path=fetched.raw_path,
        )


def _attach_darko_loader() -> None:
    """Re-register ``darko`` (registered by ROADMAP #15 without a loader) with :class:`DarkoLoader`; a source that
    already loads is left alone, so importing this module twice changes nothing."""
    current = source_registry.get(NBA_SPORT, DARKO)
    if current is not None and current.loadable:
        return
    if current is not None:
        source_registry.unregister(NBA_SPORT, DARKO)
    source_registry.register(NBA_SPORT, DARKO, label=DARKO_LABEL, loader=DarkoLoader())


_attach_darko_loader()


def day_sources(sources: ProjectionSourceRegistry = source_registry) -> ProjectionSourceRegistry:
    """The NBA sources of ``sources`` for a day's blend, in their order, with the stored ``espn`` source read as the
    per-game rate of its season line (:class:`EspnDayLoader`). ``sources`` itself is not changed."""
    day = ProjectionSourceRegistry()
    for source in sources.registered(NBA_SPORT):
        if source.name == ESPN and source.stored:
            day.register(NBA_SPORT, ESPN, label=ESPN_DAY_LABEL, loader=EspnDayLoader())
        else:
            day.register(NBA_SPORT, source.name, label=source.label, loader=source.loader, stored=source.stored)
    return day


def blend_day(
    store: Store,
    season: int,
    day: int,
    *,
    weights: BlendWeights,
    sources: ProjectionSourceRegistry = source_registry,
    save: bool = True,
    **options: Unpack[FetchOptions],
) -> PeriodBlend:
    """Blend the NBA sources' per-game lines for one day (a scoring period >= 1) and, with ``save``, store them.

    :func:`fm.model.projections.blend_period` over :func:`day_sources`: ESPN's per-game rate and every other NBA
    source (DARKO's loader; the in-house baselines of ROADMAP #43 once registered), averaged stat by stat with the
    weights file's NBA weights. Saved, the inputs land under their own source names and the blend under ``blend`` for
    the day. Raises ``ValueError`` for a day below 1 (period 0 holds season lines).
    """
    if day < 1:
        raise ValueError(f"an NBA day is a scoring period >= 1 (period 0 holds season lines), got {day}")
    return blend_period(
        store, NBA_SPORT, season, day, weights=weights, sources=day_sources(sources), save=save, **options
    )


def per_game_lines(store: Store, season: int, day: int, *, source: str = BLEND) -> dict[int, dict[str, float]]:
    """The stored per-game lines for a day by ESPN id: the blend :func:`blend_day` saved, or one source's."""
    return {row.espn_id: dict(row.stats) for row in store.projections.for_period(NBA_SPORT, season, day, source=source)}


# --- the schedule -----------------------------------------------------------------------------------------------------


def team_games(schedule: ScheduleLike, pro_team_id: int | None, periods: Iterable[int]) -> int:
    """Games a pro team plays over the scoring periods (days); 0 for a free agent or a player without a team."""
    if pro_team_id is None or pro_team_id == FREE_AGENT_TEAM:
        return 0
    return sum(len(schedule.games_for(pro_team_id, period)) for period in periods)


def period_lines(
    lines: Mapping[int, Mapping[str, float]],
    players: Iterable[PlayerRow],
    schedule: ScheduleLike,
    periods: Iterable[int],
) -> dict[int, dict[str, float]]:
    """Each player's per-game line over the games his team plays in ``periods`` (:func:`scale_line`), by ESPN id:
    the schedule-aware lines a category model is fit on. Players without a line are left out."""
    span = tuple(periods)
    return {
        player.espn_id: scale_line(lines[player.espn_id], team_games(schedule, player.pro_team_id, span))
        for player in players
        if player.espn_id in lines
    }


# --- points -----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ScheduledPoints:
    """A player's points over a span of days: ``per_game`` league points per game, the ``games`` his team plays and
    the chance ``p_active`` that he plays."""

    espn_id: int
    per_game: float
    games: int
    p_active: float = 1.0

    @property
    def expected_per_game(self) -> float:
        return self.per_game * self.p_active

    @property
    def total(self) -> float:
        return self.per_game * self.games * self.p_active


def _points_league(settings: LeagueSettings) -> None:
    if settings.game is not Game.FBA:
        raise ValueError(f"league {settings.league_id} is {settings.game.value}, not an NBA (fba) league")
    if not settings.is_points:
        raise ValueError(
            f"league {settings.league_id} competes on categories, not points; value it with fm.model.categories"
        )


def _probability(p_active: Mapping[int, float] | None, espn_id: int) -> float:
    value = 1.0 if p_active is None else p_active.get(espn_id, 1.0)
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value <= 1.0:
        raise ValueError(f"p_active of ESPN {espn_id} should be a probability, got {value!r}")
    return float(value)


def scheduled_points(
    lines: Mapping[int, Mapping[str, float]],
    players: Iterable[PlayerRow],
    settings: LeagueSettings,
    schedule: ScheduleLike,
    periods: Iterable[int],
    *,
    p_active: Mapping[int, float] | None = None,
    scorer: Scorer | None = None,
) -> dict[int, ScheduledPoints]:
    """Per-game points × games over ``periods`` (days) for each player with a line, by ESPN id (DESIGN 8.3).

    ``lines`` are per-game lines (:func:`per_game_lines`); each is scored with the league's items at the player's
    position, and his team's games come from ``schedule``. ``p_active`` (ESPN id -> probability, 1 when absent) scales
    the total. Raises ``ValueError`` for a league that is not an NBA points league or a ``p_active`` outside [0, 1].
    """
    _points_league(settings)
    scoring = scorer if scorer is not None else Scorer(settings)
    span = tuple(periods)
    values: dict[int, ScheduledPoints] = {}
    for player in players:
        line = lines.get(player.espn_id)
        if line is None:
            continue
        values[player.espn_id] = ScheduledPoints(
            espn_id=player.espn_id,
            per_game=scoring.points(line, position=player.position),
            games=team_games(schedule, player.pro_team_id, span),
            p_active=_probability(p_active, player.espn_id),
        )
    return values


# --- daily lineup value -----------------------------------------------------------------------------------------------


def _slot_instances(settings: LeagueSettings) -> tuple[int, ...]:
    """The league's active lineup slots by slot id, one entry per slot (``UTIL`` three times in a 3-UTIL league)."""
    return tuple(slot.slot_id for slot in settings.active_slots for _ in range(slot.count))


def _eligible_slots(player: PlayerRow, settings: LeagueSettings, *, plugin: SportPlugin = NBA) -> frozenset[int]:
    """The league's active slots a player may fill: his ESPN ``eligibleSlots``, or without them his position's
    (``plugin``'s table); none for a player with neither."""
    league = {slot.slot_id for slot in settings.active_slots}
    if player.eligible_slot_ids:
        return frozenset(league.intersection(player.eligible_slot_ids))
    if player.position is None:
        return frozenset()
    try:
        return frozenset(league & plugin.eligible_slots(player.position, include_reserve=False))
    except KeyError:  # a position the id maps do not know (``POS_<id>``)
        return frozenset()


@dataclass(frozen=True, slots=True)
class DayLineup:
    """One day's best lineup: the active slot each starter fills and the value of his games, the players with a game
    left on the bench, and the active slots left empty."""

    scoring_period: int
    slots: Mapping[int, int]
    values: Mapping[int, float]
    benched: tuple[int, ...]
    open_slots: int

    @property
    def value(self) -> float:
        return math.fsum(self.values.values())


@dataclass(frozen=True, slots=True)
class DailyLineupValue:
    """A roster's best daily lineups over a span of days."""

    days: tuple[DayLineup, ...]

    @property
    def total(self) -> float:
        return math.fsum(day.value for day in self.days)

    @property
    def open_slots(self) -> int:
        """Open slot-days: active slots left empty, summed over the days."""
        return sum(day.open_slots for day in self.days)

    def starts(self, espn_id: int) -> int:
        """Days the player starts."""
        return sum(espn_id in day.slots for day in self.days)

    def value_of(self, espn_id: int) -> float:
        """The value the player's started games add up to."""
        return math.fsum(day.values.get(espn_id, 0.0) for day in self.days)


def _best_lineup(period: int, playing: list[tuple[int, frozenset[int], float]], slots: tuple[int, ...]) -> DayLineup:
    """The most valuable legal lineup of the players with a game: an assignment to ``slots`` with a bench column per
    player. Only a game worth something starts. A game's value does not depend on the slot, so with every candidate's
    value positive the most valuable lineup also starts as many of them as the slots allow (the sets of players that
    fit the slots together form a matroid, where every maximal set has one size): the open slots are exactly the ones
    no candidate can fill, whichever of several equally valuable lineups the solver returns."""
    candidates = [entry for entry in playing if entry[2] > 0]
    slot_of: dict[int, int] = {}
    value_of: dict[int, float] = {}
    if candidates:
        matrix = np.zeros((len(candidates), len(slots) + len(candidates)))
        for row, (_, eligible, value) in enumerate(candidates):
            for column, slot in enumerate(slots):
                matrix[row, column] = value if slot in eligible else -np.inf
        rows, columns = linear_sum_assignment(matrix, maximize=True)
        for row, column in zip(rows.tolist(), columns.tolist(), strict=True):
            if column < len(slots):
                espn_id, _, value = candidates[row]
                slot_of[espn_id] = slots[column]
                value_of[espn_id] = value
    benched = tuple(espn_id for espn_id, _, _ in playing if espn_id not in slot_of)
    return DayLineup(period, MappingProxyType(slot_of), MappingProxyType(value_of), benched, len(slots) - len(slot_of))


def daily_lineup_value(
    players: Iterable[PlayerRow],
    per_game: Mapping[int, float],
    settings: LeagueSettings,
    schedule: ScheduleLike,
    periods: Iterable[int],
    *,
    plugin: SportPlugin = NBA,
) -> DailyLineupValue:
    """The roster's best lineup each day of ``periods`` and what it is worth (see the module docstring).

    ``players`` are the roster's players (leave injured-reserve stashes out); ``per_game`` is the expected value of
    one game by each of them, measured from an empty slot (league points times ``p_active``, or a category model's
    :meth:`~fm.model.categories.CategoryModel.contribution`). A game worth nothing or less never starts, so a player
    ruled out (``p_active`` 0) leaves his slot open; a player without a value is left out, as if he had no game. A
    player starts at most once a day, worth his value times his team's games that day.
    """
    roster = sorted({player.espn_id: player for player in players}.values(), key=lambda player: player.espn_id)
    slots = _slot_instances(settings)
    eligibility = {player.espn_id: _eligible_slots(player, settings, plugin=plugin) for player in roster}
    days: list[DayLineup] = []
    for period in dict.fromkeys(periods):
        playing = [
            (player.espn_id, eligibility[player.espn_id], per_game[player.espn_id] * games)
            for player in roster
            if player.espn_id in per_game and (games := team_games(schedule, player.pro_team_id, (period,)))
        ]
        days.append(_best_lineup(period, playing, slots))
    return DailyLineupValue(tuple(days))


def marginal_lineup_value(
    players: Iterable[PlayerRow],
    per_game: Mapping[int, float],
    settings: LeagueSettings,
    schedule: ScheduleLike,
    periods: Iterable[int],
    *,
    add: PlayerRow,
    drop: int | None = None,
    plugin: SportPlugin = NBA,
) -> float:
    """What adding ``add`` (and dropping ESPN id ``drop``) changes the roster's daily lineup value by over ``periods``.

    Raises ``ValueError`` when ``add`` is already on the roster or has no value in ``per_game``, or ``drop`` is not on
    the roster.
    """
    roster = list(players)
    ids = {player.espn_id for player in roster}
    if add.espn_id in ids:
        raise ValueError(f"ESPN {add.espn_id} is already on the roster")
    if add.espn_id not in per_game:
        raise ValueError(f"ESPN {add.espn_id} has no per-game value to add")
    if drop is not None and drop not in ids:
        raise ValueError(f"cannot drop ESPN {drop}: not on the roster")
    span = tuple(periods)
    before = daily_lineup_value(roster, per_game, settings, schedule, span, plugin=plugin)
    after_roster = [player for player in roster if player.espn_id != drop] + [add]
    after = daily_lineup_value(after_roster, per_game, settings, schedule, span, plugin=plugin)
    return after.total - before.total
