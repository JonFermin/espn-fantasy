"""The in-house NBA projection baseline: minutes x per-minute rates x context (DESIGN section 8.1, ROADMAP #43).

Importing this module registers the ``baseline_nba`` projection source for the NBA with
:func:`fm.model.projections.register_source`, the way :mod:`fm.model.value_nba` attaches ``darko``. Its rows are
per-game stat lines keyed by ESPN abbreviation (:data:`NBA_BASELINE_STATS`, checked against :data:`fm.espn.ids.FBA` at
import), never points, and :func:`fm.model.value_nba.blend_day` blends them with ESPN's and DARKO's once the module has
been imported. The weights file (``data/blend_weights.toml``, owned by ROADMAP #39) decides how much the blend trusts
them: a source it does not name is loaded and stored but left out of the blend, with a warning.

**Per-minute rates** (:func:`regressed_nba_rates`). Each count (``FGM``, ``FGA``, ``3PM``, ... ) is the player's
recency-weighted total over his stats.nba.com game-log appearances before the day, per minute played, pulled toward a
prior by ``prior_minutes`` of pseudo-evidence: DARKO's per-100 talent as a per-minute rate when DARKO has him, else the
league's minutes-weighted rate. Points and rebounds are derived from the made shots and the rebound split, so a line
always agrees with itself. A game weighs ``0.5 ** (days ago / half_life_days)``.

**Minutes** (:func:`project_nba_day`). The prior is DARKO's projected minutes, or without DARKO the player's
recency-weighted minutes per game (given enough games). Three context adjustments follow, in this order, each
recorded as a :class:`NbaMinutesAdjustment` and each bounded by :class:`NbaBaselineParams`:

1. *Teammates out.* A teammate who is out (probability ``p_out``) vacates ``p_out x`` his prior minutes. Each remaining
   teammate gets a share: the without-player split (his minutes per game in games the absent player missed, against
   games he played, from the game logs), shrunk toward a proportional share of the vacated minutes by
   ``n / (n + without_games_k)``. Shares are rescaled to conserve the vacated minutes; each player's gain is capped,
   and what the caps cut is dropped, never re-spread (the baseline errs low rather than inventing minutes). The absent
   player's *usage* (shots, free throws, turnovers, assists) is redistributed the same way: the without-player
   per-minute usage split when there is one, shrunk toward a prior from the vacated usage above what the extra minutes
   already carry, scaled by how much of a player's production the team keeps when he sits (the on/off split's
   off-court over on-court usage, :func:`absorption_from_on_off`), the bump bounded.
2. *Back-to-back rest.* On the second night of a back-to-back (``NbaDayContext.back_to_back``, from the schedule) a
   player's minutes fall by his own second-night split shrunk toward a prior that scales with his starter weight and
   age. Resting him outright is availability's business (``p_active``), not the line's. The minutes lost go to the
   bench.
3. *Blowout risk.* From the team's spread (``NbaDayContext.spreads``, the games' lines from :mod:`fm.sources.odds`), the
   chance that the margin passes ``blowout_margin`` either way times the minutes a starter loses in that case. With no
   line the adjustment is skipped and the day's notes say so. The minutes lost go to the bench.

A player's final minutes stay within ``[0, max_minutes]`` and every adjustment is a signed change in minutes, so
``prior + sum(adjustments) == minutes`` and :meth:`NbaPlayerBaseline.explain` can say where each number came from.

**Inputs are data.** :func:`project_nba_day` takes DARKO's talent rows, a :class:`NbaGameHistory` (a game-log frame)
and a :class:`NbaDayContext` and touches nothing else; only games dated before ``context.day`` are read, so a replay
never sees the future. :class:`NbaBaselineLoader` is the registered loader: it fetches the feeds (DARKO, game logs, the
schedule, the day's lines, on/off splits for the teams that need them), builds the context (who is out comes from the
``players`` table's designations through :func:`fm.model.availability.p_active_for`) and joins the result to ESPN ids
through the NBA crosswalk. A feed that cannot be fetched with nothing to fall back on is an empty ``degraded`` result
naming the reason, so the blend runs without it.

**Replaying a past day leaks.** The game logs are cut off at the day, but the loader's other inputs are read as of now:
DARKO's talent and minutes (the current projections), the on/off splits (the season to date) and the ``players``
table's injury designations. For a day before today (:func:`fm.sports.base.fantasy_day` of the clock) the result is
therefore marked ``degraded`` with a warning saying so, and a backtest on it measures a baseline that has seen the
future. Only a day that is today or ahead is clean.

The scoring period of the day is the schedule's: day 1 is opening night (:mod:`fm.sports.nba`), so period ``p`` is
the first regular-season day plus ``p - 1`` (:func:`nba_period_date`).
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from types import MappingProxyType
from typing import Final

import polars as pl

from fm.espn.ids import FBA, Game
from fm.model.availability import p_active_for
from fm.model.categories import GAMES_PLAYED
from fm.model.ids_nba import NBA_SPORT, NbaCrosswalk, nba_tricode, tricode
from fm.model.projections import Converted, register_source, source_registry
from fm.sources.base import Fetched, FetchOptions, SourceError
from fm.sources.darko import DarkoProjection, DarkoSource
from fm.sources.nba_schedule import NbaSchedule, NbaScheduleSource
from fm.sources.nba_stats import NbaStatsSource, nba_season
from fm.sources.odds import EspnScoreboardSource, Scoreboard
from fm.sports.base import fantasy_day
from fm.store import PlayerRow, ProjectionRow, Store

logger = logging.getLogger(__name__)

NBA_BASELINE_SOURCE: Final = "baseline_nba"
"""The projection source name: the ``source`` column of ``projections`` and the key in the blend weights file."""
NBA_BASELINE_LABEL: Final = "In-house NBA baseline: DARKO minutes adjusted for context x regressed per-minute rates"

NBA_BASELINE_STATS: Final = (
    "MIN",
    "PTS",
    "FGM",
    "FGA",
    "3PM",
    "3PA",
    "FTM",
    "FTA",
    "OREB",
    "DREB",
    "REB",
    "AST",
    "STL",
    "BLK",
    "TO",
    "PF",
)
"""The stats of a baseline line (ESPN ``fba`` abbreviations), plus ``GP`` = 1."""

for _abbreviation in (*NBA_BASELINE_STATS, GAMES_PLAYED):  # a typo fails at import rather than in a blend
    FBA.stat_id(_abbreviation)

NBA_RATE_COLUMNS: Final = (
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "FTM",
    "FTA",
    "OREB",
    "DREB",
    "AST",
    "STL",
    "BLK",
    "TOV",
    "PF",
)
"""The counting columns of stats.nba.com's game logs the rates are fit on (nba.com spelling)."""
NBA_LINE_STAT: Final[Mapping[str, str]] = MappingProxyType(
    {
        "FGM": "FGM",
        "FGA": "FGA",
        "FG3M": "3PM",
        "FG3A": "3PA",
        "FTM": "FTM",
        "FTA": "FTA",
        "OREB": "OREB",
        "DREB": "DREB",
        "AST": "AST",
        "STL": "STL",
        "BLK": "BLK",
        "TOV": "TO",
        "PF": "PF",
    }
)
"""nba.com game-log column to ESPN abbreviation."""
NBA_DARKO_KEYS: Final[Mapping[str, str]] = MappingProxyType(
    {
        "FGM": "fgm",
        "FGA": "fga",
        "FG3M": "fg3m",
        "FG3A": "fg3a",
        "FTM": "ftm",
        "FTA": "fta",
        "OREB": "oreb",
        "DREB": "dreb",
        "AST": "ast",
        "STL": "stl",
        "BLK": "blk",
        "TOV": "tov",
        "PF": "pf",
    }
)
"""nba.com game-log column to the key of :meth:`fm.sources.darko.DarkoProjection.per_game`."""
NBA_USAGE_COLUMNS: Final = ("FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "AST", "TOV")
"""The rates a teammate's absence moves; rebounds, steals, blocks and fouls follow minutes only."""
NBA_LOG_COLUMNS: Final = ("PLAYER_ID", "TEAM_ABBREVIATION", "GAME_ID", "GAME_DATE", "MIN", *NBA_RATE_COLUMNS)
NBA_FREE_THROW_WEIGHT: Final = 0.44
"""A free-throw attempt as a share of a possession, the usual ``FGA + 0.44 FTA + TOV`` usage estimate."""

NBA_GAME_LOG_DATASET: Final = "game_logs"


class NbaBaselineError(ValueError):
    """Game-log data the baseline cannot read, or parameters it cannot use."""


# --- parameters -------------------------------------------------------------------------------------------------------


def _bounded(name: str, value: float, low: float, high: float = math.inf) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or not low <= value <= high:
        raise NbaBaselineError(f"{name} should be a finite number within [{low}, {high}], got {value!r}")


@dataclass(frozen=True, slots=True)
class NbaBaselineParams:
    """Every knob of the baseline and the bound that limits each adjustment. The defaults are starting points to tune
    on backtests (ROADMAP #39), not fitted values; all minutes are per game."""

    half_life_days: float = 45.0
    """Recency: a game this many days old counts half as much as one played on the day."""
    prior_minutes: float = 500.0
    """Pseudo-minutes of evidence behind DARKO's per-minute rates (about 15 starter games)."""
    league_prior_minutes: float = 1000.0
    """Pseudo-minutes behind the league rate when DARKO does not have the player."""
    min_games: int = 5
    """Games of history that stand in for a missing DARKO minutes prior."""
    stale_days: int = 21
    """A player with no DARKO row is rostered for a day only if he played within this many days."""
    min_without_games: int = 3
    """Games without the absent teammate before his split counts."""
    without_games_k: float = 8.0
    """The without-player split's weight is ``n / (n + k)``."""
    max_gain: float = 10.0
    """The most minutes teammates' absences add to one player."""
    max_minutes: float = 40.0
    """No player's projected minutes pass this."""
    usage_games_k: float = 12.0
    usage_cap: float = 0.15
    """The most a per-minute usage rate rises when teammates are out."""
    usage_floor: float = -0.05
    default_absorption: float = 0.85
    """Share of an absent player's production the team keeps, when no on/off split says."""
    absorption_floor: float = 0.7
    """The on/off split's off-court over on-court usage is clamped to ``[absorption_floor, 1]``."""
    starter_floor: float = 18.0
    starter_ceiling: float = 32.0
    """Minutes at which a player is wholly bench (``starter_floor``) and wholly a starter (``starter_ceiling``)."""
    b2b_prior: float = 0.04
    """The prior share of minutes a full starter loses on the second night of a back-to-back."""
    b2b_age_extra: float = 0.03
    b2b_age: float = 33.0
    """Players at this age or older lose ``b2b_age_extra`` more."""
    b2b_games_k: float = 6.0
    b2b_cap: float = 0.15
    """The most a player's minutes fall (as a share) on the second night."""
    blowout_margin: float = 18.0
    blowout_sd: float = 12.0
    """The margin that empties benches, and the spread of one game's margin around its spread."""
    blowout_loss_win: float = 7.0
    blowout_loss_loss: float = 5.0
    """Minutes a full starter loses when his team wins or loses by ``blowout_margin`` or more."""
    blowout_cap: float = 4.0
    """The most minutes the blowout adjustment takes from one player."""
    shift_gain_cap: float = 6.0
    """The most minutes the bench absorbs per player when starters' minutes shift (back-to-back, blowout)."""

    def __post_init__(self) -> None:
        _bounded("half_life_days", self.half_life_days, 1.0)
        _bounded("prior_minutes", self.prior_minutes, 0.0)
        _bounded("league_prior_minutes", self.league_prior_minutes, 0.0)
        _bounded("min_games", self.min_games, 1.0)
        _bounded("stale_days", self.stale_days, 1.0)
        _bounded("min_without_games", self.min_without_games, 1.0)
        _bounded("without_games_k", self.without_games_k, 0.0)
        _bounded("max_gain", self.max_gain, 0.0)
        _bounded("max_minutes", self.max_minutes, 1.0, 48.0)
        _bounded("usage_games_k", self.usage_games_k, 0.0)
        _bounded("usage_cap", self.usage_cap, 0.0, 1.0)
        _bounded("usage_floor", self.usage_floor, -1.0, 0.0)
        _bounded("default_absorption", self.default_absorption, 0.0, 1.0)
        _bounded("absorption_floor", self.absorption_floor, 0.0, 1.0)
        _bounded("starter_floor", self.starter_floor, 0.0, 48.0)
        _bounded("starter_ceiling", self.starter_ceiling, self.starter_floor + 0.1, 48.0)
        _bounded("b2b_prior", self.b2b_prior, 0.0, 1.0)
        _bounded("b2b_age_extra", self.b2b_age_extra, 0.0, 1.0)
        _bounded("b2b_age", self.b2b_age, 18.0, 60.0)
        _bounded("b2b_games_k", self.b2b_games_k, 0.0)
        _bounded("b2b_cap", self.b2b_cap, 0.0, 1.0)
        _bounded("blowout_margin", self.blowout_margin, 1.0)
        _bounded("blowout_sd", self.blowout_sd, 0.5)
        _bounded("blowout_loss_win", self.blowout_loss_win, 0.0, 48.0)
        _bounded("blowout_loss_loss", self.blowout_loss_loss, 0.0, 48.0)
        _bounded("blowout_cap", self.blowout_cap, 0.0, 48.0)
        _bounded("shift_gain_cap", self.shift_gain_cap, 0.0, 48.0)


DEFAULT_NBA_PARAMS: Final = NbaBaselineParams()


# --- game logs --------------------------------------------------------------------------------------------------------


def _usage(stats: Mapping[str, float]) -> float:
    return stats["FGA"] + NBA_FREE_THROW_WEIGHT * stats["FTA"] + stats["TOV"]


@dataclass(frozen=True, slots=True)
class NbaAppearance:
    """One player-game from the game logs: who, for which team (tricode), when, how long, and the counts."""

    nba_id: int
    game_id: str
    day: date
    team: str
    minutes: float
    stats: Mapping[str, float]

    @property
    def usage(self) -> float:
        """Possessions this player used: ``FGA + 0.44 FTA + TOV``."""
        return _usage(self.stats)


@dataclass(frozen=True, slots=True)
class NbaWithoutSplit:
    """A player's games with and without an absent teammate (same team, from the teammate's first game for it)."""

    games_with: int
    games_without: int
    minutes_with: float
    minutes_without: float
    usage_with: float
    """Per-minute usage in the games the teammate played."""
    usage_without: float


@dataclass(frozen=True, slots=True)
class NbaSecondNightSplit:
    """A player's minutes on the second night of a back-to-back against his other games."""

    games_second_night: int
    games_other: int
    minutes_second_night: float
    minutes_other: float


class NbaGameHistory:
    """Player appearances from stats.nba.com game logs, with the team-game calendar they imply."""

    def __init__(self, appearances: Iterable[NbaAppearance]) -> None:
        by_player: defaultdict[int, list[NbaAppearance]] = defaultdict(list)
        for appearance in appearances:
            by_player[appearance.nba_id].append(appearance)
        for games in by_player.values():
            games.sort(key=lambda appearance: (appearance.day, appearance.game_id))
        self._by_player: dict[int, tuple[NbaAppearance, ...]] = {
            nba_id: tuple(games) for nba_id, games in by_player.items()
        }
        team_games: defaultdict[str, dict[str, date]] = defaultdict(dict)
        rosters: defaultdict[tuple[str, str], set[int]] = defaultdict(set)
        by_game: defaultdict[tuple[int, str], dict[str, NbaAppearance]] = defaultdict(dict)
        for games in self._by_player.values():
            for appearance in games:
                team_games[appearance.team][appearance.game_id] = appearance.day
                rosters[(appearance.team, appearance.game_id)].add(appearance.nba_id)
                by_game[(appearance.nba_id, appearance.team)][appearance.game_id] = appearance
        self._team_games = {
            team: dict(sorted(games.items(), key=lambda item: (item[1], item[0]))) for team, games in team_games.items()
        }
        self._rosters = rosters
        self._by_game = by_game
        self._second_nights: set[tuple[str, str]] = set()
        for team, games in self._team_games.items():
            previous: date | None = None
            for game_id, day in games.items():
                if previous is not None and (day - previous).days == 1:
                    self._second_nights.add((team, game_id))
                previous = day

    @classmethod
    def from_frame(cls, frame: pl.DataFrame) -> NbaGameHistory:
        """Appearances from a ``game_logs`` frame (:meth:`fm.sources.nba_stats.NbaStatsSource.game_logs`). Rows without
        minutes are skipped (a log row is a game the player played). Raises :class:`NbaBaselineError` for a frame
        without the columns of :data:`NBA_LOG_COLUMNS`."""
        missing = [column for column in NBA_LOG_COLUMNS if column not in frame.columns]
        if missing:
            raise NbaBaselineError(f"game log frame lacks columns {missing}")
        appearances: list[NbaAppearance] = []
        for row in frame.select(NBA_LOG_COLUMNS).iter_rows(named=True):
            minutes = row["MIN"]
            if minutes is None or not float(minutes) > 0:
                continue
            counts = {column: float(row[column] or 0.0) for column in NBA_RATE_COLUMNS}
            appearances.append(
                NbaAppearance(
                    nba_id=int(row["PLAYER_ID"]),
                    game_id=str(row["GAME_ID"]),
                    day=date.fromisoformat(str(row["GAME_DATE"])[:10]),
                    team=str(row["TEAM_ABBREVIATION"]),
                    minutes=float(minutes),
                    stats=MappingProxyType(counts),
                )
            )
        return cls(appearances)

    def __len__(self) -> int:
        return sum(len(games) for games in self._by_player.values())

    @property
    def players(self) -> tuple[int, ...]:
        return tuple(self._by_player)

    def appearances(self, nba_id: int) -> tuple[NbaAppearance, ...]:
        """The player's games, oldest first."""
        return self._by_player.get(nba_id, ())

    def all_appearances(self) -> Iterable[NbaAppearance]:
        for games in self._by_player.values():
            yield from games

    def before(self, day: date) -> NbaGameHistory:
        """The history of games dated before ``day``: what a decision for ``day`` may know."""
        return NbaGameHistory(appearance for appearance in self.all_appearances() if appearance.day < day)

    def latest_team(self, nba_id: int) -> str | None:
        games = self._by_player.get(nba_id)
        return games[-1].team if games else None

    def last_day(self, nba_id: int) -> date | None:
        games = self._by_player.get(nba_id)
        return games[-1].day if games else None

    def second_night(self, team: str, game_id: str) -> bool:
        """The team's game the day after another game of its own."""
        return (team, game_id) in self._second_nights

    def without(self, team: str, absent: int, nba_id: int) -> NbaWithoutSplit | None:
        """``nba_id``'s minutes and usage with and without ``absent``, over ``team``'s games from the first game
        ``absent`` played for it. ``None`` when ``absent`` never played for ``team`` or ``nba_id`` has no game in the
        span."""
        theirs = self._by_game.get((absent, team))
        mine = self._by_game.get((nba_id, team))
        calendar = self._team_games.get(team)
        if not theirs or not mine or not calendar:
            return None
        first = min(appearance.day for appearance in theirs.values())
        with_x: list[NbaAppearance] = []
        without_x: list[NbaAppearance] = []
        for game_id, day in calendar.items():
            appearance = mine.get(game_id)
            if day < first or appearance is None:
                continue
            (with_x if absent in self._rosters[(team, game_id)] else without_x).append(appearance)

        def mean_minutes(games: Sequence[NbaAppearance]) -> float:
            return math.fsum(game.minutes for game in games) / len(games) if games else 0.0

        def per_minute_usage(games: Sequence[NbaAppearance]) -> float:
            total = math.fsum(game.minutes for game in games)
            return math.fsum(game.usage for game in games) / total if total > 0 else 0.0

        if not with_x and not without_x:
            return None
        return NbaWithoutSplit(
            len(with_x),
            len(without_x),
            mean_minutes(with_x),
            mean_minutes(without_x),
            per_minute_usage(with_x),
            per_minute_usage(without_x),
        )

    def second_night_split(self, nba_id: int, team: str) -> NbaSecondNightSplit | None:
        """The player's minutes for ``team`` on second nights against other nights; ``None`` without games."""
        games = [appearance for appearance in self._by_player.get(nba_id, ()) if appearance.team == team]
        if not games:
            return None
        second = [game.minutes for game in games if self.second_night(team, game.game_id)]
        other = [game.minutes for game in games if not self.second_night(team, game.game_id)]
        return NbaSecondNightSplit(
            len(second),
            len(other),
            math.fsum(second) / len(second) if second else 0.0,
            math.fsum(other) / len(other) if other else 0.0,
        )


# --- per-minute rates -------------------------------------------------------------------------------------------------


def nba_recency_weight(day: date, played: date, half_life_days: float) -> float:
    """``0.5 ** (days ago / half_life_days)``; a game on or after ``day`` counts in full (callers pass earlier ones)."""
    return 0.5 ** (max(0, (day - played).days) / half_life_days)


def nba_league_rates(history: NbaGameHistory) -> dict[str, float]:
    """The league's minutes-weighted per-minute rate of every counting column: the prior of a player DARKO lacks."""
    minutes = 0.0
    totals = dict.fromkeys(NBA_RATE_COLUMNS, 0.0)
    for appearance in history.all_appearances():
        minutes += appearance.minutes
        for column in NBA_RATE_COLUMNS:
            totals[column] += appearance.stats[column]
    return {column: total / minutes for column, total in totals.items()} if minutes > 0 else {}


def darko_rates(projection: DarkoProjection) -> dict[str, float]:
    """DARKO's talent as per-minute rates by game-log column; empty for a player DARKO projects for no minutes."""
    if projection.minutes <= 0:
        return {}
    line = projection.per_game()
    return {column: line[key] / projection.minutes for column, key in NBA_DARKO_KEYS.items()}


@dataclass(frozen=True, slots=True)
class NbaRateEstimate:
    """Regressed per-minute rates by game-log column, with what they stand on: the recency-weighted ``minutes`` of
    evidence, the ``games`` behind it and the ``prior`` (``darko``, ``league`` or ``none`` without history)."""

    per_minute: Mapping[str, float]
    minutes: float
    games: int
    prior: str

    @property
    def usage(self) -> float:
        """Possessions used per minute."""
        return _usage(self.per_minute)


def regressed_nba_rates(
    appearances: Sequence[NbaAppearance],
    day: date,
    *,
    darko: Mapping[str, float],
    league: Mapping[str, float],
    params: NbaBaselineParams = DEFAULT_NBA_PARAMS,
) -> NbaRateEstimate | None:
    """Recency-weighted per-minute rates pulled toward a prior: ``(sum w x + K p) / (sum w minutes + K)`` per column,
    with ``p`` DARKO's rate (``K = prior_minutes``) or the league's (``K = league_prior_minutes``). ``None`` when there
    is neither a prior nor any game."""
    if darko:
        prior, source, weight = darko, "darko", params.prior_minutes
    elif league:
        prior, source, weight = league, "league", params.league_prior_minutes
    else:
        prior, source, weight = {}, "none", 0.0
    minutes = 0.0
    totals = dict.fromkeys(NBA_RATE_COLUMNS, 0.0)
    for appearance in appearances:
        w = nba_recency_weight(day, appearance.day, params.half_life_days)
        minutes += w * appearance.minutes
        for column in NBA_RATE_COLUMNS:
            totals[column] += w * appearance.stats[column]
    if not prior and minutes <= 0:
        return None
    rates = {
        column: (totals[column] + weight * prior.get(column, 0.0)) / (minutes + weight) for column in NBA_RATE_COLUMNS
    }
    return NbaRateEstimate(MappingProxyType(rates), minutes, len(appearances), source)


def recent_minutes(appearances: Sequence[NbaAppearance], day: date, params: NbaBaselineParams) -> float | None:
    """Recency-weighted minutes per game; ``None`` below ``min_games`` games."""
    if len(appearances) < params.min_games:
        return None
    weights = [nba_recency_weight(day, appearance.day, params.half_life_days) for appearance in appearances]
    return math.fsum(w * a.minutes for w, a in zip(weights, appearances, strict=True)) / math.fsum(weights)


def absorption_from_on_off(frame: pl.DataFrame, *, params: NbaBaselineParams = DEFAULT_NBA_PARAMS) -> dict[int, float]:
    """How much of a player's production his team keeps when he sits, by nba.com id, from a team's on/off frame
    (:meth:`fm.sources.nba_stats.NbaStatsSource.on_off`): the team's possessions used per minute with him off the court
    over with him on it (``FGA + 0.44 FTA + TOV``), clamped to ``[absorption_floor, 1]``. Players without both rows,
    or without minutes in either, are left out."""
    needed = ("VS_PLAYER_ID", "COURT_STATUS", "MIN", "FGA", "FTA", "TOV")
    missing = [column for column in needed if column not in frame.columns]
    if missing:
        raise NbaBaselineError(f"on/off frame lacks columns {missing}")
    rates: defaultdict[int, dict[str, float]] = defaultdict(dict)
    for row in frame.select(needed).iter_rows(named=True):
        minutes = row["MIN"]
        if minutes is None or not float(minutes) > 0:
            continue
        events = float(row["FGA"] or 0.0) + NBA_FREE_THROW_WEIGHT * float(row["FTA"] or 0.0) + float(row["TOV"] or 0.0)
        rates[int(row["VS_PLAYER_ID"])][str(row["COURT_STATUS"]).strip().lower()] = events / float(minutes)
    absorbed: dict[int, float] = {}
    for nba_id, sides in rates.items():
        on, off = sides.get("on"), sides.get("off")
        if on and off is not None:
            absorbed[nba_id] = min(1.0, max(params.absorption_floor, off / on))
    return absorbed


# --- the day's context ------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NbaDayContext:
    """What is known about one day: the teams that play (tricodes), who may be out, the teams' spreads and the teams on
    the second night of a back-to-back.

    ``out`` maps an nba.com id to the probability he does not play (1 for ruled out); ``spreads`` maps a tricode to
    its own spread (negative: the favorite), absent without a line; ``absorption`` maps an nba.com id to the share of
    his production his team keeps when he sits (:func:`absorption_from_on_off`).
    """

    day: date
    teams: frozenset[str]
    out: Mapping[int, float] = field(default_factory=dict)
    spreads: Mapping[str, float] = field(default_factory=dict)
    back_to_back: frozenset[str] = frozenset()
    absorption: Mapping[int, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for nba_id, probability in self.out.items():
            _bounded(f"out[{nba_id}]", probability, 0.0, 1.0)
        for team, spread in self.spreads.items():
            _bounded(f"spreads[{team}]", spread, -100.0, 100.0)
        for nba_id, share in self.absorption.items():
            _bounded(f"absorption[{nba_id}]", share, 0.0, 1.0)


# --- results ----------------------------------------------------------------------------------------------------------

MINUTES_TEAMMATES_OUT: Final = "teammates_out"
MINUTES_BACK_TO_BACK: Final = "back_to_back"
MINUTES_BLOWOUT: Final = "blowout"
MINUTES_BOUND: Final = "bound"


@dataclass(frozen=True, slots=True)
class NbaMinutesAdjustment:
    """One signed change to a player's minutes and why. ``kind`` is a ``KIND_*`` constant; a ``bound`` adjustment is the
    cut a limit made to the ones before it."""

    kind: str
    minutes: float
    detail: str


@dataclass(frozen=True, slots=True)
class NbaPlayerBaseline:
    """One player's baseline for a day: the minutes prior, the adjustments to it, the final minutes, the usage bump and
    the per-game line (ESPN abbreviations, ``GP`` 1) they give."""

    nba_id: int
    name: str
    team: str
    prior_minutes: float
    minutes: float
    adjustments: tuple[NbaMinutesAdjustment, ...]
    usage_bump: float
    usage_detail: str
    rates: NbaRateEstimate
    line: Mapping[str, float]

    def explain(self) -> str:
        """The minutes from prior to final and the rate behind the line, one adjustment a line."""
        lines = [
            f"{self.name} ({self.team}): {self.minutes:.1f} min from a prior of {self.prior_minutes:.1f}; "
            f"rates from {self.rates.games} games over a {self.rates.prior} prior"
        ]
        lines.extend(
            f"  {adjustment.minutes:+.2f} min {adjustment.kind}: {adjustment.detail}" for adjustment in self.adjustments
        )
        if self.usage_bump:
            lines.append(f"  usage {self.usage_bump:+.1%}: {self.usage_detail}")
        return "\n".join(lines)


@dataclass(frozen=True, slots=True)
class NbaDayProjection:
    """:func:`project_nba_day`'s result: a :class:`NbaPlayerBaseline` per nba.com id and what was skipped."""

    day: date
    players: Mapping[int, NbaPlayerBaseline]
    notes: tuple[str, ...] = ()


# --- the minutes model ------------------------------------------------------------------------------------------------


@dataclass(slots=True)
class _Entry:
    """A rostered player while a team's day is worked out."""

    nba_id: int
    name: str
    team: str
    prior: float
    p_out: float
    age: float | None
    rates: NbaRateEstimate
    minutes: float = 0.0
    adjustments: list[NbaMinutesAdjustment] = field(default_factory=list)
    bump: float = 0.0
    bump_notes: list[str] = field(default_factory=list)


def _clamp(value: float, low: float, high: float) -> float:
    return min(high, max(low, value))


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def starter_weight(minutes: float, params: NbaBaselineParams = DEFAULT_NBA_PARAMS) -> float:
    """0 for a player at ``starter_floor`` minutes or fewer, 1 at ``starter_ceiling`` or more, linear between."""
    return _clamp((minutes - params.starter_floor) / (params.starter_ceiling - params.starter_floor), 0.0, 1.0)


def blowout_probabilities(spread: float, params: NbaBaselineParams = DEFAULT_NBA_PARAMS) -> tuple[float, float]:
    """The chance a team wins, and the chance it loses, by ``blowout_margin`` or more, from its spread (negative: the
    favorite): the margin is normal around ``-spread`` with ``blowout_sd``."""
    expected = -spread
    wins = 1.0 - _normal_cdf((params.blowout_margin - expected) / params.blowout_sd)
    losses = _normal_cdf((-params.blowout_margin - expected) / params.blowout_sd)
    return wins, losses


def _teammates_out(
    team: list[_Entry], history: NbaGameHistory, context: NbaDayContext, params: NbaBaselineParams
) -> None:
    """Redistribute the minutes (and usage) of teammates who may be out, in place. See the module docstring."""
    absent = [entry for entry in team if entry.p_out > 0 and entry.prior > 0]
    if not absent:
        return
    candidates = [entry for entry in team if entry.p_out < 1]
    gains: dict[int, float] = defaultdict(float)
    bumps: dict[int, float] = defaultdict(float)
    for gone in absent:
        others = [entry for entry in candidates if entry.nba_id != gone.nba_id]
        weights = {entry.nba_id: entry.prior * (1.0 - entry.p_out) for entry in others}
        total_weight = math.fsum(weights.values())
        if not others or total_weight <= 0:
            continue
        vacated = gone.prior
        raw: dict[int, tuple[float, NbaWithoutSplit | None, float]] = {}
        for entry in others:
            split = history.without(entry.team, gone.nba_id, entry.nba_id)
            proportional = vacated * weights[entry.nba_id] / total_weight
            share = 0.0
            if split is not None and split.games_without >= params.min_without_games and split.games_with >= 1:
                share = split.games_without / (split.games_without + params.without_games_k)
                observed = _clamp(split.minutes_without - split.minutes_with, 0.0, params.max_gain)
                raw[entry.nba_id] = (share * observed + (1.0 - share) * proportional, split, share)
            else:
                raw[entry.nba_id] = (proportional, split, 0.0)
        scale = vacated / math.fsum(minutes for minutes, _, _ in raw.values())
        for entry in others:
            minutes, split, share = raw[entry.nba_id]
            gained = gone.p_out * minutes * scale
            gains[entry.nba_id] += gained
            how = (
                f"without-split over {split.games_without} games, weight {share:.2f}"
                if split is not None and share > 0
                else "proportional share"
            )
            entry.adjustments.append(
                NbaMinutesAdjustment(
                    MINUTES_TEAMMATES_OUT, gained, f"{gone.name} out (p={gone.p_out:.2f}, {gone.prior:.1f} min): {how}"
                )
            )
        _usage_bumps(gone, others, history, context, params, bumps)
    for entry in candidates:
        applied = gains.get(entry.nba_id, 0.0)
        if applied > 0:
            room = max(0.0, params.max_minutes - entry.prior)
            limit = min(params.max_gain, room)
            if applied > limit:
                entry.adjustments.append(
                    NbaMinutesAdjustment(
                        MINUTES_BOUND, limit - applied, f"teammates-out gain {applied:.1f} cut to {limit:.1f} (cap)"
                    )
                )
        total = bumps.get(entry.nba_id, 0.0)
        entry.bump = _clamp(total, params.usage_floor, params.usage_cap)
        if entry.bump != total:
            entry.bump_notes.append(f"bounded from {total:+.1%}")


def _usage_bumps(
    gone: _Entry,
    others: list[_Entry],
    history: NbaGameHistory,
    context: NbaDayContext,
    params: NbaBaselineParams,
    bumps: dict[int, float],
) -> None:
    """Add to ``bumps`` the per-minute usage each of ``others`` takes on from ``gone``'s absence."""
    absorption = context.absorption.get(gone.nba_id, params.default_absorption)
    minutes_total = math.fsum(entry.prior for entry in others)
    events_total = math.fsum(entry.rates.usage * entry.prior for entry in others)
    if minutes_total <= 0 or events_total <= 0:
        return
    average = events_total / minutes_total
    residual = gone.p_out * gone.prior * max(0.0, gone.rates.usage - average)
    prior_bump = absorption * residual / events_total
    for entry in others:
        split = history.without(entry.team, gone.nba_id, entry.nba_id)
        bump = prior_bump
        how = f"vacated usage x absorption {absorption:.2f}"
        if (
            split is not None
            and split.games_without >= params.min_without_games
            and split.games_with >= 1
            and split.usage_with > 0
        ):
            share = split.games_without / (split.games_without + params.usage_games_k)
            observed = gone.p_out * (split.usage_without / split.usage_with - 1.0)
            bump = share * observed + (1.0 - share) * prior_bump
            how = f"without-split over {split.games_without} games, weight {share:.2f}"
        bumps[entry.nba_id] = bumps.get(entry.nba_id, 0.0) + bump
        entry.bump_notes.append(f"{gone.name} out: {bump:+.1%} ({how})")


def _shift_to_bench(
    team: list[_Entry], losses: Mapping[int, float], kind: str, why: Mapping[int, str], params: NbaBaselineParams
) -> None:
    """Take ``losses`` (minutes, by nba.com id) from their players and give them to the bench in proportion to bench
    minutes, each gain capped at ``shift_gain_cap`` and ``max_minutes``; excess is dropped."""
    taken = math.fsum(losses.values())
    if taken <= 0:
        return
    weights = {
        entry.nba_id: entry.minutes * (1.0 - starter_weight(entry.minutes, params))
        for entry in team
        if entry.p_out < 1 and entry.nba_id not in losses
    }
    total = math.fsum(weights.values())
    if total <= 0:
        return
    for entry in team:
        lost = losses.get(entry.nba_id, 0.0)
        if lost > 0:
            entry.adjustments.append(NbaMinutesAdjustment(kind, -lost, why[entry.nba_id]))
        weight = weights.get(entry.nba_id, 0.0)
        if weight > 0:
            gained = taken * weight / total
            limit = min(params.shift_gain_cap, max(0.0, params.max_minutes - entry.minutes))
            applied = min(gained, limit)
            entry.adjustments.append(
                NbaMinutesAdjustment(kind, applied, f"picks up {applied:.1f} of the {taken:.1f} min the starters lose")
            )
    for entry in team:
        entry.minutes = _current(entry)


def _current(entry: _Entry) -> float:
    return entry.prior + math.fsum(adjustment.minutes for adjustment in entry.adjustments)


def _back_to_back(team: list[_Entry], history: NbaGameHistory, params: NbaBaselineParams) -> None:
    losses: dict[int, float] = {}
    why: dict[int, str] = {}
    for entry in team:
        if entry.p_out >= 1 or entry.minutes <= 0:
            continue
        weight = starter_weight(entry.minutes, params)
        prior = params.b2b_prior * weight + (
            params.b2b_age_extra * weight if (entry.age or 0) >= params.b2b_age else 0.0
        )
        split = history.second_night_split(entry.nba_id, entry.team)
        loss = prior
        how = f"prior {prior:.1%}"
        if split is not None and split.games_second_night >= 1 and split.games_other >= 1 and split.minutes_other > 0:
            observed = 1.0 - split.minutes_second_night / split.minutes_other
            share = split.games_second_night / (split.games_second_night + params.b2b_games_k)
            loss = share * observed + (1.0 - share) * prior
            how = f"{split.games_second_night} second nights, observed {observed:+.1%} against prior {prior:.1%}"
        loss = _clamp(loss, 0.0, params.b2b_cap)
        if loss > 0:
            losses[entry.nba_id] = loss * entry.minutes
            why[entry.nba_id] = f"second night of a back-to-back: {loss:.1%} ({how})"
    _shift_to_bench(team, losses, MINUTES_BACK_TO_BACK, why, params)


def _blowout(team: list[_Entry], spread: float, params: NbaBaselineParams) -> None:
    wins, losses_p = blowout_probabilities(spread, params)
    expected = wins * params.blowout_loss_win + losses_p * params.blowout_loss_loss
    losses: dict[int, float] = {}
    why: dict[int, str] = {}
    for entry in team:
        if entry.p_out >= 1:
            continue
        loss = min(starter_weight(entry.minutes, params) * expected, params.blowout_cap, entry.minutes)
        if loss > 0:
            losses[entry.nba_id] = loss
            why[entry.nba_id] = (
                f"spread {spread:+.1f}: P(win by {params.blowout_margin:.0f}+)={wins:.2f}, "
                f"P(lose by {params.blowout_margin:.0f}+)={losses_p:.2f}"
            )
    _shift_to_bench(team, losses, MINUTES_BLOWOUT, why, params)


def _final_bound(entry: _Entry, params: NbaBaselineParams) -> None:
    """Keep the final minutes within ``[0, max_minutes]``, recording the cut."""
    current = _current(entry)
    bounded = _clamp(current, 0.0, params.max_minutes)
    if bounded != current:
        entry.adjustments.append(
            NbaMinutesAdjustment(
                MINUTES_BOUND, bounded - current, f"minutes {current:.1f} held within [0, {params.max_minutes:.0f}]"
            )
        )
    entry.minutes = bounded


def _line(entry: _Entry) -> dict[str, float]:
    """The per-game line: rates (usage columns raised by the bump) times minutes, points and rebounds derived."""
    rates = {
        column: rate * (1.0 + entry.bump if column in NBA_USAGE_COLUMNS else 1.0)
        for column, rate in entry.rates.per_minute.items()
    }
    counts = {column: rates[column] * entry.minutes for column in NBA_RATE_COLUMNS}
    line = {NBA_LINE_STAT[column]: counts[column] for column in NBA_RATE_COLUMNS}
    line["MIN"] = entry.minutes
    line["PTS"] = 2.0 * counts["FGM"] + counts["FG3M"] + counts["FTM"]
    line["REB"] = counts["OREB"] + counts["DREB"]
    line[GAMES_PLAYED] = 1.0
    return {stat: line[stat] for stat in (*NBA_BASELINE_STATS, GAMES_PLAYED)}


def project_nba_day(
    talent: Iterable[DarkoProjection],
    history: NbaGameHistory,
    context: NbaDayContext,
    *,
    params: NbaBaselineParams = DEFAULT_NBA_PARAMS,
) -> NbaDayProjection:
    """The baseline's per-game line for every player of a team that plays on ``context.day``.

    ``talent`` is DARKO's projections (minutes prior and per-100 talent), ``history`` the game logs (only games before
    the day are read). A player is left out when he is ruled out (``p_out`` 1, or DARKO marks him unavailable), when his
    team does not play, or when neither DARKO nor his history gives a minutes prior; each is counted in ``notes``, and
    so is the context that was missing (no line for a team, no on/off split). Every adjustment is bounded by ``params``
    and recorded on the result.
    """
    known = history.before(context.day)
    league = nba_league_rates(known)
    notes: list[str] = []
    by_team: defaultdict[str, list[_Entry]] = defaultdict(list)
    seen: set[int] = set()
    unplaced = 0
    no_minutes = 0
    for projection in talent:
        team = (nba_tricode(projection.team_id) if projection.team_id is not None else None) or known.latest_team(
            projection.nba_id
        )
        seen.add(projection.nba_id)
        if team is None or team not in context.teams:
            unplaced += 1
            continue
        games = known.appearances(projection.nba_id)
        p_out = 1.0 if not projection.available or projection.minutes <= 0 else context.out.get(projection.nba_id, 0.0)
        prior = projection.minutes if projection.minutes > 0 else recent_minutes(games, context.day, params)
        rates = regressed_nba_rates(games, context.day, darko=darko_rates(projection), league=league, params=params)
        if prior is None or prior <= 0 or rates is None:
            no_minutes += p_out < 1
            continue
        by_team[team].append(_Entry(projection.nba_id, projection.name, team, prior, p_out, projection.age, rates))
    for nba_id in known.players:
        team = known.latest_team(nba_id)
        last = known.last_day(nba_id)
        if nba_id in seen or team is None or last is None or team not in context.teams:
            continue
        if (context.day - last).days > params.stale_days:
            continue
        games = known.appearances(nba_id)
        prior = recent_minutes(games, context.day, params)
        rates = regressed_nba_rates(games, context.day, darko={}, league=league, params=params)
        if prior is None or rates is None:
            continue
        p_out = context.out.get(nba_id, 0.0)
        by_team[team].append(_Entry(nba_id, f"nba {nba_id}", team, prior, p_out, None, rates))
    if no_minutes:
        notes.append(f"{no_minutes} players have no minutes prior (no DARKO minutes and too few games); left out")
    skipped_lines = sorted(team for team in by_team if team not in context.spreads)
    if skipped_lines:
        notes.append(f"blowout risk skipped for {', '.join(skipped_lines)}: no line")
    players: dict[int, NbaPlayerBaseline] = {}
    for team_name, team in sorted(by_team.items()):
        for entry in team:
            entry.minutes = entry.prior
        _teammates_out(team, known, context, params)
        for entry in team:
            entry.minutes = _current(entry)
            _final_bound(entry, params)
        if team_name in context.back_to_back:
            _back_to_back(team, known, params)
        if team_name in context.spreads:
            _blowout(team, context.spreads[team_name], params)
        for entry in team:
            _final_bound(entry, params)
            if entry.p_out >= 1 or entry.minutes <= 0:
                continue
            players[entry.nba_id] = NbaPlayerBaseline(
                nba_id=entry.nba_id,
                name=entry.name,
                team=entry.team,
                prior_minutes=entry.prior,
                minutes=entry.minutes,
                adjustments=tuple(entry.adjustments),
                usage_bump=entry.bump,
                usage_detail="; ".join(entry.bump_notes),
                rates=entry.rates,
                line=MappingProxyType(_line(entry)),
            )
    if unplaced and not players:
        notes.append(f"no talent row belongs to a team that plays on {context.day}")
    return NbaDayProjection(context.day, MappingProxyType(players), tuple(notes))


# --- feeds ------------------------------------------------------------------------------------------------------------


def nba_period_date(schedule: NbaSchedule, period: int) -> date | None:
    """The Eastern day of scoring period ``period``: day 1 is the first regular-season game day (:mod:`fm.sports.nba`).
    ``None`` for a period below 1 or a schedule without regular-season games."""
    games = schedule.regular_season()
    if period < 1 or not games:
        return None
    return min(game.day for game in games) + timedelta(days=period - 1)


def nba_schedule_context(schedule: NbaSchedule, day: date) -> tuple[frozenset[str], frozenset[str]]:
    """The tricodes that play on ``day`` and, among them, the ones on the second night of a back-to-back."""
    games = schedule.games_on(day)
    teams = frozenset(code for game in games for code in game.tricodes)
    second = frozenset(
        code for game in games for code in game.tricodes if schedule.is_second_of_back_to_back(game, code)
    )
    return teams, second


def nba_board_spreads(board: Scoreboard) -> dict[str, float]:
    """Each team's own spread (negative: the favorite) from a scoreboard's lines, by nba.com tricode. A game with no
    posted line, or whose teams the tricode table does not know, has no entry."""
    spreads: dict[str, float] = {}
    for game in board.games:
        if game.line is None or game.line.spread is None:
            continue
        home, away = tricode(game.home.abbreviation), tricode(game.away.abbreviation)
        if home is None or away is None:
            continue
        spreads[home] = game.line.spread
        spreads[away] = -game.line.spread
    return spreads


def nba_out_probabilities(players: Iterable[PlayerRow], crosswalk: NbaCrosswalk) -> dict[int, float]:
    """The chance each player with an ESPN designation is out, by nba.com id: ``1 - p_active_for`` of his
    ``injury_status`` (:func:`fm.model.availability.p_active_for`), and 1 for an inactive player. Players the crosswalk
    does not map and players with nothing to report are left out."""
    out: dict[int, float] = {}
    for player in players:
        nba_id = crosswalk.nba_id(player.espn_id)
        if nba_id is None:
            continue
        probability = 1.0 if not player.active else 1.0 - p_active_for(player.injury_status, Game.FBA)
        if probability > 0:
            out[nba_id] = min(1.0, probability)
    return out


def _close(adapter: object) -> None:
    close = getattr(adapter, "close", None)
    if callable(close):
        close()


class NbaBaselineLoader:
    """The ``baseline_nba`` source's loader: the day's baseline lines as ``baseline_nba`` rows keyed by ESPN id.

    The feeds default to fresh adapters over the cache dir (closed after each call, serving their cached data while it
    is fresh); inject ``talent``, ``stats``, ``schedule`` or ``lines`` to replace one, and ``crosswalk`` to replace the
    NBA crosswalk saved in the store. DARKO, the game logs and the schedule are required, and a day without them is an
    empty ``degraded`` result naming the reason. The lines (blowout risk) and the on/off splits (usage absorption, one
    pull per team that has a teammate out) are optional: without them the adjustment is skipped or the default
    absorption used, with a warning. After a call, :attr:`explanations` holds every player's :class:`NbaPlayerBaseline`
    by ESPN id.
    """

    def __init__(
        self,
        *,
        talent: DarkoSource | None = None,
        stats: NbaStatsSource | None = None,
        schedule: NbaScheduleSource | None = None,
        lines: EspnScoreboardSource | None = None,
        crosswalk: NbaCrosswalk | None = None,
        params: NbaBaselineParams = DEFAULT_NBA_PARAMS,
    ) -> None:
        self.talent = talent
        self.stats = stats
        self.schedule = schedule
        self.lines = lines
        self.crosswalk = crosswalk
        self.params = params
        self.explanations: dict[int, NbaPlayerBaseline] = {}

    def __call__(
        self, store: Store, season: int, scoring_period: int, options: FetchOptions
    ) -> Fetched[tuple[ProjectionRow, ...]]:
        talent = self.talent if self.talent is not None else DarkoSource()
        stats = self.stats if self.stats is not None else NbaStatsSource()
        calendar = self.schedule if self.schedule is not None else NbaScheduleSource()
        lines = self.lines if self.lines is not None else EspnScoreboardSource()
        try:
            return self._load(store, season, scoring_period, options, talent, stats, calendar, lines)
        finally:
            for adapter, injected in (
                (talent, self.talent),
                (stats, self.stats),
                (calendar, self.schedule),
                (lines, self.lines),
            ):
                if injected is None:
                    _close(adapter)

    def _empty(self, season: int, period: int, as_of: datetime, reason: str) -> Fetched[tuple[ProjectionRow, ...]]:
        self.explanations = {}
        return Fetched(
            (),
            as_of,
            NBA_BASELINE_SOURCE,
            NBA_GAME_LOG_DATASET,
            f"{NBA_SPORT}_{season}_{period}",
            degraded=True,
            warnings=(f"{NBA_BASELINE_SOURCE}: unavailable, blending without it ({reason})",),
        )

    def _load(
        self,
        store: Store,
        season: int,
        period: int,
        options: FetchOptions,
        talent: DarkoSource,
        stats: NbaStatsSource,
        calendar: NbaScheduleSource,
        lines: EspnScoreboardSource,
    ) -> Fetched[tuple[ProjectionRow, ...]]:
        now = talent.clock()
        try:
            darko = talent.projections(**options)
            logs = stats.game_logs(nba_season(season), **options)
            games = calendar.schedule(**options)
        except SourceError as exc:
            return self._empty(season, period, now, str(exc))
        day = nba_period_date(games.data, period)
        if day is None:
            return self._empty(season, period, now, f"the schedule has no regular-season day for period {period}")
        try:
            history = NbaGameHistory.from_frame(logs.data)
        except NbaBaselineError as exc:
            return self._empty(season, period, now, str(exc))
        feeds = [darko, logs, games]
        warnings = [warning for feed in feeds for warning in feed.warnings]
        teams, second = nba_schedule_context(games.data, day)
        crosswalk = self.crosswalk if self.crosswalk is not None else NbaCrosswalk.from_store(store)
        spreads: dict[str, float] = {}
        board = None
        try:
            board = lines.scoreboard(Game.FBA, day=day, **options)
        except SourceError as exc:
            warnings.append(f"{NBA_BASELINE_SOURCE}: no lines for {day}, blowout risk skipped ({exc})")
        if board is not None:
            feeds.append(board)
            warnings.extend(board.warnings)
            spreads = nba_board_spreads(board.data)
        out = nba_out_probabilities(store.players.many(NBA_SPORT, crosswalk.mapped()), crosswalk)
        absorption = self._absorption(stats, season, options, darko.data, out, teams, warnings)
        context = NbaDayContext(day, teams, out, spreads, second, absorption)
        projected = project_nba_day(darko.data, history, context, params=self.params)
        warnings.extend(f"{NBA_BASELINE_SOURCE}: {note}" for note in projected.notes)
        today = fantasy_day(now)
        replay = day < today
        if replay:
            warnings.append(
                f"{NBA_BASELINE_SOURCE}: period {period} is {day}, before today ({today}): DARKO talent and minutes, "
                "the on/off splits and the injury designations are read as of now, so this replay can see the future"
            )
        as_of = min(feed.as_of for feed in feeds)
        converted = nba_baseline_rows(projected, crosswalk, season=season, scoring_period=period, as_of=as_of)
        if converted.unmapped and not converted.rows:
            warnings.append(
                f"{NBA_BASELINE_SOURCE}: the NBA crosswalk maps none of {len(converted.unmapped)} players; "
                "run fm sync to save it"
            )
        explanations = {
            espn_id: projected.players[int(nba_id)]
            for row in converted.rows
            if (nba_id := crosswalk.nba_id(row.espn_id)) is not None
            for espn_id in (row.espn_id,)
        }
        self.explanations = explanations
        return Fetched(
            converted.rows,
            as_of,
            NBA_BASELINE_SOURCE,
            NBA_GAME_LOG_DATASET,
            f"{NBA_SPORT}_{season}_{period}",
            cached=all(feed.cached for feed in feeds),
            stale=any(feed.stale for feed in feeds),
            degraded=replay or any(feed.degraded for feed in feeds[:3]),
            warnings=tuple(warnings),
        )

    def _absorption(
        self,
        stats: NbaStatsSource,
        season: int,
        options: FetchOptions,
        talent: Sequence[DarkoProjection],
        out: Mapping[int, float],
        teams: frozenset[str],
        warnings: list[str],
    ) -> dict[int, float]:
        """On/off absorption for the teams that play and have a rostered teammate who may be out, one pull each."""
        wanted: dict[int, str] = {}
        for projection in talent:
            code = nba_tricode(projection.team_id)
            if code in teams and projection.team_id is not None and out.get(projection.nba_id, 0.0) > 0:
                wanted[projection.team_id] = code
        absorption: dict[int, float] = {}
        for team_id, code in sorted(wanted.items()):
            try:
                frame = stats.on_off(team_id, nba_season(season), **options)
                absorption.update(absorption_from_on_off(frame.data, params=self.params))
            except (SourceError, NbaBaselineError) as exc:
                warnings.append(f"{NBA_BASELINE_SOURCE}: no on/off split for {code}, default absorption used ({exc})")
        return absorption


def nba_baseline_rows(
    projection: NbaDayProjection,
    crosswalk: NbaCrosswalk,
    *,
    season: int,
    scoring_period: int,
    as_of: datetime,
) -> Converted:
    """A day's baseline as ``baseline_nba`` rows keyed by ESPN id through the crosswalk; players it does not map are
    listed in ``unmapped`` by nba.com id."""
    rows: list[ProjectionRow] = []
    unmapped: list[str] = []
    for nba_id, baseline in projection.players.items():
        espn_id = crosswalk.espn_id(nba_id)
        if espn_id is None:
            unmapped.append(str(nba_id))
            continue
        rows.append(
            ProjectionRow(
                sport=NBA_SPORT,
                espn_id=espn_id,
                source=NBA_BASELINE_SOURCE,
                kind="projected",
                season=season,
                scoring_period_id=scoring_period,
                stats=dict(baseline.line),
                as_of=as_of,
            )
        )
    return Converted(tuple(rows), tuple(unmapped), tuple(f"{NBA_BASELINE_SOURCE}: {note}" for note in projection.notes))


def _register() -> None:
    """Register ``baseline_nba`` for the NBA once; importing the module twice changes nothing."""
    if source_registry.get(NBA_SPORT, NBA_BASELINE_SOURCE) is not None:
        return
    register_source(NBA_SPORT, NBA_BASELINE_SOURCE, label=NBA_BASELINE_LABEL, loader=NbaBaselineLoader())


_register()
