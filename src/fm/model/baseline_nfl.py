"""NFL opportunity baseline (ROADMAP #42, DESIGN section 8.1): opportunity shares x implied team total x regressed
efficiency, registered as the ``opportunity`` projection source for ``nfl``.

The baseline is a stat-line source like ESPN's and Sleeper's: one :class:`fm.store.ProjectionRow` per player and week,
keyed by ESPN stat abbreviation (:data:`BASELINE_STATS`, checked against :mod:`fm.espn.ids` at import) and never points;
each league scores the lines with its own items. It projects quarterbacks, running backs, wide receivers and tight
ends, conditional on the player playing (``p_active`` is :mod:`fm.model.availability`'s job). Kickers and defenses stay
with ESPN and Sleeper, and a team on bye gets no rows.

**The model**, for a player of team ``t`` in week ``w``, from games before ``w`` only:

1. *Team volume.* Pass attempts, targets, carries and air yards of ``t`` are recency-weighted means (``decay`` per
   week) of its past games, each divided by that game's environment factor: the game's implied total over the league
   average, raised to an elasticity (:class:`OpportunityConfig`; volume follows the script weakly, touchdown rates
   strongly). A shootout then does not make a team look pass-happy. The mean is shrunk toward the league's, and the
   projection multiplies it by this week's factor.
2. *Opportunity shares.* The player's share of each team volume per appearance (attempts, carries, targets, air
   yards), recency-weighted and shrunk toward the position's mean share. His offensive snap-share trend (the last two
   games' snap percentage against his weighted mean, clamped) scales the non-quarterback shares, so a role that grew
   last week counts before the shares catch up.
3. *Regressed efficiency.* Completion rate, yards per attempt, yards per carry, catch rate, yards per target, yards per
   air yard, touchdown and interception rates and fumbles lost, each shrunk toward the position's pooled rate with a
   prior weight in opportunities (:class:`EfficiencyPriors`). Touchdown rates are environment-normalised like volume.
   Receiving yards for wide receivers and tight ends average two paths: targets x yards per target, and his share of
   the team's air yards x yards per air yard.

**Inputs.** History is nflverse weekly stats and snap counts (:mod:`fm.sources.nflverse`; snaps join through the
``ff_playerids`` PFR id) for the season and the one before, and the schedules' lines for the games already played
(spread and total become implied totals the way :func:`fm.sources.odds.implied_team_totals` does it; nflverse's
``spread_line`` is positive when the home side is favored, ESPN's ``spread`` negative). This week's totals come from
:mod:`fm.sources.odds`: ESPN's scoreboard.

**Pregame totals.** The scoreboard's ``odds`` block disappears at kickoff, so a line exists only while
``ScoreboardGame.state == "pre"``, and its cache lasts ten minutes. A run after a game started cannot read the number
from the source, so :class:`PregameSnapshot` keeps it: whenever the loader (or a caller of
:func:`capture_pregame_lines`, which a sync or tick can run right after any scoreboard read) sees a ``pre`` game with
a line, it saves both implied totals under ``cache_dir()/baseline_nfl/``. :func:`week_lines` then takes, per team, the
live scoreboard line, else the snapshot, else the nflverse schedule's line, else the league-average total with a
warning. Nothing writes a snapshot unless this code runs before kickoff (the sync job is not changed), so the schedule
line is the safety net after the fact.

**Registration.** Importing this module registers :class:`OpportunityLoader` as ``("nfl", "opportunity")`` on
:data:`fm.model.projections.source_registry`; naming ``opportunity`` in ``data/blend_weights.toml`` weights it. The
loader never raises: a failed dataset is a warning, and with nothing to project from the result is an empty
``degraded`` one, so the blend runs on the other sources.
"""

from __future__ import annotations

import json
import logging
import math
import os
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal

import polars as pl

from fm import paths
from fm.config import Sport
from fm.espn.ids import FFL, Game
from fm.model.ids import GSIS, Crosswalk, team_code
from fm.model.projections import register_source, source_registry
from fm.sources.base import Fetched, FetchOptions, SourceError, utcnow
from fm.sources.nflverse import NflverseSource
from fm.sources.odds import EspnScoreboardSource, Scoreboard, implied_team_totals
from fm.store import ProjectionRow, Store

logger = logging.getLogger(__name__)

OPPORTUNITY: Final = "opportunity"
"""The source name: the ``source`` column of ``projections`` and the key in ``data/blend_weights.toml``."""
OPPORTUNITY_LABEL: Final = "NFL opportunity baseline (shares x implied team total x regressed efficiency)"
BASELINE_SPORT: Final[Sport] = "nfl"
BASELINE_POSITIONS: Final[tuple[str, ...]] = ("QB", "RB", "WR", "TE")
BASELINE_STATS: Final[tuple[str, ...]] = (
    "PA",
    "PC",
    "PY",
    "PTD",
    "INTT",
    "RA",
    "RY",
    "RTD",
    "RET",
    "REC",
    "REY",
    "RETD",
    "FUML",
)
"""The ESPN stat abbreviations a line may carry. Derived stats (every-N yardage, brackets) are the blend's to fill."""
for _abbr in BASELINE_STATS:
    FFL.stat_id(_abbr)  # KeyError at import if the id map ever renames one

POSITION_ALIASES: Final[Mapping[str, str]] = MappingProxyType({"FB": "RB"})
SEASON_SPAN: Final = 20
"""Week slots per season when measuring how many weeks ago a game was (regular season weeks plus slack)."""
MIN_POOL: Final = 200.0
"""Opportunities a position must have pooled in the history before its pooled rate replaces :data:`DEFAULT_RATES`."""
MIN_SNAP_GAMES: Final = 3
MIN_LINE_VALUE: Final = 0.005
SNAPSHOT_DIRNAME: Final = "baseline_nfl"
SNAPSHOT_FILENAME: Final = "pregame_lines.json"
DATASET: Final = "opportunity"
REGULAR_SEASON: Final = "REG"

type LineOrigin = Literal["scoreboard", "snapshot", "schedule", "default"]


@dataclass(frozen=True, slots=True)
class EfficiencyPriors:
    """Prior weights, in opportunities (attempts, carries, targets, air yards, touches), of each regressed rate: how
    many pooled-position opportunities a player's own rate is worth before his history dominates."""

    completion: float = 150.0
    yards_per_attempt: float = 150.0
    pass_td: float = 300.0
    interception: float = 300.0
    yards_per_carry: float = 120.0
    rush_td: float = 200.0
    catch_rate: float = 80.0
    yards_per_target: float = 100.0
    receiving_td: float = 250.0
    yards_per_air_yard: float = 250.0
    fumble: float = 500.0

    def weight(self, rate: str) -> float:
        return float(getattr(self, rate))


@dataclass(frozen=True, slots=True)
class OpportunityConfig:
    """The model's constants. The defaults are modelling priors (long-run NFL structure), not fitted to any fixture;
    ROADMAP #39 may refit them against backtests."""

    decay: float = 0.92
    """Weekly recency weight: a game ``n`` weeks ago counts ``decay ** (n - 1)`` (a half-life of about 8 weeks)."""
    horizon_weeks: int = 40
    league_avg_total: float = 22.5
    """Points per team per game that stands for "an average environment" (also the fallback total when none is known).
    The loader replaces it with the mean of the history's implied totals when it has any."""
    pass_volume_elasticity: float = 0.4
    rush_volume_elasticity: float = 0.2
    pass_td_elasticity: float = 0.6
    """Touchdown rates move with the environment so that touchdowns per game scale about one for one with the total:
    volume elasticity plus rate elasticity is about 1."""
    rush_td_elasticity: float = 0.8
    environment_clamp: tuple[float, float] = (0.6, 1.6)
    team_prior_games: float = 3.0
    share_prior_games: float = 2.0
    snap_clamp: tuple[float, float] = (0.8, 1.25)
    air_yards_path_weight: float = 0.5
    """Weight of the air-yards path in a wide receiver's or tight end's receiving yards."""
    min_touches: float = 1.0
    priors: EfficiencyPriors = field(default_factory=EfficiencyPriors)


DEFAULT_CONFIG: Final = OpportunityConfig()

DEFAULT_RATES: Final[Mapping[str, Mapping[str, float]]] = MappingProxyType(
    {
        "QB": MappingProxyType(
            {
                "completion": 0.645,
                "yards_per_attempt": 6.9,
                "pass_td": 0.045,
                "interception": 0.024,
                "yards_per_carry": 5.2,
                "rush_td": 0.035,
                "catch_rate": 0.65,
                "yards_per_target": 7.0,
                "receiving_td": 0.04,
                "yards_per_air_yard": 1.0,
                "fumble": 0.010,
            }
        ),
        "RB": MappingProxyType(
            {
                "completion": 0.6,
                "yards_per_attempt": 6.0,
                "pass_td": 0.04,
                "interception": 0.03,
                "yards_per_carry": 4.3,
                "rush_td": 0.032,
                "catch_rate": 0.78,
                "yards_per_target": 6.2,
                "receiving_td": 0.03,
                "yards_per_air_yard": 2.0,
                "fumble": 0.007,
            }
        ),
        "WR": MappingProxyType(
            {
                "completion": 0.6,
                "yards_per_attempt": 6.0,
                "pass_td": 0.04,
                "interception": 0.03,
                "yards_per_carry": 6.5,
                "rush_td": 0.03,
                "catch_rate": 0.65,
                "yards_per_target": 8.4,
                "receiving_td": 0.055,
                "yards_per_air_yard": 1.05,
                "fumble": 0.004,
            }
        ),
        "TE": MappingProxyType(
            {
                "completion": 0.6,
                "yards_per_attempt": 6.0,
                "pass_td": 0.04,
                "interception": 0.03,
                "yards_per_carry": 3.0,
                "rush_td": 0.03,
                "catch_rate": 0.7,
                "yards_per_target": 7.4,
                "receiving_td": 0.06,
                "yards_per_air_yard": 1.2,
                "fumble": 0.004,
            }
        ),
    }
)
"""Typical NFL rates by position, used for a rate the history pools too few opportunities to estimate."""

type Volume = Literal["pass_att", "targets", "carries", "air_yards"]
VOLUMES: Final[tuple[Volume, ...]] = ("pass_att", "targets", "carries", "air_yards")


# --- history: nflverse frames as typed games --------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlayerGame:
    """One player's line in one regular-season game (nflverse weekly stats), with his offensive snap share when the
    snap counts have it. Teams are nflverse codes (``LA``, ``WAS``)."""

    gsis_id: str
    team: str
    position: str
    season: int
    week: int
    pass_att: float = 0.0
    completions: float = 0.0
    pass_yards: float = 0.0
    pass_tds: float = 0.0
    interceptions: float = 0.0
    carries: float = 0.0
    rush_yards: float = 0.0
    rush_tds: float = 0.0
    targets: float = 0.0
    receptions: float = 0.0
    rec_yards: float = 0.0
    rec_tds: float = 0.0
    air_yards: float = 0.0
    fumbles_lost: float = 0.0
    snap_pct: float | None = None

    @property
    def touches(self) -> float:
        """Opportunities a fumble can follow: dropbacks, carries and targets."""
        return self.pass_att + self.carries + self.targets

    @property
    def slot(self) -> int:
        return self.season * SEASON_SPAN + self.week


def _num(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        return 0.0
    return float(value)


def _text(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None


def player_games(
    stats: pl.DataFrame, snaps: pl.DataFrame | None = None, pfr_to_gsis: Mapping[str, str] | None = None
) -> tuple[PlayerGame, ...]:
    """The QB, RB, WR and TE lines of nflverse's weekly player stats (``player_stats`` from
    :class:`fm.sources.nflverse.NflverseSource`), regular season only, with ``offense_pct`` from the snap counts joined
    through ``pfr_to_gsis`` (PFR id to GSIS id, from ``ff_playerids``). Rows without a GSIS id, a team or a week, and
    other positions, are left out; a fullback counts as a running back."""
    snap_pct: dict[tuple[str, int, int], float] = {}
    if snaps is not None and pfr_to_gsis:
        for row in snaps.iter_rows(named=True):
            pfr, season, week = _text(row.get("pfr_player_id")), row.get("season"), row.get("week")
            gsis = pfr_to_gsis.get(pfr) if pfr is not None else None
            if gsis is None or not isinstance(season, int) or not isinstance(week, int):
                continue
            pct = row.get("offense_pct")
            if isinstance(pct, int | float) and math.isfinite(pct):
                key = (gsis, season, week)
                snap_pct[key] = max(snap_pct.get(key, 0.0), float(pct))
    games: list[PlayerGame] = []
    for row in stats.iter_rows(named=True):
        gsis = _text(row.get("player_id"))
        team = _text(row.get("team")) or _text(row.get("recent_team"))
        season, week = row.get("season"), row.get("week")
        position = _text(row.get("position"))
        position = POSITION_ALIASES.get(position, position) if position is not None else None
        season_type = _text(row.get("season_type"))
        if (
            gsis is None
            or team is None
            or position not in BASELINE_POSITIONS
            or not isinstance(season, int)
            or not isinstance(week, int)
            or (season_type is not None and season_type != REGULAR_SEASON)
        ):
            continue
        games.append(
            PlayerGame(
                gsis_id=gsis,
                team=team,
                position=position,
                season=season,
                week=week,
                pass_att=_num(row.get("attempts")),
                completions=_num(row.get("completions")),
                pass_yards=_num(row.get("passing_yards")),
                pass_tds=_num(row.get("passing_tds")),
                interceptions=_num(row.get("passing_interceptions")),
                carries=_num(row.get("carries")),
                rush_yards=_num(row.get("rushing_yards")),
                rush_tds=_num(row.get("rushing_tds")),
                targets=_num(row.get("targets")),
                receptions=_num(row.get("receptions")),
                rec_yards=_num(row.get("receiving_yards")),
                rec_tds=_num(row.get("receiving_tds")),
                air_yards=_num(row.get("receiving_air_yards")),
                fumbles_lost=_num(row.get("rushing_fumbles_lost"))
                + _num(row.get("receiving_fumbles_lost"))
                + _num(row.get("sack_fumbles_lost")),
                snap_pct=snap_pct.get((gsis, season, week)),
            )
        )
    return tuple(games)


# --- environments: implied team totals --------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TeamLine:
    """A team's game in one week: its opponent, its implied total when one is known, and where the number came from
    (``scoreboard``: ESPN's live line; ``snapshot``: a pregame capture; ``schedule``: nflverse's line; ``default``:
    the game exists but no line does)."""

    team: str
    opponent: str | None
    implied_total: float | None
    origin: LineOrigin


type SeasonWeekTeam = tuple[int, int, str]


def schedule_lines(frame: pl.DataFrame) -> dict[SeasonWeekTeam, TeamLine]:
    """Per (season, week, team) lines from nflverse's schedules (regular season; a game without a line has
    ``implied_total`` ``None``). ``spread_line`` is the home margin the books expect, so the home spread in ESPN's
    sign is its negation."""
    lines: dict[SeasonWeekTeam, TeamLine] = {}
    for row in frame.iter_rows(named=True):
        season, week = row.get("season"), row.get("week")
        home, away = _text(row.get("home_team")), _text(row.get("away_team"))
        game_type = _text(row.get("game_type"))
        if (
            not isinstance(season, int)
            or not isinstance(week, int)
            or home is None
            or away is None
            or (game_type is not None and game_type != REGULAR_SEASON)
        ):
            continue
        spread, total = row.get("spread_line"), row.get("total_line")
        home_total: float | None = None
        away_total: float | None = None
        if isinstance(spread, int | float) and isinstance(total, int | float) and total >= 0:
            totals = implied_team_totals(spread=-float(spread), over_under=float(total))
            home_total, away_total = totals.home, totals.away
        lines[(season, week, home)] = TeamLine(home, away, home_total, "schedule")
        lines[(season, week, away)] = TeamLine(away, home, away_total, "schedule")
    return lines


def league_average_total(lines: Mapping[SeasonWeekTeam, TeamLine], default: float) -> float:
    """The mean implied total of ``lines`` (``default`` when none has one)."""
    totals = [line.implied_total for line in lines.values() if line.implied_total is not None]
    return sum(totals) / len(totals) if totals else default


class PregameSnapshot:
    """Pregame implied totals saved while a scoreboard still carries them (JSON under ``cache_dir()/baseline_nfl/``).

    ESPN drops the odds block at kickoff and the scoreboard cache turns over every ten minutes, so this is the only
    place a started game's line survives. Entries are keyed by (season, week, team); a later capture of the same game
    replaces the earlier one (lines move). It is a deletable cache: a missing or unreadable file is an empty snapshot.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path if self._path is not None else paths.cache_dir() / SNAPSHOT_DIRNAME / SNAPSHOT_FILENAME

    def _read(self) -> dict[str, dict[str, object]]:
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(raw, dict):
            return {}
        return {key: dict(value) for key, value in raw.items() if isinstance(key, str) and isinstance(value, dict)}

    @staticmethod
    def _key(season: int, week: int, team: str) -> str:
        return f"{season}:{week}:{team}"

    def total(self, season: int, week: int, team: str) -> float | None:
        entry = self._read().get(self._key(season, week, team))
        value = entry.get("implied_total") if entry is not None else None
        return float(value) if isinstance(value, int | float) and math.isfinite(value) else None

    def record(self, season: int, week: int, lines: Iterable[TeamLine], *, captured_at: datetime) -> int:
        """Save the lines that carry a total (replacing earlier captures of the same teams); returns how many."""
        entries = self._read()
        saved = 0
        for line in lines:
            if line.implied_total is None:
                continue
            entries[self._key(season, week, line.team)] = {
                "implied_total": line.implied_total,
                "opponent": line.opponent,
                "captured_at": captured_at.isoformat(),
            }
            saved += 1
        if saved:
            target = self.path
            target.parent.mkdir(parents=True, exist_ok=True)
            scratch = target.with_suffix(".tmp")
            scratch.write_text(json.dumps(entries, indent=1, sort_keys=True), encoding="utf-8")
            os.replace(scratch, target)
        return saved


def capture_pregame_lines(
    board: Scoreboard,
    *,
    season: int,
    week: int,
    snapshot: PregameSnapshot | None = None,
    now: datetime | None = None,
) -> int:
    """Save the implied totals of every game on ``board`` that is still ``pre`` and has a line; returns how many team
    totals were saved. Call it right after a scoreboard read before kickoff (the sync or tick), so the totals outlive
    ESPN's odds block."""
    store = snapshot if snapshot is not None else PregameSnapshot()
    lines: list[TeamLine] = []
    for game in board.games:
        totals = game.implied_totals
        if game.state != "pre" or totals is None:
            continue
        home, away = team_code(game.home.abbreviation, "nflverse"), team_code(game.away.abbreviation, "nflverse")
        lines.append(TeamLine(home, away, totals.home, "scoreboard"))
        lines.append(TeamLine(away, home, totals.away, "scoreboard"))
    return store.record(season, week, lines, captured_at=now if now is not None else utcnow())


def week_lines(
    board: Scoreboard | None,
    *,
    season: int,
    week: int,
    snapshot: PregameSnapshot | None = None,
    schedule: Mapping[SeasonWeekTeam, TeamLine] | None = None,
) -> tuple[dict[str, TeamLine], tuple[str, ...]]:
    """The week's games by nflverse team code, with the best implied total known for each side, and warnings.

    Order of preference per team: the scoreboard's live line, the pregame snapshot, nflverse's schedule line, and
    finally ``None`` (the model then uses the league-average total and says so). A scoreboard game puts its teams on
    the slate even when no line is available; the schedule alone does when the scoreboard is empty. A team on neither
    has no game (a bye). A scoreboard for another season or week is ignored with a warning."""
    warnings: list[str] = []
    slate: dict[str, TeamLine] = {}
    for (line_season, line_week, team), line in (schedule or {}).items():
        if (line_season, line_week) == (season, week):
            slate[team] = line
    if board is not None and board.games and board.season is not None and board.week is not None:
        if (board.season, board.week) != (season, week):
            warnings.append(
                f"opportunity: the scoreboard is for {board.season} week {board.week}, not {season} week {week}; "
                "its lines were ignored"
            )
            board = None
    for game in board.games if board is not None else ():
        totals = game.implied_totals
        sides = (
            (team_code(game.home.abbreviation, "nflverse"), team_code(game.away.abbreviation, "nflverse"), 0),
            (team_code(game.away.abbreviation, "nflverse"), team_code(game.home.abbreviation, "nflverse"), 1),
        )
        for team, opponent, side in sides:
            if totals is not None:
                slate[team] = TeamLine(team, opponent, totals.home if side == 0 else totals.away, "scoreboard")
                continue
            kept = snapshot.total(season, week, team) if snapshot is not None else None
            if kept is not None:
                slate[team] = TeamLine(team, opponent, kept, "snapshot")
            elif team in slate and slate[team].implied_total is not None:
                slate[team] = replace(slate[team], opponent=opponent)
            else:
                slate[team] = TeamLine(team, opponent, None, "default")
    scheduled = sorted(team for team, line in slate.items() if line.origin == "schedule")
    if board is not None and board.games and scheduled:
        warnings.append(
            f"opportunity: no scoreboard or snapshot line for {', '.join(scheduled)}; used nflverse's schedule line"
        )
    return slate, tuple(warnings)


# --- the model --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class OpportunityWeek:
    """What :func:`project_week` produced: stat lines by GSIS id, the positions they were projected at, the teams whose
    total was the league average for want of a line, the players left out (a team on bye, no role) and warnings."""

    lines: Mapping[str, Mapping[str, float]]
    positions: Mapping[str, str]
    default_total_teams: tuple[str, ...] = ()
    bye: tuple[str, ...] = ()
    no_role: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _RateSpec:
    name: str
    numerator: str
    denominator: str
    environment: Literal["pass", "rush"] | None = None


_RATE_SPECS: Final[tuple[_RateSpec, ...]] = (
    _RateSpec("completion", "completions", "pass_att"),
    _RateSpec("yards_per_attempt", "pass_yards", "pass_att"),
    _RateSpec("pass_td", "pass_tds", "pass_att", "pass"),
    _RateSpec("interception", "interceptions", "pass_att"),
    _RateSpec("yards_per_carry", "rush_yards", "carries"),
    _RateSpec("rush_td", "rush_tds", "carries", "rush"),
    _RateSpec("catch_rate", "receptions", "targets"),
    _RateSpec("yards_per_target", "rec_yards", "targets"),
    _RateSpec("receiving_td", "rec_tds", "targets", "pass"),
    _RateSpec("yards_per_air_yard", "rec_yards", "air_yards"),
    _RateSpec("fumble", "fumbles_lost", "touches"),
)
_SHARED_VOLUME: Final[Mapping[Volume, str]] = MappingProxyType(
    {"pass_att": "pass_att", "targets": "targets", "carries": "carries", "air_yards": "air_yards"}
)


def _clamp(value: float, bounds: tuple[float, float]) -> float:
    return min(max(value, bounds[0]), bounds[1])


def _get(game: PlayerGame, attribute: str) -> float:
    return float(getattr(game, attribute))


class _Model:
    """One run of the model: the history before the target week, its weights and pooled statistics."""

    def __init__(
        self,
        history: Iterable[PlayerGame],
        season: int,
        week: int,
        history_lines: Mapping[SeasonWeekTeam, TeamLine],
        config: OpportunityConfig,
    ) -> None:
        self.config = config
        self.slot = season * SEASON_SPAN + week
        self.history_lines = history_lines
        self.past = tuple(
            sorted(
                (g for g in history if 0 < self.slot - g.slot <= config.horizon_weeks),
                key=lambda g: (g.slot, g.gsis_id),
            )
        )
        self.team_games: dict[SeasonWeekTeam, dict[Volume, float]] = defaultdict(lambda: dict.fromkeys(VOLUMES, 0.0))
        for game in self.past:
            totals = self.team_games[(game.season, game.week, game.team)]
            for volume in VOLUMES:
                totals[volume] += _get(game, _SHARED_VOLUME[volume])
        self.team_weights: dict[str, dict[SeasonWeekTeam, float]] = defaultdict(dict)
        for game in self.past:
            self.team_weights[game.team].setdefault((game.season, game.week, game.team), self.weight(game))
        self._team_volume: dict[str, dict[Volume, float]] = {}
        self.league_neutral = self._league_neutral()
        self.pools = self._pools()
        self.share_priors = self._share_priors()

    # environment

    def factor(self, season: int, week: int, team: str) -> float:
        """The environment factor of a past game: its implied total over the league average, clamped; 1 when the line
        is unknown."""
        line = self.history_lines.get((season, week, team))
        if line is None or line.implied_total is None or self.config.league_avg_total <= 0:
            return 1.0
        return _clamp(line.implied_total / self.config.league_avg_total, self.config.environment_clamp)

    def volume_elasticity(self, volume: Volume) -> float:
        return self.config.rush_volume_elasticity if volume == "carries" else self.config.pass_volume_elasticity

    def weight(self, game: PlayerGame) -> float:
        return self.config.decay ** (self.slot - game.slot - 1)

    # team volume

    def _neutral(self, key: SeasonWeekTeam, volume: Volume) -> float:
        return self.team_games[key][volume] / self.factor(*key) ** self.volume_elasticity(volume)

    def _league_neutral(self) -> dict[Volume, float]:
        keys = list(self.team_games)
        return {
            volume: sum(self._neutral(key, volume) for key in keys) / len(keys) if keys else 0.0 for volume in VOLUMES
        }

    def team_volume(self, team: str) -> dict[Volume, float]:
        """The team's environment-neutral weekly volume: its recency-weighted mean shrunk toward the league's."""
        cached = self._team_volume.get(team)
        if cached is not None:
            return cached
        weights = self.team_weights.get(team, {})
        prior = self.config.team_prior_games
        total_weight = sum(weights.values())
        result: dict[Volume, float] = {
            name: (sum(w * self._neutral(key, name) for key, w in weights.items()) + prior * self.league_neutral[name])
            / (total_weight + prior)
            for name in VOLUMES
        }
        self._team_volume[team] = result
        return result

    # pooled statistics

    def _pools(self) -> dict[str, dict[str, float]]:
        pools: dict[str, dict[str, float]] = {}
        for position in BASELINE_POSITIONS:
            rows = [g for g in self.past if g.position == position]
            pool: dict[str, float] = {}
            for spec in _RATE_SPECS:
                denominator = sum(_get(g, spec.denominator) for g in rows)
                numerator = sum(_get(g, spec.numerator) for g in rows)
                pool[spec.name] = (
                    numerator / denominator if denominator >= MIN_POOL else DEFAULT_RATES[position][spec.name]
                )
            pools[position] = pool
        return pools

    def _share_priors(self) -> dict[tuple[str, Volume], float]:
        priors: dict[tuple[str, Volume], float] = {}
        for position in BASELINE_POSITIONS:
            for volume in VOLUMES:
                shares = [
                    share for g in self.past if g.position == position and (share := self.share(g, volume)) is not None
                ]
                priors[(position, volume)] = sum(shares) / len(shares) if shares else 0.0
        return priors

    def share(self, game: PlayerGame, volume: Volume) -> float | None:
        team_total = self.team_games[(game.season, game.week, game.team)][volume]
        return _get(game, _SHARED_VOLUME[volume]) / team_total if team_total > 0 else None

    # one player

    def regressed(self, rows: Sequence[PlayerGame], spec: _RateSpec, position: str) -> float:
        numerator = denominator = 0.0
        for game in rows:
            weight = self.weight(game)
            divisor = 1.0
            if spec.environment is not None:
                elasticity = (
                    self.config.pass_td_elasticity if spec.environment == "pass" else self.config.rush_td_elasticity
                )
                divisor = self.factor(game.season, game.week, game.team) ** elasticity
            numerator += weight * _get(game, spec.numerator) / divisor
            denominator += weight * _get(game, spec.denominator)
        prior_weight = self.config.priors.weight(spec.name)
        return (numerator + prior_weight * self.pools[position][spec.name]) / (denominator + prior_weight)

    def shares(self, rows: Sequence[PlayerGame], position: str) -> dict[Volume, float]:
        snap = self.snap_factor(rows) if position != "QB" else 1.0
        prior_games = self.config.share_prior_games
        out: dict[Volume, float] = {}
        for volume in VOLUMES:
            numerator = denominator = 0.0
            for game in rows:
                share = self.share(game, volume)
                if share is not None:
                    weight = self.weight(game)
                    numerator += weight * share
                    denominator += weight
            prior = self.share_priors[(position, volume)]
            out[volume] = (numerator + prior_games * prior) / (denominator + prior_games) * snap
        return out

    def snap_factor(self, rows: Sequence[PlayerGame]) -> float:
        """Last two games' snap percentage over the weighted mean, clamped; 1 without enough snap data."""
        with_snaps = [g for g in rows if g.snap_pct is not None]
        if len(with_snaps) < MIN_SNAP_GAMES:
            return 1.0
        weights = [self.weight(g) for g in with_snaps]
        mean = sum(w * (g.snap_pct or 0.0) for w, g in zip(weights, with_snaps, strict=True)) / sum(weights)
        recent = sum(g.snap_pct or 0.0 for g in with_snaps[-2:]) / 2
        return _clamp(recent / mean, self.config.snap_clamp) if mean > 0 else 1.0


def _line(
    model: _Model, rows: Sequence[PlayerGame], position: str, team_line: TeamLine, total: float
) -> dict[str, float]:
    config = model.config
    factor = _clamp(total / config.league_avg_total, config.environment_clamp)
    volume = model.team_volume(team_line.team)
    shares = model.shares(rows, position)
    expected = {name: volume[name] * factor ** model.volume_elasticity(name) * shares[name] for name in VOLUMES}
    rate = {spec.name: model.regressed(rows, spec, position) for spec in _RATE_SPECS}
    pass_td_factor = factor**config.pass_td_elasticity
    attempts, carries, targets, air = (expected[name] for name in ("pass_att", "carries", "targets", "air_yards"))
    by_targets = targets * rate["yards_per_target"]
    receiving_yards = by_targets
    if position in ("WR", "TE"):
        path = config.air_yards_path_weight
        receiving_yards = (1 - path) * by_targets + path * air * rate["yards_per_air_yard"]
    line = {
        "PA": attempts,
        "PC": attempts * rate["completion"],
        "PY": attempts * rate["yards_per_attempt"],
        "PTD": attempts * rate["pass_td"] * pass_td_factor,
        "INTT": attempts * rate["interception"],
        "RA": carries,
        "RY": carries * rate["yards_per_carry"],
        "RTD": carries * rate["rush_td"] * factor**config.rush_td_elasticity,
        "RET": targets,
        "REC": targets * rate["catch_rate"],
        "REY": receiving_yards,
        "RETD": targets * rate["receiving_td"] * pass_td_factor,
        "FUML": (attempts + carries + targets) * rate["fumble"],
    }
    return {stat: round(value, 3) for stat, value in line.items() if math.isfinite(value) and value >= MIN_LINE_VALUE}


def project_week(
    history: Iterable[PlayerGame],
    *,
    season: int,
    week: int,
    games: Mapping[str, TeamLine],
    history_lines: Mapping[SeasonWeekTeam, TeamLine] | None = None,
    config: OpportunityConfig = DEFAULT_CONFIG,
) -> OpportunityWeek:
    """Stat lines by GSIS id for every QB, RB, WR and TE in ``history`` whose latest team plays in ``games`` (team
    code to :class:`TeamLine`; a team missing from it is on bye).

    Only games before ``(season, week)`` are read, however much ``history`` holds, so a backtest may pass a whole
    season. ``history_lines`` gives past games' implied totals (:func:`schedule_lines`); without them every past game
    is average. A team whose line carries no total gets the league-average total (``config.league_avg_total``) and a
    warning. A player whose expected touches fall under ``config.min_touches`` has no role and gets no line."""
    model = _Model(history, season, week, history_lines or {}, config)
    by_player: dict[str, list[PlayerGame]] = defaultdict(list)
    for game in model.past:
        by_player[game.gsis_id].append(game)
    lines: dict[str, Mapping[str, float]] = {}
    positions: dict[str, str] = {}
    bye: list[str] = []
    no_role: list[str] = []
    defaulted: set[str] = set()
    for gsis, rows in sorted(by_player.items()):
        latest = rows[-1]
        team_line = games.get(latest.team)
        if team_line is None:
            bye.append(gsis)
            continue
        total = team_line.implied_total
        if total is None:
            total = config.league_avg_total
            defaulted.add(latest.team)
        line = _line(model, rows, latest.position, team_line, total)
        if line.get("PA", 0.0) + line.get("RA", 0.0) + line.get("RET", 0.0) < config.min_touches:
            no_role.append(gsis)
            continue
        lines[gsis] = MappingProxyType(line)
        positions[gsis] = latest.position
    warnings = tuple(
        f"opportunity: no implied total for {team} in {season} week {week}; used the league average "
        f"{config.league_avg_total:.1f}"
        for team in sorted(defaulted)
    )
    return OpportunityWeek(
        MappingProxyType(lines),
        MappingProxyType(positions),
        tuple(sorted(defaulted)),
        tuple(bye),
        tuple(no_role),
        warnings,
    )


# --- the loader -------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Provenance:
    """What a loader keeps of one :class:`fm.sources.base.Fetched` it read: its age and cache state."""

    as_of: datetime
    cached: bool
    stale: bool


class OpportunityLoader:
    """The ``opportunity`` source's loader: nflverse history and the scoreboard's lines through the adapters, joined to
    ESPN ids through the crosswalk ``fm sync`` saved.

    ``nflverse`` and ``scoreboard`` default to fresh adapters over the cache dir (the scoreboard one closed after each
    call), ``crosswalk`` to the stored one and ``snapshot`` to :class:`PregameSnapshot`. A scoreboard read before
    kickoff saves its totals (:func:`capture_pregame_lines`). Failures degrade into warnings: no stats at all is an
    empty ``degraded`` result, a missing snap count, schedule or scoreboard only costs the part of the model it feeds.
    """

    def __init__(
        self,
        nflverse: NflverseSource | None = None,
        scoreboard: EspnScoreboardSource | None = None,
        *,
        crosswalk: Crosswalk | None = None,
        snapshot: PregameSnapshot | None = None,
        config: OpportunityConfig = DEFAULT_CONFIG,
    ) -> None:
        self.nflverse = nflverse
        self.scoreboard = scoreboard
        self.crosswalk = crosswalk
        self.snapshot = snapshot
        self.config = config

    def __call__(
        self, store: Store, season: int, scoring_period: int, options: FetchOptions
    ) -> Fetched[tuple[ProjectionRow, ...]]:
        nflverse = self.nflverse if self.nflverse is not None else NflverseSource()
        scoreboard = self.scoreboard if self.scoreboard is not None else EspnScoreboardSource()
        try:
            return self._load(store, nflverse, scoreboard, season, scoring_period, options)
        finally:
            if self.scoreboard is None:
                scoreboard.close()

    def _load(
        self,
        store: Store,
        nflverse: NflverseSource,
        scoreboard_source: EspnScoreboardSource,
        season: int,
        week: int,
        options: FetchOptions,
    ) -> Fetched[tuple[ProjectionRow, ...]]:
        warnings: list[str] = []
        fetched: list[_Provenance] = []

        def attempt(label: str, call: Callable[[], Fetched[pl.DataFrame]]) -> pl.DataFrame | None:
            try:
                result = call()
            except SourceError as exc:
                warnings.append(f"opportunity: {label} unavailable ({exc})")
                return None
            fetched.append(_Provenance(result.as_of, result.cached, result.stale))
            warnings.extend(result.warnings)
            return result.data

        seasons = (season - 1, season)
        stats = [attempt(f"{s} player stats", lambda s=s: nflverse.player_stats(s, **options)) for s in seasons]
        snaps = [attempt(f"{s} snap counts", lambda s=s: nflverse.snap_counts(s, **options)) for s in seasons]
        schedules = [attempt(f"{s} schedule", lambda s=s: nflverse.schedules(s, **options)) for s in seasons]
        ids = attempt("ff_playerids", lambda: nflverse.ff_playerids(**options))
        stats_in_hand = [f for f in stats if f is not None]
        key = f"{BASELINE_SPORT}_{season}_{week}"
        board = self._scoreboard(scoreboard_source, season, week, options, fetched, warnings)
        if not stats_in_hand:
            warnings.append(f"opportunity: no nflverse stats for {season} or {season - 1}; nothing to project from")
            return self._result((), fetched, key, warnings, degraded=True)
        pfr_to_gsis = _pfr_ids(ids)
        history: list[PlayerGame] = []
        for index, frame_stats in enumerate(stats):
            if frame_stats is not None:
                history.extend(player_games(frame_stats, snaps[index], pfr_to_gsis))
        history_lines: dict[SeasonWeekTeam, TeamLine] = {}
        for schedule in schedules:
            if schedule is not None:
                history_lines.update(schedule_lines(schedule))
        past_lines = {key_: line for key_, line in history_lines.items() if (key_[0], key_[1]) < (season, week)}
        config = replace(self.config, league_avg_total=league_average_total(past_lines, self.config.league_avg_total))
        snapshot = self.snapshot if self.snapshot is not None else PregameSnapshot()
        if board is not None:
            capture_pregame_lines(board, season=season, week=week, snapshot=snapshot)
        slate, slate_warnings = week_lines(board, season=season, week=week, snapshot=snapshot, schedule=history_lines)
        warnings.extend(slate_warnings)
        if not slate:
            warnings.append(f"opportunity: no games known for {season} week {week} (scoreboard and schedule empty)")
            return self._result((), fetched, key, warnings, degraded=True)
        projected = project_week(
            history, season=season, week=week, games=slate, history_lines=past_lines, config=config
        )
        warnings.extend(projected.warnings)
        walk = self.crosswalk if self.crosswalk is not None else Crosswalk.from_store(store)
        as_of = min((item.as_of for item in fetched), default=utcnow())
        rows: list[ProjectionRow] = []
        unmapped: list[str] = []
        for gsis, line in projected.lines.items():
            espn_id = walk.espn_id(GSIS, gsis)
            if espn_id is None:
                unmapped.append(gsis)
                continue
            rows.append(
                ProjectionRow(
                    sport=BASELINE_SPORT,
                    espn_id=espn_id,
                    source=OPPORTUNITY,
                    kind="projected",
                    season=season,
                    scoring_period_id=week,
                    stats=dict(line),
                    as_of=as_of,
                )
            )
        if unmapped:
            shown = ", ".join(unmapped[:5]) + (f" and {len(unmapped) - 5} more" if len(unmapped) > 5 else "")
            warnings.append(f"opportunity: {len(unmapped)} projected players have no ESPN id ({shown})")
        return self._result(tuple(rows), fetched, key, warnings, degraded=not rows, as_of=as_of)

    @staticmethod
    def _scoreboard(
        source: EspnScoreboardSource,
        season: int,
        week: int,
        options: FetchOptions,
        fetched: list[_Provenance],
        warnings: list[str],
    ) -> Scoreboard | None:
        result = source.scoreboard(Game.FFL, season=season, week=week, **options)
        warnings.extend(result.warnings)
        if result.degraded:
            warnings.append(
                "opportunity: the scoreboard is unavailable; implied totals come from snapshots and schedule"
            )
            return None
        fetched.append(_Provenance(result.as_of, result.cached, result.stale))
        return result.data

    @staticmethod
    def _result(
        rows: tuple[ProjectionRow, ...],
        fetched: Sequence[_Provenance],
        key: str,
        warnings: Sequence[str],
        *,
        degraded: bool,
        as_of: datetime | None = None,
    ) -> Fetched[tuple[ProjectionRow, ...]]:
        stamp = as_of if as_of is not None else min((item.as_of for item in fetched), default=utcnow())
        return Fetched(
            rows,
            stamp,
            OPPORTUNITY,
            DATASET,
            key,
            cached=bool(fetched) and all(item.cached for item in fetched),
            stale=any(item.stale for item in fetched),
            degraded=degraded,
            warnings=tuple(dict.fromkeys(warnings)),
        )


def _pfr_ids(frame: pl.DataFrame | None) -> dict[str, str]:
    """PFR id to GSIS id from ``ff_playerids``."""
    if frame is None or "pfr_id" not in frame.columns or "gsis_id" not in frame.columns:
        return {}
    out: dict[str, str] = {}
    for row in frame.iter_rows(named=True):
        pfr, gsis = _text(row.get("pfr_id")), _text(row.get("gsis_id"))
        if pfr is not None and gsis is not None:
            out[pfr] = gsis
    return out


def _register() -> None:
    """Register ``opportunity`` for ``nfl`` unless it already is (a module imported twice keeps one registration)."""
    if source_registry.get(BASELINE_SPORT, OPPORTUNITY) is None:
        register_source(BASELINE_SPORT, OPPORTUNITY, label=OPPORTUNITY_LABEL, loader=OpportunityLoader())


_register()
