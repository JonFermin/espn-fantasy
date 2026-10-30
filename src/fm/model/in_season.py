"""In-season evidence: what a player has actually done this season, read ahead of the preseason projections.

Every decision values a player through projections (ESPN's season line, DARKO, Sleeper). Early in a season those are
preseason opinions, and on their own they miss what the season has already shown: a healthy player out of the
rotation (an NBA DNP-coach's-decision every night) or a starter whose rest-of-season line ESPN sent empty. This module
reads the season's own record from the store and turns it into two corrections the valuations apply:

**The game log.** ``fm sync`` stores ESPN's actual line for every played scoring period the player card carries
(``projections`` rows, ``source='espn'``, ``kind='actual'``, period >= 1). ESPN sends one per game the player's team
played, empty when he did not play, and none for a day or week without a game, so a stored period *is* a team game and
an empty line in it a game he sat. Only periods before the current one count: the current period's line may be a game
not yet played.

**Healthy games** are the team games in which he was available: the stored ``availability`` row for the period (the
nearest earlier one, else the nearest later, when that period has none) was not a non-playing designation
(:data:`UNAVAILABLE`). With no availability row at all, his current ``players.injury_status`` decides. A game missed
injured therefore never counts against him.

**Role factor** (:meth:`Evidence.role_factor`): the share of his last ``role_window`` healthy games he played, shrunk
toward 1 (the projections' own assumption that he plays) with ``role_k`` pseudo-games:
``(played + role_k) / (healthy + role_k)``. No evidence is 1.0, never a penalty. It multiplies a game's value next to
``p_active``, which stays the injury model's.

**Production** (:func:`shrink_line`, :func:`shrink`): the season-to-date actual line (period 0, ``GP`` games) mixed
with the projection per game, stat by stat: ``(line_k * projected + GP * actual) / (line_k + GP)``. The season's own
numbers overtake the projection once he has played ``line_k`` games.

**Recent form** (:meth:`Evidence.form`, added 2026-10-29): the games he played in the last ``recent_periods`` periods
before the current one (NBA: 15 days) count ``recent_weight`` times more on top of the season line they are already
part of. The "actual" side of the mix is then ``(season totals + recent_weight * recent totals)`` over
``GP + recent_weight * R`` games, so a hot or cold stretch moves the line further and sooner. ``recent_weight`` 0 is
the season line alone. Rate stats (percentages) are not summed; they keep the season's.

The constants are per sport in ``data/in_season.toml`` (:class:`InSeasonParams`), starting points for a backtest to
tune.
"""

from __future__ import annotations

import math
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

from fm.config import Sport
from fm.paths import data_file
from fm.store import AvailabilityRow, ProjectionRow, Store

ESPN_SOURCE: Final = "espn"
SEASON_PERIOD: Final = 0
GAMES_STAT: Final = "GP"
MINUTES_STAT: Final = "MIN"
DEFAULT_PARAMS_FILE: Final = "in_season.toml"

UNAVAILABLE: Final = frozenset({"OUT", "INJURY_RESERVE", "SUSPENSION", "DOUBTFUL"})
"""Designations under which a missed game is the injury's, not the coach's."""


class InSeasonError(ValueError):
    """``in_season.toml`` is missing or malformed."""


@dataclass(frozen=True, slots=True)
class InSeasonParams:
    """One sport's constants: ``line_k`` pseudo-games of projection in the production mix, ``role_k`` pseudo-games of
    "he plays" in the role factor over the last ``role_window`` healthy games, and ``min_benched``: healthy games in a
    row without playing after which a decision refuses to add him outright."""

    line_k: float
    role_k: float
    role_window: int
    min_benched: int
    recent_periods: int = 15
    recent_weight: float = 0.0

    def __post_init__(self) -> None:
        weight = self.recent_weight
        if isinstance(weight, bool) or not isinstance(weight, int | float) or not math.isfinite(weight) or weight < 0:
            raise InSeasonError(f"recent_weight should be a finite number >= 0, got {weight!r}")
        for name in ("line_k", "role_k"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise InSeasonError(f"{name} should be a finite number > 0, got {value!r}")
        for name in ("role_window", "min_benched", "recent_periods"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise InSeasonError(f"{name} should be an integer >= 1, got {value!r}")


DEFAULT_PARAMS: Final[Mapping[Sport, InSeasonParams]] = MappingProxyType(
    {
        "nba": InSeasonParams(
            line_k=4.0, role_k=2.0, role_window=10, min_benched=2, recent_periods=15, recent_weight=1.0
        ),
        "nfl": InSeasonParams(line_k=4.0, role_k=1.5, role_window=6, min_benched=2, recent_periods=3),
    }
)
"""What a sport missing from the file gets; the committed file states the same numbers."""


def load_params(path: Path | None = None) -> Mapping[Sport, InSeasonParams]:
    """The per-sport constants from ``data/in_season.toml`` (or ``path``). Raises :class:`InSeasonError` for a missing
    file, an unknown sport or key, or a bad value."""
    target = data_file(DEFAULT_PARAMS_FILE) if path is None else path
    if not target.is_file():
        raise InSeasonError(f"in-season params file not found: {target}")
    return parse_params(target.read_text(encoding="utf-8"), where=str(target))


def parse_params(text: str, *, where: str = DEFAULT_PARAMS_FILE) -> Mapping[Sport, InSeasonParams]:
    try:
        raw = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise InSeasonError(f"{where}: {exc}") from exc
    params: dict[Sport, InSeasonParams] = dict(DEFAULT_PARAMS)
    fields = set(InSeasonParams.__dataclass_fields__)
    for key, table in raw.items():
        if key not in DEFAULT_PARAMS:
            raise InSeasonError(f"{where}: [{key}] is not a sport (expected nfl or nba)")
        if not isinstance(table, Mapping):
            raise InSeasonError(f"{where}: [{key}] should be a table, got {table!r}")
        unknown = sorted(set(table) - fields)
        if unknown:
            raise InSeasonError(f"{where}: [{key}] has unknown keys {unknown}")
        sport: Sport = "nfl" if key == "nfl" else "nba"
        base = DEFAULT_PARAMS[sport]
        values: dict[str, Any] = {name: table.get(name, getattr(base, name)) for name in fields}
        try:
            params[sport] = InSeasonParams(**values)
        except InSeasonError as exc:
            raise InSeasonError(f"{where}: [{key}] {exc}") from exc
    return MappingProxyType(params)


# --- one player's evidence --------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Evidence:
    """What the store shows of one player's season.

    ``log`` is his game log before the current period, oldest first: ``(period, played, healthy)`` per team game.
    ``season_line`` is ESPN's season-to-date actual line (``None`` when not stored) and ``games`` its ``GP``.
    """

    espn_id: int
    log: tuple[tuple[int, bool, bool], ...] = ()
    season_line: Mapping[str, float] | None = None
    games: float = 0.0
    lines: Mapping[int, Mapping[str, float]] = field(default_factory=lambda: MappingProxyType({}), compare=False)
    """The game log's actual lines by period (before the current one), for :meth:`form`."""
    current_period: int | None = field(default=None, compare=False)

    def recent_line(self, periods: int) -> tuple[dict[str, float], int]:
        """(summed stats, games played) over his games in the ``periods`` periods before the current one."""
        if self.current_period is None:
            return {}, 0
        since = self.current_period - periods
        total: dict[str, float] = {}
        games = 0
        for period, line in self.lines.items():
            if period < since or not played_in(line):
                continue
            games += 1
            for stat, value in line.items():
                if stat != GAMES_STAT:
                    total[stat] = total.get(stat, 0.0) + float(value)
        return total, games

    def form(self, params: InSeasonParams, *, rates: Iterable[str] = ()) -> tuple[dict[str, float] | None, float]:
        """The "actual" side of the production mix (module docs, recent form): ``(totals, games)`` with the last
        ``recent_periods`` periods' games counted ``recent_weight`` extra times, ``GP`` set to ``games``; ``(None, 0)``
        without a season line or a game. ``rates`` are stats kept at the season's value, never summed."""
        if self.season_line is None or self.games <= 0:
            return None, 0.0
        totals = {stat: float(value) for stat, value in self.season_line.items()}
        games = self.games
        if params.recent_weight > 0:
            recent, count = self.recent_line(params.recent_periods)
            if count > 0:
                kept = frozenset(rates)
                for stat, value in recent.items():
                    if stat not in kept and stat in totals:
                        totals[stat] += params.recent_weight * value
                games += params.recent_weight * count
        totals[GAMES_STAT] = games
        return totals, games

    @property
    def team_games(self) -> int:
        return len(self.log)

    def recent(self, window: int) -> tuple[tuple[int, bool, bool], ...]:
        """His last ``window`` healthy games, oldest first."""
        healthy = [entry for entry in self.log if entry[2]]
        return tuple(healthy[-window:])

    def window(self, params: InSeasonParams) -> tuple[int, int]:
        """(played, healthy) over the last ``params.role_window`` healthy games."""
        recent = self.recent(params.role_window)
        return sum(1 for _, played, _ in recent if played), len(recent)

    def role_factor(self, params: InSeasonParams) -> float:
        """The shrunk share of his recent healthy games he played; 1.0 without any."""
        played, healthy = self.window(params)
        return (played + params.role_k) / (healthy + params.role_k)

    def benched(self) -> int:
        """Healthy games in a row, up to the latest, he did not play."""
        streak = 0
        for _, played, healthy in reversed(self.log):
            if not healthy:
                continue
            if played:
                break
            streak += 1
        return streak

    def numbers(self, params: InSeasonParams) -> dict[str, Any]:
        """The evidence for a proposal's ``engine_numbers``."""
        played, healthy = self.window(params)
        return {
            "played": played,
            "healthy": healthy,
            "team_games": self.team_games,
            "games": self.games,
            "role_factor": self.role_factor(params),
            "benched": self.benched(),
        }

    def describe(self, params: InSeasonParams) -> str | None:
        """A clause for a rationale ("played 0 of 3 healthy games"); ``None`` without a healthy game to speak of."""
        played, healthy = self.window(params)
        if healthy == 0:
            return None
        return f"played {played} of {healthy} healthy game{'s' if healthy != 1 else ''}"


def role_factors(evidence: Mapping[int, Evidence], params: InSeasonParams) -> dict[int, float]:
    """Each player's :meth:`Evidence.role_factor`, by ESPN id."""
    return {espn_id: entry.role_factor(params) for espn_id, entry in evidence.items()}


def played_in(line: Mapping[str, float]) -> bool:
    """Whether an actual game line shows him playing: minutes or a game played, else any nonzero stat."""
    if line.get(MINUTES_STAT, 0.0) > 0 or line.get(GAMES_STAT, 0.0) > 0:
        return True
    return any(value != 0 for value in line.values() if isinstance(value, (int, float)))


def _designation(rows: Mapping[int, AvailabilityRow], period: int) -> str | None:
    if period in rows:
        return rows[period].designation
    earlier = [p for p in rows if p < period]
    if earlier:
        return rows[max(earlier)].designation
    later = [p for p in rows if p > period]
    return rows[min(later)].designation if later else None


def evidence_from(
    espn_id: int,
    actuals: Iterable[ProjectionRow],
    availability: Iterable[AvailabilityRow],
    *,
    current_period: int,
    injury_status: str | None = None,
) -> Evidence:
    """One player's :class:`Evidence` from his stored actual rows and availability rows."""
    log_lines: dict[int, Mapping[str, float]] = {}
    season_line: Mapping[str, float] | None = None
    for row in actuals:
        if row.espn_id != espn_id or row.kind != "actual" or row.source != ESPN_SOURCE:
            continue
        if row.scoring_period_id == SEASON_PERIOD:
            season_line = row.stats
        elif 0 < row.scoring_period_id < current_period:
            log_lines[row.scoring_period_id] = row.stats
    by_period = {row.scoring_period_id: row for row in availability if row.espn_id == espn_id}
    log: list[tuple[int, bool, bool]] = []
    for period in sorted(log_lines):
        designation = _designation(by_period, period) if by_period else injury_status
        log.append((period, played_in(log_lines[period]), designation not in UNAVAILABLE))
    games = float(season_line.get(GAMES_STAT, 0.0)) if season_line else 0.0
    if season_line is not None and games <= 0:
        games = float(sum(1 for _, played, _ in log if played))
    return Evidence(
        espn_id,
        tuple(log),
        MappingProxyType(dict(season_line)) if season_line else None,
        games,
        lines=MappingProxyType({period: MappingProxyType(dict(line)) for period, line in log_lines.items()}),
        current_period=current_period,
    )


def load_evidence(
    store: Store, sport: Sport, season: int, espn_ids: Iterable[int], *, current_period: int
) -> dict[int, Evidence]:
    """:class:`Evidence` for every one of ``espn_ids`` (a player with nothing stored gets an empty one)."""
    ids = sorted(set(espn_ids))
    actuals: dict[int, list[ProjectionRow]] = {}
    for row in store.projections.for_players(sport, season, ids, source=ESPN_SOURCE, kind="actual"):
        actuals.setdefault(row.espn_id, []).append(row)
    availability: dict[int, list[AvailabilityRow]] = {}
    for row in store.availability.for_players(sport, season, ids):
        availability.setdefault(row.espn_id, []).append(row)
    status = {player.espn_id: player.injury_status for player in store.players.many(sport, ids)}
    return {
        espn_id: evidence_from(
            espn_id,
            actuals.get(espn_id, ()),
            availability.get(espn_id, ()),
            current_period=current_period,
            injury_status=status.get(espn_id),
        )
        for espn_id in ids
    }


# --- the production mix -----------------------------------------------------------------------------------------------


def weight(games: float, params: InSeasonParams) -> float:
    """The actual line's share of the mix after ``games`` games: ``games / (games + line_k)``."""
    return games / (games + params.line_k) if games > 0 else 0.0


def shrink(projected: float, actual: float, games: float, params: InSeasonParams) -> float:
    """One number's mix of the projection and the season's own per-game value."""
    share = weight(games, params)
    return (1.0 - share) * projected + share * actual


def shrink_line(
    projected: Mapping[str, float], actual: Mapping[str, float], games: float, params: InSeasonParams
) -> dict[str, float]:
    """A per-game projected line mixed stat by stat with the season's per-game actual line. A stat the actual line
    lacks keeps its projection; ``GP`` keeps the projection's."""
    share = weight(games, params)
    if share <= 0:
        return dict(projected)
    mixed = dict(projected)
    for stat, value in projected.items():
        if stat == GAMES_STAT or stat not in actual:
            continue
        mixed[stat] = (1.0 - share) * value + share * float(actual[stat])
    return mixed
