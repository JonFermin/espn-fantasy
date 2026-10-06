"""Rest-of-season rankings sheet (DESIGN section 8.3, ROADMAP #35): every rostered player and free agent in a league,
valued for the rest of the season in that league's own scoring, as a sanity check for waiver and trade calls.

It replaces the draft sheet: both drafts are done, and what a player is worth now is what the rest of his season is
worth, in this league's points or categories, with the free agents next to the rostered players for comparison. Nothing
here proposes or writes anything (CLAUDE.md: workers propose, the executor acts); :func:`rank_league` reads the store
and returns a :class:`Rankings`, :func:`write_csv` saves one sheet per league.

**Points leagues.** The value is the player's rest-of-season points, and ``vor`` his value over replacement.

- NFL: :func:`fm.model.valuation.load_valuation` with ``include_rostered`` values every player on any roster and on the
  wire. A week is the blended projection times the chance he plays this week and later weeks are ESPN's season line per
  game times his games that week, so a bye is a zero; the league's playoff weeks count ``playoff_weight`` times a
  regular-season week. Replacement is the best player on the wire at his own position
  (:meth:`fm.model.valuation.RosterValuer.vor`).
- NBA: per-game points under the league's scoring items (:func:`fm.model.value_nba.scheduled_points`) times the games
  his pro team plays from the current day to the season's last (the pro schedule), the day's line being the blend of
  ESPN's per-game rate and DARKO (:func:`fm.model.value_nba.blend_day`). Every remaining day weighs the same: ESPN's
  playoff days are only known through its calendar (ROADMAP #31). Availability is not applied (a rest-of-season line has
  no day to apply it to), so an injured player is shown with his status, not discounted; a player off an active pro
  roster is worth nothing. Replacement is the best wire player eligible for the same active slot, as for the NFL.

**Category leagues** (NBA). Each player's per-game line is scaled to the games his team plays in the rest of the season
(:func:`fm.model.value_nba.period_lines`) and the league's own categories, in the settings' order and with
``isReverseItem`` honoured, are fit on those lines (:func:`fm.model.categories.fit_categories`): G-scores for
head-to-head leagues and z-scores for roto. The value is the sum of the categories' scores against the average player a
league could roster, ``scores`` has one column per category, and replacement is the best free agent's total, since the
categories give no slot to be scarce at. The G-score's ``tau`` (a player's matchup-to-matchup spread) needs game logs
the store does not keep, so unless the caller gives ``tau`` a G-score equals the z-score and says so in a warning.

Players nothing projects have no value, not a value of zero: they are listed last, unranked, with a note.
"""

from __future__ import annotations

import csv
import math
import re
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Final

from fm.espn.ids import Game, ids_for
from fm.espn.settings import LeagueSettings
from fm.model.categories import CategoryMetric, CategoryModel, fit_categories, metric_for
from fm.model.ids_nba import NBA_SPORT
from fm.model.projections import ESPN, BlendWeights, ProjectionSourceRegistry, source_registry
from fm.model.valuation import (
    BASIS_NONE,
    BASIS_SEASON,
    BASIS_THIS_WEEK,
    DEFAULT_PLAYOFF_WEIGHT,
    ValuationError,
    eligible_active_slots,
    last_scoring_period,
    league_settings,
    load_valuation,
    slot_instances,
)
from fm.model.value_nba import SEASON_PERIOD, blend_day, period_lines, scheduled_points
from fm.sports.base import ScheduleLike
from fm.store import LeagueRow, PlayerRow, Store

OWNER_FREE_AGENT: Final = "FA"
"""``owner`` of a player on no roster."""
OWNER_WAIVERS: Final = "waivers"
"""``owner`` of an unrostered player the caller says is on waivers (he cannot be added until a claim runs)."""
CSV_SUFFIX: Final = ".csv"
FIXED_COLUMNS: Final = ("rank", "player", "espn_id", "pos", "team", "owner", "mine", "injury")
"""The columns every sheet starts with; the value columns and the notes follow (:meth:`Rankings.header`)."""
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


class RankingsError(ValueError):
    """A league cannot be ranked as asked: not synced, no projections, or no pro schedule for the NBA's games."""


class RankingMetric(StrEnum):
    """What a ranking's value is: league points, or a category league's G-scores (head-to-head) or z-scores (roto)."""

    POINTS = "points"
    G = "g"
    Z = "z"

    @property
    def unit(self) -> str:
        return {"points": "points", "g": "G-score", "z": "z-score"}[self.value]


# --- the sheet --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RankedPlayer:
    """One row of the sheet.

    ``rank`` counts valued players by ``value`` (``None`` for a player nothing projects). ``owner`` is the fantasy
    team's name, :data:`OWNER_FREE_AGENT` or :data:`OWNER_WAIVERS`; ``team_id`` its id (``None`` on the wire). ``games``
    is the games the value covers (the pro team's remaining games for the NBA, ESPN's projected games for the NFL);
    ``per_game`` the league points of one game (points leagues); ``scores`` the score in each category (category
    leagues).
    """

    rank: int | None
    espn_id: int
    name: str
    positions: tuple[str, ...]
    pro_team: str | None
    team_id: int | None
    owner: str
    mine: bool
    injury: str | None
    value: float | None
    vor: float | None
    games: float | None = None
    per_game: float | None = None
    scores: Mapping[str, float] = MappingProxyType({})
    note: str = ""


@dataclass(frozen=True, slots=True)
class Rankings:
    """One league's rest-of-season sheet: ``players`` best first (the unvalued last), the ``categories`` that have a
    column (none in a points league), the replacement level behind ``vor`` as lines for the output, and the warnings
    collected on the way. ``scoring_period`` is the period the store's latest roster snapshot is for and
    ``last_period`` the season's last."""

    league: LeagueRow
    metric: RankingMetric
    scoring_period: int
    last_period: int
    players: tuple[RankedPlayer, ...]
    categories: tuple[str, ...] = ()
    replacement: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    as_of: datetime | None = None

    @property
    def ranked(self) -> tuple[RankedPlayer, ...]:
        """The players with a value."""
        return tuple(player for player in self.players if player.rank is not None)

    def header(self) -> tuple[str, ...]:
        """The CSV header: the fixed columns, then ``ros_value`` and ``vor`` with ``per_game`` and ``games`` (points) or
        ``games`` and one column per category (categories), then ``note``."""
        if self.metric is RankingMetric.POINTS:
            return (*FIXED_COLUMNS, "ros_value", "vor", "per_game", "games", "note")
        return (*FIXED_COLUMNS, "ros_value", "vor", "games", *self.categories, "note")

    def table(self) -> list[tuple[str, ...]]:
        """The header and every player's row as text, numbers to two places."""
        rows = [self.header()]
        for player in self.players:
            fixed = (
                "" if player.rank is None else str(player.rank),
                player.name,
                str(player.espn_id),
                "/".join(player.positions),
                player.pro_team or "",
                player.owner,
                "yes" if player.mine else "",
                player.injury or "",
            )
            games = _count(player.games)
            if self.metric is RankingMetric.POINTS:
                values = (_number(player.value), _number(player.vor), _number(player.per_game), games)
            else:
                categories = tuple(_number(player.scores.get(stat)) for stat in self.categories)
                values = (_number(player.value), _number(player.vor), games, *categories)
            rows.append((*fixed, *values, player.note))
        return rows

    @property
    def filename(self) -> str:
        """``rankings_<league key>.csv``, the key made safe for a file name."""
        return f"rankings_{_UNSAFE.sub('_', self.league.key)}{CSV_SUFFIX}"

    def describe(self) -> str:
        """One line for the CLI: the league, what the value is and how many players it ranks."""
        span = f"periods {self.scoring_period} to {self.last_period}"
        valued = len(self.ranked)
        return (
            f"{self.league.key}: {valued} of {len(self.players)} players valued in {self.metric.unit} ({span}), "
            f"{sum(player.owner not in (OWNER_FREE_AGENT, OWNER_WAIVERS) for player in self.players)} rostered"
        )


def _number(value: float | None) -> str:
    return "" if value is None or not math.isfinite(value) else f"{value:.2f}"


def _count(value: float | None) -> str:
    return "" if value is None or not math.isfinite(value) else f"{value:g}"


def write_csv(rankings: Rankings, directory: Path) -> Path:
    """Write the sheet to ``directory / rankings.filename`` (the directory is created) and return the path. UTF-8, one
    header row, ``\\n`` line ends."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / rankings.filename
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle, lineterminator="\n").writerows(rankings.table())
    return path


# --- one league -------------------------------------------------------------------------------------------------------


def rank_league(
    store: Store,
    league: LeagueRow,
    *,
    now: datetime,
    schedule: ScheduleLike | None = None,
    weights: BlendWeights | None = None,
    sources: ProjectionSourceRegistry | None = None,
    playoff_weight: float = DEFAULT_PLAYOFF_WEIGHT,
    tau: Mapping[str, float] | None = None,
    waivers: Collection[int] = (),
) -> Rankings:
    """The league's rest-of-season sheet from what ``fm sync`` stored (``now`` must be an aware datetime).

    ``schedule`` is the sport's pro schedule: it times byes and off days, and the NBA needs it to count games (without
    one, or one that does not cover the current period, an NFL league is valued as if no team had a bye and an NBA
    league raises :class:`RankingsError`). ``weights`` (default ``data/blend_weights.toml``) and ``sources`` (default:
    every registered projection source, DARKO included for the NBA) are the projection blend's; ``playoff_weight`` is
    the NFL playoff weeks' weight; ``tau`` is a category league's G-score ``tau`` by category; ``waivers`` are the ESPN
    ids on waivers rather than free to add. Raises :class:`RankingsError` for a league the store cannot value.
    """
    try:
        settings = league_settings(store, league)
        period = store.rosters.latest_period(league.row_id)
        if period is None:
            raise RankingsError(f"league {league.key!r} has no roster snapshot; run fm sync")
        last = last_scoring_period(settings)
        if last is None:
            raise RankingsError(f"league {league.key!r}: the season's last scoring period is unknown")
        warnings: list[str] = []
        usable = _usable_schedule(schedule, period, league.key, warnings)
        blend_weights = weights if weights is not None else BlendWeights.load()
        if league.sport == "nfl":
            built = _rank_nfl(store, league, settings, period, now, usable, blend_weights, playoff_weight)
        else:
            built = _rank_nba(store, league, settings, period, last, now, usable, blend_weights, sources, tau)
    except RankingsError:
        raise
    except ValuationError as exc:
        raise RankingsError(str(exc)) from exc
    except ValueError as exc:
        raise RankingsError(f"league {league.key!r}: {exc}") from exc
    sheet = _Sheet(store, league, period, waivers)
    players = sheet.assemble(built.candidates)
    return Rankings(
        league=league,
        metric=built.metric,
        scoring_period=period,
        last_period=last,
        players=players,
        categories=built.categories,
        replacement=built.replacement,
        warnings=(*warnings, *built.warnings),
        as_of=now,
    )


def _usable_schedule(schedule: ScheduleLike | None, period: int, key: str, warnings: list[str]) -> ScheduleLike | None:
    """``schedule`` when it covers ``period``; otherwise ``None`` with a warning (availability refuses a schedule that
    does not cover the period)."""
    if schedule is None:
        warnings.append(f"{key}: no pro schedule; byes and off days are not counted")
        return None
    if period not in tuple(schedule.scoring_periods):
        warnings.append(f"{key}: the pro schedule does not cover scoring period {period}; ignored")
        return None
    return schedule


@dataclass(frozen=True, slots=True)
class _Candidate:
    """A player's numbers before ranks and owners are attached."""

    espn_id: int
    value: float | None
    vor: float | None
    games: float | None = None
    per_game: float | None = None
    scores: Mapping[str, float] = MappingProxyType({})
    note: str = ""


@dataclass(frozen=True, slots=True)
class _Built:
    metric: RankingMetric
    candidates: tuple[_Candidate, ...]
    categories: tuple[str, ...] = ()
    replacement: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class _Sheet:
    """Attaches who owns each player and his name, positions and status, and orders the rows."""

    def __init__(self, store: Store, league: LeagueRow, period: int, waivers: Collection[int]) -> None:
        self.league = league
        self.waivers = frozenset(waivers)
        entries = store.rosters.league(league.row_id, period)
        self.team_of = {entry.espn_id: entry.team_id for entry in entries}
        self.team_names = {team.team_id: team.name for team in store.teams.for_league(league.row_id)}
        self.store = store

    def assemble(self, candidates: Iterable[_Candidate]) -> tuple[RankedPlayer, ...]:
        found = list(candidates)
        known = self.store.players.many(self.league.sport, {candidate.espn_id for candidate in found})
        rows = {player.espn_id: player for player in known}
        valued = sorted(
            (candidate for candidate in found if candidate.value is not None),
            key=lambda candidate: (-(candidate.value or 0.0), rows[candidate.espn_id].full_name, candidate.espn_id),
        )
        unvalued = sorted(
            (candidate for candidate in found if candidate.value is None),
            key=lambda candidate: (rows[candidate.espn_id].full_name, candidate.espn_id),
        )
        ranked: list[RankedPlayer] = []
        for index, candidate in enumerate((*valued, *unvalued), start=1):
            player = rows[candidate.espn_id]
            team_id = self.team_of.get(player.espn_id)
            ranked.append(
                RankedPlayer(
                    rank=index if candidate.value is not None else None,
                    espn_id=player.espn_id,
                    name=player.full_name,
                    positions=_positions(player),
                    pro_team=player.pro_team,
                    team_id=team_id,
                    owner=self._owner(player.espn_id, team_id),
                    mine=team_id == self.league.team_id,
                    injury=player.injury_status,
                    value=candidate.value,
                    vor=candidate.vor,
                    games=candidate.games,
                    per_game=candidate.per_game,
                    scores=candidate.scores,
                    note=candidate.note,
                )
            )
        return tuple(ranked)

    def _owner(self, espn_id: int, team_id: int | None) -> str:
        if team_id is None:
            return OWNER_WAIVERS if espn_id in self.waivers else OWNER_FREE_AGENT
        return self.team_names.get(team_id, f"Team {team_id}")


def _positions(player: PlayerRow) -> tuple[str, ...]:
    """His default position first, then every other position ESPN lists him eligible at (``PG/SG`` in the NBA)."""
    ids = ids_for(player.sport)
    labels = set(ids.positions.values())
    found: list[str] = [player.position] if player.position else []
    for slot_id in sorted(player.eligible_slot_ids):
        label = ids.slot_label(slot_id)
        if label in labels and label not in found:
            found.append(label)
    return tuple(found)


# --- NFL --------------------------------------------------------------------------------------------------------------


def _rank_nfl(
    store: Store,
    league: LeagueRow,
    settings: LeagueSettings,
    period: int,
    now: datetime,
    schedule: ScheduleLike | None,
    weights: BlendWeights,
    playoff_weight: float,
) -> _Built:
    valuation = load_valuation(
        store,
        league,
        now=now,
        settings=settings,
        schedule=schedule,
        weights=weights,
        playoff_weight=playoff_weight,
        include_rostered=True,
    )
    valuer = valuation.valuer()
    candidates: list[_Candidate] = []
    for espn_id, outlook in valuation.outlooks.items():
        player = valuation.players[espn_id]
        notes: list[str] = []
        if not player.active:
            notes.append("not on an active pro roster")
        if outlook.basis == BASIS_NONE:
            notes.append("no projection")
        elif outlook.basis == BASIS_THIS_WEEK:
            notes.append("this week's line stands in for every week")
        elif outlook.basis != BASIS_SEASON:
            notes.append("season line without a games count")
        known = outlook.has_projection
        candidates.append(
            _Candidate(
                espn_id=espn_id,
                value=valuer.ros(espn_id) if known else None,
                vor=valuer.vor(espn_id) if known else None,
                games=outlook.games,
                per_game=outlook.per_game if known else None,
                note="; ".join(notes),
            )
        )
    replacement = tuple(
        f"{level.label}: {level.name} ({level.value:.1f})"
        for level in valuer.replacement.values()
        if level.name is not None
    )
    return _Built(RankingMetric.POINTS, tuple(candidates), replacement=replacement, warnings=valuation.warnings)


# --- NBA --------------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _NbaInputs:
    players: Mapping[int, PlayerRow]
    rostered: frozenset[int]
    wire: frozenset[int]
    lines: Mapping[int, Mapping[str, float]]
    periods: range
    schedule: ScheduleLike
    warnings: tuple[str, ...]


def _rank_nba(
    store: Store,
    league: LeagueRow,
    settings: LeagueSettings,
    period: int,
    last: int,
    now: datetime,
    schedule: ScheduleLike | None,
    weights: BlendWeights,
    sources: ProjectionSourceRegistry | None,
    tau: Mapping[str, float] | None,
) -> _Built:
    if settings.game is not Game.FBA:
        raise RankingsError(f"league {league.key!r} is {settings.game.value}, not an NBA (fba) league")
    if schedule is None:
        raise RankingsError(
            f"league {league.key!r}: the NBA's games are counted from the pro schedule, which is missing"
        )
    inputs = _nba_inputs(store, league, period, last, now, schedule, weights, sources)
    if settings.is_points:
        return _nba_points(inputs, settings)
    return _nba_categories(inputs, settings, tau)


def _nba_inputs(
    store: Store,
    league: LeagueRow,
    period: int,
    last: int,
    now: datetime,
    schedule: ScheduleLike,
    weights: BlendWeights,
    sources: ProjectionSourceRegistry | None,
) -> _NbaInputs:
    """Everyone rostered in the league and on the wire (an ESPN-projected player on no roster), and the day's blended
    per-game lines. The blend is computed, not saved: a ranking changes nothing in the store."""
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError(f"now must be an aware datetime, got a naive {now.isoformat()}")
    rostered = frozenset(store.rosters.rostered_ids(league.row_id, period))
    espn_rows = store.projections.for_period(NBA_SPORT, league.season, SEASON_PERIOD, source=ESPN)
    wire = frozenset(row.espn_id for row in espn_rows) - rostered
    players = {player.espn_id: player for player in store.players.many(NBA_SPORT, rostered | wire)}
    blended = blend_day(
        store,
        league.season,
        period,
        weights=weights,
        sources=sources if sources is not None else source_registry,
        save=False,
    )
    lines = {row.espn_id: dict(row.stats) for row in blended.rows if row.espn_id in players}
    warnings = list(blended.warnings)
    days = range(period, last + 1)
    listed = set(schedule.scoring_periods)
    scheduled = sum(day in listed for day in days)
    if scheduled < len(days):
        warnings.append(
            f"{league.key}: the pro schedule lists games for {scheduled} of the {len(days)} days left; "
            "values count the scheduled games only"
        )
    gone = sorted((rostered | wire) - players.keys())
    if gone:
        warnings.append(f"{league.key}: {len(gone)} players have no players row and were not valued: {gone[:5]}")
    if not lines:
        raise RankingsError(f"league {league.key!r}: no NBA projections are stored; run fm sync")
    return _NbaInputs(
        players=MappingProxyType(players),
        rostered=rostered,
        wire=wire,
        lines=MappingProxyType(lines),
        periods=days,
        schedule=schedule,
        warnings=tuple(warnings),
    )


def _nba_points(inputs: _NbaInputs, settings: LeagueSettings) -> _Built:
    scheduled = scheduled_points(inputs.lines, inputs.players.values(), settings, inputs.schedule, inputs.periods)
    values: dict[int, float] = {}
    for espn_id, points in scheduled.items():
        values[espn_id] = points.total if inputs.players[espn_id].active else 0.0
    eligible = {espn_id: eligible_active_slots(player, settings) for espn_id, player in inputs.players.items()}
    levels = _slot_levels(values, eligible, inputs.wire, slot_instances(settings))
    labels = {slot.slot_id: slot.label for slot in settings.active_slots}
    candidates: list[_Candidate] = []
    for espn_id, player in inputs.players.items():
        points = scheduled.get(espn_id)
        if points is None:
            candidates.append(_Candidate(espn_id, None, None, note="no projection"))
            continue
        mine = [levels[slot][1] for slot in eligible[espn_id] if slot in levels]
        notes = [] if player.active else ["not on an active pro roster"]
        if points.games == 0:
            notes.append("no games on the pro schedule")
        candidates.append(
            _Candidate(
                espn_id=espn_id,
                value=values[espn_id],
                vor=values[espn_id] - min(mine) if mine else None,
                games=float(points.games),
                per_game=points.per_game,
                note="; ".join(notes),
            )
        )
    replacement = tuple(
        f"{labels.get(slot, f'SLOT_{slot}')}: {inputs.players[best].full_name} ({value:.1f})"
        for slot, (best, value) in sorted(levels.items())
        if best is not None
    )
    return _Built(RankingMetric.POINTS, tuple(candidates), replacement=replacement, warnings=inputs.warnings)


def _slot_levels(
    values: Mapping[int, float],
    eligible: Mapping[int, frozenset[int]],
    wire: Collection[int],
    slots: Sequence[int],
) -> dict[int, tuple[int | None, float]]:
    """For each active slot, the best wire player who can fill it by value (his ESPN id and value; ``(None, 0.0)`` when
    nobody on the wire can): replacement level at the slot, as :class:`fm.model.valuation.RosterValuer` has it."""
    levels: dict[int, tuple[int | None, float]] = {}
    for slot in dict.fromkeys(slots):
        fits = [espn_id for espn_id in wire if espn_id in values and slot in eligible.get(espn_id, frozenset())]
        best = min(fits, key=lambda espn_id: (-values[espn_id], espn_id), default=None)
        levels[slot] = (best, values[best] if best is not None else 0.0)
    return levels


def _nba_categories(inputs: _NbaInputs, settings: LeagueSettings, tau: Mapping[str, float] | None) -> _Built:
    scaled = period_lines(inputs.lines, inputs.players.values(), inputs.schedule, inputs.periods)
    if len(scaled) < 2:
        raise RankingsError(f"league {settings.league_id}: category scores need at least two projected players")
    model: CategoryModel = fit_categories(scaled, settings, tau=tau)
    metric = RankingMetric.G if metric_for(settings) is CategoryMetric.G else RankingMetric.Z
    warnings = [*inputs.warnings, *model.warnings]
    if metric is RankingMetric.G and not any(entry.tau > 0 for entry in model.stats):
        warnings.append(
            f"{settings.league_id}: no game history to estimate the G-score's tau from, so G-scores equal z-scores"
        )
    totals: dict[int, float] = {}
    candidates: list[_Candidate] = []
    scored = {entry.espn_id: entry for entry in model.rank(scaled)}
    for espn_id in inputs.players:
        entry = scored.get(espn_id)
        if entry is not None and not inputs.players[espn_id].active:
            totals[espn_id] = 0.0
        elif entry is not None:
            totals[espn_id] = entry.total
    best_free = max((totals[espn_id] for espn_id in inputs.wire if espn_id in totals), default=None)
    for espn_id, player in inputs.players.items():
        entry = scored.get(espn_id)
        if entry is None:
            candidates.append(_Candidate(espn_id, None, None, note="no projection"))
            continue
        games = scaled[espn_id].get("GP")
        notes = [] if player.active else ["not on an active pro roster"]
        if games == 0:
            notes.append("no games on the pro schedule")
        candidates.append(
            _Candidate(
                espn_id=espn_id,
                value=totals[espn_id],
                vor=totals[espn_id] - best_free if best_free is not None else None,
                games=games,
                scores=entry.scores,
                note="; ".join(notes),
            )
        )
    leader = max(
        (espn_id for espn_id in inputs.wire if espn_id in totals), key=lambda espn_id: totals[espn_id], default=None
    )
    replacement = (
        (f"best free agent: {inputs.players[leader].full_name} ({totals[leader]:.2f})",) if leader is not None else ()
    )
    return _Built(metric, tuple(candidates), model.categories, replacement, tuple(warnings))
