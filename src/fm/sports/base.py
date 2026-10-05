"""Sport plugins: the per-sport rules the engine needs, behind one interface (DESIGN section 5, ``sports/``).

A :class:`SportPlugin` answers four questions about its sport:

- **Stat schema** (:class:`StatSchema`): the sport's stat vocabulary, ESPN stat ids and their abbreviations, which are
  the keys of every stat line in the store (projections are stat lines, never points; CLAUDE.md).
- **Slot eligibility**: which lineup slots a position may fill (``FLEX`` and ``OP`` in NFL; ``G``, ``F`` and ``UTIL``
  in NBA). ESPN sends ``eligibleSlots`` per player and that wins where present; the plugin table is the rule for a
  position and the fallback for a player without one.
- **Scoring periods** (:class:`PeriodKind`): a week in NFL, a day in NBA, and where one period gives way to the next
  given the pro schedule. ESPN turns its periods at 03:00 US Eastern (:data:`PERIOD_TURN`), not at midnight and not
  when the last game ends.
- **Lock times**: when each player locks, from the pro schedule and the league's lock types
  (``LeagueSettings.lineup_lock_type`` for lineups, ``roster_lock_type`` for adds, drops and trades: at each player's
  own game, or everyone at the period's first game). Nothing here is hardcoded to a kickoff time or a weekday; league
  settings are data.

The pro schedule is ESPN's ``proTeamSchedules_wl`` view, which the ESPN read client parses into
``fm.espn.models.ProSchedule`` (ROADMAP #9). This module does not import that model: it asks for the small
:class:`ScheduleLike` surface (games per period, games per team, idle teams) so any object of that shape serves,
including test doubles over recorded fixtures. Start times are aware UTC datetimes (ESPN's ``date``, epoch
milliseconds upstream). A game with ``startTimeTBD`` (flex scheduling in NFL weeks 16-18) carries a placeholder early
on its day (03:01 US Eastern in the 2026 capture), earlier than any real kickoff and so a conservative lock estimate;
``start_time_tbd`` / ``valid_for_locking`` mark such a lock provisional.

Plugins are stateless singletons found by convention, like command modules: ``plugin_for("nfl")`` imports
``fm.sports.nfl`` and reads its ``PLUGIN`` attribute, so adding a sport (``fm.sports.nba``, ROADMAP #17) touches no
shared file. Decision modules register per sport in :mod:`fm.decide.registry`.
"""

from __future__ import annotations

import importlib
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from enum import StrEnum
from functools import cached_property
from types import MappingProxyType
from typing import Annotated, Any, ClassVar, Final, Protocol
from zoneinfo import ZoneInfo

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict

from fm.config import Sport
from fm.espn.ids import Game, IdMaps, StatDef, ids_for
from fm.espn.settings import LeagueSettings, LockType

PLUGIN_ATTR = "PLUGIN"
FREE_AGENT_TEAM = 0  # ESPN's pro team id for unsigned players; it never has a game
_PLACEHOLDER_STAT = re.compile(r"^STAT_(\d+)$")  # the label IdMaps gives an unknown stat id

EASTERN: Final = ZoneInfo("America/New_York")
"""ESPN's fantasy calendar runs on US Eastern time."""
PERIOD_TURN: Final = time(3)
"""When ESPN's scoring periods turn, US Eastern time. Its web client's calendar runs every period from 3 a.m. to 3 a.m.
ET (docs/espn-api.md section 3), so a game still on after midnight is in the period it started in, and a lineup set
before 3 a.m. is still for that period (``ROSTER``, not ``FUTURE_ROSTER``)."""


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


type UtcInstant = Annotated[AwareDatetime, AfterValidator(_to_utc)]
"""An aware datetime, normalised to UTC on validation; naive values are rejected."""


def _require_aware(at: datetime, what: str = "at") -> None:
    if at.tzinfo is None or at.utcoffset() is None:
        raise ValueError(f"{what} must be an aware datetime (UTC); got a naive {at.isoformat()}")


def fantasy_day(at: datetime) -> date:
    """The day of ESPN's fantasy calendar an aware instant falls in: its US Eastern date, except that the day turns at
    :data:`PERIOD_TURN` (03:00 ET) rather than at midnight."""
    _require_aware(at)
    local = at.astimezone(EASTERN)
    return local.date() if local.time() >= PERIOD_TURN else local.date() - timedelta(days=1)


def period_turn(day: date) -> datetime:
    """When ESPN's fantasy ``day`` begins, and the day before it ends: :data:`PERIOD_TURN` (03:00 US Eastern) on that
    date, as an aware UTC instant."""
    return datetime.combine(day, PERIOD_TURN, tzinfo=EASTERN).astimezone(UTC)


class PeriodKind(StrEnum):
    """What one ESPN ``scoringPeriodId`` spans in this sport."""

    WEEK = "week"
    DAY = "day"


# --- the schedule surface ---------------------------------------------------------------------------------------------


class GameLike(Protocol):
    """What the plugins read from a pro game; ``fm.espn.models.ProGame`` has this shape.

    ``date`` is ESPN's name for the start instant (kickoff or tip), an aware UTC datetime. ``id`` may be ``None`` for a
    game ESPN has not numbered yet.
    """

    @property
    def id(self) -> int | None: ...

    @property
    def date(self) -> datetime: ...

    @property
    def home_pro_team_id(self) -> int: ...

    @property
    def away_pro_team_id(self) -> int: ...

    @property
    def start_time_tbd(self) -> bool: ...

    @property
    def valid_for_locking(self) -> bool: ...


class ScheduleLike(Protocol):
    """What the plugins read from a pro schedule; ``fm.espn.models.ProSchedule`` has this shape.

    ``games`` lists each game of a scoring period once (ESPN files it under both teams); ``idle_teams`` are the pro
    teams without a game in the period (NFL byes, NBA off days), excluding the free-agent pseudo-team.
    """

    @property
    def scoring_periods(self) -> Sequence[int]: ...

    def games(self, scoring_period: int) -> Sequence[GameLike]: ...

    def games_for(self, pro_team_id: int, scoring_period: int) -> Sequence[GameLike]: ...

    def idle_teams(self, scoring_period: int) -> Sequence[int]: ...


def is_provisional(game: GameLike) -> bool:
    """True when ESPN has not fixed the start time, so ``date`` is a placeholder rather than the kickoff."""
    return game.start_time_tbd or not game.valid_for_locking


def game_for(schedule: ScheduleLike, pro_team_id: int, scoring_period: int) -> GameLike | None:
    """A pro team's game in the period (the earliest, should a sport ever list two), or ``None`` on a bye or off day."""
    games = schedule.games_for(pro_team_id, scoring_period)
    return min(games, key=lambda game: game.date) if games else None


def start_times(schedule: ScheduleLike, scoring_period: int) -> tuple[datetime, ...]:
    """Distinct start instants in a period, ascending: the lineup windows (TNF, early Sunday, late, SNF, MNF)."""
    return tuple(sorted({game.date for game in schedule.games(scoring_period)}))


def first_start(schedule: ScheduleLike, scoring_period: int) -> datetime | None:
    starts = start_times(schedule, scoring_period)
    return starts[0] if starts else None


def last_start(schedule: ScheduleLike, scoring_period: int) -> datetime | None:
    starts = start_times(schedule, scoring_period)
    return starts[-1] if starts else None


def teams_playing(schedule: ScheduleLike, scoring_period: int) -> frozenset[int]:
    return frozenset(
        team_id for game in schedule.games(scoring_period) for team_id in (game.home_pro_team_id, game.away_pro_team_id)
    )


def teams_in(schedule: ScheduleLike, scoring_period: int) -> frozenset[int]:
    """Every pro team the schedule knows in a period, playing or idle; empty for a period it has no games in."""
    playing = teams_playing(schedule, scoring_period)
    return playing | frozenset(schedule.idle_teams(scoring_period)) if playing else frozenset()


# --- periods and locks ------------------------------------------------------------------------------------------------


class PeriodWindow(BaseModel):
    """The span of a scoring period's games: first start, last start, and ``end`` (last start plus the sport's game
    duration), when its games should be over. Its lineups stay current until ESPN's calendar turns, later
    (:meth:`SportPlugin.scoring_period_at`)."""

    model_config = ConfigDict(frozen=True)

    period: int
    first_start: UtcInstant
    last_start: UtcInstant
    end: UtcInstant


class LineupLock(BaseModel):
    """When players on one pro team lock in a period. ``game_id`` is ``None`` for a team without a game under a
    first-game lock (or a game ESPN has not numbered); ``provisional`` says the start time is still a placeholder."""

    model_config = ConfigDict(frozen=True)

    team_id: int
    period: int
    at: UtcInstant
    game_id: int | None = None
    provisional: bool = False


# --- stat schema ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class StatSchema:
    """A sport's stat vocabulary: ESPN stat ids and the abbreviations that key stat lines.

    ESPN's player views carry stats as ``{"53": 5.0, ...}`` (stat id to value); the store keeps stat lines keyed by
    abbreviation (``{"REC": 5.0}``) so a line reads like the league's scoring page and survives a renumbering.
    :meth:`from_espn` and :meth:`to_espn` convert, keeping an unknown id as the ``STAT_<id>`` placeholder that
    :class:`fm.espn.ids.IdMaps` uses, so a new ESPN stat shows up in output instead of vanishing.
    """

    game: Game
    stats: Mapping[int, StatDef]
    _ids_by_abbr: Mapping[str, int] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        by_abbr = {stat.abbr: stat_id for stat_id, stat in self.stats.items()}
        object.__setattr__(self, "_ids_by_abbr", MappingProxyType(by_abbr))

    @classmethod
    def for_game(cls, game: Game | str) -> StatSchema:
        ids = ids_for(game)
        return cls(game=ids.game, stats=ids.stats)

    @property
    def abbreviations(self) -> tuple[str, ...]:
        return tuple(self._ids_by_abbr)

    def __len__(self) -> int:
        return len(self.stats)

    def __contains__(self, key: object) -> bool:
        """``53 in schema`` by stat id, ``"REC" in schema`` by abbreviation; placeholders are not in the schema."""
        if isinstance(key, bool):
            return False
        if isinstance(key, int):
            return key in self.stats
        return isinstance(key, str) and key in self._ids_by_abbr

    def stat_id(self, abbr: str) -> int:
        """Reverse lookup by abbreviation; a ``STAT_<id>`` placeholder maps back to its id. Raises ``KeyError``."""
        stat_id = self._ids_by_abbr.get(abbr)
        if stat_id is not None:
            return stat_id
        placeholder = _PLACEHOLDER_STAT.match(abbr)
        if placeholder:
            return int(placeholder.group(1))
        raise KeyError(f"{self.game.value}: unknown stat abbreviation {abbr!r}")

    def abbr(self, stat_id: int) -> str:
        stat = self.stats.get(stat_id)
        return stat.abbr if stat else f"STAT_{stat_id}"

    def label(self, stat: int | str) -> str:
        stat_id = stat if isinstance(stat, int) else self.stat_id(stat)
        known = self.stats.get(stat_id)
        return known.label if known else f"Stat {stat_id}"

    def from_espn(self, raw: Mapping[Any, Any]) -> dict[str, float]:
        """An ESPN stats dict (ids as ints or strings) as an abbreviation-keyed stat line; ``None`` values are dropped.

        A non-numeric id or value raises ``ValueError``: a stat line that lost a stat silently would score wrong.
        """
        line: dict[str, float] = {}
        for key, value in raw.items():
            stat_id = _optional_int(key)
            if stat_id is None:
                raise ValueError(f"{self.game.value}: non-numeric ESPN stat id {key!r}")
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{self.game.value}: stat {stat_id} has a non-numeric value {value!r}")
            line[self.abbr(stat_id)] = float(value)
        return line

    def to_espn(self, line: Mapping[str, float]) -> dict[int, float]:
        """The reverse of :meth:`from_espn`. Raises ``KeyError`` for an abbreviation the sport does not have."""
        return {self.stat_id(abbr): float(value) for abbr, value in line.items()}

    def unknown(self, line: Mapping[str, Any]) -> tuple[str, ...]:
        """Keys of ``line`` that are not abbreviations of this sport (placeholders included), in line order."""
        return tuple(key for key in line if key not in self._ids_by_abbr)

    def scored_by(self, settings: LeagueSettings) -> tuple[str, ...]:
        """Abbreviations of the stats a league scores (or competes on), by stat id. The league must be this sport."""
        if settings.game is not self.game:
            raise ValueError(f"league {settings.league_id} is {settings.game.value}, not {self.game.value}")
        return tuple(item.stat for item in settings.scoring_items)


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value)
    return None


# --- the plugin interface ---------------------------------------------------------------------------------------------


class SportPlugin(ABC):
    """The sport protocol: what the engine asks a sport.

    A subclass sets ``game`` (ESPN game key), ``sport`` (the ``config.toml`` value), ``period_kind`` and
    ``game_duration``, and provides ``slot_positions``; everything else is derived here so NFL and NBA share one
    implementation of eligibility, period windows and lock times. ``game_duration`` is how long after its start a game
    is assumed to run; it only puts the end of a period's games (:class:`PeriodWindow`). Which period is current
    follows ESPN's calendar instead (:meth:`scoring_period_at`).

    Plugins are stateless; each module exposes one instance as ``PLUGIN`` for :func:`plugin_for`.
    """

    game: ClassVar[Game]
    sport: ClassVar[Sport]
    period_kind: ClassVar[PeriodKind]
    game_duration: ClassVar[timedelta]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        game = getattr(cls, "game", None)
        sport = getattr(cls, "sport", None)
        if isinstance(game, Game) and isinstance(sport, str) and Game.coerce(sport) is not game:
            raise TypeError(f"{cls.__name__}: sport {sport!r} is not ESPN game {game.value!r}")

    @property
    def ids(self) -> IdMaps:
        return ids_for(self.game)

    @cached_property
    def stat_schema(self) -> StatSchema:
        """The sport's stat ids and abbreviations (the keys of its stat lines)."""
        return StatSchema.for_game(self.game)

    @property
    @abstractmethod
    def slot_positions(self) -> Mapping[int, frozenset[str]]:
        """Active lineup slot id to the position labels that may fill it. Bench and IR are implied (every position)."""

    # --- slot eligibility

    def position_label(self, position: str | int) -> str:
        """Normalise a position given as a label or ESPN ``defaultPositionId``. Raises ``KeyError`` when unknown."""
        if isinstance(position, int):
            label = self.ids.positions.get(position)
            if label is None:
                raise KeyError(f"{self.game.value}: unknown position id {position}")
            return label
        self.ids.position_id(position)  # raises for an unknown label
        return position

    def positions_for_slot(self, slot_id: int) -> frozenset[str]:
        """Positions that may fill a slot; every position for bench and IR, none for a slot the sport lacks."""
        if slot_id in (self.ids.bench_slot, self.ids.ir_slot):
            return frozenset(self.ids.positions.values())
        return self.slot_positions.get(slot_id, frozenset())

    def eligible_slots(self, position: str | int, *, include_reserve: bool = True) -> frozenset[int]:
        """Slot ids a position may occupy, bench and IR included unless ``include_reserve`` is false."""
        label = self.position_label(position)
        slots = {slot_id for slot_id, positions in self.slot_positions.items() if label in positions}
        if include_reserve:
            slots |= {self.ids.bench_slot, self.ids.ir_slot}
        return frozenset(slots)

    def is_eligible(self, position: str | int, slot_id: int) -> bool:
        return slot_id in self.eligible_slots(position)

    # --- scoring periods

    def period_window(self, period: int, schedule: ScheduleLike) -> PeriodWindow | None:
        """First start, last start and end of a period's games; ``None`` when the schedule has no games in it."""
        first, last = first_start(schedule, period), last_start(schedule, period)
        if first is None or last is None:
            return None
        return PeriodWindow(period=period, first_start=first, last_start=last, end=last + self.game_duration)

    def scoring_period_at(self, at: datetime, schedule: ScheduleLike) -> int | None:
        """The scoring period whose lineups are current at ``at``, from the schedule alone.

        ESPN moves on from a period at :data:`PERIOD_TURN` (03:00 US Eastern) after the fantasy day of its last game:
        Tuesday 3 a.m. after Monday night, Monday 3 a.m. after a week that ends on Sunday. A period is current from the
        previous period's turn until its own; before the schedule's first game it is the first period, after the last
        period's turn ``None``. ESPN's own ``scoringPeriodId`` (``LeagueSettings.current_scoring_period``) is
        authoritative when fresh; this is the estimate for the ticks between syncs, and it never hardcodes a rollover
        weekday.
        """
        _require_aware(at)
        for period in sorted(schedule.scoring_periods):
            last = last_start(schedule, period)
            if last is not None and at < period_turn(fantasy_day(last) + timedelta(days=1)):
                return period
        return None

    # --- lock times

    def lock_time(
        self,
        team_id: int,
        period: int,
        schedule: ScheduleLike,
        *,
        lock_type: LockType = LockType.INDIVIDUAL_GAME,
    ) -> datetime | None:
        """When players on pro team ``team_id`` lock in ``period``.

        Per-game locking (``INDIVIDUAL_GAME``) locks at the team's own start time and never for a team without a game
        (a bye player can be moved all period, as on ESPN). Any first-game lock type locks everyone, bye teams
        included, at the period's first start. ``LockType.UNKNOWN`` raises ``ValueError``: guessing a lock rule could
        move a locked player, so an unrecognised ESPN value must be mapped first (ROADMAP #14).
        """
        if lock_type is LockType.UNKNOWN:
            raise ValueError(
                "lineup lock type is UNKNOWN; map ESPN's value (LeagueSettings.lineup_lock_type_raw) to "
                "INDIVIDUAL_GAME or a first-game lock before computing lock times"
            )
        if lock_type is LockType.INDIVIDUAL_GAME:
            game = game_for(schedule, team_id, period)
            return None if game is None else game.date
        return first_start(schedule, period)

    def is_locked(
        self,
        team_id: int,
        period: int,
        at: datetime,
        schedule: ScheduleLike,
        *,
        lock_type: LockType = LockType.INDIVIDUAL_GAME,
    ) -> bool:
        """True once ``team_id``'s players are locked at ``at`` (an aware datetime); a team without a lock never is."""
        _require_aware(at)
        lock = self.lock_time(team_id, period, schedule, lock_type=lock_type)
        return lock is not None and at >= lock

    def lock_windows(
        self,
        period: int,
        schedule: ScheduleLike,
        *,
        lock_type: LockType = LockType.INDIVIDUAL_GAME,
    ) -> tuple[datetime, ...]:
        """Distinct instants at which something locks in ``period``, ascending: the deadlines a tick works toward."""
        if lock_type is LockType.UNKNOWN:
            self.lock_time(FREE_AGENT_TEAM, period, schedule, lock_type=lock_type)  # raises with the explanation
        if lock_type is LockType.INDIVIDUAL_GAME:
            return start_times(schedule, period)
        first = first_start(schedule, period)
        return () if first is None else (first,)

    def locks(
        self,
        period: int,
        schedule: ScheduleLike,
        *,
        lock_type: LockType = LockType.INDIVIDUAL_GAME,
    ) -> tuple[LineupLock, ...]:
        """One :class:`LineupLock` per pro team that locks in ``period``, by lock time then team id.

        Under per-game locking that is every team with a game; under a first-game lock it is every team the schedule
        knows in the period, at the first start, with ``provisional`` reflecting the games that start then.
        """
        if lock_type is LockType.UNKNOWN:
            self.lock_time(FREE_AGENT_TEAM, period, schedule, lock_type=lock_type)  # raises with the explanation
        games = sorted(schedule.games(period), key=lambda game: game.date)
        if not games:
            return ()
        locks: list[LineupLock] = []
        if lock_type is LockType.INDIVIDUAL_GAME:
            for game in games:
                for team_id in (game.home_pro_team_id, game.away_pro_team_id):
                    locks.append(
                        LineupLock(
                            team_id=team_id,
                            period=period,
                            at=game.date,
                            game_id=game.id,
                            provisional=is_provisional(game),
                        )
                    )
        else:
            first = games[0].date
            provisional = any(is_provisional(game) for game in games if game.date == first)
            for team_id in teams_in(schedule, period):
                game = game_for(schedule, team_id, period)
                game_id = game.id if game is not None else None
                locks.append(
                    LineupLock(team_id=team_id, period=period, at=first, game_id=game_id, provisional=provisional)
                )
        return tuple(sorted(locks, key=lambda lock: (lock.at, lock.team_id)))

    def transaction_cutoff(
        self,
        team_id: int,
        period: int,
        schedule: ScheduleLike,
        *,
        lock_type: LockType,
    ) -> datetime | None:
        """When adds, drops and trades of players on pro team ``team_id`` close in ``period``, under the league's roster
        lock type (``LeagueSettings.roster_lock_type``, ESPN's ``rosterLocktimeType``; docs/espn-api.md section 1 #4).

        The rule is :meth:`lock_time`'s, applied to the roster lock instead of the lineup lock. ``INDIVIDUAL_GAME``
        closes them at the team's own start and never for a team without a game; a first-game lock
        (``FIRSTGAME_SCORINGPERIOD``, the real NBA league's) closes them for everyone at the period's first start, so
        a streamer must be added before the day's first tip (DESIGN section 9.3). ``LockType.UNKNOWN``, which ESPN's
        weekly types parse as, raises ``ValueError``, and there is no default: a guessed cutoff could plan an add that
        can no longer play or a drop of a player who is already locked.
        """
        if lock_type is LockType.UNKNOWN:
            raise ValueError(
                "roster lock type is UNKNOWN; map ESPN's value (LeagueSettings.roster_lock_type_raw) to "
                "INDIVIDUAL_GAME or a first-game lock before computing when adds and drops close"
            )
        return self.lock_time(team_id, period, schedule, lock_type=lock_type)


# --- discovery --------------------------------------------------------------------------------------------------------

PLUGIN_PACKAGE = __name__.rpartition(".")[0]  # fm.sports


def plugin_for(sport: Game | str, *, package: str = PLUGIN_PACKAGE) -> SportPlugin:
    """The plugin for a sport, given a ``Game``, a game key (``ffl``) or a sport (``nfl``).

    Imports ``<package>.<sport>`` and returns its ``PLUGIN``. An unsupported sport raises ``ValueError``; a sport
    whose module does not exist yet raises ``LookupError``; a module without a proper ``PLUGIN`` raises ``TypeError``.
    ``package`` exists so discovery can be tested against a synthetic package.
    """
    game = Game.coerce(sport)
    module_name = f"{package}.{game.sport}"
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        missing = exc.name or ""
        if missing != module_name and not module_name.startswith(f"{missing}."):
            raise  # the plugin exists but one of its own imports is broken
        raise LookupError(f"no sport plugin for {game.sport!r}: module {module_name} does not exist") from exc
    plugin = getattr(module, PLUGIN_ATTR, None)
    if not isinstance(plugin, SportPlugin):
        raise TypeError(f"{module_name}.{PLUGIN_ATTR} should be a SportPlugin instance, got {plugin!r}")
    if plugin.game is not game:
        raise TypeError(f"{module_name}.{PLUGIN_ATTR} is the {plugin.game.value} plugin, not {game.value}")
    return plugin
