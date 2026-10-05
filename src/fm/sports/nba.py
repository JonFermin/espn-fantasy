"""NBA plugin for ESPN ``fba`` leagues: daily scoring periods, per-game locks at tip, the add/drop cutoff at the day's
first tip, and slot eligibility including ``G`` (PG/SG), ``F`` (SF/PF), the combination slots and ``UTIL``.

**Eligibility** follows ESPN's position-to-slot rules for ``fba``, by position label from
:data:`fm.espn.ids.FBA_POSITIONS`; the ids come from :data:`fm.espn.ids.FBA`, never literals. The table matches the
``eligibleSlots`` ESPN sends (``G/F`` takes every guard and forward, ``F/C`` forwards and centers). ESPN grants many
players a second position (an ``SG`` with ``SF`` eligibility has ``SF``, ``F`` and ``F/C`` in his own
``eligibleSlots``), which callers prefer when they have it; this table is the rule for a position and the fallback.
``Rookie`` (slot 15) depends on experience, not position, so no position fills it here.

**Scoring periods are days.** ESPN numbers NBA scoring periods by US Eastern calendar day from opening night (day 1 is
Tue Oct 20, 2026 in the 2027 season; DESIGN section 9.3), and ``proTeamSchedules_wl`` files every game under its
Eastern day: a 10:30 p.m. ET tip is 02:30 UTC the next morning and still belongs to its day. Days without games
(Election Day, Thanksgiving, Christmas Eve, the All-Star break, an unscheduled NBA Cup knockout window) keep their
numbers and are simply absent from the schedule. So :meth:`NbaPlugin.scoring_period_at` answers from the calendar
rather than from game ends: the day turns at midnight ET, and a game still running then is a locked slot in the
previous day's lineup. The schedule anchors the count (its first day with a confirmed tip), and every other day's
first confirmed tip must sit at the matching offset; otherwise the plugin raises rather than guess which day a
lineup belongs to.

**Locks and adds.** Lineups lock per game (the base class) or, under ``FIRSTGAME_SCORINGPERIOD``, for everyone at the
day's first tip; the lock type is the league's (``LeagueSettings.lineup_lock_type``), passed in by the caller. A weekly
lock spans several daily periods and needs the league's matchup periods, which a pro schedule does not carry, so it
is refused like ``UNKNOWN``. Adds, drops and trades for a day close at its first tip even though lineups lock per game
(DESIGN section 9.3): :meth:`NbaPlugin.transaction_cutoff`.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from types import MappingProxyType
from typing import Final
from zoneinfo import ZoneInfo

from fm.espn.ids import FBA, Game
from fm.espn.settings import LockType
from fm.sports.base import LineupLock, PeriodKind, ScheduleLike, SportPlugin, first_start, is_provisional

EASTERN: Final = ZoneInfo("America/New_York")
"""ESPN's fantasy basketball day is the US Eastern calendar day."""

NBA_GAME_DURATION: Final = timedelta(hours=3)
"""How long after tip an NBA game is assumed to run (about 2 h 15 min typical, longer with overtime); only used for
``PeriodWindow.end``."""

WEEKLY_LOCK_TYPES: Final = frozenset({LockType.FIRST_GAME_OF_WEEK})
"""Lock types that span a matchup week. A day is the NBA scoring period, so these cannot be computed per period."""

_SLOT_POSITIONS_BY_LABEL: Mapping[str, frozenset[str]] = {
    "PG": frozenset({"PG"}),
    "SG": frozenset({"SG"}),
    "SF": frozenset({"SF"}),
    "PF": frozenset({"PF"}),
    "C": frozenset({"C"}),
    "G": frozenset({"PG", "SG"}),
    "F": frozenset({"SF", "PF"}),
    "SG/SF": frozenset({"SG", "SF"}),
    "G/F": frozenset({"PG", "SG", "SF", "PF"}),
    "PF/C": frozenset({"PF", "C"}),
    "F/C": frozenset({"SF", "PF", "C"}),
    "UTIL": frozenset({"PG", "SG", "SF", "PF", "C"}),
}


def _by_slot_id(table: Mapping[str, frozenset[str]]) -> Mapping[int, frozenset[str]]:
    """Resolve slot and position labels through the fba id maps, so a typo fails at import rather than in a lineup."""
    resolved: dict[int, frozenset[str]] = {}
    for slot_label, positions in table.items():
        for position in positions:
            FBA.position_id(position)  # KeyError on an unknown position label
        resolved[FBA.slot_id(slot_label)] = positions
    return MappingProxyType(resolved)


NBA_SLOT_POSITIONS: Mapping[int, frozenset[str]] = _by_slot_id(_SLOT_POSITIONS_BY_LABEL)
"""Active ``fba`` lineup slot id to the position labels that may fill it (bench and IR take every position)."""


def eastern_day(at: datetime) -> date:
    """The US Eastern calendar day an aware instant falls on: the fantasy day of a game that tips at ``at``."""
    if at.tzinfo is None or at.utcoffset() is None:
        raise ValueError(f"at must be an aware datetime (UTC); got a naive {at.isoformat()}")
    return at.astimezone(EASTERN).date()


@dataclass(frozen=True, slots=True)
class _DayNumbering:
    """A schedule's days: ``anchor_period`` is ``anchor_day``, and the span runs from its first period with games to
    its last."""

    anchor_period: int
    anchor_day: date
    first_period: int
    last_period: int

    def day(self, period: int) -> date:
        return self.anchor_day + timedelta(days=period - self.anchor_period)

    def period(self, day: date) -> int:
        return self.anchor_period + (day - self.anchor_day).days


class NbaPlugin(SportPlugin):
    """ESPN fantasy basketball: daily periods, per-game locks at tip, the first-tip add/drop cutoff, G/F/UTIL."""

    game = Game.FBA
    sport = "nba"
    period_kind = PeriodKind.DAY
    game_duration = NBA_GAME_DURATION

    @property
    def slot_positions(self) -> Mapping[int, frozenset[str]]:
        return NBA_SLOT_POSITIONS

    # --- days

    def period_day(self, period: int, schedule: ScheduleLike) -> date | None:
        """The US Eastern calendar day of a scoring period, days without games included; ``None`` outside the span
        from the schedule's first day with games to its last."""
        days = self._numbering(schedule)
        if days is None or not days.first_period <= period <= days.last_period:
            return None
        return days.day(period)

    def period_for_day(self, day: date, schedule: ScheduleLike) -> int | None:
        """The scoring period of a US Eastern calendar day; ``None`` outside the schedule's span."""
        days = self._numbering(schedule)
        if days is None or not days.day(days.first_period) <= day <= days.day(days.last_period):
            return None
        return days.period(day)

    def scoring_period_at(self, at: datetime, schedule: ScheduleLike) -> int | None:
        """The scoring period current at ``at``: the US Eastern calendar day it falls on, by the schedule's numbering.

        Before the schedule's first day it is the first period (opening night's lineup is the next one to set);
        after its last day, or for a schedule without games, ``None``. The pro schedule runs to the end of the NBA
        regular season, past a league's ``final_scoring_period``, which callers compare against. ESPN's own
        ``scoringPeriodId`` (``LeagueSettings.current_scoring_period``) is authoritative when fresh; this is the
        estimate for the ticks between syncs.
        """
        day = eastern_day(at)
        days = self._numbering(schedule)
        if days is None or day > days.day(days.last_period):
            return None
        return max(days.period(day), days.first_period)

    def _numbering(self, schedule: ScheduleLike) -> _DayNumbering | None:
        """Number the schedule's days from its first confirmed tip and check every other day against that count.

        Each period's earliest confirmed tip (placeholders for unscheduled games are skipped) must fall on the Eastern
        day its number implies; a schedule that breaks the pattern raises ``ValueError`` naming the period, because
        a misnumbered day would put a lineup on the wrong date.
        """
        with_games = [period for period in sorted(schedule.scoring_periods) if schedule.games(period)]
        anchors: list[tuple[int, date]] = []
        for period in with_games:
            confirmed = [game.date for game in schedule.games(period) if not is_provisional(game)]
            if confirmed:
                anchors.append((period, eastern_day(min(confirmed))))
        if not anchors:
            return None
        anchor_period, anchor_day = anchors[0]
        days = _DayNumbering(anchor_period, anchor_day, with_games[0], with_games[-1])
        for period, day in anchors[1:]:
            expected = days.day(period)
            if day != expected:
                raise ValueError(
                    f"{self.game.value} schedule: scoring period {period} first tips on {day} (US Eastern), but "
                    f"period {anchor_period} is {anchor_day}, which puts period {period} on {expected}; NBA scoring "
                    "periods are consecutive US Eastern days, so this schedule needs a look before a lineup uses it"
                )
        return days

    # --- adds and drops

    def transaction_cutoff(self, period: int, schedule: ScheduleLike) -> datetime | None:
        """When adds, drops and trades for ``period`` close: the day's first tip (DESIGN section 9.3), so a streamer
        must be added before it to play that day. ``None`` for a day without games. A placeholder start (an
        unscheduled game) counts, which errs early, like the lock times."""
        return first_start(schedule, period)

    # --- lock times

    def lock_time(
        self,
        team_id: int,
        period: int,
        schedule: ScheduleLike,
        *,
        lock_type: LockType = LockType.INDIVIDUAL_GAME,
    ) -> datetime | None:
        """As :meth:`SportPlugin.lock_time`, refusing a weekly lock type (see the module docstring)."""
        self._require_daily_lock(lock_type)
        return super().lock_time(team_id, period, schedule, lock_type=lock_type)

    def lock_windows(
        self,
        period: int,
        schedule: ScheduleLike,
        *,
        lock_type: LockType = LockType.INDIVIDUAL_GAME,
    ) -> tuple[datetime, ...]:
        """As :meth:`SportPlugin.lock_windows`, refusing a weekly lock type."""
        self._require_daily_lock(lock_type)
        return super().lock_windows(period, schedule, lock_type=lock_type)

    def locks(
        self,
        period: int,
        schedule: ScheduleLike,
        *,
        lock_type: LockType = LockType.INDIVIDUAL_GAME,
    ) -> tuple[LineupLock, ...]:
        """As :meth:`SportPlugin.locks`, refusing a weekly lock type."""
        self._require_daily_lock(lock_type)
        return super().locks(period, schedule, lock_type=lock_type)

    def _require_daily_lock(self, lock_type: LockType) -> None:
        if lock_type in WEEKLY_LOCK_TYPES:
            raise ValueError(
                f"lineup lock type {lock_type.value} locks a whole matchup week, but an {self.game.value} scoring "
                "period is one day; a weekly lock needs the league's matchup periods, which the pro schedule does "
                "not carry, so it is not computed per day"
            )


NBA: Final = NbaPlugin()
PLUGIN: Final = NBA  # discovered by fm.sports.base.plugin_for("nba")
