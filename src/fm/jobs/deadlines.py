"""Deadlines and decision windows for the tick, computed from the pro schedule and the league settings (DESIGN 9.1, 13).

Nothing here is a weekday or a clock time: every instant comes from the league's own data.

- **Deadlines** (:class:`Deadline`) are when something closes in a scoring period: each distinct **lineup lock**
  (``plugin.lock_windows`` under ``LeagueSettings.lineup_lock_type``: every kickoff in the NFL league, international
  and holiday slots included, since each game's start is a lock; every tip in the NBA league), each **roster lock**
  (adds, drops and trades, under ``roster_lock_type``: the day's first tip in the real NBA league), the next **waiver
  runs** (``acquisitionSettings.waiverProcessDays`` / ``waiverProcessHour``, US Eastern), and the **period turn**,
  when ESPN moves on to the next period (03:00 ET after the fantasy day of the period's last game,
  :data:`fm.sports.base.PERIOD_TURN`). A lock type the plugins refuse to read (``UNKNOWN``) yields no deadlines of its
  kind and a warning, never a guess.
- **Windows** (:class:`RunWindow`) are when a decision should run, each derived from a deadline: the period's opening
  (its turn) runs every registered decision; :data:`LOCK_LEAD` before each lineup lock (90 minutes in the NFL, when
  the inactives are out; 30 before an NBA tip, for late scratches) runs the lineup decisions; the same lead before a
  roster lock that is not also a lineup lock runs the acquisition decisions; :data:`WAIVER_LEAD` before a waiver run
  runs the waiver decision. A window is **due** between its opening and its deadline until it has run once (the tick
  records each run), and **missed** when it closed since the previous tick without running: the PC was asleep or the
  tick did not fire, and the first tick after reports it (DESIGN section 13).

The tick (:mod:`fm.jobs.tick`) asks :func:`upcoming` for the deadlines between its previous run and its horizon and
:func:`run_windows` for the windows, then :func:`due_windows` and :func:`missed_windows` against its recorded runs.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, time, timedelta
from enum import StrEnum
from types import MappingProxyType
from typing import Final
from zoneinfo import ZoneInfo

from fm.config import Sport
from fm.espn.settings import AcquisitionSettings, LeagueSettings, LockType
from fm.model.availability import RESOLUTION_LEAD
from fm.sports.base import (
    EASTERN,
    ScheduleLike,
    SportPlugin,
    fantasy_day,
    first_start,
    last_start,
    period_turn,
    plugin_for,
)

LOCK_LEAD: Mapping[Sport, timedelta] = RESOLUTION_LEAD
"""How long before a lock its decision window opens: when a player's status for the game is known (NFL inactives 90
minutes before kickoff, NBA late scratches 30 minutes before the tip; :data:`fm.model.availability.RESOLUTION_LEAD`)."""
WAIVER_LEAD: Final = timedelta(hours=2)
"""How long before the waiver run the waiver decision runs, so the claims are in when ESPN processes them."""
DEFAULT_HORIZON: Final = timedelta(days=7)
"""How far ahead :func:`upcoming` looks by default: a week covers the next NFL period and waiver run."""
FIRST_PERIOD_LEAD: Final = timedelta(days=7)
"""How long before the schedule's first game the first period counts as open (there is no earlier turn)."""
WEEKDAYS: Final[tuple[str, ...]] = ("MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY")
"""ESPN's ``waiverProcessDays`` names, Monday first, as ``date.weekday`` numbers them."""


class DeadlineKind(StrEnum):
    LINEUP_LOCK = "lineup_lock"
    ROSTER_LOCK = "roster_lock"
    WAIVER_RUN = "waiver_run"
    PERIOD_TURN = "period_turn"


class WindowKind(StrEnum):
    PERIOD_OPEN = "period_open"
    PRE_LOCK = "pre_lock"
    PRE_ROSTER_LOCK = "pre_roster_lock"
    PRE_WAIVER = "pre_waiver"


LINEUP_DECISIONS: Final = frozenset({"lineup", "lineup_daily"})
"""Decision kinds a lineup-lock window runs (:mod:`fm.decide.registry` kinds)."""
ACQUISITION_DECISIONS: Final = frozenset({"waivers", "streaming"})
"""Decision kinds a roster-lock window runs."""
WAIVER_DECISIONS: Final = frozenset({"waivers"})
WINDOW_DECISIONS: Mapping[WindowKind, frozenset[str] | None] = MappingProxyType(
    {
        WindowKind.PERIOD_OPEN: None,
        WindowKind.PRE_LOCK: LINEUP_DECISIONS,
        WindowKind.PRE_ROSTER_LOCK: ACQUISITION_DECISIONS,
        WindowKind.PRE_WAIVER: WAIVER_DECISIONS,
    }
)
"""Which registered decision kinds each window runs; ``None`` runs every decision registered for the sport."""


@dataclass(frozen=True, slots=True, order=True)
class Deadline:
    """Something closes at ``at`` (aware UTC) in ``period``: ``teams`` are the pro teams locking then (lock kinds),
    ``provisional`` says a start time is still a placeholder (NFL flex scheduling)."""

    at: datetime
    kind: DeadlineKind
    league_key: str
    period: int
    teams: tuple[int, ...] = ()
    provisional: bool = False

    def describe(self) -> str:
        local = self.at.astimezone(EASTERN)
        what = {
            DeadlineKind.LINEUP_LOCK: f"lineup lock ({len(self.teams)} teams)",
            DeadlineKind.ROSTER_LOCK: f"roster lock ({len(self.teams)} teams)",
            DeadlineKind.WAIVER_RUN: "waiver run",
            DeadlineKind.PERIOD_TURN: "period turns",
        }[self.kind]
        note = " (provisional)" if self.provisional else ""
        return f"{local:%a %Y-%m-%d %H:%M} ET {self.league_key} period {self.period}: {what}{note}"


@dataclass(frozen=True, slots=True)
class RunWindow:
    """A decision run: due from ``opens_at`` until ``closes_at`` (the deadline it serves), once."""

    league_key: str
    kind: WindowKind
    opens_at: datetime
    closes_at: datetime
    deadline: Deadline
    decisions: frozenset[str] | None = None
    """The decision kinds to run; ``None`` means every decision registered for the league's sport."""

    @property
    def key(self) -> str:
        """``nfl:pre_lock:2026-10-04T15:30Z``: what the tick records once the window has run."""
        return f"{self.league_key}:{self.kind.value}:{self.opens_at.astimezone(UTC):%Y-%m-%dT%H:%MZ}"

    @property
    def period(self) -> int:
        return self.deadline.period

    def runs(self, decision_kind: str) -> bool:
        return self.decisions is None or decision_kind in self.decisions

    def describe(self) -> str:
        local = self.closes_at.astimezone(EASTERN)
        return f"{self.league_key} {self.kind.value} before {local:%a %H:%M} ET ({self.deadline.kind.value})"


# --- deadlines --------------------------------------------------------------------------------------------------------


def next_waiver_run(acquisition: AcquisitionSettings, after: datetime, *, zone: ZoneInfo = EASTERN) -> datetime | None:
    """The first waiver run strictly after ``after``: the next of ``waiver_process_days`` at ``waiver_process_hour``
    (US Eastern wall clock, as ESPN schedules it), as aware UTC. ``None`` when the settings carry no run days or
    hour."""
    hour = acquisition.waiver_process_hour
    days = {WEEKDAYS.index(day) for day in acquisition.waiver_process_days if day in WEEKDAYS}
    if hour is None or not 0 <= hour <= 23 or not days:
        return None
    local = after.astimezone(zone)
    for offset in range(8):
        day = local.date() + timedelta(days=offset)
        if day.weekday() not in days:
            continue
        run = datetime.combine(day, time(hour), tzinfo=zone)
        if run > local:
            return run.astimezone(UTC)
    return None


def period_opens_at(schedule: ScheduleLike, period: int) -> datetime | None:
    """When ``period`` becomes current: the turn after the previous period's last game, or
    :data:`FIRST_PERIOD_LEAD` before its first game when the schedule has no earlier period. ``None`` when the
    schedule has no games in the period."""
    first = first_start(schedule, period)
    if first is None:
        return None
    earlier = [p for p in sorted(schedule.scoring_periods) if p < period and last_start(schedule, p) is not None]
    if not earlier:
        return first - FIRST_PERIOD_LEAD
    previous = last_start(schedule, earlier[-1])
    assert previous is not None
    return period_turn(fantasy_day(previous) + timedelta(days=1))


def period_turns_at(schedule: ScheduleLike, period: int) -> datetime | None:
    """When ESPN moves on from ``period``: 03:00 ET after the fantasy day of its last game."""
    last = last_start(schedule, period)
    return None if last is None else period_turn(fantasy_day(last) + timedelta(days=1))


def lock_deadlines(
    plugin: SportPlugin,
    schedule: ScheduleLike,
    period: int,
    *,
    league_key: str,
    lock_type: LockType,
    kind: DeadlineKind,
) -> tuple[Deadline, ...]:
    """One deadline per distinct lock instant in the period, with the pro teams that lock then. Raises ``ValueError``
    for an ``UNKNOWN`` lock type (the plugin's refusal)."""
    by_instant: dict[datetime, list[int]] = {}
    provisional: dict[datetime, bool] = {}
    for lock in plugin.locks(period, schedule, lock_type=lock_type):
        by_instant.setdefault(lock.at, []).append(lock.team_id)
        provisional[lock.at] = provisional.get(lock.at, False) or lock.provisional
    return tuple(
        Deadline(
            at=at,
            kind=kind,
            league_key=league_key,
            period=period,
            teams=tuple(sorted(set(teams))),
            provisional=provisional[at],
        )
        for at, teams in sorted(by_instant.items())
    )


def upcoming(
    league_key: str,
    sport: Sport,
    settings: LeagueSettings,
    schedule: ScheduleLike,
    *,
    now: datetime,
    since: datetime | None = None,
    period: int | None = None,
    horizon: timedelta = DEFAULT_HORIZON,
) -> tuple[tuple[Deadline, ...], tuple[str, ...]]:
    """Every deadline from ``since`` (default ``now``; the tick passes its previous run, so what closed in between
    is seen) to ``now + horizon`` for the league, ascending, plus warnings for what could not be computed (an
    ``UNKNOWN`` lock type, no waiver schedule).

    Covers the period current at ``since`` through every period that starts inside the horizon, so the next period's
    early locks are visible before it turns. ``period`` pins the first period instead of reading it off the schedule.
    """
    plugin = plugin_for(sport)
    start = now if since is None or since > now else since
    end = now + horizon
    first_period = period if period is not None else plugin.scoring_period_at(start, schedule)
    if first_period is None:
        return (), (f"{league_key}: the pro schedule has no scoring period left at {start.isoformat()}",)
    found: list[Deadline] = []
    warnings: list[str] = []
    seen_unknown: set[DeadlineKind] = set()
    for target in (p for p in sorted(schedule.scoring_periods) if p >= first_period):
        first = first_start(schedule, target)
        if first is None or (target != first_period and first > end):
            continue
        for kind, lock_type in (
            (DeadlineKind.LINEUP_LOCK, settings.lineup_lock_type),
            (DeadlineKind.ROSTER_LOCK, settings.roster_lock_type),
        ):
            try:
                found.extend(
                    lock_deadlines(plugin, schedule, target, league_key=league_key, lock_type=lock_type, kind=kind)
                )
            except ValueError as exc:
                if kind not in seen_unknown:
                    seen_unknown.add(kind)
                    warnings.append(f"{league_key}: no {kind.value} deadlines: {exc}")
        turn = period_turns_at(schedule, target)
        if turn is not None:
            found.append(Deadline(at=turn, kind=DeadlineKind.PERIOD_TURN, league_key=league_key, period=target))
    run_after = start - timedelta(seconds=1)
    while (run := next_waiver_run(settings.acquisition, run_after)) is not None and run <= end:
        run_period = plugin.scoring_period_at(run, schedule)
        found.append(
            Deadline(
                at=run,
                kind=DeadlineKind.WAIVER_RUN,
                league_key=league_key,
                period=run_period if run_period is not None else first_period,
            )
        )
        run_after = run
    if next_waiver_run(settings.acquisition, now) is None:
        warnings.append(f"{league_key}: the league settings carry no waiver run days or hour; no waiver deadlines")
    within = sorted(deadline for deadline in found if start <= deadline.at <= end)
    return tuple(within), tuple(warnings)


# --- windows ----------------------------------------------------------------------------------------------------------


def run_windows(
    league_key: str,
    sport: Sport,
    deadlines: Iterable[Deadline],
    *,
    schedule: ScheduleLike,
    now: datetime,
    lock_lead: Mapping[Sport, timedelta] = LOCK_LEAD,
    waiver_lead: timedelta = WAIVER_LEAD,
) -> tuple[RunWindow, ...]:
    """The decision windows the deadlines imply, plus the opening of the period current at ``now`` (from its turn
    until its first game), ascending by opening time."""
    plugin = plugin_for(sport)
    listed = sorted(deadlines)
    windows: list[RunWindow] = []
    lineup_locks = {d.at for d in listed if d.kind is DeadlineKind.LINEUP_LOCK}
    lead = lock_lead[sport]
    for deadline in listed:
        if deadline.kind is DeadlineKind.LINEUP_LOCK:
            windows.append(_window(deadline, WindowKind.PRE_LOCK, deadline.at - lead))
        elif deadline.kind is DeadlineKind.ROSTER_LOCK and deadline.at not in lineup_locks:
            windows.append(_window(deadline, WindowKind.PRE_ROSTER_LOCK, deadline.at - lead))
        elif deadline.kind is DeadlineKind.WAIVER_RUN:
            windows.append(_window(deadline, WindowKind.PRE_WAIVER, deadline.at - waiver_lead))
    current = plugin.scoring_period_at(now, schedule)
    if current is not None:
        opens = period_opens_at(schedule, current)
        closes = first_start(schedule, current)
        if opens is not None and closes is not None and opens < closes:
            first_lock = Deadline(at=closes, kind=DeadlineKind.PERIOD_TURN, league_key=league_key, period=current)
            windows.append(_window(first_lock, WindowKind.PERIOD_OPEN, opens))
    return tuple(sorted(windows, key=lambda window: (window.opens_at, window.closes_at, window.kind.value)))


def _window(deadline: Deadline, kind: WindowKind, opens_at: datetime) -> RunWindow:
    return RunWindow(
        league_key=deadline.league_key,
        kind=kind,
        opens_at=opens_at,
        closes_at=deadline.at,
        deadline=deadline,
        decisions=WINDOW_DECISIONS[kind],
    )


def due_windows(windows: Iterable[RunWindow], *, now: datetime, ran: Iterable[str]) -> tuple[RunWindow, ...]:
    """The windows open at ``now`` that have not run (``ran`` holds the keys of the runs recorded)."""
    done = set(ran)
    return tuple(w for w in windows if w.opens_at <= now < w.closes_at and w.key not in done)


def missed_windows(
    windows: Iterable[RunWindow], *, now: datetime, last_tick: datetime | None, ran: Iterable[str]
) -> tuple[RunWindow, ...]:
    """The windows that closed since ``last_tick`` without running: what the first tick after a gap reports. With no
    previous tick nothing counts as missed."""
    if last_tick is None:
        return ()
    done = set(ran)
    return tuple(w for w in windows if last_tick < w.closes_at <= now and w.key not in done)


__all__ = [
    "ACQUISITION_DECISIONS",
    "DEFAULT_HORIZON",
    "FIRST_PERIOD_LEAD",
    "LINEUP_DECISIONS",
    "LOCK_LEAD",
    "WAIVER_DECISIONS",
    "WAIVER_LEAD",
    "WEEKDAYS",
    "WINDOW_DECISIONS",
    "Deadline",
    "DeadlineKind",
    "RunWindow",
    "WindowKind",
    "due_windows",
    "lock_deadlines",
    "missed_windows",
    "next_waiver_run",
    "period_opens_at",
    "period_turns_at",
    "run_windows",
    "upcoming",
]
