"""ESPN's season calendar: which scoring periods (days) each matchup period spans (ROADMAP #31, DESIGN section 9.3).

``scheduleSettings.matchupPeriods`` lists each matchup's schedule-period ids, and the ids are periods of the type
``scheduleSettings.periodTypeId`` names, not necessarily scoring periods. ``ffl`` and the stand-in NBA settings use type
1, where period N is scoring period N. The real NBA league uses the weekly type (2): matchup 1 is ``[1]``, and only
ESPN's web client knows that its week 1 is days 1-6 (Tue Oct 20 - Sun Oct 25, 2026), weeks 2-17 are Monday-Sunday
(7 days each), week 18 is the 14 days around the All-Star break (days 119-132) and weeks 19-21, the playoffs, are days
133-153. No read view carries that table: the web app bundles it as a constant, ``scripts/capture/webclient.py`` extracts
it, and this module is where the engine reads it from.

**The data.** One JSON file per game and season under ``data/calendars/`` (``fba_2027.json``, :func:`calendar_file`),
read through :func:`fm.paths.data_file`. The file is what ``capture.py webclient`` writes as ``calendar_<game>.json``
without its ``errorCodes`` (:func:`extract_calendar` makes that cut and validates the rest). To refresh a season,
re-run ``uv run python scripts/capture/capture.py webclient`` (it matches the bundle's calendar against the league's pro
schedule), then replace the file with ``extract_calendar``'s output. A season with no file has no calendar:
:func:`find_calendar` answers ``None``, :func:`load_calendar` raises :class:`CalendarError` naming the fix, and callers
fall back to what they did before the calendar existed (the weekly transaction cap counts the trailing seven days).

**Not here.** ESPN's weekly lock types (``FIRSTGAME_WEEKLY`` and ``INDIVIDUAL_FIRSTGAME_WEEKLY``) parse as ``UNKNOWN`` in
:mod:`fm.espn.settings` and the sport plugins refuse ``UNKNOWN``; this calendar is the data a future parser change
would resolve them against (:meth:`SeasonCalendar.matchup_days`), but nothing guesses a weekly lock until then.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any, Final

from pydantic import BaseModel, ConfigDict, ValidationError

from fm import paths
from fm.espn.ids import Game
from fm.espn.settings import LeagueSettings, ScheduleSettings

CALENDAR_DIR: Final = "calendars"
"""Where the per-season files live, under :func:`fm.paths.data_dir`."""
DATA_KEYS: Final = ("game", "season", "webClientBuild", "scoringPeriods", "periodTypes")
"""The keys of a calendar data file: ``capture.py webclient``'s ``calendar_<game>.json`` without ``errorCodes``."""


class CalendarError(ValueError):
    """A calendar file is missing or malformed, or a matchup period names a schedule period the calendar lacks."""


class ScoringDay(BaseModel):
    """One scoring period of the season: ``id``, its ESPN window (``start`` inclusive, ``end`` exclusive, aware UTC)
    and whether it belongs to the preseason (period 0) or the postseason. Period 1's ``start`` is a placeholder."""

    model_config = ConfigDict(frozen=True)

    id: int
    start: datetime
    end: datetime
    pre_season: bool = False
    post_season: bool = False


class CalendarPeriod(BaseModel):
    """One schedule period of a period type: the scoring periods ``first`` through ``last`` it spans."""

    model_config = ConfigDict(frozen=True)

    id: int
    first: int
    last: int

    @property
    def scoring_periods(self) -> tuple[int, ...]:
        return tuple(range(self.first, self.last + 1))


class CalendarPeriodType(BaseModel):
    """ESPN's grouping of scoring periods into ``daily``, ``weekly`` or season-long schedule periods."""

    model_config = ConfigDict(frozen=True)

    id: int
    daily: bool = False
    weekly: bool = False
    season_long: bool = False
    periods: tuple[CalendarPeriod, ...] = ()

    def period(self, period_id: int) -> CalendarPeriod:
        for period in self.periods:
            if period.id == period_id:
                return period
        raise CalendarError(f"period type {self.id} has no period {period_id}")


class SeasonCalendar(BaseModel):
    """One game's calendar for one season: its scoring periods and the period types that group them."""

    model_config = ConfigDict(frozen=True)

    game: Game
    season: int
    web_client_build: str | None = None
    scoring_periods: tuple[ScoringDay, ...]
    period_types: tuple[CalendarPeriodType, ...]

    def period_type(self, period_type_id: int) -> CalendarPeriodType:
        for period_type in self.period_types:
            if period_type.id == period_type_id:
                return period_type
        known = ", ".join(str(period_type.id) for period_type in self.period_types) or "none"
        raise CalendarError(f"{self.game.value} {self.season}: no period type {period_type_id}; known: {known}")

    def scoring_day(self, scoring_period: int) -> ScoringDay | None:
        return next((day for day in self.scoring_periods if day.id == scoring_period), None)

    @property
    def last_regular_period(self) -> int:
        """The season's last regular-season scoring period (the pro schedule's last; the postseason holds one more)."""
        regular = [day.id for day in self.scoring_periods if day.id > 0 and not day.post_season]
        return max(regular, default=0)

    def matchup_days(
        self, matchup_periods: Mapping[int, Sequence[int]], period_type_id: int
    ) -> dict[int, tuple[int, ...]]:
        """The scoring periods of each matchup period: ``scheduleSettings.matchupPeriods`` resolved through the period
        type ``period_type_id`` names, by matchup. Raises :class:`CalendarError` for a type or period the calendar
        lacks."""
        period_type = self.period_type(period_type_id)
        resolved: dict[int, tuple[int, ...]] = {}
        for matchup, period_ids in matchup_periods.items():
            days: list[int] = []
            for period_id in period_ids:
                days.extend(period_type.period(period_id).scoring_periods)
            resolved[matchup] = tuple(sorted(set(days)))
        return dict(sorted(resolved.items()))

    def schedule_days(self, schedule: ScheduleSettings) -> dict[int, tuple[int, ...]]:
        """:meth:`matchup_days` for a league's schedule settings. Under :data:`fm.espn.settings.SCORING_PERIOD_TYPE`
        the ids already are scoring periods and come back as they are; a schedule that names no period type raises
        :class:`CalendarError`, since its ids cannot be read."""
        if schedule.lists_scoring_periods:
            return dict(schedule.matchup_periods)
        if schedule.period_type_id is None:
            raise CalendarError("the schedule settings name no periodTypeId, so their matchup periods cannot be read")
        return self.matchup_days(schedule.matchup_periods, schedule.period_type_id)


# --- reading and extracting -------------------------------------------------------------------------------------------


def _instant(milliseconds: Any, what: str) -> datetime:
    if isinstance(milliseconds, bool) or not isinstance(milliseconds, int | float):
        raise CalendarError(f"{what} should be epoch milliseconds, got {milliseconds!r}")
    return datetime.fromtimestamp(milliseconds / 1000, tz=UTC)


def parse_calendar(data: Mapping[str, Any]) -> SeasonCalendar:
    """A calendar from its data file or from ``capture.py webclient``'s ``calendar_<game>.json`` (the extra
    ``errorCodes`` key is ignored). Raises :class:`CalendarError` for a missing key or a malformed entry."""
    missing = [key for key in ("game", "season", "scoringPeriods", "periodTypes") if key not in data]
    if missing:
        raise CalendarError(f"calendar is missing {', '.join(missing)}")
    try:
        days = tuple(
            ScoringDay(
                id=int(raw["id"]),
                start=_instant(raw["startDate"], f"scoring period {raw['id']} startDate"),
                end=_instant(raw["endDate"], f"scoring period {raw['id']} endDate"),
                pre_season=bool(raw.get("preSeason", False)),
                post_season=bool(raw.get("postSeason", False)),
            )
            for raw in data["scoringPeriods"]
        )
        types = tuple(
            CalendarPeriodType(
                id=int(raw["id"]),
                daily=bool(raw.get("daily", False)),
                weekly=bool(raw.get("weekly", False)),
                season_long=bool(raw.get("seasonLong", False)),
                periods=tuple(
                    CalendarPeriod(
                        id=int(period["id"]),
                        first=int(period["scoringPeriodStart"]),
                        last=int(period["scoringPeriodEnd"]),
                    )
                    for period in raw.get("periods") or ()
                ),
            )
            for raw in data["periodTypes"]
        )
        build = data.get("webClientBuild")
        return SeasonCalendar(
            game=Game.coerce(data["game"]),
            season=int(data["season"]),
            web_client_build=build if isinstance(build, str) else None,
            scoring_periods=days,
            period_types=types,
        )
    except (KeyError, TypeError, ValueError, ValidationError) as exc:
        if isinstance(exc, CalendarError):
            raise
        raise CalendarError(f"calendar is malformed: {type(exc).__name__}: {exc}") from exc


def extract_calendar(capture: Mapping[str, Any]) -> dict[str, Any]:
    """The data file for ``capture.py webclient``'s ``calendar_<game>.json``: its :data:`DATA_KEYS`, nothing else,
    after checking that it parses and that every period type's scoring periods exist in the calendar. Raises
    :class:`CalendarError`. Write the result with ``json.dump(..., separators=(",", ":"), sort_keys=True)`` to
    :func:`calendar_file`."""
    calendar = parse_calendar(capture)
    known = {day.id for day in calendar.scoring_periods}
    for period_type in calendar.period_types:
        for period in period_type.periods:
            absent = [day for day in period.scoring_periods if day not in known]
            if absent:
                raise CalendarError(
                    f"period type {period_type.id} period {period.id} spans scoring period {absent[0]}, "
                    "which the calendar does not list"
                )
    return {key: capture[key] for key in DATA_KEYS if key in capture}


def calendar_file(game: Game | str, season: int, *, root: Path | None = None) -> Path:
    """``data/calendars/<game>_<season>.json`` (``fba_2027.json``), through :func:`fm.paths.data_file` unless ``root``
    names another directory."""
    name = f"{Game.coerce(game).value}_{season}.json"
    if root is not None:
        return root / name
    return paths.data_file(f"{CALENDAR_DIR}/{name}")


@cache
def _read(path: str) -> SeasonCalendar | None:
    file = Path(path)
    if not file.is_file():
        return None
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise CalendarError(f"{file}: cannot be read ({type(exc).__name__}: {exc})") from exc
    if not isinstance(data, dict):
        raise CalendarError(f"{file}: should hold a JSON object")
    try:
        return parse_calendar(data)
    except CalendarError as exc:
        raise CalendarError(f"{file}: {exc}") from exc


def find_calendar(game: Game | str, season: int, *, root: Path | None = None) -> SeasonCalendar | None:
    """The season's calendar, or ``None`` when no file exists for it. A file that cannot be read raises
    :class:`CalendarError`. Files are read once per process."""
    return _read(str(calendar_file(game, season, root=root)))


def load_calendar(game: Game | str, season: int, *, root: Path | None = None) -> SeasonCalendar:
    """:func:`find_calendar`, raising :class:`CalendarError` that names the fix when there is no file."""
    calendar = find_calendar(game, season, root=root)
    if calendar is None:
        raise CalendarError(
            f"no {Game.coerce(game).value} {season} calendar at {calendar_file(game, season, root=root)}; "
            "run `uv run python scripts/capture/capture.py webclient` and save its calendar there "
            "(fm.espn.calendar.extract_calendar)"
        )
    return calendar


# --- a league's matchup weeks -----------------------------------------------------------------------------------------


def league_matchup_days(settings: LeagueSettings, *, root: Path | None = None) -> dict[int, tuple[int, ...]] | None:
    """The scoring periods of each of a league's matchup periods, or ``None`` when they cannot be told: the league
    lists its matchups in schedule periods the season's calendar has to resolve and there is no calendar file for the
    season (or it lacks a period the league lists)."""
    schedule = settings.schedule
    if schedule.lists_scoring_periods:
        return dict(schedule.matchup_periods)
    calendar = find_calendar(settings.game, settings.season, root=root)
    if calendar is None:
        return None
    try:
        return calendar.schedule_days(schedule)
    except CalendarError:
        return None


def matchup_scoring_periods(
    settings: LeagueSettings, scoring_period: int, *, root: Path | None = None
) -> tuple[int, ...] | None:
    """The scoring periods of the matchup that contains ``scoring_period``: the matchup week of a day. ``None`` when
    the period is in no listed matchup or the league's matchups cannot be resolved (:func:`league_matchup_days`)."""
    days = league_matchup_days(settings, root=root)
    if days is None:
        return None
    return next((span for span in days.values() if scoring_period in span), None)


def matchup_period_of(settings: LeagueSettings, scoring_period: int, *, root: Path | None = None) -> int | None:
    """The matchup period that contains ``scoring_period``, or ``None`` as in :func:`matchup_scoring_periods`."""
    days = league_matchup_days(settings, root=root)
    if days is None:
        return None
    return next((matchup for matchup, span in days.items() if scoring_period in span), None)
