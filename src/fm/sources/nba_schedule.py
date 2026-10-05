"""The NBA's CDN schedule (``scheduleLeagueV2.json``): games per day and week, back-to-backs, NBA Cup placeholders.

``https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json`` is the file nba.com's schedule pages load: every
game of the season (preseason, regular season, the NBA Cup final) with ET and UTC tip times, arena, labels and a
``gameId`` whose third digit encodes the game type. It sits behind the same Akamai checks as stats.nba.com, and a bare
UA gets a 403, so :class:`NbaScheduleSource` sends the full Chrome header set and, like every NBA source, runs from the
home PC. The file is about 5 MB; one pull a day is plenty (``ttl``), and callers ``force`` a re-pull in early December
because the NBA Cup knockout games (and the consolation games for the 22 eliminated teams) are not scheduled until
group play ends. Until then the file carries placeholder knockout games with ``TBD`` teams and times:
:attr:`NbaGame.teams_tbd` flags them, :meth:`NbaSchedule.unscheduled_cup_games` lists them and the ``Fetched``
carries a warning.

Fantasy semantics belong to the consumers (ESPN scoring periods are ET days; matchup weeks come from league
settings), so this module speaks in ET calendar days (:attr:`NbaGame.day`, from ``gameDateEst``) and UTC tips.
Regular-season games (``gameId`` prefix ``002``) include the Cup group stage and knockout rounds; the Cup final
(``006``) does not count toward regular-season statistics and so is not a fantasy game.
"""

from __future__ import annotations

import dataclasses
import logging
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from enum import StrEnum
from itertools import pairwise
from types import MappingProxyType
from typing import Any, ClassVar, Unpack

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from fm.sources.base import Fetched, FetchOptions, HttpSource, SourceSchemaError, parse_json

logger = logging.getLogger(__name__)

CDN_URL = "https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json"

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
)

CHROME_HEADERS: Mapping[str, str] = MappingProxyType(
    {
        "User-Agent": BROWSER_USER_AGENT,
        "Accept": "application/json, text/plain, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Origin": "https://www.nba.com",
        "Referer": "https://www.nba.com/",
        "Sec-Ch-Ua": '"Google Chrome";v="145", "Chromium";v="145", "Not:A-Brand";v="99"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
    }
)
"""What Chrome sends when nba.com loads the schedule; the CDN answers 403 to a bare client UA."""

CUP_SUBTYPE_PREFIX = "in-season"
CUP_KNOCKOUT_SUBTYPE = "in-season-knockout"


class GameType(StrEnum):
    """Encoded in the third digit of ``gameId`` (``0022600119`` is a regular-season game)."""

    PRESEASON = "preseason"
    REGULAR_SEASON = "regular_season"
    ALL_STAR = "all_star"
    PLAYOFFS = "playoffs"
    PLAY_IN = "play_in"
    CUP_FINAL = "cup_final"
    UNKNOWN = "unknown"


_GAME_TYPES: Mapping[str, GameType] = MappingProxyType(
    {
        "1": GameType.PRESEASON,
        "2": GameType.REGULAR_SEASON,
        "3": GameType.ALL_STAR,
        "4": GameType.PLAYOFFS,
        "5": GameType.PLAY_IN,
        "6": GameType.CUP_FINAL,
    }
)


def game_type(game_id: str) -> GameType:
    return _GAME_TYPES.get(game_id[2:3], GameType.UNKNOWN) if len(game_id) >= 3 else GameType.UNKNOWN


def _parse_instant(value: object) -> datetime | None:
    """The CDN writes ``2026-10-26T23:00:00Z``; placeholders carry ``0001-01-01T00:00:00Z``."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        parsed = datetime.fromisoformat(value)
    else:
        raise ValueError(f"not a timestamp: {value!r}")
    if parsed.year <= 1:
        return None
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


class ScheduleModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)


class NbaTeamRef(ScheduleModel):
    """A side of a game. Knockout placeholders have ``teamId`` 0 and no tricode, which read as ``None``."""

    team_id: int | None = Field(default=None, alias="teamId")
    tricode: str | None = Field(default=None, alias="teamTricode")
    name: str | None = Field(default=None, alias="teamName")
    city: str | None = Field(default=None, alias="teamCity")
    slug: str | None = Field(default=None, alias="teamSlug")
    score: int = 0
    seed: int = 0

    @field_validator("team_id", mode="before")
    @classmethod
    def _zero_is_unknown(cls, value: object) -> object:
        return None if value in (0, "0", "") else value

    @field_validator("tricode", "name", "city", "slug", mode="before")
    @classmethod
    def _blank_or_tbd_is_unknown(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            return None if not value or value.upper() == "TBD" else value
        return value

    @property
    def is_tbd(self) -> bool:
        return self.team_id is None or self.tricode is None


class NbaGame(ScheduleModel):
    """One scheduled game. ``day`` is the ET calendar day (ESPN's NBA scoring period); ``tip_utc`` the tip-off."""

    game_id: str = Field(alias="gameId")
    game_code: str | None = Field(default=None, alias="gameCode")
    status: int = Field(default=1, alias="gameStatus")
    """1 scheduled, 2 in progress, 3 final."""
    status_text: str = Field(default="", alias="gameStatusText")
    day: date = Field(alias="gameDateEst")
    tip_utc: datetime | None = Field(default=None, alias="gameDateTimeUTC")
    """``None`` while the time is TBD (knockout placeholders)."""
    week_number: int = Field(default=0, alias="weekNumber")
    week_name: str = Field(default="", alias="weekName")
    label: str = Field(default="", alias="gameLabel")
    sub_label: str = Field(default="", alias="gameSubLabel")
    subtype: str = Field(default="", alias="gameSubtype")
    series_text: str = Field(default="", alias="seriesText")
    arena: str = Field(default="", alias="arenaName")
    arena_city: str = Field(default="", alias="arenaCity")
    arena_state: str = Field(default="", alias="arenaState")
    is_neutral: bool = Field(default=False, alias="isNeutral")
    postponed_status: str = Field(default="N", alias="postponedStatus")
    home: NbaTeamRef = Field(alias="homeTeam")
    away: NbaTeamRef = Field(alias="awayTeam")

    @model_validator(mode="before")
    @classmethod
    def _tbd_has_no_tip(cls, data: object) -> object:
        if isinstance(data, dict) and str(data.get("gameStatusText", "")).strip().upper() == "TBD":
            data = {**data, "gameDateTimeUTC": None}
        return data

    @field_validator("day", mode="before")
    @classmethod
    def _et_day(cls, value: object) -> object:
        if isinstance(value, str):
            return datetime.fromisoformat(value).date()
        return value

    @field_validator("tip_utc", mode="before")
    @classmethod
    def _instant(cls, value: object) -> datetime | None:
        return _parse_instant(value)

    @field_validator(
        "game_code", "status_text", "week_name", "label", "sub_label", "subtype", "series_text", "arena", mode="before"
    )
    @classmethod
    def _strip(cls, value: object) -> object:
        return value.strip() if isinstance(value, str) else value

    @field_validator("game_code", mode="after")
    @classmethod
    def _blank_code(cls, value: str | None) -> str | None:
        return value or None

    @property
    def game_type(self) -> GameType:
        return game_type(self.game_id)

    @property
    def is_regular_season(self) -> bool:
        """Counts toward regular-season statistics, so toward fantasy: includes Cup group and knockout games."""
        return self.game_type is GameType.REGULAR_SEASON

    @property
    def is_cup(self) -> bool:
        return self.subtype.startswith(CUP_SUBTYPE_PREFIX) or "nba cup" in self.label.lower()

    @property
    def is_cup_knockout(self) -> bool:
        return self.subtype == CUP_KNOCKOUT_SUBTYPE

    @property
    def teams_tbd(self) -> bool:
        return self.home.is_tbd or self.away.is_tbd

    @property
    def is_final(self) -> bool:
        return self.status == 3

    @property
    def tricodes(self) -> frozenset[str]:
        return frozenset(code for code in (self.home.tricode, self.away.tricode) if code is not None)

    @property
    def matchup(self) -> str:
        return f"{self.away.tricode or 'TBD'} @ {self.home.tricode or 'TBD'}"


class NbaWeek(ScheduleModel):
    """The league's own week numbering (``weeks`` in the file); fantasy weeks come from league settings instead."""

    number: int = Field(alias="weekNumber")
    name: str = Field(default="", alias="weekName")
    start: date = Field(alias="startDate")
    end: date = Field(alias="endDate")

    @field_validator("start", "end", mode="before")
    @classmethod
    def _day(cls, value: object) -> object:
        return datetime.fromisoformat(value).date() if isinstance(value, str) else value


def _order(game: NbaGame) -> tuple[date, datetime, str]:
    return (game.day, game.tip_utc if game.tip_utc is not None else datetime.max.replace(tzinfo=UTC), game.game_id)


@dataclass(frozen=True)
class NbaSchedule:
    """The season's games, sorted by day and tip, with the derived views the decision modules need."""

    season_year: str
    league_id: str
    generated_at: datetime | None
    games: tuple[NbaGame, ...]
    weeks: tuple[NbaWeek, ...]
    _second_nights: frozenset[tuple[str, str]] = field(init=False, repr=False, compare=False, default=frozenset())
    """``(game_id, tricode)`` pairs where the team also played the day before (regular season only)."""

    def __post_init__(self) -> None:
        object.__setattr__(self, "games", tuple(sorted(self.games, key=_order)))
        by_team: defaultdict[str, list[NbaGame]] = defaultdict(list)
        for game in self.regular_season():
            for tricode in game.tricodes:
                by_team[tricode].append(game)
        second_nights = {
            (later.game_id, tricode)
            for tricode, games in by_team.items()
            for earlier, later in pairwise(games)
            if (later.day - earlier.day).days == 1
        }
        object.__setattr__(self, "_second_nights", frozenset(second_nights))

    def regular_season(self) -> list[NbaGame]:
        return [game for game in self.games if game.is_regular_season]

    def tricodes(self) -> frozenset[str]:
        """Every team with a regular-season game."""
        return frozenset(code for game in self.regular_season() for code in game.tricodes)

    def games_on(self, day: date, *, regular_season_only: bool = True) -> list[NbaGame]:
        """Games on an ET calendar day, in tip order."""
        return [game for game in self.games if game.day == day and (game.is_regular_season or not regular_season_only)]

    def games_between(self, start: date, end: date, *, regular_season_only: bool = True) -> list[NbaGame]:
        """Games on the ET days ``start`` through ``end`` inclusive."""
        return [
            game
            for game in self.games
            if start <= game.day <= end and (game.is_regular_season or not regular_season_only)
        ]

    def games_per_day(self, start: date, end: date) -> dict[date, int]:
        """Regular-season game count for every day of the range, off days included as 0."""
        counts = {start + timedelta(days=offset): 0 for offset in range((end - start).days + 1)}
        for game in self.games_between(start, end):
            counts[game.day] += 1
        return counts

    def team_games(self, start: date, end: date) -> dict[str, int]:
        """Regular-season games per team (tricode) over the range; every team appears, 0 when it is idle."""
        counts = dict.fromkeys(sorted(self.tricodes()), 0)
        for game in self.games_between(start, end):
            for tricode in game.tricodes:
                counts[tricode] += 1
        return counts

    def team_schedule(self, tricode: str, *, regular_season_only: bool = True) -> list[NbaGame]:
        return [
            game
            for game in self.games
            if tricode in game.tricodes and (game.is_regular_season or not regular_season_only)
        ]

    def is_second_of_back_to_back(self, game: NbaGame, tricode: str) -> bool:
        """True when ``tricode`` plays ``game`` the day after another regular-season game."""
        return (game.game_id, tricode) in self._second_nights

    def back_to_backs(self, tricode: str) -> list[tuple[NbaGame, NbaGame]]:
        """Consecutive-day regular-season game pairs for a team, in season order."""
        games = self.team_schedule(tricode)
        return [(earlier, later) for earlier, later in pairwise(games) if (later.day - earlier.day).days == 1]

    def unscheduled_cup_games(self) -> list[NbaGame]:
        """NBA Cup knockout placeholders whose teams (and so tips) are not set yet."""
        return [game for game in self.games if game.is_cup and game.teams_tbd]

    def week_for(self, day: date) -> NbaWeek | None:
        return next((week for week in self.weeks if week.start <= day <= week.end), None)


def parse_schedule(payload: bytes) -> NbaSchedule:
    """``scheduleLeagueV2.json`` to a :class:`NbaSchedule`. Games that fail validation are skipped and counted; a
    file with no valid game is a schema error."""
    raw = parse_json(payload)
    if not isinstance(raw, dict) or not isinstance(raw.get("leagueSchedule"), dict):
        raise SourceSchemaError("nba_schedule: expected an object with a leagueSchedule object")
    league: dict[str, Any] = raw["leagueSchedule"]
    game_dates = league.get("gameDates")
    if not isinstance(game_dates, list) or not game_dates:
        raise SourceSchemaError("nba_schedule: leagueSchedule.gameDates is missing or empty")
    games: list[NbaGame] = []
    skipped = 0
    total = 0
    for game_date in game_dates:
        entries = game_date.get("games") if isinstance(game_date, dict) else None
        if not isinstance(entries, list):
            raise SourceSchemaError("nba_schedule: a gameDates entry has no games list")
        for entry in entries:
            total += 1
            try:
                games.append(NbaGame.model_validate(entry))
            except ValidationError:
                skipped += 1
    if not games:
        raise SourceSchemaError(f"nba_schedule: none of {total} games validated")
    if skipped:
        logger.warning("nba_schedule: skipped %d of %d games that did not validate", skipped, total)
    weeks: list[NbaWeek] = []
    for entry in league.get("weeks") or []:
        try:
            weeks.append(NbaWeek.model_validate(entry))
        except ValidationError:
            logger.warning("nba_schedule: skipped a week entry that did not validate: %r", entry)
    meta_raw = raw.get("meta")
    meta: dict[str, Any] = meta_raw if isinstance(meta_raw, dict) else {}
    try:
        generated_at = _parse_instant(meta.get("time"))
    except ValueError:
        generated_at = None
    return NbaSchedule(
        season_year=str(league.get("seasonYear", "")),
        league_id=str(league.get("leagueId", "")),
        generated_at=generated_at,
        games=tuple(games),
        weeks=tuple(weeks),
    )


class NbaScheduleSource(HttpSource):
    """The CDN schedule as a :class:`NbaSchedule`; one download a day unless forced."""

    name: ClassVar[str] = "nba_schedule"
    min_interval: ClassVar[float] = 1.0
    base_headers: ClassVar[Mapping[str, str]] = CHROME_HEADERS
    ttl: ClassVar[Mapping[str, timedelta]] = {"schedule": timedelta(hours=24)}

    def schedule(self, **options: Unpack[FetchOptions]) -> Fetched[NbaSchedule]:
        """The full schedule. ``warnings`` notes NBA Cup knockout games whose teams are not set yet."""
        result = self.fetch(
            "schedule",
            "league",
            download=lambda: self.get_bytes(CDN_URL),
            parse=parse_schedule,
            meta={"url": CDN_URL},
            **options,
        )
        pending = result.data.unscheduled_cup_games()
        if pending:
            first, last = pending[0].day.isoformat(), pending[-1].day.isoformat()
            note = (
                f"{len(pending)} NBA Cup knockout games ({first} to {last}) are not scheduled yet; "
                "re-pull after group play"
            )
            result = dataclasses.replace(result, warnings=(*result.warnings, note))
        return result
