"""Game environment, part 1: betting lines as implied team totals, from ESPN's scoreboard and The Odds API (DESIGN 7).

ESPN's public scoreboard (``site.api.espn.com``; no key, but send a browser UA) comes first. Per game it carries the
DraftKings spread and over/under, ESPN's AccuWeather snippet and the venue's ``indoor`` flag. ESPN drops the ``odds``
block once a game kicks off, so a line exists only while ``state == "pre"``. ``spread`` is the home team's spread:
negative when the home team is favored (``details`` names the favorite, ``"NO -1.5"``). :func:`implied_team_totals`
turns (spread, over/under) into a projected score per side::

    home = (over_under - spread) / 2        away = (over_under + spread) / 2

The Odds API (``the-odds-api.com``; optional, needs ``ODDS_API_KEY``) adds the sportsbooks' *posted* team totals, one
event per call. The free tier is 500 credits a month: the ``events`` list is free and a ``team_totals`` call costs one
credit per region, hence the long TTLs. The key travels as a query parameter, so this adapter strips it from error
messages and never records it in cache metadata (httpx still logs request URLs at INFO; keep that logger quieter).

Both adapters degrade instead of raising: an upstream failure yields the last good copy as ``stale`` or an empty
``degraded`` result with the reason in ``warnings``. Team environment is an enrichment and must never block a decision.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, Unpack

import httpx
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError, field_validator, model_validator

from fm.espn.ids import Game
from fm.sources.base import (
    Fetched,
    FetchOptions,
    HttpSource,
    Primitive,
    RateLimiter,
    SourceError,
    SourceSchemaError,
    parse_json,
    utcnow,
)

logger = logging.getLogger(__name__)

BROWSER_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36"
)
"""ESPN's site APIs are meant for the website; DESIGN section 7 says to send a browser UA."""

PREFERRED_PROVIDERS: tuple[str, ...] = ("DraftKings",)
"""Odds providers in order of preference when ESPN lists several; DraftKings is ESPN's partner book."""

SITE_PATHS: Mapping[Game, str] = {Game.FFL: "football/nfl", Game.FBA: "basketball/nba"}
ODDS_API_SPORTS: Mapping[Game, str] = {Game.FFL: "americanfootball_nfl", Game.FBA: "basketball_nba"}
REDACTED = "***"


@dataclass(frozen=True, slots=True)
class TeamTotals:
    """Projected points per side implied by a spread and an over/under."""

    home: float
    away: float

    @property
    def total(self) -> float:
        return self.home + self.away

    @property
    def spread(self) -> float:
        """The home team's spread these totals came from (negative when the home side is favored)."""
        return self.away - self.home


def implied_team_totals(*, spread: float, over_under: float) -> TeamTotals:
    """Split an over/under by the spread. ``spread`` is the home team's (negative = home favored)."""
    if over_under < 0:
        raise ValueError("over_under must be non-negative")
    return TeamTotals(home=(over_under - spread) / 2, away=(over_under + spread) / 2)


def _to_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _is_number(value: object) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int | float):
        return True
    return isinstance(value, str) and value.strip().lstrip("-").replace(".", "", 1).isdigit()


def _opt_int(value: object) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip():
        try:
            return int(float(value))
        except ValueError:
            return None
    return None


def _opt_float(value: object) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str) and value.strip():
        try:
            return float(value)
        except ValueError:
            return None
    return None


class OddsModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)


class ScoreboardTeam(OddsModel):
    """One side of a scoreboard game. ``id`` is ESPN's pro team id (the ``fm.espn.ids`` pro-team maps)."""

    id: int
    abbreviation: str
    display_name: str | None = Field(default=None, alias="displayName")
    """Full name as the sportsbooks write it (``"New Orleans Saints"``)."""
    location: str | None = None
    nickname: str | None = Field(default=None, alias="name")


class ScoreboardVenue(OddsModel):
    id: str | None = None
    name: str | None = Field(default=None, alias="fullName")
    city: str | None = None
    state: str | None = None
    country: str | None = None
    indoor: bool | None = None

    @model_validator(mode="before")
    @classmethod
    def _flatten_address(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        out: dict[str, Any] = dict(data)
        address = out.pop("address", None)
        if isinstance(address, Mapping):
            for key in ("city", "state", "country"):
                out.setdefault(key, address.get(key))
        if out.get("id") is not None:
            out["id"] = str(out["id"])
        return out


class ScoreboardWeather(OddsModel):
    """ESPN's AccuWeather snippet for the event: a description, AccuWeather's condition id, temperatures in F."""

    description: str | None = None
    condition_id: str | None = None
    temperature_f: float | None = Field(default=None, alias="temperature")
    high_temperature_f: float | None = Field(default=None, alias="highTemperature")
    link: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _untangle(cls, data: object) -> object:
        if not isinstance(data, Mapping):
            return data
        out: dict[str, Any] = dict(data)
        display = out.pop("displayValue", out.get("description"))
        condition = out.pop("conditionId", out.get("condition_id"))
        # ESPN serves the pair swapped at times ({"displayValue": "7", "conditionId": "Cloudy"}); the number is the id.
        if _is_number(display) and not _is_number(condition):
            display, condition = condition, display
        out["description"] = display if isinstance(display, str) else None
        out["condition_id"] = None if condition is None else str(condition)
        link = out.get("link")
        if isinstance(link, Mapping):
            out["link"] = link.get("href")
        return out


class GameLine(OddsModel):
    """One provider's pregame line. ``spread`` is the home team's: negative when the home team is favored."""

    provider: str
    details: str | None = None
    """ESPN's summary naming the favorite, e.g. ``"NO -1.5"``."""
    spread: float | None = None
    over_under: float | None = None
    home_favorite: bool | None = None

    @property
    def implied_totals(self) -> TeamTotals | None:
        if self.spread is None or self.over_under is None:
            return None
        return implied_team_totals(spread=self.spread, over_under=self.over_under)


class ScoreboardGame(OddsModel):
    event_id: str
    kickoff: datetime
    name: str
    short_name: str
    season: int | None = None
    season_type: int | None = None
    """ESPN season type: 1 preseason, 2 regular season, 3 postseason."""
    week: int | None = None
    state: str
    """``pre`` (scheduled), ``in`` (live) or ``post`` (final), from ``status.type.state``; ``unknown`` when absent."""
    completed: bool = False
    neutral_site: bool = False
    home: ScoreboardTeam
    away: ScoreboardTeam
    home_score: int | None = None
    away_score: int | None = None
    venue: ScoreboardVenue | None = None
    line: GameLine | None = None
    """Absent once the game has started (ESPN drops the odds block) or when no book has posted a line."""
    weather: ScoreboardWeather | None = None

    @field_validator("kickoff")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return _to_utc(value)

    @property
    def indoor(self) -> bool | None:
        """ESPN's venue flag; ``None`` when the scoreboard carries no venue."""
        return self.venue.indoor if self.venue is not None else None

    @property
    def implied_totals(self) -> TeamTotals | None:
        return self.line.implied_totals if self.line is not None else None

    def has_team(self, team: str) -> bool:
        return team in (self.home.abbreviation, self.away.abbreviation)

    def implied_total(self, team: str) -> float | None:
        """Implied points for ``team`` (ESPN abbreviation); ``None`` when no line is posted."""
        totals = self.implied_totals
        if totals is None:
            return None
        if team == self.home.abbreviation:
            return totals.home
        if team == self.away.abbreviation:
            return totals.away
        raise ValueError(f"{team} is not playing in {self.short_name}")


class Scoreboard(OddsModel):
    """One scoreboard payload: the slate ESPN answered with, games in kickoff order."""

    season: int | None = None
    season_type: int | None = None
    week: int | None = None
    games: list[ScoreboardGame] = Field(default_factory=list)

    def game(self, event_id: str) -> ScoreboardGame | None:
        return next((game for game in self.games if game.event_id == event_id), None)

    def for_team(self, team: str) -> ScoreboardGame | None:
        return next((game for game in self.games if game.has_team(team)), None)


def pick_line(entries: Sequence[object], *, preferred: Sequence[str] = PREFERRED_PROVIDERS) -> GameLine | None:
    """The preferred provider's line, else the one ESPN ranks first (lowest ``priority``); ``None`` when unposted."""
    candidates = [entry for entry in entries if isinstance(entry, Mapping)]
    if not candidates:
        return None

    def rank(entry: Mapping[str, Any]) -> tuple[int, int]:
        provider = _mapping(entry.get("provider"))
        name = provider.get("name")
        priority = provider.get("priority")
        preference = preferred.index(name) if isinstance(name, str) and name in preferred else len(preferred)
        return preference, priority if isinstance(priority, int) else 1_000

    chosen = min(candidates, key=rank)
    provider_name = _mapping(chosen.get("provider")).get("name")
    details = chosen.get("details")
    favorite = _mapping(chosen.get("homeTeamOdds")).get("favorite")
    return GameLine(
        provider=provider_name if isinstance(provider_name, str) and provider_name else "unknown",
        details=details if isinstance(details, str) else None,
        spread=_opt_float(chosen.get("spread")),
        over_under=_opt_float(chosen.get("overUnder")),
        home_favorite=favorite if isinstance(favorite, bool) else None,
    )


def _parse_event(event: Mapping[str, Any], preferred: Sequence[str]) -> ScoreboardGame:
    competitions = event.get("competitions")
    if not isinstance(competitions, list) or not competitions or not isinstance(competitions[0], Mapping):
        raise ValueError("event has no competition")
    competition: Mapping[str, Any] = competitions[0]
    competitors = competition.get("competitors")
    if not isinstance(competitors, list):
        raise ValueError("competition has no competitors")
    sides = {side.get("homeAway"): side for side in competitors if isinstance(side, Mapping)}
    home, away = sides.get("home"), sides.get("away")
    if home is None or away is None:
        raise ValueError("competition is missing its home or away side")
    status_type = _mapping(_mapping(event.get("status") or competition.get("status")).get("type"))
    season = _mapping(event.get("season"))
    week = _mapping(event.get("week"))
    odds = competition.get("odds")
    venue = competition.get("venue")
    weather = event.get("weather")
    return ScoreboardGame.model_validate(
        {
            "event_id": str(event["id"]),
            "kickoff": competition.get("date") or event["date"],
            "name": str(event.get("name") or ""),
            "short_name": str(event.get("shortName") or ""),
            "season": _opt_int(season.get("year")),
            "season_type": _opt_int(season.get("type")),
            "week": _opt_int(week.get("number")),
            "state": str(status_type.get("state") or "unknown"),
            "completed": bool(status_type.get("completed", False)),
            "neutral_site": bool(competition.get("neutralSite", False)),
            "home": ScoreboardTeam.model_validate(home["team"]),
            "away": ScoreboardTeam.model_validate(away["team"]),
            "home_score": _opt_int(home.get("score")),
            "away_score": _opt_int(away.get("score")),
            "venue": ScoreboardVenue.model_validate(venue) if isinstance(venue, Mapping) else None,
            "line": pick_line(odds, preferred=preferred) if isinstance(odds, list) else None,
            "weather": ScoreboardWeather.model_validate(weather) if isinstance(weather, Mapping) else None,
        }
    )


def parse_scoreboard(payload: bytes, *, preferred: Sequence[str] = PREFERRED_PROVIDERS) -> Scoreboard:
    """Games from a scoreboard payload, in kickoff order. An empty ``events`` list is a quiet day; a payload without
    one is a schema error, as is a non-empty list in which no event parses. Bad events are skipped and counted."""
    raw = parse_json(payload)
    if not isinstance(raw, Mapping):
        raise SourceSchemaError(f"espn scoreboard: expected a JSON object, got {type(raw).__name__}")
    events = raw.get("events")
    if not isinstance(events, list):
        raise SourceSchemaError("espn scoreboard: expected an 'events' list")
    games: list[ScoreboardGame] = []
    skipped = 0
    for event in events:
        if not isinstance(event, Mapping):
            skipped += 1
            continue
        try:
            games.append(_parse_event(event, preferred))
        except (KeyError, TypeError, ValueError, ValidationError) as exc:
            skipped += 1
            logger.warning("espn scoreboard: skipped event %s (%s)", event.get("id"), exc)
    if events and not games:
        raise SourceSchemaError(f"espn scoreboard: none of {len(events)} events parsed")
    if skipped:
        logger.warning("espn scoreboard: skipped %d of %d events that did not parse", skipped, len(events))
    season = _mapping(raw.get("season"))
    week = _mapping(raw.get("week"))
    return Scoreboard(
        season=_opt_int(season.get("year")),
        season_type=_opt_int(season.get("type")),
        week=_opt_int(week.get("number")),
        games=sorted(games, key=lambda game: (game.kickoff, game.event_id)),
    )


class EspnScoreboardSource(HttpSource):
    """ESPN's public scoreboard: lines, the AccuWeather snippet and the indoor flag per game. Degrades, never raises."""

    name: ClassVar[str] = "espn_scoreboard"
    SITE_API: ClassVar[str] = "https://site.api.espn.com/apis/site/v2/sports"
    base_headers: ClassVar[Mapping[str, str]] = {"User-Agent": BROWSER_USER_AGENT, "Accept": "application/json"}
    min_interval: ClassVar[float] = 0.5
    preferred_providers: ClassVar[tuple[str, ...]] = PREFERRED_PROVIDERS
    ttl: ClassVar[Mapping[str, timedelta]] = {"scoreboard": timedelta(minutes=10)}  # lines move, scores update live

    def scoreboard(
        self,
        game: Game | str = Game.FFL,
        *,
        season: int | None = None,
        week: int | None = None,
        season_type: int | None = None,
        day: date | None = None,
        **options: Unpack[FetchOptions],
    ) -> Fetched[Scoreboard]:
        """The slate for one NFL ``week`` (of ``season``; regular season unless ``season_type`` says otherwise), one
        calendar ``day`` (the NBA shape), or ESPN's current slate when neither is given."""
        game = Game.coerce(game)
        url = f"{self.SITE_API}/{SITE_PATHS[game]}/scoreboard"
        params: dict[str, Primitive] = {}
        if day is not None:
            params["dates"] = day.strftime("%Y%m%d")
            key = f"{game.sport}_{params['dates']}"
        elif week is not None:
            params["week"] = week
            params["seasontype"] = season_type if season_type is not None else 2
            if season is not None:
                params["dates"] = season
            key = f"{game.sport}_{season if season is not None else 'current'}_t{params['seasontype']}_w{week}"
        else:
            key = f"{game.sport}_current"
        try:
            return self.fetch(
                "scoreboard",
                key,
                download=lambda: self.get_bytes(url, params=params),
                parse=lambda payload: parse_scoreboard(payload, preferred=self.preferred_providers),
                meta={"url": url, **params},
                **options,
            )
        except SourceError as exc:
            logger.warning("%s/scoreboard[%s]: unavailable (%s); continuing without lines", self.name, key, exc)
            return Fetched(
                Scoreboard(), self.clock(), self.name, "scoreboard", key, degraded=True, warnings=(str(exc),)
            )


class OddsEvent(OddsModel):
    """One upcoming game as The Odds API lists it (``/events`` costs no credits)."""

    id: str
    sport_key: str
    commence_time: datetime
    home_team: str
    away_team: str

    @field_validator("commence_time")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return _to_utc(value)

    def is_game(self, home_name: str, away_name: str) -> bool:
        """Match against a scoreboard game's ``display_name`` pair."""
        return self.home_team == home_name and self.away_team == away_name


class PostedTeamTotal(OddsModel):
    """A sportsbook's team total market: the posted line plus the over and under prices (American odds)."""

    event_id: str
    bookmaker: str
    team: str
    """Team name as the book writes it (``"New Orleans Saints"``), matching ``ScoreboardTeam.display_name``."""
    point: float
    over_price: int | None = None
    under_price: int | None = None
    last_update: datetime | None = None

    @field_validator("last_update")
    @classmethod
    def _utc(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _to_utc(value)


def parse_odds_events(payload: bytes) -> list[OddsEvent]:
    raw = parse_json(payload)
    if not isinstance(raw, list):
        raise SourceSchemaError(f"odds api events: expected a JSON list, got {type(raw).__name__}")
    try:
        return [OddsEvent.model_validate(item) for item in raw]
    except ValidationError as exc:
        raise SourceSchemaError(f"odds api events: {exc}") from exc


def parse_team_totals(payload: bytes) -> list[PostedTeamTotal]:
    """Team totals from a per-event odds payload: one entry per (bookmaker, team) at the book's main line."""
    raw = parse_json(payload)
    if not isinstance(raw, Mapping):
        raise SourceSchemaError(f"odds api event: expected a JSON object, got {type(raw).__name__}")
    event_id = raw.get("id")
    bookmakers = raw.get("bookmakers")
    if not isinstance(event_id, str) or not isinstance(bookmakers, list):
        raise SourceSchemaError("odds api event: expected an 'id' and a 'bookmakers' list")
    totals: dict[tuple[str, str], dict[str, Any]] = {}
    for book in bookmakers:
        if not isinstance(book, Mapping):
            continue
        bookmaker = str(book.get("key") or book.get("title") or "unknown")
        for market in book.get("markets") or []:
            if not isinstance(market, Mapping) or market.get("key") != "team_totals":
                continue
            for outcome in market.get("outcomes") or []:
                if not isinstance(outcome, Mapping):
                    continue
                team, side, point = outcome.get("description"), outcome.get("name"), _opt_float(outcome.get("point"))
                if not isinstance(team, str) or point is None or side not in ("Over", "Under"):
                    continue
                entry = totals.setdefault(
                    (bookmaker, team),
                    {
                        "event_id": event_id,
                        "bookmaker": bookmaker,
                        "team": team,
                        "point": point,
                        "last_update": market.get("last_update") or book.get("last_update"),
                    },
                )
                if entry["point"] != point:
                    continue  # an alternate line; the first point seen is the main one
                entry["over_price" if side == "Over" else "under_price"] = _opt_int(outcome.get("price"))
    try:
        return [PostedTeamTotal.model_validate(entry) for entry in totals.values()]
    except ValidationError as exc:
        raise SourceSchemaError(f"odds api event: {exc}") from exc


class OddsApiSource(HttpSource):
    """The Odds API v4, optional. Without a key every method returns an empty ``degraded`` result and sends nothing."""

    name: ClassVar[str] = "odds_api"
    BASE_URL: ClassVar[str] = "https://api.the-odds-api.com/v4"
    min_interval: ClassVar[float] = 1.0
    ttl: ClassVar[Mapping[str, timedelta]] = {
        "events": timedelta(hours=6),  # free, and the slate rarely changes
        "team_totals": timedelta(hours=6),  # one credit per call out of 500 a month
    }

    def __init__(
        self,
        api_key: SecretStr | str | None = None,
        *,
        regions: str = "us",
        client: httpx.Client | None = None,
        sleep: Callable[[float], object] = time.sleep,
        cache_root: Path | None = None,
        limiter: RateLimiter | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        super().__init__(client=client, sleep=sleep, cache_root=cache_root, limiter=limiter, clock=clock)
        secret = api_key if isinstance(api_key, SecretStr) or api_key is None else SecretStr(api_key)
        self._api_key = secret if secret is not None and secret.get_secret_value().strip() else None
        self.regions = regions

    @property
    def configured(self) -> bool:
        return self._api_key is not None

    def events(self, game: Game | str = Game.FFL, **options: Unpack[FetchOptions]) -> Fetched[list[OddsEvent]]:
        """Upcoming events with the names the books use; free (no credits)."""
        game = Game.coerce(game)
        path = f"/sports/{ODDS_API_SPORTS[game]}/events"
        return self._fetch("events", game.sport, path, {}, parse_odds_events, options)

    def team_totals(
        self, event_id: str, game: Game | str = Game.FFL, **options: Unpack[FetchOptions]
    ) -> Fetched[list[PostedTeamTotal]]:
        """Posted team totals for one event from every book in ``regions``; costs one credit per region."""
        game = Game.coerce(game)
        path = f"/sports/{ODDS_API_SPORTS[game]}/events/{event_id}/odds"
        params: dict[str, Primitive] = {"regions": self.regions, "markets": "team_totals", "oddsFormat": "american"}
        return self._fetch("team_totals", f"{game.sport}_{event_id}", path, params, parse_team_totals, options)

    def _fetch[T](
        self,
        dataset: str,
        key: str,
        path: str,
        params: Mapping[str, Primitive],
        parse: Callable[[bytes], list[T]],
        options: FetchOptions,
    ) -> Fetched[list[T]]:
        url = f"{self.BASE_URL}{path}"
        empty: list[T] = []
        if self._api_key is None:
            warning = "ODDS_API_KEY is not configured; posted team totals skipped"
            return Fetched(empty, self.clock(), self.name, dataset, key, degraded=True, warnings=(warning,))
        try:
            return self.fetch(
                dataset,
                key,
                download=lambda: self._get(url, params),
                parse=parse,
                meta={"url": url, **params},
                **options,
            )
        except SourceError as exc:
            reason = self._redact(str(exc))
            logger.warning("%s/%s[%s]: unavailable (%s); continuing without it", self.name, dataset, key, reason)
            return Fetched(empty, self.clock(), self.name, dataset, key, degraded=True, warnings=(reason,))

    def _get(self, url: str, params: Mapping[str, Primitive]) -> bytes:
        if self._api_key is None:
            raise SourceError(f"{self.name}: no API key")
        try:
            return self.get_bytes(url, params={**params, "apiKey": self._api_key.get_secret_value()})
        except SourceError as exc:
            # The failing URL carries the key; re-raise without it and without the chained original.
            raise SourceError(self._redact(str(exc))) from None

    def _redact(self, text: str) -> str:
        if self._api_key is None:
            return text
        return text.replace(self._api_key.get_secret_value(), REDACTED)
