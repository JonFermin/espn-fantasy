"""Game environment, part 2: stadium coordinates and Open-Meteo forecasts for outdoor games (DESIGN section 7).

``data/stadiums.csv`` lists every NFL venue: home team (ESPN abbreviation), venue name, roof (``outdoors``, ``dome`` or
``retractable``) and coordinates, in the one-row-per-venue spirit of ``greerreNFL/stadiums``. Shared venues (MetLife,
SoFi) appear once per home team; neutral sites (London, Munich, ...) carry no team and match by venue name only.
:meth:`Stadiums.for_game` picks a game's venue: the scoreboard's venue name when it is known (which also catches a
relocated game), else the home team's stadium (ESPN's venue names drift: ``"Reliant Stadium"`` for NRG), else nothing
for an unknown neutral site, in which case there is no forecast to fetch.

Open-Meteo (``api.open-meteo.com``; no key; free for non-commercial use, 10k calls a day) serves hourly forecasts at
the coordinates, requested in F, mph and inches on UTC hours. :meth:`OpenMeteoSource.game_weather` aggregates the hours
from kickoff into a :class:`GameWeather`. Domes and retractable roofs skip the request (``covered=True``); ESPN's
``indoor`` flag, when the caller passes it, overrides the CSV either way. Failures degrade to an empty forecast with the
reason in ``warnings``; nothing here raises for an upstream problem.

Typical use with a scoreboard game::

    stadium = stadiums.for_game(game.home.abbreviation, venue=game.venue.name, neutral_site=game.neutral_site)
    if stadium is not None:
        weather = meteo.game_weather(stadium, game.kickoff, indoor=game.indoor)
"""

from __future__ import annotations

import csv
import logging
import re
import unicodedata
from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar, Literal, Unpack

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from fm import paths
from fm.sources.base import Fetched, FetchOptions, HttpSource, Primitive, SourceError, SourceSchemaError, parse_json

logger = logging.getLogger(__name__)

type Roof = Literal["outdoors", "dome", "retractable"]

STADIUMS_CSV: Path = paths.data_file("stadiums.csv")
"""The repo's ``data/stadiums.csv``; ``data/`` sits beside ``src/`` (DESIGN section 14), outside the package."""

COVERED_ROOFS: frozenset[str] = frozenset({"dome", "retractable"})
"""Roofs that take weather out of the game. A retractable roof closes in bad weather, so it counts as covered."""

FIELD_FOR_VARIABLE: Mapping[str, str] = {
    "temperature_2m": "temperature_f",
    "apparent_temperature": "apparent_temperature_f",
    "precipitation_probability": "precipitation_probability",
    "precipitation": "precipitation_in",
    "rain": "rain_in",
    "snowfall": "snowfall_in",
    "wind_speed_10m": "wind_mph",
    "wind_gusts_10m": "gust_mph",
    "weather_code": "weather_code",
}
"""Open-Meteo hourly variable -> :class:`ForecastHour` field."""

HOURLY_VARIABLES: tuple[str, ...] = tuple(FIELD_FOR_VARIABLE)
EXPECTED_UNITS: Mapping[str, str] = {"temperature_2m": "°F", "wind_speed_10m": "mp/h", "precipitation": "inch"}
"""What ``hourly_units`` must report for the units we ask for; anything else is a schema error, not a silent slip."""

DEFAULT_WINDOW_HOURS = 3
MAX_FORECAST_DAYS = 16


def _to_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def floor_hour(when: datetime) -> datetime:
    """The UTC hour containing ``when``; a naive datetime is taken as UTC."""
    return _to_utc(when).replace(minute=0, second=0, microsecond=0)


def normalize_venue(name: str) -> str:
    """Lowercase ASCII letters and digits only, so ``"Levi's Stadium"`` and ``"Santiago Bernabéu"`` match loosely."""
    ascii_name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", ascii_name.lower())


class WeatherModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)


class Stadium(WeatherModel):
    """One row of ``data/stadiums.csv``."""

    team: str | None = None
    """Home team (ESPN abbreviation); ``None`` for a neutral site."""
    venue: str
    city: str
    state: str | None = None
    country: str = "USA"
    roof: Roof
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)

    @field_validator("team", "state", mode="before")
    @classmethod
    def _blank_is_none(cls, value: object) -> object:
        if isinstance(value, str):
            value = value.strip()
            return value or None
        return value

    @property
    def covered(self) -> bool:
        return self.roof in COVERED_ROOFS


class Stadiums:
    """The stadium table with lookups by home team and by venue name."""

    def __init__(self, stadiums: Iterable[Stadium]) -> None:
        self._rows: tuple[Stadium, ...] = tuple(stadiums)
        self._by_team: dict[str, Stadium] = {row.team: row for row in self._rows if row.team is not None}
        self._by_venue: dict[str, list[Stadium]] = {}
        for row in self._rows:
            self._by_venue.setdefault(normalize_venue(row.venue), []).append(row)

    @classmethod
    def load(cls, path: Path = STADIUMS_CSV) -> Stadiums:
        """Read the CSV; a bad row raises ``ValueError`` naming the file and line."""
        rows: list[Stadium] = []
        with path.open(encoding="utf-8", newline="") as handle:
            for line_number, record in enumerate(csv.DictReader(handle), start=2):
                try:
                    rows.append(Stadium.model_validate(record))
                except ValidationError as exc:
                    raise ValueError(f"{path}:{line_number}: {exc}") from exc
        if not rows:
            raise ValueError(f"{path}: no stadiums")
        return cls(rows)

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterator[Stadium]:
        return iter(self._rows)

    @property
    def teams(self) -> frozenset[str]:
        return frozenset(self._by_team)

    def for_team(self, team: str) -> Stadium | None:
        """The home stadium of ``team`` (ESPN abbreviation)."""
        return self._by_team.get(team)

    def by_venue(self, name: str, *, prefer_team: str | None = None) -> Stadium | None:
        """The venue called ``name`` (loosely matched); for a shared venue, ``prefer_team``'s row when it has one."""
        matches = self._by_venue.get(normalize_venue(name), [])
        if not matches:
            return None
        if prefer_team is not None:
            for match in matches:
                if match.team == prefer_team:
                    return match
        return matches[0]

    def for_game(self, home_team: str, *, venue: str | None = None, neutral_site: bool = False) -> Stadium | None:
        """Where a game is played: the named venue when known, else the home stadium unless the site is neutral."""
        if venue:
            match = self.by_venue(venue, prefer_team=home_team)
            if match is not None:
                return match
        if not neutral_site:
            return self.for_team(home_team)
        return None


def load_stadiums(path: Path = STADIUMS_CSV) -> Stadiums:
    return Stadiums.load(path)


class ForecastHour(WeatherModel):
    """One forecast hour in US units (F, mph, inches); ``weather_code`` is the WMO code (0 clear, 99 thunderstorm)."""

    time: datetime
    temperature_f: float | None = None
    apparent_temperature_f: float | None = None
    precipitation_probability: float | None = None
    """Percent."""
    precipitation_in: float | None = None
    rain_in: float | None = None
    snowfall_in: float | None = None
    wind_mph: float | None = None
    gust_mph: float | None = None
    weather_code: int | None = None

    @field_validator("time")
    @classmethod
    def _utc(cls, value: datetime) -> datetime:
        return _to_utc(value)


class HourlyForecast(WeatherModel):
    latitude: float
    longitude: float
    elevation_m: float | None = None
    hours: list[ForecastHour] = Field(default_factory=list)

    @classmethod
    def empty(cls, latitude: float, longitude: float) -> HourlyForecast:
        return cls(latitude=latitude, longitude=longitude)

    @property
    def start(self) -> datetime | None:
        return self.hours[0].time if self.hours else None

    @property
    def end(self) -> datetime | None:
        """The first hour not covered."""
        return self.hours[-1].time + timedelta(hours=1) if self.hours else None

    def at(self, when: datetime) -> ForecastHour | None:
        """The hour containing ``when``."""
        target = floor_hour(when)
        return next((hour for hour in self.hours if hour.time == target), None)

    def window(self, start: datetime, hours: int) -> list[ForecastHour]:
        """The ``hours`` consecutive forecast hours from the hour containing ``start``, as far as the horizon goes."""
        first = floor_hour(start)
        last = first + timedelta(hours=hours)
        return [hour for hour in self.hours if first <= hour.time < last]


def parse_forecast(payload: bytes) -> HourlyForecast:
    """An Open-Meteo forecast response. The ``hourly`` arrays must be parallel to ``hourly.time`` and the units must be
    the ones we asked for; a missing variable is tolerated as nulls."""
    raw = parse_json(payload)
    if not isinstance(raw, Mapping):
        raise SourceSchemaError(f"open-meteo: expected a JSON object, got {type(raw).__name__}")
    hourly = raw.get("hourly")
    if not isinstance(hourly, Mapping) or not isinstance(hourly.get("time"), list):
        raise SourceSchemaError("open-meteo: expected 'hourly' with a 'time' list")
    times: list[Any] = hourly["time"]
    units = raw.get("hourly_units")
    if isinstance(units, Mapping):
        for variable, expected in EXPECTED_UNITS.items():
            unit = units.get(variable)
            if unit is not None and unit != expected:
                raise SourceSchemaError(f"open-meteo: {variable} is in {unit!r}, expected {expected!r}")
    columns: dict[str, list[Any]] = {}
    for variable in HOURLY_VARIABLES:
        values = hourly.get(variable)
        if values is None:
            values = [None] * len(times)
        if not isinstance(values, list) or len(values) != len(times):
            raise SourceSchemaError(f"open-meteo: {variable} does not line up with 'time' ({len(times)} entries)")
        columns[FIELD_FOR_VARIABLE[variable]] = values
    try:
        hours = [
            ForecastHour.model_validate({"time": time, **{field: column[index] for field, column in columns.items()}})
            for index, time in enumerate(times)
        ]
        return HourlyForecast.model_validate(
            {
                "latitude": raw.get("latitude"),
                "longitude": raw.get("longitude"),
                "elevation_m": raw.get("elevation"),
                "hours": hours,
            }
        )
    except ValidationError as exc:
        raise SourceSchemaError(f"open-meteo: {exc}") from exc


class GameWeather(WeatherModel):
    """The forecast for the hours from kickoff at one venue.

    A ``covered`` game carries no hours (nothing was fetched), and neither does a kickoff beyond the forecast horizon;
    ``available`` tells a real forecast from both. Aggregates: mean temperatures, peak wind, gust and precipitation
    probability, total precipitation and snowfall, and the highest (roughly: worst) WMO weather code in the window."""

    venue: str
    roof: Roof
    covered: bool
    kickoff: datetime
    latitude: float
    longitude: float
    hours: list[ForecastHour] = Field(default_factory=list)
    temperature_f: float | None = None
    apparent_temperature_f: float | None = None
    wind_mph: float | None = None
    gust_mph: float | None = None
    precipitation_probability: float | None = None
    precipitation_in: float | None = None
    snowfall_in: float | None = None
    weather_code: int | None = None

    @property
    def available(self) -> bool:
        return bool(self.hours)


def _mean(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _max[T: (int, float)](values: Iterable[T | None]) -> T | None:
    present = [value for value in values if value is not None]
    return max(present) if present else None


def _total(values: Iterable[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return sum(present) if present else None


def covered_game(stadium: Stadium, kickoff: datetime) -> GameWeather:
    """Weather is moot under a roof: no hours, no aggregates."""
    return GameWeather(
        venue=stadium.venue,
        roof=stadium.roof,
        covered=True,
        kickoff=_to_utc(kickoff),
        latitude=stadium.latitude,
        longitude=stadium.longitude,
    )


def weather_at_kickoff(
    stadium: Stadium, kickoff: datetime, forecast: HourlyForecast, *, hours: int = DEFAULT_WINDOW_HOURS
) -> GameWeather:
    """Aggregate ``hours`` forecast hours from kickoff; empty (``available=False``) outside the forecast horizon."""
    if hours < 1:
        raise ValueError("hours must be >= 1")
    window = forecast.window(kickoff, hours)
    return GameWeather(
        venue=stadium.venue,
        roof=stadium.roof,
        covered=False,
        kickoff=_to_utc(kickoff),
        latitude=stadium.latitude,
        longitude=stadium.longitude,
        hours=window,
        temperature_f=_mean(hour.temperature_f for hour in window),
        apparent_temperature_f=_mean(hour.apparent_temperature_f for hour in window),
        wind_mph=_max(hour.wind_mph for hour in window),
        gust_mph=_max(hour.gust_mph for hour in window),
        precipitation_probability=_max(hour.precipitation_probability for hour in window),
        precipitation_in=_total(hour.precipitation_in for hour in window),
        snowfall_in=_total(hour.snowfall_in for hour in window),
        weather_code=_max(hour.weather_code for hour in window),
    )


class OpenMeteoSource(HttpSource):
    """Open-Meteo hourly forecasts, cached per 0.01 degree cell. Degrades, never raises."""

    name: ClassVar[str] = "open_meteo"
    FORECAST_URL: ClassVar[str] = "https://api.open-meteo.com/v1/forecast"
    min_interval: ClassVar[float] = 0.2
    ttl: ClassVar[Mapping[str, timedelta]] = {"forecast": timedelta(hours=1)}  # the model output updates hourly

    @staticmethod
    def cache_key(latitude: float, longitude: float, forecast_days: int) -> str:
        return f"{latitude:.2f}_{longitude:.2f}_{forecast_days}d"

    def forecast(
        self, latitude: float, longitude: float, *, forecast_days: int = 7, **options: Unpack[FetchOptions]
    ) -> Fetched[HourlyForecast]:
        """Hourly forecast from today's 00:00 UTC for ``forecast_days`` days (1 to 16; the NFL week fits in 7)."""
        if not 1 <= forecast_days <= MAX_FORECAST_DAYS:
            raise ValueError(f"forecast_days must be between 1 and {MAX_FORECAST_DAYS}")
        params: dict[str, Primitive] = {
            "latitude": round(latitude, 4),
            "longitude": round(longitude, 4),
            "hourly": ",".join(HOURLY_VARIABLES),
            "temperature_unit": "fahrenheit",
            "wind_speed_unit": "mph",
            "precipitation_unit": "inch",
            "timezone": "UTC",
            "forecast_days": forecast_days,
        }
        key = self.cache_key(latitude, longitude, forecast_days)
        try:
            return self.fetch(
                "forecast",
                key,
                download=lambda: self.get_bytes(self.FORECAST_URL, params=params),
                parse=parse_forecast,
                meta={"url": self.FORECAST_URL, **params},
                **options,
            )
        except SourceError as exc:
            logger.warning("%s/forecast[%s]: unavailable (%s); continuing without weather", self.name, key, exc)
            empty = HourlyForecast.empty(latitude, longitude)
            return Fetched(empty, self.clock(), self.name, "forecast", key, degraded=True, warnings=(str(exc),))

    def game_weather(
        self,
        stadium: Stadium,
        kickoff: datetime,
        *,
        indoor: bool | None = None,
        hours: int = DEFAULT_WINDOW_HOURS,
        forecast_days: int = 7,
        **options: Unpack[FetchOptions],
    ) -> Fetched[GameWeather]:
        """The kickoff window's weather at ``stadium``. Covered venues (``indoor`` when given, else the table's roof)
        return ``covered=True`` without a request."""
        kickoff = _to_utc(kickoff)
        key = self.cache_key(stadium.latitude, stadium.longitude, forecast_days)
        covered = indoor if indoor is not None else stadium.covered
        if covered:
            return Fetched(covered_game(stadium, kickoff), self.clock(), self.name, "forecast", key)
        forecast = self.forecast(stadium.latitude, stadium.longitude, forecast_days=forecast_days, **options)
        weather = weather_at_kickoff(stadium, kickoff, forecast.data, hours=hours)
        warnings = forecast.warnings
        if not weather.available and not forecast.degraded:
            horizon = f"{forecast.data.start} to {forecast.data.end}" if forecast.data.hours else "no hours"
            warnings = (
                *warnings,
                f"{stadium.venue}: kickoff {kickoff.isoformat()} is outside the forecast horizon ({horizon})",
            )
        return Fetched(
            weather,
            forecast.as_of,
            forecast.source,
            forecast.dataset,
            forecast.key,
            cached=forecast.cached,
            stale=forecast.stale,
            degraded=forecast.degraded,
            warnings=warnings,
            raw_path=forecast.raw_path,
        )
