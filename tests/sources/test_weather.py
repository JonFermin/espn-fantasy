"""Stadium table and Open-Meteo adapter against a recorded forecast (respx; no network).

``open_meteo_hourly_metlife.json`` is a two-day hourly forecast recorded on 2026-10-04 for MetLife Stadium's
coordinates in F, mph and inches with ``timezone=UTC`` (which Open-Meteo echoes as ``GMT``).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx

from fm.espn.ids import FFL_PRO_TEAMS
from fm.sources.base import RateLimiter, SourceSchemaError
from fm.sources.weather import (
    HOURLY_VARIABLES,
    STADIUMS_CSV,
    OpenMeteoSource,
    Stadium,
    Stadiums,
    floor_hour,
    load_stadiums,
    normalize_venue,
    parse_forecast,
    weather_at_kickoff,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "sources" / "weather"
FORECAST = "open_meteo_hourly_metlife.json"
T0 = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)
METEO = "api.open-meteo.com"
SUNDAY_EARLY = datetime(2026, 10, 5, 17, 0, tzinfo=UTC)  # inside the recorded 48 hours


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def ok(name: str) -> httpx.Response:
    return httpx.Response(200, content=fixture(name), headers={"content-type": "application/json; charset=utf-8"})


class FakeClock:
    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> datetime:
        self.now += timedelta(**delta)
        return self.now


class Routes:
    def __init__(self, router: respx.MockRouter) -> None:
        self.forecast = router.get(host=METEO, path="/v1/forecast").mock(return_value=ok(FORECAST))


@pytest.fixture
def routes() -> Iterator[Routes]:
    with respx.mock(assert_all_called=False) as router:
        yield Routes(router)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def sleeps() -> list[float]:
    return []


@pytest.fixture
def source(routes: Routes, clock: FakeClock, tmp_path: Path, sleeps: list[float]) -> Iterator[OpenMeteoSource]:
    with OpenMeteoSource(
        cache_root=tmp_path / "cache", limiter=RateLimiter(0), clock=clock, sleep=sleeps.append
    ) as src:
        yield src


@pytest.fixture(scope="module")
def stadiums() -> Stadiums:
    return load_stadiums()


def stadium_of(stadiums: Stadiums, team: str) -> Stadium:
    stadium = stadiums.for_team(team)
    assert stadium is not None, team
    return stadium


# --- data/stadiums.csv ---


def test_every_nfl_team_has_a_stadium_in_the_repo_table(stadiums: Stadiums) -> None:
    assert STADIUMS_CSV.is_file() and (STADIUMS_CSV.parents[1] / "pyproject.toml").is_file()
    assert stadiums.teams == frozenset(FFL_PRO_TEAMS.values()) - {"FA"}
    assert len(stadiums.teams) == 32 and len(stadiums) > 32  # plus neutral sites
    neutral = [stadium for stadium in stadiums if stadium.team is None]
    assert neutral and all(stadium.country != "USA" and stadium.state is None for stadium in neutral)
    assert all(stadium.roof in ("outdoors", "dome", "retractable") for stadium in stadiums)
    assert {stadium.team for stadium in stadiums if stadium.roof == "dome"} == {"DET", "LV", "LAC", "LAR", "MIN", "NO"}
    assert {stadium.team for stadium in stadiums if stadium.roof == "retractable" and stadium.team} == {
        "ARI",
        "ATL",
        "DAL",
        "HOU",
        "IND",
    }


def test_shared_venues_and_covered_roofs(stadiums: Stadiums) -> None:
    giants, jets = stadium_of(stadiums, "NYG"), stadium_of(stadiums, "NYJ")
    assert giants.venue == jets.venue == "MetLife Stadium"
    assert (giants.latitude, giants.longitude) == (jets.latitude, jets.longitude) == (40.8135, -74.0745)
    assert giants.covered is False and giants.roof == "outdoors"
    assert stadium_of(stadiums, "NO").covered is True and stadium_of(stadiums, "NO").roof == "dome"
    assert stadium_of(stadiums, "DAL").covered is True and stadium_of(stadiums, "DAL").roof == "retractable"
    assert stadiums.for_team("XYZ") is None and stadiums.by_venue("Nowhere Field") is None


def test_for_game_prefers_the_named_venue_then_the_home_team(stadiums: Stadiums) -> None:
    # ESPN still calls NRG Stadium "Reliant Stadium": an unknown name on a home game falls back to the home team.
    assert stadium_of(stadiums, "HOU") == stadiums.for_game("HOU", venue="Reliant Stadium")
    # A shared venue resolves to the home side's row.
    jets_home = stadiums.for_game("NYJ", venue="MetLife Stadium")
    assert jets_home is not None and jets_home.team == "NYJ"
    # A known neutral site matches by name; an unknown one yields nothing (so weather is skipped, not guessed).
    london = stadiums.for_game("WSH", venue="Tottenham Hotspur Stadium", neutral_site=True)
    assert london is not None and (london.team, london.city, london.country) == (None, "London", "England")
    assert stadiums.for_game("WSH", venue="Some Unknown Arena", neutral_site=True) is None
    assert stadiums.for_game("WSH", venue=None, neutral_site=True) is None
    # No venue at all: the home team's stadium. A relocated game: the named venue wins over the home team.
    assert stadiums.for_game("GB") == stadium_of(stadiums, "GB")
    relocated = stadiums.for_game("KC", venue="Levi's Stadium")
    assert relocated is not None and relocated.team == "SF"


def test_venue_names_match_loosely() -> None:
    assert normalize_venue("Santiago Bernabéu") == normalize_venue("santiago bernabeu") == "santiagobernabeu"
    assert (
        normalize_venue("U.S. Bank Stadium") == "usbankstadium" and normalize_venue("Levi's Stadium") == "levisstadium"
    )


def test_load_rejects_bad_rows_and_an_empty_table(tmp_path: Path) -> None:
    header = "team,venue,city,state,country,roof,latitude,longitude\n"
    bad = tmp_path / "stadiums.csv"
    bad.write_text(header + "KC,Arrowhead,Kansas City,MO,USA,open,39.0,-94.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"stadiums\.csv:2"):
        Stadiums.load(bad)
    bad.write_text(header + "KC,Arrowhead,Kansas City,MO,USA,outdoors,99.0,-94.5\n", encoding="utf-8")
    with pytest.raises(ValueError, match="latitude"):
        Stadiums.load(bad)
    bad.write_text(header, encoding="utf-8")
    with pytest.raises(ValueError, match="no stadiums"):
        Stadiums.load(bad)


# --- Open-Meteo ---


def test_forecast_requests_us_units_on_utc_hours_and_parses_them(
    source: OpenMeteoSource, routes: Routes, stadiums: Stadiums
) -> None:
    metlife = stadium_of(stadiums, "NYG")
    result = source.forecast(metlife.latitude, metlife.longitude, forecast_days=2)
    params = dict(routes.forecast.calls.last.request.url.params)
    assert params == {
        "latitude": "40.8135",
        "longitude": "-74.0745",
        "hourly": ",".join(HOURLY_VARIABLES),
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "UTC",
        "forecast_days": "2",
    }
    assert routes.forecast.calls.last.request.headers["user-agent"].startswith("espn-fantasy/")
    forecast = result.data
    assert len(forecast.hours) == 48 and (result.cached, result.degraded, result.as_of) == (False, False, T0)
    assert forecast.start == datetime(2026, 10, 5, 0, 0, tzinfo=UTC)
    assert forecast.end == datetime(2026, 10, 7, 0, 0, tzinfo=UTC)
    assert (forecast.latitude, forecast.longitude, forecast.elevation_m) == (40.80899, -74.06947, 4.0)
    hour = forecast.at(datetime(2026, 10, 5, 2, 30, tzinfo=UTC))
    assert hour is not None and hour.time == datetime(2026, 10, 5, 2, 0, tzinfo=UTC) and hour.time.tzinfo is UTC
    assert (hour.temperature_f, hour.apparent_temperature_f, hour.precipitation_probability) == (57.4, 56.6, 0)
    assert (hour.precipitation_in, hour.rain_in, hour.snowfall_in) == (0.016, 0.016, 0.0)
    assert (hour.wind_mph, hour.gust_mph, hour.weather_code) == (5.2, 9.8, 51)
    assert forecast.at(datetime(2026, 10, 9, tzinfo=UTC)) is None
    assert result.key == "40.81_-74.07_2d" and result.raw_path is not None
    assert result.raw_path.name == "40.81_-74.07_2d.json" and result.raw_path.read_bytes() == fixture(FORECAST)


def test_game_weather_aggregates_the_three_hours_from_kickoff(source: OpenMeteoSource, stadiums: Stadiums) -> None:
    metlife = stadium_of(stadiums, "NYG")
    result = source.game_weather(metlife, SUNDAY_EARLY, forecast_days=2)
    weather = result.data
    assert (weather.available, weather.covered, len(weather.hours)) == (True, False, 3)
    assert [hour.time for hour in weather.hours] == [SUNDAY_EARLY + timedelta(hours=offset) for offset in range(3)]
    assert (weather.venue, weather.roof, weather.kickoff) == ("MetLife Stadium", "outdoors", SUNDAY_EARLY)
    assert (weather.latitude, weather.longitude) == (metlife.latitude, metlife.longitude)

    hourly = json.loads(fixture(FORECAST))["hourly"]
    rows = [hourly["time"].index(f"2026-10-05T{hour:02d}:00") for hour in (17, 18, 19)]
    assert weather.temperature_f == pytest.approx(sum(hourly["temperature_2m"][i] for i in rows) / 3)
    assert weather.apparent_temperature_f == pytest.approx(sum(hourly["apparent_temperature"][i] for i in rows) / 3)
    assert weather.wind_mph == max(hourly["wind_speed_10m"][i] for i in rows)
    assert weather.gust_mph == max(hourly["wind_gusts_10m"][i] for i in rows)
    assert weather.precipitation_probability == max(hourly["precipitation_probability"][i] for i in rows)
    assert weather.precipitation_in == pytest.approx(sum(hourly["precipitation"][i] for i in rows))
    assert weather.snowfall_in == pytest.approx(sum(hourly["snowfall"][i] for i in rows))
    assert weather.weather_code == max(hourly["weather_code"][i] for i in rows)
    assert (result.degraded, result.stale, result.warnings) == (False, False, ())
    assert result.dataset == "forecast" and result.key == "40.81_-74.07_2d"


def test_kickoff_minutes_and_naive_datetimes_snap_to_the_utc_hour(stadiums: Stadiums) -> None:
    forecast = parse_forecast(fixture(FORECAST))
    metlife = stadium_of(stadiums, "NYG")
    late = weather_at_kickoff(metlife, datetime(2026, 10, 6, 0, 15, tzinfo=UTC), forecast, hours=2)
    assert [hour.time for hour in late.hours] == [
        datetime(2026, 10, 6, 0, tzinfo=UTC),
        datetime(2026, 10, 6, 1, tzinfo=UTC),
    ]
    naive = weather_at_kickoff(metlife, datetime(2026, 10, 6, 0, 15), forecast, hours=2)
    assert naive.hours == late.hours and naive.kickoff == late.kickoff and naive.kickoff.tzinfo is UTC
    assert floor_hour(datetime(2026, 10, 6, 0, 59, 59)) == datetime(2026, 10, 6, 0, tzinfo=UTC)
    with pytest.raises(ValueError):
        weather_at_kickoff(metlife, SUNDAY_EARLY, forecast, hours=0)


def test_domes_skip_weather(source: OpenMeteoSource, routes: Routes, stadiums: Stadiums) -> None:
    superdome = stadium_of(stadiums, "NO")
    kickoff = datetime(2026, 10, 6, 0, 15, tzinfo=UTC)
    result = source.game_weather(superdome, kickoff)
    weather = result.data
    assert (weather.covered, weather.available, weather.hours) == (True, False, [])
    assert (weather.venue, weather.roof, weather.kickoff) == ("Caesars Superdome", "dome", kickoff)
    assert weather.temperature_f is None and weather.wind_mph is None and weather.precipitation_in is None
    assert (result.degraded, result.stale, result.warnings, result.as_of) == (False, False, (), T0)
    assert routes.forecast.call_count == 0

    # A retractable roof closes in bad weather, so it counts as covered too.
    cowboys = source.game_weather(stadium_of(stadiums, "DAL"), kickoff)
    assert cowboys.data.covered is True and cowboys.data.roof == "retractable"
    assert routes.forecast.call_count == 0


def test_espn_indoor_flag_overrides_the_table(source: OpenMeteoSource, routes: Routes, stadiums: Stadiums) -> None:
    # ESPN flags the venue indoor: skip, whatever the table says.
    flagged = source.game_weather(stadium_of(stadiums, "NYG"), SUNDAY_EARLY, indoor=True)
    assert flagged.data.covered is True and routes.forecast.call_count == 0
    # ESPN flags it outdoor: fetch, even for a venue the table calls a dome.
    opened = source.game_weather(stadium_of(stadiums, "NO"), SUNDAY_EARLY, indoor=False, forecast_days=2)
    assert opened.data.covered is False and opened.data.available is True and routes.forecast.call_count == 1
    assert opened.data.roof == "dome"  # the table's roof is still reported


def test_kickoff_outside_the_forecast_horizon_is_flagged(source: OpenMeteoSource, stadiums: Stadiums) -> None:
    far_off = datetime(2026, 10, 20, 17, 0, tzinfo=UTC)
    result = source.game_weather(stadium_of(stadiums, "NYG"), far_off, forecast_days=2)
    assert (result.data.available, result.data.covered, result.degraded) == (False, False, False)
    assert result.data.temperature_f is None and result.data.hours == []
    assert len(result.warnings) == 1 and "outside the forecast horizon" in result.warnings[0]
    assert "2026-10-20T17:00:00+00:00" in result.warnings[0] and "MetLife Stadium" in result.warnings[0]


def test_forecast_degrades_on_server_errors(
    source: OpenMeteoSource, routes: Routes, stadiums: Stadiums, sleeps: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    routes.forecast.mock(return_value=httpx.Response(500))
    with caplog.at_level(logging.WARNING, logger="fm.sources.weather"):
        result = source.game_weather(stadium_of(stadiums, "GB"), SUNDAY_EARLY)
    assert (result.degraded, result.stale, result.cached, result.as_of) == (True, False, False, T0)
    assert result.data.available is False and result.data.covered is False and result.data.venue == "Lambeau Field"
    assert len(result.warnings) == 1 and "HTTP 500" in result.warnings[0]  # no horizon warning on top of the outage
    assert "continuing without weather" in caplog.text
    assert routes.forecast.call_count == 3 and sleeps == [1.0, 2.0]  # retried, then gave up

    direct = source.forecast(0.0, 0.0)
    assert (
        direct.degraded is True and direct.data.hours == [] and (direct.data.latitude, direct.data.longitude) == (0, 0)
    )


def test_forecast_serves_the_stale_copy_when_the_refresh_fails(
    source: OpenMeteoSource, routes: Routes, clock: FakeClock, stadiums: Stadiums
) -> None:
    metlife = stadium_of(stadiums, "NYG")
    good = source.game_weather(metlife, SUNDAY_EARLY, forecast_days=2)
    clock.advance(minutes=61)
    routes.forecast.mock(return_value=httpx.Response(503))
    stale = source.game_weather(metlife, SUNDAY_EARLY, forecast_days=2)
    assert (stale.stale, stale.degraded, stale.cached, stale.as_of) == (True, False, True, T0)
    assert stale.data.hours == good.data.hours and stale.data.available is True
    assert stale.warnings and "HTTP 503" in stale.warnings[0]


def test_forecasts_are_cached_per_rounded_coordinate_cell(
    source: OpenMeteoSource, routes: Routes, clock: FakeClock
) -> None:
    source.forecast(40.8135, -74.0745, forecast_days=2)
    nearby = source.forecast(40.8101, -74.0699, forecast_days=2)  # the same 0.01 degree cell
    assert nearby.cached is True and routes.forecast.call_count == 1
    assert source.forecast(40.8135, -74.0745, forecast_days=3).cached is False  # a different horizon
    clock.advance(minutes=61)
    assert source.forecast(40.8135, -74.0745, forecast_days=2).cached is False
    assert routes.forecast.call_count == 3
    assert OpenMeteoSource.cache_key(29.9511, -90.0812, 7) == "29.95_-90.08_7d"


def test_forecast_days_are_bounded(source: OpenMeteoSource) -> None:
    for days in (0, 17):
        with pytest.raises(ValueError, match="forecast_days"):
            source.forecast(40.8, -74.1, forecast_days=days)


def test_parse_forecast_rejects_other_units_and_ragged_arrays() -> None:
    raw = json.loads(fixture(FORECAST))
    raw["hourly_units"]["temperature_2m"] = "°C"
    with pytest.raises(SourceSchemaError, match="temperature_2m"):
        parse_forecast(json.dumps(raw).encode())

    raw = json.loads(fixture(FORECAST))
    raw["hourly"]["wind_speed_10m"] = raw["hourly"]["wind_speed_10m"][:-1]
    with pytest.raises(SourceSchemaError, match="wind_speed_10m"):
        parse_forecast(json.dumps(raw).encode())

    for payload in (b"[]", b'{"hourly": {"time": "x"}}', b'{"hourly": []}'):
        with pytest.raises(SourceSchemaError):
            parse_forecast(payload)

    raw = json.loads(fixture(FORECAST))
    del raw["hourly"]["snowfall"]  # a missing variable is tolerated as nulls
    forecast = parse_forecast(json.dumps(raw).encode())
    assert len(forecast.hours) == 48 and all(hour.snowfall_in is None for hour in forecast.hours)
    assert parse_forecast(b'{"latitude": 1, "longitude": 2, "hourly": {"time": []}}').hours == []


def test_rejected_forecast_is_kept_beside_the_cache(
    source: OpenMeteoSource, routes: Routes, tmp_path: Path, stadiums: Stadiums
) -> None:
    routes.forecast.mock(return_value=httpx.Response(200, content=b'{"error": true, "reason": "nope"}'))
    result = source.game_weather(stadium_of(stadiums, "NYG"), SUNDAY_EARLY, forecast_days=2)
    assert result.degraded is True and "expected 'hourly'" in result.warnings[0]
    cache = tmp_path / "cache" / "open_meteo" / "forecast"
    assert (cache / "40.81_-74.07_2d.rejected.json").read_bytes() == b'{"error": true, "reason": "nope"}'
    assert not (cache / "40.81_-74.07_2d.json").exists()


def test_accidental_live_calls_are_not_swallowed(source: OpenMeteoSource, routes: Routes, stadiums: Stadiums) -> None:
    routes.forecast.mock(side_effect=RuntimeError("outbound network is disabled in unit tests"))
    with pytest.raises(RuntimeError, match="outbound network"):
        source.game_weather(stadium_of(stadiums, "NYG"), SUNDAY_EARLY)
