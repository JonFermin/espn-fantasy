"""ESPN scoreboard and The Odds API adapters against recorded and hand-built responses (respx; no network).

``espn_scoreboard_nfl_2026_w4.json`` is the public NFL scoreboard recorded on 2026-10-04 (week 4), trimmed to three
events: ATL @ NO (scheduled, DraftKings line, dome), DET @ CAR (in progress: odds gone, AccuWeather fields swapped) and
IND vs WSH (final, neutral site in London). The Odds API fixtures are hand-built in the documented v4 shapes.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx
from hypothesis import given
from hypothesis import strategies as st
from pydantic import SecretStr

from fm.espn.ids import Game
from fm.sources.base import RateLimiter, SourceSchemaError
from fm.sources.odds import (
    REDACTED,
    EspnScoreboardSource,
    OddsApiSource,
    RedactApiKeyFilter,
    TeamTotals,
    implied_team_totals,
    parse_odds_events,
    parse_scoreboard,
    parse_team_totals,
    pick_line,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "sources" / "odds"
SCOREBOARD = "espn_scoreboard_nfl_2026_w4.json"
T0 = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)
ESPN = "site.api.espn.com"
ODDS = "api.the-odds-api.com"
KEY = "0123456789abcdef0123456789abcdef"
EVENT = "9a3f1c2e5b7d4e6f8a1b2c3d4e5f6a7b"


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
        self.scoreboard = router.get(host=ESPN, path__regex=r"^/apis/site/v2/sports/[a-z]+/[a-z]+/scoreboard$").mock(
            return_value=ok(SCOREBOARD)
        )
        self.events = router.get(host=ODDS, path="/v4/sports/americanfootball_nfl/events").mock(
            return_value=ok("odds_api_events.json")
        )
        self.team_totals = router.get(
            host=ODDS, path__regex=r"^/v4/sports/americanfootball_nfl/events/[^/]+/odds$"
        ).mock(return_value=ok("odds_api_event_team_totals.json"))


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
def espn(routes: Routes, clock: FakeClock, tmp_path: Path, sleeps: list[float]) -> Iterator[EspnScoreboardSource]:
    source = EspnScoreboardSource(
        cache_root=tmp_path / "cache", limiter=RateLimiter(0), clock=clock, sleep=sleeps.append
    )
    with source:
        yield source


@pytest.fixture
def odds_api(routes: Routes, clock: FakeClock, tmp_path: Path, sleeps: list[float]) -> Iterator[OddsApiSource]:
    source = OddsApiSource(KEY, cache_root=tmp_path / "cache", limiter=RateLimiter(0), clock=clock, sleep=sleeps.append)
    with source:
        yield source


# --- implied totals math ---


def test_implied_team_totals_split_the_total_by_the_spread() -> None:
    home_favored = implied_team_totals(spread=-1.5, over_under=47.5)  # NO -1.5, O/U 47.5 with the Saints at home
    assert (home_favored.home, home_favored.away) == (24.5, 23.0)
    away_favored = implied_team_totals(spread=6.5, over_under=44.0)
    assert (away_favored.home, away_favored.away) == (18.75, 25.25)
    pick = implied_team_totals(spread=0.0, over_under=41.0)
    assert (pick.home, pick.away) == (20.5, 20.5)
    assert home_favored.total == 47.5 and home_favored.spread == -1.5


half_points = st.integers(min_value=-60, max_value=60).map(lambda n: n / 2)  # lines are posted in half points
totals_lines = st.integers(min_value=40, max_value=160).map(lambda n: n / 2)


@given(spread=half_points, over_under=totals_lines)
def test_implied_totals_add_up_to_the_total_and_differ_by_the_spread(spread: float, over_under: float) -> None:
    totals = implied_team_totals(spread=spread, over_under=over_under)
    assert totals.home + totals.away == pytest.approx(over_under)
    assert totals.away - totals.home == pytest.approx(spread)
    if spread < 0:
        assert totals.home > totals.away  # the home team is favored
    elif spread > 0:
        assert totals.away > totals.home
    else:
        assert totals.home == totals.away


def test_implied_totals_reject_a_negative_total() -> None:
    with pytest.raises(ValueError):
        implied_team_totals(spread=0.0, over_under=-1.0)


# --- ESPN scoreboard ---


def test_scoreboard_games_are_typed_and_in_kickoff_order(espn: EspnScoreboardSource) -> None:
    result = espn.scoreboard(season=2026, week=4)
    board = result.data
    assert (board.season, board.season_type, board.week) == (2026, 2, 4)
    assert [game.short_name for game in board.games] == ["IND VS WSH", "DET @ CAR", "ATL @ NO"]
    assert result.degraded is False and result.warnings == ()

    saints = board.game("401872979")
    assert saints is not None and board.for_team("ATL") is saints and board.for_team("KC") is None
    assert saints.kickoff == datetime(2026, 10, 6, 0, 15, tzinfo=UTC) and saints.kickoff.tzinfo is UTC
    assert (saints.season, saints.season_type, saints.week) == (2026, 2, 4)
    assert (saints.home.id, saints.home.abbreviation, saints.home.display_name) == (18, "NO", "New Orleans Saints")
    assert (saints.away.abbreviation, saints.away.nickname, saints.away.location) == ("ATL", "Falcons", "Atlanta")
    assert (saints.state, saints.completed, saints.neutral_site) == ("pre", False, False)
    assert saints.venue is not None
    assert (saints.venue.id, saints.venue.name, saints.venue.city) == ("3493", "Caesars Superdome", "New Orleans")
    assert (saints.venue.state, saints.venue.country, saints.venue.indoor, saints.indoor) == ("LA", "USA", True, True)
    assert saints.line is not None
    assert (saints.line.provider, saints.line.details) == ("DraftKings", "NO -1.5")
    assert (saints.line.spread, saints.line.over_under, saints.line.home_favorite) == (-1.5, 47.5, True)


def test_scoreboard_lines_become_implied_team_totals(espn: EspnScoreboardSource) -> None:
    saints = espn.scoreboard(season=2026, week=4).data.game("401872979")
    assert saints is not None and saints.line is not None
    assert saints.implied_totals == saints.line.implied_totals == TeamTotals(home=24.5, away=23.0)
    assert saints.implied_total("NO") == 24.5 and saints.implied_total("ATL") == 23.0
    with pytest.raises(ValueError, match="KC is not playing"):
        saints.implied_total("KC")


def test_started_games_lose_their_line_but_keep_scores_and_venue(espn: EspnScoreboardSource) -> None:
    board = espn.scoreboard(season=2026, week=4).data
    panthers = board.game("401872978")
    assert panthers is not None
    assert (panthers.state, panthers.completed, panthers.line, panthers.implied_totals) == ("in", False, None, None)
    assert panthers.implied_total("DET") is None
    assert (panthers.home.abbreviation, panthers.home_score, panthers.away.abbreviation, panthers.away_score) == (
        "CAR",
        7,
        "DET",
        3,
    )
    london = board.game("401872965")
    assert london is not None and london.venue is not None
    assert (london.state, london.completed, london.neutral_site) == ("post", True, True)
    assert (london.venue.name, london.venue.city, london.venue.state, london.venue.country, london.venue.indoor) == (
        "Tottenham Hotspur Stadium",
        "London",
        None,
        "England",
        False,
    )
    assert (london.home.abbreviation, london.away.abbreviation, london.home_score, london.away_score) == (
        "WSH",
        "IND",
        13,
        30,
    )
    assert london.weather is None


def test_accuweather_snippet_is_parsed_even_when_espn_swaps_its_fields(espn: EspnScoreboardSource) -> None:
    board = espn.scoreboard(season=2026, week=4).data
    saints = board.game("401872979")
    assert saints is not None and saints.weather is not None
    assert (saints.weather.description, saints.weather.condition_id) == ("Cloudy", "7")
    assert (saints.weather.temperature_f, saints.weather.high_temperature_f) == (78, 78)
    assert saints.weather.link is not None and saints.weather.link.startswith("http://www.accuweather.com/")
    panthers = board.game("401872978")  # recorded as {"displayValue": "7", "conditionId": "Cloudy"}
    assert panthers is not None and panthers.weather is not None
    assert (panthers.weather.description, panthers.weather.condition_id, panthers.weather.temperature_f) == (
        "Cloudy",
        "7",
        65,
    )


def test_scoreboard_request_cache_key_and_browser_user_agent(
    espn: EspnScoreboardSource, routes: Routes, clock: FakeClock, tmp_path: Path
) -> None:
    first = espn.scoreboard(Game.FFL, season=2026, week=4)
    request = routes.scoreboard.calls.last.request
    assert request.url.path == "/apis/site/v2/sports/football/nfl/scoreboard"
    assert dict(request.url.params) == {"week": "4", "seasontype": "2", "dates": "2026"}
    assert request.headers["user-agent"].startswith("Mozilla/5.0")
    assert (first.key, first.cached, first.as_of) == ("nfl_2026_t2_w4", False, T0)

    clock.advance(minutes=9)
    again = espn.scoreboard("nfl", season=2026, week=4)
    assert again.cached is True and again.as_of == T0 and routes.scoreboard.call_count == 1
    raw = tmp_path / "cache" / "espn_scoreboard" / "scoreboard" / "nfl_2026_t2_w4.json"
    assert first.raw_path == raw and raw.read_bytes() == fixture(SCOREBOARD)
    meta = json.loads((raw.parent / "nfl_2026_t2_w4.meta.json").read_text(encoding="utf-8"))
    assert meta["url"] == "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    assert (meta["week"], meta["seasontype"], meta["dates"]) == (4, 2, 2026)

    clock.advance(minutes=2)
    assert espn.scoreboard(season=2026, week=4).cached is False and routes.scoreboard.call_count == 2


def test_scoreboard_keys_for_a_day_the_current_slate_and_the_postseason(
    espn: EspnScoreboardSource, routes: Routes
) -> None:
    by_day = espn.scoreboard(Game.FBA, day=date(2026, 10, 4))
    request = routes.scoreboard.calls.last.request
    assert request.url.path == "/apis/site/v2/sports/basketball/nba/scoreboard"
    assert dict(request.url.params) == {"dates": "20261004"} and by_day.key == "nba_20261004"

    current = espn.scoreboard()
    assert dict(routes.scoreboard.calls.last.request.url.params) == {} and current.key == "nfl_current"

    playoffs = espn.scoreboard(week=1, season_type=3)
    assert dict(routes.scoreboard.calls.last.request.url.params) == {"week": "1", "seasontype": "3"}
    assert playoffs.key == "nfl_current_t3_w1"


def test_scoreboard_degrades_on_server_errors(
    espn: EspnScoreboardSource, routes: Routes, sleeps: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    routes.scoreboard.mock(return_value=httpx.Response(503))
    with caplog.at_level(logging.WARNING, logger="fm.sources.odds"):
        result = espn.scoreboard(season=2026, week=4)
    assert (result.degraded, result.stale, result.cached, result.as_of) == (True, False, False, T0)
    assert result.data.games == [] and result.data.game("401872979") is None
    assert "HTTP 503" in result.warnings[0] and "continuing without lines" in caplog.text
    assert routes.scoreboard.call_count == 3 and sleeps == [1.0, 2.0]  # retried, then gave up


def test_scoreboard_serves_the_stale_copy_when_the_refresh_fails(
    espn: EspnScoreboardSource, routes: Routes, clock: FakeClock
) -> None:
    good = espn.scoreboard(season=2026, week=4)
    clock.advance(minutes=11)
    routes.scoreboard.mock(return_value=httpx.Response(500))
    stale = espn.scoreboard(season=2026, week=4)
    assert (stale.stale, stale.degraded, stale.cached, stale.as_of) == (True, False, True, T0)
    assert [game.event_id for game in stale.data.games] == [game.event_id for game in good.data.games]
    assert stale.warnings and "HTTP 500" in stale.warnings[0]


def test_scoreboard_rejects_the_wrong_shape_and_keeps_it_beside_the_cache(
    espn: EspnScoreboardSource, routes: Routes, tmp_path: Path
) -> None:
    routes.scoreboard.mock(return_value=httpx.Response(200, content=b'{"leagues": []}'))
    result = espn.scoreboard(season=2026, week=4)
    assert result.degraded is True and result.data.games == []
    assert "expected an 'events' list" in result.warnings[0]
    cache = tmp_path / "cache" / "espn_scoreboard" / "scoreboard"
    assert (cache / "nfl_2026_t2_w4.rejected.json").read_bytes() == b'{"leagues": []}'
    assert not (cache / "nfl_2026_t2_w4.json").exists()

    for payload in (b"[]", b'{"events": {}}', b'"nope"'):
        with pytest.raises(SourceSchemaError):
            parse_scoreboard(payload)
    assert parse_scoreboard(b'{"events": []}').games == []  # a quiet day, not a broken feed


def test_scoreboard_skips_bad_events_unless_none_parse(caplog: pytest.LogCaptureFixture) -> None:
    raw = json.loads(fixture(SCOREBOARD))
    raw["events"].extend([{"id": "broken"}, 42, {"id": "no-sides", "date": "2026-10-04T17:00Z", "competitions": [{}]}])
    with caplog.at_level(logging.WARNING, logger="fm.sources.odds"):
        board = parse_scoreboard(json.dumps(raw).encode())
    assert len(board.games) == 3 and "skipped 3 of 6" in caplog.text
    with pytest.raises(SourceSchemaError, match="none of 1 events parsed"):
        parse_scoreboard(b'{"events": [{"id": "broken"}]}')


def test_pick_line_prefers_draftkings_then_espn_priority() -> None:
    caesars = {"provider": {"name": "Caesars", "priority": 2}, "spread": 3.0, "overUnder": 40.0}
    draftkings = {
        "provider": {"name": "DraftKings", "priority": 5},
        "details": "HOME -2.5",
        "spread": -2.5,
        "overUnder": "44.5",
        "homeTeamOdds": {"favorite": True},
    }
    espn_bet = {"provider": {"name": "ESPN BET", "priority": 1}, "spread": 1.0, "overUnder": 42.0}

    line = pick_line([caesars, draftkings, espn_bet])
    assert line is not None
    assert (line.provider, line.spread, line.over_under, line.home_favorite, line.details) == (
        "DraftKings",
        -2.5,
        44.5,
        True,
        "HOME -2.5",
    )
    fallback = pick_line([caesars, espn_bet])
    assert fallback is not None and (fallback.provider, fallback.home_favorite) == ("ESPN BET", None)
    assert pick_line([]) is None and pick_line(["junk", 1]) is None

    posted_off = pick_line([{"provider": {"name": "DraftKings"}, "details": "OFF"}])
    assert posted_off is not None and posted_off.spread is None and posted_off.implied_totals is None
    unnamed = pick_line([{"spread": -3.0, "overUnder": 41.0}])
    assert unnamed is not None and unnamed.provider == "unknown" and unnamed.implied_totals == TeamTotals(22.0, 19.0)


def test_accidental_live_calls_are_not_swallowed(espn: EspnScoreboardSource, routes: Routes) -> None:
    # The harness blocks the network with a bare RuntimeError; graceful degradation must never hide one.
    routes.scoreboard.mock(side_effect=RuntimeError("outbound network is disabled in unit tests"))
    with pytest.raises(RuntimeError, match="outbound network"):
        espn.scoreboard(season=2026, week=4)


# --- The Odds API ---


def test_odds_api_without_a_key_is_degraded_and_sends_nothing(routes: Routes, clock: FakeClock, tmp_path: Path) -> None:
    source = OddsApiSource(None, cache_root=tmp_path / "cache", limiter=RateLimiter(0), clock=clock)
    assert source.configured is False
    for result in (source.events(), source.team_totals(EVENT)):
        assert (result.degraded, result.data, result.as_of) == (True, [], T0)
        assert "ODDS_API_KEY" in result.warnings[0]
    assert routes.events.call_count == 0 and routes.team_totals.call_count == 0
    assert OddsApiSource("   ").configured is False
    assert OddsApiSource(SecretStr(KEY)).configured is True and OddsApiSource(KEY).configured is True


def test_odds_api_events_are_typed_and_free(odds_api: OddsApiSource, routes: Routes) -> None:
    result = odds_api.events()
    events = result.data
    assert [event.home_team for event in events] == ["New Orleans Saints", "Philadelphia Eagles"]
    assert events[0].commence_time == datetime(2026, 10, 6, 0, 15, tzinfo=UTC)
    assert (events[0].id, events[0].sport_key, events[0].away_team) == (
        EVENT,
        "americanfootball_nfl",
        "Atlanta Falcons",
    )
    assert events[0].is_game("New Orleans Saints", "Atlanta Falcons") is True
    assert events[0].is_game("Atlanta Falcons", "New Orleans Saints") is False
    request = routes.events.calls.last.request
    assert dict(request.url.params) == {"apiKey": KEY} and result.key == "nfl"


def test_odds_api_team_totals_one_per_book_and_team(odds_api: OddsApiSource, routes: Routes) -> None:
    result = odds_api.team_totals(EVENT)
    totals = result.data
    assert len(totals) == 4 and {total.event_id for total in totals} == {EVENT}
    draftkings = {total.team: total for total in totals if total.bookmaker == "draftkings"}
    saints = draftkings["New Orleans Saints"]
    assert (saints.point, saints.over_price, saints.under_price) == (24.5, -115, -105)
    assert saints.last_update == datetime(2026, 10, 4, 18, 30, 12, tzinfo=UTC)
    assert draftkings["Atlanta Falcons"].point == 22.5
    fanduel = {total.team: total for total in totals if total.bookmaker == "fanduel"}
    assert fanduel["Atlanta Falcons"].point == 23.5 and fanduel["New Orleans Saints"].under_price == 100
    params = dict(routes.team_totals.calls.last.request.url.params)
    assert params == {"regions": "us", "markets": "team_totals", "oddsFormat": "american", "apiKey": KEY}
    assert routes.team_totals.calls.last.request.url.path == f"/v4/sports/americanfootball_nfl/events/{EVENT}/odds"
    assert result.key == f"nfl_{EVENT}"


def test_odds_api_key_never_reaches_cache_metadata_warnings_or_logs(
    odds_api: OddsApiSource, routes: Routes, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    # Capture at INFO with no logger filter: httpx logs every request URL, query string included, at that level.
    with caplog.at_level(logging.INFO):
        odds_api.events()
    meta = (tmp_path / "cache" / "odds_api" / "events" / "nfl.meta.json").read_text(encoding="utf-8")
    assert KEY not in meta and "api.the-odds-api.com/v4/sports/americanfootball_nfl/events" in meta
    request_lines = [record.getMessage() for record in caplog.records if record.name == "httpx"]
    assert len(request_lines) == 1 and request_lines[0].startswith("HTTP Request: GET https://api.the-odds-api.com/")
    assert f"/v4/sports/americanfootball_nfl/events?apiKey={REDACTED} " in request_lines[0]
    assert KEY not in caplog.text
    assert f"apiKey={KEY}" in str(routes.events.calls.last.request.url)  # the request itself did carry it

    caplog.clear()
    routes.team_totals.mock(return_value=httpx.Response(401))
    with caplog.at_level(logging.INFO):
        result = odds_api.team_totals(EVENT)
    assert result.degraded is True and result.data == []
    assert "HTTP 401" in result.warnings[0] and REDACTED in result.warnings[0] and KEY not in result.warnings[0]
    assert KEY not in caplog.text and "unavailable" in caplog.text
    assert f'&apiKey={REDACTED} "HTTP/1.1 401 Unauthorized"' in caplog.text  # logged, but redacted
    assert f"apiKey={KEY}" in str(routes.team_totals.calls.last.request.url)


def test_httpx_request_log_filter_redacts_only_the_api_key() -> None:
    def record(url: str) -> logging.LogRecord:
        # The exact shape httpx logs: the URL object is a format argument, not part of the message.
        args = ("GET", httpx.URL(url), "HTTP/1.1", 200, "OK")
        return logging.LogRecord("httpx", logging.INFO, "_client.py", 1, 'HTTP Request: %s %s "%s %d %s"', args, None)

    redactor = RedactApiKeyFilter()
    middle = record(f"https://api.the-odds-api.com/v4/x/odds?regions=us&apiKey={KEY}&markets=team_totals%2Ctotals")
    assert redactor.filter(middle) is True
    assert middle.getMessage() == (  # the %2C survives: the rewritten record is not formatted a second time
        f"HTTP Request: GET https://api.the-odds-api.com/v4/x/odds?regions=us&apiKey={REDACTED}&markets=team_totals%2C"
        'totals "HTTP/1.1 200 OK"'
    )
    last = record(f"https://api.the-odds-api.com/v4/x/events?apikey={KEY}")
    redactor.filter(last)
    assert (
        last.getMessage()
        == f'HTTP Request: GET https://api.the-odds-api.com/v4/x/events?apikey={REDACTED} "HTTP/1.1 200 OK"'
    )

    untouched = record("https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard?week=4&dates=100%25")
    assert redactor.filter(untouched) is True
    assert untouched.args is not None and len(untouched.args) == 5  # left for the handler to format as usual
    assert untouched.getMessage().endswith('scoreboard?week=4&dates=100%25 "HTTP/1.1 200 OK"')

    for _ in range(3):  # idempotent: one filter per logger however many sources are built
        OddsApiSource(KEY)
    assert sum(isinstance(f, RedactApiKeyFilter) for f in logging.getLogger("httpx").filters) == 1


def test_odds_api_rejects_the_wrong_shapes() -> None:
    with pytest.raises(SourceSchemaError, match="expected a JSON list"):
        parse_odds_events(b"{}")
    with pytest.raises(SourceSchemaError):
        parse_odds_events(b'[{"id": "x"}]')
    with pytest.raises(SourceSchemaError, match="expected a JSON object"):
        parse_team_totals(b"[]")
    with pytest.raises(SourceSchemaError, match="bookmakers"):
        parse_team_totals(b'{"id": "x"}')
    assert parse_team_totals(b'{"id": "x", "bookmakers": []}') == [] and parse_odds_events(b"[]") == []


def test_odds_api_ignores_other_markets_and_alternate_lines() -> None:
    payload = {
        "id": "x",
        "bookmakers": [
            {
                "key": "draftkings",
                "markets": [
                    {"key": "totals", "outcomes": [{"name": "Over", "price": -110, "point": 47.5}]},
                    {
                        "key": "team_totals",
                        "outcomes": [
                            {"name": "Over", "description": "Team A", "price": -110, "point": 24.5},
                            {"name": "Over", "description": "Team A", "price": 150, "point": 27.5},
                            {"name": "Under", "description": "Team A", "price": -110, "point": 24.5},
                            {"name": "Over", "description": "Team B", "price": -105},
                        ],
                    },
                ],
            }
        ],
    }
    (total,) = parse_team_totals(json.dumps(payload).encode())
    assert (total.team, total.point, total.over_price, total.under_price, total.last_update) == (
        "Team A",
        24.5,
        -110,
        -110,
        None,
    )


def test_odds_api_ttls_respect_the_free_tier() -> None:
    assert OddsApiSource.ttl["team_totals"] >= timedelta(hours=6) and OddsApiSource.ttl["events"] >= timedelta(hours=1)
    assert EspnScoreboardSource.ttl["scoreboard"] <= timedelta(minutes=15)  # lines move
