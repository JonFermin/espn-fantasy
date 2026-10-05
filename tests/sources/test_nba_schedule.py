"""NBA CDN schedule adapter against a trimmed real capture (respx; no network).

tests/fixtures/sources/nba_schedule/scheduleLeagueV2.json was captured on 2026-10-04 (the 2026-27 schedule) and cut to
the one finished preseason game, the opening-week games of ATL, CHA, OKC, IND and PHI (back-to-backs included), one NBA
Cup group game and the seven knockout placeholders, with broadcasters and pointsLeaders dropped.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx

from fm.sources.base import RateLimiter, SourceSchemaError, SourceUnavailable
from fm.sources.nba_schedule import (
    CDN_URL,
    GameType,
    NbaGame,
    NbaScheduleSource,
    NbaTeamRef,
    game_type,
    parse_schedule,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "sources" / "nba_schedule"
T0 = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)
OPENING_WEEK = (date(2026, 10, 20), date(2026, 10, 26))


def fixture() -> bytes:
    return (FIXTURES / "scheduleLeagueV2.json").read_bytes()


def ok() -> httpx.Response:
    return httpx.Response(200, content=fixture(), headers={"content-type": "text/plain"})


class FakeClock:
    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> datetime:
        self.now += timedelta(**delta)
        return self.now


@pytest.fixture
def route() -> Iterator[respx.Route]:
    with respx.mock(assert_all_called=False) as router:
        yield router.get(CDN_URL).mock(return_value=ok())


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def sleeps() -> list[float]:
    return []


@pytest.fixture
def source(route: respx.Route, clock: FakeClock, tmp_path: Path, sleeps: list[float]) -> Iterator[NbaScheduleSource]:
    with NbaScheduleSource(
        cache_root=tmp_path / "cache", limiter=RateLimiter(0), clock=clock, sleep=sleeps.append
    ) as src:
        yield src


def game(source: NbaScheduleSource, day: date, matchup: str) -> NbaGame:
    return next(g for g in source.schedule().data.games_on(day, regular_season_only=False) if g.matchup == matchup)


# --- parsing ---


def test_schedule_parses_games_weeks_and_meta(source: NbaScheduleSource) -> None:
    result = source.schedule()
    schedule = result.data
    assert (schedule.season_year, schedule.league_id) == ("2026-27", "00")
    assert schedule.generated_at == datetime(2026, 10, 4, 21, 10, 18, 101800, tzinfo=UTC)
    assert len(schedule.games) == 28 and len(schedule.regular_season()) == 26  # minus 1 preseason, 1 Cup final
    assert schedule.games == tuple(sorted(schedule.games, key=lambda g: (g.day, g.game_id)))
    assert [week.number for week in schedule.weeks] == [1, 2, 7, 8, 8]
    assert schedule.week_for(date(2026, 10, 21)) is not None
    assert schedule.week_for(date(2026, 10, 21)).number == 1  # type: ignore[union-attr]
    assert schedule.week_for(date(2026, 9, 1)) is None
    assert (result.source, result.dataset, result.key, result.as_of) == ("nba_schedule", "schedule", "league", T0)


def test_days_are_eastern_and_tips_are_utc(source: NbaScheduleSource) -> None:
    opening = source.schedule().data.games_on(date(2026, 10, 20))
    assert [g.matchup for g in opening] == ["PHI @ NYK", "OKC @ SAS"]
    late = opening[1]
    assert late.tip_utc == datetime(2026, 10, 21, 1, 30, tzinfo=UTC)  # 9:30 pm ET is already the 21st in UTC
    assert late.day == date(2026, 10, 20) and late.status_text == "9:30 pm ET"
    assert (late.game_id, late.game_code, late.arena, late.week_number) == (
        "0022600003",
        "20261020/OKCSAS",
        "Frost Bank Center",
        1,
    )
    assert (late.home.team_id, late.home.tricode, late.home.name, late.home.city) == (
        1610612759,
        "SAS",
        "Spurs",
        "San Antonio",
    )
    assert late.away.tricode == "OKC" and late.tricodes == frozenset({"OKC", "SAS"})
    assert late.game_type is GameType.REGULAR_SEASON and late.is_regular_season and not late.is_cup


def test_preseason_games_are_excluded_unless_asked(source: NbaScheduleSource) -> None:
    schedule = source.schedule().data
    assert schedule.games_on(date(2026, 10, 3)) == []
    (preseason,) = schedule.games_on(date(2026, 10, 3), regular_season_only=False)
    assert preseason.game_type is GameType.PRESEASON and preseason.is_final and preseason.status == 3
    assert (preseason.matchup, preseason.away.score, preseason.home.score) == ("MIA @ TOR", 129, 105)
    assert preseason.label == "Preseason" and preseason.sub_label == "NBA Canada Game"
    assert "MIA" in schedule.tricodes()  # MIA also has a regular-season game in the fixture
    assert "TOR" not in schedule.tricodes()  # TOR appears only in the preseason game


def test_game_counts_per_team_and_per_day(source: NbaScheduleSource) -> None:
    schedule = source.schedule().data
    counts = schedule.team_games(*OPENING_WEEK)
    assert {team: n for team, n in counts.items() if n >= 2} == {
        "ATL": 4,
        "CHA": 4,
        "OKC": 4,
        "IND": 4,
        "PHI": 4,
        "BKN": 2,
        "HOU": 2,
        "MIL": 2,
    }
    assert len(counts) == 20 and sum(counts.values()) == 2 * 19
    assert counts["DET"] == 1  # the Cup group game on Oct 30 is outside the range
    assert schedule.games_per_day(date(2026, 10, 20), date(2026, 10, 27)) == {
        date(2026, 10, 20): 2,
        date(2026, 10, 21): 3,
        date(2026, 10, 22): 2,
        date(2026, 10, 23): 2,
        date(2026, 10, 24): 3,
        date(2026, 10, 25): 3,
        date(2026, 10, 26): 4,
        date(2026, 10, 27): 0,
    }
    assert len(schedule.games_between(*OPENING_WEEK)) == 19
    assert [g.day for g in schedule.team_schedule("ATL")] == [date(2026, 10, d) for d in (21, 23, 24, 26)]


def test_back_to_backs(source: NbaScheduleSource) -> None:
    schedule = source.schedule().data
    first, second = game(source, date(2026, 10, 23), "ATL @ CHA"), game(source, date(2026, 10, 24), "HOU @ ATL")
    assert schedule.back_to_backs("ATL") == [(first, second)]
    assert schedule.is_second_of_back_to_back(second, "ATL") is True
    assert schedule.is_second_of_back_to_back(second, "HOU") is False  # Houston's previous game was not the day before
    assert schedule.is_second_of_back_to_back(first, "ATL") is False
    assert [(a.day, b.day) for a, b in schedule.back_to_backs("CHA")] == [(date(2026, 10, 23), date(2026, 10, 24))]
    assert [(a.matchup, b.matchup) for a, b in schedule.back_to_backs("OKC")] == [("LAC @ OKC", "PHX @ OKC")]
    assert [(a.matchup, b.matchup) for a, b in schedule.back_to_backs("IND")] == [("IND @ BKN", "IND @ MIL")]
    assert [(a.matchup, b.matchup) for a, b in schedule.back_to_backs("PHI")] == [("MIL @ PHI", "DET @ PHI")]
    assert schedule.back_to_backs("HOU") == [] and schedule.back_to_backs("TOR") == []


def test_cup_group_games_count_and_knockout_placeholders_are_flagged(source: NbaScheduleSource) -> None:
    result = source.schedule()
    schedule = result.data
    (group,) = schedule.games_on(date(2026, 10, 30))
    assert (group.matchup, group.label, group.sub_label, group.subtype) == (
        "DET @ BKN",
        "Emirates NBA Cup",
        "East Group A",
        "in-season",
    )
    assert group.is_cup and not group.is_cup_knockout and not group.teams_tbd and group.is_regular_season

    pending = schedule.unscheduled_cup_games()
    assert len(pending) == 7
    assert [g.day for g in pending] == [date(2026, 12, d) for d in (4, 4, 5, 5, 8, 8, 11)]
    assert all(g.is_cup and g.is_cup_knockout and g.teams_tbd and g.tip_utc is None for g in pending)
    assert {g.status_text for g in pending} == {"TBD"} and {g.matchup for g in pending} == {"TBD @ TBD"}
    assert [g.sub_label for g in pending] == ["Quarterfinal"] * 4 + ["Semifinal"] * 2 + ["Championship"]
    quarterfinal, championship = pending[0], pending[-1]
    assert (quarterfinal.game_id, quarterfinal.game_code, quarterfinal.week_number) == ("0022601201", None, 7)
    assert quarterfinal.is_regular_season and quarterfinal.home.is_tbd and quarterfinal.tricodes == frozenset()
    assert championship.game_type is GameType.CUP_FINAL and not championship.is_regular_season
    assert championship.arena == "Hinkle Fieldhouse" and championship not in schedule.regular_season()
    assert schedule.games_on(date(2026, 12, 4)) == pending[:2]  # placeholders still occupy their ET day
    assert result.warnings == (
        "7 NBA Cup knockout games (2026-12-04 to 2026-12-11) are not scheduled yet; re-pull after group play",
    )


def test_game_type_is_the_third_digit_of_the_id() -> None:
    assert game_type("0012600009") is GameType.PRESEASON
    assert game_type("0022600119") is GameType.REGULAR_SEASON
    assert game_type("0032600001") is GameType.ALL_STAR
    assert game_type("0042500402") is GameType.PLAYOFFS
    assert game_type("0052500101") is GameType.PLAY_IN
    assert game_type("0062600001") is GameType.CUP_FINAL
    assert game_type("09") is GameType.UNKNOWN and game_type("0092600001") is GameType.UNKNOWN


def test_tbd_team_refs_read_as_unknown() -> None:
    tbd = NbaTeamRef.model_validate({"teamId": 0, "teamTricode": "TBD", "teamName": "", "teamCity": None})
    assert tbd.is_tbd and (tbd.team_id, tbd.tricode, tbd.name) == (None, None, None)
    real = NbaTeamRef.model_validate({"teamId": 1610612737, "teamTricode": "ATL", "teamName": "Hawks"})
    assert not real.is_tbd and real.tricode == "ATL"


def test_parse_rejects_wrong_shapes_and_skips_bad_games(caplog: pytest.LogCaptureFixture) -> None:
    for payload in (b"[]", b'{"leagueSchedule": []}', b'{"leagueSchedule": {"gameDates": []}}'):
        with pytest.raises(SourceSchemaError):
            parse_schedule(payload)
    with pytest.raises(SourceSchemaError, match="no games list"):
        parse_schedule(b'{"leagueSchedule": {"gameDates": [{"gameDate": "x"}]}}')
    with pytest.raises(SourceSchemaError, match="none of 1 games validated"):
        parse_schedule(b'{"leagueSchedule": {"gameDates": [{"gameDate": "x", "games": [{"gameId": 1}]}]}}')
    good = {
        "gameId": "0022600002",
        "gameStatusText": "7:00 pm ET",
        "gameDateEst": "2026-10-20T00:00:00Z",
        "gameDateTimeUTC": "2026-10-20T23:00:00Z",
        "homeTeam": {"teamId": 1610612752, "teamTricode": "NYK"},
        "awayTeam": {"teamId": 1610612755, "teamTricode": "PHI"},
    }
    payload = {"leagueSchedule": {"seasonYear": "2026-27", "gameDates": [{"games": [good, {"gameId": 1}]}]}}
    with caplog.at_level(logging.WARNING, logger="fm.sources.nba_schedule"):
        schedule = parse_schedule(httpx.Response(200, json=payload).content)
    assert len(schedule.games) == 1 and schedule.weeks == () and schedule.generated_at is None
    assert "skipped 1 of 2 games" in caplog.text


# --- transport ---


def test_sends_the_full_chrome_header_set(source: NbaScheduleSource, route: respx.Route) -> None:
    source.schedule()
    headers = route.calls.last.request.headers
    assert "Chrome/145" in headers["user-agent"] and "espn-fantasy" not in headers["user-agent"]
    assert headers["origin"] == "https://www.nba.com" and headers["referer"] == "https://www.nba.com/"
    assert headers["sec-fetch-mode"] == "cors" and headers["sec-ch-ua-platform"] == '"Windows"'
    assert "sec-ch-ua" in headers


def test_cached_for_a_day_then_refreshed_or_forced(
    source: NbaScheduleSource, route: respx.Route, clock: FakeClock
) -> None:
    first = source.schedule()
    clock.advance(hours=23)
    assert source.schedule().cached is True and route.call_count == 1
    later = clock.advance(hours=2)
    refreshed = source.schedule()
    assert refreshed.cached is False and refreshed.as_of == later and route.call_count == 2
    assert refreshed.data.games == first.data.games
    assert source.schedule().cached is True and route.call_count == 2
    forced = source.schedule(force=True)  # the December re-pull after Cup group play
    assert forced.cached is False and forced.as_of == later and route.call_count == 3


def test_403_without_a_copy_raises_and_with_one_serves_stale(
    source: NbaScheduleSource, route: respx.Route, clock: FakeClock, sleeps: list[float]
) -> None:
    route.mock(return_value=httpx.Response(403, text="Access Denied"))
    with pytest.raises(SourceUnavailable, match="HTTP 403"):
        source.schedule()
    assert route.call_count == 1 and sleeps == []  # a 403 is not retried

    route.mock(return_value=ok())
    good = source.schedule()
    clock.advance(hours=25)
    route.mock(return_value=httpx.Response(403, text="Access Denied"))
    stale = source.schedule()
    assert (stale.stale, stale.cached, stale.as_of) == (True, True, T0)
    assert stale.data.games == good.data.games and "HTTP 403" in stale.warnings[0]
    assert stale.warnings[-1].startswith("7 NBA Cup knockout games")


def test_raw_capture_is_the_cdn_file(source: NbaScheduleSource, tmp_path: Path) -> None:
    result = source.schedule()
    raw = tmp_path / "cache" / "nba_schedule" / "schedule" / "league.json"
    assert result.raw_path == raw and raw.read_bytes() == fixture()
    assert source.schedule().data is result.data  # parsed value memoized while the cached bytes are unchanged
