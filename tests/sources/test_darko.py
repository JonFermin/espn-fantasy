"""DARKO adapter against trimmed exports of the public Google Sheet (respx; no network).

tests/fixtures/sources/darko/talent.csv and daily.csv were exported on 2026-10-04 (the sheet still held the June 2026
end-of-season run) and cut to a dozen players: free agents (``tm_id`` -999), a player projected for zero minutes
(``available`` 0) and, in the daily tab, two players with a game that day.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import httpx
import pytest
import respx

from fm.sources.base import RateLimiter, SourceSchemaError
from fm.sources.darko import DAILY_URL, TALENT_URL, DarkoSource, parse_daily, parse_talent

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "sources" / "darko"
T0 = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)
JOKIC, WEMBANYAMA, BRUNSON, SGA = 203999, 1641705, 1628973, 1628983
LOGIN_PAGE = b"<!DOCTYPE html><html><head><title>Google Sheets: Sign-in</title></head><body></body></html>"


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def ok(name: str) -> httpx.Response:
    return httpx.Response(200, content=fixture(name), headers={"content-type": "text/csv; charset=utf-8"})


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
        self.talent = router.get(TALENT_URL).mock(return_value=ok("talent.csv"))
        self.daily = router.get(DAILY_URL).mock(return_value=ok("daily.csv"))


@pytest.fixture
def routes() -> Iterator[Routes]:
    with respx.mock(assert_all_called=False) as router:
        yield Routes(router)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def source(routes: Routes, clock: FakeClock, tmp_path: Path) -> Iterator[DarkoSource]:
    with DarkoSource(cache_root=tmp_path / "cache", limiter=RateLimiter(0), clock=clock, sleep=lambda _: None) as src:
        yield src


# --- talent ---


def test_talent_rows_are_typed(source: DarkoSource) -> None:
    result = source.projections()
    rows = result.data
    assert len(rows) == 13 and result.degraded is False
    assert [row.name for row in rows][:2] == ["Nikola Jokic", "Victor Wembanyama"]

    jokic = rows[0]
    assert (jokic.nba_id, jokic.position, jokic.x_position, jokic.career_games) == (JOKIC, "C-F", "c_pos", 984)
    assert (jokic.team_id, jokic.team, jokic.available, jokic.age) == (None, None, True, 31.0)  # free agent
    assert (jokic.minutes, jokic.pace) == (29.3, 94.42)
    assert jokic.per_100["pts"] == 31.73 and jokic.per_100["ast"] == 10.67 and jokic.per_100["rim_fga"] == 4.89
    assert set(jokic.per_100) == {
        "pts",
        "orb",
        "drb",
        "ast",
        "pf",
        "blk",
        "stl",
        "tov",
        "fta",
        "fga",
        "rim_fga",
        "fg3a",
    }
    assert (jokic.rates["fg_pct"], jokic.rates["fg3_pct"], jokic.rates["ft_pct"], jokic.rates["usg_pct"]) == (
        0.53,
        0.34,
        0.79,
        0.28,
    )
    assert (jokic.dpm["dpm"], jokic.dpm["o_dpm"], jokic.dpm["d_dpm"]) == (7.03, 5.13, 1.9)

    wemby = rows[1]
    assert (wemby.nba_id, wemby.team_id, wemby.team, wemby.position) == (
        WEMBANYAMA,
        1610612759,
        "San Antonio Spurs",
        "F-C",
    )
    coward = rows[-1]
    assert (coward.name, coward.available, coward.minutes, coward.team) == (
        "Cedric Coward",
        False,
        0.0,
        "Memphis Grizzlies",
    )
    assert all(isinstance(row.nba_id, int) for row in rows)


def test_per_game_follows_minutes_pace_and_percentages(source: DarkoSource) -> None:
    wemby = next(row for row in source.projections().data if row.nba_id == WEMBANYAMA)
    possessions = 96.29 * 36.2 / 48
    assert wemby.possessions == pytest.approx(possessions)
    line = wemby.per_game()
    assert set(line) == {
        "min",
        "pts",
        "oreb",
        "dreb",
        "reb",
        "ast",
        "stl",
        "blk",
        "tov",
        "pf",
        "fga",
        "fgm",
        "fg3a",
        "fg3m",
        "fta",
        "ftm",
    }
    assert line["min"] == 36.2
    assert line["pts"] == pytest.approx(36.36 * possessions / 100)
    assert line["pts"] == pytest.approx(26.4, abs=0.05)  # a 26-a-night projection, in line with darko.app's table
    assert (
        line["reb"] == pytest.approx(line["oreb"] + line["dreb"]) == pytest.approx((3.45 + 13.44) * possessions / 100)
    )
    assert line["blk"] == pytest.approx(4.74 * possessions / 100)
    assert line["fgm"] == pytest.approx(line["fga"] * 0.50) and line["fga"] == pytest.approx(25.42 * possessions / 100)
    assert line["fg3m"] == pytest.approx(line["fg3a"] * 0.33) and line["ftm"] == pytest.approx(line["fta"] * 0.82)

    coward = next(row for row in source.projections().data if row.name == "Cedric Coward")
    assert coward.possessions == 0 and all(value == 0 for value in coward.per_game().values())


# --- daily ---


def test_daily_rows_are_typed(source: DarkoSource) -> None:
    rows = source.daily().data
    assert len(rows) == 12
    brunson = next(row for row in rows if row.nba_id == BRUNSON)
    assert (brunson.name, brunson.day, brunson.team, brunson.opponent) == (
        "Jalen Brunson",
        date(2026, 6, 5),
        "New York Knicks",
        "San Antonio Spurs",
    )
    assert brunson.game_id == "0042500402" and brunson.has_game and brunson.available  # zero-padded nba.com id
    assert brunson.minutes == 39.01 and brunson.stats["min"] == 39.01
    assert (brunson.stats["pts"], brunson.stats["ast"], brunson.stats["fg3m"]) == (26.64, 6.39, 2.31)
    assert brunson.stats["reb"] == pytest.approx(0.53 + 3.03)
    assert set(brunson.stats) == {"min", "pts", "oreb", "dreb", "reb", "ast", "stl", "blk", "fg3a", "fg3m"}

    sga = next(row for row in rows if row.nba_id == SGA)
    assert (sga.day, sga.team, sga.opponent, sga.game_id, sga.has_game) == (date(2026, 6, 4), None, None, None, False)
    assert sga.minutes == 42.31 and sga.stats["pts"] == 29.89


# --- caching and failure ---


def test_csv_is_captured_and_cached(source: DarkoSource, routes: Routes, clock: FakeClock, tmp_path: Path) -> None:
    first = source.projections()
    raw = tmp_path / "cache" / "darko" / "talent" / "current.csv"
    assert first.raw_path == raw and raw.read_bytes() == fixture("talent.csv")
    assert (first.source, first.dataset, first.key, first.as_of) == ("darko", "talent", "current", T0)
    clock.advance(hours=5)
    assert source.projections().cached is True and routes.talent.call_count == 1
    clock.advance(hours=2)
    assert source.projections().cached is False and routes.talent.call_count == 2
    assert routes.talent.calls.last.request.headers["user-agent"].startswith("espn-fantasy/")
    assert source.daily().raw_path == tmp_path / "cache" / "darko" / "daily" / "current.csv"
    assert DarkoSource.ttl["daily"] < DarkoSource.ttl["talent"]


def test_google_error_page_is_rejected_and_the_good_copy_kept(
    source: DarkoSource, routes: Routes, clock: FakeClock, tmp_path: Path
) -> None:
    routes.talent.mock(return_value=httpx.Response(200, content=LOGIN_PAGE, headers={"content-type": "text/html"}))
    with pytest.raises(SourceSchemaError, match=r"missing columns \['nba_id'.*\(not a CSV\?\)"):
        source.projections()
    cache = tmp_path / "cache" / "darko" / "talent"
    assert (cache / "current.rejected.csv").read_bytes() == LOGIN_PAGE and not (cache / "current.csv").exists()

    routes.talent.mock(return_value=ok("talent.csv"))
    good = source.projections()
    clock.advance(hours=7)
    routes.talent.mock(return_value=httpx.Response(200, content=LOGIN_PAGE, headers={"content-type": "text/html"}))
    stale = source.projections()
    assert (stale.stale, stale.cached, stale.as_of) == (True, True, T0) and stale.data == good.data
    assert "did not parse" in stale.warnings[0] and (cache / "current.csv").read_bytes() == fixture("talent.csv")


def test_urls_are_configurable(clock: FakeClock, tmp_path: Path) -> None:
    mirror = "https://mirror.test/darko/talent.csv"
    with respx.mock(assert_all_called=True) as router:
        route = router.get(mirror).mock(return_value=ok("talent.csv"))
        source = DarkoSource(talent_url=mirror, cache_root=tmp_path / "cache", limiter=RateLimiter(0), clock=clock)
        assert len(source.projections().data) == 13 and route.call_count == 1


# --- parsing ---


def test_parse_skips_bad_rows_and_rejects_bad_sheets(caplog: pytest.LogCaptureFixture) -> None:
    header = (
        "nba_id,player_name,position,available,tm_id,team_name,minutes,pace,fg_pct,fg3_pct,ft_pct,"
        "pts_100,orb_100,drb_100,ast_100,stl_100,blk_100,tov_100,fga_100,fg3a_100,fta_100\n"
    )
    good = (
        "203999.00,Nikola Jokic,C-F,1,-999,,29.30,94.42,0.53,0.34,0.79,"
        "31.73,3.75,13.50,10.67,1.90,1.23,5.02,24.12,6.60,8.85\n"
    )
    bad = "abc,Nobody,G,1,-999,,10,100,0.5,0.3,0.8,1,1,1,1,1,1,1,1,1,1\n"
    blank_minutes = "2,Blank Minutes,G,1,-999,,,100,0.5,0.3,0.8,1,1,1,1,1,1,1,1,1,1\n"
    with caplog.at_level(logging.WARNING, logger="fm.sources.darko"):
        (jokic,) = parse_talent((header + good + bad + blank_minutes).encode())
    assert jokic.nba_id == JOKIC and jokic.per_100 == {
        "pts": 31.73,
        "orb": 3.75,
        "drb": 13.5,
        "ast": 10.67,
        "stl": 1.9,
        "blk": 1.23,
        "tov": 5.02,
        "fta": 8.85,
        "fga": 24.12,
        "fg3a": 6.6,
    }
    assert jokic.dpm == {} and jokic.x_position is None
    assert "skipped 2 of 3 rows" in caplog.text
    with pytest.raises(SourceSchemaError, match="none of 1 rows validated"):
        parse_talent((header + bad).encode())
    assert parse_talent(header.encode()) == []
    with pytest.raises(SourceSchemaError, match=r"missing columns \['pace'\]"):
        parse_talent(header.replace("pace", "tempo").encode())
    with pytest.raises(SourceSchemaError, match="not a CSV"):
        parse_talent(b"")


def test_parse_daily_handles_sentinels_and_padding() -> None:
    header = (
        "nba_id,date,player_name,team_name,opp_tm_name,game_id,available,minutes,pts,blk,orb,drb,ast,stl,fg3a,fg3m\n"
    )
    with_game = (
        "1628973,2026-06-05,Jalen Brunson,New York Knicks,San Antonio Spurs,42500402,1.00,"
        "39.01,26.64,0.06,0.53,3.03,6.39,0.82,6.89,2.31\n"
    )
    no_game = (
        "1628983,2026-06-04 0:00:00,Shai Gilgeous-Alexander,,,-999,0,42.31,29.89,1.03,0.84,3.92,7.40,1.71,5.00,1.78\n"
    )
    brunson, sga = parse_daily((header + with_game + no_game).encode("utf-8-sig"))
    assert (brunson.game_id, brunson.available, brunson.day) == ("0042500402", True, date(2026, 6, 5))
    assert (sga.game_id, sga.available, sga.day, sga.team) == (None, False, date(2026, 6, 4), None)
    with pytest.raises(SourceSchemaError, match=r"missing columns \['fg3m'\]"):
        parse_daily(header.replace("fg3m", "threes").encode())
