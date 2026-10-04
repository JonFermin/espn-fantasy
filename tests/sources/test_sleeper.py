"""Sleeper adapter against recorded responses (respx; no network).

Fixtures under tests/fixtures/sources/sleeper were recorded on 2026-10-04 (NFL week 4) and trimmed to a dozen players.
``projections_legacy_junk.json`` is what the broken ``api.sleeper.app/v1/projections`` endpoint answered that day.
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

from fm.sources.base import RateLimiter, SourceSchemaError, SourceUnavailable
from fm.sources.sleeper import (
    SleeperPlayer,
    SleeperSnapCount,
    SleeperSource,
    SleeperStatLine,
    epoch_ms,
    parse_players,
    parse_stat_lines,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "sources" / "sleeper"
T0 = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)
V1 = "api.sleeper.app"
DATA = "api.sleeper.com"


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
        self.state = router.get(host=V1, path="/v1/state/nfl").mock(return_value=ok("state_nfl.json"))
        self.players = router.get(host=V1, path="/v1/players/nfl").mock(return_value=ok("players_nfl.json"))
        self.add = router.get(host=V1, path="/v1/players/nfl/trending/add").mock(return_value=ok("trending_add.json"))
        self.drop = router.get(host=V1, path="/v1/players/nfl/trending/drop").mock(
            return_value=ok("trending_drop.json")
        )
        self.projections = router.get(host=DATA, path="/projections/nfl/2026/4").mock(
            return_value=ok("projections_2026_4.json")
        )
        self.stats = router.get(host=DATA, path="/stats/nfl/2026/4").mock(return_value=ok("stats_2026_4.json"))


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
def source(routes: Routes, clock: FakeClock, tmp_path: Path, sleeps: list[float]) -> Iterator[SleeperSource]:
    with SleeperSource(cache_root=tmp_path / "cache", limiter=RateLimiter(0), clock=clock, sleep=sleeps.append) as src:
        yield src


# --- documented endpoints ---


def test_state(source: SleeperSource) -> None:
    state = source.state().data
    assert (state.season, state.week, state.season_type) == (2026, 4, "regular")
    assert state.season_start_date == date(2026, 9, 9)


def test_players_are_typed_and_cleaned(source: SleeperSource) -> None:
    players = source.players().data
    assert len(players) == 12
    assert all(player.player_id == key for key, player in players.items())
    allen = players["4984"]
    assert (allen.name, allen.position, allen.team) == ("Josh Allen", "QB", "BUF")
    assert (allen.espn_id, allen.gsis_id, allen.fantasy_positions) == (3918298, "00-0034857", ["QB"])
    assert allen.news_updated is not None and allen.news_updated.tzinfo is UTC and allen.news_updated.year == 2026
    chase = players["7564"]
    assert (chase.injury_status, chase.injury_body_part, chase.espn_id) == ("Out", "Concussion", None)
    eagles = players["PHI"]
    assert (eagles.position, eagles.name, eagles.team, eagles.full_name) == ("DEF", "Philadelphia Eagles", "PHI", None)
    for player in players.values():
        assert player.gsis_id is None or player.gsis_id == player.gsis_id.strip()
        assert player.injury_status != ""


def test_players_strip_padded_gsis_ids() -> None:
    raw = {"1": {"player_id": "1", "gsis_id": " 00-0035057", "injury_status": "", "fantasy_positions": None}}
    (player,) = parse_players(json.dumps(raw).encode()).values()
    assert (player.gsis_id, player.injury_status, player.fantasy_positions, player.active) == (
        "00-0035057",
        None,
        [],
        False,
    )


def test_players_are_cached_and_captured(
    source: SleeperSource, routes: Routes, clock: FakeClock, tmp_path: Path
) -> None:
    first = source.players()
    clock.advance(hours=23)
    again = source.players()
    assert routes.players.call_count == 1
    assert again.cached is True and again.as_of == first.as_of == T0
    raw = tmp_path / "cache" / "sleeper" / "players" / "nfl.json"
    assert first.raw_path == raw and raw.read_bytes() == fixture("players_nfl.json")
    meta = json.loads((raw.parent / "nfl.meta.json").read_text(encoding="utf-8"))
    assert meta["url"] == "https://api.sleeper.app/v1/players/nfl"
    assert routes.players.calls.last.request.headers["user-agent"].startswith("espn-fantasy/")


def test_players_skip_bad_entries_but_reject_an_empty_db(caplog: pytest.LogCaptureFixture) -> None:
    raw = json.loads(fixture("players_nfl.json"))
    raw["bogus"] = {"espn_id": "not-a-number"}
    raw["junk"] = "string"
    with caplog.at_level(logging.WARNING, logger="fm.sources.sleeper"):
        players = parse_players(json.dumps(raw).encode())
    assert len(players) == 12 and "skipped 2 of 14" in caplog.text
    for payload in (b"{}", b"[]", b'{"1": "x"}'):
        with pytest.raises(SourceSchemaError):
            parse_players(payload)


def test_documented_endpoint_failure_raises(source: SleeperSource, routes: Routes) -> None:
    routes.players.mock(return_value=httpx.Response(500))
    with pytest.raises(SourceUnavailable, match="HTTP 500"):
        source.players()


def test_trending_is_sorted_and_parameterized(source: SleeperSource, routes: Routes) -> None:
    adds = source.trending("add").data
    assert [entry.player_id for entry in adds][:2] == ["11435", "7049"]
    assert [entry.count for entry in adds] == sorted((entry.count for entry in adds), reverse=True)
    assert dict(routes.add.calls.last.request.url.params) == {"lookback_hours": "24", "limit": "25"}

    drops = source.trending("drop", lookback_hours=6, limit=5)
    assert drops.data[0].player_id == "13286" and drops.key == "drop_6h_5"
    assert dict(routes.drop.calls.last.request.url.params) == {"lookback_hours": "6", "limit": "5"}


# --- undocumented endpoints ---


def test_projections_are_stat_lines_without_placeholders(source: SleeperSource, routes: Routes) -> None:
    result = source.projections(2026, 4)
    lines = result.data
    assert len(lines) == 10 and result.degraded is False  # the fixture holds 10 projections + 2 ADP-only entries
    assert {line.category for line in lines} == {"proj"} and {line.company for line in lines} == {"rotowire"}
    assert {(line.season, line.week, line.season_type) for line in lines} == {(2026, 4, "regular")}
    gibbs = next(line for line in lines if line.player_id == "9221")
    assert (gibbs.position, gibbs.team, gibbs.opponent, gibbs.game_id) == ("RB", "DET", "CAR", "202610405")
    assert gibbs.game_date == date(2026, 10, 4)
    assert gibbs.stats["rush_yd"] == 94.01 and gibbs.stats["rec"] == 4.95 and gibbs.stats["pts_ppr"] == 24.8
    assert gibbs.updated_at is not None and gibbs.updated_at.tzinfo is UTC and gibbs.is_placeholder is False
    assert all(isinstance(value, float) for line in lines for value in line.stats.values())

    params = routes.projections.calls.last.request.url.params
    assert params["season_type"] == "regular" and params["order_by"] == "ppr"
    assert params.get_list("position[]") == ["QB", "RB", "WR", "TE", "K", "DEF"]
    assert routes.projections.calls.last.request.url.host == "api.sleeper.com"


def test_placeholder_detection() -> None:
    adp_only = SleeperStatLine(player_id="1", season=2026, week=4, category="proj", stats={"adp_dd_ppr": 1000.0})
    assert adp_only.is_placeholder is True
    assert parse_stat_lines(json.dumps([adp_only.model_dump(by_alias=True)]).encode(), drop_placeholders=True) == []
    assert (
        len(parse_stat_lines(json.dumps([adp_only.model_dump(by_alias=True)]).encode(), drop_placeholders=False)) == 1
    )


def test_week_stats_and_same_day_snaps(source: SleeperSource, routes: Routes) -> None:
    stats = source.week_stats(2026, 4)
    assert len(stats.data) == 7 and {line.category for line in stats.data} == {"stat"}
    snaps = source.snap_counts(2026, 4)
    assert routes.stats.call_count == 1  # both views share one cached payload
    (judkins,) = snaps.data
    assert isinstance(judkins, SleeperSnapCount)
    assert (judkins.player_id, judkins.team, judkins.opponent, judkins.position) == ("12512", "CLE", "PIT", "RB")
    assert (judkins.offensive_snaps, judkins.team_offensive_snaps) == (42.0, 64.0)
    assert judkins.snap_share == pytest.approx(42 / 64)
    assert judkins.game_date == date(2026, 10, 1) and (snaps.as_of, snaps.cached) == (T0, True)


def test_snap_count_requires_snaps() -> None:
    line = SleeperStatLine(player_id="1", season=2026, week=4, category="stat", stats={"rec": 3.0})
    assert SleeperSnapCount.from_line(line) is None
    partial = SleeperStatLine(player_id="1", season=2026, week=4, category="stat", stats={"off_snp": 10.0})
    snap = SleeperSnapCount.from_line(partial)
    assert snap is not None and snap.team_offensive_snaps is None and snap.snap_share is None


def test_projections_degrade_on_server_errors(
    source: SleeperSource, routes: Routes, sleeps: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    routes.projections.mock(return_value=httpx.Response(500))
    with caplog.at_level(logging.WARNING, logger="fm.sources.sleeper"):
        result = source.projections(2026, 4)
    assert result.degraded is True and result.data == [] and result.as_of == T0
    assert "HTTP 500" in result.warnings[0] and "continuing without it" in caplog.text
    assert routes.projections.call_count == 3 and sleeps == [1.0, 2.0]  # retried, then gave up


def test_projections_degrade_when_the_shape_changes(source: SleeperSource, routes: Routes, tmp_path: Path) -> None:
    # The legacy endpoint "broke" by answering 200 with a dict of ADP junk; the new one must not be trusted blindly.
    routes.projections.mock(return_value=ok("projections_legacy_junk.json"))
    result = source.projections(2026, 4)
    assert result.degraded is True and result.data == []
    assert "expected a JSON list" in result.warnings[0]
    rejected = tmp_path / "cache" / "sleeper" / "projections" / "regular_2026_w4.rejected.json"
    assert rejected.read_bytes() == fixture("projections_legacy_junk.json")
    assert not (rejected.parent / "regular_2026_w4.json").exists()


def test_projections_serve_the_last_good_copy_when_the_endpoint_breaks(
    source: SleeperSource, routes: Routes, clock: FakeClock
) -> None:
    good = source.projections(2026, 4)
    clock.advance(minutes=31)
    routes.projections.mock(return_value=ok("projections_legacy_junk.json"))
    stale = source.projections(2026, 4)
    assert (stale.stale, stale.degraded, stale.cached, stale.as_of) == (True, False, True, T0)
    assert [line.player_id for line in stale.data] == [line.player_id for line in good.data]
    assert stale.warnings and "did not parse" in stale.warnings[0]


def test_snap_counts_degrade_with_week_stats(source: SleeperSource, routes: Routes) -> None:
    routes.stats.mock(return_value=httpx.Response(503))
    result = source.snap_counts(2026, 4)
    assert result.degraded is True and result.data == [] and result.warnings


def test_accidental_live_calls_are_not_swallowed(source: SleeperSource, routes: Routes) -> None:
    # The harness blocks the network with a bare RuntimeError; graceful degradation must never hide one.
    routes.projections.mock(side_effect=RuntimeError("outbound network is disabled in unit tests"))
    with pytest.raises(RuntimeError, match="outbound network"):
        source.projections(2026, 4)


def test_stat_line_parse_rejects_everything_invalid() -> None:
    with pytest.raises(SourceSchemaError, match="expected a JSON list"):
        parse_stat_lines(b'{"a": 1}', drop_placeholders=True)
    with pytest.raises(SourceSchemaError, match="expected objects"):
        parse_stat_lines(b"[1, 2]", drop_placeholders=True)
    with pytest.raises(SourceSchemaError, match="none of 2 entries validated"):
        parse_stat_lines(b'[{"player_id": "1"}, {"week": 4}]', drop_placeholders=True)
    assert parse_stat_lines(b"[]", drop_placeholders=True) == []


def test_ttls_follow_sleepers_guidance() -> None:
    assert SleeperSource.ttl["players"] == timedelta(hours=24)
    assert SleeperSource.ttl["stats"] <= timedelta(minutes=10) < SleeperSource.ttl["projections"]


# --- helpers ---


def test_epoch_ms_handles_sleepers_timestamp_shapes() -> None:
    expected = datetime.fromtimestamp(1791144019.647, UTC)
    assert epoch_ms(1791144019647) == expected
    assert epoch_ms("1791144019647") == expected
    assert epoch_ms(1791144019) == expected.replace(microsecond=0)
    assert epoch_ms(expected) == expected
    assert epoch_ms(expected.replace(tzinfo=None)) == expected
    assert epoch_ms("2026-10-04T20:00:19.647+00:00") == expected
    assert epoch_ms(None) is None and epoch_ms("") is None
    with pytest.raises(ValueError):
        epoch_ms(True)
    with pytest.raises(ValueError):
        epoch_ms("tomorrow")


def test_player_model_ignores_unknown_fields() -> None:
    player = SleeperPlayer.model_validate({"player_id": "9", "new_field": 1, "espn_id": "123", "age": None})
    assert player.espn_id == 123 and player.age is None and player.name == ""
