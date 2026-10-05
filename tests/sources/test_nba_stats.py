"""stats.nba.com adapter against recorded payloads (no network).

``nba_api`` fetches with ``requests``, so respx cannot intercept it. The adapter's transport seam serves the payloads
under tests/fixtures/sources/nba_stats by endpoint name; the default ``NbaApiTransport`` is exercised by faking the
``requests`` session beneath ``nba_api``, which checks the full header set, parameter sorting and timeout without a
socket. stats.nba.com answered only Akamai read timeouts while the fixtures were built (2026-10-04), so the payloads
are hand-built in the documented ``resultSets`` shape using ``nba_api``'s own header lists.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
import requests
from nba_api.stats.library.http import NBAStatsHTTP

from fm.sources.base import RateLimiter, SourceSchemaError, SourceUnavailable
from fm.sources.nba_stats import (
    FULL_HEADERS,
    NbaApiTransport,
    NbaStatsSource,
    nba_season,
    on_off_frame,
    parse_result_sets,
    pick_result_set,
    stats_date,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "sources" / "nba_stats"
T0 = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)
SEASON = "2025-26"
OKC = 1610612760
JOKIC, SGA, WEMBANYAMA = 203999, 1628983, 1641705


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


class FakeClock:
    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> datetime:
        self.now += timedelta(**delta)
        return self.now


class RecordedTransport:
    """Serves the fixtures by endpoint (and MeasureType for the dashboard endpoint) and records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.override: bytes | Exception | None = None

    def get(self, endpoint: str, parameters: Mapping[str, object]) -> bytes:
        self.calls.append((endpoint, dict(parameters)))
        if isinstance(self.override, Exception):
            raise self.override
        if self.override is not None:
            return self.override
        if endpoint == "leaguegamelog":
            return fixture(f"leaguegamelog_{parameters['Season']}.json")
        if endpoint == "leaguedashplayerstats":
            return fixture(f"leaguedashplayerstats_{parameters['MeasureType']}_{parameters['Season']}.json")
        if endpoint == "teamplayeronoffdetails":
            return fixture(f"teamplayeronoffdetails_OKC_{parameters['Season']}.json")
        raise AssertionError(f"no recorded fixture for {endpoint}")


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def transport() -> RecordedTransport:
    return RecordedTransport()


@pytest.fixture
def source(transport: RecordedTransport, clock: FakeClock, tmp_path: Path) -> NbaStatsSource:
    return NbaStatsSource(transport=transport, cache_root=tmp_path / "cache", limiter=RateLimiter(0), clock=clock)


# --- datasets ---


def test_game_logs_are_a_frame_with_stats_nba_columns(
    source: NbaStatsSource, transport: RecordedTransport, tmp_path: Path
) -> None:
    result = source.game_logs(SEASON, date_from=date(2026, 4, 10), date_to=date(2026, 4, 12))
    frame = result.data
    assert frame.height == 6
    assert frame.columns[:9] == [
        "SEASON_ID",
        "PLAYER_ID",
        "PLAYER_NAME",
        "TEAM_ID",
        "TEAM_ABBREVIATION",
        "TEAM_NAME",
        "GAME_ID",
        "GAME_DATE",
        "MATCHUP",
    ]
    assert frame.schema["PLAYER_ID"] == pl.Int64
    assert frame.schema["FG_PCT"] == pl.Float64 and frame.schema["FANTASY_PTS"] == pl.Float64
    assert frame.schema["GAME_DATE"] == pl.String and frame.schema["MIN"] == pl.Int64
    jokic = frame.filter(pl.col("PLAYER_ID") == JOKIC).sort("GAME_DATE")
    assert jokic["PTS"].to_list() == [30, 20] and jokic["MATCHUP"].to_list() == ["DEN vs. OKC", "DEN @ UTA"]

    ((endpoint, params),) = transport.calls
    assert endpoint == "leaguegamelog"
    assert (params["PlayerOrTeam"], params["Season"], params["SeasonType"]) == ("P", SEASON, "Regular Season")
    assert (params["DateFrom"], params["DateTo"]) == ("04/10/2026", "04/12/2026")

    assert (result.source, result.dataset, result.as_of, result.cached) == ("nba_stats", "game_logs", T0, False)
    key = "2025-26_Regular_Season_P_2026-04-10_2026-04-12"
    raw = tmp_path / "cache" / "nba_stats" / "game_logs" / f"{key}.json"
    assert result.raw_path == raw and raw.read_bytes() == fixture("leaguegamelog_2025-26.json")
    meta = json.loads((raw.parent / f"{key}.meta.json").read_text(encoding="utf-8"))
    assert (meta["endpoint"], meta["DateFrom"], meta["PlayerOrTeam"]) == ("leaguegamelog", "04/10/2026", "P")


def test_team_game_logs_use_the_same_endpoint(source: NbaStatsSource, transport: RecordedTransport) -> None:
    source.game_logs(SEASON, player_or_team="T", season_type="Playoffs")
    ((_, params),) = transport.calls
    assert (params["PlayerOrTeam"], params["SeasonType"], params["DateFrom"]) == ("T", "Playoffs", "")


def test_player_splits_select_the_measure_type(source: NbaStatsSource, transport: RecordedTransport) -> None:
    base = source.player_splits(SEASON).data
    advanced = source.player_splits(SEASON, "Advanced").data
    usage = source.player_splits(SEASON, "Usage", per_mode="Totals", last_n_games=5).data
    assert {"NBA_FANTASY_PTS", "DD2", "TD3", "PLUS_MINUS_RANK", "CFPARAMS"} <= set(base.columns)
    assert {"USG_PCT", "PACE", "POSS", "PIE", "TS_PCT", "E_NET_RATING"} <= set(advanced.columns)
    assert {"PCT_FGA", "PCT_AST", "PCT_PTS"} <= set(usage.columns)
    assert base.filter(pl.col("PLAYER_ID") == WEMBANYAMA)["BLK"].to_list() == [3.8]
    assert advanced.filter(pl.col("PLAYER_ID") == SGA)["USG_PCT"].to_list() == [0.343]
    assert usage["PLAYER_ID"].to_list() == [JOKIC, SGA, WEMBANYAMA]

    assert [params["MeasureType"] for _, params in transport.calls] == ["Base", "Advanced", "Usage"]
    assert [params["PerMode"] for _, params in transport.calls] == ["PerGame", "PerGame", "Totals"]
    assert [params["LastNGames"] for _, params in transport.calls] == ["0", "0", "5"]


def test_split_variants_cache_separately(source: NbaStatsSource, transport: RecordedTransport, tmp_path: Path) -> None:
    base = source.player_splits(SEASON)
    advanced = source.player_splits(SEASON, "Advanced")
    cache = tmp_path / "cache" / "nba_stats" / "player_splits"
    assert base.raw_path == cache / "2025-26_Regular_Season_Base_PerGame_last0.json"
    assert advanced.raw_path == cache / "2025-26_Regular_Season_Advanced_PerGame_last0.json"
    assert source.player_splits(SEASON, "Advanced").cached is True and len(transport.calls) == 2


def test_on_off_stacks_on_and_off_court_rows(source: NbaStatsSource, transport: RecordedTransport) -> None:
    result = source.on_off(OKC, SEASON)
    frame = result.data
    assert frame.height == 4
    assert frame["VS_PLAYER_ID"].to_list() == [SGA, SGA, 1631096, 1631096]
    assert frame["COURT_STATUS"].to_list() == ["On", "Off", "On", "Off"]
    sga_on = frame.filter((pl.col("VS_PLAYER_ID") == SGA) & (pl.col("COURT_STATUS") == "On"))
    assert sga_on["PTS"].to_list() == [86.4] and sga_on["GROUP_SET"].to_list() == ["On Court"]
    assert frame["TEAM_ABBREVIATION"].unique().to_list() == ["OKC"]

    ((endpoint, params),) = transport.calls
    assert endpoint == "teamplayeronoffdetails"
    assert (params["TeamID"], params["MeasureType"], params["PerMode"]) == (OKC, "Base", "PerGame")
    assert result.key == "1610612760_2025-26_Regular Season_Base_PerGame"


# --- caching, pacing, failures ---


def test_second_call_within_ttl_is_cached(
    source: NbaStatsSource, transport: RecordedTransport, clock: FakeClock
) -> None:
    first = source.game_logs(SEASON)
    clock.advance(hours=5)
    again = source.game_logs(SEASON)
    assert len(transport.calls) == 1
    assert again.cached is True and again.as_of == first.as_of == T0 and again.data.equals(first.data)
    later = clock.advance(hours=2)
    refreshed = source.game_logs(SEASON)
    assert refreshed.cached is False and refreshed.as_of == later and len(transport.calls) == 2


def test_default_pacing_and_transport(clock: FakeClock, tmp_path: Path) -> None:
    source = NbaStatsSource(cache_root=tmp_path / "cache", clock=clock)
    assert source.limiter.min_interval == 0.6
    assert isinstance(source.transport, NbaApiTransport)
    assert source.transport.timeout == 30.0 and source.transport.headers == FULL_HEADERS
    assert NbaStatsSource.ttl["game_logs"] <= timedelta(hours=6) < NbaStatsSource.ttl["on_off"]


def test_downloads_are_paced_but_cache_hits_are_not(clock: FakeClock, tmp_path: Path) -> None:
    limiter = RateLimiter(0.6, monotonic=lambda: 0.0, sleep=lambda _: None)
    source = NbaStatsSource(transport=RecordedTransport(), cache_root=tmp_path / "cache", limiter=limiter, clock=clock)
    for _ in range(3):
        source.game_logs(SEASON)
    assert limiter.calls == 1
    source.player_splits(SEASON)
    assert limiter.calls == 2


def test_block_page_is_kept_as_rejected_and_the_good_copy_survives(
    source: NbaStatsSource, transport: RecordedTransport, clock: FakeClock, tmp_path: Path
) -> None:
    good = source.game_logs(SEASON)
    clock.advance(hours=7)
    transport.override = fixture("access_denied.html")
    stale = source.game_logs(SEASON)
    assert (stale.stale, stale.cached, stale.as_of) == (True, True, T0) and stale.data.equals(good.data)
    assert "did not parse" in stale.warnings[0]
    cache = tmp_path / "cache" / "nba_stats" / "game_logs"
    assert (cache / "2025-26_Regular_Season_P.rejected.json").read_bytes() == fixture("access_denied.html")
    assert (cache / "2025-26_Regular_Season_P.json").read_bytes() == fixture("leaguegamelog_2025-26.json")


def test_block_page_without_a_good_copy_is_a_schema_error(source: NbaStatsSource, transport: RecordedTransport) -> None:
    transport.override = fixture("access_denied.html")
    with pytest.raises(SourceSchemaError, match="did not parse"):
        source.game_logs(SEASON)


def test_timeout_without_a_good_copy_raises_unavailable(source: NbaStatsSource, transport: RecordedTransport) -> None:
    # requests' timeouts are IOErrors, which the base source treats as recoverable (stale copy when there is one).
    transport.override = requests.exceptions.ReadTimeout("Read timed out. (read timeout=30)")
    with pytest.raises(SourceUnavailable, match="Read timed out"):
        source.game_logs(SEASON)


def test_accidental_live_calls_are_not_swallowed(source: NbaStatsSource, transport: RecordedTransport) -> None:
    transport.override = RuntimeError("outbound network is disabled in unit tests")
    with pytest.raises(RuntimeError, match="outbound network"):
        source.game_logs(SEASON)


def test_shape_change_in_a_fresh_payload_is_a_schema_error(
    source: NbaStatsSource, transport: RecordedTransport
) -> None:
    payload = json.loads(fixture("leaguegamelog_2025-26.json"))
    headers = payload["resultSets"][0]["headers"]
    headers[headers.index("PTS")] = "POINTS"
    transport.override = json.dumps(payload).encode("utf-8")
    with pytest.raises(SourceSchemaError, match=r"missing columns \['PTS'\]"):
        source.game_logs(SEASON)


# --- parsing helpers ---


def test_result_set_parser_rejects_bad_shapes() -> None:
    with pytest.raises(SourceSchemaError, match="expected a JSON object"):
        parse_result_sets(b"[]")
    with pytest.raises(SourceSchemaError, match="no resultSets"):
        parse_result_sets(b'{"resource": "x"}')
    with pytest.raises(SourceSchemaError, match="lacks name, headers or rowSet"):
        parse_result_sets(b'{"resultSets": [{"headers": [], "rowSet": []}]}')
    with pytest.raises(SourceSchemaError, match="does not match its 2 headers"):
        parse_result_sets(b'{"resultSets": [{"name": "A", "headers": ["X", "Y"], "rowSet": [[1]]}]}')
    with pytest.raises(ValueError):
        parse_result_sets(fixture("access_denied.html"))


def test_legacy_single_result_set_and_empty_rows_parse() -> None:
    frames = parse_result_sets(b'{"resultSet": {"name": "A", "headers": ["PLAYER_ID", "MIN"], "rowSet": []}}')
    assert frames["A"].columns == ["PLAYER_ID", "MIN"] and frames["A"].height == 0


def test_mixed_int_and_float_columns_infer_floats() -> None:
    payload = (
        b'{"resultSets": [{"name": "A", "headers": ["MIN", "WL"], "rowSet": [[34, "W"], [31.5, null], [null, "L"]]}]}'
    )
    frame = parse_result_sets(payload)["A"]
    assert frame.schema["MIN"] == pl.Float64 and frame["MIN"].to_list() == [34.0, 31.5, None]


def test_pick_result_set_requires_columns() -> None:
    frames = parse_result_sets(fixture("leaguegamelog_2025-26.json"))
    assert pick_result_set(frames, "LeagueGameLog", ("PLAYER_ID",)).height == 6
    with pytest.raises(SourceSchemaError, match=r"missing columns \['NOPE'\]"):
        pick_result_set(frames, "LeagueGameLog", ("NOPE",))
    with pytest.raises(SourceSchemaError, match="result set 'Other' missing"):
        pick_result_set(frames, "Other", ())


def test_on_off_frame_needs_both_court_sets() -> None:
    frames = parse_result_sets(fixture("teamplayeronoffdetails_OKC_2025-26.json"))
    assert set(frames) == {
        "OverallTeamPlayerOnOffDetails",
        "PlayersOnCourtTeamPlayerOnOffDetails",
        "PlayersOffCourtTeamPlayerOnOffDetails",
    }
    del frames["PlayersOffCourtTeamPlayerOnOffDetails"]
    with pytest.raises(SourceSchemaError, match="PlayersOffCourtTeamPlayerOnOffDetails"):
        on_off_frame(frames)


def test_season_and_date_helpers() -> None:
    assert nba_season(2027) == "2026-27" and nba_season(2000) == "1999-00"
    with pytest.raises(ValueError):
        nba_season(1900)
    assert stats_date(date(2026, 4, 10)) == "04/10/2026" and stats_date(None) == ""


# --- the default transport ---


class FakeResponse:
    def __init__(self, url: str, text: str) -> None:
        self.url = url
        self.status_code = 200
        self.text = text


class FakeSession:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def get(
        self, url: str, params: object = None, headers: object = None, proxies: object = None, timeout: object = None
    ):
        self.calls.append({"url": url, "params": params, "headers": headers, "proxies": proxies, "timeout": timeout})
        return FakeResponse(url, fixture("leaguegamelog_2025-26.json").decode("utf-8"))


def test_default_transport_sends_the_full_header_set_through_nba_api(monkeypatch: pytest.MonkeyPatch) -> None:
    session = FakeSession()
    monkeypatch.setattr(NBAStatsHTTP, "get_session", classmethod(lambda cls: session))
    transport = NbaApiTransport(timeout=12.5)
    body = transport.get("leaguegamelog", {"Season": SEASON, "PlayerOrTeam": "P", "Counter": 0})
    assert body == fixture("leaguegamelog_2025-26.json")
    (call,) = session.calls
    assert call["url"] == "https://stats.nba.com/stats/leaguegamelog" and call["timeout"] == 12.5
    assert call["params"] == [("Counter", 0), ("PlayerOrTeam", "P"), ("Season", SEASON)]  # nba_api sorts by key
    headers = call["headers"]
    assert isinstance(headers, dict) and headers == dict(FULL_HEADERS)
    for name in (
        "Host",
        "Origin",
        "Referer",
        "Accept-Language",
        "Sec-Ch-Ua",
        "x-nba-stats-origin",
        "x-nba-stats-token",
    ):
        assert name in headers
    assert "Chrome" in headers["User-Agent"] and headers["Host"] == "stats.nba.com"
    assert call["proxies"] is None
