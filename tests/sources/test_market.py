"""Market value adapter against recorded responses (respx; no network).

Fixtures under tests/fixtures/sources/market were recorded on 2026-10-04 and trimmed: 12 of FantasyCalc's 197 redraft
values (12 teams, 1 QB, full PPR) and 10 NFL / 7 NBA players from ESPN's public ``kona_player_info`` pool with the stat
arrays and outlook text removed. No cookies were sent or saved; the pool is ESPN's league-less default, so every entry
reads ``onTeamId: 0`` and no manager appears.
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

from fm.espn.ids import Game, InjuryStatus
from fm.sources.base import RateLimiter, SourceSchemaError
from fm.sources.market import (
    ESPN_FILTER_HEADER,
    FANTASYCALC_PPR,
    FANTASYCALC_QBS,
    FANTASYCALC_TEAMS,
    EspnOwnership,
    EspnPlayerMarket,
    FantasyCalcValue,
    LeagueShape,
    MarketSource,
    MarketValue,
    espn_filter,
    index_by_espn_id,
    nearest_shape,
    parse_espn_players,
    parse_fantasycalc,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "sources" / "market"
FC_FIXTURE = "fantasycalc_redraft_12t_1qb_ppr1.json"
FFL_FIXTURE = "espn_ffl_kona_player_info.json"
FBA_FIXTURE = "espn_fba_kona_player_info.json"
T0 = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)
FANTASYCALC = "api.fantasycalc.com"
ESPN_READS = "lm-api-reads.fantasy.espn.com"
ESPN_PATH = "/apis/v3/games/{game}/seasons/{season}/segments/0/leaguedefaults/3"

GIBBS = 4429795
CHASE = 4362628
ALLEN = 3918298
AUBREY = 3953687  # kicker: in ESPN's pool, never valued by FantasyCalc
BOWERS = 4432665  # valued by FantasyCalc, outside the recorded ESPN pool
WEMBANYAMA = 5104157
JOKIC = 3112335


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
        self.fantasycalc = router.get(host=FANTASYCALC, path="/values/current").mock(return_value=ok(FC_FIXTURE))
        self.espn_ffl = router.get(host=ESPN_READS, path=ESPN_PATH.format(game="ffl", season=2026)).mock(
            return_value=ok(FFL_FIXTURE)
        )
        self.espn_fba = router.get(host=ESPN_READS, path=ESPN_PATH.format(game="fba", season=2027)).mock(
            return_value=ok(FBA_FIXTURE)
        )

    @property
    def all(self) -> tuple[respx.Route, ...]:
        return (self.fantasycalc, self.espn_ffl, self.espn_fba)


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
def source(routes: Routes, clock: FakeClock, tmp_path: Path, sleeps: list[float]) -> Iterator[MarketSource]:
    with MarketSource(cache_root=tmp_path / "cache", limiter=RateLimiter(0), clock=clock, sleep=sleeps.append) as src:
        yield src


# --- FantasyCalc ---


def test_fantasycalc_values_are_typed_and_keyed_by_espn_id(source: MarketSource, routes: Routes) -> None:
    result = source.fantasycalc()
    values = result.data
    assert len(values) == 12 and result.degraded is False and result.key == "redraft_12t_1qb_ppr1"
    assert [value.overall_rank for value in values] == sorted(value.overall_rank for value in values)
    gibbs = values[0]
    assert isinstance(gibbs, FantasyCalcValue)
    assert (gibbs.espn_id, gibbs.sleeper_id, gibbs.mfl_id, gibbs.fantasycalc_id) == (GIBBS, "9221", "16162", 9821)
    assert (gibbs.name, gibbs.position, gibbs.team) == ("Jahmyr Gibbs", "RB", "DET")
    assert (gibbs.value, gibbs.overall_rank, gibbs.position_rank, gibbs.trend_30d, gibbs.tier) == (10697, 1, 1, 620, 1)
    assert gibbs.adp is None and gibbs.trade_frequency == 0.0095 and gibbs.roster_percent == 1.0
    chase = next(value for value in values if value.espn_id == CHASE)
    assert chase.trend_30d == -714 and chase.position_rank == 2
    by_id = index_by_espn_id(values)
    assert set(by_id) == {value.espn_id for value in values} and by_id[GIBBS] is gibbs
    request = routes.fantasycalc.calls.last.request
    assert dict(request.url.params) == {"isDynasty": "false", "numQbs": "1", "numTeams": "12", "ppr": "1"}
    assert request.headers["user-agent"].startswith("espn-fantasy/")


def test_fantasycalc_snaps_the_league_shape_onto_its_grid(source: MarketSource, routes: Routes) -> None:
    assert (FANTASYCALC_TEAMS, FANTASYCALC_PPR, FANTASYCALC_QBS) == ((8, 10, 12, 14), (0.0, 0.5, 1.0), (1, 2))
    assert nearest_shape(12, 1.0) == LeagueShape(12, 1.0, 1)
    assert nearest_shape(9, 0.5) == LeagueShape(10, 0.5, 1)  # ties go up
    assert nearest_shape(16, 1.0, 3) == LeagueShape(14, 1.0, 2)
    assert nearest_shape(8, 0.25, 0) == LeagueShape(8, 0.5, 1)
    assert LeagueShape(10, 0.5, 2).key == "redraft_10t_2qb_ppr0.5" and LeagueShape(8, 0.0).params()["ppr"] == "0"

    result = source.fantasycalc(num_teams=9, ppr=0.5, num_qbs=2)
    assert result.key == "redraft_10t_2qb_ppr0.5"
    params = dict(routes.fantasycalc.calls.last.request.url.params)
    assert params == {"isDynasty": "false", "numQbs": "2", "numTeams": "10", "ppr": "0.5"}


def test_fantasycalc_is_cached_and_captured(
    source: MarketSource, routes: Routes, clock: FakeClock, tmp_path: Path
) -> None:
    first = source.fantasycalc()
    clock.advance(hours=11)
    again = source.fantasycalc()
    assert routes.fantasycalc.call_count == 1
    assert again.cached is True and again.as_of == first.as_of == T0
    raw = tmp_path / "cache" / "market" / "fantasycalc" / "redraft_12t_1qb_ppr1.json"
    assert first.raw_path == raw and raw.read_bytes() == fixture(FC_FIXTURE)
    meta = json.loads((raw.parent / "redraft_12t_1qb_ppr1.meta.json").read_text(encoding="utf-8"))
    assert meta["url"] == "https://api.fantasycalc.com/values/current" and meta["numTeams"] == "12"
    clock.advance(hours=2)
    assert source.fantasycalc().cached is False and routes.fantasycalc.call_count == 2


def test_fantasycalc_skips_bad_entries_but_rejects_junk(caplog: pytest.LogCaptureFixture) -> None:
    raw = json.loads(fixture(FC_FIXTURE))
    raw.append({"player": {"id": 1, "name": "No Value", "position": "RB"}})
    raw.append("junk")
    with caplog.at_level(logging.WARNING, logger="fm.sources.market"):
        values = parse_fantasycalc(json.dumps(raw).encode())
    assert len(values) == 12 and "skipped 2 of 14" in caplog.text
    for payload in (b"[]", b"{}", b'{"values": []}', b"[1, 2]", b'[{"value": 5}]'):
        with pytest.raises(SourceSchemaError):
            parse_fantasycalc(payload)


def test_fantasycalc_tolerates_missing_or_odd_ids() -> None:
    entry = json.loads(fixture(FC_FIXTURE))[0]
    entry["player"]["espnId"] = ""
    entry["player"]["sleeperId"] = 9221
    entry["player"]["maybeTeam"] = None
    entry["trend30Day"] = None
    (value,) = parse_fantasycalc(json.dumps([entry]).encode())
    assert (value.espn_id, value.sleeper_id, value.team, value.trend_30d) == (None, "9221", None, None)
    entry["player"]["espnId"] = "n/a"
    (value,) = parse_fantasycalc(json.dumps([entry]).encode())
    assert value.espn_id is None and index_by_espn_id([value]) == {}


def test_index_keeps_the_better_ranked_duplicate() -> None:
    def valued(rank: int, value: int) -> FantasyCalcValue:
        fields = {"fantasycalc_id": rank, "name": "A", "position": "RB", "espn_id": 7, "position_rank": 1}
        return FantasyCalcValue.model_validate({**fields, "value": value, "overall_rank": rank})

    low, high = valued(50, 10), valued(5, 90)
    assert index_by_espn_id([low, high]) == {7: high}


def test_fantasycalc_degrades_on_server_errors(
    source: MarketSource, routes: Routes, sleeps: list[float], caplog: pytest.LogCaptureFixture
) -> None:
    routes.fantasycalc.mock(return_value=httpx.Response(500))
    with caplog.at_level(logging.WARNING, logger="fm.sources.market"):
        result = source.fantasycalc()
    assert (result.degraded, result.stale, result.data, result.as_of) == (True, False, [], T0)
    assert "HTTP 500" in result.warnings[0] and "continuing without it" in caplog.text
    assert routes.fantasycalc.call_count == 3 and sleeps == [1.0, 2.0]  # retried, then gave up


def test_fantasycalc_serves_the_last_good_copy_when_the_api_breaks(
    source: MarketSource, routes: Routes, clock: FakeClock, tmp_path: Path
) -> None:
    good = source.fantasycalc()
    clock.advance(hours=13)
    routes.fantasycalc.mock(return_value=httpx.Response(200, content=b'{"error": "maintenance"}'))
    stale = source.fantasycalc()
    assert (stale.stale, stale.degraded, stale.cached, stale.as_of) == (True, False, True, T0)
    assert [value.espn_id for value in stale.data] == [value.espn_id for value in good.data]
    assert "did not parse" in stale.warnings[0]
    cache = tmp_path / "cache" / "market" / "fantasycalc"
    assert (cache / "redraft_12t_1qb_ppr1.json").read_bytes() == fixture(FC_FIXTURE)
    assert (cache / "redraft_12t_1qb_ppr1.rejected.json").read_bytes() == b'{"error": "maintenance"}'


# --- ESPN public pool ---


def test_espn_nfl_pool_is_typed_with_ranks_and_ownership(source: MarketSource) -> None:
    result = source.espn_players("ffl", 2026)
    players = result.data
    assert len(players) == 10 and result.degraded is False and result.key == "ffl_2026_top300"
    gibbs = players[0]
    assert isinstance(gibbs, EspnPlayerMarket)
    assert (gibbs.espn_id, gibbs.name, gibbs.position, gibbs.pro_team) == (GIBBS, "Jahmyr Gibbs", "RB", "DET")
    assert (gibbs.position_id, gibbs.pro_team_id, gibbs.injured, gibbs.active) == (2, 8, False, True)
    assert gibbs.injury_status is InjuryStatus.ACTIVE
    assert gibbs.draft_ranks == {"STANDARD": 1, "PPR": 1, "ELIMINATION": 1, "SUPERFLEX": 7}
    assert (gibbs.positional_ranking, gibbs.total_ranking, gibbs.total_rating) == (1, 2, 98.3)
    ownership = gibbs.ownership
    assert isinstance(ownership, EspnOwnership)
    assert (ownership.percent_owned, ownership.percent_started, ownership.percent_change) == (99.95, 99.86, 0.0)
    assert (ownership.average_draft_position, ownership.adp_percent_change) == (1.69, 0.01)
    assert (ownership.auction_value_average, ownership.auction_value_change) == (73.38, 0.75)
    assert ownership.league_count is None
    assert ownership.updated_at == datetime.fromtimestamp(1791160220436 / 1000, UTC)
    assert ownership.updated_at is not None and ownership.updated_at.tzinfo is UTC

    chase = next(player for player in players if player.espn_id == CHASE)
    assert chase.injury_status is InjuryStatus.QUESTIONABLE and chase.injured is False
    assert (chase.draft_ranks["PPR"], chase.draft_ranks["STANDARD"]) == (3, 4)
    allen = next(player for player in players if player.espn_id == ALLEN)
    assert (allen.position, allen.draft_ranks["SUPERFLEX"], allen.draft_ranks["PPR"]) == ("QB", 1, 26)
    aubrey = next(player for player in players if player.espn_id == AUBREY)
    assert (aubrey.position, aubrey.pro_team, aubrey.total_ranking, aubrey.draft_ranks["PPR"]) == ("K", "DAL", 123, 255)


def test_espn_nba_pool_uses_nba_ids_and_rank_types(source: MarketSource, routes: Routes) -> None:
    result = source.espn_players(Game.FBA, 2027)
    players = result.data
    assert len(players) == 7 and result.key == "fba_2027_top300"
    wembanyama = players[0]
    assert (wembanyama.espn_id, wembanyama.name, wembanyama.position, wembanyama.pro_team) == (
        WEMBANYAMA,
        "Victor Wembanyama",
        "C",
        "SAS",
    )
    assert wembanyama.draft_ranks == {"STANDARD": 4, "ROTO": 2}
    assert (wembanyama.ownership.percent_owned, wembanyama.ownership.percent_started) == (99.92, 99.53)
    jokic = next(player for player in players if player.espn_id == JOKIC)
    assert (jokic.draft_ranks, jokic.positional_ranking, jokic.total_ranking) == ({"STANDARD": 1, "ROTO": 1}, 1, 1)
    assert jokic.ownership.percent_change == 0.01 and jokic.pro_team == "DEN"
    request = routes.espn_fba.calls.last.request
    assert request.url.path == "/apis/v3/games/fba/seasons/2027/segments/0/leaguedefaults/3"


def test_espn_request_is_public_read_only_and_filtered(source: MarketSource, routes: Routes) -> None:
    result = source.espn_players("nfl", 2026, limit=40)
    assert result.key == "ffl_2026_top40"
    request = routes.espn_ffl.calls.last.request
    assert (request.url.scheme, request.url.host) == ("https", ESPN_READS)
    assert request.url.path == "/apis/v3/games/ffl/seasons/2026/segments/0/leaguedefaults/3"
    assert dict(request.url.params) == {"view": "kona_player_info"}
    assert "cookie" not in request.headers and "authorization" not in request.headers
    assert request.headers["accept"] == "application/json"
    assert json.loads(request.headers[ESPN_FILTER_HEADER]) == {
        "players": {
            "limit": 40,
            "sortPercOwned": {"sortAsc": False, "sortPriority": 1},
            "filterStatsForTopScoringPeriodIds": {"value": 1, "additionalValue": []},
        }
    }
    assert json.loads(espn_filter(5))["players"]["limit"] == 5
    with pytest.raises(ValueError, match="limit"):
        source.espn_players("ffl", 2026, limit=0)


def test_espn_pool_rejects_the_wrong_shape_and_skips_bad_entries(caplog: pytest.LogCaptureFixture) -> None:
    for payload in (b"[]", b"{}", b'{"players": {}}', b'{"players": []}', b'{"players": [1, {"player": {}}]}'):
        with pytest.raises(SourceSchemaError):
            parse_espn_players(payload, Game.FFL)
    raw = json.loads(fixture(FFL_FIXTURE))
    raw["players"].append({"player": {"id": "x", "fullName": "Bad Id", "defaultPositionId": 2}})
    raw["players"].append({"ratings": {}})
    with caplog.at_level(logging.WARNING, logger="fm.sources.market"):
        players = parse_espn_players(json.dumps(raw).encode(), "ffl")
    assert len(players) == 10 and "skipped 2 of 12" in caplog.text


def test_espn_player_with_minimal_fields() -> None:
    entry = {"player": {"id": 99, "firstName": "New", "lastName": "Guy", "defaultPositionId": 16, "proTeamId": 0}}
    (player,) = parse_espn_players(json.dumps({"players": [entry]}).encode(), "ffl")
    assert (player.name, player.position, player.pro_team) == ("New Guy", "D/ST", "FA")
    assert player.injury_status is InjuryStatus.UNKNOWN and player.ownership == EspnOwnership()
    assert player.draft_ranks == {} and player.total_ranking is None and player.ownership.percent_owned is None
    ranked = {**entry, "ratings": {"0": {"totalRanking": 5}}}
    ranked["player"] = {**entry["player"], "draftRanksByRankType": {"PPR": {"rank": 0}, "STANDARD": {"rank": 12.0}}}
    (player,) = parse_espn_players(json.dumps({"players": [ranked]}).encode(), "ffl")
    assert player.draft_ranks == {"STANDARD": 12} and player.total_ranking == 5 and player.positional_ranking is None


def test_espn_pool_degrades_then_keeps_the_last_good_copy(
    source: MarketSource, routes: Routes, clock: FakeClock, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    routes.espn_ffl.mock(return_value=httpx.Response(503))
    with caplog.at_level(logging.WARNING, logger="fm.sources.market"):
        result = source.espn_players("ffl", 2026)
    assert (result.degraded, result.data, result.as_of) == (True, [], T0)
    assert "HTTP 503" in result.warnings[0] and "continuing without it" in caplog.text

    routes.espn_ffl.mock(return_value=ok(FFL_FIXTURE))
    good = source.espn_players("ffl", 2026)
    assert good.degraded is False and len(good.data) == 10
    clock.advance(hours=7)
    routes.espn_ffl.mock(return_value=httpx.Response(200, content=b'{"players": []}'))
    stale = source.espn_players("ffl", 2026)
    assert (stale.stale, stale.degraded, stale.cached, stale.as_of, len(stale.data)) == (True, False, True, T0, 10)
    assert "player pool is empty" in stale.warnings[0]
    cache = tmp_path / "cache" / "market" / "espn_players"
    assert (cache / "ffl_2026_top300.json").read_bytes() == fixture(FFL_FIXTURE)
    assert (cache / "ffl_2026_top300.rejected.json").read_bytes() == b'{"players": []}'


# --- merged market values ---


def test_market_values_merge_fantasycalc_with_espn_for_nfl(source: MarketSource, routes: Routes) -> None:
    result = source.market_values("ffl", 2026, rank_type="PPR")
    values = result.data
    assert (result.source, result.dataset, result.key) == ("market", "values", "ffl_2026_top300_redraft_12t_1qb_ppr1")
    assert (result.cached, result.stale, result.degraded) == (False, False, False)
    assert result.warnings == () and result.as_of == T0
    assert len(values) == 15  # 10 pooled ESPN players + 5 FantasyCalc players outside the recorded pool

    gibbs = values[GIBBS]
    assert isinstance(gibbs, MarketValue)
    assert (gibbs.name, gibbs.position, gibbs.trade_value, gibbs.value_rank, gibbs.value_trend_30d) == (
        "Jahmyr Gibbs",
        "RB",
        10697,
        1,
        620,
    )
    assert (gibbs.espn_rank, gibbs.espn_ranks["SUPERFLEX"]) == (1, 7)
    assert (gibbs.positional_ranking, gibbs.total_ranking) == (1, 2)
    assert (gibbs.percent_owned, gibbs.percent_started, gibbs.percent_change) == (99.95, 99.86, 0.0)
    allen = values[ALLEN]
    assert (allen.espn_rank, allen.trade_value, allen.value_rank, allen.value_trend_30d) == (26, 6609, 13, 1265)
    aubrey = values[AUBREY]  # ESPN only: FantasyCalc does not value kickers
    assert (aubrey.trade_value, aubrey.value_rank, aubrey.espn_rank, aubrey.percent_owned) == (None, None, 255, 99.49)
    bowers = values[BOWERS]  # FantasyCalc only: outside the recorded ESPN pool
    assert (bowers.name, bowers.position, bowers.trade_value, bowers.value_rank) == ("Brock Bowers", "TE", 6252, 14)
    assert (bowers.espn_rank, bowers.espn_ranks, bowers.percent_owned, bowers.total_ranking) == (None, {}, None, None)
    assert routes.fantasycalc.call_count == 1 and routes.espn_ffl.call_count == 1


def test_market_values_select_the_rank_type(source: MarketSource) -> None:
    standard = source.market_values("ffl", 2026).data
    assert (standard[ALLEN].espn_rank, standard[GIBBS].espn_rank) == (36, 1)
    assert source.market_values("ffl", 2026, rank_type="SUPERFLEX").data[ALLEN].espn_rank == 1
    unknown = source.market_values("ffl", 2026, rank_type="NOPE").data[ALLEN]
    assert unknown.espn_rank is None and unknown.espn_ranks["PPR"] == 26


def test_market_values_for_nba_skip_fantasycalc(source: MarketSource, routes: Routes) -> None:
    result = source.market_values("nba", 2027, rank_type="ROTO")
    assert routes.fantasycalc.call_count == 0 and routes.espn_fba.call_count == 1
    assert result.key == "fba_2027_top300" and len(result.data) == 7 and result.degraded is False
    jokic = result.data[JOKIC]
    assert (jokic.trade_value, jokic.value_rank, jokic.value_trend_30d) == (None, None, None)
    assert (jokic.espn_rank, jokic.espn_ranks, jokic.percent_change) == (1, {"STANDARD": 1, "ROTO": 1}, 0.01)
    wembanyama = result.data[WEMBANYAMA]
    assert (wembanyama.espn_rank, wembanyama.position, wembanyama.percent_owned) == (2, "C", 99.92)


def test_market_values_carry_degradation_and_the_oldest_as_of(
    source: MarketSource, routes: Routes, clock: FakeClock
) -> None:
    source.espn_players("ffl", 2026)  # ESPN cached at T0
    clock.advance(hours=2)
    routes.fantasycalc.mock(return_value=httpx.Response(500))
    degraded = source.market_values("ffl", 2026)
    assert (degraded.degraded, degraded.stale, degraded.cached, degraded.as_of) == (True, False, False, T0)
    assert len(degraded.data) == 10 and all(value.trade_value is None for value in degraded.data.values())
    assert degraded.data[GIBBS].percent_owned == 99.95 and "HTTP 500" in degraded.warnings[0]

    routes.fantasycalc.mock(return_value=ok(FC_FIXTURE))
    fresh = source.market_values("ffl", 2026)
    assert (fresh.degraded, fresh.cached, fresh.as_of) == (False, False, T0)  # ESPN from cache, FantasyCalc downloaded
    assert fresh.data[GIBBS].trade_value == 10697
    assert source.market_values("ffl", 2026).cached is True

    clock.advance(hours=13)  # both TTLs have passed; ESPN refreshes, FantasyCalc fails and serves its T0+2h copy
    routes.fantasycalc.mock(return_value=httpx.Response(500))
    stale = source.market_values("ffl", 2026)
    assert (stale.stale, stale.degraded, stale.cached) == (True, False, False)
    assert stale.as_of == T0 + timedelta(hours=2) and stale.data[GIBBS].trade_value == 10697
    assert any("HTTP 500" in warning for warning in stale.warnings)


# --- safety ---


def test_only_public_read_hosts_are_contacted_without_cookies(source: MarketSource, routes: Routes) -> None:
    source.market_values("ffl", 2026)
    source.market_values("fba", 2027)
    calls = [call for route in routes.all for call in route.calls]
    assert {call.request.url.host for call in calls} == {FANTASYCALC, ESPN_READS}
    assert all(call.request.method == "GET" for call in calls)
    assert all("cookie" not in call.request.headers for call in calls)


def test_accidental_live_calls_are_not_swallowed(source: MarketSource, routes: Routes) -> None:
    # The harness blocks the network with a bare RuntimeError; graceful degradation must never hide one.
    routes.fantasycalc.mock(side_effect=RuntimeError("outbound network is disabled in unit tests"))
    with pytest.raises(RuntimeError, match="outbound network"):
        source.fantasycalc()
    routes.espn_ffl.mock(side_effect=RuntimeError("outbound network is disabled in unit tests"))
    with pytest.raises(RuntimeError, match="outbound network"):
        source.market_values("ffl", 2026)


def test_ttls_match_source_cadence() -> None:
    assert MarketSource.ttl["fantasycalc"] == timedelta(hours=12)
    assert MarketSource.ttl["espn_players"] == timedelta(hours=6)
    assert MarketSource.min_interval >= 0.5
    assert MarketSource.ESPN_READS.startswith("https://lm-api-reads.fantasy.espn.com/")


def test_models_ignore_unknown_fields_and_accept_field_names() -> None:
    ownership = EspnOwnership.model_validate({"percentOwned": 50.0, "newField": 1, "date": "not-a-timestamp"})
    assert ownership.percent_owned == 50.0 and ownership.updated_at is None and ownership.percent_change is None
    value = FantasyCalcValue(fantasycalc_id=1, name="X", position="RB", value=10, overall_rank=1, position_rank=1)
    assert value.espn_id is None and value.trend_30d is None and value.team is None
    market = MarketValue(espn_id=1, name="X")
    assert market.trade_value is None and market.espn_ranks == {} and market.percent_owned is None
