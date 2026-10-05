"""ESPN read client over a mocked transport (respx; no network): every view, the filters, retries, auth failures,
schema errors and raw capture.

The fixtures under ``tests/fixtures/espn/`` are hand-built stand-ins in the shape of ESPN's responses for the same
10-team NFL league as ``ffl_settings_ppr.json`` (week 4 of 2026) and the NBA 9-cat league of ``fba_settings_9cat.json``.
No real league or manager names, no cookies. ROADMAP #14 replaces them with scrubbed real-league captures.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx

from fm import paths
from fm.config import League
from fm.espn.auth import AuthError, EspnSession
from fm.espn.client import (
    API_ROOT,
    DEFAULT_TRANSACTION_TYPES,
    FILTER_HEADER,
    PLAYER_CARD_BATCH,
    READS_HOST,
    ClientOptions,
    EspnAuthError,
    EspnClient,
    EspnClientError,
    EspnHttpError,
    EspnRead,
    EspnSchemaError,
    View,
    filter_header,
    player_filter,
    schedule_filter,
    stat_entry_id,
    transaction_filter,
)
from fm.espn.ids import FBA, FFL, Game
from fm.espn.models import (
    Matchup,
    PlayerStats,
    PoolEntry,
    RosterEntry,
    Team,
    Transaction,
    TransactionsView,
)
from fm.espn.settings import LeagueSettings

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "espn"
T0 = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)
LEAGUE_ID = 1234567
SEASON = 2026
LEAGUE_PATH = f"/apis/v3/games/ffl/seasons/{SEASON}/segments/0/leagues/{LEAGUE_ID}"
SEASON_PATH = f"/apis/v3/games/ffl/seasons/{SEASON}"
NBA_LEAGUE_ID = 3456789
NBA_SEASON = 2027

S2 = "s2-secret-value"
SWID = "{TEST-SWID}"
SESSION = EspnSession(espn_s2=S2, swid=SWID, expires_at=T0 + timedelta(days=200))

# Transaction ids in the fixtures.
T_WAIVER_WON = "3b6b7b9e-0000-4000-8000-0000000000a1"
T_WAIVER_LOST = "3b6b7b9e-0000-4000-8000-0000000000a2"
T_FA = "3b6b7b9e-0000-4000-8000-0000000000b1"
T_TRADE = "3b6b7b9e-0000-4000-8000-0000000000c1"
T_CLAIM = "3b6b7b9e-0000-4000-8000-00000000c1a1"
T_TRADE_IN = "3b6b7b9e-0000-4000-8000-0000000000d1"

# Player ids in the fixtures.
ALLEN, GIBBS, CHASE, MCBRIDE, ALLGEIER, SHAHEED, OLAVE = 3918298, 4429795, 4362628, 4361307, 4373626, 4249836, 4361370
LAMAR, KELCE, DOWDLE, WARREN, LIKELY, DAVIS, JENNINGS, BIGSBY = (
    3916387,
    15847,
    4038815,
    4569173,
    4361050,
    4429160,
    4360438,
    4569587,
)
JOKIC = 3112335

LEAGUE_VIEWS = {
    "mSettings": "ffl_settings_ppr.json",
    "mTeam+mStandings": "ffl_teams.json",
    "mRoster": "ffl_rosters_week4.json",
    "mMatchup": "ffl_matchups.json",
    "mMatchupScore+mScoreboard": "ffl_scoreboard_week4.json",
    "kona_player_info": "ffl_free_agents_week4.json",
    "kona_playercard": "ffl_player_cards_week4.json",
    "mTransactions2": "ffl_transactions_week4.json",
    "mPendingTransactions": "ffl_pending_transactions.json",
}
NBA_VIEWS = {
    "mMatchupScore+mScoreboard": "fba_scoreboard_9cat_day1.json",
}
SEASON_VIEWS = {"ffl": "ffl_pro_schedule_2026.json", "fba": "fba_pro_schedule_2027.json"}
NEW_FIXTURES = sorted(set(LEAGUE_VIEWS.values()) | set(NBA_VIEWS.values()) | set(SEASON_VIEWS.values()))


def fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def ok(name: str) -> httpx.Response:
    return httpx.Response(200, content=fixture(name), headers={"content-type": "application/json;charset=UTF-8"})


def espn_error(status: int, kind: str, message: str) -> httpx.Response:
    body = {"messages": [message], "details": [{"message": message, "shortMessage": message, "type": kind}]}
    return httpx.Response(status, json=body)


def view_key(request: httpx.Request) -> str:
    return "+".join(request.url.params.get_list("view"))


class FakeClock:
    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> datetime:
        self.now += timedelta(**delta)
        return self.now


class FakeEspn:
    """Serves each view from its fixture, keyed by the joined ``view`` params, and records every request.

    ``queue`` replaces a view's answers with a sequence (the last one repeats); an ``Exception`` in the sequence is
    raised as the transport error it is.
    """

    def __init__(self, router: respx.MockRouter) -> None:
        self.requests: dict[str, list[httpx.Request]] = defaultdict(list)
        self.queued: dict[str, list[httpx.Response | Exception]] = {}
        self.league = router.get(
            host=READS_HOST, path__regex=r"^/apis/v3/games/(?P<game>ffl|fba)/seasons/\d+/segments/0/leagues/\d+$"
        ).mock(side_effect=self._league)
        self.season = router.get(host=READS_HOST, path__regex=r"^/apis/v3/games/(?P<game>ffl|fba)/seasons/\d+$").mock(
            side_effect=self._season
        )

    def queue(self, key: str, *answers: httpx.Response | Exception) -> None:
        self.queued[key] = list(answers)

    def _answer(self, key: str, default: str | None) -> httpx.Response:
        if key in self.queued:
            answers = self.queued[key]
            answer = answers.pop(0) if len(answers) > 1 else answers[0]
            if isinstance(answer, Exception):
                raise answer
            return answer
        if default is None:
            raise AssertionError(f"no fixture for view {key!r}")
        return ok(default)

    def _league(self, request: httpx.Request, game: str) -> httpx.Response:
        key = view_key(request)
        self.requests[key].append(request)
        views = LEAGUE_VIEWS if game == "ffl" else NBA_VIEWS
        return self._answer(key, views.get(key))

    def _season(self, request: httpx.Request, game: str) -> httpx.Response:
        key = view_key(request)
        self.requests[key].append(request)
        return self._answer(key, SEASON_VIEWS[game] if key == "proTeamSchedules_wl" else None)

    def last(self, key: str) -> httpx.Request:
        return self.requests[key][-1]

    def filter_of(self, key: str) -> Any:
        return json.loads(self.last(key).headers[FILTER_HEADER])


@pytest.fixture
def espn() -> Iterator[FakeEspn]:
    with respx.mock(assert_all_called=False) as router:
        yield FakeEspn(router)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def sleeps() -> list[float]:
    return []


@pytest.fixture
def cache_root(tmp_path: Path) -> Path:
    return tmp_path / "cache"


@pytest.fixture
def options(cache_root: Path, clock: FakeClock, sleeps: list[float]) -> ClientOptions:
    return {"cache_root": cache_root, "clock": clock, "sleep": sleeps.append, "min_interval_s": 0.0}


@pytest.fixture
def client(espn: FakeEspn, options: ClientOptions) -> Iterator[EspnClient]:
    with EspnClient("ffl", LEAGUE_ID, SEASON, SESSION, **options) as client:
        yield client


@pytest.fixture
def nba(espn: FakeEspn, options: ClientOptions) -> Iterator[EspnClient]:
    with EspnClient(Game.FBA, NBA_LEAGUE_ID, NBA_SEASON, SESSION, **options) as client:
        yield client


# --- request shape ----------------------------------------------------------------------------------------------------


def test_league_url_cookies_and_headers(client: EspnClient, espn: FakeEspn) -> None:
    read = client.settings()
    request = espn.last("mSettings")
    assert str(request.url) == f"https://{READS_HOST}{LEAGUE_PATH}?view=mSettings"
    assert client.league_url == f"{API_ROOT}/ffl/seasons/2026/segments/0/leagues/{LEAGUE_ID}"
    assert request.headers["cookie"] == f"espn_s2={S2}; SWID={SWID}"
    assert request.headers["user-agent"].startswith("espn-fantasy/")
    assert request.headers["accept"] == "application/json"
    assert FILTER_HEADER.lower() not in request.headers
    assert isinstance(read, EspnRead) and read.as_of == T0


def test_public_league_sends_no_cookie(espn: FakeEspn, options: ClientOptions) -> None:
    with EspnClient("nfl", LEAGUE_ID, SEASON, **options) as client:
        client.settings()
    assert "cookie" not in espn.last("mSettings").headers


def test_for_league_uses_the_configured_game_ids_and_season(options: ClientOptions) -> None:
    nfl = League(key="nfl", sport="nfl", espn_league_id=LEAGUE_ID, season=SEASON, team_id=1)
    nba = League(key="nba", sport="nba", espn_league_id=NBA_LEAGUE_ID, season=NBA_SEASON, team_id=4)
    with EspnClient.for_league(nfl, SESSION, **options) as client:
        assert (client.game, client.league_id, client.season) == (Game.FFL, LEAGUE_ID, SEASON)
        assert client.league_url.endswith(LEAGUE_PATH)
    with EspnClient.for_league(nba, **options) as client:
        assert client.game is Game.FBA
        assert client.season_url == f"{API_ROOT}/fba/seasons/{NBA_SEASON}"
        assert repr(client) == f"EspnClient(fba league {NBA_LEAGUE_ID}, season {NBA_SEASON})"


def test_rejects_nonsense_ids() -> None:
    with pytest.raises(ValueError, match="league_id"):
        EspnClient("ffl", 0, SEASON, capture=False)
    with pytest.raises(ValueError, match="season"):
        EspnClient("ffl", LEAGUE_ID, 0, capture=False)
    with pytest.raises(ValueError, match="sport"):
        EspnClient("mlb", LEAGUE_ID, SEASON, capture=False)


def test_timeouts_are_explicit(options: ClientOptions) -> None:
    with EspnClient("ffl", LEAGUE_ID, SEASON, **options) as client:
        assert client.client.timeout == httpx.Timeout(30.0, connect=10.0)
    with EspnClient("ffl", LEAGUE_ID, SEASON, **{**options, "timeout": 5.0}) as client:
        assert client.client.timeout == httpx.Timeout(5.0)


def test_owned_client_is_closed_and_an_injected_one_is_not(options: ClientOptions) -> None:
    with EspnClient("ffl", LEAGUE_ID, SEASON, **options) as owned:
        http = owned.client
        assert http is owned.client
    assert http.is_closed
    injected = httpx.Client()
    with EspnClient("ffl", LEAGUE_ID, SEASON, **{**options, "client": injected}) as client:
        assert client.client is injected
    assert not injected.is_closed
    injected.close()


def test_requests_are_paced(espn: FakeEspn, cache_root: Path, clock: FakeClock) -> None:
    now = [100.0]
    slept: list[float] = []

    def sleep(seconds: float) -> None:
        slept.append(seconds)
        now[0] += seconds

    client = EspnClient(
        "ffl", LEAGUE_ID, SEASON, SESSION, cache_root=cache_root, clock=clock, sleep=sleep, monotonic=lambda: now[0]
    )
    client.teams()
    now[0] += 0.1
    client.teams()
    now[0] += 5
    client.teams()
    assert slept == [pytest.approx(0.15)]
    assert client.requests == 3 and len(espn.requests["mTeam+mStandings"]) == 3


# --- settings, teams, rosters -----------------------------------------------------------------------------------------


def test_settings_parse_through_the_settings_module(client: EspnClient) -> None:
    settings = client.settings().data
    assert isinstance(settings, LeagueSettings)
    assert (settings.league_id, settings.season, settings.game) == (LEAGUE_ID, SEASON, Game.FFL)
    assert settings.points_for("REC") == 1.0 and settings.slot_count("FLEX") == 1


def test_settings_that_do_not_parse_raise_a_schema_error(client: EspnClient, espn: FakeEspn) -> None:
    espn.queue("mSettings", httpx.Response(200, json={"gameId": 1, "id": LEAGUE_ID, "seasonId": SEASON}))
    with pytest.raises(EspnSchemaError, match="mSettings: .*'settings'") as excinfo:
        client.settings()
    assert isinstance(excinfo.value, EspnClientError) and isinstance(excinfo.value, ValueError)


def test_teams_with_standings(client: EspnClient, espn: FakeEspn) -> None:
    read = client.teams()
    assert espn.last("mTeam+mStandings").url.params.get_list("view") == ["mTeam", "mStandings"]
    view = read.data
    assert (view.league_id, view.season_id, view.scoring_period_id, view.game_id) == (LEAGUE_ID, SEASON, 4, 1)
    assert view.status is not None and view.status.current_matchup_period == 4
    assert view.status.waiver_next_execution_date == datetime(2026, 10, 7, 7, 0, tzinfo=UTC)  # 3 a.m. ET
    assert view.status.final_scoring_period == 18
    assert len(view.teams) == 10 and [team.id for team in view.teams] == list(range(1, 11))

    leader = view.team(2)
    assert isinstance(leader, Team)
    assert (leader.record.overall.wins, leader.record.overall.losses, leader.record.overall.points_for) == (3, 0, 402.1)
    assert (leader.playoff_seed, leader.waiver_rank, leader.transaction_counter.acquisition_budget_spent) == (1, 10, 41)
    assert leader.display_name == "Fixture Team 2" and leader.abbrev == "FT2"

    ours = view.team(1)
    assert ours.transaction_counter.move_to_ir == 1 and ours.transaction_counter.matchup_acquisition_totals == {
        1: 1,
        3: 1,
    }
    assert ours.values_by_stat == {3: 812.0, 4: 7.0, 24: 896.0, 53: 60.0}  # the junk key is dropped
    assert ours.record.home is not None and ours.record.overall.streak_type == "WIN"
    assert ours.owners == ("{00000000-0000-0000-0000-000000000001}",)
    assert view.team(10).display_name == "Fixture Ten"  # location + nickname, no name
    assert view.team(3).record.home is None

    assert len(view.members) == 10
    manager = view.member("{00000000-0000-0000-0000-000000000003}")
    assert manager is not None and manager.is_league_manager and manager.display_name == "Manager 3"
    assert view.member("{nobody}") is None
    with pytest.raises(KeyError, match="no team 11"):
        view.team(11)


def test_rosters_for_a_scoring_period(client: EspnClient, espn: FakeEspn) -> None:
    view = client.rosters(4).data
    assert espn.last("mRoster").url.params["scoringPeriodId"] == "4"
    assert [roster.team_id for roster in view.teams] == [1, 2]
    ours = view.roster(1)
    assert len(ours.entries) == 11 and ours.applied_stat_total == 31.4
    assert ours.player_ids[:3] == (ALLEN, GIBBS, 4430807)

    gibbs = ours.entry(GIBBS)
    assert isinstance(gibbs, RosterEntry) and gibbs.lineup_slot_id == FFL.slot_id("RB")
    assert gibbs.lineup_locked is True and gibbs.player_pool_entry.roster_locked is True
    assert gibbs.player.full_name == "Jahmyr Gibbs" and gibbs.player.pro_team_id == 8
    assert gibbs.player.eligible_slots == (2, 3, 23, 7, 20, 21) and gibbs.player.default_position_id == 2
    assert gibbs.acquisition_type == "DRAFT" and gibbs.acquisition_date == datetime(2026, 8, 31, 0, 15, tzinfo=UTC)

    allgeier = ours.entry(ALLGEIER)
    assert allgeier.acquisition_type == "ADD"
    assert allgeier.acquisition_date == datetime(2026, 9, 30, 18, 12, tzinfo=UTC)
    assert ours.entry(SHAHEED).pending_transaction_ids == (T_CLAIM,)

    olave = ours.entry(OLAVE)
    assert olave.lineup_slot_id == FFL.ir_slot and olave.player.injury_status == "INJURY_RESERVE"
    assert olave.player.injured is True and olave.player.active is False and olave.player.stats == ()
    assert [entry.player_id for entry in ours.in_slot(FFL.bench_slot)] == [ALLGEIER, SHAHEED]
    assert ours.in_slot(99) == ()

    assert view.team_of(LAMAR) == 2 and view.team_of(GIBBS) == 1 and view.team_of(999) is None
    assert view.roster(2).entry(LAMAR).player.projection(SEASON, 4) is None  # trimmed in the fixture
    with pytest.raises(KeyError, match="no roster for team 3"):
        view.roster(3)
    with pytest.raises(KeyError, match="not on this roster"):
        ours.entry(LAMAR)


def test_rosters_default_to_espn_current_period(client: EspnClient, espn: FakeEspn) -> None:
    client.rosters()
    assert "scoringPeriodId" not in espn.last("mRoster").url.params


def test_player_stat_lines_are_keyed_by_stat_id(client: EspnClient) -> None:
    gibbs = client.rosters(4).data.roster(1).entry(GIBBS).player
    projected = gibbs.projection(SEASON, 4)
    assert isinstance(projected, PlayerStats) and projected.id == "1120264"
    assert projected.is_projection and projected.is_single_period and not projected.is_season
    assert projected.stats[FFL.stat_id("RY")] == 92.1 and projected.stats[FFL.stat_id("REC")] == 4.6
    assert projected.applied_stats[24] == pytest.approx(9.21) and projected.applied_total == pytest.approx(21.91)
    assert projected.pro_team_id == 8

    actual = gibbs.actual(SEASON, 4)
    assert actual is not None and actual.is_actual and actual.stats[24] == 104.0 and actual.id == "0120264"
    season = gibbs.projection(SEASON, 0)
    assert season is not None and season.is_season and season.stats[24] == 1380.0 and season.scoring_period_id == 0
    assert gibbs.actual(SEASON, 0) is not None and gibbs.actual(SEASON, 0) is not season
    assert gibbs.projection(SEASON, 5) is None and gibbs.projection(2025, 4) is None
    assert gibbs.stat_entry(season=SEASON, scoring_period=4, projected=True) is projected
    assert all(isinstance(value, float) for entry in gibbs.stats for value in entry.stats.values())


# --- matchups ---------------------------------------------------------------------------------------------------------


def test_matchups_schedule(client: EspnClient) -> None:
    view = client.matchups().data
    assert len(view.schedule) == 9 and view.matchup_periods == (3, 4, 15)
    assert len(view.for_period(4)) == 5 and view.for_period(99) == ()

    (ours,) = view.for_team(1, 4)
    assert isinstance(ours, Matchup) and ours.id == 16 and not ours.is_decided and not ours.is_playoff
    assert ours.team_ids == (1, 2) and ours.home is not None and ours.home.team_id == 1
    opponent = ours.opponent(1)
    assert opponent is not None and opponent.team_id == 2 and ours.opponent(3) is None
    assert ours.home.cumulative_score is not None and ours.home.cumulative_score.score_by_stat == {}  # null in ffl
    assert ours.home.roster is None

    week3 = next(matchup for matchup in view.schedule if matchup.id == 11)
    assert week3.winner == "AWAY" and week3.is_decided
    assert week3.home is not None and week3.home.points_by_scoring_period == {3: 118.9}
    assert week3.away is not None and week3.away.total_points == 140.3

    bye = next(matchup for matchup in view.schedule if matchup.id == 71)
    assert bye.is_bye and bye.away is None and bye.is_playoff and bye.playoff_tier_type == "WINNERS_BRACKET"
    assert bye.sides == (bye.home,) and bye.opponent(2) is None and bye.side(2) is bye.home
    assert len(view.for_team(2)) == 3


def test_matchups_can_be_narrowed_to_one_period(client: EspnClient, espn: FakeEspn) -> None:
    read = client.matchups(matchup_period=3)
    assert [matchup.id for matchup in read.data.schedule] == [11, 12]
    assert read.data.league_id == LEAGUE_ID and read.capture is not None and read.capture.kind == "mMatchup"
    assert len(espn.requests["mMatchup"]) == 1


def test_scoreboard_carries_live_scores_and_lineups(client: EspnClient, espn: FakeEspn) -> None:
    view = client.scoreboard(4, scoring_period=4).data
    request = espn.last("mMatchupScore+mScoreboard")
    assert request.url.params.get_list("view") == ["mMatchupScore", "mScoreboard"]
    assert request.url.params["scoringPeriodId"] == "4"
    assert request.headers[FILTER_HEADER] == '{"schedule":{"filterMatchupPeriodIds":{"value":[4]}}}'

    (ours,) = view.for_team(1, 4)
    assert ours.home is not None and ours.away is not None
    assert (ours.home.total_points, ours.home.total_points_live, ours.home.total_projected_points_live) == (
        31.4,
        31.4,
        128.6,
    )
    assert ours.home.roster is not None and ours.home.roster.player_ids == (ALLEN, GIBBS, MCBRIDE)
    assert ours.home.roster.entry(GIBBS).lineup_locked and not ours.home.roster.entry(ALLEN).lineup_locked
    assert ours.away.roster is not None and len(ours.away.roster.entries) == 2
    other = next(matchup for matchup in view.schedule if matchup.id == 17)
    assert other.home is not None and other.home.roster is None and other.home.total_projected_points_live == 110.3


# --- player pool and player cards -------------------------------------------------------------------------------------


def test_free_agents_send_the_pool_filter(client: EspnClient, espn: FakeEspn) -> None:
    view = client.free_agents(4, slot_ids=[FFL.slot_id("RB")], limit=25).data
    request = espn.last("kona_player_info")
    assert request.url.params["scoringPeriodId"] == "4"
    assert espn.filter_of("kona_player_info") == {
        "players": {
            "filterStatus": {"value": ["FREEAGENT", "WAIVERS"]},
            "filterSlotIds": {"value": [2]},
            "limit": 25,
            "offset": 0,
            "sortPercOwned": {"sortPriority": 1, "sortAsc": False},
            "sortDraftRanks": {"sortPriority": 100, "sortAsc": True, "value": "STANDARD"},
        }
    }
    assert view.player_ids == (DOWDLE, WARREN, LIKELY, DAVIS)

    dowdle = view.entry(DOWDLE)
    assert isinstance(dowdle, PoolEntry) and dowdle.is_free_agent and not dowdle.is_on_waivers
    assert dowdle.rostered_team_id is None and dowdle.on_team_id == 0
    assert dowdle.player.ownership is not None and dowdle.player.ownership.percent_owned == 61.4
    assert dowdle.player.ownership.average_draft_position == 141.3
    assert dowdle.ratings[0].positional_ranking == 24 and dowdle.ratings[4].total_ranking == 55
    assert dowdle.player.last_news_date == datetime(2026, 10, 2, 18, 30, tzinfo=UTC)

    warren = view.entry(WARREN)
    assert warren.is_on_waivers and warren.waiver_process_date == datetime(2026, 10, 5, 16, 40, tzinfo=UTC)
    davis = view.entry(DAVIS).player.projection(SEASON, 4)
    assert davis is not None and davis.stats[24] == 27.0
    assert view.entry(LIKELY).ratings == {}
    with pytest.raises(KeyError):
        view.entry(GIBBS)


def test_free_agents_defaults_and_paging(client: EspnClient, espn: FakeEspn) -> None:
    client.free_agents()
    request = espn.last("kona_player_info")
    assert "scoringPeriodId" not in request.url.params
    players = espn.filter_of("kona_player_info")["players"]
    assert players["limit"] == 50 and players["offset"] == 0 and "filterSlotIds" not in players

    client.free_agents(4, statuses=["WAIVERS"], offset=50)
    players = espn.filter_of("kona_player_info")["players"]
    assert players["filterStatus"] == {"value": ["WAIVERS"]} and players["offset"] == 50
    with pytest.raises(ValueError, match="limit"):
        client.free_agents(limit=0)
    with pytest.raises(ValueError, match="offset"):
        client.free_agents(offset=-1)


def test_player_cards_ask_for_the_season_and_period_stat_lines(client: EspnClient, espn: FakeEspn) -> None:
    view = client.player_cards([GIBBS, CHASE, GIBBS], scoring_period=4).data
    request = espn.last("kona_playercard")
    assert request.url.params["scoringPeriodId"] == "4"
    assert espn.filter_of("kona_playercard") == {
        "players": {
            "filterIds": {"value": [GIBBS, CHASE]},
            "filterStatsForTopScoringPeriodIds": {
                "value": 4,  # the actual week lines: ESPN keys them by pro game id, so "0120264" would return nothing
                "additionalValue": ["002026", "102026", "1120264"],
            },
        }
    }
    assert view.player_ids == (GIBBS, CHASE)
    gibbs = view.entry(GIBBS)
    assert gibbs.rostered_team_id == 1 and gibbs.lineup_locked and gibbs.status == "ONTEAM"
    assert gibbs.player.actual(SEASON, 4) is not None and gibbs.ratings[0].total_rating == 38.6

    client.player_cards([CHASE], top_scoring_periods=2)
    assert espn.filter_of("kona_playercard")["players"]["filterStatsForTopScoringPeriodIds"] == {
        "value": 2,
        "additionalValue": ["002026", "102026"],
    }
    with pytest.raises(ValueError, match=f"at most {PLAYER_CARD_BATCH}"):
        client.player_cards(range(1, PLAYER_CARD_BATCH + 2))
    with pytest.raises(ValueError, match="at least one player id"):
        client.player_cards([])


def test_projections_are_stat_lines_by_player_fetched_in_batches(client: EspnClient, espn: FakeEspn) -> None:
    projections = client.projections([GIBBS, CHASE], 4)
    assert set(projections) == {GIBBS, CHASE}
    assert projections[GIBBS].stats[FFL.stat_id("RY")] == 92.1 and projections[GIBBS].is_projection
    assert projections[CHASE].stats[FFL.stat_id("REY")] == 93.6
    assert len(espn.requests["kona_playercard"]) == 1

    client.projections(range(1000, 1000 + PLAYER_CARD_BATCH + 5), 4)  # two batches, same fixture answers both
    assert len(espn.requests["kona_playercard"]) == 3
    sizes = [
        len(json.loads(r.headers[FILTER_HEADER])["players"]["filterIds"]["value"])
        for r in espn.requests["kona_playercard"][1:]
    ]
    assert sizes == [PLAYER_CARD_BATCH, 5]


def test_stat_entry_ids_follow_espns_composite_format() -> None:
    assert stat_entry_id(2026) == "002026"
    assert stat_entry_id(2026, projected=True) == "102026"
    assert stat_entry_id(2026, 4, projected=True) == "1120264"
    assert stat_entry_id(2026, 4) == "0120264"
    assert stat_entry_id(2027, 25) == "01202725"


def test_filter_builders() -> None:
    assert transaction_filter(["WAIVER"]) == {"transactions": {"filterType": {"value": ["WAIVER"]}}}
    assert schedule_filter([4, 5]) == {"schedule": {"filterMatchupPeriodIds": {"value": [4, 5]}}}
    assert player_filter() == {"players": {}}
    assert player_filter(ids=[1], stat_ids=["002026"]) == {
        "players": {
            "filterIds": {"value": [1]},
            "filterStatsForTopScoringPeriodIds": {"value": 1, "additionalValue": ["002026"]},
        }
    }
    assert filter_header({"a": [1, 2]}) == {FILTER_HEADER: '{"a":[1,2]}'}


# --- transactions and pending offers ----------------------------------------------------------------------------------


def test_transactions_with_bids(client: EspnClient, espn: FakeEspn) -> None:
    view = client.transactions(4).data
    request = espn.last("mTransactions2")
    assert request.url.params["scoringPeriodId"] == "4"
    assert espn.filter_of("mTransactions2") == {
        "transactions": {"filterType": {"value": list(DEFAULT_TRANSACTION_TYPES)}}
    }
    assert [t.id for t in view.transactions] == [T_WAIVER_WON, T_WAIVER_LOST, T_FA, T_TRADE, T_CLAIM]

    won = view.transactions[0]
    assert isinstance(won, Transaction) and won.type == "WAIVER" and won.executed and won.is_waiver
    assert (won.team_id, won.bid_amount, won.adds, won.drops) == (3, 17, (JENNINGS,), (BIGSBY,))
    assert won.date == won.process_date == datetime(2026, 9, 30, 7, 2, tzinfo=UTC)  # the 3 a.m. ET waiver run
    assert won.items[0].to_lineup_slot_id == FFL.bench_slot and won.scoring_period_id == 4

    lost = view.transactions[1]
    assert lost.failed and lost.status == "FAILED_INVALIDPLAYERSOURCE" and lost.bid_amount == 9  # outbid, bid kept
    assert not lost.pending and not lost.executed
    assert [t.bid_amount for t in view.with_bids()] == [17, 9, 0, 12]

    trade = view.transactions[3]
    assert trade.is_trade and trade.pending and trade.team_ids == (2, 1) and trade.bid_amount is None
    assert trade.expiration_date is not None and trade.proposed_date is not None
    assert trade.expiration_date - trade.proposed_date == timedelta(days=2)
    assert trade.player_ids == (KELCE, MCBRIDE) and trade.date == trade.proposed_date
    assert view.of_type("TRADE_PROPOSAL") == (trade,) and view.pending() == (trade, view.transactions[4])
    assert view.transactions[4].adds == (WARREN,) and view.transactions[4].bid_amount == 12


def test_transactions_filter_statuses_on_our_side(client: EspnClient, espn: FakeEspn) -> None:
    read = client.transactions(4, types=["WAIVER", "TRADE_PROPOSAL"], statuses=["PENDING"])
    assert [t.id for t in read.data.transactions] == [T_TRADE, T_CLAIM]
    assert read.data.league_id == LEAGUE_ID and read.capture is not None and read.capture.kind == "mTransactions2"
    assert espn.filter_of("mTransactions2") == {"transactions": {"filterType": {"value": ["WAIVER", "TRADE_PROPOSAL"]}}}
    client.transactions()
    assert "scoringPeriodId" not in espn.last("mTransactions2").url.params
    with pytest.raises(ValueError, match="transaction type"):
        client.transactions(4, types=[])


def test_pending_transactions_view(client: EspnClient, espn: FakeEspn) -> None:
    view = client.pending_transactions().data
    assert espn.last("mPendingTransactions").url.params.get_list("view") == ["mPendingTransactions"]
    assert [t.id for t in view.transactions] == [T_TRADE, T_CLAIM, T_TRADE_IN]  # read from ``pendingTransactions``
    assert all(t.pending for t in view.transactions)
    incoming = view.transactions[2]
    assert incoming.team_id == 6 and incoming.items[0].to_team_id == 1 and incoming.items[1].from_team_id == 1
    # The other candidate key parses the same way.
    direct = TransactionsView.model_validate({"transactions": [{"id": 7, "type": "WAIVER", "status": "PENDING"}]})
    assert direct.transactions[0].id == "7" and direct.pending() == direct.transactions
    assert TransactionsView.model_validate({}).transactions == ()


def test_pending_offers_merge_both_candidate_views(client: EspnClient, espn: FakeEspn) -> None:
    earlier = T0 - timedelta(hours=1)
    offers = client.pending_offers(now=earlier)
    assert [offer.id for offer in offers] == [T_TRADE, T_TRADE_IN, T_CLAIM]  # oldest first, each once
    assert all(offer.pending for offer in offers)
    assert len(espn.requests["mPendingTransactions"]) == 1 and len(espn.requests["mTransactions2"]) == 1
    assert espn.filter_of("mTransactions2") == {"transactions": {"filterType": {"value": ["WAIVER", "TRADE_PROPOSAL"]}}}

    # At the client's clock team 2's offer has reached its expirationDate: still PENDING at ESPN, no longer open.
    assert [offer.id for offer in client.pending_offers()] == [T_TRADE_IN, T_CLAIM]

    espn.queue("mPendingTransactions", httpx.Response(200, json={"id": LEAGUE_ID, "pendingTransactions": []}))
    assert [offer.id for offer in client.pending_offers(4, now=earlier)] == [T_TRADE, T_CLAIM]
    assert espn.last("mPendingTransactions").url.params["scoringPeriodId"] == "4"


def test_a_cancel_record_closes_the_offer_it_names(client: EspnClient, espn: FakeEspn) -> None:
    """ESPN records a cancelled or expired offer as a separate CANCEL record that still says isPending (ROADMAP #14):
    neither it nor the offer it names is open, whatever their flags say."""
    earlier = T0 - timedelta(hours=1)
    body = json.loads(fixture("ffl_transactions_week4.json"))
    offer = next(record for record in body["transactions"] if record["id"] == T_TRADE)
    cancel = {
        **offer,
        "id": "3b6b7b9e-0000-4000-8000-0000000000e1",
        "status": "CANCELED",
        "executionType": "CANCEL",
        "isPending": True,
        "relatedTransactionId": T_TRADE,
    }
    body["transactions"].append(cancel)
    espn.queue("mTransactions2", httpx.Response(200, json=body))
    espn.queue("mPendingTransactions", httpx.Response(200, json={"id": LEAGUE_ID, "pendingTransactions": []}))
    assert [offer.id for offer in client.pending_offers(now=earlier)] == [T_CLAIM]

    view = TransactionsView.model_validate(body)
    closing = view.transactions[-1]
    assert closing.is_cancellation and not closing.pending and closing.is_pending
    assert T_TRADE in {t.id for t in view.pending()} and T_TRADE not in {t.id for t in view.open(earlier)}
    assert [t.id for t in view.open(earlier)] == [T_CLAIM]


def test_a_status_less_record_counts_as_pending_only_in_the_pending_view(client: EspnClient, espn: FakeEspn) -> None:
    claim = {"id": 41, "type": "WAIVER", "teamId": 1, "items": []}
    espn.queue("mPendingTransactions", httpx.Response(200, json={"id": LEAGUE_ID, "pendingTransactions": [claim]}))
    espn.queue("mTransactions2", httpx.Response(200, json={"id": LEAGUE_ID, "transactions": []}))
    assert [offer.id for offer in client.pending_offers()] == ["41"]
    loose = TransactionsView.model_validate({"transactions": [claim]})
    assert loose.pending() == () and loose.open(T0) == ()


# --- pro schedules ----------------------------------------------------------------------------------------------------


def test_pro_schedule_is_season_level(client: EspnClient, espn: FakeEspn) -> None:
    read = client.pro_schedule()
    request = espn.last("proTeamSchedules_wl")
    assert str(request.url) == f"https://{READS_HOST}{SEASON_PATH}?view=proTeamSchedules_wl"
    assert request.headers["cookie"].startswith("espn_s2=")
    schedule = read.data
    assert len(schedule.pro_teams) == 15 and schedule.scoring_periods == (4, 5)

    week4 = schedule.games(4)
    assert [game.id for game in week4] == [401772001, 401772002, 401772003, 401772004, 401772005, 401772006]
    assert [game.date for game in week4] == sorted(game.date for game in week4)
    first = schedule.first_game(4)
    assert first is not None and first.date == datetime(2026, 10, 2, 0, 15, tzinfo=UTC)  # Thursday night
    assert (first.home_pro_team_id, first.away_pro_team_id) == (8, 22) and first.opponent_of(8) == 22
    assert first.involves(22) and not first.involves(2) and first.opponent_of(2) is None
    assert first.valid_for_locking and not first.start_time_tbd and first.scoring_period_id == 4
    assert week4[-1].date == datetime(2026, 10, 6, 0, 15, tzinfo=UTC)  # Monday night

    assert schedule.has_game(8, 4) and not schedule.has_game(23, 4) and not schedule.has_game(999, 4)
    assert schedule.idle_teams(4) == (23, 25) and schedule.idle_teams(5) == (1, 21)
    atlanta = schedule.team(1)
    assert atlanta is not None and atlanta.bye_week == 5 and atlanta.abbrev == "ATL" and atlanta.games(5) == ()
    assert len(schedule.games_for(14, 5)) == 1 and schedule.games_for(0, 4) == ()
    free_agents = schedule.team(0)
    assert free_agents is not None and free_agents.pro_games_by_scoring_period == {}
    assert schedule.first_game(6) is None and schedule.games(6) == ()

    assert read.capture is not None and read.capture.league_id is None
    assert read.capture.relative_path.startswith("espn/ffl/2026/game/proTeamSchedules_wl/")


def test_nba_first_tip_is_the_days_add_cutoff(nba: EspnClient, espn: FakeEspn) -> None:
    schedule = nba.pro_schedule().data
    assert str(espn.last("proTeamSchedules_wl").url).startswith(f"{API_ROOT}/fba/seasons/{NBA_SEASON}?")
    first = schedule.first_game(1)
    assert first is not None and first.date == datetime(2026, 10, 20, 23, 30, tzinfo=UTC)  # 7:30 p.m. ET
    assert (first.home_pro_team_id, first.away_pro_team_id) == (2, 18)
    assert len(schedule.games(2)) == 2 and schedule.idle_teams(1) == (24, 25)
    assert FBA.pro_team(first.home_pro_team_id) == "BOS"


# --- NBA categories ---------------------------------------------------------------------------------------------------


def test_nba_category_scoreboard(nba: EspnClient, espn: FakeEspn) -> None:
    view = nba.scoreboard(1, scoring_period=1).data
    request = espn.last("mMatchupScore+mScoreboard")
    assert request.url.path == f"/apis/v3/games/fba/seasons/{NBA_SEASON}/segments/0/leagues/{NBA_LEAGUE_ID}"
    assert (view.league_id, view.season_id, view.scoring_period_id) == (NBA_LEAGUE_ID, NBA_SEASON, 1)

    (matchup,) = view.schedule
    assert matchup.home is not None and matchup.away is not None
    score = matchup.home.cumulative_score
    assert score is not None and (score.wins, score.losses, score.ties) == (5, 3, 1)
    assert len(score.score_by_stat) == 9 and set(score.score_by_stat) == {0, 1, 2, 3, 6, 11, 17, 19, 20}
    fg = score.score_by_stat[FBA.stat_id("FG%")]
    assert (fg.score, fg.result, fg.ineligible) == (0.531, "WIN", False)
    assert score.score_by_stat[FBA.stat_id("FT%")].result == "LOSS" and score.score_by_stat[17].result == "TIE"
    assert matchup.away.cumulative_score is not None and matchup.away.cumulative_score.score_by_stat[2].result == "WIN"
    assert matchup.home.games_played == 2

    roster = matchup.home.roster
    assert roster is not None and roster.player_ids == (JOKIC, 3945274)
    jokic = roster.entry(JOKIC)
    assert jokic.lineup_slot_id == FBA.slot_id("C") and jokic.lineup_locked
    day = jokic.player.actual(NBA_SEASON, 1)
    assert day is not None and day.stats[FBA.stat_id("REB")] == 14.0 and day.id == "05401800002"
    assert day.stat_split_type_id == 5 and day.is_single_period  # fba's "Game" split, keyed by pro game id
    assert (
        jokic.player.actual(NBA_SEASON, 1, game="fba") is day and jokic.player.actual(NBA_SEASON, 1, game="ffl") is None
    )
    season = jokic.player.actual(NBA_SEASON, 0)
    assert season is not None and season.is_season and jokic.player.projection(NBA_SEASON, 1) is None


# --- raw views and capture --------------------------------------------------------------------------------------------


def test_get_view_returns_the_raw_object(client: EspnClient, espn: FakeEspn) -> None:
    read = client.get_view(View.TEAM, "mStandings")
    assert isinstance(read.data, dict) and len(read.data["teams"]) == 10
    assert espn.last("mTeam+mStandings").url.params.get_list("view") == ["mTeam", "mStandings"]
    with pytest.raises(ValueError, match="at least one view"):
        client.get_view()


def test_every_response_is_captured_with_metadata(client: EspnClient, cache_root: Path, clock: FakeClock) -> None:
    read = client.rosters(4)
    capture = read.capture
    assert capture is not None
    folder = cache_root / "espn" / "ffl" / "2026" / str(LEAGUE_ID) / "mRoster"
    assert capture.path == folder / "20261004T150000000000Z_sp4.json"
    assert capture.path.read_bytes() == fixture("ffl_rosters_week4.json")
    assert capture.relative_path == f"espn/ffl/2026/{LEAGUE_ID}/mRoster/20261004T150000000000Z_sp4.json"
    assert (capture.kind, capture.status_code, capture.league_id, capture.scoring_period_id) == (
        "mRoster",
        200,
        LEAGUE_ID,
        4,
    )
    assert capture.sha256 == hashlib.sha256(fixture("ffl_rosters_week4.json")).hexdigest()
    assert capture.size_bytes == len(fixture("ffl_rosters_week4.json")) and capture.fetched_at == T0
    assert capture.url == f"https://{READS_HOST}{LEAGUE_PATH}?view=mRoster&scoringPeriodId=4"
    assert capture.params == {"view": ["mRoster"], "scoringPeriodId": 4}

    meta = json.loads(capture.meta_path.read_text(encoding="utf-8"))
    assert capture.meta_path == folder / "20261004T150000000000Z_sp4.meta.json"
    assert meta["url"] == capture.url and meta["params"] == capture.params and meta["sha256"] == capture.sha256
    assert (meta["kind"], meta["bytes"], meta["status_code"], meta["fetched_at"]) == (
        "mRoster",
        capture.size_bytes,
        200,
        T0.isoformat(),
    )
    assert (meta["game"], meta["season"], meta["league_id"], meta["scoring_period_id"]) == ("ffl", 2026, LEAGUE_ID, 4)
    meta_text = capture.meta_path.read_text(encoding="utf-8").lower()
    assert S2 not in meta_text and "cookie" not in meta_text and "swid" not in meta_text

    # Every read is kept, so a second read at the same instant gets a new file rather than overwriting the first.
    again = client.rosters(4).capture
    assert again is not None and again.path == folder / "20261004T150000000000Z_sp4-1.json"
    clock.advance(minutes=1)
    later = client.rosters(5).capture
    assert later is not None and later.path.name == "20261004T150100000000Z_sp5.json"
    assert sorted(p.name for p in folder.glob("*.json") if ".meta." not in p.name) == [
        "20261004T150000000000Z_sp4-1.json",
        "20261004T150000000000Z_sp4.json",
        "20261004T150100000000Z_sp5.json",
    ]


def test_capture_keys_name_the_request(client: EspnClient, cache_root: Path) -> None:
    stamp = "20261004T150000000000Z"

    def name(read: EspnRead[Any]) -> str:
        assert read.capture is not None and read.capture.path.name.startswith(stamp)
        return read.capture.path.name.removeprefix(stamp)

    league = cache_root / "espn" / "ffl" / "2026" / str(LEAGUE_ID)
    assert name(client.teams()) == ".json" and (league / "mTeam_mStandings").is_dir()
    assert name(client.scoreboard(4, scoring_period=4)) == "_mp4_sp4.json"
    assert name(client.free_agents(4, slot_ids=[2, 23], offset=50)) == "_freeagent_waivers_sp4_slots2-23_o50.json"
    cards = name(client.player_cards([GIBBS, CHASE], scoring_period=4))
    assert cards.startswith("_2ids_") and cards.endswith("_sp4.json") and (league / "kona_playercard").is_dir()
    assert name(client.transactions(4)) == "_8types_sp4.json"
    assert name(client.transactions(4, types=["WAIVER"])) == "_waiver_sp4.json"
    assert name(client.pending_transactions()) == "_current.json"
    assert name(client.matchups()) == ".json" and (league / "mMatchup").is_dir()


def test_capture_can_be_disabled(espn: FakeEspn, options: ClientOptions, cache_root: Path) -> None:
    with EspnClient("ffl", LEAGUE_ID, SEASON, SESSION, **{**options, "capture": False}) as client:
        read = client.teams()
    assert read.capture is None and read.as_of == T0 and len(read.data.teams) == 10
    assert not cache_root.exists()


def test_default_cache_root_is_the_cache_dir(clock: FakeClock) -> None:
    client = EspnClient("ffl", LEAGUE_ID, SEASON, clock=clock)
    assert client.cache_root == paths.cache_dir()


# --- failures ---------------------------------------------------------------------------------------------------------


def test_429_is_retried_honoring_retry_after(client: EspnClient, espn: FakeEspn, sleeps: list[float]) -> None:
    espn.queue("mRoster", httpx.Response(429, headers={"Retry-After": "2"}), ok("ffl_rosters_week4.json"))
    assert len(client.rosters(4).data.teams) == 2
    assert sleeps == [2.0] and client.requests == 2 and len(espn.requests["mRoster"]) == 2


def test_5xx_backs_off_exponentially_then_gives_up(client: EspnClient, espn: FakeEspn, sleeps: list[float]) -> None:
    espn.queue("mRoster", httpx.Response(503))
    with pytest.raises(EspnHttpError, match="HTTP 503 .*gave up after 4 attempts") as excinfo:
        client.rosters(4)
    assert excinfo.value.status_code == 503 and "scoringPeriodId=4" in excinfo.value.url
    assert sleeps == [1.0, 2.0, 4.0] and client.requests == 4


def test_transport_errors_are_retried(client: EspnClient, espn: FakeEspn, sleeps: list[float]) -> None:
    espn.queue("mTeam+mStandings", httpx.ConnectError("refused"), ok("ffl_teams.json"))
    assert len(client.teams().data.teams) == 10
    assert sleeps == [1.0]

    espn.queue("mTeam+mStandings", httpx.ReadTimeout("slow"))
    with pytest.raises(EspnHttpError, match="no response .*ReadTimeout") as excinfo:
        client.teams()
    assert excinfo.value.status_code is None and isinstance(excinfo.value, EspnClientError)


def test_other_4xx_raise_at_once_with_espns_detail(client: EspnClient, espn: FakeEspn, sleeps: list[float]) -> None:
    espn.queue("mSettings", espn_error(404, "LEAGUE_NOT_FOUND", "League not found."))
    with pytest.raises(EspnHttpError, match="HTTP 404 .*LEAGUE_NOT_FOUND: League not found") as excinfo:
        client.settings()
    assert excinfo.value.status_code == 404 and sleeps == [] and client.requests == 1

    espn.queue("mSettings", httpx.Response(400, text="<html>bad request</html>"))
    with pytest.raises(EspnHttpError, match="HTTP 400 .*<html>bad request"):
        client.settings()
    espn.queue("mSettings", httpx.Response(400, json={"messages": ["Unknown view."]}))
    with pytest.raises(EspnHttpError, match="Unknown view"):
        client.settings()


def test_401_means_the_session_is_gone(client: EspnClient, espn: FakeEspn, sleeps: list[float]) -> None:
    espn.queue("mRoster", espn_error(401, "AUTH_LEAGUE_NOT_VISIBLE", "You are not authorized to view this League."))
    with pytest.raises(EspnAuthError, match="HTTP 401.*AUTH_LEAGUE_NOT_VISIBLE.*fm login") as excinfo:
        client.rosters(4)
    assert isinstance(excinfo.value, AuthError) and isinstance(excinfo.value, EspnClientError)
    assert sleeps == [] and client.requests == 1

    espn.queue("mRoster", espn_error(403, "AUTH_MISSING_CREDENTIALS", "Missing credentials."))
    with pytest.raises(EspnAuthError, match="HTTP 403"):
        client.rosters(4)


def test_bodies_that_do_not_parse_raise_schema_errors_and_are_kept(
    client: EspnClient, espn: FakeEspn, cache_root: Path
) -> None:
    espn.queue("mTeam+mStandings", httpx.Response(200, text="<html>maintenance</html>"))
    with pytest.raises(EspnSchemaError, match="mTeam\\+mStandings: response is not JSON") as excinfo:
        client.teams()
    saved = str(excinfo.value).rsplit("raw body saved at ", 1)[1]
    assert Path(saved).read_text(encoding="utf-8") == "<html>maintenance</html>"
    assert Path(saved).parent == cache_root / "espn" / "ffl" / "2026" / str(LEAGUE_ID) / "mTeam_mStandings"

    espn.queue("mTeam+mStandings", httpx.Response(200, json={"id": LEAGUE_ID, "teams": "nope"}))
    with pytest.raises(EspnSchemaError, match="did not parse as TeamsView .*teams") as excinfo:
        client.teams()
    assert isinstance(excinfo.value, ValueError) and "raw body saved at" in str(excinfo.value)

    espn.queue("mTeam+mStandings", httpx.Response(200, json=[1, 2]))
    with pytest.raises(EspnSchemaError, match="expected a JSON object, got list"):
        client.teams()


def test_a_single_element_list_is_unwrapped(client: EspnClient, espn: FakeEspn) -> None:
    espn.queue("mTeam+mStandings", httpx.Response(200, content=b"[" + fixture("ffl_teams.json") + b"]"))
    assert len(client.teams().data.teams) == 10


def test_accidental_live_calls_are_not_swallowed(client: EspnClient, espn: FakeEspn) -> None:
    # The harness blocks the network with a bare RuntimeError; the retry loop must never hide one.
    espn.queue("mRoster", RuntimeError("outbound network is disabled in unit tests"))
    with pytest.raises(RuntimeError, match="outbound network") as excinfo:
        client.rosters(4)
    assert not isinstance(excinfo.value, EspnClientError)


# --- models and fixtures ----------------------------------------------------------------------------------------------


def test_models_tolerate_espn_shapes() -> None:
    entry = RosterEntry.model_validate(
        {"lineupSlotId": 20, "player": {"id": 5, "fullName": "Someone", "eligibleSlots": None}, "newField": 1}
    )
    assert entry.player_id == 5 and entry.player.full_name == "Someone" and entry.player.eligible_slots == ()
    assert entry.player_pool_entry.id == 5 and not entry.lineup_locked

    team = Team.model_validate({"id": 9, "owners": None, "record": {"overall": {"wins": 2}}, "valuesByStat": None})
    assert team.display_name == "Team 9" and team.owners == () and team.record.overall.losses == 0
    assert Team.model_validate({"id": 9, "abbrev": "NINE"}).display_name == "NINE"

    stats = PlayerStats.model_validate(
        {"id": 102026, "seasonId": 2026, "stats": {"3": 1, "x": 2}, "appliedStats": None}
    )
    assert stats.id == "102026" and stats.stats == {3: 1.0} and stats.applied_stats == {} and stats.is_season

    transaction = Transaction.model_validate(
        {"id": 4, "type": "WAIVER", "proposedDate": 0, "processDate": None, "items": None}
    )
    assert transaction.id == "4" and transaction.date is None and transaction.items == () and not transaction.pending
    assert transaction.team_ids == () and not transaction.failed
    with pytest.raises(ValueError):
        Transaction.model_validate({"id": 4, "type": "WAIVER", "proposedDate": True})
    with pytest.raises(ValueError):
        Transaction.model_validate({"id": 4, "type": "WAIVER", "proposedDate": 10**30})
    with pytest.raises(ValueError):
        Transaction.model_validate({"id": 4, "type": "WAIVER", "proposedDate": "tomorrow"})


def test_models_are_frozen(client: EspnClient) -> None:
    team = client.teams().data.team(1)
    with pytest.raises(ValueError):
        team.name = "renamed"  # type: ignore[misc]


@pytest.mark.parametrize("name", NEW_FIXTURES)
def test_fixtures_are_scrubbed(name: str) -> None:
    text = fixture(name).decode("utf-8")
    assert "espn_s2" not in text.lower() and "swid" not in text.lower()
    data = json.loads(text)
    for member in data.get("members", []):
        assert member["displayName"].startswith("Manager ") and member["id"].startswith("{00000000-")
    for team in data.get("teams", []):
        assert all(owner.startswith("{00000000-") for owner in team.get("owners", []))
    for transaction in data.get("transactions", data.get("pendingTransactions", [])):
        assert transaction["memberId"].startswith("{00000000-")
