"""The sync job and ``fm sync`` over fixture-backed ESPN, nflverse, Sleeper, stats.nba.com and DARKO (respx, a
patched nflreadpy download and a recorded stats.nba.com transport; no network, no browser).

ESPN views come from tests/fixtures/espn (NFL league 1234567 in week 4 of 2026; the NBA 9-cat league 3456789 on its
opening day, with a one-team roster built from the scoreboard fixture). Player cards are assembled from the roster,
free-agent and player-card fixtures so a card request answers exactly the ids it asked for. The nflverse ID map is the
recorded ff_playerids file plus synthetic rows for the fixture's other rostered players, so the gate passes, and
leaving one out makes it fail. The NBA crosswalk matches Jokic through the recorded stats.nba.com splits
(tests/fixtures/sources/nba_stats), DARKO's talent export (tests/fixtures/sources/darko) and nba_api's bundled table.

Containment: ``fm sync`` harvests the ESPN session from the browser profile and writes the state DB, so a test that
lost the isolation tests/conftest.py sets up would sync a real league. :func:`_contained` refuses to run a test unless
the config and cache dirs are that test's temp dirs, and turns a real browser launch into an error. Nothing here calls
``monkeypatch.undo()``: the fixture is shared with the conftest, and undoing it would drop that isolation mid-test.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, NoReturn, cast

import httpx
import polars as pl
import pytest
import respx
import typer
from nflreadpy.downloader import NflverseDownloader
from typer.testing import CliRunner, Result

from fm import paths
from fm.browser import session as browser_session
from fm.commands import sync as sync_cmd
from fm.config import Config
from fm.espn import client as espn_client
from fm.espn.auth import EspnSession
from fm.espn.client import FILTER_HEADER, READS_HOST, ClientOptions, EspnHttpError
from fm.jobs import sync as sync_job
from fm.jobs.sync import ESPN_SOURCE, KEPT_CROSSWALK, SourceSync, SyncError, SyncReport, select_leagues, sync
from fm.model.ids import GSIS, SLEEPER, Crosswalk, CrosswalkError, UnmappedPlayer, check_rostered
from fm.model.ids_nba import NBA_SOURCE, NbaCrosswalk, UnmappedNbaPlayersError, check_nba_rostered
from fm.proposals.policy import stored_settings
from fm.sources.base import RateLimiter, SourceError
from fm.sources.darko import TALENT_URL, DarkoSource
from fm.sources.nba_stats import NbaStatsSource, NbaStatsTransport
from fm.sources.nflverse import NflreadLoader, NflverseSource
from fm.sources.sleeper import SleeperSource
from fm.store import Store

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
T0 = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)
SESSION = EspnSession(
    espn_s2="s2-secret-value", swid="{TEST-SWID}", expires_at=datetime.now(UTC).replace(microsecond=0) + timedelta(200)
)

NFL_LEAGUE, NFL_SEASON, WEEK = 1234567, 2026, 4
NBA_LEAGUE, NBA_SEASON, NBA_TEAM = 3456789, 2027, 9
ALLEN, GIBBS, CHASE, KELCE, OLAVE, DOWDLE, DAVIS, EAGLES_DST, JOKIC = (
    3918298,
    4429795,
    4362628,
    15847,
    4361370,
    4038815,
    4429160,
    -16021,
    3112335,
)
ROSTERED = 14  # 11 players on team 1 and 3 on team 2 in ffl_rosters_week4.json
POOL = 4  # ffl_free_agents_week4.json
STAT_LINES = 23  # ESPN lines on the rostered and pooled players' cards (season and week 4, projected and actual)
ROSTERED_STAT_LINES = 15  # the rostered players' share; each pooled player carries a season and a week projection
# Rostered in the fixture but absent from the recorded ff_playerids file; the test's ID map adds them.
NOT_IN_ID_MAP = {
    4430807: "Bijan Robinson",
    4361307: "Trey McBride",
    4241416: "Chuba Hubbard",
    3055899: "Harrison Butker",
    4373626: "Tyler Allgeier",
    4249836: "Rashid Shaheed",
    OLAVE: "Chris Olave",
    3916387: "Lamar Jackson",
    3929630: "Saquon Barkley",
    KELCE: "Travis Kelce",
}
NFL_VIEWS = {
    "mSettings": "ffl_settings_ppr.json",
    "mTeam+mStandings": "ffl_teams.json",
    "mRoster": "ffl_rosters_week4.json",
}
SLEEPER_LINES = 10  # stat lines in projections_2026_4.json once ADP-only placeholders are dropped
NBA_SPLITS = FIXTURES / "sources" / "nba_stats" / "leaguedashplayerstats_Base_2025-26.json"  # Jokic, SGA, Wembanyama
DARKO_TALENT = FIXTURES / "sources" / "darko" / "talent.csv"
JOKIC_NBA_ID = 203999

runner = CliRunner()


# --- containment ------------------------------------------------------------------------------------------------------


def _no_browser(*_: object) -> NoReturn:
    raise AssertionError("real browser launch in a unit test; fm sync must get its session from a stub here")


@pytest.fixture(autouse=True)
def _contained(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail unless FM_CONFIG_DIR and FM_CACHE_DIR are this test's temp dirs; make a real browser launch an error."""
    for name in (paths.CONFIG_DIR_ENV, paths.CACHE_DIR_ENV):
        value = os.environ.get(name)
        assert value, f"{name} is not set; tests/conftest.py isolation is missing"
        assert Path(value).resolve().is_relative_to(tmp_path.resolve()), f"{name}={value} is not under {tmp_path}"
    monkeypatch.setattr(browser_session, "sync_playwright", _no_browser)


# --- fixture material -------------------------------------------------------------------------------------------------


def espn_json(name: str) -> dict[str, Any]:
    return json.loads((FIXTURES / "espn" / name).read_text(encoding="utf-8"))


def nfl_card_pool() -> dict[int, dict[str, Any]]:
    """Every player pool entry the NFL fixtures carry, by player id, so a card request can be answered per id."""
    pool: dict[int, dict[str, Any]] = {}
    for team in espn_json("ffl_rosters_week4.json")["teams"]:
        for entry in team["roster"]["entries"]:
            pool[entry["playerId"]] = {
                **entry["playerPoolEntry"],
                "id": entry["playerId"],
                "onTeamId": team["id"],
                "status": "ONTEAM",
            }
    for entry in espn_json("ffl_free_agents_week4.json")["players"]:
        pool[entry["id"]] = entry
    for entry in espn_json("ffl_player_cards_week4.json")["players"]:
        pool[entry["id"]] = entry
    return pool


def synthetic_wire(count: int) -> list[dict[str, Any]]:
    """``count`` free agents cloned from the fixture's first one, with ids from 9,000,000 up."""
    template = espn_json("ffl_free_agents_week4.json")["players"][0]
    wire: list[dict[str, Any]] = []
    for index in range(count):
        espn_id = 9_000_000 + index
        player = {**template["player"], "id": espn_id, "fullName": f"Wire Player {index}"}
        wire.append({**template, "id": espn_id, "player": player})
    return wire


def jokic_entry() -> dict[str, Any]:
    matchup = espn_json("fba_scoreboard_9cat_day1.json")["schedule"][0]
    return matchup["home"]["rosterForCurrentScoringPeriod"]["entries"][0]


def nba_views() -> dict[str, dict[str, Any]]:
    """The NBA league's views: the 9-cat settings fixture plus a hand-built one-team league around Jokic."""
    envelope = {"gameId": 3, "id": NBA_LEAGUE, "seasonId": NBA_SEASON, "scoringPeriodId": 1}
    team = {"id": NBA_TEAM, "name": "Fixture Nine", "abbrev": "FN", "record": {"overall": {"wins": 1, "losses": 0}}}
    return {
        "mSettings": espn_json("fba_settings_9cat.json"),
        "mTeam+mStandings": {**envelope, "teams": [team], "members": []},
        "mRoster": {**envelope, "teams": [{"id": NBA_TEAM, "roster": {"entries": [jokic_entry()]}}]},
    }


def id_map(*, without: Iterable[int] = ()) -> pl.DataFrame:
    """The recorded ff_playerids file plus synthetic ids for the other rostered players (except ``without``)."""
    base = pl.read_csv(FIXTURES / "sources" / "nflverse" / "db_playerids.csv", null_values=["NA", "NULL", ""])
    skipped = set(without)
    extra = [(espn_id, name) for espn_id, name in NOT_IN_ID_MAP.items() if espn_id not in skipped]
    rows = pl.DataFrame(
        {
            "espn_id": [espn_id for espn_id, _ in extra],
            "sleeper_id": [90000 + index for index in range(len(extra))],
            "gsis_id": [f"00-009{index:04d}" for index in range(len(extra))],
            "name": [name for _, name in extra],
        },
        schema={"espn_id": pl.Int64, "sleeper_id": pl.Int64, "gsis_id": pl.String, "name": pl.String},
    )
    return pl.concat([base, rows], how="diagonal")


def no_espn_id_warning() -> str:
    """The crosswalk build's note about the one ID-map row (Trinidad Chambliss) without an ESPN id."""
    return f"ff_playerids: 1 of {id_map().height} rows have no ESPN id (Trinidad Chambliss)"


# --- fakes ------------------------------------------------------------------------------------------------------------


class FakeClock:
    def __init__(self, now: datetime = T0) -> None:
        self.now = now

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> datetime:
        self.now += timedelta(**delta)
        return self.now


class FakeLoader:
    """Only ``load_ff_playerids``; any other nflreadpy call is a test failure."""

    def __init__(self, data: pl.DataFrame) -> None:
        self.data = data
        self.calls = 0
        self.fail_with: Exception | None = None

    def load_ff_playerids(self) -> pl.DataFrame:
        self.calls += 1
        if self.fail_with is not None:
            raise self.fail_with
        return self.data

    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"unexpected loader call {name}")


class SplitsTransport:
    """stats.nba.com's ``leaguedashplayerstats`` from the recorded payload, for whichever season is asked; anything
    else is a test failure. ``fail_with`` makes every call raise instead."""

    def __init__(self) -> None:
        self.seasons: list[str] = []
        self.fail_with: Exception | None = None

    def get(self, endpoint: str, parameters: Mapping[str, object]) -> bytes:
        assert endpoint == "leaguedashplayerstats" and parameters["MeasureType"] == "Base"
        self.seasons.append(str(parameters["Season"]))
        if self.fail_with is not None:
            raise self.fail_with
        return NBA_SPLITS.read_bytes()


def sleeper_ok(name: str) -> httpx.Response:
    return httpx.Response(
        200,
        content=(FIXTURES / "sources" / "sleeper" / name).read_bytes(),
        headers={"content-type": "application/json; charset=utf-8"},
    )


class Routes:
    """ESPN league views from the fixtures (player cards answered per requested id) and Sleeper's projections."""

    def __init__(self, router: respx.MockRouter) -> None:
        self.requests: dict[str, list[httpx.Request]] = defaultdict(list)
        self.answers: dict[str, httpx.Response] = {}
        """View key (``mRoster``, ``kona_playercard``, ...) -> a canned answer in place of the fixtures, both games."""
        self.wire: list[dict[str, Any]] = espn_json("ffl_free_agents_week4.json")["players"]
        """The NFL pool ``kona_player_info`` pages through, most-owned first."""
        self.cards: dict[int, dict[str, Any]] = nfl_card_pool()
        """What an NFL ``kona_playercard`` request answers per id (wire players are added under these)."""
        self.espn = router.get(
            host=READS_HOST, path__regex=r"^/apis/v3/games/(?P<game>ffl|fba)/seasons/\d+/segments/0/leagues/\d+$"
        ).mock(side_effect=self._league)
        self.sleeper = router.get(host="api.sleeper.com", path=f"/projections/nfl/{NFL_SEASON}/{WEEK}").mock(
            return_value=sleeper_ok("projections_2026_4.json")
        )
        self.darko = router.get(TALENT_URL).mock(
            return_value=httpx.Response(200, content=DARKO_TALENT.read_bytes(), headers={"content-type": "text/csv"})
        )

    def _league(self, request: httpx.Request, game: str) -> httpx.Response:
        key = "+".join(request.url.params.get_list("view"))
        self.requests[key].append(request)
        if key in self.answers:
            return self.answers[key]
        players = json.loads(request.headers[FILTER_HEADER])["players"] if FILTER_HEADER in request.headers else {}
        if game == "ffl":
            envelope = {"gameId": 1, "id": NFL_LEAGUE, "seasonId": NFL_SEASON, "scoringPeriodId": WEEK}
            if key == "kona_player_info":
                start = players.get("offset", 0)
                return httpx.Response(200, json={**envelope, "players": self.wire[start : start + players["limit"]]})
            if key == "kona_playercard":
                pool = {**{entry["id"]: entry for entry in self.wire}, **self.cards}
                wanted = players["filterIds"]["value"]
                return httpx.Response(200, json={**envelope, "players": [pool[i] for i in wanted if i in pool]})
            return httpx.Response(200, json=espn_json(NFL_VIEWS[key]))
        envelope = {"gameId": 3, "id": NBA_LEAGUE, "seasonId": NBA_SEASON, "scoringPeriodId": 1}
        if key == "kona_player_info":
            return httpx.Response(200, json={**envelope, "players": []})
        if key == "kona_playercard":
            return httpx.Response(200, json={**envelope, "players": [jokic_entry()["playerPoolEntry"]]})
        return httpx.Response(200, json=nba_views()[key])

    def filters(self, key: str) -> list[dict[str, Any]]:
        return [json.loads(request.headers[FILTER_HEADER])["players"] for request in self.requests[key]]


@pytest.fixture
def routes() -> Iterator[Routes]:
    with respx.mock(assert_all_called=False) as router:  # unmatched requests raise; nothing leaves the process
        yield Routes(router)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def store() -> Iterator[Store]:
    with Store.open() as opened:  # the state DB under the per-test FM_CONFIG_DIR, as the commands open it
        yield opened


@pytest.fixture
def config() -> Config:
    return Config.model_validate(
        {
            "league": [
                {"key": "nfl", "sport": "nfl", "espn_league_id": NFL_LEAGUE, "season": NFL_SEASON, "team_id": 1},
                {"key": "nba", "sport": "nba", "espn_league_id": NBA_LEAGUE, "season": NBA_SEASON, "team_id": NBA_TEAM},
            ]
        }
    )


@pytest.fixture
def espn_options(clock: FakeClock) -> ClientOptions:
    return {"clock": clock, "sleep": lambda _: None, "min_interval_s": 0.0}


@pytest.fixture
def loader() -> FakeLoader:
    return FakeLoader(id_map())


@pytest.fixture
def nflverse(loader: FakeLoader, clock: FakeClock) -> NflverseSource:
    return NflverseSource(loader=cast(NflreadLoader, loader), limiter=RateLimiter(0), clock=clock)


@pytest.fixture
def sleeper(clock: FakeClock) -> Iterator[SleeperSource]:
    with SleeperSource(limiter=RateLimiter(0), clock=clock, sleep=lambda _: None) as source:
        yield source


@pytest.fixture
def splits() -> SplitsTransport:
    return SplitsTransport()


@pytest.fixture
def nba_stats(splits: SplitsTransport, clock: FakeClock) -> NbaStatsSource:
    return NbaStatsSource(transport=cast(NbaStatsTransport, splits), limiter=RateLimiter(0), clock=clock)


@pytest.fixture
def darko(clock: FakeClock) -> Iterator[DarkoSource]:
    with DarkoSource(limiter=RateLimiter(0), clock=clock, sleep=lambda _: None) as source:
        yield source


class Runner:
    """``sync`` with every seam injected; ``run(leagues=...)`` syncs the NFL league by default."""

    def __init__(
        self,
        store: Store,
        config: Config,
        nflverse: NflverseSource,
        sleeper: SleeperSource,
        nba_stats: NbaStatsSource,
        darko: DarkoSource,
        espn_options: ClientOptions,
    ) -> None:
        self.store = store
        self.config = config
        self.nflverse = nflverse
        self.sleeper = sleeper
        self.nba_stats = nba_stats
        self.darko = darko
        self.espn_options = espn_options

    def run(self, leagues: Iterable[str] | str | None = ("nfl",), **options: Any) -> SyncReport:
        return sync(
            self.store,
            self.config,
            session=SESSION,
            leagues=leagues,
            nflverse=self.nflverse,
            sleeper=self.sleeper,
            nba_stats=self.nba_stats,
            darko=self.darko,
            espn_options=self.espn_options,
            **options,
        )


@pytest.fixture
def job(
    routes: Routes,
    store: Store,
    config: Config,
    nflverse: NflverseSource,
    sleeper: SleeperSource,
    nba_stats: NbaStatsSource,
    darko: DarkoSource,
    espn_options: ClientOptions,
) -> Runner:
    return Runner(store, config, nflverse, sleeper, nba_stats, darko, espn_options)


def rostered_ids() -> set[int]:
    return {
        entry["playerId"]
        for team in espn_json("ffl_rosters_week4.json")["teams"]
        for entry in team["roster"]["entries"]
    }


# --- the store after a sync -------------------------------------------------------------------------------------------


def test_sync_populates_the_store_from_fixtures(job: Runner, store: Store) -> None:
    report = job.run()
    assert report.ok and report.gate_errors == () and report.failed == ()
    (league,) = report.leagues
    assert (league.key, league.sport, league.espn_league_id, league.season) == ("nfl", "nfl", NFL_LEAGUE, NFL_SEASON)
    assert (league.scoring_period_id, league.name, league.as_of) == (WEEK, "Fixture League (PPR)", T0)
    assert (league.teams, league.rosters, league.rostered, league.players) == (10, 2, ROSTERED, ROSTERED + POOL)
    assert (league.projections, league.snapshots, league.warnings) == (STAT_LINES, 5, ())
    assert league.describe() == (
        f"nfl: Fixture League (PPR) (ESPN {NFL_LEAGUE}, {NFL_SEASON}), scoring period {WEEK}: 10 teams, "
        f"{ROSTERED} rostered on 2 rosters, {ROSTERED + POOL} players, {STAT_LINES} stat lines, 5 captures"
    )

    row = store.leagues.by_key("nfl")
    assert row is not None and row.row_id == league.league_id
    assert (row.sport, row.espn_league_id, row.season, row.team_id, row.name, row.as_of) == (
        "nfl",
        NFL_LEAGUE,
        NFL_SEASON,
        1,
        "Fixture League (PPR)",
        T0,
    )
    settings = stored_settings(store, row)
    assert settings is not None and settings.points_for("REC") == 1.0 and settings.slot_count("FLEX") == 1
    assert settings.acquisition.budget == 100 and settings.current_scoring_period == WEEK

    teams = store.teams.for_league(row.row_id)
    assert [team.team_id for team in teams] == list(range(1, 11))
    leader = teams[1]
    assert (leader.name, leader.abbrev, leader.wins, leader.losses, leader.points_for) == (
        "Fixture Team 2",
        "FT2",
        3,
        0,
        402.1,
    )
    assert (leader.playoff_seed, leader.waiver_rank, leader.acquisition_budget_spent) == (1, 10, 41)
    assert teams[9].name == "Fixture Ten" and teams[0].acquisition_budget_spent == 23

    ours = store.rosters.team(row.row_id, WEEK, 1)
    assert len(ours) == 11 and len(store.rosters.team(row.row_id, WEEK, 2)) == 3
    assert store.rosters.latest_period(row.row_id) == WEEK
    assert store.rosters.rostered_ids(row.row_id, WEEK) == rostered_ids()
    gibbs = next(entry for entry in ours if entry.espn_id == GIBBS)
    assert (gibbs.lineup_slot_id, gibbs.lineup_locked, gibbs.acquisition_type) == (2, True, "DRAFT")
    assert gibbs.acquisition_date == datetime(2026, 8, 31, 0, 15, tzinfo=UTC) and gibbs.as_of == T0
    assert next(entry for entry in ours if entry.espn_id == OLAVE).lineup_slot_id == 21
    assert not next(entry for entry in ours if entry.espn_id == ALLEN).lineup_locked


def test_players_are_decoded_and_projections_are_abbreviation_keyed_stat_lines(job: Runner, store: Store) -> None:
    job.run()
    gibbs = store.players.get("nfl", GIBBS)
    assert gibbs is not None
    assert (gibbs.full_name, gibbs.position, gibbs.pro_team, gibbs.default_position_id, gibbs.pro_team_id) == (
        "Jahmyr Gibbs",
        "RB",
        "DET",
        2,
        8,
    )
    assert gibbs.eligible_slot_ids == [2, 3, 23, 7, 20, 21] and gibbs.injury_status == "ACTIVE" and gibbs.as_of == T0
    eagles = store.players.get("nfl", EAGLES_DST)
    assert eagles is not None and (eagles.full_name, eagles.position, eagles.pro_team) == ("Eagles D/ST", "D/ST", "PHI")
    olave = store.players.get("nfl", OLAVE)
    assert olave is not None and (olave.injury_status, olave.injured, olave.active) == ("INJURY_RESERVE", True, False)
    dowdle = store.players.get("nfl", DOWDLE)  # from the free-agent pool
    assert dowdle is not None and (dowdle.position, dowdle.pro_team) == ("RB", "CAR")
    pooled = {entry["id"] for entry in espn_json("ffl_free_agents_week4.json")["players"]}
    assert {player.espn_id for player in store.players.many("nfl", rostered_ids() | pooled)} == rostered_ids() | pooled

    week = store.projections.get("nfl", GIBBS, ESPN_SOURCE, NFL_SEASON, WEEK)
    assert week is not None and week.kind == "projected" and week.as_of == T0
    assert week.stats["RY"] == 92.1 and week.stats["REC"] == 4.6 and "24" not in week.stats
    season = store.projections.get("nfl", GIBBS, ESPN_SOURCE, NFL_SEASON, 0)
    assert season is not None and season.stats["RY"] == 1380.0
    actual = store.projections.get("nfl", GIBBS, ESPN_SOURCE, NFL_SEASON, WEEK, kind="actual")
    assert actual is not None and actual.stats["RY"] == 104.0
    assert store.projections.get("nfl", GIBBS, ESPN_SOURCE, NFL_SEASON, 0, kind="actual") is not None
    davis = store.projections.get("nfl", DAVIS, ESPN_SOURCE, NFL_SEASON, WEEK)  # a free agent's line
    assert davis is not None and davis.stats["RY"] == 27.0
    assert store.projections.get("nfl", ALLEN, ESPN_SOURCE, NFL_SEASON, WEEK) is None  # no line in the fixture
    assert len(store.projections.for_period("nfl", NFL_SEASON, WEEK, source=ESPN_SOURCE)) == 9
    assert len(store.projections.for_period("nfl", NFL_SEASON, WEEK, source=SLEEPER)) == SLEEPER_LINES
    assert len(store.projections.for_period("nfl", NFL_SEASON, 0, source=ESPN_SOURCE)) == 9


def test_an_empty_espn_line_is_kept_as_a_projection_of_zero(job: Runner, routes: Routes, store: Store) -> None:
    card = routes.cards[CHASE]
    stats = [
        {**entry, "stats": {}} if (entry["statSourceId"], entry["scoringPeriodId"]) == (1, WEEK) else entry
        for entry in card["player"]["stats"]
    ]
    routes.cards[CHASE] = {**card, "player": {**card["player"], "stats": stats}}  # on bye: ESPN projects nothing
    job.run()
    bye = store.projections.get("nfl", CHASE, ESPN_SOURCE, NFL_SEASON, WEEK)
    assert bye is not None and bye.stats == {}
    season = store.projections.get("nfl", CHASE, ESPN_SOURCE, NFL_SEASON, 0)
    assert season is not None and season.stats  # the other lines are untouched


def test_every_espn_read_is_indexed_as_a_raw_snapshot(job: Runner, store: Store) -> None:
    report = job.run()
    league = report.leagues[0]
    snapshots = store.raw_snapshots.find(ESPN_SOURCE)
    assert [snapshot.kind for snapshot in snapshots] == [
        "mSettings",
        "mTeam+mStandings",
        "mRoster",
        "kona_player_info",
        "kona_playercard",
    ]
    for snapshot in snapshots:
        file = paths.cache_dir() / snapshot.path
        assert snapshot.path.startswith(f"espn/ffl/{NFL_SEASON}/{NFL_LEAGUE}/") and file.is_file()
        assert snapshot.sha256 == hashlib.sha256(file.read_bytes()).hexdigest()
        assert snapshot.size_bytes == file.stat().st_size and snapshot.status_code == 200
        assert (snapshot.league_id, snapshot.fetched_at) == (league.league_id, T0)
        assert snapshot.url is not None and snapshot.url.startswith(f"https://{READS_HOST}/apis/v3/games/ffl/")
    rosters = snapshots[2]
    assert rosters.scoring_period_id == WEEK and rosters.params == {"view": ["mRoster"], "scoringPeriodId": WEEK}
    assert "view=mRoster" in str(rosters.url)
    cards = snapshots[4]
    assert cards.params["filter"]["players"]["filterIds"]["value"][:2] == [ALLEN, GIBBS]
    assert snapshots[0].params == {"view": ["mSettings"]} and snapshots[0].scoring_period_id is None
    settings = store.settings.get(league.league_id)
    assert settings is not None and settings.raw_snapshot_id == snapshots[0].id
    meta = (paths.cache_dir() / rosters.path).with_name(Path(rosters.path).stem + ".meta.json").read_text("utf-8")
    assert "cookie" not in meta.lower() and SESSION.espn_s2 not in meta and SESSION.swid not in meta


# --- crosswalk and gate -----------------------------------------------------------------------------------------------


def test_crosswalk_is_saved_and_the_gate_passes(job: Runner, store: Store) -> None:
    report = job.run()
    walk = Crosswalk.from_store(store)
    assert walk.gsis_id(ALLEN) == "00-0034857" and walk.sleeper_id(EAGLES_DST) == "PHI"
    assert walk.sleeper_id(KELCE) is not None and walk.espn_id(SLEEPER, walk.sleeper_id(KELCE) or "") == KELCE
    assert walk.unmapped(store.rosters.rostered_ids(report.leagues[0].league_id, WEEK)) == []

    (league,) = report.leagues
    assert league.gate is not None and league.gate.passed and league.gate.checked == ROSTERED
    assert league.gate.unmapped == () and league.ok

    crosswalk, projections = report.sources
    assert (crosswalk.source, crosswalk.dataset, crosswalk.key, crosswalk.as_of) == (
        "nflverse",
        "ff_playerids",
        "all",
        T0,
    )
    assert crosswalk.fresh and crosswalk.stored == len(walk) and crosswalk.detail == f"{len(walk)} id mappings saved"
    assert crosswalk.warnings == (no_espn_id_warning(),)
    assert crosswalk.describe() == (
        f"nflverse/ff_playerids[all]: as of 2026-10-04 15:00 UTC, fresh; {len(walk)} id mappings saved"
    )
    assert (projections.source, projections.dataset, projections.key) == ("sleeper", "projections", "regular_2026_w4")
    assert projections.fresh and projections.detail == f"{SLEEPER_LINES} stat lines, {SLEEPER_LINES} stored"
    assert projections.warnings == () and projections.stored == SLEEPER_LINES  # every line maps through the crosswalk
    stored = store.projections.for_period("nfl", NFL_SEASON, WEEK, source=SLEEPER)
    assert len(stored) == SLEEPER_LINES and all(row.as_of == T0 for row in stored)
    allen = store.projections.get("nfl", ALLEN, SLEEPER, NFL_SEASON, WEEK)
    assert allen is not None and allen.stats["PY"] > 0  # keyed by ESPN id and ESPN abbreviation, never points
    assert report.warnings == crosswalk.warnings


def test_gate_fails_naming_the_unmapped_rostered_player(job: Runner, store: Store, loader: FakeLoader) -> None:
    loader.data = id_map(without=[KELCE])
    report = job.run()
    assert not report.ok
    (league,) = report.leagues
    assert report.failed == (league,)
    assert league.gate is not None and not league.gate.passed and league.gate.checked == ROSTERED
    assert league.gate.unmapped == (UnmappedPlayer(KELCE, (GSIS, SLEEPER), "Travis Kelce", "TE", "KC"),)
    (error,) = report.gate_errors
    assert error is league.gate.error
    assert str(error) == (
        f"rostered players in league {league.league_id} (scoring period {WEEK}): 1 unmapped; "
        "add rows to id_overrides.csv and sync again:\n"
        f"  Travis Kelce (TE KC, ESPN {KELCE}): no gsis, sleeper id"
    )
    # Everything else was still synced and reported: the fix is an override line, not a re-read.
    assert len(store.rosters.team(league.league_id, WEEK, 2)) == 3 and store.players.get("nfl", KELCE) is not None
    assert Crosswalk.from_store(store).sleeper_id(KELCE) is None and len(report.sources) == 2


def test_the_gate_is_picked_by_the_stored_league_sport(
    job: Runner, store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    checked: dict[str, list[int]] = {"nfl": [], "nba": []}
    real_nfl, real_nba = sync_job.check_rostered, sync_job.check_nba_rostered

    def recording_nfl(store: Store, league_id: int, **options: Any) -> set[int]:
        checked["nfl"].append(league_id)
        return real_nfl(store, league_id, **options)

    def recording_nba(store: Store, league_id: int, **options: Any) -> set[int]:
        checked["nba"].append(league_id)
        return real_nba(store, league_id, **options)

    monkeypatch.setattr(sync_job, "check_rostered", recording_nfl)
    monkeypatch.setattr(sync_job, "check_nba_rostered", recording_nba)
    report = job.run(leagues=None)
    nfl, nba = report.leagues
    assert checked == {"nfl": [nfl.league_id], "nba": [nba.league_id]}  # each gate sees its own sport only
    assert nfl.gate is not None and nfl.gate.passed and nba.gate is not None and nba.gate.passed
    row = store.leagues.get(nba.league_id)
    assert row is not None and (row.sport, nba.sport) == ("nba", "nba")
    with pytest.raises(CrosswalkError, match="is an nba league"):  # what the dispatch keeps each gate away from
        check_rostered(store, nba.league_id)
    with pytest.raises(CrosswalkError, match="is an nfl league"):
        check_nba_rostered(store, nfl.league_id)


def test_a_broken_overrides_file_fails_naming_its_line(job: Runner, store: Store, tmp_path: Path) -> None:
    overrides = tmp_path / "id_overrides.csv"
    overrides.write_text(
        "espn_id,source,source_id,name,note\n15847,sleeper,not-an-id,Travis Kelce,typo\n", encoding="utf-8"
    )
    with pytest.raises(CrosswalkError, match=r"id_overrides\.csv:2: 'not-an-id' does not look like a sleeper id"):
        job.run(overrides=overrides)
    assert store.leagues.by_key("nfl") is not None  # the league itself was synced before the crosswalk was built
    assert len(Crosswalk.from_store(store)) == 0


# --- source freshness -------------------------------------------------------------------------------------------------


def test_sleeper_degradation_and_staleness_are_surfaced(job: Runner, routes: Routes, clock: FakeClock) -> None:
    routes.sleeper.mock(return_value=sleeper_ok("projections_legacy_junk.json"))  # a 200 full of ADP junk
    degraded = job.run().sources[1]
    assert isinstance(degraded, SourceSync) and degraded.degraded and not degraded.stale
    assert degraded.detail == "0 stat lines, 0 stored" and "expected a JSON list" in degraded.warnings[0]
    assert degraded.state.startswith("DEGRADED") and "DEGRADED" in degraded.describe()

    routes.sleeper.mock(return_value=sleeper_ok("projections_2026_4.json"))
    assert job.run().sources[1].fresh

    clock.advance(minutes=31)  # past Sleeper's projections TTL, and the endpoint breaks again
    routes.sleeper.mock(return_value=sleeper_ok("projections_legacy_junk.json"))
    report = job.run()
    stale = report.sources[1]
    assert (stale.stale, stale.cached, stale.degraded, stale.as_of) == (True, True, False, T0)
    assert stale.detail == f"{SLEEPER_LINES} stat lines, {SLEEPER_LINES} stored"  # the last good copy
    assert "did not parse" in stale.warnings[0]
    assert stale.state.startswith("STALE") and stale.describe().startswith(
        "sleeper/projections[regular_2026_w4]: as of 2026-10-04 15:00 UTC, STALE"
    )
    assert report.ok and stale.warnings[0] in report.warnings  # reported, not a failure


def test_a_stale_crosswalk_still_saves_and_gates(
    job: Runner, loader: FakeLoader, clock: FakeClock, store: Store
) -> None:
    first = job.run().sources[0]
    assert first.fresh and loader.calls == 1

    clock.advance(hours=25)  # past the 24 h TTL
    loader.fail_with = ConnectionError("Failed to download: offline")
    report = job.run()
    crosswalk = report.sources[0]
    assert (crosswalk.stale, crosswalk.cached, crosswalk.as_of, loader.calls) == (True, True, T0, 2)
    assert "offline" in crosswalk.warnings[0] and crosswalk.warnings[1:] == first.warnings  # adapter's, then build's
    assert crosswalk.stored == first.stored and len(Crosswalk.from_store(store)) == first.stored
    assert report.ok and report.leagues[0].gate is not None and report.leagues[0].gate.passed


def test_force_refreshes_cached_sources(job: Runner, loader: FakeLoader, routes: Routes) -> None:
    job.run()
    cached = job.run()
    assert all(source.cached for source in cached.sources) and loader.calls == 1 and routes.sleeper.call_count == 1
    assert [source.state for source in cached.sources] == ["cached", "cached"]
    forced = job.run(force=True)
    assert all(source.fresh for source in forced.sources) and loader.calls == 2 and routes.sleeper.call_count == 2


# --- shape of the job -------------------------------------------------------------------------------------------------


def test_sync_is_idempotent_and_replaces_rather_than_duplicates(job: Runner, store: Store, clock: FakeClock) -> None:
    first = job.run().leagues[0]
    later = clock.advance(minutes=10)
    again = job.run().leagues[0]
    assert again.league_id == first.league_id and again.as_of == later
    assert (again.teams, again.rosters, again.rostered, again.players, again.projections) == (
        first.teams,
        first.rosters,
        first.rostered,
        first.players,
        first.projections,
    )
    assert (
        len(store.rosters.team(first.league_id, WEEK, 1)) == 11 and len(store.teams.for_league(first.league_id)) == 10
    )
    assert len(store.raw_snapshots.find(ESPN_SOURCE)) == 10  # every read is kept
    gibbs = store.players.get("nfl", GIBBS)
    assert gibbs is not None and gibbs.as_of == later


def test_a_failed_read_leaves_the_previous_sync_in_place(
    job: Runner, routes: Routes, store: Store, clock: FakeClock
) -> None:
    first = job.run().leagues[0]
    clock.advance(minutes=10)
    routes.answers["kona_playercard"] = httpx.Response(400, json={"messages": ["bad filter"]})
    with pytest.raises(EspnHttpError, match="HTTP 400"):
        job.run()
    row = store.leagues.get(first.league_id)
    assert row is not None and row.as_of == T0  # reads first, then one transaction: nothing of the second sync landed
    assert len(store.raw_snapshots.find(ESPN_SOURCE)) == first.snapshots
    gibbs = store.players.get("nfl", GIBBS)
    assert gibbs is not None and gibbs.as_of == T0
    assert len(store.rosters.team(first.league_id, WEEK, 1)) == 11


def test_pool_size_zero_syncs_rosters_only(job: Runner, routes: Routes, store: Store) -> None:
    league = job.run(pool_size=0).leagues[0]
    assert routes.requests["kona_player_info"] == [] and league.players == ROSTERED
    assert league.projections == ROSTERED_STAT_LINES
    assert len(routes.requests["kona_playercard"]) == 1 and store.players.get("nfl", DOWDLE) is None
    with pytest.raises(ValueError, match="pool_size must be >= 0"):
        job.run(pool_size=-1)


def test_the_pool_is_paged_up_to_the_requested_size(job: Runner, routes: Routes, store: Store) -> None:
    job.run(pool_size=120)  # the fixture wire holds 4 players, so paging stops after the first page
    pages = routes.filters("kona_player_info")
    assert [(page["limit"], page["offset"]) for page in pages] == [(50, 0)]
    assert pages[0]["filterStatus"] == {"value": ["FREEAGENT", "WAIVERS"]}

    routes.wire = synthetic_wire(53)
    routes.requests.clear()
    league = job.run(pool_size=60).leagues[0]
    assert [(page["limit"], page["offset"]) for page in routes.filters("kona_player_info")] == [(50, 0), (10, 50)]
    assert league.players == ROSTERED + 53 and store.players.get("nfl", 9_000_052) is not None
    batches = [batch["filterIds"]["value"] for batch in routes.filters("kona_playercard")]
    assert [len(batch) for batch in batches] == [40, ROSTERED + 53 - 40] and batches[0][0] == ALLEN

    routes.requests.clear()
    job.run(pool_size=50)  # a full last page asks for nothing more
    assert [(page["limit"], page["offset"]) for page in routes.filters("kona_player_info")] == [(50, 0)]


def test_nba_league_syncs_espn_state_and_gates_through_the_nba_crosswalk(
    job: Runner, store: Store, splits: SplitsTransport, routes: Routes
) -> None:
    report = job.run(leagues=["nba"])
    assert report.ok
    (league,) = report.leagues
    assert (league.sport, league.espn_league_id, league.scoring_period_id) == ("nba", NBA_LEAGUE, 1)
    assert league.gate is not None and league.gate.passed and league.gate.checked == 1
    (crosswalk,) = report.sources
    assert (crosswalk.source, crosswalk.dataset, crosswalk.key) == ("nba_stats+darko", "crosswalk", "2026-27+2025-26")
    assert crosswalk.fresh and crosswalk.stored == 1 and crosswalk.detail == "1 id mappings saved"
    assert splits.seasons == ["2026-27", "2025-26"] and routes.darko.call_count == 1  # this season and last
    assert NbaCrosswalk.from_store(store).nba_id(JOKIC) == JOKIC_NBA_ID
    assert [row.source for row in store.player_ids.for_player("nba", JOKIC)] == [NBA_SOURCE]
    assert (league.teams, league.rosters, league.rostered, league.players, league.projections) == (1, 1, 1, 1, 2)
    assert league.name == "Fixture League (NBA 9-cat)" and league.warnings == ()
    row = store.leagues.by_key("nba")
    assert row is not None and row.sport == "nba"
    settings = stored_settings(store, row)
    assert settings is not None and settings.is_categories and settings.team_count == 10
    jokic = store.players.get("nba", JOKIC)
    assert jokic is not None and (jokic.position, jokic.pro_team) == ("C", "DEN")
    day = store.projections.get("nba", JOKIC, ESPN_SOURCE, NBA_SEASON, 1, kind="actual")
    assert day is not None and day.stats["REB"] == 14.0 and day.stats["PTS"] == 29.0
    assert store.projections.get("nba", JOKIC, ESPN_SOURCE, NBA_SEASON, 0, kind="actual") is not None
    assert store.projections.get("nba", JOKIC, ESPN_SOURCE, NBA_SEASON, 1) is None  # no projection in the fixture
    assert len(store.rosters.team(row.row_id, 1, NBA_TEAM)) == 1
    assert sync_cmd.render(report)[-1] == "gate nba: 1 rostered players mapped"


def test_nba_gate_fails_naming_the_unmapped_rostered_player(job: Runner, store: Store, tmp_path: Path) -> None:
    overrides = tmp_path / "id_overrides_nba.csv"
    overrides.write_text(f"espn_id,source,source_id,name,note\n{JOKIC},nba,,Nikola Jokic,held back\n", "utf-8")
    report = job.run(leagues=["nba"], nba_overrides=overrides)
    assert not report.ok
    (league,) = report.leagues
    assert report.failed == (league,) and league.gate is not None and not league.gate.passed
    (error,) = report.gate_errors
    assert isinstance(error, UnmappedNbaPlayersError) and [player.espn_id for player in error.unmapped] == [JOKIC]
    assert "add rows to id_overrides_nba.csv" in str(error) and "keeps him unmapped" in str(error)
    assert store.players.get("nba", JOKIC) is not None  # the ESPN state is kept; the fix is an override line


def test_a_degraded_nba_crosswalk_keeps_the_one_an_earlier_sync_saved(
    job: Runner, store: Store, splits: SplitsTransport, routes: Routes, clock: FakeClock
) -> None:
    first = job.run(leagues=["nba"])
    saved = NbaCrosswalk.from_store(store).rows
    assert first.ok and len(saved) == 1 and saved[0].as_of == T0

    clock.advance(hours=25)
    for source in ("nba_stats", "darko"):  # no cached copy left to serve stale
        shutil.rmtree(paths.cache_dir() / "sources" / source)
    splits.fail_with = SourceError("Akamai read timeout")
    routes.darko.mock(return_value=httpx.Response(500))
    report = job.run(leagues=["nba"])
    (crosswalk,) = report.sources
    assert crosswalk.degraded and (crosswalk.stored, crosswalk.detail) == (0, KEPT_CROSSWALK)
    assert any("unavailable, matching without it" in warning for warning in crosswalk.warnings)
    assert NbaCrosswalk.from_store(store).rows == saved  # not replaced by the degraded build
    assert report.ok and report.leagues[0].gate is not None and report.leagues[0].gate.passed


def test_a_degraded_nba_crosswalk_is_saved_when_none_was(
    job: Runner, store: Store, splits: SplitsTransport, routes: Routes
) -> None:
    splits.fail_with = SourceError("Akamai read timeout")
    routes.darko.mock(return_value=httpx.Response(500))
    report = job.run(leagues=["nba"])
    (crosswalk,) = report.sources
    assert crosswalk.degraded and crosswalk.stored == 1 and crosswalk.detail == "1 id mappings saved"
    assert NbaCrosswalk.from_store(store).nba_id(JOKIC) == JOKIC_NBA_ID  # nba_api's bundled table still knows him
    assert report.ok


def test_every_league_by_default_and_unknown_keys_fail(job: Runner, config: Config) -> None:
    report = job.run(leagues=None)
    assert [league.key for league in report.leagues] == ["nfl", "nba"] and report.ok
    assert [source.source for source in report.sources] == ["nflverse", "sleeper", "nba_stats+darko"]  # once each
    assert [league.key for league in job.run(leagues=["nfl", "nfl"]).leagues] == ["nfl"]
    assert [league.key for league in job.run(leagues="nba").leagues] == ["nba"]  # a bare key is one key
    with pytest.raises(SyncError, match="no league 'mlb' in config.toml; known: nfl, nba"):
        job.run(leagues=["mlb"])
    assert [league.key for league in select_leagues(config, ["nba", "nfl"])] == ["nba", "nfl"]


def test_our_team_missing_from_the_league_is_a_warning(job: Runner) -> None:
    job.config = Config.model_validate(
        {"league": [{"key": "nfl", "sport": "nfl", "espn_league_id": NFL_LEAGUE, "season": NFL_SEASON, "team_id": 11}]}
    )
    report = job.run()
    assert report.ok and report.leagues[0].warnings == (
        f"team 11 is not in ESPN league {NFL_LEAGUE} (teams: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10); "
        "check team_id in config.toml",
    )
    assert report.warnings == (*report.leagues[0].warnings, no_espn_id_warning())


def test_a_missing_scoring_period_is_an_error(job: Runner, routes: Routes, store: Store) -> None:
    settings = {key: value for key, value in espn_json("ffl_settings_ppr.json").items() if key != "scoringPeriodId"}
    teams = {key: value for key, value in espn_json("ffl_teams.json").items() if key != "scoringPeriodId"}
    routes.answers["mSettings"] = httpx.Response(200, json=settings)
    routes.answers["mTeam+mStandings"] = httpx.Response(200, json=teams)
    with pytest.raises(SyncError, match="nfl: ESPN did not report the current scoring period"):
        job.run()
    assert store.leagues.by_key("nfl") is None  # reads first, then one transaction: nothing was written


# --- fm sync ----------------------------------------------------------------------------------------------------------


def _root() -> None:
    pass


def cli() -> typer.Typer:
    root = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)
    root.callback()(_root)
    sync_cmd.register(root)
    return root


def run_cli(*args: str, expect: int = 0) -> Result:
    result = runner.invoke(cli(), ["sync", *args], catch_exceptions=False)
    assert result.exit_code == expect, result.output
    return result


def write_config() -> None:
    paths.config_file().write_text(
        f'[[league]]\nkey = "nfl"\nsport = "nfl"\nespn_league_id = {NFL_LEAGUE}\nseason = {NFL_SEASON}\nteam_id = 1\n',
        encoding="utf-8",
    )


@pytest.fixture
def logged_in(monkeypatch: pytest.MonkeyPatch) -> None:
    """A saved ESPN session without a browser (the command's ``load_session`` returns the test session), and no
    pacing between the command's ESPN requests."""
    monkeypatch.setattr(sync_cmd, "load_session", lambda: SESSION)
    monkeypatch.setattr(espn_client, "DEFAULT_MIN_INTERVAL_S", 0.0)


class RecordedIdMap:
    """Serves nflreadpy's ff_playerids download from the test's ID map, so the real ``NflverseSource`` works offline."""

    def __init__(self, frame: pl.DataFrame) -> None:
        self.frame = frame
        self.urls: list[str] = []

    def __call__(self, url: str, **kwargs: object) -> pl.DataFrame:
        self.urls.append(url)
        assert url.endswith("db_playerids.csv"), f"unexpected nflreadpy download {url}"
        return self.frame


@pytest.fixture
def recorded(monkeypatch: pytest.MonkeyPatch) -> RecordedIdMap:
    served = RecordedIdMap(id_map())
    monkeypatch.setattr(NflverseDownloader, "_download_file", served)
    return served


def test_cli_help_lists_the_options() -> None:
    out = " ".join(run_cli("--help").stdout.split())
    assert "--league" in out and "--force" in out and "--pool" in out
    assert "unmapped-rostered-player gate" in out


def test_cli_syncs_and_reports(routes: Routes, recorded: RecordedIdMap, logged_in: None) -> None:
    write_config()
    result = run_cli()
    lines = result.stdout.strip().splitlines()
    assert lines[0].startswith("ESPN session OK until ")
    assert lines[1] == (
        f"nfl: Fixture League (PPR) (ESPN {NFL_LEAGUE}, {NFL_SEASON}), scoring period {WEEK}: 10 teams, "
        f"{ROSTERED} rostered on 2 rosters, {ROSTERED + POOL} players, {STAT_LINES} stat lines, 5 captures"
    )
    assert lines[2].startswith("nflverse/ff_playerids[all]: as of ") and ", fresh; " in lines[2]
    assert lines[2].endswith(" id mappings saved")
    assert lines[3] == f"  warning: {no_espn_id_warning()}"
    assert lines[4].startswith("sleeper/projections[regular_2026_w4]: as of ")
    assert lines[4].endswith(f", fresh; {SLEEPER_LINES} stat lines, {SLEEPER_LINES} stored")
    assert lines[5] == f"gate nfl: {ROSTERED} rostered players mapped" and len(lines) == 6
    assert result.stderr == "" and recorded.urls and routes.sleeper.call_count == 1
    with Store.open() as store:
        row = store.leagues.by_key("nfl")
        assert row is not None and store.rosters.rostered_ids(row.row_id, WEEK) == rostered_ids()
        assert len(store.raw_snapshots.find(ESPN_SOURCE)) == 5 and len(Crosswalk.from_store(store)) > 0
    assert (paths.cache_dir() / "sources" / "nflverse" / "ff_playerids" / "all.parquet").is_file()

    again = run_cli("--league", "nfl", "--pool", "0").stdout.strip().splitlines()
    assert f"{ROSTERED} players, {ROSTERED_STAT_LINES} stat lines, 4 captures" in again[1]
    assert ", cached; " in again[2] and ", cached; " in again[4]


def test_cli_exits_one_on_an_unmapped_rostered_player(routes: Routes, recorded: RecordedIdMap, logged_in: None) -> None:
    write_config()
    recorded.frame = id_map(without=[KELCE])
    result = run_cli(expect=1)
    assert f"gate nfl: FAILED, 1 of {ROSTERED} rostered players unmapped" in result.stdout
    assert result.stderr.startswith(
        f"error: nfl: rostered players in league 1 (scoring period {WEEK}): 1 unmapped; add rows to id_overrides.csv"
    )
    assert f"  Travis Kelce (TE KC, ESPN {KELCE}): no gsis, sleeper id" in result.stderr
    with Store.open() as store:
        assert store.players.get("nfl", KELCE) is not None  # the state is kept; the fix is an override line


def test_cli_surfaces_a_degraded_source(routes: Routes, recorded: RecordedIdMap, logged_in: None) -> None:
    write_config()
    routes.sleeper.mock(return_value=sleeper_ok("projections_legacy_junk.json"))  # HTTP 200, junk body
    lines = run_cli().stdout.strip().splitlines()  # a degraded source is reported, not a failure
    (sleeper_line,) = [line for line in lines if line.startswith("sleeper/projections[")]
    assert sleeper_line.endswith(", DEGRADED (nothing usable; see warnings); 0 stat lines, 0 stored")
    warning = lines[lines.index(sleeper_line) + 1]
    assert warning.startswith("  warning: sleeper/projections[regular_2026_w4]: payload did not parse")


def test_cli_needs_a_config_and_known_league_keys(logged_in: None) -> None:
    result = run_cli(expect=1)
    assert result.stderr.startswith("error: ") and "config.toml not found" in result.stderr
    write_config()
    result = run_cli("--league", "mlb", expect=1)
    assert result.stderr == "error: no league 'mlb' in config.toml; known: nfl\n"
    assert not paths.state_db().exists()  # refused before the store opened


def test_cli_needs_a_saved_session() -> None:
    write_config()  # the real load_session: there is no browser profile under this test's config dir
    assert not paths.browser_profile_dir().exists()
    result = run_cli(expect=1)
    assert result.stderr == "error: no browser profile yet; run `fm login`\n"
    assert not paths.state_db().exists()


def test_a_profile_never_starts_a_real_browser_here() -> None:
    write_config()
    profile = paths.browser_profile_dir()
    profile.mkdir()
    (profile / "Local State").write_text("{}", encoding="utf-8")  # a profile in this test's temp config dir
    with pytest.raises(AssertionError, match="real browser launch in a unit test"):  # _contained stops the launch
        runner.invoke(cli(), ["sync"], catch_exceptions=False)
    assert not paths.state_db().exists()


def test_cli_reports_espn_failures(routes: Routes, recorded: RecordedIdMap, logged_in: None) -> None:
    write_config()
    routes.answers["mSettings"] = httpx.Response(401, json={"messages": ["Not authorized."]})
    result = run_cli(expect=1)
    assert result.stderr.startswith("error: ESPN rejected the session (HTTP 401") and "fm login" in result.stderr
