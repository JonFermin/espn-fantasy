"""No-history NBA players downstream of the gate (ROADMAP #57, DESIGN sections 7, 8.1 and 9.3).

A rostered player nba.com lists nobody for (a rookie it has not added yet) passes the NBA gate without an nba.com id.
These tests follow him from the crosswalk through the day's projection blend into the daily lineup and streaming:
the sources keyed by nba.com id (DARKO here) skip him, ESPN's projection alone values him, and a player no source
projects is held at an unknown value rather than a made-up one. Nothing crashes on the way.

The league is the real NBA league's settings (tests/fixtures/espn/real/fba/mSettings.json: H2H points, PG SG SF PF C G
F, three UTIL, three bench) with a synthetic roster of thirteen: eleven veterans with nba.com ids (a hand-built person
list stands in for stats.nba.com and DARKO's sheet), ``Rookie One`` with only ESPN's season projection and ``Rookie
Two`` with none. Two free agents: a veteran and a third rookie. DARKO is a fake adapter and the pro schedule a test
double with the surface of ``fm.sports.base.ScheduleLike`` (one 7:30 p.m. ET game per team on the days it plays; Tue
Oct 20, 2026 is day 1, and the decisions run on day 3's morning, when teams 2, 5 and 10 are off). No network.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import cast

import pytest

from fm.decide.lineup_daily import plan_daily_lineup
from fm.decide.streaming import plan_streaming
from fm.decide.waivers import Wire, WireEntry, WireStatus
from fm.espn.ids import FBA
from fm.espn.settings import LeagueSettings, load_league_settings
from fm.model.ids_nba import NbaCrosswalk, NbaPerson, build_nba_crosswalk, nba_gate
from fm.model.projections import DARKO, ESPN, BlendWeights, ProjectionSourceRegistry
from fm.model.scoring import Scorer
from fm.model.value_nba import DarkoLoader, blend_day, per_game_lines
from fm.sources.base import Fetched
from fm.sources.darko import DarkoProjection, DarkoSource
from fm.sports.nba import NBA
from fm.store import LeagueRow, LeagueSettingsRow, PlayerRow, ProjectionRow, RosterEntryRow, Store, TeamRow

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SEASON = 2027
OUR_TEAM, THEIR_TEAM = 1, 2
PG, SG, SF, PF, C, G, F, UTIL = (FBA.slot_id(label) for label in ("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"))
BENCH = FBA.bench_slot
SYNCED = datetime(2026, 10, 22, 11, 0, tzinfo=UTC)  # 7 a.m. ET Thursday, day 3
MORNING = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)  # 8 a.m. ET, before the day's tips
DAY = 3
GAMES = 70.0  # the games in every ESPN season projection here
DARKO_EXTRA = 2.0  # DARKO projects every veteran this many points a game above ESPN, so a blend shows both sources

# (espn_id, name, position, pro team, slot, ESPN points per game or None for no projection)
VETERANS = [
    (101, "Pat Guard", "PG", 1, PG, 30.0),
    (102, "Sam Wing", "SG", 2, SG, 28.0),
    (103, "Sid Small", "SF", 3, SF, 26.0),
    (104, "Pete Power", "PF", 4, PF, 24.0),
    (105, "Cal Center", "C", 5, C, 22.0),
    (106, "Gus Shooter", "SG", 6, G, 20.0),
    (107, "Fred Forward", "SF", 7, F, 18.0),
    (108, "Uri Util", "PG", 8, UTIL, 16.0),
    (109, "Una Util", "PF", 9, UTIL, 14.0),
    (110, "Ulf Util", "C", 10, UTIL, 12.0),
    (111, "Ben Bench", "SG", 11, BENCH, 8.0),
]
ROOKIE_ONE, ROOKIE_TWO, ROOKIE_THREE, FREE_VET = 201, 202, 302, 301
ROOKIES = [
    (ROOKIE_ONE, "Rookie One", "PG", 12, BENCH, 9.0),  # ESPN projects him; nba.com lists nobody by his name
    (ROOKIE_TWO, "Rookie Two", "SF", 13, BENCH, None),  # nobody projects him
]
FREE_AGENTS = [(FREE_VET, "Free Vet", "C", 22, 10.0), (ROOKIE_THREE, "Rookie Three", "PG", 20, 12.0)]
WITH_HISTORY = {espn_id for espn_id, *_ in VETERANS} | {FREE_VET}


def nba_id(espn_id: int) -> int:
    """A made-up nba.com person id for a player with history."""
    return 1_700_000 + espn_id


# --- the pro schedule double ------------------------------------------------------------------------------------------


def tip(day: int) -> datetime:
    return datetime(2026, 10, 20, 23, 30, tzinfo=UTC) + timedelta(days=day - 1)


@dataclass(frozen=True)
class FakeGame:
    id: int | None
    date: datetime
    home_pro_team_id: int
    away_pro_team_id: int
    start_time_tbd: bool = False
    valid_for_locking: bool = True


class FakeSchedule:
    """``plays`` maps a day to the pro teams with a game, each at home against the pseudo-team 99."""

    def __init__(self, plays: Mapping[int, Iterable[int]], *, teams: Iterable[int] = range(1, 31)) -> None:
        self.teams = tuple(teams)
        self._games = {
            day: [FakeGame(team * 1000 + day, tip(day), team, 99) for team in sorted(set(playing))]
            for day, playing in plays.items()
        }

    @property
    def scoring_periods(self) -> tuple[int, ...]:
        return tuple(sorted(self._games))

    def games(self, scoring_period: int) -> tuple[FakeGame, ...]:
        return tuple(sorted(self._games.get(scoring_period, ()), key=lambda game: game.date))

    def games_for(self, pro_team_id: int, scoring_period: int) -> tuple[FakeGame, ...]:
        return tuple(game for game in self.games(scoring_period) if game.home_pro_team_id == pro_team_id)

    def idle_teams(self, scoring_period: int) -> tuple[int, ...]:
        playing = {game.home_pro_team_id for game in self.games(scoring_period)}
        return tuple(team for team in self.teams if team not in playing)


ALL = set(range(1, 31))
WEEK = {1: ALL, 2: ALL, 3: ALL - {2, 5, 10}, 4: ALL, 5: ALL, 6: ALL}


def schedule() -> FakeSchedule:
    return FakeSchedule(WEEK)


# --- the fake DARKO adapter -------------------------------------------------------------------------------------------


class FakeDarko:
    """What :class:`fm.model.value_nba.DarkoLoader` asks of the DARKO adapter: ``projections`` and ``clock``."""

    def __init__(self, rows: list[DarkoProjection]) -> None:
        self.rows = rows

    def clock(self) -> datetime:
        return SYNCED

    def projections(self, **_: object) -> Fetched[list[DarkoProjection]]:
        return Fetched(self.rows, SYNCED, DARKO, "talent", "current")


def darko_row(espn_id: int, name: str, ppg: float) -> DarkoProjection:
    """A talent row worth ``ppg + DARKO_EXTRA`` points a game: 36 minutes at a pace of 100 is 75 possessions."""
    return DarkoProjection(
        nba_id=nba_id(espn_id),
        name=name,
        minutes=36.0,
        pace=100.0,
        per_100={"pts": (ppg + DARKO_EXTRA) / 0.75},
        rates={},
        dpm={},
    )


# --- the store --------------------------------------------------------------------------------------------------------


def real_points() -> LeagueSettings:
    return load_league_settings(FIXTURES / "espn" / "real" / "fba" / "mSettings.json")


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        yield opened


def player_row(espn_id: int, name: str, position: str, team: int) -> PlayerRow:
    return PlayerRow(
        sport="nba",
        espn_id=espn_id,
        full_name=name,
        default_position_id=FBA.position_id(position),
        position=position,
        pro_team_id=team,
        pro_team=FBA.pro_teams[team],
        eligible_slot_ids=sorted(NBA.eligible_slots(position, include_reserve=False)),
        as_of=SYNCED,
    )


def espn_season_row(espn_id: int, ppg: float) -> ProjectionRow:
    """ESPN's season projection as the sync stores it (period 0): totals over ``GAMES`` games."""
    return ProjectionRow(
        sport="nba",
        espn_id=espn_id,
        source=ESPN,
        kind="projected",
        season=SEASON,
        scoring_period_id=0,
        stats={"PTS": ppg * GAMES, "GP": GAMES},
        as_of=SYNCED,
    )


@dataclass(frozen=True)
class Synced:
    league: LeagueRow
    crosswalk: NbaCrosswalk
    darko: FakeDarko


def seed(store: Store) -> Synced:
    """What ``fm sync`` leaves for the league: settings, our roster for day 3, the players, ESPN's season projections
    and the NBA crosswalk built from the persons nba.com lists (none for the rookies), saved."""
    settings = real_points()
    league = store.leagues.upsert(
        LeagueRow(
            key="nba", sport="nba", espn_league_id=settings.league_id, season=SEASON, team_id=OUR_TEAM, as_of=SYNCED
        )
    )
    store.settings.upsert(
        LeagueSettingsRow(league_id=league.row_id, settings=settings.model_dump(mode="json"), as_of=SYNCED)
    )
    for team_id in (OUR_TEAM, THEIR_TEAM):
        store.teams.upsert(TeamRow(league_id=league.row_id, team_id=team_id, name=f"Team {team_id}", as_of=SYNCED))
    rows: list[PlayerRow] = []
    entries: list[RosterEntryRow] = []
    for espn_id, name, position, team, slot, ppg in [*VETERANS, *ROOKIES]:
        rows.append(player_row(espn_id, name, position, team))
        entries.append(
            RosterEntryRow(
                league_id=league.row_id,
                scoring_period_id=DAY,
                team_id=OUR_TEAM,
                espn_id=espn_id,
                lineup_slot_id=slot,
                as_of=SYNCED,
            )
        )
        if ppg is not None:
            store.projections.upsert(espn_season_row(espn_id, ppg))
    for espn_id, name, position, team, ppg in FREE_AGENTS:
        rows.append(player_row(espn_id, name, position, team))
        store.projections.upsert(espn_season_row(espn_id, ppg))
    store.players.upsert_many(rows)
    store.rosters.replace(league.row_id, DAY, OUR_TEAM, entries)

    by_id = {row.espn_id: row for row in rows}
    persons = [
        NbaPerson(nba_id(espn_id), row.full_name, team=row.pro_team, position=row.position)
        for espn_id, row in by_id.items()
        if espn_id in WITH_HISTORY
    ]
    crosswalk = build_nba_crosswalk(rows, persons, as_of=SYNCED)
    crosswalk.save(store)
    ppg_of = {espn_id: ppg for espn_id, *_, ppg in VETERANS} | {FREE_VET: 10.0}
    darko = FakeDarko([darko_row(espn_id, by_id[espn_id].full_name, ppg_of[espn_id]) for espn_id in sorted(ppg_of)])
    return Synced(league, crosswalk, darko)


def blend(store: Store, synced: Synced) -> None:
    """The tick's step before the decisions: blend and store the day's lines from ESPN and DARKO (the stats baseline
    skips a no-history player the same way DARKO does)."""
    sources = ProjectionSourceRegistry()
    sources.register("nba", ESPN, stored=True)
    sources.register("nba", DARKO, loader=DarkoLoader(cast(DarkoSource, synced.darko)))
    day = blend_day(store, SEASON, DAY, weights=BlendWeights.load(), sources=sources)
    assert day.blend.sources_for(ROOKIE_ONE, SEASON, DAY) == (ESPN,)  # DARKO keys by nba.com id: none for him
    assert day.blend.sources_for(ROOKIE_THREE, SEASON, DAY) == (ESPN,)
    assert day.blend.sources_for(101, SEASON, DAY) == (DARKO, ESPN)
    assert day.blend.sources_for(ROOKIE_TWO, SEASON, DAY) == ()  # nothing projects him


# --- the tests --------------------------------------------------------------------------------------------------------


def test_rookies_pass_the_gate_without_an_nba_com_id(store: Store) -> None:
    synced = seed(store)
    walk = synced.crosswalk
    assert walk.mapped() == WITH_HISTORY
    assert all(walk.no_history(espn_id) for espn_id in (ROOKIE_ONE, ROOKIE_TWO, ROOKIE_THREE))
    verdict = nba_gate(store, synced.league.row_id, crosswalk=walk)
    assert verdict.passed and [entry.name for entry in verdict.no_history] == ["Rookie One", "Rookie Two"]


def test_the_blend_values_a_no_history_player_on_espns_projection_alone(store: Store) -> None:
    synced = seed(store)
    blend(store, synced)
    lines = per_game_lines(store, SEASON, DAY)
    assert lines[ROOKIE_ONE] == {"PTS": 9.0, "GP": 1.0}  # ESPN's per-game rate as it is
    assert lines[101]["PTS"] == pytest.approx(30.0 + DARKO_EXTRA / 2)  # a veteran averages ESPN and DARKO
    assert ROOKIE_TWO not in lines  # no source projects him: no line rather than a made-up one


def test_the_daily_lineup_starts_a_no_history_player_and_holds_one_nobody_projects(store: Store) -> None:
    synced = seed(store)
    blend(store, synced)
    decision = plan_daily_lineup(store, synced.league, schedule=schedule(), now=MORNING, period=DAY)
    per_game = decision.inputs.values.per_game
    scorer = Scorer(real_points())
    assert per_game[ROOKIE_ONE] == pytest.approx(scorer.points({"PTS": 9.0, "GP": 1.0}, position="PG"))
    assert ROOKIE_TWO not in per_game  # his value is unknown, not zero
    today = decision.best.day(DAY)
    # Day 3: the SG, the C and a UTIL are off, so the bench's guards play; the rookie nobody projects stays put.
    assert today.slots[ROOKIE_ONE] != BENCH and today.slots[ROOKIE_TWO] == BENCH
    assert all(move.espn_id != ROOKIE_TWO for move in today.moves)
    assert any("no projection for Rookie Two" in warning for warning in decision.warnings)


def test_streaming_values_a_no_history_free_agent_and_never_drops_an_unprojected_rookie(store: Store) -> None:
    synced = seed(store)
    blend(store, synced)
    wire = Wire(
        MappingProxyType({espn_id: WireEntry(espn_id, WireStatus.FREE_AGENT) for espn_id, *_ in FREE_AGENTS}), MORNING
    )
    decision = plan_streaming(store, synced.league, schedule=schedule(), now=MORNING, period=DAY, wire=wire)
    ranked = {score.espn_id: score for score in decision.ranked}
    assert ranked[ROOKIE_THREE].games > 0 and ranked[ROOKIE_THREE].value > 0  # valued from ESPN, like anyone
    assert ROOKIE_TWO in decision.protected  # unknown value: never a free drop
    assert decision.plan is None or all(move.drop != ROOKIE_TWO for move in decision.plan.moves)
