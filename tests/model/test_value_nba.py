"""NBA valuation (ROADMAP #24): per-game lines from ESPN and DARKO, the day's blend, points over the schedule and the
daily lineup value.

Real data, offline: ESPN's 2027 season projections of the 16 rostered players in
``tests/fixtures/espn/real/fba/mRoster.json`` (team 1's thirteen, plus Jokic, Jaylen Brown and Deni Avdija of team 8)
under the real league's points scoring (``.../mSettings.json``, ESPN's default H2H points); DARKO's talent sheet
(``tests/fixtures/sources/darko/talent.csv``, served by a mock transport to the real adapter), which shares Jokic and
Jaylen Brown with them; and ESPN's real pro schedule for days 1-3 and Christmas
(``tests/fixtures/sports/fba_pro_schedule_2027.json``). The 9-cat stand-in
(``tests/fixtures/espn/fba_settings_9cat.json``) is the category league.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterator
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fm.espn.ids import FBA
from fm.espn.models import ProSchedule, RostersView
from fm.espn.settings import LeagueSettings, load_league_settings, parse_league_settings
from fm.model import value_nba
from fm.model.categories import fit_categories
from fm.model.ids_nba import NbaCrosswalk
from fm.model.projections import (
    BLEND,
    DARKO,
    ESPN,
    BlendWeights,
    ProjectionSourceRegistry,
    blend_period,
    projection_source,
)
from fm.model.scoring import Scorer, ScoringError
from fm.model.value_nba import (
    DARKO_LABEL,
    DARKO_STATS,
    DarkoLoader,
    EspnDayLoader,
    blend_day,
    daily_lineup_value,
    darko_line,
    darko_rows,
    day_sources,
    espn_day_rows,
    marginal_lineup_value,
    per_game_line,
    per_game_lines,
    period_lines,
    scale_line,
    scheduled_points,
    team_games,
)
from fm.sources.base import RateLimiter
from fm.sources.darko import TALENT_URL, DarkoProjection, DarkoSource, parse_talent
from fm.sports.base import StatSchema
from fm.store import PlayerIdRow, PlayerRow, ProjectionRow, Store

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
REAL = FIXTURES / "espn" / "real" / "fba"
TALENT = FIXTURES / "sources" / "darko" / "talent.csv"
SCHEDULE = FIXTURES / "sports" / "fba_pro_schedule_2027.json"
NINE_CAT = FIXTURES / "espn" / "fba_settings_9cat.json"
SEASON = 2027
DAYS = (1, 2, 3, 67)
ESPN_AS_OF = datetime(2026, 10, 5, 12, 0, tzinfo=UTC)
DARKO_AS_OF = datetime(2026, 10, 5, 13, 0, tzinfo=UTC)

JOKIC, JAYLEN_BROWN = 3112335, 3917376
JOKIC_NBA, JAYLEN_BROWN_NBA = 203999, 1627759
CADE, SENGUN, BANCHERO, MOBLEY, DANIELS, MURPHY = 4432166, 4871144, 4432573, 4432158, 4869342, 4397688
OKONGWU, EDGECOMBE, QUICKLEY, WIGGINS, JAQUEZ, JABARI, BEY = (
    4431680,
    5124612,
    4395724,
    3059319,
    4432848,
    4432639,
    4397136,
)
DAY_2 = {SENGUN, BANCHERO, DANIELS, MURPHY, OKONGWU, QUICKLEY, WIGGINS, JAQUEZ, JABARI, BEY}
DET, PHI, MIA, LAC, NO, FREE_AGENT = 8, 20, 14, 12, 3, 0
EQUAL = "[nba.default]\nespn = 1\ndarko = 1\n"


# --- fixtures ---


@cache
def league() -> LeagueSettings:
    return load_league_settings(REAL / "mSettings.json")


@cache
def schedule() -> ProSchedule:
    return ProSchedule.model_validate(json.loads(SCHEDULE.read_text(encoding="utf-8")))


@cache
def rosters() -> RostersView:
    return RostersView.model_validate(json.loads((REAL / "mRoster.json").read_text(encoding="utf-8")))


@cache
def talent() -> list[DarkoProjection]:
    return parse_talent(TALENT.read_bytes())


def darko(nba_id: int) -> DarkoProjection:
    return next(projection for projection in talent() if projection.nba_id == nba_id)


def players() -> list[PlayerRow]:
    """The ``players`` rows the sync job writes for the 16 rostered players."""
    rows: list[PlayerRow] = []
    for team in rosters().teams:
        for entry in team.entries:
            player = entry.player
            assert player.default_position_id is not None
            rows.append(
                PlayerRow(
                    sport="nba",
                    espn_id=player.id,
                    full_name=player.full_name,
                    default_position_id=player.default_position_id,
                    position=FBA.position_label(player.default_position_id),
                    pro_team_id=player.pro_team_id,
                    eligible_slot_ids=list(player.eligible_slots),
                    as_of=ESPN_AS_OF,
                )
            )
    return rows


def team1() -> list[PlayerRow]:
    team = next(team for team in rosters().teams if team.team_id == 1)
    return [player for player in players() if player.espn_id in team.player_ids]


def espn_season_rows() -> list[ProjectionRow]:
    """ESPN's 2027 season projections as the sync job stores them: source ``espn``, period 0."""
    schema = StatSchema.for_game("fba")
    rows: list[ProjectionRow] = []
    for team in rosters().teams:
        for entry in team.entries:
            projection = entry.player.projection(SEASON, 0)
            assert projection is not None
            rows.append(
                ProjectionRow(
                    sport="nba",
                    espn_id=entry.player_id,
                    source=ESPN,
                    kind="projected",
                    season=SEASON,
                    scoring_period_id=0,
                    stats=schema.from_espn(projection.stats),
                    as_of=ESPN_AS_OF,
                )
            )
    return rows


def espn_per_game() -> dict[int, dict[str, float]]:
    lines: dict[int, dict[str, float]] = {}
    for row in espn_season_rows():
        line = per_game_line(row.stats)
        assert line is not None
        lines[row.espn_id] = line
    return lines


def crosswalk() -> NbaCrosswalk:
    return NbaCrosswalk(
        PlayerIdRow(sport="nba", espn_id=espn_id, source="nba", source_id=str(nba_id), origin="test", as_of=ESPN_AS_OF)
        for espn_id, nba_id in ((JOKIC, JOKIC_NBA), (JAYLEN_BROWN, JAYLEN_BROWN_NBA))
    )


def by_id(rows: tuple[ProjectionRow, ...] | list[ProjectionRow]) -> dict[int, ProjectionRow]:
    return {row.espn_id: row for row in rows}


def ids(rows: list[PlayerRow]) -> dict[int, PlayerRow]:
    return {player.espn_id: player for player in rows}


def slot_counts(settings: LeagueSettings) -> Counter[int]:
    """The league's active slots and how many of each, straight from its settings."""
    return Counter({slot.slot_id: slot.count for slot in settings.active_slots})


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        yield opened


class DarkoServer:
    """The real DARKO adapter over a mock transport serving the talent sheet (or failing with ``status``)."""

    def __init__(self, cache_root: Path, *, status: int = 200, body: bytes | None = None) -> None:
        self.status = status
        self.body = TALENT.read_bytes() if body is None else body
        self.requests: list[httpx.Request] = []
        self.source = DarkoSource(
            client=httpx.Client(transport=httpx.MockTransport(self.handle)),
            cache_root=cache_root,
            limiter=RateLimiter(0),
            clock=lambda: DARKO_AS_OF,
            sleep=lambda _: None,
        )

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert str(request.url) == TALENT_URL
        return httpx.Response(self.status, content=self.body, headers={"content-type": "text/csv"})


def nba_registry(loader: DarkoLoader) -> ProjectionSourceRegistry:
    """The NBA sources on a registry of the test's own: ESPN stored, DARKO through ``loader``."""
    registry = ProjectionSourceRegistry()
    registry.register("nba", ESPN, stored=True)
    registry.register("nba", DARKO, loader=loader)
    return registry


def seed(store: Store) -> None:
    store.projections.upsert_many(espn_season_rows())
    store.players.upsert_many(players())
    crosswalk().save(store)


# --- DARKO's lines ---


def test_darko_keys_map_onto_espn_abbreviations() -> None:
    for projection in talent():
        assert set(projection.per_game()) == set(DARKO_STATS)
    jokic = darko(JOKIC_NBA)
    per_game = jokic.per_game()
    line = darko_line(jokic)
    assert line == {**{DARKO_STATS[key]: value for key, value in per_game.items()}, "GP": 1.0}
    assert (line["TO"], line["3PM"], line["3PA"], line["MIN"]) == (
        per_game["tov"],
        per_game["fg3m"],
        per_game["fg3a"],
        jokic.minutes,
    )
    assert line["REB"] == pytest.approx(line["OREB"] + line["DREB"])
    assert set(DARKO_STATS.values()) <= set(StatSchema.for_game("fba").abbreviations)


def test_darko_rows_join_nba_ids_to_espn_ids_through_the_crosswalk() -> None:
    converted = darko_rows(talent(), crosswalk(), season=SEASON, scoring_period=3, as_of=DARKO_AS_OF)
    rows = by_id(converted.rows)
    assert set(rows) == {JOKIC, JAYLEN_BROWN}
    jokic = rows[JOKIC]
    assert (jokic.sport, jokic.source, jokic.kind, jokic.season, jokic.scoring_period_id, jokic.as_of) == (
        "nba",
        DARKO,
        "projected",
        SEASON,
        3,
        DARKO_AS_OF,
    )
    assert jokic.stats == darko_line(darko(JOKIC_NBA))
    playing = {str(projection.nba_id) for projection in talent() if projection.available and projection.minutes > 0}
    assert len(playing) == 12  # Cedric Coward is marked unavailable at zero minutes
    assert set(converted.unmapped) == playing - {str(JOKIC_NBA), str(JAYLEN_BROWN_NBA)}
    assert converted.warnings == ("darko: left out 1 players projected for no minutes or marked unavailable",)

    before_sync = darko_rows(talent(), NbaCrosswalk(()), season=SEASON, scoring_period=3, as_of=DARKO_AS_OF)
    assert before_sync.rows == ()
    assert before_sync.warnings[0] == (
        "darko: the NBA crosswalk maps none of the 12 players DARKO projects; run fm sync to save it"
    )


def test_darko_loader_reads_the_sheet_and_the_saved_crosswalk(store: Store, tmp_path: Path) -> None:
    crosswalk().save(store)
    server = DarkoServer(tmp_path / "cache")
    loader = DarkoLoader(server.source)
    fetched = loader(store, SEASON, 3, {})
    assert (fetched.source, fetched.dataset, fetched.as_of, fetched.degraded, fetched.cached) == (
        DARKO,
        "talent",
        DARKO_AS_OF,
        False,
        False,
    )
    assert {row.espn_id for row in fetched.data} == {JOKIC, JAYLEN_BROWN}
    again = loader(store, SEASON, 4, {})
    assert again.cached and len(server.requests) == 1  # the cached sheet serves another day
    assert {row.scoring_period_id for row in again.data} == {4}
    loader(store, SEASON, 4, {"force": True})
    assert len(server.requests) == 2


@pytest.mark.parametrize(
    ("status", "body"),
    [(404, b"not found"), (200, b"<!DOCTYPE html><html><title>Google Sheets: Sign-in</title></html>")],
)
def test_a_darko_outage_degrades_instead_of_raising(store: Store, tmp_path: Path, status: int, body: bytes) -> None:
    server = DarkoServer(tmp_path / "cache", status=status, body=body)
    fetched = DarkoLoader(server.source, crosswalk=crosswalk())(store, SEASON, 1, {})
    assert fetched.data == () and fetched.degraded and fetched.source == DARKO
    assert len(fetched.warnings) == 1 and fetched.warnings[0].startswith("darko: unavailable, blending without it")


def test_darko_is_registered_with_a_loader_and_espn_stays_stored() -> None:
    source = projection_source("nba", DARKO)
    assert isinstance(source.loader, DarkoLoader) and not source.stored and source.label == DARKO_LABEL
    assert projection_source("nba", ESPN).stored  # the sync job still writes ESPN's lines
    value_nba._attach_darko_loader()  # what a second import would do: nothing
    assert projection_source("nba", DARKO) is source


# --- ESPN's per-game rate ---


def test_espns_season_projection_per_game_is_its_applied_average() -> None:
    # ESPN publishes no NBA daily projections: a day's is the season line per game, and under the real league's
    # scoring each rostered player's comes to ESPN's own appliedAverage (appliedTotal / GP)
    schema = StatSchema.for_game("fba")
    scorer = Scorer(league())
    checked = 0
    for team in rosters().teams:
        for entry in team.entries:
            projection = entry.player.projection(SEASON, 0)
            assert projection is not None and projection.applied_average is not None
            line = per_game_line(schema.from_espn(projection.stats))
            assert line is not None and line["GP"] == 1.0
            position = FBA.position_label(entry.player.default_position_id or 0)
            assert scorer.points(line, position=position) == pytest.approx(projection.applied_average, abs=1e-9)
            checked += 1
    assert checked == 16


def test_a_per_game_line_divides_the_totals_and_keeps_the_rates() -> None:
    season = by_id(espn_season_rows())[JOKIC].stats
    games = season["GP"]
    assert games == 72.0
    line = per_game_line(season)
    assert line is not None
    assert line["PTS"] == pytest.approx(season["PTS"] / games) and line["MIN"] == pytest.approx(season["MPG"], rel=1e-2)
    assert (line["FG%"], line["FT%"], line["APG"], line["A/TO"], line["GP"]) == (
        season["FG%"],
        season["FT%"],
        season["APG"],
        season["A/TO"],
        1.0,
    )
    assert scale_line(line, games) == pytest.approx(season)  # the round trip
    assert scale_line(line, 0)["PTS"] == 0.0 and scale_line(line, 0)["GP"] == 0.0
    assert per_game_line({}) is None and per_game_line({"GP": 0.0, "PTS": 0.0}) is None
    assert per_game_line({"PTS": 10.0}) is None
    with pytest.raises(ScoringError, match="'PTS'"):
        per_game_line({"GP": 10.0, "PTS": float("inf")})
    with pytest.raises(ValueError, match="games"):
        scale_line(line, -1.0)


def test_espn_day_rows_and_their_loader(store: Store) -> None:
    rows = espn_season_rows()
    jokic = by_id(rows)[JOKIC]
    out_for_the_season = jokic.model_copy(update={"espn_id": 1, "stats": {"GP": 0.0}})
    actual = jokic.model_copy(update={"kind": "actual"})
    converted = espn_day_rows([*rows, out_for_the_season, actual], scoring_period=5)
    assert len(converted.rows) == 16
    assert by_id(converted.rows)[JOKIC].stats == per_game_line(jokic.stats)
    assert all((row.source, row.scoring_period_id, row.as_of) == (ESPN, 5, ESPN_AS_OF) for row in converted.rows)
    assert converted.warnings == ("espn: 1 season projections have no games (GP 0 or missing); no per-game line",)

    nothing = EspnDayLoader()(store, SEASON, 5, {})
    assert nothing.data == () and nothing.degraded
    assert nothing.warnings == ("espn: no nba projections stored for 2027 period 0; run fm sync",)
    store.projections.upsert_many(rows)
    loaded = EspnDayLoader()(store, SEASON, 5, {})
    assert (loaded.source, loaded.as_of, loaded.cached, loaded.degraded, loaded.warnings) == (
        ESPN,
        ESPN_AS_OF,
        True,
        False,
        (),
    )
    assert by_id(loaded.data) == by_id(converted.rows)


def test_day_sources_read_espn_per_game_and_leave_the_registry_alone() -> None:
    loader = DarkoLoader()
    registry = nba_registry(loader)
    registry.register("nba", "minutes", stored=True)  # an in-house baseline (ROADMAP #43)
    registry.register("nfl", ESPN, stored=True)
    day = day_sources(registry)
    assert [source.name for source in day.registered("nba")] == [ESPN, DARKO, "minutes"]
    assert day.registered("nfl") == ()
    espn = day.lookup("nba", ESPN)
    assert isinstance(espn.loader, EspnDayLoader) and not espn.stored
    assert day.lookup("nba", DARKO).loader is loader and day.lookup("nba", "minutes").stored
    assert registry.lookup("nba", ESPN).stored and registry.lookup("nba", ESPN).loader is None


# --- the day's blend ---


def test_blend_day_blends_espns_rate_with_darko(store: Store, tmp_path: Path) -> None:
    seed(store)
    registry = nba_registry(DarkoLoader(DarkoServer(tmp_path / "cache").source))
    weights = BlendWeights.parse(EQUAL, sources=registry)
    result = blend_day(store, SEASON, 1, weights=weights, sources=registry)
    assert result.warnings == ("darko: left out 1 players projected for no minutes or marked unavailable",)
    assert len(result.rows) == 16 and result.saved == 16 + 2 + 16  # ESPN's per-game rows, DARKO's, the blend
    assert result.blend.sources_for(JOKIC, SEASON, 1) == (DARKO, ESPN)
    assert result.blend.sources_for(CADE, SEASON, 1) == (ESPN,)

    blended = per_game_lines(store, SEASON, 1)
    espn, from_darko = per_game_lines(store, SEASON, 1, source=ESPN), per_game_lines(store, SEASON, 1, source=DARKO)
    assert set(blended) == set(espn) == {player.espn_id for player in players()}
    assert set(from_darko) == {JOKIC, JAYLEN_BROWN}
    scorer = Scorer(league())
    for player in (JOKIC, JAYLEN_BROWN):
        line = blended[player]
        # both sources carry every stat the league scores, so the blend scores as the mean of their points
        assert scorer.points(line) == pytest.approx(
            (scorer.points(espn[player]) + scorer.points(from_darko[player])) / 2
        )
        assert line["FG%"] == pytest.approx(line["FGM"] / line["FGA"])  # recomputed, not averaged
        rebounds = (espn[player]["REB"] + from_darko[player]["OREB"] + from_darko[player]["DREB"]) / 2
        assert line["REB"] == pytest.approx(rebounds) and line["GP"] == 1.0
    assert scorer.points(espn[JOKIC]) == pytest.approx(67.194444, abs=1e-6)
    assert scorer.points(from_darko[JOKIC]) == pytest.approx(43.014961, abs=1e-6)
    assert scorer.points(blended[JOKIC]) == pytest.approx(55.104703, abs=1e-6)
    assert blended[CADE] == espn[CADE]  # ESPN alone: its per-game rate as it is

    # saved, ESPN's per-game rows serve the plain stored source as well
    replay = blend_period(store, "nba", SEASON, 1, weights=weights, sources=registry, save=False)
    assert by_id(replay.rows)[JOKIC].stats == pytest.approx(blended[JOKIC])
    with pytest.raises(ValueError, match="scoring period >= 1"):
        blend_day(store, SEASON, 0, weights=weights, sources=registry)


def test_blend_day_runs_on_espn_alone_when_darko_is_down(store: Store, tmp_path: Path) -> None:
    seed(store)
    registry = nba_registry(DarkoLoader(DarkoServer(tmp_path / "cache", status=404).source))
    result = blend_day(store, SEASON, 2, weights=BlendWeights.parse(EQUAL, sources=registry), sources=registry)
    assert len(result.rows) == 16 and all(sources == (ESPN,) for sources in result.blend.inputs.values())
    load = result.load(DARKO)
    assert load is not None and load.degraded
    assert any(warning.startswith("darko: unavailable, blending without it") for warning in result.warnings)
    assert by_id(store.projections.for_period("nba", SEASON, 2, source=BLEND))[JOKIC].stats == espn_per_game()[JOKIC]


# --- points over the schedule ---


def test_team_games_count_the_schedule() -> None:
    assert team_games(schedule(), PHI, DAYS) == 3  # days 1, 3 and Christmas
    assert team_games(schedule(), MIA, DAYS) == 2
    assert team_games(schedule(), DET, (2, 3)) == 0
    assert team_games(schedule(), FREE_AGENT, DAYS) == 0 and team_games(schedule(), None, DAYS) == 0


def test_scheduled_points_are_points_per_game_times_games() -> None:
    lines = espn_per_game()
    values = scheduled_points(lines, team1(), league(), schedule(), DAYS)
    assert set(values) == {player.espn_id for player in team1()}
    edgecombe = values[EDGECOMBE]
    assert edgecombe.games == 3 and edgecombe.per_game == pytest.approx(32.666667, abs=1e-6)
    assert edgecombe.total == pytest.approx(3 * edgecombe.per_game)
    assert (values[WIGGINS].games, values[MOBLEY].games, values[CADE].games, values[BEY].games) == (2, 1, 1, 1)

    doubtful = scheduled_points(lines, team1(), league(), schedule(), DAYS, p_active={EDGECOMBE: 0.5})
    assert doubtful[EDGECOMBE].total == pytest.approx(edgecombe.total / 2)
    assert doubtful[EDGECOMBE].expected_per_game == pytest.approx(edgecombe.per_game / 2)
    assert doubtful[CADE].total == values[CADE].total
    unsigned = team1()[0].model_copy(update={"pro_team_id": FREE_AGENT})
    assert scheduled_points(lines, [unsigned], league(), schedule(), DAYS)[unsigned.espn_id].total == 0.0
    with pytest.raises(ValueError, match="should be a probability"):
        scheduled_points(lines, team1(), league(), schedule(), DAYS, p_active={CADE: 1.5})
    with pytest.raises(ValueError, match="competes on categories"):
        scheduled_points(lines, team1(), load_league_settings(NINE_CAT), schedule(), DAYS)


def test_period_lines_scale_each_line_by_its_teams_games() -> None:
    lines = espn_per_game()
    spans = period_lines(lines, team1(), schedule(), DAYS)
    assert set(spans) == {player.espn_id for player in team1()}
    assert spans[EDGECOMBE]["GP"] == 3.0 and spans[EDGECOMBE]["PTS"] == pytest.approx(3 * lines[EDGECOMBE]["PTS"])
    assert spans[EDGECOMBE]["FG%"] == lines[EDGECOMBE]["FG%"] and spans[CADE]["GP"] == 1.0
    # fit on the span, three games of Edgecombe's outscore one of Cunningham's in points
    model = fit_categories(spans, load_league_settings(NINE_CAT))
    assert lines[CADE]["PTS"] > lines[EDGECOMBE]["PTS"]
    assert model.scores(spans[EDGECOMBE])["PTS"] > model.scores(spans[CADE])["PTS"]


# --- the daily lineup ---


def test_a_player_without_espns_slots_plays_where_his_position_may() -> None:
    template = team1()[0]

    def player(espn_id: int, position: str | None) -> PlayerRow:
        return template.model_copy(
            update={"espn_id": espn_id, "pro_team_id": NO, "position": position, "eligible_slot_ids": []}
        )

    roster = [player(1, "C"), player(2, "PG"), player(3, "POS_9"), player(4, None)]
    centers = [player(10 + n, "C") for n in range(4)]
    (day,) = daily_lineup_value(roster, {1: 30.0, 2: 20.0, 3: 50.0, 4: 40.0}, league(), schedule(), (2,)).days
    assert day.slots[1] in {4, 11} and day.slots[2] in {0, 5, 11}  # the NBA plugin's C and PG rows
    assert set(day.benched) == {3, 4}  # no slot for an unknown position, or for none
    assert day.open_slots == 8
    values = {1: 30.0} | {10 + n: 31.0 + n for n in range(4)}
    (crowded,) = daily_lineup_value([roster[0], *centers], values, league(), schedule(), (2,)).days
    assert set(crowded.slots.values()) == {4, 11} and crowded.benched == (1,)  # C and three UTIL for five centers


def per_game_points() -> dict[int, float]:
    values = scheduled_points(espn_per_game(), team1(), league(), schedule(), DAYS)
    return {espn_id: value.expected_per_game for espn_id, value in values.items()}


def test_the_daily_lineup_on_the_real_schedule() -> None:
    per_game = per_game_points()
    lineup = daily_lineup_value(team1(), per_game, league(), schedule(), DAYS)
    days = {day.scoring_period: day for day in lineup.days}
    assert list(days) == list(DAYS)
    assert set(days[1].slots) == {CADE, EDGECOMBE} and days[1].open_slots == 8
    assert set(days[2].slots) == DAY_2 and days[2].open_slots == 0 and days[2].benched == ()
    assert set(days[3].slots) == {MOBLEY, EDGECOMBE} and set(days[67].slots) == {WIGGINS, EDGECOMBE}
    roster = ids(team1())
    counts = slot_counts(league())
    assert sum(counts.values()) == 10 and counts[FBA.slot_id("UTIL")] == 3
    for day in lineup.days:  # legal: each starter in a slot ESPN lets him fill, no slot used more than the league has
        assert all(slot in roster[espn_id].eligible_slot_ids for espn_id, slot in day.slots.items())
        assert Counter(day.slots.values()) <= counts
        assert day.value == pytest.approx(sum(per_game[espn_id] for espn_id in day.slots))
    assert lineup.open_slots == 24
    assert lineup.total == pytest.approx(sum(day.value for day in lineup.days))
    assert lineup.starts(EDGECOMBE) == 3 and lineup.value_of(EDGECOMBE) == pytest.approx(3 * per_game[EDGECOMBE])
    assert lineup.starts(BEY) == 1 and lineup.starts(JOKIC) == 0


def test_a_crowded_day_benches_by_eligibility_not_by_value_alone() -> None:
    template = team1()[0]
    centers = [
        template.model_copy(update={"espn_id": 100 + n, "pro_team_id": NO, "eligible_slot_ids": [4, 11, 12, 13]})
        for n in range(5)
    ]
    guards = [
        template.model_copy(update={"espn_id": 200 + n, "pro_team_id": NO, "eligible_slot_ids": [0, 1, 5, 11, 12, 13]})
        for n in range(7)
    ]
    per_game = {100 + n: 50.0 - n for n in range(5)} | {200 + n: 10.0 + n for n in range(7)}
    (day,) = daily_lineup_value([*centers, *guards], per_game, league(), schedule(), (2,)).days
    # centers fill only C and the three UTIL slots, so the fifth (46) sits behind guards worth 14-16
    assert set(day.slots) == {100, 101, 102, 103, 204, 205, 206}
    assert set(day.benched) == {104, 200, 201, 202, 203}
    assert day.open_slots == 3  # no one here plays SF, PF or F
    assert day.value == pytest.approx(50 + 49 + 48 + 47 + 14 + 15 + 16)


@cache
def small_league() -> LeagueSettings:
    """The real league cut to PG, C and two UTIL slots."""
    view: dict[str, Any] = json.loads((REAL / "mSettings.json").read_text(encoding="utf-8"))
    counts = view["settings"]["rosterSettings"]["lineupSlotCounts"]
    for slot in counts:
        counts[slot] = 0
    counts.update({"0": 1, "4": 1, "11": 2, "12": 3})
    return parse_league_settings(view)


def brute_force(players: list[tuple[frozenset[int], float]], slots: tuple[int, ...]) -> tuple[float, int]:
    """By trying every legal lineup: the highest value any has (only a game worth something starts), and the most
    starters among the lineups worth that much."""
    best = (0.0, 0)

    def search(index: int, free: tuple[int, ...], value: float, starters: int) -> None:
        nonlocal best
        if index == len(players):
            best = max(best, (value, starters))
            return
        eligible, worth = players[index]
        search(index + 1, free, value, starters)
        if worth > 0:
            for position, slot in enumerate(free):
                if slots[slot] in eligible:
                    search(index + 1, free[:position] + free[position + 1 :], value + worth, starters + 1)

    search(0, tuple(range(len(slots))), 0.0, 0)
    return best


@settings(deadline=None)
@given(
    st.lists(
        st.tuples(st.frozensets(st.sampled_from((0, 4, 11))), st.integers(min_value=-20, max_value=60).map(float)),
        max_size=6,
    )
)
def test_the_lineup_is_the_most_valuable_legal_one(drawn: list[tuple[frozenset[int], float]]) -> None:
    template = team1()[0]
    roster = [
        template.model_copy(update={"espn_id": n, "pro_team_id": NO, "eligible_slot_ids": sorted(eligible | {12, 13})})
        for n, (eligible, _) in enumerate(drawn)
    ]
    per_game = {n: value for n, (_, value) in enumerate(drawn)}
    (day,) = daily_lineup_value(roster, per_game, small_league(), schedule(), (2,)).days
    slots = tuple(slot for slot, count in sorted(slot_counts(small_league()).items()) for _ in range(count))
    assert slots == (0, 4, 11, 11)
    value, starters = brute_force(list(drawn), slots)
    assert day.value == value  # whole numbers: the sums are exact
    # no equally valuable lineup starts more (with positive values the best one fills every slot it can)
    assert len(day.slots) == starters and day.open_slots == len(slots) - starters
    assert all(per_game[espn_id] > 0 for espn_id in day.slots)
    assert set(day.slots) | set(day.benched) == set(per_game) and not set(day.slots) & set(day.benched)
    for espn_id, slot in day.slots.items():
        assert slot in drawn[espn_id][0]


def test_a_player_ruled_out_leaves_his_slot_open() -> None:
    per_game = {**per_game_points(), CADE: 0.0, EDGECOMBE: -1.0}  # Cunningham out; a game of Edgecombe's hurts
    lineup = daily_lineup_value(team1(), per_game, league(), schedule(), DAYS)
    day_1 = lineup.days[0]
    assert day_1.slots == {} and set(day_1.benched) == {CADE, EDGECOMBE} and day_1.open_slots == 10
    assert lineup.starts(EDGECOMBE) == 0 and lineup.open_slots == 24 + 2 + 1 + 1
    worth = {**per_game, 1: 12.5}
    gain = marginal_lineup_value(team1(), worth, league(), schedule(), DAYS, add=streamer(1, DET))
    assert gain == pytest.approx(12.5)  # Detroit's day-1 game fills Cunningham's slot
    with pytest.raises(ValueError, match="no per-game value"):
        marginal_lineup_value(team1(), per_game, league(), schedule(), DAYS, add=streamer(1, DET))


def streamer(espn_id: int, pro_team_id: int) -> PlayerRow:
    """A free agent who may fill any active slot."""
    return team1()[0].model_copy(
        update={"espn_id": espn_id, "pro_team_id": pro_team_id, "eligible_slot_ids": [*range(12), 12, 13]}
    )


def test_a_streamer_is_worth_his_games_on_open_days_only() -> None:
    per_game = per_game_points()
    values = {**per_game, 1: 20.0, 2: 20.0, 3: 100.0}

    def gain(add: PlayerRow, drop: int | None = None) -> float:
        return marginal_lineup_value(team1(), values, league(), schedule(), DAYS, add=add, drop=drop)

    assert gain(streamer(1, DET)) == pytest.approx(20.0)  # Detroit plays on day 1, when eight slots are open
    assert gain(streamer(2, LAC)) == pytest.approx(0.0, abs=1e-9)  # the Clippers only on day 2, which is full
    weakest = min(per_game[espn_id] for espn_id in DAY_2)
    assert weakest == per_game[BEY]
    assert gain(streamer(3, LAC)) == pytest.approx(100.0 - weakest)
    assert gain(streamer(3, LAC), drop=BEY) == pytest.approx(100.0 - per_game[BEY])
    assert gain(streamer(3, LAC), drop=EDGECOMBE) == pytest.approx(100.0 - weakest - 3 * per_game[EDGECOMBE])
    with pytest.raises(ValueError, match="already on the roster"):
        gain(ids(team1())[CADE])
    with pytest.raises(ValueError, match="not on the roster"):
        gain(streamer(1, DET), drop=JOKIC)


def test_a_category_league_values_a_game_by_its_contribution() -> None:
    nine_cat = load_league_settings(NINE_CAT)
    lines = espn_per_game()
    model = fit_categories(lines, nine_cat)
    per_game = {espn_id: model.contribution(lines[espn_id]) for espn_id in lines}
    lineup = daily_lineup_value(team1(), per_game, nine_cat, schedule(), DAYS)
    assert lineup.open_slots == 24  # the stand-in has the real league's lineup slots
    assert lineup.total == pytest.approx(sum(per_game[espn_id] for day in lineup.days for espn_id in day.slots))
    jokic_on_det = ids(players())[JOKIC].model_copy(update={"pro_team_id": DET})
    gain = marginal_lineup_value(team1(), per_game, nine_cat, schedule(), DAYS, add=jokic_on_det)
    assert gain == pytest.approx(per_game[JOKIC]) and gain > 0  # one game on an open day
