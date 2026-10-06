"""The rest-of-season rankings sheet (ROADMAP #35): the sheet's numbers for an NBA points league and an NBA category
league on hand-built stores, the CSV, and ``fm rankings`` over the recorded fixtures (offline, NFL and NBA) and over the
fixture home (the synced NFL league).

The category and NBA points leagues are built from ``tests/fixtures/espn/fba_settings_*.json`` with a handful of players
whose lines are easy to reason about; every expected number below follows from those lines. Nothing here reaches the
network: the NBA blend is ESPN's stored line alone (DARKO is fetched over the network), and the command tests run the
real sync job over recorded views or read a private copy of the fixture home.
"""

from __future__ import annotations

import csv
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from fm import paths
from fm.commands import rankings as rankings_command
from fm.decide.rankings import (
    OWNER_FREE_AGENT,
    OWNER_WAIVERS,
    RankingMetric,
    Rankings,
    RankingsError,
    rank_league,
    write_csv,
)
from fm.espn.models import ProSchedule
from fm.espn.settings import LeagueSettings, load_league_settings
from fm.model.projections import ESPN, BlendWeights, ProjectionSourceRegistry
from fm.model.scoring import Scorer
from fm.model.value_nba import per_game_line
from fm.store import (
    LeagueRow,
    LeagueSettingsRow,
    PlayerRow,
    ProjectionRow,
    RosterEntryRow,
    Store,
    TeamRow,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ESPN_FIXTURES = FIXTURES / "espn"
HOME = FIXTURES / "home"
NOW = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)
NBA_NOW = datetime(2026, 10, 20, 15, 0, tzinfo=UTC)
DAYS = 3
GAMES = 70
"""Games in the season totals the stored ESPN lines carry (``GP``)."""
BENCH = 13
UTIL_SLOTS = [0, 5, 11, 12, 13]

runner = CliRunner()


# --- an NBA league of eight players -----------------------------------------------------------------------------------

# name: (pro team, ESPN eligible slots, per-game stats); FGM/FGA and FTM/FTA are what the percentages are made of.
PER_GAME: dict[int, tuple[str, int, list[int], dict[str, float]]] = {
    1: (
        "Big",
        1,
        [4, 6, 11, 12, 13],
        dict(PTS=18, REB=11, AST=2, STL=0.8, BLK=2.0, TO=2.0, FGM=7.5, FGA=12, FTM=3, FTA=5),
    ),
    2: (
        "Scorer",
        2,
        [1, 5, 11, 12, 13],
        dict(PTS=25, REB=4, AST=6, STL=1.2, BLK=0.3, TO=3.5, FGM=8.5, FGA=19, FTM=6, FTA=7),
    ),
    3: (
        "Wing",
        3,
        [2, 6, 11, 12, 13],
        dict(PTS=14, REB=6, AST=3, STL=1.1, BLK=0.5, TO=2.0, FGM=5.5, FGA=12.5, FTM=2, FTA=2.4),
    ),
    4: (
        "Passer",
        4,
        [0, 5, 11, 12, 13],
        dict(PTS=12, REB=3, AST=9, STL=1.5, BLK=0.2, TO=2.0, FGM=4.5, FGA=10, FTM=2, FTA=2.5),
    ),
    5: (
        "Reserve",
        5,
        [0, 5, 11, 12, 13],
        dict(PTS=6, REB=2, AST=1, STL=0.5, BLK=0.1, TO=0.8, FGM=2.2, FGA=5.5, FTM=0.8, FTA=1),
    ),
    6: (
        "FreeGood",
        6,
        [2, 3, 6, 11, 12, 13],
        dict(PTS=16, REB=7, AST=3, STL=1.0, BLK=1.0, TO=1.5, FGM=6, FGA=13, FTM=3, FTA=4),
    ),
    7: (
        "FreeBad",
        6,
        [2, 6, 11, 12, 13],
        dict(PTS=5, REB=2, AST=1, STL=0.4, BLK=0.1, TO=0.8, FGM=2, FGA=5, FTM=0.5, FTA=1),
    ),
    8: ("Unprojected", 1, [4, 11, 12, 13], {}),
}
ROSTERED = {1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 8: 1}
"""ESPN id -> fantasy team: ours (1) and one rival (2); players 6 and 7 are free agents."""
SHORT_SCHEDULE = 5
"""The one team (``Reserve``'s) that plays on day 1 only."""


def schedule(days: int = DAYS) -> ProSchedule:
    """Every team plays every day, except team 5 after day 1."""
    teams = []
    for team in range(1, 7):
        games = {
            str(day): [
                {
                    "id": team * 1000 + day,
                    "date": int(datetime(2026, 10, 20 + day, 23, 0, tzinfo=UTC).timestamp() * 1000),
                    "scoringPeriodId": day,
                    "homeProTeamId": team,
                    "awayProTeamId": 99,
                }
            ]
            for day in range(1, days + 1)
            if team != SHORT_SCHEDULE or day == 1
        }
        teams.append({"id": team, "proGamesByScoringPeriod": games})
    return ProSchedule.model_validate({"proTeams": teams})


def espn_only() -> ProjectionSourceRegistry:
    registry = ProjectionSourceRegistry()
    registry.register("nba", ESPN, label="stored ESPN line", stored=True)
    return registry


def nba_settings(name: str) -> LeagueSettings:
    """A fixture NBA league's settings, ending its season on day ``DAYS``."""
    return load_league_settings(ESPN_FIXTURES / name).model_copy(update={"final_scoring_period": DAYS})


@pytest.fixture
def store() -> Iterator[Store]:
    with Store.open(":memory:") as opened:
        yield opened


def nba_league(store: Store, settings: LeagueSettings, *, lines: bool = True) -> LeagueRow:
    league = store.leagues.upsert(
        LeagueRow(key="nba", sport="nba", espn_league_id=settings.league_id, season=2027, team_id=1, as_of=NOW)
    )
    store.settings.upsert(
        LeagueSettingsRow(
            league_id=league.row_id, settings=settings.model_dump(mode="json"), raw_snapshot_id=None, as_of=NOW
        )
    )
    store.teams.upsert_many(TeamRow(league_id=league.row_id, team_id=n, name=f"Team {n}", as_of=NOW) for n in (1, 2))
    store.players.upsert_many(
        PlayerRow(
            sport="nba",
            espn_id=espn_id,
            full_name=name,
            position="PG",
            pro_team_id=team,
            pro_team=f"T{team}",
            eligible_slot_ids=slots,
            injury_status="ACTIVE",
            as_of=NOW,
        )
        for espn_id, (name, team, slots, _) in PER_GAME.items()
    )
    for team in (1, 2):
        store.rosters.replace(
            league.row_id,
            1,
            team,
            [
                RosterEntryRow(
                    league_id=league.row_id,
                    scoring_period_id=1,
                    team_id=team,
                    espn_id=espn_id,
                    lineup_slot_id=BENCH,
                    as_of=NOW,
                )
                for espn_id, owner in ROSTERED.items()
                if owner == team
            ],
        )
    if lines:
        store.projections.upsert_many(
            ProjectionRow(
                sport="nba",
                espn_id=espn_id,
                source=ESPN,
                kind="projected",
                season=2027,
                scoring_period_id=0,
                stats={stat: value * GAMES for stat, value in per_game.items()} | {"GP": float(GAMES)},
                as_of=NOW,
            )
            for espn_id, (_, _, _, per_game) in PER_GAME.items()
            if per_game
        )
    return league


def rank_nba(store: Store, league: LeagueRow, **options: object) -> Rankings:
    arguments: dict[str, object] = {"schedule": schedule(), "weights": BlendWeights.load(), "sources": espn_only()}
    arguments.update(options)
    return rank_league(store, league, now=NBA_NOW, **arguments)  # type: ignore[arg-type]


# --- a category league ------------------------------------------------------------------------------------------------


def test_category_league_ranks_by_the_sum_of_its_own_categories(store: Store) -> None:
    league = nba_league(store, nba_settings("fba_settings_9cat.json"))
    sheet = rank_nba(store, league)

    assert sheet.metric is RankingMetric.G
    assert sheet.categories == (
        "PTS",
        "BLK",
        "STL",
        "AST",
        "REB",
        "TO",
        "3PM",
        "FG%",
        "FT%",
    )  # the league's own, in order
    ranked = sheet.ranked
    assert [player.rank for player in ranked] == list(range(1, 8))
    assert [player.value for player in ranked] == sorted((player.value for player in ranked), reverse=True)  # type: ignore[type-var]
    for player in ranked:
        assert player.value == pytest.approx(sum(player.scores.values()))
        assert list(player.scores) == list(sheet.categories)

    players = {player.name: player for player in sheet.players}
    assert ranked[0].name in {"Scorer", "Big", "FreeGood"}
    # Turnovers count against: the scorer's 3.5 a game is the worst in the pool, the reserve's 0.8 the best.
    assert players["Scorer"].scores["TO"] < 0 < players["Reserve"].scores["TO"]
    # The big man's blocks and boards beat the passer's; the passer's assists beat the big man's.
    assert players["Big"].scores["BLK"] > players["Passer"].scores["BLK"]
    assert players["Passer"].scores["AST"] > players["Big"].scores["AST"]


def test_a_players_games_are_his_teams_remaining_scheduled_games(store: Store) -> None:
    league = nba_league(store, nba_settings("fba_settings_9cat.json"))
    players = {player.name: player for player in rank_nba(store, league).players}
    assert players["Big"].games == DAYS
    assert players["Reserve"].games == 1  # his team's schedule ends after day 1


def test_owners_mine_and_free_agent_status(store: Store) -> None:
    league = nba_league(store, nba_settings("fba_settings_9cat.json"))
    sheet = rank_nba(store, league, waivers={7})
    players = {player.name: player for player in sheet.players}
    assert (players["Big"].owner, players["Big"].mine) == ("Team 1", True)
    assert (players["Passer"].owner, players["Passer"].mine) == ("Team 2", False)
    assert players["FreeGood"].owner == OWNER_FREE_AGENT and players["FreeGood"].team_id is None
    assert players["FreeBad"].owner == OWNER_WAIVERS
    assert players["Big"].positions == ("PG", "C")  # his default position, then the other positions ESPN lists


def test_value_over_replacement_is_measured_from_the_best_free_agent(store: Store) -> None:
    league = nba_league(store, nba_settings("fba_settings_9cat.json"))
    sheet = rank_nba(store, league)
    players = {player.name: player for player in sheet.players}
    best_free = players["FreeGood"].value
    assert best_free is not None and players["FreeBad"].value is not None
    assert players["FreeGood"].vor == pytest.approx(0.0)
    assert players["FreeBad"].vor == pytest.approx(players["FreeBad"].value - best_free)
    assert players["Scorer"].vor == pytest.approx((players["Scorer"].value or 0.0) - best_free)
    assert any("best free agent: FreeGood" in line for line in sheet.replacement)


def test_a_player_nothing_projects_is_unranked_not_zero(store: Store) -> None:
    league = nba_league(store, nba_settings("fba_settings_9cat.json"))
    sheet = rank_nba(store, league)
    ghost = {player.name: player for player in sheet.players}["Unprojected"]
    assert ghost.rank is None and ghost.value is None and ghost.vor is None
    assert ghost.note == "no projection"
    assert sheet.players[-1] is ghost  # unvalued players come last


def test_g_scores_equal_z_scores_until_tau_is_known(store: Store) -> None:
    league = nba_league(store, nba_settings("fba_settings_9cat.json"))
    plain = rank_nba(store, league)
    assert any("G-scores equal z-scores" in warning for warning in plain.warnings)
    noisy = rank_nba(store, league, tau={"STL": 1.0, "BLK": 1.0})
    assert not any("G-scores equal z-scores" in warning for warning in noisy.warnings)
    shrunk = {player.name: player.scores for player in noisy.players}["Big"]["BLK"]
    assert abs(shrunk) < abs({player.name: player.scores for player in plain.players}["Big"]["BLK"])


# --- an NBA points league ---------------------------------------------------------------------------------------------


def test_nba_points_are_per_game_points_times_scheduled_games(store: Store) -> None:
    settings = nba_settings("fba_settings_points.json")
    league = nba_league(store, settings)
    sheet = rank_nba(store, league)
    assert sheet.metric is RankingMetric.POINTS and sheet.categories == ()

    scorer = Scorer(settings)
    players = {player.name: player for player in sheet.players}
    for name, _, _, stats in PER_GAME.values():
        if not stats:
            continue
        line = per_game_line({stat: value * GAMES for stat, value in stats.items()} | {"GP": float(GAMES)})
        assert line is not None
        per_game = scorer.points(line, position="PG")
        games = 1 if name == "Reserve" else DAYS
        assert players[name].per_game == pytest.approx(per_game)
        assert players[name].value == pytest.approx(per_game * games), name
    ranked = [player.name for player in sheet.ranked]
    assert ranked[0] in {"Big", "Scorer"} and ranked[-1] in {"Reserve", "FreeBad"}


def test_nba_points_replacement_is_the_best_wire_player_at_a_slot(store: Store) -> None:
    league = nba_league(store, nba_settings("fba_settings_points.json"))
    sheet = rank_nba(store, league)
    players = {player.name: player for player in sheet.players}
    good, bad = players["FreeGood"], players["FreeBad"]
    assert good.value is not None and bad.value is not None
    # FreeGood is the best wire player at every slot he fills, and the lowest of those levels is the slot the big man
    # also fills, so a player's vor is his value less the best wire player's value at his scarcest slot.
    assert good.vor == pytest.approx(0.0)
    assert bad.vor == pytest.approx(bad.value - good.value)
    assert any("FreeGood" in line for line in sheet.replacement)


def test_an_nba_league_cannot_be_valued_without_the_pro_schedule(store: Store) -> None:
    league = nba_league(store, nba_settings("fba_settings_points.json"))
    with pytest.raises(RankingsError, match="pro schedule"):
        rank_nba(store, league, schedule=None)


def test_a_schedule_short_of_the_season_is_warned_about(store: Store) -> None:
    league = nba_league(store, nba_settings("fba_settings_points.json"))
    sheet = rank_nba(store, league, schedule=schedule(days=2))
    assert any("lists games for 2 of the 3 days left" in warning for warning in sheet.warnings)


def test_a_league_without_projections_is_refused_with_the_fix(store: Store) -> None:
    league = nba_league(store, nba_settings("fba_settings_points.json"), lines=False)
    with pytest.raises(RankingsError, match="run fm sync"):
        rank_nba(store, league)


def test_a_league_that_was_never_synced_is_refused(store: Store) -> None:
    league = store.leagues.upsert(
        LeagueRow(key="nba", sport="nba", espn_league_id=1, season=2027, team_id=1, as_of=NOW)
    )
    with pytest.raises(RankingsError, match="no synced settings"):
        rank_league(store, league, now=NBA_NOW)


# --- the CSV ----------------------------------------------------------------------------------------------------------


def read(path: Path) -> list[list[str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.reader(handle))


def test_the_category_sheet_is_a_csv_with_a_column_per_category(store: Store, tmp_path: Path) -> None:
    league = nba_league(store, nba_settings("fba_settings_9cat.json"))
    sheet = rank_nba(store, league)
    path = write_csv(sheet, tmp_path / "out")

    assert path == tmp_path / "out" / "rankings_nba.csv"
    rows = read(path)
    assert rows[0] == [
        "rank", "player", "espn_id", "pos", "team", "owner", "mine", "injury", "ros_value", "vor", "games",
        "PTS", "BLK", "STL", "AST", "REB", "TO", "3PM", "FG%", "FT%", "note",
    ]  # fmt: skip
    assert len(rows) == 1 + len(PER_GAME)
    assert all(len(row) == len(rows[0]) for row in rows)
    first = dict(zip(rows[0], rows[1], strict=True))
    assert first["rank"] == "1" and first["owner"] in {"Team 1", "FA"}
    assert float(first["ros_value"]) == pytest.approx(sum(float(first[stat]) for stat in sheet.categories), abs=0.1)
    ghost = dict(zip(rows[0], rows[-1], strict=True))
    assert ghost["player"] == "Unprojected" and ghost["rank"] == "" and ghost["ros_value"] == ""


def test_the_points_sheet_has_per_game_and_games_columns(store: Store, tmp_path: Path) -> None:
    league = nba_league(store, nba_settings("fba_settings_points.json"))
    rows = read(write_csv(rank_nba(store, league), tmp_path))
    assert rows[0][-5:] == ["ros_value", "vor", "per_game", "games", "note"]


def test_a_league_key_is_made_safe_for_a_file_name() -> None:
    league = LeagueRow(key="nba-2_x", sport="nba", espn_league_id=1, season=2027, team_id=1, as_of=NOW)
    sheet = Rankings(league, RankingMetric.POINTS, 1, 2, ())
    assert sheet.filename == "rankings_nba-2_x.csv"


# --- fm rankings ------------------------------------------------------------------------------------------------------


def _root() -> None:
    pass


def invoke(*args: str, expect: int = 0) -> str:
    root = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)
    root.callback()(_root)
    rankings_command.register(root)
    result = runner.invoke(root, ["rankings", *args], catch_exceptions=False)
    assert result.exit_code == expect, result.output
    return result.output


def test_the_fixtures_give_a_sheet_per_league(tmp_path: Path) -> None:
    out = tmp_path / "sheets"
    output = invoke("--fixtures", str(ESPN_FIXTURES), "--out", str(out))

    assert sorted(path.name for path in out.iterdir()) == ["rankings_nba.csv", "rankings_nfl.csv"]
    nfl = read(out / "rankings_nfl.csv")
    assert nfl[0][:3] == ["rank", "player", "espn_id"]
    assert [row[1] for row in nfl[1:3]] == ["Chase Brown", "Rashee Rice"]  # the two players the recorded cards project
    assert nfl[1][0] == "1" and nfl[1][6] == "yes"
    assert all(row[0] == "" for row in nfl[3:])  # the rest have no projection and no rank
    nba = read(out / "rankings_nba.csv")
    assert {row[1] for row in nba[1:3]} == {"Evan Mobley", "Paolo Banchero"}
    assert "offline" in output and "rankings_nfl.csv" in output and "rankings_nba.csv" in output
    assert "lists games for 1 of the 153 days left" in output  # the recorded schedule holds day 1 only


def test_a_league_filter_and_a_folder_with_one_game(tmp_path: Path) -> None:
    out = tmp_path / "sheets"
    output = invoke("--fixtures", str(ESPN_FIXTURES), "--out", str(out), "--league", "nba")
    assert [path.name for path in out.iterdir()] == ["rankings_nba.csv"] and "nfl" not in output

    only_ffl = tmp_path / "ffl-only"
    shutil.copytree(ESPN_FIXTURES / "real" / "ffl", only_ffl)
    output = invoke("--fixtures", str(only_ffl), "--out", str(tmp_path / "one"))
    assert (tmp_path / "one" / "rankings_nfl.csv").is_file()
    assert "skipped nba: no recorded fba views" in output


def test_nothing_written_is_an_error(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    output = invoke("--fixtures", str(empty), "--out", str(tmp_path / "out"), expect=1)
    assert "skipped nfl" in output and "skipped nba" in output
    assert not (tmp_path / "out").exists()


def test_the_default_folder_is_under_the_config_dir() -> None:
    invoke("--fixtures", str(ESPN_FIXTURES), "--league", "nfl")
    assert (paths.config_dir() / "rankings" / "rankings_nfl.csv").is_file()


def test_team_and_schedule_options_are_checked(tmp_path: Path) -> None:
    assert "--team applies only with --fixtures" in invoke("--team", "1", expect=1)
    schedule_file = ESPN_FIXTURES / "ffl_pro_schedule_2026.json"
    assert "--schedule applies" in invoke("--fixtures", str(ESPN_FIXTURES), "--schedule", str(schedule_file), expect=1)


def test_the_fixture_home_ranks_the_synced_league(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    shutil.copytree(HOME, home, ignore=shutil.ignore_patterns("snapshots", "build.py", "README.md", "__pycache__"))
    monkeypatch.setenv(paths.CONFIG_DIR_ENV, str(home))
    monkeypatch.setenv(paths.CACHE_DIR_ENV, str(home / "cache"))
    out = tmp_path / "sheets"
    output = invoke("--as-of", "2026-10-04T15:00Z", "--out", str(out))

    rows = read(out / "rankings_nfl.csv")
    header = rows[0]
    table = [dict(zip(header, row, strict=True)) for row in rows[1:]]
    ranked = [row for row in table if row["rank"]]
    assert [float(row["ros_value"]) for row in ranked] == sorted(
        (float(row["ros_value"]) for row in ranked), reverse=True
    )
    owners = {row["player"]: row["owner"] for row in table}
    assert owners["Bijan Robinson"] == "Fixture Team 1"
    assert owners["Jaylen Warren"] == OWNER_WAIVERS  # the synced pool says so
    assert owners["Ray Davis"] == OWNER_FREE_AGENT
    assert "as of 2026-10-04 15:00Z" in output
    assert next(row for row in table if row["player"] == "Chris Olave")["rank"] == ""  # on IR, nothing projects him


def test_a_synced_store_missing_a_league_is_skipped_with_the_fix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    home = tmp_path / "home"
    shutil.copytree(HOME, home, ignore=shutil.ignore_patterns("snapshots", "build.py", "README.md", "__pycache__"))
    (home / "state.db").unlink()
    monkeypatch.setenv(paths.CONFIG_DIR_ENV, str(home))
    monkeypatch.setenv(paths.CACHE_DIR_ENV, str(home / "cache"))
    output = invoke("--out", str(tmp_path / "sheets"), expect=1)
    assert "skipped nfl: not synced; run fm sync" in output
