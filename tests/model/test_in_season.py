"""In-season evidence (``fm.model.in_season``): the game log ``fm sync`` stores read into a role factor, a benched
streak and the production mix, with the constants from ``data/in_season.toml``.

The player of record is Wolf-shaped: a healthy second-round pick ESPN projected for 24 minutes a night who has been a
DNP-coach's-decision every game. His stored actual lines are empty for every team game and his season line shows no
games played; nothing about him is injured. Two other shapes sit beside him: a rotation player who sat one game out
healthy and one game out injured, and a player with no evidence at all, who must come out at 1.0.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fm.model.in_season import (
    DEFAULT_PARAMS,
    Evidence,
    InSeasonError,
    InSeasonParams,
    evidence_from,
    load_evidence,
    load_params,
    parse_params,
    played_in,
    role_factors,
    shrink,
    shrink_line,
    weight,
)
from fm.store import AvailabilityRow, PlayerRow, ProjectionRow, Store

SEASON = 2027
AS_OF = datetime(2026, 10, 24, 12, 0, tzinfo=UTC)
WOLF, ROTATION, NOBODY = 5_000_001, 5_000_002, 5_000_003
NBA = DEFAULT_PARAMS["nba"]


def actual(espn_id: int, period: int, stats: Mapping[str, float], *, source: str = "espn") -> ProjectionRow:
    return ProjectionRow(
        sport="nba",
        espn_id=espn_id,
        source=source,
        kind="actual",
        season=SEASON,
        scoring_period_id=period,
        stats=dict(stats),
        as_of=AS_OF,
    )


def projected(espn_id: int, period: int, stats: Mapping[str, float]) -> ProjectionRow:
    return actual(espn_id, period, stats).model_copy(update={"kind": "projected"})


def availability(espn_id: int, period: int, designation: str | None, p_active: float) -> AvailabilityRow:
    return AvailabilityRow(
        sport="nba",
        espn_id=espn_id,
        season=SEASON,
        scoring_period_id=period,
        designation=designation,
        p_active=p_active,
        as_of=AS_OF,
    )


def player(espn_id: int, name: str, *, injury_status: str | None = "ACTIVE") -> PlayerRow:
    return PlayerRow(sport="nba", espn_id=espn_id, full_name=name, injury_status=injury_status, as_of=AS_OF)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        yield opened


# --- the constants ----------------------------------------------------------------------------------------------------


def test_the_committed_file_states_the_defaults() -> None:
    loaded = load_params()
    assert dict(loaded) == dict(DEFAULT_PARAMS)
    assert loaded["nba"] == InSeasonParams(line_k=8.0, role_k=2.0, role_window=10, min_benched=2)
    assert loaded["nfl"] == InSeasonParams(line_k=4.0, role_k=1.5, role_window=6, min_benched=2)


def test_parse_params_overrides_one_sport_and_keeps_the_rest(tmp_path: Path) -> None:
    parsed = parse_params("[nba]\nline_k = 4\nmin_benched = 3\n")
    assert (parsed["nba"].line_k, parsed["nba"].min_benched, parsed["nba"].role_k) == (4.0, 3, 2.0)
    assert parsed["nfl"] == DEFAULT_PARAMS["nfl"]
    assert parse_params("") == DEFAULT_PARAMS
    file = tmp_path / "in_season.toml"
    file.write_text("[nfl]\nrole_window = 3\n", encoding="utf-8")
    assert load_params(file)["nfl"].role_window == 3


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("[mlb]\nline_k = 1\n", r"\[mlb\] is not a sport"),
        ("nba = 3\n", r"\[nba\] should be a table"),
        ("[nba]\nlinek = 1\n", r"unknown keys \['linek'\]"),
        ("[nba]\nline_k = 0\n", "line_k should be a finite number > 0"),
        ("[nba]\nrole_k = -1.5\n", "role_k should be a finite number > 0"),
        ("[nba]\nrole_k = true\n", "role_k should be a finite number > 0"),
        ("[nba]\nrole_window = 0\n", "role_window should be an integer >= 1"),
        ("[nba]\nmin_benched = 1.5\n", "min_benched should be an integer >= 1"),
        ("[nba\n", "in_season.toml"),
    ],
)
def test_parse_params_refuses_a_bad_file_naming_the_problem(text: str, match: str) -> None:
    with pytest.raises(InSeasonError, match=match):
        parse_params(text)


def test_load_params_needs_the_file(tmp_path: Path) -> None:
    with pytest.raises(InSeasonError, match="not found"):
        load_params(tmp_path / "missing.toml")
    with pytest.raises(InSeasonError, match="line_k"):
        InSeasonParams(line_k=float("inf"), role_k=1.0, role_window=1, min_benched=1)


# --- the role factor and the benched streak ---------------------------------------------------------------------------


def test_no_evidence_is_a_role_factor_of_one_and_never_a_penalty() -> None:
    nobody = Evidence(NOBODY)
    assert nobody.role_factor(NBA) == 1.0 and nobody.benched() == 0 and nobody.team_games == 0
    assert nobody.window(NBA) == (0, 0) and nobody.describe(NBA) is None
    assert nobody.numbers(NBA) == {
        "played": 0,
        "healthy": 0,
        "team_games": 0,
        "games": 0.0,
        "role_factor": 1.0,
        "benched": 0,
    }
    assert evidence_from(NOBODY, (), (), current_period=5) == nobody
    assert role_factors({NOBODY: nobody}, NBA) == {NOBODY: 1.0}


def test_a_healthy_dnp_lowers_the_role_factor_and_counts_toward_the_benched_streak() -> None:
    """Wolf sat three team games healthy: 0 of 3 with role_k 2 is 0.4, and the streak is 3. The current period's line
    is a game not yet played, so it is left out."""
    log = [actual(WOLF, period, {}) for period in (1, 2, 3)] + [actual(WOLF, 4, {})]
    wolf = evidence_from(WOLF, log, (), current_period=4, injury_status="ACTIVE")
    assert wolf.log == ((1, False, True), (2, False, True), (3, False, True))
    assert wolf.team_games == 3 and wolf.window(NBA) == (0, 3)
    assert wolf.role_factor(NBA) == pytest.approx(0.4) and wolf.benched() == 3
    assert wolf.describe(NBA) == "played 0 of 3 healthy games"
    assert wolf.numbers(NBA)["role_factor"] == pytest.approx(0.4)
    # one game played after the streak ends it, and the factor climbs
    played = evidence_from(
        WOLF, [*log, actual(WOLF, 4, {"MIN": 12.0, "PTS": 4.0})], (), current_period=5, injury_status="ACTIVE"
    )
    assert played.benched() == 0 and played.role_factor(NBA) == pytest.approx((1 + 2) / (4 + 2))
    assert played.describe(NBA) == "played 1 of 4 healthy games"


def test_games_missed_injured_are_not_held_against_him() -> None:
    """The rotation player played game 1, sat game 2 healthy and game 3 ruled OUT: 1 of 2 healthy games, a streak of 1
    (game 3 is skipped, not counted)."""
    log = [actual(ROTATION, 1, {"MIN": 30.0, "PTS": 18.0}), actual(ROTATION, 2, {}), actual(ROTATION, 3, {})]
    rows = [availability(ROTATION, 3, "OUT", 0.0)]
    rotation = evidence_from(ROTATION, log, rows, current_period=4)
    # with availability rows, the nearest earlier row decides a period without one: game 1 and 2 read game 3's OUT
    # only when there is no earlier row, so here they fall back to the nearest later one and read OUT as well
    assert rotation.log == ((1, True, False), (2, False, False), (3, False, False))
    assert rotation.window(NBA) == (0, 0) and rotation.role_factor(NBA) == 1.0 and rotation.benched() == 0
    cleared = [availability(ROTATION, 1, "ACTIVE", 1.0), availability(ROTATION, 3, "OUT", 0.0)]
    rotation = evidence_from(ROTATION, log, cleared, current_period=4)
    assert rotation.log == ((1, True, True), (2, False, True), (3, False, False))  # game 2 reads game 1's row
    assert rotation.window(NBA) == (1, 2) and rotation.role_factor(NBA) == pytest.approx((1 + 2) / (2 + 2))
    assert rotation.benched() == 1 and rotation.describe(NBA) == "played 1 of 2 healthy games"
    # without any availability row the current injury status stands for every game
    hurt = evidence_from(WOLF, [actual(WOLF, 1, {}), actual(WOLF, 2, {})], (), current_period=3, injury_status="OUT")
    assert hurt.role_factor(NBA) == 1.0 and hurt.benched() == 0 and hurt.window(NBA) == (0, 0)


def test_the_role_window_and_the_pseudo_games_are_the_constants() -> None:
    log = [actual(WOLF, period, {}) for period in range(1, 13)] + [actual(WOLF, 13, {"MIN": 20.0})]
    wolf = evidence_from(WOLF, log, (), current_period=14, injury_status="ACTIVE")
    assert wolf.team_games == 13 and wolf.window(NBA) == (1, 10)  # the last ten healthy games
    assert wolf.role_factor(NBA) == pytest.approx(3 / 12)
    forgiving = InSeasonParams(line_k=8.0, role_k=10.0, role_window=2, min_benched=1)
    assert wolf.window(forgiving) == (1, 2) and wolf.role_factor(forgiving) == pytest.approx(11 / 12)


def test_played_in_reads_minutes_games_or_any_stat() -> None:
    assert played_in({"MIN": 12.0}) and played_in({"GP": 1.0}) and played_in({"REB": 2.0})
    assert not played_in({}) and not played_in({"PTS": 0.0, "MIN": 0.0})
    # the actuals of other sources are not the game log
    foreign = evidence_from(WOLF, [actual(WOLF, 1, {"MIN": 30.0}, source="darko")], (), current_period=2)
    assert foreign.team_games == 0 and foreign.season_line is None
    assert evidence_from(WOLF, [projected(WOLF, 1, {"MIN": 30.0})], (), current_period=2).team_games == 0


def test_the_season_line_and_its_games() -> None:
    season = evidence_from(WOLF, [actual(WOLF, 0, {"GP": 5.0, "PTS": 50.0})], (), current_period=6)
    assert season.games == 5.0 and season.season_line == {"GP": 5.0, "PTS": 50.0}
    # a season line without a games count takes the game log's count
    counted = evidence_from(
        WOLF,
        [actual(WOLF, 0, {"PTS": 20.0}), actual(WOLF, 1, {"MIN": 10.0}), actual(WOLF, 2, {"MIN": 10.0})],
        (),
        current_period=3,
        injury_status="ACTIVE",
    )
    assert counted.games == 2.0
    assert evidence_from(WOLF, [actual(WOLF, 1, {"MIN": 10.0})], (), current_period=2).games == 0.0


# --- the production mix -----------------------------------------------------------------------------------------------


def test_the_mix_weighs_the_actuals_by_games_over_games_plus_line_k() -> None:
    assert weight(0, NBA) == 0.0 and weight(8, NBA) == pytest.approx(0.5) and weight(24, NBA) == pytest.approx(0.75)
    assert shrink(10.0, 20.0, 0, NBA) == 10.0
    assert shrink(10.0, 20.0, 8, NBA) == pytest.approx(15.0)
    assert shrink(10.0, 20.0, 8_000_000, NBA) == pytest.approx(20.0, abs=1e-4)  # a season of games is the season
    projected_line = {"PTS": 10.0, "REB": 4.0, "FGM": 4.0, "GP": 1.0}
    actual_line = {"PTS": 20.0, "FGM": 8.0, "GP": 8.0}
    mixed = shrink_line(projected_line, actual_line, 8, NBA)
    assert mixed == {"PTS": 15.0, "REB": 4.0, "FGM": 6.0, "GP": 1.0}  # REB keeps the projection, GP too
    assert shrink_line(projected_line, actual_line, 0, NBA) == projected_line
    assert shrink_line(projected_line, actual_line, 0, NBA) is not projected_line
    assert shrink_line(projected_line, {}, 8, NBA) == projected_line  # nothing to mix with
    nfl = DEFAULT_PARAMS["nfl"]
    assert shrink_line({"RY": 60.0}, {"RY": 100.0}, 4, nfl)["RY"] == pytest.approx(80.0)


# --- the store --------------------------------------------------------------------------------------------------------


def test_load_evidence_reads_the_game_log_the_availability_rows_and_the_status(store: Store) -> None:
    store.players.upsert_many(
        [player(WOLF, "Danny Wolf"), player(ROTATION, "Rota Tion"), player(NOBODY, "No Body", injury_status=None)]
    )
    store.projections.upsert_many(
        [
            actual(WOLF, 0, {"GP": 0.0}),
            actual(WOLF, 1, {}),
            actual(WOLF, 2, {}),
            actual(ROTATION, 0, {"GP": 1.0, "PTS": 18.0, "MIN": 30.0}),
            actual(ROTATION, 1, {"PTS": 18.0, "MIN": 30.0}),
            actual(ROTATION, 2, {}),
            projected(WOLF, 0, {"GP": 71.0, "MIN": 1711.0}),  # the preseason line: not evidence
            actual(WOLF, 1, {"MIN": 30.0}, source="darko"),  # another source's actuals are not the game log
        ]
    )
    store.availability.upsert_many([availability(ROTATION, 2, "OUT", 0.0), availability(WOLF, 2, "ACTIVE", 1.0)])
    evidence = load_evidence(store, "nba", SEASON, [NOBODY, WOLF, ROTATION, WOLF], current_period=3)
    assert list(evidence) == [WOLF, ROTATION, NOBODY]
    wolf, rotation, nobody = evidence[WOLF], evidence[ROTATION], evidence[NOBODY]
    assert wolf.log == ((1, False, True), (2, False, True)) and wolf.benched() == 2
    assert wolf.role_factor(NBA) == pytest.approx(0.5) and wolf.games == 0.0 and wolf.season_line == {"GP": 0.0}
    assert rotation.log == ((1, True, False), (2, False, False))  # one row, OUT, stands for every game
    assert rotation.games == 1.0 and rotation.role_factor(NBA) == 1.0
    assert nobody == Evidence(NOBODY)

    # the repositories behind it
    log = store.projections.for_players("nba", SEASON, [ROTATION, WOLF], source="espn")
    assert [(row.espn_id, row.scoring_period_id) for row in log] == [(WOLF, 0), (WOLF, 1), (WOLF, 2)] + [
        (ROTATION, 0),
        (ROTATION, 1),
        (ROTATION, 2),
    ]
    assert all(row.kind == "actual" and row.source == "espn" for row in log)
    assert store.projections.for_players("nba", SEASON, [WOLF], source="espn", kind="projected") == [
        projected(WOLF, 0, {"GP": 71.0, "MIN": 1711.0})
    ]
    assert store.projections.for_players("nba", SEASON, [], source="espn") == []
    assert store.projections.for_players("nba", SEASON + 1, [WOLF], source="espn") == []
    rows = store.availability.for_players("nba", SEASON, [ROTATION, WOLF, NOBODY])
    assert [(row.espn_id, row.scoring_period_id, row.designation) for row in rows] == [
        (WOLF, 2, "ACTIVE"),
        (ROTATION, 2, "OUT"),
    ]
    assert store.availability.for_players("nba", SEASON, []) == []
