"""Basic availability (ROADMAP #15): ESPN designation -> ``p_active``, zeroed by the pro schedule and roster status.

Players come from ``tests/fixtures/model/espn_ffl_pool_week4.json`` (ESPN's own records, Ja'Marr Chase listed
QUESTIONABLE) and the schedules are the real ``proTeamSchedules_wl`` captures: NFL weeks 4 and 5 of 2026 (KC and CAR
on bye in week 5) and the first two NBA days of 2026-27 (SAS and OKC idle on day 1).
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

import pytest

from fm.espn.ids import Game, InjuryStatus, ids_for
from fm.espn.models import Player, PlayersView, ProSchedule
from fm.model.availability import (
    BASE_RATES,
    INACTIVE_DESIGNATIONS,
    MODEL,
    NBA_RATES,
    NFL_RATES,
    REASON_DESIGNATION,
    REASON_INACTIVE,
    REASON_NO_GAME,
    REASON_NO_TEAM,
    UNRECOGNIZED_AS,
    assess,
    assess_many,
    assess_stored,
    base_rate,
    designation,
    expected_points,
    is_unrecognized,
    p_active_for,
    rates_for,
)
from fm.store import AvailabilityRow, PlayerRow, Sport, Store

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
POOL = FIXTURES / "model" / "espn_ffl_pool_week4.json"
NFL_SCHEDULE = FIXTURES / "sports" / "ffl_pro_schedule_2026.json"
NBA_SCHEDULE = FIXTURES / "espn" / "fba_pro_schedule_2027.json"
AS_OF = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)
SEASON = 2026

ALLEN, MAHOMES, CHASE, EAGLES = 3918298, 3139477, 4362628, -16021
BUF_WEEK4_GAME = 401872971
LADDER = (
    InjuryStatus.ACTIVE,
    InjuryStatus.PROBABLE,
    InjuryStatus.QUESTIONABLE,
    InjuryStatus.DOUBTFUL,
    InjuryStatus.OUT,
)


def schedule_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@cache
def nfl_schedule() -> ProSchedule:
    return ProSchedule.model_validate(schedule_json(NFL_SCHEDULE))


@cache
def nba_schedule() -> ProSchedule:
    return ProSchedule.model_validate(schedule_json(NBA_SCHEDULE))


def row_from_espn(player: Player, sport: Sport = "nfl") -> PlayerRow:
    """The ``players`` row the sync job builds from an ESPN player record."""
    ids = ids_for(sport)
    return PlayerRow(
        sport=sport,
        espn_id=player.id,
        full_name=player.full_name,
        default_position_id=player.default_position_id,
        position=ids.position_label(player.default_position_id) if player.default_position_id is not None else None,
        pro_team_id=player.pro_team_id,
        pro_team=ids.pro_team(player.pro_team_id) if player.pro_team_id is not None else None,
        eligible_slot_ids=list(player.eligible_slots),
        injury_status=player.injury_status,
        injured=player.injured,
        active=player.active,
        as_of=AS_OF,
    )


@cache
def pool_rows() -> dict[int, PlayerRow]:
    view = PlayersView.model_validate(json.loads(POOL.read_text(encoding="utf-8")))
    return {entry.id: row_from_espn(entry.player) for entry in view.players}


def pooled(espn_id: int, **changes: Any) -> PlayerRow:
    return pool_rows()[espn_id].model_copy(update=changes)


def nba_player(espn_id: int, pro_team_id: int | None, injury_status: str | None = None, **changes: Any) -> PlayerRow:
    return PlayerRow(
        sport="nba",
        espn_id=espn_id,
        full_name=f"Player {espn_id}",
        pro_team_id=pro_team_id,
        injury_status=injury_status,
        as_of=AS_OF,
        **changes,
    )


# --- the designation mapping ---


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, 1.0),
        ("", 1.0),
        ("ACTIVE", 1.0),
        ("NORMAL", 1.0),
        ("PROBABLE", 0.95),
        ("QUESTIONABLE", 0.70),
        (" questionable ", 0.70),
        ("DOUBTFUL", 0.05),
        ("OUT", 0.0),
        ("INJURY_RESERVE", 0.0),
        ("IR", 0.0),
        ("SUSPENSION", 0.0),
        ("SUSPENDED", 0.0),
        ("DAY_TO_DAY", 0.75),
        ("PHYSICALLY_UNABLE", 0.70),  # unrecognised: the questionable rate, never the healthy one
    ],
)
def test_nfl_designation_mapping(raw: str | None, expected: float) -> None:
    assert p_active_for(raw, "nfl") == pytest.approx(expected)
    assert p_active_for(raw, Game.FFL) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, 1.0),
        ("ACTIVE", 1.0),
        ("PROBABLE", 0.95),
        ("DAY_TO_DAY", 0.60),
        ("QUESTIONABLE", 0.50),
        ("DOUBTFUL", 0.10),
        ("OUT", 0.0),
        ("SUSPENSION", 0.0),
        ("GAME_TIME_DECISION", 0.50),  # unrecognised
    ],
)
def test_nba_designation_mapping(raw: str | None, expected: float) -> None:
    assert p_active_for(raw, "nba") == pytest.approx(expected)
    assert p_active_for(raw, "fba") == pytest.approx(expected)


@pytest.mark.parametrize("sport", ["nfl", "nba"])
def test_every_designation_has_a_rate_and_the_ladder_descends(sport: Sport) -> None:
    rates = BASE_RATES[sport]
    assert set(rates) == set(InjuryStatus)
    assert all(0.0 <= rate <= 1.0 for rate in rates.values())
    assert all(rates[status] == 0.0 for status in INACTIVE_DESIGNATIONS)
    assert rates[InjuryStatus.ACTIVE] == rates[InjuryStatus.UNKNOWN] == 1.0
    ladder = [rates[status] for status in LADDER]
    assert ladder == sorted(ladder, reverse=True)
    assert len(set(ladder)) == len(ladder)
    assert 0.0 < rates[UNRECOGNIZED_AS] < 1.0


def test_sports_have_their_own_tables() -> None:
    assert BASE_RATES["nfl"] is NFL_RATES and BASE_RATES["nba"] is NBA_RATES
    assert NFL_RATES[InjuryStatus.QUESTIONABLE] != NBA_RATES[InjuryStatus.QUESTIONABLE]


def test_designation_normalises_and_tells_missing_from_unrecognised() -> None:
    assert designation("NORMAL", "nfl") is InjuryStatus.ACTIVE
    assert designation(None, "nfl") is InjuryStatus.UNKNOWN
    assert designation("NEW_THING", "nba") is InjuryStatus.UNKNOWN
    assert is_unrecognized("NEW_THING", "nba")
    assert not is_unrecognized(None, "nba") and not is_unrecognized("  ", "nfl") and not is_unrecognized("OUT", "nfl")


def test_rates_can_be_overridden_and_are_validated() -> None:
    tuned = {InjuryStatus.QUESTIONABLE: 0.6}
    assert rates_for("nfl", tuned)[InjuryStatus.QUESTIONABLE] == 0.6
    assert rates_for("nfl", tuned)[InjuryStatus.DOUBTFUL] == NFL_RATES[InjuryStatus.DOUBTFUL]
    assert base_rate(InjuryStatus.QUESTIONABLE, "nfl", rates=tuned) == 0.6
    assert p_active_for("QUESTIONABLE", "nfl", rates=tuned) == 0.6
    assert NFL_RATES[InjuryStatus.QUESTIONABLE] == 0.70  # the module table is untouched
    with pytest.raises(ValueError, match="QUESTIONABLE"):
        rates_for("nfl", {InjuryStatus.QUESTIONABLE: 1.2})
    with pytest.raises(ValueError, match="OUT"):
        rates_for("nba", {InjuryStatus.OUT: -0.1})


# --- assessing players against the schedule ---


def test_a_healthy_player_with_a_game() -> None:
    row = assess(pooled(ALLEN), season=SEASON, scoring_period=4, as_of=AS_OF, schedule=nfl_schedule())
    assert isinstance(row, AvailabilityRow)
    assert (row.sport, row.espn_id, row.season, row.scoring_period_id) == ("nfl", ALLEN, SEASON, 4)
    assert row.p_active == 1.0
    assert row.has_game
    assert row.game_time == datetime(2026, 10, 4, 17, 0, tzinfo=UTC)  # BUF's 1:00 p.m. ET kickoff
    assert row.designation == "ACTIVE"
    assert row.inputs["model"] == MODEL
    assert row.inputs["game_id"] == BUF_WEEK4_GAME
    assert row.inputs["provisional"] is False
    assert "reason" not in row.inputs
    assert row.as_of == AS_OF


def test_a_questionable_player_gets_the_questionable_rate() -> None:
    chase = pooled(CHASE)
    assert chase.injury_status == "QUESTIONABLE"  # as ESPN listed him
    row = assess(chase, season=SEASON, scoring_period=4, as_of=AS_OF, schedule=nfl_schedule())
    assert row.p_active == pytest.approx(0.70)
    assert row.designation == "QUESTIONABLE"
    assert row.inputs["base_rate"] == pytest.approx(0.70)
    assert row.has_game


def test_a_bye_week_is_zero_whatever_the_designation() -> None:
    mahomes = pooled(MAHOMES)  # KC is on bye in week 5
    row = assess(mahomes, season=SEASON, scoring_period=5, as_of=AS_OF, schedule=nfl_schedule())
    assert (row.p_active, row.has_game, row.game_time) == (0.0, False, None)
    assert row.inputs["reason"] == REASON_NO_GAME
    assert row.designation == "ACTIVE"
    week4 = assess(mahomes, season=SEASON, scoring_period=4, as_of=AS_OF, schedule=nfl_schedule())
    assert week4.p_active == 1.0 and week4.has_game


def test_out_players_have_their_game_but_no_chance_to_play() -> None:
    out = pooled(ALLEN, injury_status="OUT")
    row = assess(out, season=SEASON, scoring_period=4, as_of=AS_OF, schedule=nfl_schedule())
    assert row.p_active == 0.0
    assert row.has_game  # the game happens; he is not in it
    assert row.inputs["reason"] == REASON_DESIGNATION
    assert row.designation == "OUT"


def test_inactive_players_and_players_without_a_team_are_zero() -> None:
    schedule = nfl_schedule()
    inactive = assess(pooled(ALLEN, active=False), season=SEASON, scoring_period=4, as_of=AS_OF, schedule=schedule)
    assert inactive.p_active == 0.0 and inactive.inputs["reason"] == REASON_INACTIVE
    for team in (0, None):  # ESPN's free-agent pseudo-team, and a record without a team
        unsigned = pooled(ALLEN, pro_team_id=team)
        with_schedule = assess(unsigned, season=SEASON, scoring_period=4, as_of=AS_OF, schedule=schedule)
        without = assess(unsigned, season=SEASON, scoring_period=4, as_of=AS_OF)
        for row in (with_schedule, without):
            assert (row.p_active, row.has_game) == (0.0, False)
            assert row.inputs["reason"] == REASON_NO_TEAM


def test_a_d_st_has_no_designation_and_counts_as_active() -> None:
    row = assess(pooled(EAGLES), season=SEASON, scoring_period=4, as_of=AS_OF, schedule=nfl_schedule())
    assert row.p_active == 1.0
    assert row.designation is None
    assert row.inputs["designation"] == InjuryStatus.UNKNOWN.value


def test_an_unrecognised_designation_is_flagged_not_trusted() -> None:
    row = assess(pooled(ALLEN, injury_status="NEW_LIST"), season=SEASON, scoring_period=4, as_of=AS_OF)
    assert row.p_active == pytest.approx(NFL_RATES[UNRECOGNIZED_AS])
    assert row.inputs["designation_unrecognized"] is True
    assert row.inputs["designation_raw"] == "NEW_LIST"
    assert row.designation == InjuryStatus.UNKNOWN.value


def test_without_a_schedule_the_game_is_assumed() -> None:
    row = assess(pooled(CHASE), season=SEASON, scoring_period=4, as_of=AS_OF)
    assert row.has_game and row.game_time is None
    assert row.p_active == pytest.approx(0.70)
    assert row.inputs["schedule"] is False


@pytest.mark.parametrize("period", [3, 6])
def test_a_schedule_that_does_not_cover_the_period_raises(period: int) -> None:
    with pytest.raises(ValueError, match=f"periods 4-5, not scoring period {period}"):
        assess(pooled(ALLEN), season=SEASON, scoring_period=period, as_of=AS_OF, schedule=nfl_schedule())
    with pytest.raises(ValueError, match="no periods"):
        assess(pooled(ALLEN), season=SEASON, scoring_period=period, as_of=AS_OF, schedule=ProSchedule())


def test_a_provisional_start_is_recorded() -> None:
    raw = schedule_json(NFL_SCHEDULE)
    for team in raw["settings"]["proTeams"]:
        for game in team.get("proGamesByScoringPeriod", {}).get("4", []):
            if game["id"] == BUF_WEEK4_GAME:
                game["startTimeTBD"] = True
                game["validForLocking"] = False
    schedule = ProSchedule.model_validate(raw)
    row = assess(pooled(ALLEN), season=SEASON, scoring_period=4, as_of=AS_OF, schedule=schedule)
    assert row.inputs["provisional"] is True
    assert row.has_game and row.p_active == 1.0


def test_nba_days_follow_the_daily_schedule() -> None:
    schedule = nba_schedule()  # day 1: BOS, NYK, DEN, LAL play; SAS and OKC are idle
    boston = assess(nba_player(1, 2, "DAY_TO_DAY"), season=2027, scoring_period=1, as_of=AS_OF, schedule=schedule)
    assert boston.p_active == pytest.approx(0.60)
    assert boston.game_time == datetime(2026, 10, 20, 23, 30, tzinfo=UTC)
    san_antonio = assess(nba_player(2, 24), season=2027, scoring_period=1, as_of=AS_OF, schedule=schedule)
    assert (san_antonio.p_active, san_antonio.has_game) == (0.0, False)
    assert san_antonio.inputs["reason"] == REASON_NO_GAME
    day2 = assess(nba_player(2, 24), season=2027, scoring_period=2, as_of=AS_OF, schedule=schedule)
    assert day2.p_active == 1.0 and day2.sport == "nba"


def test_assess_many_keeps_the_order_given() -> None:
    players = [pooled(CHASE), pooled(ALLEN), pooled(MAHOMES)]
    rows = assess_many(players, season=SEASON, scoring_period=5, as_of=AS_OF, schedule=nfl_schedule())
    assert [row.espn_id for row in rows] == [CHASE, ALLEN, MAHOMES]
    assert [row.p_active for row in rows] == pytest.approx([0.70, 1.0, 0.0])


# --- the store ---


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        yield opened


def test_assess_stored_writes_and_replaces_rows(store: Store) -> None:
    store.players.upsert_many(pool_rows().values())
    schedule = nfl_schedule()
    rows = assess_stored(
        store, "nfl", [MAHOMES, CHASE, 999999], season=SEASON, scoring_period=5, as_of=AS_OF, schedule=schedule
    )
    assert [row.espn_id for row in rows] == [MAHOMES, CHASE]  # by ESPN id; the unknown id is skipped
    saved = store.availability.for_period("nfl", SEASON, 5)
    assert {(row.espn_id, row.p_active) for row in saved} == {(MAHOMES, 0.0), (CHASE, 0.70)}
    assert store.availability.get("nfl", CHASE, SEASON, 5) == rows[1]

    store.players.upsert(pooled(CHASE, injury_status="OUT"))
    assess_stored(store, Game.FFL, [CHASE], season=SEASON, scoring_period=5, as_of=AS_OF, schedule=schedule)
    replaced = store.availability.get("nfl", CHASE, SEASON, 5)
    assert replaced is not None and replaced.p_active == 0.0 and replaced.designation == "OUT"

    dry = assess_stored(store, "nfl", [ALLEN], season=SEASON, scoring_period=5, as_of=AS_OF, save=False)
    assert dry[0].p_active == 1.0
    assert store.availability.get("nfl", ALLEN, SEASON, 5) is None


# --- expected value ---


def test_expected_points_is_p_active_times_the_projection() -> None:
    assert expected_points(20.0, 0.7) == pytest.approx(14.0)
    row = assess(pooled(CHASE), season=SEASON, scoring_period=4, as_of=AS_OF, schedule=nfl_schedule())
    assert expected_points(20.43, row) == pytest.approx(0.70 * 20.43)
    bye = assess(pooled(MAHOMES), season=SEASON, scoring_period=5, as_of=AS_OF, schedule=nfl_schedule())
    assert expected_points(18.7, bye) == 0.0
    with pytest.raises(ValueError, match="within"):
        expected_points(10.0, 1.5)
