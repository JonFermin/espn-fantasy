"""Availability (ROADMAP #15, #22): designations, NFL practice trends, the NBA's official injury report and bounded news
signals -> ``p_active``, zeroed by the pro schedule and roster status, plus late-game pivots.

Players come from ``tests/fixtures/model/espn_ffl_pool_week4.json`` (ESPN's own records, Ja'Marr Chase listed
QUESTIONABLE) and the schedules are the real ``proTeamSchedules_wl`` captures: NFL weeks 4 and 5 of 2026 (KC and CAR
on bye in week 5), the first two NBA days of 2026-27 as hand-built for the read client (SAS and OKC idle on day 1) and
the real NBA capture of days 1-3 (``tests/fixtures/sports/``, with GSW at LAL on day 2, Oct 21). Practice reports come
from nflverse's injuries capture joined to ESPN ids through the crosswalk built from ``db_playerids.csv``; the official
injury report is the one-page PDF under ``tests/fixtures/sources/nba_injuries/``, whose GSW at LAL rows match the real
day-2 game (GSW had not submitted). News signals are built here in the shape the advisor stores.
"""

from __future__ import annotations

import json
import logging
import math
from collections.abc import Iterable, Iterator
from datetime import UTC, date, datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Any

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from fm.espn.ids import FBA, FFL, Game, IdMaps, InjuryStatus, ids_for
from fm.espn.models import Player, PlayersView, ProSchedule
from fm.espn.settings import LockType
from fm.model.availability import (
    BASE_RATES,
    INACTIVE_DESIGNATIONS,
    MODEL,
    NBA_RATES,
    NEWS_BOUND,
    NFL_RATES,
    PRACTICE_SHIFTS,
    REASON_DESIGNATION,
    REASON_INACTIVE,
    REASON_NO_GAME,
    REASON_NO_TEAM,
    RESOLUTION_LEAD,
    SOURCE_ESPN,
    SOURCE_OFFICIAL,
    TREND_STEP,
    UNRECOGNIZED_AS,
    AvailabilityParams,
    LineupPlayer,
    OfficialReport,
    Participation,
    PracticeReport,
    assess,
    assess_many,
    assess_stored,
    base_rate,
    can_pivot,
    designation,
    expected_points,
    is_rest_day,
    is_uncertain,
    is_unrecognized,
    late_pivots,
    merge_practice,
    news_window,
    p_active_for,
    participation,
    plan_pivots,
    practice_from_nflverse,
    practice_trend,
    rates_for,
    recorded_practice,
    resolves_at,
    weigh_news,
)
from fm.model.ids import Crosswalk, build_crosswalk
from fm.sources.nba_injuries import OfficialInjuryReport, parse_official_report, parse_report_lines
from fm.sports.nba import NBA
from fm.sports.nfl import NFL
from fm.store import AvailabilityRow, NewsItemRow, NewsSignalRow, PlayerRow, SignalKind, Sport, Store

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
POOL = FIXTURES / "model" / "espn_ffl_pool_week4.json"
NFL_SCHEDULE = FIXTURES / "sports" / "ffl_pro_schedule_2026.json"
NBA_SCHEDULE = FIXTURES / "espn" / "fba_pro_schedule_2027.json"
NBA_REAL_SCHEDULE = FIXTURES / "sports" / "fba_pro_schedule_2027.json"
NFLVERSE = FIXTURES / "sources" / "nflverse"
OFFICIAL_REPORT = FIXTURES / "sources" / "nba_injuries" / "Injury-Report_2026-10-21_05PM.pdf"
AS_OF = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)
SEASON = 2026
NBA_SEASON = 2027
LOGGER = "fm.model.availability"

ALLEN, MAHOMES, CHASE, EAGLES = 3918298, 3139477, 4362628, -16021
HENRY, GIBBS = 3043078, 4429795
PRICE, SHRADER = 4685512, 4571557  # in nflverse's week-4 injury report, with Mahomes
MAHOMES_GSIS = "00-0033873"
LUKA, LEBRON, REAVES, CURRY, SGA = 3945274, 1966, 4066457, 3975, 4278073
BUF_WEEK4_GAME = 401872971
WEDNESDAY = date(2026, 9, 30)  # the first practice day before week 4's Sunday games
LADDER = (
    InjuryStatus.ACTIVE,
    InjuryStatus.PROBABLE,
    InjuryStatus.QUESTIONABLE,
    InjuryStatus.DOUBTFUL,
    InjuryStatus.OUT,
)
RB, WR = FFL.slot_id("RB"), FFL.slot_id("WR")
FLEX = FFL.slot_id("FLEX")
UTIL = FBA.slot_id("UTIL")


def schedule_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@cache
def nfl_schedule() -> ProSchedule:
    return ProSchedule.model_validate(schedule_json(NFL_SCHEDULE))


@cache
def nba_schedule() -> ProSchedule:
    return ProSchedule.model_validate(schedule_json(NBA_SCHEDULE))


@cache
def nba_real_schedule() -> ProSchedule:
    return ProSchedule.model_validate(schedule_json(NBA_REAL_SCHEDULE))


@cache
def official_report() -> OfficialInjuryReport:
    return parse_official_report(OFFICIAL_REPORT.read_bytes())


@cache
def injuries_frame() -> pl.DataFrame:
    return pl.read_parquet(NFLVERSE / "injuries_2026.parquet")


@cache
def crosswalk() -> Crosswalk:
    return build_crosswalk(pl.read_csv(NFLVERSE / "db_playerids.csv", null_values=["NA", "NULL", ""]), as_of=AS_OF)


def team_id(ids: IdMaps, abbrev: str) -> int:
    return next(team for team, code in ids.pro_teams.items() if code == abbrev)


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


def nfl_player(espn_id: int, team: str, position: str, injury_status: str | None = None) -> PlayerRow:
    """An NFL player the pool fixture lacks, with the plugin's slots for his position."""
    return PlayerRow(
        sport="nfl",
        espn_id=espn_id,
        full_name=f"Player {espn_id}",
        position=position,
        pro_team_id=team_id(FFL, team),
        pro_team=team,
        eligible_slot_ids=sorted(NFL.eligible_slots(position)),
        injury_status=injury_status,
        as_of=AS_OF,
    )


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


def nba_named(espn_id: int, name: str, team: str, injury_status: str | None = None) -> PlayerRow:
    return PlayerRow(
        sport="nba",
        espn_id=espn_id,
        full_name=name,
        pro_team_id=team_id(FBA, team),
        pro_team=team,
        injury_status=injury_status,
        as_of=AS_OF,
    )


def practice(*levels: str, start: date = WEDNESDAY, rest: Iterable[int] = ()) -> list[PracticeReport]:
    """One report per day from ``start``, in the order given; ``rest`` are the indexes of veteran rest days."""
    rested = set(rest)
    return [
        PracticeReport(day=start + timedelta(days=offset), participation=Participation(level), rest=offset in rested)
        for offset, level in enumerate(levels)
    ]


def signal(
    delta: float,
    *,
    confidence: float = 1.0,
    kind: SignalKind = "injury",
    published: datetime = datetime(2026, 10, 2, 16, 0, tzinfo=UTC),
    espn_id: int = CHASE,
    sport: Sport = "nfl",
    source_url: str | None = "https://www.espn.com/nfl/story/_/id/46000001",
    signal_id: int = 1,
    news_item_id: int = 1,
) -> NewsSignalRow:
    """A news signal as the advisor stores it (``news_signals``)."""
    return NewsSignalRow(
        id=signal_id,
        news_item_id=news_item_id,
        sport=sport,
        espn_id=espn_id,
        kind=kind,
        severity="moderate",
        p_active_delta=delta,
        confidence=confidence,
        summary="test signal",
        source_url=source_url,
        published_at=published,
        created_at=published + timedelta(minutes=5),
    )


def week4(player: PlayerRow, **options: Any) -> AvailabilityRow:
    return assess(player, season=SEASON, scoring_period=4, as_of=AS_OF, schedule=nfl_schedule(), **options)


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
    assert row.inputs["designation_source"] == SOURCE_ESPN
    assert row.inputs["resolves_at"] == "2026-10-04T15:30:00.000000Z"  # inactives, 90 minutes before kickoff
    assert "reason" not in row.inputs
    assert not {"practice", "official", "news"} & set(row.inputs)  # nothing beyond the designation was given
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
    assert "resolves_at" not in row.inputs


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
    assert boston.inputs["resolves_at"] == "2026-10-20T23:00:00.000000Z"  # late scratches, 30 minutes before the tip
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


# --- NFL practice reports ---


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Did Not Participate In Practice", Participation.DNP),  # nflverse's wording
        ("Limited Participation in Practice", Participation.LIMITED),
        ("Full Participation in Practice", Participation.FULL),
        ("DNP", Participation.DNP),
        ("did not participate", Participation.DNP),
        ("LP", Participation.LIMITED),
        ("Limited", Participation.LIMITED),
        ("FP", Participation.FULL),
        (" full ", Participation.FULL),
        ("Out", None),  # a game status, not a practice participation
        ("", None),
        (None, None),
    ],
)
def test_practice_participation_is_read_from_nflverse_and_short_forms(
    raw: str | None, expected: Participation | None
) -> None:
    assert participation(raw) is expected


def test_veteran_rest_days_are_recognised() -> None:
    assert is_rest_day("Knee", "Not injury related - resting player")  # Ja'Marr Chase, week 1
    assert is_rest_day("Rest")
    assert not is_rest_day("Not injury related - personal matter", None)
    assert not is_rest_day("Hamstring", None) and not is_rest_day()


@pytest.mark.parametrize(
    ("levels", "shift"),
    [
        (("FP",), PRACTICE_SHIFTS[Participation.FULL]),
        (("LP", "LP", "LP"), 0.0),
        (("DNP", "DNP", "DNP"), PRACTICE_SHIFTS[Participation.DNP]),
        (("DNP", "LP", "FP"), PRACTICE_SHIFTS[Participation.FULL] + 2 * TREND_STEP),
        (("DNP", "LP"), TREND_STEP),
        (("FP", "LP", "DNP"), PRACTICE_SHIFTS[Participation.DNP] - 2 * TREND_STEP),
        (("FP", "LP"), -TREND_STEP),
    ],
)
def test_the_trend_is_the_latest_day_plus_the_direction(levels: tuple[str, ...], shift: float) -> None:
    trend = practice_trend(practice(*levels))
    assert trend.shift == pytest.approx(shift)
    assert trend.latest is Participation(levels[-1]) and trend.first is Participation(levels[0])


@pytest.mark.parametrize(
    ("levels", "expected"),
    [
        (("FP",), 0.85),
        (("DNP", "LP", "FP"), 0.95),  # trending up: practised fully by the end of the week
        (("LP", "LP", "LP"), 0.70),
        (("DNP", "LP"), 0.75),
        (("DNP", "DNP", "DNP"), 0.45),
        (("FP", "LP", "DNP"), 0.35),  # trending down: a setback during the week
    ],
)
def test_practice_trends_shift_a_questionable_player(levels: tuple[str, ...], expected: float) -> None:
    row = week4(pooled(CHASE), practice=practice(*levels))
    assert row.p_active == pytest.approx(expected)
    recorded = row.inputs["practice"]
    assert recorded["applied"] is True
    assert [report["participation"] for report in recorded["reports"]] == list(levels)
    assert recorded["latest"] == levels[-1] and recorded["shift"] == pytest.approx(expected - 0.70)


def test_improving_beats_flat_beats_declining() -> None:
    chase = pooled(CHASE)
    improving, flat, declining = (
        week4(chase, practice=practice(*levels)).p_active for levels in (("DNP", "LP"), ("LP", "LP"), ("FP", "LP"))
    )
    assert improving > flat > declining


def test_practice_shifts_are_clamped_to_zero_and_one() -> None:
    healthy = week4(pooled(ALLEN), practice=practice("DNP", "LP", "FP"))
    assert healthy.p_active == 1.0  # a healthy full practice cannot go above certain
    missed = week4(pooled(ALLEN), practice=practice("DNP"))
    assert missed.p_active == pytest.approx(0.75)  # no designation yet, but he missed practice
    doubtful = week4(pooled(CHASE, injury_status="DOUBTFUL"), practice=practice("FP", "LP", "DNP"))
    assert doubtful.p_active == 0.0
    assert doubtful.inputs["practice"]["shift"] == pytest.approx(-0.35)


def test_rest_days_carry_no_signal() -> None:
    rested = week4(pooled(CHASE), practice=practice("DNP", "FP", "FP", rest=[0]))
    assert rested.p_active == pytest.approx(0.85)  # the Wednesday rest day is not a missed practice
    only_rest = week4(pooled(CHASE), practice=practice("DNP", rest=[0]))
    assert only_rest.p_active == pytest.approx(0.70)
    assert only_rest.inputs["practice"]["latest"] is None and only_rest.inputs["practice"]["shift"] == 0.0


def test_practice_cannot_lift_a_hard_zero() -> None:
    out = week4(pooled(CHASE, injury_status="OUT"), practice=practice("LP", "FP", "FP"))
    assert out.p_active == 0.0 and out.inputs["reason"] == REASON_DESIGNATION
    assert out.inputs["practice"]["applied"] is False  # recorded, not applied
    bye = assess(
        pooled(MAHOMES),
        season=SEASON,
        scoring_period=5,
        as_of=AS_OF,
        schedule=nfl_schedule(),
        practice=practice("FP", start=date(2026, 10, 7)),
    )
    assert bye.p_active == 0.0 and bye.inputs["practice"]["applied"] is False


def test_reports_merge_one_per_day_later_wins() -> None:
    wednesday, thursday = practice("DNP", "LP")
    corrected = PracticeReport(day=wednesday.day, participation=Participation.LIMITED, source="correction")
    merged = merge_practice([thursday, wednesday], [corrected])
    assert [(report.day, report.participation) for report in merged] == [
        (WEDNESDAY, Participation.LIMITED),
        (WEDNESDAY + timedelta(days=1), Participation.LIMITED),
    ]
    assert practice_trend([thursday, wednesday]).steps == 1  # ordered by day, not by input order


@given(
    designation_raw=st.sampled_from([None, "ACTIVE", "PROBABLE", "QUESTIONABLE", "DOUBTFUL", "DAY_TO_DAY", "NEW"]),
    days=st.lists(st.tuples(st.sampled_from(list(Participation)), st.booleans()), min_size=1, max_size=6),
)
def test_practice_keeps_p_active_a_probability_and_a_better_latest_day_never_hurts(
    designation_raw: str | None, days: list[tuple[Participation, bool]]
) -> None:
    reports = [
        PracticeReport(day=WEDNESDAY + timedelta(days=offset), participation=level, rest=rest)
        for offset, (level, rest) in enumerate(days)
    ]
    player = pooled(CHASE, injury_status=designation_raw)
    row = week4(player, practice=reports)
    assert 0.0 <= row.p_active <= 1.0
    counted = [offset for offset, report in enumerate(reports) if not report.rest]
    if not counted:
        assert row.p_active == week4(player).p_active  # rest days alone say nothing
        return
    improved = [
        report.model_copy(update={"participation": Participation.FULL}) if offset == counted[-1] else report
        for offset, report in enumerate(reports)
    ]
    assert week4(player, practice=improved).p_active >= row.p_active


def test_practice_reports_come_from_nflverse_through_the_crosswalk() -> None:
    week4_reports = practice_from_nflverse(injuries_frame(), crosswalk(), season=SEASON, week=4, observed=AS_OF)
    assert set(week4_reports.by_player) == {MAHOMES, PRICE, SHRADER}
    assert week4_reports.unmapped == () and week4_reports.warnings == ()
    (mahomes,) = week4_reports.by_player[MAHOMES]
    assert (mahomes.participation, mahomes.injury, mahomes.rest) == (Participation.FULL, "Knee", False)
    assert mahomes.day == date(2026, 10, 4) and mahomes.source == "nflverse"  # the fantasy day it was observed on
    (price,) = week4_reports.by_player[PRICE]
    assert price.participation is Participation.DNP and price.injury == "Chest"

    week1 = practice_from_nflverse(injuries_frame(), crosswalk(), season=SEASON, week=1, observed=AS_OF)
    (chase,) = week1.by_player[CHASE]
    assert chase.participation is Participation.DNP and chase.rest  # "Not injury related - resting player"
    assert chase.injury == "Knee"


def test_nflverse_rows_that_cannot_be_read_are_reported() -> None:
    frame = injuries_frame().with_columns(
        pl.when(pl.col("gsis_id") == MAHOMES_GSIS)
        .then(pl.lit("Did not practice (estimated)"))
        .otherwise(pl.col("practice_status"))
        .alias("practice_status")
    )
    empty = Crosswalk(())
    converted = practice_from_nflverse(frame, empty, season=SEASON, week=4, observed=AS_OF)
    assert converted.by_player == {}
    assert sorted(converted.unmapped) == ["00-0039576", "00-0041512"]  # Shrader and Price, unmapped here
    assert any("does not map" in warning for warning in converted.warnings)
    assert any("Did not practice (estimated)" in warning for warning in converted.warnings)
    with pytest.raises(ValueError, match="practice_status"):
        practice_from_nflverse(frame.drop("practice_status"), crosswalk(), season=SEASON, week=4, observed=AS_OF)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        yield opened


def with_mahomes_status(status: str | None) -> pl.DataFrame:
    """nflverse's injuries frame as it read on a day Mahomes' latest practice was ``status`` (week 4)."""
    if status is None:
        return injuries_frame()
    return injuries_frame().with_columns(
        pl.when((pl.col("gsis_id") == MAHOMES_GSIS) & (pl.col("week") == 4))
        .then(pl.lit(status))
        .otherwise(pl.col("practice_status"))
        .alias("practice_status")
    )


def test_assess_stored_builds_the_weeks_trend_from_daily_observations(store: Store) -> None:
    store.players.upsert(pooled(MAHOMES, injury_status="QUESTIONABLE"))
    observations = [
        (datetime(2026, 9, 30, 22, 0, tzinfo=UTC), "Did Not Participate In Practice", 0.45),
        (datetime(2026, 10, 1, 22, 0, tzinfo=UTC), "Limited Participation in Practice", 0.75),
        (datetime(2026, 10, 2, 22, 0, tzinfo=UTC), None, 0.95),  # Friday: the capture itself, a full practice
        (datetime(2026, 10, 2, 23, 0, tzinfo=UTC), None, 0.95),  # fetched again the same day: no new day
    ]
    for observed, status, expected in observations:
        reports = practice_from_nflverse(
            with_mahomes_status(status), crosswalk(), season=SEASON, week=4, observed=observed
        )
        (row,) = assess_stored(
            store,
            "nfl",
            [MAHOMES],
            season=SEASON,
            scoring_period=4,
            as_of=observed,
            schedule=nfl_schedule(),
            practice=reports.by_player,
        )
        assert row.p_active == pytest.approx(expected)
    saved = store.availability.get("nfl", MAHOMES, SEASON, 4)
    assert saved is not None
    log = recorded_practice(saved)
    assert [(report.day, report.participation) for report in log] == [
        (date(2026, 9, 30), Participation.DNP),
        (date(2026, 10, 1), Participation.LIMITED),
        (date(2026, 10, 2), Participation.FULL),
    ]
    assert saved.inputs["practice"]["steps"] == 2
    # A later assessment without new reports keeps the week's log.
    (again,) = assess_stored(
        store, "nfl", [MAHOMES], season=SEASON, scoring_period=4, as_of=AS_OF, schedule=nfl_schedule()
    )
    assert again.p_active == pytest.approx(0.95) and len(again.inputs["practice"]["reports"]) == 3


def test_a_recorded_log_that_does_not_read_is_skipped(caplog: pytest.LogCaptureFixture) -> None:
    good = practice("LP")[0].model_dump(mode="json")
    row = week4(pooled(CHASE)).model_copy(
        update={"inputs": {"practice": {"reports": [good, {"day": "Wednesday", "participation": "??"}]}}}
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        assert recorded_practice(row) == (PracticeReport.model_validate(good),)
    assert "does not read" in caplog.text
    assert recorded_practice(None) == () and recorded_practice(week4(pooled(CHASE))) == ()


def test_practice_is_an_nfl_input() -> None:
    with pytest.raises(ValueError, match="NFL"):
        assess(nba_player(1, 2), season=NBA_SEASON, scoring_period=1, as_of=AS_OF, practice=practice("FP"))
    routed = assess_many(
        [pooled(CHASE), nba_player(CHASE, 2)],
        season=SEASON,
        scoring_period=4,
        as_of=AS_OF,
        practice={CHASE: practice("FP")},
    )
    assert routed[0].p_active == pytest.approx(0.85) and "practice" not in routed[1].inputs


# --- the NBA's official injury report ---


def day2(
    player: PlayerRow, report: OfficialReport | OfficialInjuryReport | None = None, **options: Any
) -> AvailabilityRow:
    """Day 2 (Wed Oct 21, 2026) of the real NBA schedule, where GSW play at LAL at 10:00 p.m. ET."""
    official = report if report is not None else official_report()
    return assess(
        player,
        season=NBA_SEASON,
        scoring_period=2,
        as_of=AS_OF,
        schedule=nba_real_schedule(),
        official=official,
        **options,
    )


def test_the_official_report_sets_the_designation_for_the_game() -> None:
    luka = day2(nba_named(LUKA, "Luka Dončić", "LAL", "DAY_TO_DAY"))
    assert luka.p_active == pytest.approx(NBA_RATES[InjuryStatus.DOUBTFUL])  # ESPN's day-to-day was 0.60
    assert luka.designation == "DOUBTFUL"
    assert luka.inputs["designation"] == "DAY_TO_DAY" and luka.inputs["designation_source"] == SOURCE_OFFICIAL
    assert luka.inputs["official"] == {
        "listed": True,
        "status": "Doubtful",
        "designation": "DOUBTFUL",
        "reason": "Injury/Illness - Left Calf; Strain",
        "game_day": "2026-10-21",
        "game_time": "10:00 (ET)",
        "matchup": "GSW@LAL",
        "team": "Los Angeles Lakers",
        "player": "Doncic, Luka",
    }

    lebron = day2(nba_named(LEBRON, "LeBron James", "LAL"))
    assert (lebron.p_active, lebron.designation) == (0.0, "OUT")
    assert lebron.inputs["reason"] == REASON_DESIGNATION
    assert lebron.inputs["official"]["reason"] == "Not With Team"

    reaves = day2(nba_named(REAVES, "Austin Reaves", "LAL"))
    assert reaves.p_active == 1.0 and reaves.designation is None
    assert reaves.inputs["official"] == {"listed": False, "not_submitted": False}
    assert reaves.inputs["designation_source"] == SOURCE_ESPN

    curry = day2(nba_named(CURRY, "Stephen Curry", "GSW", "DAY_TO_DAY"))  # GSW had not submitted its report
    assert curry.p_active == pytest.approx(NBA_RATES[InjuryStatus.DAY_TO_DAY])
    assert curry.inputs["official"] == {"listed": False, "not_submitted": True}


@pytest.mark.parametrize(
    ("status", "espn", "expected"),
    [
        ("Out", None, 0.0),
        ("Doubtful", "DAY_TO_DAY", 0.10),
        ("Questionable", "DAY_TO_DAY", 0.50),
        ("Probable", "DAY_TO_DAY", 0.95),
        ("Available", "OUT", 1.0),  # cleared for this game, whatever ESPN still shows
    ],
)
def test_official_statuses_shift_p_active_both_ways(status: str, espn: str | None, expected: float) -> None:
    entry = f"Los Angeles Lakers Hachimura, Rui {status} Injury/Illness - Right Knee; Soreness"
    report = parse_report_lines(["Injury Report: 10/21/26 05:00 PM", f"10/21/2026 10:00 (ET) GSW@LAL {entry}"])
    row = day2(nba_named(4066648, "Rui Hachimura", "LAL", espn), report)
    assert row.p_active == pytest.approx(expected)
    assert row.inputs["official"]["status"] == status
    assert row.inputs.get("reason") == (REASON_DESIGNATION if status == "Out" else None)  # ESPN's OUT no longer rules
    with_news = day2(
        nba_named(4066648, "Rui Hachimura", "LAL", espn), report, signals=[signal(-0.2, espn_id=4066648, sport="nba")]
    )
    assert with_news.p_active == pytest.approx(max(0.0, expected - 0.2))  # news moves all but a hard zero


def test_the_report_applies_only_to_its_game_day() -> None:
    sga = nba_named(SGA, "Shai Gilgeous-Alexander", "OKC", "DAY_TO_DAY")  # listed Probable for Oct 21
    schedule = nba_real_schedule()  # OKC plays at SAS on day 1 and hosts DEN on day 3, and is idle on Oct 21
    for period in (1, 3):
        row = assess(
            sga, season=NBA_SEASON, scoring_period=period, as_of=AS_OF, schedule=schedule, official=official_report()
        )
        assert row.p_active == pytest.approx(0.60) and row.inputs["official"]["listed"] is False
    idle = assess(sga, season=NBA_SEASON, scoring_period=2, as_of=AS_OF, schedule=schedule, official=official_report())
    assert idle.p_active == 0.0 and "official" not in idle.inputs
    unscheduled = assess(sga, season=NBA_SEASON, scoring_period=2, as_of=AS_OF, official=official_report())
    assert unscheduled.p_active == pytest.approx(0.95)  # without a schedule the entry is taken for the game


def test_report_names_match_espn_spellings() -> None:
    index = OfficialReport(
        parse_report_lines(
            [
                "10/21/2026 08:30 (ET) DAL@HOU Houston Rockets Smith Jr., Jabari Questionable Injury/Illness - Ankle",
                "10/21/2026 08:00 (ET) IND@NOP New Orleans Pelicans Jones, Herbert Out Injury/Illness - Back; Strain",
                "10/21/2026 07:00 (ET) MIA@CHA Charlotte Hornets Martin, Cody Doubtful Injury/Illness - Knee",
            ]
        )
    )
    jabari = index.entry_for(nba_named(1, "Jabari Smith Jr.", "HOU"), date(2026, 10, 21))
    assert jabari is not None and jabari.status == "Questionable"  # suffix dropped on both sides
    herb = index.entry_for(nba_named(2, "Herb Jones", "NOP"))  # one spelling shortens the other
    assert herb is not None and herb.status == "Out"
    assert index.entry_for(nba_named(3, "Caleb Martin", "CHA")) is None  # Cody's twin: neither name shortens the other
    assert index.entry_for(nba_named(4, "Jabari Smith Jr.", "DAL")) is None  # another team
    assert index.entry_for(nba_named(5, "Jabari Smith Jr.", "HOU"), date(2026, 10, 22)) is None  # another day
    assert index.entry_for(nba_named(6, "Free Agent", "FA")) is None


def test_conflicting_entries_leave_espns_designation(caplog: pytest.LogCaptureFixture) -> None:
    report = parse_report_lines(
        [
            "10/21/2026 10:00 (ET) GSW@LAL Los Angeles Lakers Doncic, Luka Out Injury/Illness - Left Calf; Strain",
            "Doncic, Luka Probable Injury/Illness - Left Calf; Strain",
        ]
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        row = day2(nba_named(LUKA, "Luka Doncic", "LAL", "DAY_TO_DAY"), report)
    assert row.p_active == pytest.approx(0.60) and row.inputs["official"]["listed"] is False
    assert "lists Luka Doncic (LAL) 2 times" in caplog.text


def test_the_official_report_is_an_nba_input() -> None:
    with pytest.raises(ValueError, match="NBA"):
        week4(pooled(CHASE), official=official_report())


def test_assess_stored_reads_the_report_for_nba_players(store: Store) -> None:
    store.players.upsert_many(
        [nba_named(LUKA, "Luka Doncic", "LAL", "DAY_TO_DAY"), nba_named(LEBRON, "LeBron James", "LAL")]
    )
    rows = assess_stored(
        store,
        "nba",
        [LUKA, LEBRON],
        season=NBA_SEASON,
        scoring_period=2,
        as_of=AS_OF,
        schedule=nba_real_schedule(),
        official=official_report(),
    )
    assert {row.espn_id: row.p_active for row in rows} == {LEBRON: 0.0, LUKA: pytest.approx(0.10)}
    saved = store.availability.get("nba", LUKA, NBA_SEASON, 2)
    assert saved is not None and saved.inputs["official"]["status"] == "Doubtful"


# --- Claude's news signals ---


def test_a_news_signal_moves_p_active_by_its_confidence() -> None:
    row = week4(pooled(CHASE), signals=[signal(-0.2, confidence=0.5)])
    assert row.p_active == pytest.approx(0.60)
    news = row.inputs["news"]
    assert news["before"] == pytest.approx(0.70) and news["applied"] is True
    assert news["delta"] == pytest.approx(-0.10) and news["clamped"] is False
    (entry,) = news["signals"]
    assert entry["status"] == "counted" and entry["clamped"] is False
    assert (entry["proposed"], entry["confidence"], entry["weighted"]) == (-0.2, 0.5, pytest.approx(-0.1))
    assert entry["source_url"] == "https://www.espn.com/nfl/story/_/id/46000001"
    assert entry["published_at"] == "2026-10-02T16:00:00.000000Z"


@pytest.mark.parametrize(
    ("delta", "expected"),
    [(-0.9, 0.40), (0.8, 1.0), (-NEWS_BOUND, 0.40), (NEWS_BOUND, 1.0), (-0.1, 0.60)],
)
def test_each_signal_is_clamped_to_the_bound(delta: float, expected: float) -> None:
    row = week4(pooled(CHASE), signals=[signal(delta)])
    assert row.p_active == pytest.approx(expected)
    (entry,) = row.inputs["news"]["signals"]
    assert entry["proposed"] == delta
    assert abs(entry["bounded"]) <= NEWS_BOUND and entry["clamped"] is (abs(delta) > NEWS_BOUND)


def test_signals_together_are_clamped_to_the_bound(caplog: pytest.LogCaptureFixture) -> None:
    signals = [signal(-0.3, kind="injury"), signal(-0.3, kind="suspension", signal_id=2, news_item_id=2)]
    with caplog.at_level(logging.INFO, logger=LOGGER):
        row = week4(pooled(CHASE), signals=signals)
    news = row.inputs["news"]
    assert news["requested"] == pytest.approx(-0.6) and news["delta"] == pytest.approx(-NEWS_BOUND)
    assert news["clamped"] is True
    assert row.p_active == pytest.approx(0.40)
    assert "together asked for -0.60; clamped to -0.30" in caplog.text


def test_the_latest_signal_of_a_kind_supersedes_earlier_ones() -> None:
    signals = [
        signal(0.1, published=datetime(2026, 10, 3, 18, 0, tzinfo=UTC), signal_id=2, news_item_id=2),
        signal(-0.3, published=datetime(2026, 10, 1, 18, 0, tzinfo=UTC)),
        signal(-0.2, kind="rest", published=datetime(2026, 10, 2, 18, 0, tzinfo=UTC), signal_id=3, news_item_id=3),
    ]
    row = week4(pooled(CHASE), signals=signals)
    statuses = [(entry["signal_id"], entry["status"]) for entry in row.inputs["news"]["signals"]]
    assert statuses == [(1, "superseded"), (3, "counted"), (2, "counted")]  # in publication order
    assert row.p_active == pytest.approx(0.70 + 0.1 - 0.2)


def test_the_same_news_twice_counts_once() -> None:
    espn = signal(-0.2)
    rotowire = signal(-0.2, published=espn.published_at + timedelta(minutes=30), signal_id=2, news_item_id=2)
    assert week4(pooled(CHASE), signals=[espn, rotowire]).p_active == pytest.approx(0.50)


def test_uncited_and_invalid_signals_are_ignored_and_logged(caplog: pytest.LogCaptureFixture) -> None:
    uncited = signal(-0.3, source_url=None)
    blank = signal(-0.3, source_url="  ", signal_id=2, news_item_id=2)
    invalid = signal(math.nan, signal_id=3, news_item_id=3)
    with caplog.at_level(logging.INFO, logger=LOGGER):
        row = week4(pooled(CHASE), signals=[uncited, blank, invalid])
    assert row.p_active == pytest.approx(0.70)
    assert [entry["status"] for entry in row.inputs["news"]["signals"]] == ["uncited", "uncited", "invalid"]
    assert row.inputs["news"]["signals"][2]["proposed"] is None  # NaN is not JSON
    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert sum("cites no source; ignored" in message for message in warnings) == 2
    assert any("non-finite change (nan); ignored" in message for message in warnings)


def test_applied_signals_are_logged_with_their_source(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER):
        week4(pooled(CHASE), signals=[signal(-0.9, confidence=0.8, signal_id=7)])
    messages = [(record.levelno, record.getMessage()) for record in caplog.records if record.name == LOGGER]
    assert (
        logging.WARNING,
        "availability: nfl 4362628 (Ja'Marr Chase) period 4: news signal 7 (injury) proposed -0.90, "
        "outside the +/-0.30 bound; clamped to -0.30",
    ) in messages
    assert (
        logging.INFO,
        "availability: nfl 4362628 (Ja'Marr Chase) period 4: news signal 7 (injury), published "
        "2026-10-02T16:00:00.000000Z, moves p_active -0.24 (-0.30 x confidence 0.80), "
        "source https://www.espn.com/nfl/story/_/id/46000001",
    ) in messages


def test_news_counts_from_the_teams_previous_game_until_kickoff() -> None:
    chase = pooled(CHASE)  # CIN: home to JAX in week 4 (Oct 4, 17:00 UTC), at MIA in week 5 (Oct 11, 17:00 UTC)
    saturday = datetime(2026, 10, 10, 12, 0, tzinfo=UTC)
    since, until = news_window(chase, 5, as_of=saturday, schedule=nfl_schedule())
    assert (since, until) == (datetime(2026, 10, 4, 17, 0, tzinfo=UTC), saturday)
    after_kickoff = datetime(2026, 10, 12, 12, 0, tzinfo=UTC)
    assert news_window(chase, 5, as_of=after_kickoff, schedule=nfl_schedule())[1] == datetime(
        2026, 10, 11, 17, 0, tzinfo=UTC
    )
    signals = [
        signal(-0.3, published=datetime(2026, 10, 4, 12, 0, tzinfo=UTC)),  # week 4's pregame news
        signal(-0.2, kind="rest", published=datetime(2026, 10, 5, 12, 0, tzinfo=UTC), signal_id=2, news_item_id=2),
        signal(-0.3, kind="other", published=datetime(2026, 10, 10, 18, 0, tzinfo=UTC), signal_id=3, news_item_id=3),
    ]
    row = assess(chase, season=SEASON, scoring_period=5, as_of=saturday, schedule=nfl_schedule(), signals=signals)
    assert [entry["status"] for entry in row.inputs["news"]["signals"]] == ["stale", "counted", "future"]
    assert row.p_active == pytest.approx(0.50)
    assert row.inputs["news"]["since"] == "2026-10-04T17:00:00.000000Z"


def test_without_an_earlier_game_news_counts_over_the_lookback() -> None:
    since, until = news_window(pooled(CHASE), 4, as_of=AS_OF, schedule=nfl_schedule())  # no week 3 in the capture
    assert (since, until) == (AS_OF - timedelta(days=7), AS_OF)
    nba = news_window(nba_named(LUKA, "Luka Doncic", "LAL"), 2, as_of=AS_OF)
    assert nba == (AS_OF - timedelta(days=2), AS_OF)
    assert news_window(pooled(CHASE), 4, as_of=AS_OF, lookback=timedelta(days=1))[0] == AS_OF - timedelta(days=1)
    with pytest.raises(ValueError, match="timezone-aware"):
        news_window(pooled(CHASE), 4, as_of=datetime(2026, 10, 4, 15, 0))


def test_news_never_lifts_or_lowers_a_hard_zero(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger=LOGGER):
        out = week4(pooled(CHASE, injury_status="OUT"), signals=[signal(0.3)])
    assert out.p_active == 0.0 and out.inputs["news"]["applied"] is False
    assert "not applied: p_active is a hard zero" in caplog.text
    bye = assess(
        pooled(MAHOMES),
        season=SEASON,
        scoring_period=5,
        as_of=AS_OF,
        schedule=nfl_schedule(),
        signals=[signal(0.3, espn_id=MAHOMES)],
    )
    assert bye.p_active == 0.0 and bye.inputs["reason"] == REASON_NO_GAME


def test_a_claude_only_signal_cannot_bench_a_healthy_player() -> None:
    row = week4(pooled(ALLEN), signals=[signal(-1.0, espn_id=ALLEN)])
    assert row.p_active == pytest.approx(1.0 - NEWS_BOUND)
    assert row.inputs["news"]["before"] == 1.0  # the engine's number without Claude


def test_signals_must_be_about_the_player() -> None:
    with pytest.raises(ValueError, match="pass each player his own signals"):
        week4(pooled(ALLEN), signals=[signal(-0.2)])
    rows = assess_many(
        [pooled(CHASE), pooled(ALLEN)],
        season=SEASON,
        scoring_period=4,
        as_of=AS_OF,
        schedule=nfl_schedule(),
        signals=[signal(-0.2), signal(-0.2, espn_id=LUKA, sport="nba", signal_id=2)],
    )
    assert [row.p_active for row in rows] == pytest.approx([0.50, 1.0])
    assert "news" not in rows[1].inputs


@given(
    deltas=st.lists(
        st.tuples(
            st.floats(min_value=-2.0, max_value=2.0),
            st.floats(min_value=0.0, max_value=1.0),
            st.sampled_from(["injury", "role", "rest", "suspension", "other"]),
        ),
        min_size=1,
        max_size=6,
    ),
    designation_raw=st.sampled_from([None, "QUESTIONABLE", "DOUBTFUL", "PROBABLE"]),
)
def test_news_moves_p_active_at_most_the_bound(
    deltas: list[tuple[float, float, SignalKind]], designation_raw: str | None
) -> None:
    signals = [
        signal(
            delta,
            confidence=confidence,
            kind=kind,
            signal_id=index + 1,
            news_item_id=index + 1,
            published=datetime(2026, 10, 1, tzinfo=UTC) + timedelta(hours=index),
        )
        for index, (delta, confidence, kind) in enumerate(deltas)
    ]
    row = week4(pooled(CHASE, injury_status=designation_raw), signals=signals)
    before = row.inputs["news"]["before"]
    assert 0.0 <= row.p_active <= 1.0
    assert abs(row.p_active - before) <= NEWS_BOUND + 1e-9
    effect = weigh_news(signals, since=AS_OF - timedelta(days=7), until=AS_OF)
    assert abs(effect.delta) <= NEWS_BOUND and len(effect.counted) <= len({kind for _, _, kind in deltas})


def test_assess_stored_reads_each_players_signals_from_the_store(store: Store) -> None:
    store.players.upsert_many(pool_rows().values())
    item = store.news.ingest(
        NewsItemRow(
            source="espn",
            external_id="46000001",
            sport="nfl",
            title="Chase limited by ankle",
            url="https://www.espn.com/nfl/story/_/id/46000001",
            espn_ids=[CHASE],
            published_at=datetime(2026, 10, 2, 16, 0, tzinfo=UTC),
            fetched_at=datetime(2026, 10, 2, 16, 5, tzinfo=UTC),
        )
    )
    assert item is not None
    store.news_signals.insert(signal(-0.2, news_item_id=item.row_id).model_copy(update={"id": None}))
    store.news_signals.insert(
        signal(-0.3, news_item_id=item.row_id, published=datetime(2026, 9, 20, tzinfo=UTC)).model_copy(
            update={"id": None}
        )
    )
    rows = assess_stored(
        store, "nfl", [CHASE, ALLEN], season=SEASON, scoring_period=4, as_of=AS_OF, schedule=nfl_schedule()
    )
    by_id = {row.espn_id: row for row in rows}
    assert by_id[CHASE].p_active == pytest.approx(0.50)
    assert [entry["status"] for entry in by_id[CHASE].inputs["news"]["signals"]] == ["counted"]  # the old one unread
    assert "news" not in by_id[ALLEN].inputs
    explicit = assess_stored(
        store,
        "nfl",
        [CHASE],
        season=SEASON,
        scoring_period=4,
        as_of=AS_OF,
        schedule=nfl_schedule(),
        signals=[],
        save=False,
    )
    assert explicit[0].p_active == pytest.approx(0.70)  # the caller's (empty) list replaces the store's


# --- parameters ---


def test_parameters_are_validated_and_overridable() -> None:
    with pytest.raises(ValueError, match="no shift for FP"):
        AvailabilityParams(practice_shifts={Participation.DNP: -0.2, Participation.LIMITED: 0.0})
    with pytest.raises(ValueError, match="within"):
        AvailabilityParams(practice_shifts={**PRACTICE_SHIFTS, Participation.FULL: 1.5})
    with pytest.raises(ValueError, match="trend_step"):
        AvailabilityParams(trend_step=-0.1)
    with pytest.raises(ValueError, match="news_bound"):
        AvailabilityParams(news_bound=math.nan)
    with pytest.raises(ValueError, match="news_lookback"):
        AvailabilityParams(news_lookback={"nfl": timedelta(days=7)})
    tight = AvailabilityParams(news_bound=0.1, practice_shifts={**PRACTICE_SHIFTS, Participation.FULL: 0.2})
    assert week4(pooled(CHASE), signals=[signal(-0.3)], params=tight).p_active == pytest.approx(0.60)
    assert week4(pooled(CHASE), practice=practice("FP"), params=tight).p_active == pytest.approx(0.90)


# --- the store ---


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


def test_full_rows_round_trip_through_the_store(store: Store) -> None:
    store.players.upsert(pooled(CHASE))
    (row,) = assess_stored(
        store,
        "nfl",
        [CHASE],
        season=SEASON,
        scoring_period=4,
        as_of=AS_OF,
        schedule=nfl_schedule(),
        practice={CHASE: practice("DNP", "LP")},
        signals=[signal(-0.9, confidence=0.5)],
    )
    assert store.availability.get("nfl", CHASE, SEASON, 4) == row  # every input is plain JSON
    assert row.p_active == pytest.approx(0.75 - 0.15)


# --- expected value ---


def test_expected_points_is_p_active_times_the_projection() -> None:
    assert expected_points(20.0, 0.7) == pytest.approx(14.0)
    row = assess(pooled(CHASE), season=SEASON, scoring_period=4, as_of=AS_OF, schedule=nfl_schedule())
    assert expected_points(20.43, row) == pytest.approx(0.70 * 20.43)
    bye = assess(pooled(MAHOMES), season=SEASON, scoring_period=5, as_of=AS_OF, schedule=nfl_schedule())
    assert expected_points(18.7, bye) == 0.0
    with pytest.raises(ValueError, match="within"):
        expected_points(10.0, 1.5)


# --- late-game pivots ---


def lineup_player(
    player: PlayerRow,
    points: float,
    *,
    period: int = 4,
    lock_type: LockType = LockType.INDIVIDUAL_GAME,
    schedule: ProSchedule | None = None,
    **options: Any,
) -> LineupPlayer:
    """A player as the lineup sees him: assessed, projected, and locked by the NFL plugin under ``lock_type``."""
    games = schedule if schedule is not None else nfl_schedule()
    row = assess(player, season=SEASON, scoring_period=period, as_of=AS_OF, schedule=games, **options)
    assert player.pro_team_id is not None
    lock = NFL.lock_time(player.pro_team_id, period, games, lock_type=lock_type)
    return LineupPlayer(row, points, frozenset(player.eligible_slot_ids), lock)


GIBBS_Q = pooled(GIBBS, injury_status="QUESTIONABLE")  # DET, Sunday night (00:20 UTC Oct 5)
OLAVE = nfl_player(4361370, "NO", "WR")  # NO, Monday night
SMITH_NJIGBA = nfl_player(4430878, "SEA", "WR")  # SEA, Sunday 4:25 p.m. ET (20:25 UTC)


def test_a_players_status_settles_before_his_game() -> None:
    gibbs = week4(GIBBS_Q)
    assert resolves_at(gibbs) == datetime(2026, 10, 4, 22, 50, tzinfo=UTC)  # 90 minutes before Sunday night
    assert resolves_at(gibbs, lead=timedelta(minutes=60)) == datetime(2026, 10, 4, 23, 20, tzinfo=UTC)
    luka = day2(nba_named(LUKA, "Luka Doncic", "LAL"))
    assert resolves_at(luka) == datetime(2026, 10, 22, 1, 30, tzinfo=UTC)  # 30 minutes before the 10:00 p.m. tip
    assert RESOLUTION_LEAD["nba"] < RESOLUTION_LEAD["nfl"]
    assert resolves_at(assess(pooled(CHASE), season=SEASON, scoring_period=4, as_of=AS_OF)) is None
    assert is_uncertain(gibbs) and not is_uncertain(week4(pooled(ALLEN))) and not is_uncertain(0.0)


def test_a_late_questionable_player_keeps_a_later_pivot_in_flex() -> None:
    gibbs = lineup_player(GIBBS_Q, 20.0)
    henry = lineup_player(pooled(HENRY), 15.0)  # BAL, Sunday 1:00 p.m.: locked before Gibbs' status is known
    olave = lineup_player(OLAVE, 12.0)
    assert late_pivots(gibbs, RB, [henry, olave]) == []  # a WR cannot take an RB slot
    assert late_pivots(gibbs, FLEX, [henry, olave]) == [olave]
    assert not can_pivot(gibbs, FLEX, henry) and can_pivot(gibbs, FLEX, olave)

    rb_slot = plan_pivots([(RB, gibbs), (FLEX, henry)], [olave])
    flex_slot = plan_pivots([(RB, henry), (FLEX, gibbs)], [olave])
    assert rb_slot.base == flex_slot.base == pytest.approx(0.70 * 20.0 + 15.0)
    assert rb_slot.pivots == {} and rb_slot.gain == 0.0
    assert flex_slot.pivots == {GIBBS: 4361370}
    assert flex_slot.gain == pytest.approx(0.30 * 12.0)
    assert flex_slot.expected > rb_slot.expected  # the FLEX trick


def test_pivots_are_assigned_optimally_not_greedily() -> None:
    chase = lineup_player(pooled(CHASE), 18.0, practice=practice("DNP", "DNP"))  # 0.45, Sunday 1:00 p.m.
    gibbs = lineup_player(GIBBS_Q, 20.0)  # 0.70, Sunday night
    olave = lineup_player(OLAVE, 12.0)  # Monday night: can cover either
    smith_njigba = lineup_player(SMITH_NJIGBA, 10.0)  # 4:25 p.m.: can cover Chase only
    assert chase.availability.p_active == pytest.approx(0.45)
    assert late_pivots(chase, WR, [olave, smith_njigba]) == [olave, smith_njigba]
    assert late_pivots(gibbs, FLEX, [olave, smith_njigba]) == [olave]

    plan = plan_pivots([(WR, chase), (FLEX, gibbs)], [olave, smith_njigba])
    assert plan.pivots == {CHASE: smith_njigba.espn_id, GIBBS: olave.espn_id}
    assert plan.gain == pytest.approx(0.55 * 10.0 + 0.30 * 12.0)
    greedy = 0.55 * 12.0  # Olave to Chase first leaves Gibbs uncovered
    assert plan.gain > greedy


def test_a_swap_needs_both_players_unlocked_once_the_status_is_known() -> None:
    gibbs = lineup_player(GIBBS_Q, 20.0)  # DET, Sunday night: his status is known at 22:50 UTC
    same_game = lineup_player(nfl_player(4686000, "CAR", "WR"), 9.0)  # CAR hosts DET
    assert can_pivot(gibbs, FLEX, same_game)  # DET's inactives come out 90 minutes before the shared kickoff
    olave = lineup_player(OLAVE, 12.0)
    assert not can_pivot(gibbs, FLEX, olave, lead=timedelta(0))  # no notice: Gibbs is locked as the news lands
    smith_njigba = lineup_player(SMITH_NJIGBA, 10.0)  # locks at 20:25 UTC, 3 h 55 min before Gibbs' kickoff
    assert gibbs.availability.game_time is not None and smith_njigba.lock is not None
    notice = gibbs.availability.game_time - smith_njigba.lock
    assert not can_pivot(gibbs, FLEX, smith_njigba, lead=notice)  # he locks the moment the status lands
    assert can_pivot(gibbs, FLEX, smith_njigba, lead=notice + timedelta(minutes=1))


def test_no_pivot_once_everyone_locks_at_the_first_game() -> None:
    first_game = LockType.FIRSTGAME_SCORINGPERIOD  # all lock at Thursday night's kickoff
    gibbs = lineup_player(GIBBS_Q, 20.0, lock_type=first_game)
    olave = lineup_player(OLAVE, 12.0, lock_type=first_game)
    assert gibbs.lock == datetime(2026, 10, 2, 0, 15, tzinfo=UTC)
    assert not can_pivot(gibbs, FLEX, olave)
    assert plan_pivots([(FLEX, gibbs)], [olave]).gain == 0.0


def test_no_pivot_for_a_placeholder_start() -> None:
    raw = schedule_json(NFL_SCHEDULE)
    for team in raw["settings"]["proTeams"]:
        for game in team.get("proGamesByScoringPeriod", {}).get("4", []):
            if game["id"] == 401872978:  # DET at CAR, Sunday night
                game["startTimeTBD"] = True
                game["validForLocking"] = False
    flexed = ProSchedule.model_validate(raw)
    gibbs = lineup_player(GIBBS_Q, 20.0, schedule=flexed)
    olave = lineup_player(OLAVE, 12.0, schedule=flexed)
    assert gibbs.availability.inputs["provisional"] is True
    assert not can_pivot(gibbs, FLEX, olave)  # the placeholder would make his status look known too early


def test_certain_players_and_useless_pivots_are_left_out() -> None:
    olave = lineup_player(OLAVE, 12.0)
    healthy = lineup_player(pooled(GIBBS), 20.0)
    out = lineup_player(pooled(GIBBS, injury_status="OUT"), 20.0)
    gibbs = lineup_player(GIBBS_Q, 20.0)
    assert not can_pivot(healthy, FLEX, olave) and not can_pivot(out, FLEX, olave)
    hurt_pivot = lineup_player(nfl_player(4361370, "NO", "WR", "OUT"), 12.0)
    quarterback = lineup_player(pooled(MAHOMES), 25.0)  # healthy and projected, but a QB cannot take FLEX
    assert not can_pivot(gibbs, FLEX, hurt_pivot) and not can_pivot(gibbs, FLEX, quarterback)
    assert not can_pivot(gibbs, FLEX, gibbs)
    plan = plan_pivots([(FLEX, gibbs)], [gibbs, hurt_pivot])
    assert plan.pivots == {} and plan.expected == plan.base == pytest.approx(14.0)
    assert plan_pivots([], [olave]).expected == 0.0


def test_nba_util_pivots_need_a_later_tip() -> None:
    schedule = nba_real_schedule()  # day 2: tips from 7:00 p.m. ET; SAC at LAC at 10:30 p.m. ET
    luka = day2(nba_named(LUKA, "Luka Doncic", "LAL"))  # Doubtful on the official report

    def nba_lineup(row: AvailabilityRow, team: str, points: float) -> LineupPlayer:
        lock = NBA.lock_time(team_id(FBA, team), 2, schedule, lock_type=LockType.INDIVIDUAL_GAME)
        return LineupPlayer(row, points, frozenset({UTIL, FBA.bench_slot}), lock)

    def healthy(espn_id: int, team: str) -> AvailabilityRow:
        return assess(
            nba_named(espn_id, f"Player {espn_id}", team),
            season=NBA_SEASON,
            scoring_period=2,
            as_of=AS_OF,
            schedule=schedule,
        )

    starter = nba_lineup(luka, "LAL", 50.0)  # Doubtful (0.10), 10:00 p.m. tip: known at 9:30 p.m.
    late = nba_lineup(healthy(901, "LAC"), "LAC", 30.0)  # 10:30 p.m. tip
    early = nba_lineup(healthy(902, "ORL"), "ORL", 35.0)  # 7:00 p.m. tip: locked by then
    assert late_pivots(starter, UTIL, [early, late]) == [late]
    plan = plan_pivots([(UTIL, starter)], [early, late])
    assert plan.pivots == {LUKA: 901} and plan.gain == pytest.approx(0.90 * 30.0)
