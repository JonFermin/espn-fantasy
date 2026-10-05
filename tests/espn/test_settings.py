"""ESPN id maps and the ``mSettings`` parser on the NFL PPR, NBA points and NBA 9-cat fixtures.

The fixtures under ``tests/fixtures/espn/`` are hand-built stand-ins in the shape of ESPN's ``mSettings`` response
(no real league or manager names); ROADMAP #14 replaces them with scrubbed real-league captures.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from fm.espn.ids import FBA, FFL, Game, InjuryStatus, ids_for
from fm.espn.settings import (
    AcquisitionType,
    LeagueSettings,
    LockType,
    ScoringKind,
    ScoringType,
    SettingsParseError,
    SlotKind,
    load_league_settings,
    parse_league_settings,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "espn"
FFL_PPR = FIXTURES / "ffl_settings_ppr.json"
FBA_POINTS = FIXTURES / "fba_settings_points.json"
FBA_9CAT = FIXTURES / "fba_settings_9cat.json"


def _view(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def ffl_ppr() -> LeagueSettings:
    return load_league_settings(FFL_PPR)


@pytest.fixture
def fba_points() -> LeagueSettings:
    return load_league_settings(FBA_POINTS)


@pytest.fixture
def fba_9cat() -> LeagueSettings:
    return load_league_settings(FBA_9CAT)


# --- ids --------------------------------------------------------------------------------------------------------------


def test_game_keys_ids_and_sports() -> None:
    assert Game.FFL.game_id == 1 and Game.FBA.game_id == 3
    assert Game.from_game_id(3) is Game.FBA
    assert Game.from_sport("NBA") is Game.FBA
    assert Game.FFL.sport == "nfl"
    assert Game.coerce("ffl") is Game.FFL
    assert Game.coerce("nba") is Game.FBA
    assert Game.coerce(Game.FFL) is Game.FFL
    with pytest.raises(ValueError, match="gameId"):
        Game.from_game_id(2)  # baseball
    with pytest.raises(ValueError, match="sport"):
        Game.from_sport("mlb")


def test_ids_for_accepts_game_key_sport_or_enum() -> None:
    assert ids_for("ffl") is FFL
    assert ids_for("nba") is FBA
    assert ids_for(Game.FBA) is FBA


def test_ffl_maps() -> None:
    assert FFL.slot_label(23) == "RB/WR/TE"
    assert FFL.slot_id("FLEX") == 23 and FFL.slot_id("RB/WR/TE") == 23
    assert FFL.slot_id("D/ST") == 16 and FFL.slot_id("DST") == 16
    assert FFL.stat_abbr(53) == "REC" and FFL.stat_label(4) == "TD Pass"
    assert FFL.stat_id("PY") == 3
    assert FFL.position_label(16) == "D/ST" and FFL.position_id("QB") == 1
    assert FFL.pro_team(14) == "LAR" and FFL.pro_team(28) == "WSH" and FFL.pro_team(0) == "FA"
    assert (FFL.bench_slot, FFL.ir_slot) == (20, 21)
    assert FFL.is_active_slot(0) and not FFL.is_active_slot(20) and not FFL.is_active_slot(21)


def test_fba_maps() -> None:
    assert FBA.slot_label(11) == "UTIL"
    assert FBA.slot_id("UTIL") == 11 and FBA.slot_id("UT") == 11
    assert FBA.stat_abbr(19) == "FG%" and FBA.stat_id("3PM") == 17
    assert FBA.position_label(1) == "PG" and FBA.position_label(5) == "C"
    assert FBA.pro_team(21) == "PHX" and FBA.pro_team(9) == "GSW" and FBA.pro_team(3) == "NOP"
    assert (FBA.bench_slot, FBA.ir_slot) == (12, 13)


def test_unknown_ids_get_placeholders_and_reverse_lookups_raise() -> None:
    assert FFL.stat_abbr(999) == "STAT_999" and FFL.stat_label(999) == "Stat 999"
    assert FFL.slot_label(22) == "SLOT_22"  # unused by ESPN, deliberately unmapped
    assert FBA.position_label(9) == "POS_9"
    assert FBA.pro_team(99) == "TEAM_99"
    with pytest.raises(KeyError):
        FFL.stat_id("NOPE")
    with pytest.raises(KeyError):
        FBA.slot_id("FLEX")  # an NFL alias
    with pytest.raises(KeyError):
        FBA.position_id("K")


def test_injury_status_normalization() -> None:
    assert FFL.injury_status("QUESTIONABLE") is InjuryStatus.QUESTIONABLE
    assert FFL.injury_status("injury_reserve") is InjuryStatus.INJURY_RESERVE
    assert FBA.injury_status("DAY_TO_DAY") is InjuryStatus.DAY_TO_DAY
    assert FBA.injury_status(" OUT ") is InjuryStatus.OUT
    assert FFL.injury_status(None) is InjuryStatus.UNKNOWN
    assert FFL.injury_status("") is InjuryStatus.UNKNOWN
    assert FFL.injury_status("SOMETHING_NEW") is InjuryStatus.UNKNOWN


def test_id_tables_are_read_only() -> None:
    table: Any = FFL.lineup_slots
    with pytest.raises(TypeError):
        table[0] = "X"


# --- NFL PPR ----------------------------------------------------------------------------------------------------------


def test_ffl_identity_and_season_position(ffl_ppr: LeagueSettings) -> None:
    assert ffl_ppr.game is Game.FFL and ffl_ppr.ids is FFL
    assert (ffl_ppr.league_id, ffl_ppr.season, ffl_ppr.team_count) == (1234567, 2026, 10)
    assert ffl_ppr.name == "Fixture League (PPR)"
    assert (ffl_ppr.current_scoring_period, ffl_ppr.current_matchup_period) == (4, 4)
    assert (ffl_ppr.first_scoring_period, ffl_ppr.final_scoring_period) == (1, 18)


def test_ffl_ppr_scoring_items(ffl_ppr: LeagueSettings) -> None:
    assert ffl_ppr.scoring_type is ScoringType.H2H_POINTS
    assert ffl_ppr.scoring_kind is ScoringKind.POINTS and ffl_ppr.is_points and not ffl_ppr.is_categories
    assert ffl_ppr.categories == ()
    assert ffl_ppr.points_for("REC") == 1.0  # full PPR
    assert ffl_ppr.points_for(3) == 0.04 and ffl_ppr.points_for("RY") == 0.1
    assert ffl_ppr.points_for("PTD") == 4 and ffl_ppr.points_for("RTD") == 6
    assert ffl_ppr.points_for("INTT") == -2 and ffl_ppr.points_for("FUML") == -2
    assert ffl_ppr.points_for("TK") == 0.0 and ffl_ppr.scoring_item("TK") is None  # not scored
    rec = ffl_ppr.scoring_item(53)
    assert rec is not None and (rec.stat, rec.label, rec.is_reverse) == ("REC", "Each reception", False)
    assert len(ffl_ppr.scoring_items) == 45
    assert [item.stat_id for item in ffl_ppr.scoring_items] == sorted(item.stat_id for item in ffl_ppr.scoring_items)


def test_ffl_points_overrides_by_position(ffl_ppr: LeagueSettings) -> None:
    interception = ffl_ppr.scoring_item("INT")
    assert interception is not None and interception.points_overrides == {16: 2.0}
    assert ffl_ppr.points_for("INT") == 3  # individual defenders
    assert ffl_ppr.points_for("INT", position_id=FFL.position_id("D/ST")) == 2
    assert ffl_ppr.points_for("INT", position_id=FFL.position_id("LB")) == 3


def test_ffl_lineup_slots(ffl_ppr: LeagueSettings) -> None:
    assert ffl_ppr.slot_counts == {0: 1, 2: 2, 4: 2, 6: 1, 16: 1, 17: 1, 20: 7, 21: 1, 23: 1}
    assert ffl_ppr.slot_count("QB") == 1 and ffl_ppr.slot_count("RB") == 2 and ffl_ppr.slot_count("WR") == 2
    assert ffl_ppr.slot_count("FLEX") == 1 and ffl_ppr.slot_count(23) == 1
    assert ffl_ppr.slot_count("OP") == 0  # a known slot the league does not use
    assert tuple(slot.label for slot in ffl_ppr.active_slots) == ("QB", "RB", "WR", "TE", "D/ST", "K", "RB/WR/TE")
    assert (ffl_ppr.active_slot_count, ffl_ppr.bench_count, ffl_ppr.ir_count) == (9, 7, 1)
    assert ffl_ppr.roster_size == 16
    kinds = {slot.slot_id: slot.kind for slot in ffl_ppr.lineup_slots}
    assert kinds[20] is SlotKind.BENCH and kinds[21] is SlotKind.IR and kinds[0] is SlotKind.ACTIVE


def test_ffl_position_limits(ffl_ppr: LeagueSettings) -> None:
    assert ffl_ppr.position_limit(FFL.position_id("QB")) is None  # -1 means unlimited
    assert ffl_ppr.position_limit(0) == 0
    assert ffl_ppr.position_limit(99) is None
    assert ffl_ppr.slot_stat_limits == {} and ffl_ppr.games_played_limit(0) is None


def test_ffl_lock_type(ffl_ppr: LeagueSettings) -> None:
    assert ffl_ppr.lineup_lock_type is LockType.INDIVIDUAL_GAME
    assert ffl_ppr.lineup_lock_type_raw == "INDIVIDUAL_GAME"
    assert ffl_ppr.roster_lock_type_raw == "INDIVIDUAL_GAME"


def test_ffl_faab_and_waiver_timing(ffl_ppr: LeagueSettings) -> None:
    acq = ffl_ppr.acquisition
    assert acq.type is AcquisitionType.WAIVERS_TRADITIONAL and acq.type_raw == "WAIVERS_TRADITIONAL"
    assert acq.uses_faab and acq.budget == 100 and acq.minimum_bid == 0
    assert acq.season_limit is None and acq.matchup_limit is None
    assert not acq.matchup_limit_per_scoring_period
    assert acq.waiver_hours == 24
    assert acq.waiver_process_days == ("WEDNESDAY",) and acq.waiver_process_hour == 3
    assert acq.waiver_order_reset is True and acq.transaction_locking_enabled is False


def test_ffl_trade_deadline(ffl_ppr: LeagueSettings) -> None:
    deadline = ffl_ppr.trade.deadline
    assert deadline == datetime(2026, 11, 25, 17, 0, tzinfo=UTC)
    assert deadline is not None and deadline.tzinfo is not None
    assert ffl_ppr.trade.is_open(deadline - timedelta(hours=1))
    assert not ffl_ppr.trade.is_open(deadline)
    assert ffl_ppr.trade.max_trades is None
    assert (ffl_ppr.trade.revision_hours, ffl_ppr.trade.veto_votes_required) == (48, 4)


def test_ffl_schedule_and_playoff_weeks(ffl_ppr: LeagueSettings) -> None:
    schedule = ffl_ppr.schedule
    assert schedule.regular_season_matchups == 14 and schedule.playoff_team_count == 6
    assert schedule.playoff_matchup_period_length == 1 and not schedule.variable_playoff_matchup_period_length
    assert schedule.playoff_seeding_rule == "TOTAL_POINTS_SCORED"
    assert schedule.playoff_matchup_periods == (15, 16, 17) and ffl_ppr.playoff_matchup_periods == (15, 16, 17)
    assert schedule.playoff_scoring_periods == (15, 16, 17)
    assert schedule.scoring_periods(3) == (3,) and schedule.scoring_periods(99) == ()
    assert schedule.matchup_period_for(16) == 16 and schedule.matchup_period_for(99) is None
    assert schedule.is_playoff(15) and not schedule.is_playoff(14)


# --- NBA points -------------------------------------------------------------------------------------------------------


def test_fba_points_scoring(fba_points: LeagueSettings) -> None:
    assert fba_points.game is Game.FBA and fba_points.ids is FBA
    assert (fba_points.league_id, fba_points.season) == (2345678, 2027)
    assert fba_points.scoring_type is ScoringType.H2H_POINTS and fba_points.is_points
    assert {item.stat: item.points for item in fba_points.scoring_items} == {
        "PTS": 1,
        "BLK": 4,
        "STL": 4,
        "AST": 2,
        "REB": 1,
        "TO": -2,
        "FGM": 2,
        "FGA": -1,
        "FTM": 1,
        "FTA": -1,
        "3PM": 1,
    }
    assert not any(item.is_reverse for item in fba_points.scoring_items)
    assert fba_points.points_for("FGA") == -1 and fba_points.points_for("MIN") == 0.0


def test_fba_lineup_slots(fba_points: LeagueSettings) -> None:
    assert tuple(slot.label for slot in fba_points.active_slots) == ("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL")
    assert fba_points.slot_count("UTIL") == 3 and fba_points.slot_count("UT") == 3 and fba_points.slot_count(11) == 3
    assert (fba_points.active_slot_count, fba_points.bench_count, fba_points.ir_count) == (10, 3, 1)
    assert fba_points.roster_size == 13
    assert fba_points.position_limit(FBA.position_id("C")) is None
    assert fba_points.lineup_lock_type is LockType.INDIVIDUAL_GAME


def _fba_with_matchup_limit(value: Any, *, per_period: bool) -> LeagueSettings:
    raw = _view(FBA_POINTS)
    acq = raw["settings"]["acquisitionSettings"]
    acq["matchupAcquisitionLimit"] = value
    acq["matchupLimitPerScoringPeriod"] = per_period
    return parse_league_settings(raw)


def test_per_scoring_period_matchup_limit_keeps_the_fractional_rate() -> None:
    # A real NBA league's "3 adds per weekly matchup" arrives as 3/7 per day; truncating it to int gave 0.
    acq = _fba_with_matchup_limit(0.42857142857142855, per_period=True).acquisition
    assert acq.matchup_limit_per_scoring_period
    assert acq.matchup_limit is None
    assert acq.matchup_limit_rate == pytest.approx(3 / 7)
    assert acq.matchup_limit_for(7) == 3
    assert acq.matchup_limit_for(14) == 6
    assert acq.matchup_limit_for(13) == 5


@pytest.mark.parametrize("value", [-1, -1.0, None])
def test_per_scoring_period_matchup_limit_unlimited(value: Any) -> None:
    acq = _fba_with_matchup_limit(value, per_period=True).acquisition
    assert acq.matchup_limit_rate is None and acq.matchup_limit is None
    assert acq.matchup_limit_for(7) is None


def test_per_scoring_period_matchup_limit_rejects_non_numbers() -> None:
    with pytest.raises(SettingsParseError, match="matchupAcquisitionLimit"):
        _fba_with_matchup_limit("three", per_period=True)


def test_fixed_matchup_limit_ignores_matchup_length(fba_points: LeagueSettings) -> None:
    acq = fba_points.acquisition
    assert acq.matchup_limit_rate is None
    assert acq.matchup_limit_for(7) == acq.matchup_limit_for(14) == 4


def test_fba_acquisition_limits_without_faab(fba_points: LeagueSettings) -> None:
    acq = fba_points.acquisition
    assert acq.type is AcquisitionType.WAIVERS_TRADITIONAL
    assert not acq.uses_faab and acq.budget is None  # ESPN still sends acquisitionBudget; it is meaningless here
    assert acq.minimum_bid == 0
    assert acq.season_limit is None and acq.matchup_limit == 4
    assert acq.waiver_hours == 24 and len(acq.waiver_process_days) == 7 and acq.waiver_process_hour == 3


def test_fba_daily_scoring_periods_and_playoffs(fba_points: LeagueSettings) -> None:
    schedule = fba_points.schedule
    assert schedule.regular_season_matchups == 19
    assert schedule.scoring_periods(1) == (1, 2, 3, 4, 5, 6)  # opening Tuesday through Sunday
    assert schedule.scoring_periods(2) == (7, 8, 9, 10, 11, 12, 13)
    assert schedule.matchup_period_for(9) == 2
    assert schedule.matchup_period_for(fba_points.current_scoring_period or 0) == fba_points.current_matchup_period == 4
    assert schedule.playoff_matchup_periods == (20, 21, 22)
    assert len(schedule.playoff_scoring_periods) == 21 and schedule.playoff_scoring_periods[0] == 133
    assert schedule.playoff_matchup_period_length == 7
    assert fba_points.trade.deadline == datetime(2027, 2, 4, 20, 0, tzinfo=UTC)
    assert (fba_points.first_scoring_period, fba_points.final_scoring_period) == (1, 175)


# --- NBA 9-cat --------------------------------------------------------------------------------------------------------


def test_fba_9cat_categories(fba_9cat: LeagueSettings) -> None:
    assert fba_9cat.scoring_type is ScoringType.H2H_MOST_CATEGORIES
    assert fba_9cat.scoring_kind is ScoringKind.CATEGORIES and fba_9cat.is_categories and not fba_9cat.is_points
    assert fba_9cat.categories == ("PTS", "BLK", "STL", "AST", "REB", "TO", "3PM", "FG%", "FT%")
    turnovers = fba_9cat.scoring_item("TO")
    assert turnovers is not None and turnovers.is_reverse
    assert [item.stat for item in fba_9cat.scoring_items if item.is_reverse] == ["TO"]
    assert all(item.points == 0 for item in fba_9cat.scoring_items)
    assert fba_9cat.points_for("PTS") == 0.0


def test_fba_9cat_faab_season_limit_and_no_deadline(fba_9cat: LeagueSettings) -> None:
    acq = fba_9cat.acquisition
    assert acq.type is AcquisitionType.WAIVERS_CONTINUOUS
    assert acq.uses_faab and acq.budget == 200 and acq.minimum_bid == 1
    assert acq.season_limit == 60 and acq.matchup_limit is None
    assert acq.waiver_hours == 48 and acq.waiver_process_hour == 11 and acq.waiver_order_reset is False
    assert fba_9cat.trade.deadline is None
    assert fba_9cat.trade.is_open(datetime(2027, 4, 1, tzinfo=UTC))
    assert (fba_9cat.current_scoring_period, fba_9cat.current_matchup_period) == (1, 1)


def test_fba_9cat_games_played_limits(fba_9cat: LeagueSettings) -> None:
    assert fba_9cat.games_played_limit(0) == 82
    assert fba_9cat.games_played_limit(FBA.slot_id("UTIL")) == 246
    assert fba_9cat.games_played_limit(FBA.bench_slot) is None
    assert fba_9cat.slot_stat_limits[5] == {42: 82}


def test_slot_stat_limits_read_bare_integers_and_treat_negative_or_null_as_no_cap() -> None:
    view = _view(FBA_9CAT)
    limits = view["settings"]["rosterSettings"]["lineupSlotStatLimits"]
    limits["0"]["42"] = -1  # ESPN's "unlimited"
    limits["1"]["42"] = None
    limits["2"] = {}
    settings = parse_league_settings(view)
    assert settings.games_played_limit(0) is None and settings.games_played_limit(1) is None
    assert settings.games_played_limit(2) is None and settings.games_played_limit(3) == 82
    assert set(settings.slot_stat_limits) == {3, 4, 5, 6, 11}


@pytest.mark.parametrize(
    ("per_stat", "where"),
    [
        ({"42": {"limit": 246}}, r"lineupSlotStatLimits\[11\]\[42\] should be an integer cap .* got \{'limit': 246\}"),
        ({"42": "two hundred"}, r"lineupSlotStatLimits\[11\]\[42\] should be an integer cap"),
        ({"42": True}, r"lineupSlotStatLimits\[11\]\[42\] should be an integer cap"),
        ({"GP": 246}, r"lineupSlotStatLimits\[11\] has a non-numeric stat id 'GP'"),
        (246, r"lineupSlotStatLimits\[11\] should be an object keyed by stat id"),
        ([{"42": 246}], r"lineupSlotStatLimits\[11\] should be an object keyed by stat id"),
    ],
)
def test_slot_stat_limits_of_an_unknown_shape_raise_instead_of_dropping_the_cap(per_stat: Any, where: str) -> None:
    view = _view(FBA_9CAT)
    view["settings"]["rosterSettings"]["lineupSlotStatLimits"]["11"] = per_stat
    with pytest.raises(SettingsParseError, match=where):
        parse_league_settings(view)


def test_slot_stat_limits_container_must_be_keyed_by_slot_id() -> None:
    view = _view(FBA_9CAT)
    roster = view["settings"]["rosterSettings"]
    roster["lineupSlotStatLimits"] = [{"42": 82}]
    with pytest.raises(SettingsParseError, match=r"lineupSlotStatLimits should be an object keyed by slot id"):
        parse_league_settings(view)
    roster["lineupSlotStatLimits"] = {"UTIL": {"42": 246}}
    with pytest.raises(SettingsParseError, match=r"non-numeric slot id 'UTIL'"):
        parse_league_settings(view)
    roster["lineupSlotStatLimits"] = None
    assert parse_league_settings(view).slot_stat_limits == {}
    del roster["lineupSlotStatLimits"]
    assert parse_league_settings(view).slot_stat_limits == {}


# --- variants and errors ----------------------------------------------------------------------------------------------


@pytest.mark.parametrize("scoring_type", ["H2H_CATEGORY", "ROTO"])
def test_each_category_and_roto_are_category_leagues(scoring_type: str) -> None:
    view = _view(FBA_9CAT)
    view["settings"]["scoringSettings"]["scoringType"] = scoring_type
    settings = parse_league_settings(view)
    assert settings.scoring_type is ScoringType(scoring_type)
    assert settings.scoring_kind is ScoringKind.CATEGORIES


def test_total_season_points_is_a_points_league() -> None:
    view = _view(FBA_POINTS)
    view["settings"]["scoringSettings"]["scoringType"] = "TOTAL_SEASON_POINTS"
    assert parse_league_settings(view).scoring_kind is ScoringKind.POINTS


def test_unknown_scoring_type_keeps_raw_and_infers_kind_from_items() -> None:
    points_view = _view(FBA_POINTS)
    points_view["settings"]["scoringSettings"]["scoringType"] = "H2H_SOMETHING_NEW"
    points = parse_league_settings(points_view)
    assert points.scoring_type is ScoringType.UNKNOWN and points.scoring_type_raw == "H2H_SOMETHING_NEW"
    assert points.scoring_kind is ScoringKind.POINTS

    cat_view = _view(FBA_9CAT)
    del cat_view["settings"]["scoringSettings"]["scoringType"]
    categories = parse_league_settings(cat_view)
    assert categories.scoring_type is ScoringType.UNKNOWN and categories.scoring_type_raw is None
    assert categories.scoring_kind is ScoringKind.CATEGORIES


def test_lock_type_variants_keep_raw_value() -> None:
    view = _view(FFL_PPR)
    roster = view["settings"]["rosterSettings"]

    roster["lineupLocktimeType"] = "FIRST_GAME_OF_WEEK"
    assert parse_league_settings(view).lineup_lock_type is LockType.FIRST_GAME_OF_WEEK

    roster["lineupLocktimeType"] = "FIRSTGAME_SCORINGPERIOD"  # the value ESPN's typeNames.locktimeTypes lists
    assert parse_league_settings(view).lineup_lock_type is LockType.FIRSTGAME_SCORINGPERIOD

    roster["lineupLocktimeType"] = "FIRST_GAME_OF_DAY"
    unknown = parse_league_settings(view)
    assert unknown.lineup_lock_type is LockType.UNKNOWN and unknown.lineup_lock_type_raw == "FIRST_GAME_OF_DAY"

    del roster["lineupLocktimeType"]
    del roster["rosterLocktimeType"]
    missing = parse_league_settings(view)
    assert missing.lineup_lock_type is LockType.UNKNOWN and missing.lineup_lock_type_raw is None
    assert missing.roster_lock_type_raw is None


def test_unknown_acquisition_type_keeps_raw_value() -> None:
    view = _view(FFL_PPR)
    view["settings"]["acquisitionSettings"]["acquisitionType"] = "SOMETHING_NEW"
    acq = parse_league_settings(view).acquisition
    assert acq.type is AcquisitionType.UNKNOWN and acq.type_raw == "SOMETHING_NEW"


def test_game_resolution_from_payload_and_argument() -> None:
    view = _view(FFL_PPR)
    assert parse_league_settings(view, game="nfl").game is Game.FFL
    assert parse_league_settings(view, game=Game.FFL).game is Game.FFL
    with pytest.raises(SettingsParseError, match="ffl"):
        parse_league_settings(view, game="fba")

    del view["gameId"]
    with pytest.raises(SettingsParseError, match="gameId"):
        parse_league_settings(view)
    assert parse_league_settings(view, game="ffl").game is Game.FFL

    view["gameId"] = 2  # baseball
    with pytest.raises(SettingsParseError, match="gameId"):
        parse_league_settings(view)
    assert parse_league_settings(view, game="ffl").game is Game.FFL


def test_missing_required_blocks_raise() -> None:
    view = _view(FFL_PPR)
    del view["settings"]["scoringSettings"]
    with pytest.raises(SettingsParseError, match="scoringSettings"):
        parse_league_settings(view)

    view = _view(FFL_PPR)
    del view["settings"]["rosterSettings"]
    with pytest.raises(SettingsParseError, match="rosterSettings"):
        parse_league_settings(view)

    view = _view(FFL_PPR)
    del view["seasonId"]
    with pytest.raises(SettingsParseError, match="season"):
        parse_league_settings(view)

    with pytest.raises(SettingsParseError, match="settings"):
        parse_league_settings({"gameId": 1, "id": 1, "seasonId": 2026})

    view = _view(FFL_PPR)
    view["settings"]["scoringSettings"]["scoringItems"].append({"points": 1.0})
    with pytest.raises(SettingsParseError, match="scoring item"):
        parse_league_settings(view)


def test_negative_limits_mean_unlimited_and_zero_stays_zero() -> None:
    view = _view(FFL_PPR)
    acquisition = view["settings"]["acquisitionSettings"]
    acquisition["acquisitionLimit"] = 0
    acquisition["matchupAcquisitionLimit"] = -1
    view["settings"]["tradeSettings"]["max"] = 2
    view["settings"]["tradeSettings"]["deadlineDate"] = 0
    settings = parse_league_settings(view)
    assert settings.acquisition.season_limit == 0
    assert settings.acquisition.matchup_limit is None
    assert settings.trade.max_trades == 2
    assert settings.trade.deadline is None


def test_slot_counts_keep_only_used_slots_and_ignore_junk_keys() -> None:
    view = _view(FFL_PPR)
    counts = view["settings"]["rosterSettings"]["lineupSlotCounts"]
    counts["7"] = 1  # add an OP slot
    counts["junk"] = 1
    settings = parse_league_settings(view)
    assert settings.slot_count("OP") == 1
    assert all(slot.count > 0 for slot in settings.lineup_slots)
    assert settings.active_slot_count == 10 and settings.roster_size == 17


def test_unknown_stat_ids_get_placeholder_labels() -> None:
    view = _view(FBA_POINTS)
    view["settings"]["scoringSettings"]["scoringItems"].append({"statId": 999, "points": 1.5})
    item = parse_league_settings(view).scoring_item(999)
    assert item is not None and (item.stat, item.label, item.points) == ("STAT_999", "Stat 999", 1.5)


def test_load_matches_parse_and_models_round_trip(ffl_ppr: LeagueSettings) -> None:
    assert ffl_ppr == parse_league_settings(_view(FFL_PPR))
    restored = LeagueSettings.model_validate_json(ffl_ppr.model_dump_json())
    assert restored == ffl_ppr
    assert restored.trade.deadline == ffl_ppr.trade.deadline and restored.slot_counts == ffl_ppr.slot_counts


def test_settings_are_immutable(ffl_ppr: LeagueSettings) -> None:
    with pytest.raises(ValidationError):
        ffl_ppr.name = "renamed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ffl_ppr.acquisition.budget = 1  # type: ignore[misc]


def test_fixtures_are_scrubbed() -> None:
    for path in (FFL_PPR, FBA_POINTS, FBA_9CAT):
        text = path.read_text(encoding="utf-8")
        assert _view(path)["members"] == []
        assert "espn_s2" not in text.lower() and "swid" not in text.lower()
