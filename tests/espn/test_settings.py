"""ESPN id maps and the ``mSettings`` parser on the two real leagues and the NBA 9-cat stand-in.

The NFL and NBA points sections read the scrubbed ``mSettings`` captures of the configured leagues
(``tests/fixtures/espn/real/{ffl,fba}/mSettings.json``, ROADMAP #14; docs/espn-api.md section 3 lists what they hold).
Neither real league uses FAAB, categories or games-played caps, so those cases keep the hand-built stand-ins under
``tests/fixtures/espn/``: ``fba_settings_9cat.json`` (categories, FAAB with a minimum bid, a season acquisition limit,
per-slot games-played caps, no trade deadline) and ``ffl_settings_ppr.json`` for FAAB without a minimum bid. The
variant and error tests mutate copies of the real captures.
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
    SCORING_PERIOD_TYPE,
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
REAL_FFL = FIXTURES / "real" / "ffl" / "mSettings.json"
REAL_FBA = FIXTURES / "real" / "fba" / "mSettings.json"
FFL_PPR = FIXTURES / "ffl_settings_ppr.json"
"""Hand-built stand-in: the FAAB case without a minimum bid (neither real league uses FAAB)."""
FBA_9CAT = FIXTURES / "fba_settings_9cat.json"
"""Hand-built stand-in: categories, FAAB, a season limit and games-played caps (no real league has them)."""
DST = 16
"""The D/ST position id in ``ffl``, the only position the real league overrides points for."""


def _view(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture
def ffl() -> LeagueSettings:
    return load_league_settings(REAL_FFL)


@pytest.fixture
def fba() -> LeagueSettings:
    return load_league_settings(REAL_FBA)


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


# --- the real NFL league (ffl/mSettings.json: 2026, week 4) -----------------------------------------------------------


def test_ffl_identity_and_season_position(ffl: LeagueSettings) -> None:
    assert ffl.game is Game.FFL and ffl.ids is FFL
    assert (ffl.league_id, ffl.season, ffl.team_count) == (1010101, 2026, 10)  # the scrubber's league id
    assert ffl.name == "Fixture ffl League"
    assert (ffl.current_scoring_period, ffl.current_matchup_period) == (4, 4)
    assert (ffl.first_scoring_period, ffl.final_scoring_period) == (1, 17)


def test_ffl_scoring_items(ffl: LeagueSettings) -> None:
    """Full PPR with ESPN's default yardage and touchdown values, 46 items (docs/espn-api.md section 3)."""
    assert ffl.scoring_type is ScoringType.H2H_POINTS
    assert ffl.scoring_kind is ScoringKind.POINTS and ffl.is_points and not ffl.is_categories
    assert ffl.categories == ()
    assert ffl.points_for("REC") == 1.0  # full PPR
    assert ffl.points_for(3) == 0.04 and ffl.points_for("RY") == 0.1 and ffl.points_for("REY") == 0.1
    assert ffl.points_for("PTD") == 4 and ffl.points_for("RTD") == 6 and ffl.points_for("RETD") == 6
    assert ffl.points_for("INTT") == -2 and ffl.points_for("FUML") == -2
    assert ffl.points_for("FG0") == 3 and ffl.points_for("FG40") == 4 and ffl.points_for("FG50") == 5
    assert ffl.points_for("FGM") == -1 and ffl.points_for("PAT") == 1
    assert ffl.points_for("TK") == 0.0 and ffl.scoring_item("TK") is None  # not scored
    rec = ffl.scoring_item(53)
    assert rec is not None and (rec.stat, rec.label, rec.is_reverse) == ("REC", "Each reception", False)
    assert len(ffl.scoring_items) == 46
    assert [item.stat_id for item in ffl.scoring_items] == sorted(item.stat_id for item in ffl.scoring_items)


def test_ffl_points_overrides_by_position(ffl: LeagueSettings) -> None:
    """The D/ST tiers (points allowed, yards allowed, sacks, takeaways) score only through a position override."""
    shutout = ffl.scoring_item("PA0")
    assert shutout is not None and shutout.points == 0.0 and shutout.points_overrides == {DST: 5.0}
    assert ffl.points_for("PA0") == 0 and ffl.points_for("PA0", position_id=DST) == 5
    assert ffl.points_for("INT", position_id=DST) == 2 and ffl.points_for("SK", position_id=DST) == 1
    assert ffl.points_for("PA46", position_id=DST) == -5 and ffl.points_for("YA550", position_id=DST) == -7
    assert ffl.points_for("INT", position_id=FFL.position_id("LB")) == 0  # no IDP scoring
    overridden = {item.stat for item in ffl.scoring_items if item.points_overrides}
    assert len(overridden) == 27 and all(
        set(item.points_overrides) == {DST} for item in ffl.scoring_items if item.points_overrides
    )


def test_ffl_lineup_slots(ffl: LeagueSettings) -> None:
    assert ffl.slot_counts == {0: 1, 2: 2, 4: 2, 6: 1, 16: 1, 17: 1, 20: 7, 21: 1, 23: 1}
    assert ffl.slot_count("QB") == 1 and ffl.slot_count("RB") == 2 and ffl.slot_count("WR") == 2
    assert ffl.slot_count("FLEX") == 1 and ffl.slot_count(23) == 1
    assert ffl.slot_count("OP") == 0  # a known slot the league does not use
    assert tuple(slot.label for slot in ffl.active_slots) == ("QB", "RB", "WR", "TE", "D/ST", "K", "RB/WR/TE")
    assert (ffl.active_slot_count, ffl.bench_count, ffl.ir_count) == (9, 7, 1)
    assert ffl.roster_size == 16
    kinds = {slot.slot_id: slot.kind for slot in ffl.lineup_slots}
    assert kinds[20] is SlotKind.BENCH and kinds[21] is SlotKind.IR and kinds[0] is SlotKind.ACTIVE


def test_ffl_position_limits(ffl: LeagueSettings) -> None:
    limits = {label: ffl.position_limit(FFL.position_id(label)) for label in ("QB", "RB", "WR", "TE", "K", "D/ST")}
    assert limits == {"QB": 4, "RB": 8, "WR": 8, "TE": 3, "K": 3, "D/ST": 3}
    assert ffl.position_limit(FFL.position_id("LB")) is None  # -1 means unlimited
    assert ffl.position_limit(0) == 0
    assert ffl.position_limit(99) is None
    assert ffl.slot_stat_limits == {} and ffl.games_played_limit(0) is None


def test_ffl_lock_types(ffl: LeagueSettings) -> None:
    assert ffl.lineup_lock_type is LockType.INDIVIDUAL_GAME
    assert ffl.lineup_lock_type_raw == "INDIVIDUAL_GAME"
    assert ffl.roster_lock_type is LockType.INDIVIDUAL_GAME
    assert ffl.roster_lock_type_raw == "INDIVIDUAL_GAME"


def test_ffl_waivers_without_faab(ffl: LeagueSettings) -> None:
    """Traditional waivers: ESPN still sends ``acquisitionBudget: 100`` and ``minimumBid: 1``, meaningless here."""
    acq = ffl.acquisition
    assert acq.type is AcquisitionType.WAIVERS_TRADITIONAL and acq.type_raw == "WAIVERS_TRADITIONAL"
    assert not acq.uses_faab and acq.budget is None and acq.minimum_bid == 1
    assert acq.season_limit is None and acq.matchup_limit is None and acq.matchup_limit_rate is None
    assert acq.matchup_limit_per_scoring_period  # true even with no limit set (-1)
    assert acq.waiver_hours == 24
    assert acq.waiver_process_days == ("MONDAY", "WEDNESDAY", "THURSDAY", "FRIDAY", "SATURDAY", "SUNDAY")
    assert acq.waiver_process_hour == 11  # not an ET hour: runs on record were at about 03:00 ET (docs section 1 #8)
    assert acq.waiver_order_reset is True and acq.transaction_locking_enabled is False


def test_ffl_trade_deadline(ffl: LeagueSettings) -> None:
    deadline = ffl.trade.deadline
    assert deadline == datetime(2026, 12, 2, 17, 0, tzinfo=UTC)  # Wed Dec 2 2026, noon ET
    assert deadline is not None and deadline.tzinfo is not None
    assert ffl.trade.is_open(deadline - timedelta(hours=1))
    assert not ffl.trade.is_open(deadline)
    assert ffl.trade.max_trades is None
    assert (ffl.trade.revision_hours, ffl.trade.veto_votes_required) == (24, 4)


def test_ffl_schedule_and_playoff_weeks(ffl: LeagueSettings) -> None:
    """13 one-week matchups, then a 4-team playoff of two two-week rounds (matchup 14 = weeks 14-15, 15 = 16-17)."""
    schedule = ffl.schedule
    assert schedule.period_type_id == SCORING_PERIOD_TYPE and schedule.lists_scoring_periods
    assert schedule.regular_season_matchups == 13 and schedule.playoff_team_count == 4
    assert schedule.playoff_matchup_period_length == 2 and not schedule.variable_playoff_matchup_period_length
    assert schedule.playoff_seeding_rule == "TOTAL_POINTS_SCORED"
    assert schedule.playoff_matchup_periods == (14, 15) and ffl.playoff_matchup_periods == (14, 15)
    assert schedule.playoff_scoring_periods == (14, 15, 16, 17)
    assert schedule.scoring_periods(3) == (3,) and schedule.scoring_periods(14) == (14, 15)
    assert schedule.scoring_periods(99) == ()
    assert schedule.matchup_period_for(4) == 4 and schedule.matchup_period_for(16) == 15
    assert schedule.matchup_period_for(99) is None
    assert schedule.is_playoff(14) and not schedule.is_playoff(13)


# --- the real NBA league (fba/mSettings.json: 2027, preseason day 1) --------------------------------------------------


def test_fba_identity_and_points_scoring(fba: LeagueSettings) -> None:
    """ESPN's default NBA points scoring, 11 items."""
    assert fba.game is Game.FBA and fba.ids is FBA
    assert (fba.league_id, fba.season, fba.team_count) == (2020202, 2027, 10)
    assert fba.name == "Fixture fba League"
    assert (fba.current_scoring_period, fba.current_matchup_period) == (1, 1)
    assert (fba.first_scoring_period, fba.final_scoring_period) == (1, 153)
    assert fba.scoring_type is ScoringType.H2H_POINTS and fba.is_points
    assert {item.stat: item.points for item in fba.scoring_items} == {
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
    assert not any(item.is_reverse or item.points_overrides for item in fba.scoring_items)
    assert fba.points_for("FGA") == -1 and fba.points_for("MIN") == 0.0


def test_fba_lineup_slots(fba: LeagueSettings) -> None:
    assert tuple(slot.label for slot in fba.active_slots) == ("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL")
    assert fba.slot_count("UTIL") == 3 and fba.slot_count("UT") == 3 and fba.slot_count(11) == 3
    assert (fba.active_slot_count, fba.bench_count, fba.ir_count) == (10, 3, 3)
    assert fba.roster_size == 13
    assert fba.position_limit(FBA.position_id("C")) == 4  # the only position the league caps
    assert fba.position_limit(FBA.position_id("PG")) is None
    assert fba.slot_stat_limits == {} and fba.games_played_limit(FBA.slot_id("UTIL")) is None  # no games-played caps


def test_fba_lock_types(fba: LeagueSettings) -> None:
    """Lineups lock per game; adds, drops and trades lock at the day's first tip (docs/espn-api.md section 1 #4)."""
    assert fba.lineup_lock_type is LockType.INDIVIDUAL_GAME
    assert fba.roster_lock_type is LockType.FIRSTGAME_SCORINGPERIOD
    assert fba.roster_lock_type_raw == "FIRSTGAME_SCORINGPERIOD"


def test_fba_acquisition_limit_is_a_daily_rate(fba: LeagueSettings) -> None:
    """ "3 adds per weekly matchup" arrives as 3/7 per day with ``matchupLimitPerScoringPeriod``; truncating it to an
    int gave 0 once."""
    acq = fba.acquisition
    assert acq.type is AcquisitionType.WAIVERS_TRADITIONAL
    assert not acq.uses_faab and acq.budget is None and acq.minimum_bid == 1
    assert acq.season_limit is None
    assert acq.matchup_limit_per_scoring_period and acq.matchup_limit is None
    assert acq.matchup_limit_rate == pytest.approx(3 / 7)
    assert acq.matchup_limit_for(7) == 3 and acq.matchup_limit_for(14) == 6 and acq.matchup_limit_for(13) == 5
    assert acq.matchup_limit_for(6) == 2  # how ESPN rounds the 6-day opening matchup is not yet visible
    assert acq.waiver_hours == 24 and acq.waiver_process_days == ("SUNDAY",) and acq.waiver_process_hour == 8
    assert acq.waiver_order_reset is False and acq.transaction_locking_enabled is False


def test_fba_trade_deadline(fba: LeagueSettings) -> None:
    assert fba.trade.deadline == datetime(2027, 2, 26, 17, 0, tzinfo=UTC)  # Fri Feb 26 2027, noon ET
    assert fba.trade.max_trades is None
    assert (fba.trade.revision_hours, fba.trade.veto_votes_required) == (24, 3)


def test_fba_matchups_list_weeks_not_days(fba: LeagueSettings) -> None:
    """The real NBA league lists its matchups as weeks (``periodTypeId`` 2; docs/espn-api.md section 1 #1): matchup 1 is
    week 1, days 1-6, and the playoffs are days 133-153. Read as days, matchup N would be the single day N, so the
    schedule answers ``None`` instead until ESPN's web-client calendar maps weeks to days (ROADMAP #31)."""
    schedule = fba.schedule
    assert schedule.period_type_id == 2 and not schedule.lists_scoring_periods
    assert schedule.regular_season_matchups == 18 and schedule.playoff_team_count == 6
    assert schedule.playoff_matchup_period_length == 1 and schedule.playoff_seeding_rule == "H2H_RECORD"
    assert schedule.matchup_periods[1] == (1,) and schedule.matchup_periods[21] == (21,)  # week ids, as ESPN sends them
    assert schedule.matchup_period_for(5) is None  # day 5 is in matchup 1, not "matchup 5"
    assert schedule.scoring_periods(1) is None and schedule.scoring_periods(5) is None
    assert schedule.playoff_scoring_periods is None  # not (19, 20, 21)
    assert schedule.playoff_matchup_periods == fba.playoff_matchup_periods == (19, 20, 21)  # matchup ids still hold


def _fba_with_matchup_limit(value: Any, *, per_period: bool) -> LeagueSettings:
    raw = _view(REAL_FBA)
    acq = raw["settings"]["acquisitionSettings"]
    acq["matchupAcquisitionLimit"] = value
    acq["matchupLimitPerScoringPeriod"] = per_period
    return parse_league_settings(raw)


@pytest.mark.parametrize("value", [-1, -1.0, None])
def test_per_scoring_period_matchup_limit_unlimited(value: Any) -> None:
    acq = _fba_with_matchup_limit(value, per_period=True).acquisition
    assert acq.matchup_limit_rate is None and acq.matchup_limit is None
    assert acq.matchup_limit_for(7) is None


def test_per_scoring_period_matchup_limit_rejects_non_numbers() -> None:
    with pytest.raises(SettingsParseError, match="matchupAcquisitionLimit"):
        _fba_with_matchup_limit("three", per_period=True)


def test_fixed_matchup_limit_ignores_matchup_length() -> None:
    """A per-matchup limit (no real league has one): the same count whatever the matchup's length."""
    acq = _fba_with_matchup_limit(4, per_period=False).acquisition
    assert not acq.matchup_limit_per_scoring_period and acq.matchup_limit == 4
    assert acq.matchup_limit_rate is None
    assert acq.matchup_limit_for(7) == acq.matchup_limit_for(14) == 4


def test_matchups_of_an_unknown_period_type_are_not_read_as_scoring_periods(ffl: LeagueSettings) -> None:
    view = _view(REAL_FFL)
    del view["settings"]["scheduleSettings"]["periodTypeId"]
    missing = parse_league_settings(view).schedule
    assert missing.period_type_id is None and not missing.lists_scoring_periods
    assert missing.matchup_period_for(4) is None and missing.scoring_periods(4) is None
    # Settings stored before periodTypeId was read validate too, as unknown, until the next sync re-parses them.
    stored = ffl.model_dump(mode="json")
    del stored["schedule"]["period_type_id"]
    del stored["roster_lock_type"]
    restored = LeagueSettings.model_validate(stored)
    assert restored.schedule.period_type_id is None and restored.roster_lock_type is LockType.UNKNOWN


# --- FAAB (stand-ins: neither real league uses it) ------------------------------------------------------------------


def test_faab_budget_and_minimum_bid_from_the_stand_ins(fba_9cat: LeagueSettings) -> None:
    ppr = load_league_settings(FFL_PPR).acquisition
    assert ppr.type is AcquisitionType.WAIVERS_TRADITIONAL
    assert ppr.uses_faab and ppr.budget == 100 and ppr.minimum_bid == 0
    assert ppr.waiver_process_days == ("WEDNESDAY",) and ppr.waiver_process_hour == 3
    acq = fba_9cat.acquisition
    assert acq.type is AcquisitionType.WAIVERS_CONTINUOUS
    assert acq.uses_faab and acq.budget == 200 and acq.minimum_bid == 1
    assert acq.season_limit == 60 and acq.matchup_limit is None
    assert acq.waiver_hours == 48 and acq.waiver_process_hour == 11 and acq.waiver_order_reset is False


# --- NBA 9-cat (stand-in: no real league plays categories or caps games played) --------------------------------------


def test_fba_9cat_categories(fba_9cat: LeagueSettings) -> None:
    assert fba_9cat.scoring_type is ScoringType.H2H_MOST_CATEGORIES
    assert fba_9cat.scoring_kind is ScoringKind.CATEGORIES and fba_9cat.is_categories and not fba_9cat.is_points
    assert fba_9cat.categories == ("PTS", "BLK", "STL", "AST", "REB", "TO", "3PM", "FG%", "FT%")
    turnovers = fba_9cat.scoring_item("TO")
    assert turnovers is not None and turnovers.is_reverse
    assert [item.stat for item in fba_9cat.scoring_items if item.is_reverse] == ["TO"]
    assert all(item.points == 0 for item in fba_9cat.scoring_items)
    assert fba_9cat.points_for("PTS") == 0.0


def test_fba_9cat_no_deadline(fba_9cat: LeagueSettings) -> None:
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


# --- variants and errors (mutated copies of the real captures) --------------------------------------------------------


@pytest.mark.parametrize("scoring_type", ["H2H_CATEGORY", "ROTO"])
def test_each_category_and_roto_are_category_leagues(scoring_type: str) -> None:
    view = _view(FBA_9CAT)
    view["settings"]["scoringSettings"]["scoringType"] = scoring_type
    settings = parse_league_settings(view)
    assert settings.scoring_type is ScoringType(scoring_type)
    assert settings.scoring_kind is ScoringKind.CATEGORIES


def test_total_season_points_is_a_points_league() -> None:
    view = _view(REAL_FBA)
    view["settings"]["scoringSettings"]["scoringType"] = "TOTAL_SEASON_POINTS"
    assert parse_league_settings(view).scoring_kind is ScoringKind.POINTS


def test_unknown_scoring_type_keeps_raw_and_infers_kind_from_items() -> None:
    points_view = _view(REAL_FBA)
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
    view = _view(REAL_FFL)
    roster = view["settings"]["rosterSettings"]

    roster["lineupLocktimeType"] = "FIRSTGAME_SCORINGPERIOD"  # the value ESPN's typeNames.locktimeTypes lists
    assert parse_league_settings(view).lineup_lock_type is LockType.FIRSTGAME_SCORINGPERIOD

    # ESPN's two weekly values, and the earlier guess no league carries: all UNKNOWN, which no plugin computes.
    for weekly in ("FIRSTGAME_WEEKLY", "INDIVIDUAL_FIRSTGAME_WEEKLY", "FIRST_GAME_OF_WEEK"):
        roster["lineupLocktimeType"] = weekly
        parsed = parse_league_settings(view)
        assert parsed.lineup_lock_type is LockType.UNKNOWN and parsed.lineup_lock_type_raw == weekly

    roster["lineupLocktimeType"] = "FIRST_GAME_OF_DAY"
    unknown = parse_league_settings(view)
    assert unknown.lineup_lock_type is LockType.UNKNOWN and unknown.lineup_lock_type_raw == "FIRST_GAME_OF_DAY"

    # rosterLocktimeType (adds, drops and trades) parses the same way, independently of the lineup lock.
    roster["rosterLocktimeType"] = "FIRSTGAME_SCORINGPERIOD"  # the real NBA league's value
    first_game = parse_league_settings(view)
    assert first_game.roster_lock_type is LockType.FIRSTGAME_SCORINGPERIOD
    assert first_game.roster_lock_type_raw == "FIRSTGAME_SCORINGPERIOD"
    assert first_game.lineup_lock_type is LockType.UNKNOWN
    for weekly in ("FIRSTGAME_WEEKLY", "INDIVIDUAL_FIRSTGAME_WEEKLY"):
        roster["rosterLocktimeType"] = weekly
        parsed = parse_league_settings(view)
        assert parsed.roster_lock_type is LockType.UNKNOWN and parsed.roster_lock_type_raw == weekly

    del roster["lineupLocktimeType"]
    del roster["rosterLocktimeType"]
    missing = parse_league_settings(view)
    assert missing.lineup_lock_type is LockType.UNKNOWN and missing.lineup_lock_type_raw is None
    assert missing.roster_lock_type is LockType.UNKNOWN and missing.roster_lock_type_raw is None


def test_unknown_acquisition_type_keeps_raw_value() -> None:
    view = _view(REAL_FFL)
    view["settings"]["acquisitionSettings"]["acquisitionType"] = "SOMETHING_NEW"
    acq = parse_league_settings(view).acquisition
    assert acq.type is AcquisitionType.UNKNOWN and acq.type_raw == "SOMETHING_NEW"


def test_game_resolution_from_payload_and_argument() -> None:
    view = _view(REAL_FFL)
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
    view = _view(REAL_FFL)
    del view["settings"]["scoringSettings"]
    with pytest.raises(SettingsParseError, match="scoringSettings"):
        parse_league_settings(view)

    view = _view(REAL_FFL)
    del view["settings"]["rosterSettings"]
    with pytest.raises(SettingsParseError, match="rosterSettings"):
        parse_league_settings(view)

    view = _view(REAL_FFL)
    del view["seasonId"]
    with pytest.raises(SettingsParseError, match="season"):
        parse_league_settings(view)

    with pytest.raises(SettingsParseError, match="settings"):
        parse_league_settings({"gameId": 1, "id": 1, "seasonId": 2026})

    view = _view(REAL_FFL)
    view["settings"]["scoringSettings"]["scoringItems"].append({"points": 1.0})
    with pytest.raises(SettingsParseError, match="scoring item"):
        parse_league_settings(view)


def test_negative_limits_mean_unlimited_and_zero_stays_zero() -> None:
    view = _view(REAL_FFL)
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
    view = _view(REAL_FFL)
    counts = view["settings"]["rosterSettings"]["lineupSlotCounts"]
    counts["7"] = 1  # add an OP slot
    counts["junk"] = 1
    settings = parse_league_settings(view)
    assert settings.slot_count("OP") == 1
    assert all(slot.count > 0 for slot in settings.lineup_slots)
    assert settings.active_slot_count == 10 and settings.roster_size == 17


def test_unknown_stat_ids_get_placeholder_labels() -> None:
    view = _view(REAL_FBA)
    view["settings"]["scoringSettings"]["scoringItems"].append({"statId": 999, "points": 1.5})
    item = parse_league_settings(view).scoring_item(999)
    assert item is not None and (item.stat, item.label, item.points) == ("STAT_999", "Stat 999", 1.5)


def test_load_matches_parse_and_models_round_trip(ffl: LeagueSettings) -> None:
    assert ffl == parse_league_settings(_view(REAL_FFL))
    restored = LeagueSettings.model_validate_json(ffl.model_dump_json())
    assert restored == ffl
    assert restored.trade.deadline == ffl.trade.deadline and restored.slot_counts == ffl.slot_counts


def test_settings_are_immutable(ffl: LeagueSettings) -> None:
    with pytest.raises(ValidationError):
        ffl.name = "renamed"  # type: ignore[misc]
    with pytest.raises(ValidationError):
        ffl.acquisition.budget = 1  # type: ignore[misc]


def test_stand_ins_are_scrubbed() -> None:
    for path in (FFL_PPR, FBA_9CAT, FIXTURES / "fba_settings_points.json"):
        text = path.read_text(encoding="utf-8")
        assert _view(path)["members"] == []
        assert "espn_s2" not in text.lower() and "swid" not in text.lower()
