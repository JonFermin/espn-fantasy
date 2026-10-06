"""ESPN's season calendar (ROADMAP #31): the shipped per-season files, the matchup weeks they resolve and the extractor.

Real data, offline: ``data/calendars/fba_2027.json`` and ``ffl_2026.json`` are the web client's calendars as
``capture.py webclient`` saved them (``tests/fixtures/espn/real/*/calendar.json`` holds the same capture).
"""

from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import pytest

from fm import paths
from fm.espn.calendar import (
    CalendarError,
    calendar_file,
    extract_calendar,
    find_calendar,
    league_matchup_days,
    load_calendar,
    matchup_period_of,
    matchup_scoring_periods,
    parse_calendar,
)
from fm.espn.ids import Game
from fm.espn.settings import load_league_settings

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
REAL = FIXTURES / "espn" / "real"


def test_the_files_live_under_data_and_are_read_through_the_paths_helper() -> None:
    assert calendar_file("fba", 2027) == paths.data_file("calendars/fba_2027.json")
    assert calendar_file(Game.FFL, 2026).name == "ffl_2026.json"
    assert calendar_file("nba", 2027, root=Path("elsewhere")) == Path("elsewhere") / "fba_2027.json"
    assert calendar_file("fba", 2027).is_file()


def test_the_real_nba_weeks_resolve_to_days() -> None:
    settings = load_league_settings(REAL / "fba" / "mSettings.json")
    days = league_matchup_days(settings)
    assert days is not None and len(days) == 21
    assert days[1] == (1, 2, 3, 4, 5, 6)  # Tue Oct 20 - Sun Oct 25, 2026
    assert days[2] == tuple(range(7, 14)) and days[17] == tuple(range(112, 119))  # Monday-Sunday weeks
    assert days[18] == tuple(range(119, 133))  # the 14 days around the All-Star break
    assert days[19] == tuple(range(133, 140)) and days[21] == tuple(range(147, 154))  # the playoffs
    assert sum(len(span) for span in days.values()) == 153  # 6 + 16 x 7 + 14 + 3 x 7: every day exactly once
    assert settings.final_scoring_period == 153
    assert matchup_scoring_periods(settings, 5) == (1, 2, 3, 4, 5, 6)
    assert matchup_period_of(settings, 120) == 18
    assert matchup_period_of(settings, 999) is None and matchup_scoring_periods(settings, 0) is None


def test_a_league_that_lists_scoring_periods_needs_no_calendar() -> None:
    settings = load_league_settings(REAL / "ffl" / "mSettings.json")
    assert settings.schedule.lists_scoring_periods
    assert matchup_scoring_periods(settings, 4) == (4,)
    stand_in = load_league_settings(FIXTURES / "espn" / "fba_settings_points.json")
    assert matchup_scoring_periods(stand_in, 8) == tuple(range(7, 14))
    assert league_matchup_days(stand_in.model_copy(update={"season": 2099})) is not None  # identity: no file needed


def test_a_season_without_a_calendar_answers_none_and_load_names_the_fix() -> None:
    settings = load_league_settings(REAL / "fba" / "mSettings.json").model_copy(update={"season": 2099})
    assert find_calendar("fba", 2099) is None
    assert league_matchup_days(settings) is None and matchup_scoring_periods(settings, 3) is None
    with pytest.raises(CalendarError, match="capture.py webclient"):
        load_calendar("fba", 2099)


def test_the_calendar_reads_the_web_clients_windows() -> None:
    calendar = load_calendar("fba", 2027)
    assert (calendar.game, calendar.season) == (Game.FBA, 2027)
    assert calendar.last_regular_period == 174 and calendar.scoring_day(175) is not None
    day_two = calendar.scoring_day(2)
    assert day_two is not None and day_two.end - day_two.start == timedelta(days=1)
    assert calendar.scoring_day(0) is not None and calendar.scoring_day(0).pre_season  # type: ignore[union-attr]
    assert calendar.period_type(2).weekly and calendar.period_type(0).season_long
    with pytest.raises(CalendarError, match="no period type 9"):
        calendar.period_type(9)
    with pytest.raises(CalendarError, match="no period 99"):
        calendar.period_type(2).period(99)
    with pytest.raises(CalendarError, match="no period 99"):
        calendar.matchup_days({1: [99]}, 2)


def test_the_extractor_keeps_the_data_keys_and_checks_them() -> None:
    capture = json.loads((REAL / "fba" / "calendar.json").read_text(encoding="utf-8"))
    assert "errorCodes" in capture
    data = extract_calendar(capture)
    assert set(data) == {"game", "season", "webClientBuild", "scoringPeriods", "periodTypes"}
    assert parse_calendar(data) == load_calendar("fba", 2027)  # the shipped file is the capture, extracted
    assert json.loads(calendar_file("fba", 2027).read_text(encoding="utf-8")) == data
    with pytest.raises(CalendarError, match="missing periodTypes"):
        parse_calendar({"game": "fba", "season": 2027, "scoringPeriods": []})
    with pytest.raises(CalendarError, match="malformed"):
        parse_calendar({"game": "fba", "season": 2027, "scoringPeriods": [{"id": 1}], "periodTypes": []})
    broken = {**capture, "scoringPeriods": capture["scoringPeriods"][:10]}
    with pytest.raises(CalendarError, match="does not list"):
        extract_calendar(broken)


def test_a_malformed_file_leaves_a_leagues_matchup_days_unknown_instead_of_raising(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    settings = load_league_settings(REAL / "fba" / "mSettings.json")
    calendar_file(settings.game, settings.season, root=tmp_path).write_text("{not json", encoding="utf-8")
    with caplog.at_level("WARNING", logger="fm.espn.calendar"):
        assert league_matchup_days(settings, root=tmp_path) is None
        assert matchup_scoring_periods(settings, 5, root=tmp_path) is None
        assert matchup_period_of(settings, 5, root=tmp_path) is None
    assert "cannot be read" in caplog.text
    with pytest.raises(CalendarError, match="cannot be read"):  # the strict readers still say so
        find_calendar(settings.game, settings.season, root=tmp_path)
    with pytest.raises(CalendarError, match="cannot be read"):
        load_calendar(settings.game, settings.season, root=tmp_path)


def test_a_malformed_file_is_an_error_not_silence(tmp_path: Path) -> None:
    (tmp_path / "fba_2027.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(CalendarError, match="cannot be read"):
        find_calendar("fba", 2027, root=tmp_path)
    (tmp_path / "fba_2028.json").write_text("[]", encoding="utf-8")
    with pytest.raises(CalendarError, match="JSON object"):
        find_calendar("fba", 2028, root=tmp_path)
