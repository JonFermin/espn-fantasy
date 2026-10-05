"""The NBA sport plugin (ROADMAP #17): eligibility, stat schema, daily scoring periods, per-game locks and the add/drop
cutoff at the day's first tip.

``tests/fixtures/sports/fba_pro_schedule_2027.json`` is a real ``proTeamSchedules_wl`` capture for ``fba`` 2027 (public,
no cookies, 2026-10-05) trimmed to days 1-3 and Christmas Day (day 67), every game listed under both teams as ESPN
sends it. In the full capture all 1,200 games tip on the US Eastern day their period number implies, counted from
opening night (Tue Oct 20, 2026), and the days without games (Election Day, Thanksgiving, Christmas Eve, the NBA Cup
knockout window, the All-Star break) are simply absent. It is parsed by ``fm.espn.models.ProSchedule``, the model the
sync reads it with. Instants below are UTC, with the Eastern wall-clock time beside them.

ESPN spells six teams its own way in that view (``NY``, ``SA``, ``GS``, ``NO``, ``UTAH``, ``WSH``); the id maps use
nba.com tricodes, so the tests name teams through :data:`fm.espn.ids.FBA`.
"""

from __future__ import annotations

import copy
import json
from datetime import UTC, date, datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from fm.espn.ids import FBA, Game
from fm.espn.models import ProSchedule
from fm.espn.settings import LeagueSettings, LockType, load_league_settings, parse_league_settings
from fm.sports.base import (
    FREE_AGENT_TEAM,
    LineupLock,
    PeriodKind,
    PeriodWindow,
    ScheduleLike,
    SportPlugin,
    StatSchema,
    plugin_for,
    start_times,
    teams_in,
)
from fm.sports.nba import (
    EASTERN,
    NBA,
    NBA_GAME_DURATION,
    NBA_SLOT_POSITIONS,
    PLUGIN,
    WEEKLY_LOCK_TYPES,
    NbaPlugin,
    eastern_day,
)
from fm.sports.nfl import NFL

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PRO_SCHEDULE = FIXTURES / "sports" / "fba_pro_schedule_2027.json"
POOL = FIXTURES / "sources" / "market" / "espn_fba_kona_player_info.json"  # real ESPN pool entries with eligibleSlots
FBA_POINTS = FIXTURES / "espn" / "fba_settings_points.json"
FBA_9CAT = FIXTURES / "espn" / "fba_settings_9cat.json"
FFL_PPR = FIXTURES / "espn" / "ffl_settings_ppr.json"

OPENING_NIGHT = date(2026, 10, 20)  # day 1
CHRISTMAS = 67
DAY1_FIRST = datetime(2026, 10, 20, 19, 0, tzinfo=UTC)  # BOS at DET, 3:00 p.m. ET
DAY1_SECOND = datetime(2026, 10, 20, 23, 0, tzinfo=UTC)  # PHI at NYK, 7:00 p.m. ET
DAY1_LAST = datetime(2026, 10, 21, 1, 30, tzinfo=UTC)  # OKC at SAS, 9:30 p.m. ET on Oct 20
DAY2_FIRST = datetime(2026, 10, 21, 23, 0, tzinfo=UTC)  # ATL at ORL and MIL at WAS, 7:00 p.m. ET
DAY2_LAST = datetime(2026, 10, 22, 2, 30, tzinfo=UTC)  # SAC at LAC, 10:30 p.m. ET on Oct 21
XMAS_FIRST = datetime(2026, 12, 25, 17, 0, tzinfo=UTC)  # SAS at NYK, noon ET
XMAS_LAST = datetime(2026, 12, 26, 3, 30, tzinfo=UTC)  # DEN at GSW, 10:30 p.m. ET on Dec 25
BOS_AT_DET = 401909088
OKC_AT_SAS = 401909090
SAC_AT_LAC = 401909842
SAS_AT_NYK_XMAS = 401909097


def team(abbrev: str) -> int:
    """ESPN pro team id by nba.com tricode, through the id map so the tests read as team names."""
    return next(team_id for team_id, label in FBA.pro_teams.items() if label == abbrev)


def slots(*labels: str) -> frozenset[int]:
    return frozenset(FBA.slot_id(label) for label in labels)


def eastern(at: datetime) -> str:
    return at.astimezone(EASTERN).strftime("%a %m-%d %H:%M")


def _view(path: Path = PRO_SCHEDULE) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


@cache
def _recorded_view() -> dict[str, Any]:
    return _view()


def _set_game(view: dict[str, Any], game_id: int, **fields: Any) -> None:
    """Edit a game in a ``proTeamSchedules_wl`` view under both teams that list it."""
    for pro_team in view["settings"]["proTeams"]:
        for games in pro_team.get("proGamesByScoringPeriod", {}).values():
            for game in games:
                if game["id"] == game_id:
                    game.update(fields)


def _epoch_ms(at: datetime) -> int:
    return int(at.timestamp() * 1000)


def load_schedule(view: dict[str, Any] | None = None) -> ScheduleLike:
    schedule: ScheduleLike = ProSchedule.model_validate(view if view is not None else _recorded_view())
    return schedule


@pytest.fixture
def schedule() -> ScheduleLike:
    return load_schedule()


@pytest.fixture
def fba_points() -> LeagueSettings:
    return load_league_settings(FBA_POINTS)


# --- plugin identity and discovery ------------------------------------------------------------------------------------


def test_nba_plugin_identity() -> None:
    assert isinstance(NBA, SportPlugin) and isinstance(NBA, NbaPlugin)
    assert PLUGIN is NBA
    assert NBA.game is Game.FBA and NBA.sport == "nba"
    assert NBA.period_kind is PeriodKind.DAY
    assert NBA.ids is FBA
    assert NBA.game_duration == NBA_GAME_DURATION == timedelta(hours=3)


def test_plugin_for_finds_the_nba_plugin_by_convention() -> None:
    assert plugin_for("nba") is NBA
    assert plugin_for("fba") is NBA
    assert plugin_for(Game.FBA) is NBA
    assert plugin_for("NBA") is NBA
    assert plugin_for("nfl") is NFL  # the NFL plugin is untouched


# --- slot eligibility -------------------------------------------------------------------------------------------------


def test_util_takes_everyone_and_g_and_f_split_guards_from_forwards() -> None:
    assert NBA.positions_for_slot(FBA.slot_id("UTIL")) == frozenset({"PG", "SG", "SF", "PF", "C"})
    assert NBA.positions_for_slot(FBA.slot_id("UT")) == NBA.positions_for_slot(FBA.slot_id("UTIL"))  # ESPN's alias
    assert NBA.positions_for_slot(FBA.slot_id("G")) == frozenset({"PG", "SG"})
    assert NBA.positions_for_slot(FBA.slot_id("F")) == frozenset({"SF", "PF"})
    assert not NBA.is_eligible("C", FBA.slot_id("F")) and not NBA.is_eligible("SF", FBA.slot_id("G"))
    assert NBA.is_eligible("C", FBA.slot_id("UTIL")) and NBA.is_eligible("PG", FBA.slot_id("UTIL"))


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        ("PG", ("PG", "G", "G/F", "UTIL", "BE", "IR")),
        ("SG", ("SG", "G", "SG/SF", "G/F", "UTIL", "BE", "IR")),
        ("SF", ("SF", "F", "SG/SF", "G/F", "F/C", "UTIL", "BE", "IR")),
        ("PF", ("PF", "F", "G/F", "PF/C", "F/C", "UTIL", "BE", "IR")),
        ("C", ("C", "PF/C", "F/C", "UTIL", "BE", "IR")),
    ],
)
def test_eligible_slots_by_position(position: str, expected: tuple[str, ...]) -> None:
    assert NBA.eligible_slots(position) == slots(*expected)
    assert NBA.eligible_slots(position, include_reserve=False) == slots(*expected) - slots("BE", "IR")
    assert NBA.eligible_slots(FBA.position_id(position)) == NBA.eligible_slots(position)  # by ESPN position id


def test_bench_ir_rookie_and_unknown_slots() -> None:
    every = frozenset(FBA.positions.values())
    assert NBA.positions_for_slot(FBA.bench_slot) == every and NBA.positions_for_slot(FBA.ir_slot) == every
    assert NBA.positions_for_slot(FBA.slot_id("Rookie")) == frozenset()  # experience, not position
    assert NBA.positions_for_slot(14) == frozenset() and NBA.positions_for_slot(99) == frozenset()
    assert NBA.position_label(5) == "C" and NBA.position_label("C") == "C"
    with pytest.raises(KeyError):
        NBA.eligible_slots("G")  # a slot, not a position
    with pytest.raises(KeyError):
        NBA.eligible_slots(0)  # position ids are 1-based


def test_slot_table_uses_espn_ids_and_is_read_only() -> None:
    assert NBA.slot_positions is NBA_SLOT_POSITIONS
    assert len(NBA_SLOT_POSITIONS) == 12 and set(NBA_SLOT_POSITIONS) == set(range(12))  # PG (0) through UTIL (11)
    assert FBA.bench_slot not in NBA_SLOT_POSITIONS and FBA.ir_slot not in NBA_SLOT_POSITIONS
    assert all(positions <= set(FBA.positions.values()) for positions in NBA_SLOT_POSITIONS.values())
    table: Any = NBA_SLOT_POSITIONS
    with pytest.raises(TypeError):
        table[11] = frozenset()


def test_eligibility_reproduces_espn_eligible_slots_for_real_players() -> None:
    """ESPN's own ``eligibleSlots`` are the union of the slots of every position a player qualifies at."""
    single = {slot_id: label for slot_id, label in FBA.lineup_slots.items() if label in FBA.positions.values()}
    entries = json.loads(POOL.read_text(encoding="utf-8"))["players"]
    assert len(entries) == 7
    multi_position = 0
    for entry in entries:
        player = entry["player"]
        espn_slots = frozenset(player["eligibleSlots"])
        positions = {single[slot_id] for slot_id in espn_slots if slot_id in single}
        assert FBA.positions[player["defaultPositionId"]] in positions, player["fullName"]
        derived = frozenset[int]().union(*(NBA.eligible_slots(position) for position in positions))
        assert derived == espn_slots, player["fullName"]
        multi_position += len(positions) > 1
    assert multi_position == 3  # Edwards (SG/SF), Antetokounmpo (PF/C), Flagg (SF/PF)


@pytest.mark.parametrize("path", [FBA_POINTS, FBA_9CAT])
def test_eligibility_covers_the_league_fixture_slots(path: Path) -> None:
    league = load_league_settings(path)
    assert all(NBA.positions_for_slot(slot.slot_id) for slot in league.active_slots)
    center_slots = {slot_id for slot_id in NBA.eligible_slots("C") if league.slot_count(slot_id)}
    assert center_slots == slots("C", "UTIL", "BE", "IR")
    guard_slots = {slot_id for slot_id in NBA.eligible_slots("PG") if league.slot_count(slot_id)}
    assert guard_slots == slots("PG", "G", "UTIL", "BE", "IR")


# --- stat schema ------------------------------------------------------------------------------------------------------


def test_stat_schema_is_espns_nba_vocabulary() -> None:
    schema = NBA.stat_schema
    assert schema.game is Game.FBA and len(schema) == len(FBA.stats)
    assert schema.abbr(0) == "PTS" and schema.stat_id("3PM") == 17 and schema.stat_id("TO") == 11
    assert schema.label("FG%") == "Field Goal Percentage"
    assert schema.from_espn({"0": 27, "6": 11.5, "17": 3, "42": 1}) == {"PTS": 27.0, "REB": 11.5, "3PM": 3.0, "GP": 1.0}
    assert schema.to_espn({"AST": 9.0}) == {3: 9.0}
    assert NBA.stat_schema is schema and schema == StatSchema.for_game("nba") != NFL.stat_schema


def test_stat_schema_knows_what_a_league_scores(fba_points: LeagueSettings) -> None:
    scored = NBA.stat_schema.scored_by(fba_points)
    assert set(scored) == {"PTS", "3PM", "FGM", "FGA", "FTM", "FTA", "REB", "AST", "STL", "BLK", "TO"}
    nine_cat = NBA.stat_schema.scored_by(load_league_settings(FBA_9CAT))
    assert {"FG%", "FT%", "TO"} <= set(nine_cat)
    with pytest.raises(ValueError, match="is ffl, not fba"):
        NBA.stat_schema.scored_by(load_league_settings(FFL_PPR))


# --- the recorded schedule --------------------------------------------------------------------------------------------


def test_fixture_is_a_trimmed_real_capture() -> None:
    text = PRO_SCHEDULE.read_text(encoding="utf-8")
    assert len(text.encode()) < 15_000
    lowered = text.lower()
    assert "espn_s2" not in lowered and "swid" not in lowered and "members" not in lowered
    view = _view()
    pro_teams = view["settings"]["proTeams"]
    assert len(pro_teams) == 31 and {pro_team["id"] for pro_team in pro_teams} == set(range(31))  # FA is id 0
    espn_spellings = {pro_team["abbrev"] for pro_team in pro_teams} - set(FBA.pro_teams.values())
    assert espn_spellings == {"NY", "SA", "GS", "NO", "UTAH", "WSH"}
    periods = {period for pro_team in pro_teams for period in pro_team["proGamesByScoringPeriod"]}
    assert periods == {"1", "2", "3", "67"}
    listed: dict[int, list[dict[str, Any]]] = {}
    for pro_team in pro_teams:
        for games in pro_team["proGamesByScoringPeriod"].values():
            for game in games:
                listed.setdefault(game["id"], []).append(game)
    assert len(listed) == 21 and all(len(copies) == 2 and copies[0] == copies[1] for copies in listed.values())


def test_days_carry_their_games_and_idle_teams() -> None:
    recorded = ProSchedule.model_validate(_recorded_view())
    assert recorded.scoring_periods == (1, 2, 3, CHRISTMAS)
    assert [len(recorded.games(period)) for period in recorded.scoring_periods] == [3, 11, 2, 5]
    assert [len(recorded.idle_teams(period)) for period in recorded.scoring_periods] == [24, 8, 26, 20]
    assert all(len(teams_in(recorded, period)) == 30 for period in recorded.scoring_periods)
    first = recorded.first_game(1)
    assert first is not None and first.id == BOS_AT_DET and first.date == DAY1_FIRST
    assert (first.away_pro_team_id, first.home_pro_team_id) == (team("BOS"), team("DET"))
    assert recorded.games(38) == () and recorded.first_game(38) is None  # Thanksgiving: no games


# --- days -------------------------------------------------------------------------------------------------------------


def test_periods_count_us_eastern_days_from_opening_night(schedule: ScheduleLike) -> None:
    assert NBA.period_day(1, schedule) == OPENING_NIGHT
    assert NBA.period_day(2, schedule) == date(2026, 10, 21)
    assert NBA.period_day(38, schedule) == date(2026, 11, 26)  # Thanksgiving keeps its number without games
    assert NBA.period_day(66, schedule) == date(2026, 12, 24)
    assert NBA.period_day(CHRISTMAS, schedule) == date(2026, 12, 25)
    assert NBA.period_day(0, schedule) is None and NBA.period_day(68, schedule) is None  # outside the schedule
    assert NBA.period_for_day(OPENING_NIGHT, schedule) == 1
    assert NBA.period_for_day(date(2026, 11, 26), schedule) == 38
    assert NBA.period_for_day(date(2026, 12, 25), schedule) == CHRISTMAS
    assert NBA.period_for_day(date(2026, 10, 19), schedule) is None
    assert NBA.period_for_day(date(2026, 12, 26), schedule) is None


def test_every_tip_falls_on_its_periods_eastern_day(schedule: ScheduleLike) -> None:
    late = 0
    for period in schedule.scoring_periods:
        for game in schedule.games(period):
            assert eastern_day(game.date) == NBA.period_day(period, schedule), game.id
            late += game.date.date() != eastern_day(game.date)  # from 8 p.m. EDT (7 p.m. EST) it is tomorrow in UTC
    assert late == 10
    assert eastern(DAY2_LAST) == "Wed 10-21 22:30" and DAY2_LAST.date() == date(2026, 10, 22)


def test_scoring_period_turns_at_eastern_midnight(schedule: ScheduleLike) -> None:
    assert NBA.scoring_period_at(datetime(2026, 10, 5, tzinfo=UTC), schedule) == 1  # preseason: opening night is next
    assert NBA.scoring_period_at(DAY1_FIRST - timedelta(hours=8), schedule) == 1
    assert NBA.scoring_period_at(DAY1_LAST + timedelta(hours=1), schedule) == 1  # 10:30 p.m. ET, Oct 20
    midnight = datetime(2026, 10, 21, 4, 0, tzinfo=UTC)  # 00:00 EDT on Oct 21
    assert NBA.scoring_period_at(midnight - timedelta(seconds=1), schedule) == 1
    assert NBA.scoring_period_at(midnight, schedule) == 2
    # OKC at SAS may still be on (its window runs to 04:30 UTC), but its slots are locked in day 1's lineup; the
    # game-end rule of the base plugin would still say day 1.
    assert midnight < DAY1_LAST + NBA_GAME_DURATION
    assert SportPlugin.scoring_period_at(NBA, midnight, schedule) == 1
    assert NBA.scoring_period_at(datetime(2026, 11, 26, 17, 0, tzinfo=UTC), schedule) == 38  # a day without games
    assert NBA.scoring_period_at(XMAS_LAST, schedule) == CHRISTMAS
    christmas_midnight = datetime(2026, 12, 26, 5, 0, tzinfo=UTC)  # 00:00 EST on Dec 26
    assert NBA.scoring_period_at(christmas_midnight - timedelta(seconds=1), schedule) == CHRISTMAS
    assert NBA.scoring_period_at(christmas_midnight, schedule) is None  # past the trimmed schedule's last day
    with pytest.raises(ValueError, match="aware"):
        NBA.scoring_period_at(datetime(2026, 10, 21, 12, 0), schedule)


def test_eastern_days_follow_daylight_saving_time(schedule: ScheduleLike) -> None:
    # EDT ends at 06:00 UTC on Sun Nov 1, 2026: midnight ET is 04:00 UTC before it and 05:00 UTC after.
    assert eastern_day(datetime(2026, 11, 1, 3, 59, tzinfo=UTC)) == date(2026, 10, 31)
    assert eastern_day(datetime(2026, 11, 1, 4, 0, tzinfo=UTC)) == date(2026, 11, 1)
    assert eastern_day(datetime(2026, 11, 2, 4, 59, tzinfo=UTC)) == date(2026, 11, 1)
    assert eastern_day(datetime(2026, 11, 2, 5, 0, tzinfo=UTC)) == date(2026, 11, 2)
    assert NBA.scoring_period_at(datetime(2026, 11, 2, 4, 59, tzinfo=UTC), schedule) == 13
    assert NBA.scoring_period_at(datetime(2026, 11, 2, 5, 0, tzinfo=UTC), schedule) == 14
    assert eastern_day(datetime(2026, 10, 20, 23, 0, tzinfo=EASTERN)) == OPENING_NIGHT  # any aware zone works
    with pytest.raises(ValueError, match="aware"):
        eastern_day(datetime(2026, 10, 20, 23, 0))


def test_a_schedule_that_breaks_the_day_count_raises() -> None:
    view = copy.deepcopy(_recorded_view())
    _set_game(view, SAS_AT_NYK_XMAS, date=_epoch_ms(XMAS_FIRST - timedelta(days=1)))  # Christmas Eve, filed under 67
    broken = load_schedule(view)
    with pytest.raises(ValueError, match=r"scoring period 67 first tips on 2026-12-24 .* puts period 67 on 2026-12-25"):
        NBA.scoring_period_at(DAY1_FIRST, broken)
    with pytest.raises(ValueError, match="consecutive US Eastern days"):
        NBA.period_day(1, broken)
    # Locks and the add/drop cutoff read the schedule as it is; only the day count refuses to guess.
    assert NBA.transaction_cutoff(1, broken) == DAY1_FIRST


def test_placeholder_tips_neither_anchor_nor_break_the_day_count() -> None:
    view = copy.deepcopy(_recorded_view())
    placeholder = datetime(2026, 10, 20, 0, 0, tzinfo=UTC)  # UTC midnight: 8 p.m. ET on Oct 19
    _set_game(view, BOS_AT_DET, date=_epoch_ms(placeholder), startTimeTBD=True, validForLocking=False)
    schedule = load_schedule(view)
    assert NBA.period_day(1, schedule) == OPENING_NIGHT and NBA.period_day(CHRISTMAS, schedule) == date(2026, 12, 25)
    assert NBA.scoring_period_at(DAY1_SECOND, schedule) == 1
    assert NBA.transaction_cutoff(1, schedule) == placeholder  # an unscheduled start errs early, like a lock
    provisional = {lock.team_id for lock in NBA.locks(1, schedule) if lock.provisional}
    assert provisional == {team("BOS"), team("DET")}


def test_a_day_of_placeholders_is_counted_from_the_next_confirmed_day() -> None:
    view = copy.deepcopy(_recorded_view())
    for game_id in (BOS_AT_DET, 401909089, OKC_AT_SAS):  # all of day 1, at a wrong placeholder
        _set_game(view, game_id, date=_epoch_ms(datetime(2026, 10, 19, 7, 1, tzinfo=UTC)), startTimeTBD=True)
    schedule = load_schedule(view)
    assert NBA.period_day(1, schedule) == OPENING_NIGHT  # day 2 anchors the count, which still starts at day 1
    assert NBA.period_for_day(OPENING_NIGHT, schedule) == 1
    assert NBA.scoring_period_at(datetime(2026, 10, 20, 16, 0, tzinfo=UTC), schedule) == 1


def test_a_schedule_without_confirmed_games_has_no_days() -> None:
    empty = load_schedule({"settings": {"proTeams": [{"id": 0, "abbrev": "FA", "proGamesByScoringPeriod": {}}]}})
    assert NBA.scoring_period_at(DAY1_FIRST, empty) is None
    assert NBA.period_day(1, empty) is None and NBA.period_for_day(OPENING_NIGHT, empty) is None
    view = copy.deepcopy(_recorded_view())
    for game_id in (BOS_AT_DET, 401909089, OKC_AT_SAS):
        _set_game(view, game_id, startTimeTBD=True)
    for pro_team in view["settings"]["proTeams"]:
        pro_team["proGamesByScoringPeriod"] = {
            period: games for period, games in pro_team["proGamesByScoringPeriod"].items() if period == "1"
        }
    only_placeholders = load_schedule(view)
    assert NBA.period_day(1, only_placeholders) is None and NBA.scoring_period_at(DAY1_FIRST, only_placeholders) is None


@cache
def _property_schedule() -> ScheduleLike:
    return load_schedule()


@given(st.datetimes(min_value=datetime(2026, 9, 1), max_value=datetime(2027, 1, 31), timezones=st.just(UTC)))
def test_scoring_period_is_the_eastern_day_number(at: datetime) -> None:
    schedule = _property_schedule()
    day = eastern_day(at)
    period = NBA.scoring_period_at(at, schedule)
    if day > date(2026, 12, 25):
        assert period is None
    elif day < OPENING_NIGHT:
        assert period == 1
    else:
        expected = (day - OPENING_NIGHT).days + 1
        assert period == expected
        assert NBA.period_day(expected, schedule) == day and NBA.period_for_day(day, schedule) == expected
    later = NBA.scoring_period_at(at + timedelta(hours=7), schedule)
    assert period is None or later is None or later >= period  # never runs backwards


# --- adds and drops ---------------------------------------------------------------------------------------------------


def test_adds_and_drops_close_at_the_days_first_tip(schedule: ScheduleLike) -> None:
    assert NBA.transaction_cutoff(1, schedule) == DAY1_FIRST and eastern(DAY1_FIRST) == "Tue 10-20 15:00"
    assert NBA.transaction_cutoff(2, schedule) == DAY2_FIRST
    assert NBA.transaction_cutoff(CHRISTMAS, schedule) == XMAS_FIRST and eastern(XMAS_FIRST) == "Fri 12-25 12:00"
    assert NBA.transaction_cutoff(38, schedule) is None and NBA.transaction_cutoff(999, schedule) is None
    recorded = ProSchedule.model_validate(_recorded_view())
    for period in recorded.scoring_periods:
        first = recorded.first_game(period)
        assert first is not None and NBA.transaction_cutoff(period, schedule) == first.date
        cutoff = NBA.transaction_cutoff(period, schedule)
        assert cutoff is not None and all(cutoff <= lock.at for lock in NBA.locks(period, schedule))
    # Lineups still move per game after the cutoff: PHI at NYK locks four hours later.
    assert not NBA.is_locked(team("NYK"), 1, DAY1_FIRST, schedule)
    assert NBA.lock_time(team("NYK"), 1, schedule) == DAY1_SECOND


# --- lock times -------------------------------------------------------------------------------------------------------


def test_lineups_lock_at_each_teams_tip(schedule: ScheduleLike) -> None:
    assert NBA.lock_time(team("DET"), 1, schedule) == DAY1_FIRST
    assert NBA.lock_time(team("NYK"), 1, schedule) == DAY1_SECOND
    assert NBA.lock_time(team("SAS"), 1, schedule) == DAY1_LAST
    assert NBA.lock_time(team("LAC"), 2, schedule) == DAY2_LAST
    assert NBA.lock_time(team("LAL"), 1, schedule) is None  # off day: movable all day
    assert NBA.lock_time(FREE_AGENT_TEAM, 1, schedule) is None and NBA.lock_time(team("DET"), 38, schedule) is None
    assert not NBA.is_locked(team("SAS"), 1, DAY1_LAST - timedelta(seconds=1), schedule)
    assert NBA.is_locked(team("SAS"), 1, DAY1_LAST, schedule)
    assert NBA.is_locked(team("DET"), 1, DAY1_SECOND, schedule) and not NBA.is_locked(
        team("LAL"), 1, XMAS_LAST, schedule
    )
    with pytest.raises(ValueError, match="aware"):
        NBA.is_locked(team("DET"), 1, datetime(2026, 10, 20, 20, 0), schedule)


def test_lock_windows_and_locks_per_game(schedule: ScheduleLike) -> None:
    windows = NBA.lock_windows(2, schedule)
    assert windows == start_times(schedule, 2)
    assert [eastern(window)[-5:] for window in windows] == ["19:00", "19:30", "20:00", "20:30", "22:00", "22:30"]
    locks = NBA.locks(2, schedule)
    assert len(locks) == 22 and [lock.at for lock in locks] == sorted(lock.at for lock in locks)
    assert locks[-2:] == (
        LineupLock(team_id=team("LAC"), period=2, at=DAY2_LAST, game_id=SAC_AT_LAC),
        LineupLock(team_id=team("SAC"), period=2, at=DAY2_LAST, game_id=SAC_AT_LAC),
    )
    assert len(NBA.locks(CHRISTMAS, schedule)) == 10 and NBA.locks(38, schedule) == ()
    assert not any(lock.provisional for lock in locks)


def test_first_game_lock_locks_everyone_at_the_first_tip(schedule: ScheduleLike) -> None:
    first = LockType.FIRSTGAME_SCORINGPERIOD
    assert NBA.lock_time(team("SAS"), 1, schedule, lock_type=first) == DAY1_FIRST
    assert NBA.lock_time(team("LAL"), 1, schedule, lock_type=first) == DAY1_FIRST  # off day, locked all the same
    assert NBA.lock_windows(1, schedule, lock_type=first) == (DAY1_FIRST,)
    locks = NBA.locks(1, schedule, lock_type=first)
    assert len(locks) == 30 and {lock.at for lock in locks} == {DAY1_FIRST}
    by_team = {lock.team_id: lock for lock in locks}
    assert by_team[team("LAL")].game_id is None and by_team[team("SAS")].game_id == OKC_AT_SAS


def test_weekly_and_unknown_lock_types_are_refused(schedule: ScheduleLike) -> None:
    # A day is the period, so a lock "at the first game of the week" is not the day's first tip.
    assert WEEKLY_LOCK_TYPES == {LockType.FIRST_GAME_OF_WEEK}
    weekly = LockType.FIRST_GAME_OF_WEEK
    with pytest.raises(ValueError, match="locks a whole matchup week"):
        NBA.lock_time(team("SAS"), 1, schedule, lock_type=weekly)
    with pytest.raises(ValueError, match="locks a whole matchup week"):
        NBA.is_locked(team("SAS"), 1, DAY1_LAST, schedule, lock_type=weekly)
    with pytest.raises(ValueError, match="locks a whole matchup week"):
        NBA.lock_windows(1, schedule, lock_type=weekly)
    with pytest.raises(ValueError, match="locks a whole matchup week"):
        NBA.locks(1, schedule, lock_type=weekly)
    with pytest.raises(ValueError, match="UNKNOWN"):
        NBA.locks(1, schedule, lock_type=LockType.UNKNOWN)
    with pytest.raises(ValueError, match="UNKNOWN"):
        NBA.lock_time(team("SAS"), 1, schedule, lock_type=LockType.UNKNOWN)


def test_lock_type_comes_from_league_settings(schedule: ScheduleLike, fba_points: LeagueSettings) -> None:
    assert fba_points.lineup_lock_type is LockType.INDIVIDUAL_GAME
    assert NBA.lock_time(team("SAS"), 1, schedule, lock_type=fba_points.lineup_lock_type) == DAY1_LAST
    view = json.loads(FBA_POINTS.read_text(encoding="utf-8"))
    view["settings"]["rosterSettings"]["lineupLocktimeType"] = "FIRSTGAME_SCORINGPERIOD"
    daily = parse_league_settings(view)
    assert NBA.lock_time(team("SAS"), 1, schedule, lock_type=daily.lineup_lock_type) == DAY1_FIRST
    # ESPN's fba typeNames also list FIRSTGAME_WEEKLY and INDIVIDUAL_FIRSTGAME_WEEKLY; until they are mapped they
    # parse as UNKNOWN, which no plugin computes a lock for.
    view["settings"]["rosterSettings"]["lineupLocktimeType"] = "FIRSTGAME_WEEKLY"
    weekly = parse_league_settings(view)
    assert weekly.lineup_lock_type is LockType.UNKNOWN and weekly.lineup_lock_type_raw == "FIRSTGAME_WEEKLY"
    with pytest.raises(ValueError, match="UNKNOWN"):
        NBA.lock_time(team("SAS"), 1, schedule, lock_type=weekly.lineup_lock_type)


def test_tbd_games_make_their_locks_provisional() -> None:
    view = copy.deepcopy(_recorded_view())
    _set_game(view, SAC_AT_LAC, startTimeTBD=True, validForLocking=False)
    schedule = load_schedule(view)
    assert NBA.lock_time(team("LAC"), 2, schedule) == DAY2_LAST  # the placeholder is still the best estimate
    provisional = {lock.team_id for lock in NBA.locks(2, schedule) if lock.provisional}
    assert provisional == {team("LAC"), team("SAC")}
    assert not any(lock.provisional for lock in NBA.locks(2, schedule, lock_type=LockType.FIRSTGAME_SCORINGPERIOD))


# --- period windows ---------------------------------------------------------------------------------------------------


def test_period_window_spans_first_to_last_tip(schedule: ScheduleLike) -> None:
    window = NBA.period_window(1, schedule)
    assert window == PeriodWindow(
        period=1, first_start=DAY1_FIRST, last_start=DAY1_LAST, end=DAY1_LAST + timedelta(hours=3)
    )
    assert NBA.period_window(38, schedule) is None
