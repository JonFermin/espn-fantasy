"""The NFL sport plugin (eligibility, stat schema, scoring periods, lock times) and the decision registry (ROADMAP #7).

``tests/fixtures/sports/ffl_pro_schedule_2026.json`` is a real ``proTeamSchedules_wl`` capture trimmed to 2026 weeks 4
and 5 (see the README there): week 4 has Thursday night, the 9:30 a.m. ET London slot, three Sunday windows, Sunday
night and Monday night with no byes; week 5 has CAR and KC on bye. :class:`FixtureSchedule` reads it with the surface
``fm.sports.base.ScheduleLike`` asks for, which is the shape of ``fm.espn.models.ProSchedule`` (ROADMAP #9), so the
plugin is exercised exactly as the sync will drive it. Instants below are UTC; Eastern renderings are checked where a
reader would think in kickoff times.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from fm.decide import registry as decide_registry
from fm.decide.registry import (
    DecisionRegistry,
    DuplicateDecisionError,
    Registration,
    UnknownDecisionError,
    normalize_kind,
    normalize_sport,
)
from fm.espn.ids import FFL, Game
from fm.espn.settings import LeagueSettings, LockType, load_league_settings, parse_league_settings
from fm.sports.base import (
    FREE_AGENT_TEAM,
    PLUGIN_ATTR,
    LineupLock,
    PeriodKind,
    PeriodWindow,
    ScheduleLike,
    SportPlugin,
    StatSchema,
    first_start,
    game_for,
    is_provisional,
    last_start,
    plugin_for,
    start_times,
    teams_in,
    teams_playing,
)
from fm.sports.nfl import NFL, NFL_GAME_DURATION, NFL_SLOT_POSITIONS, PLUGIN, NflPlugin

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
PRO_SCHEDULE = FIXTURES / "sports" / "ffl_pro_schedule_2026.json"
FFL_PPR = FIXTURES / "espn" / "ffl_settings_ppr.json"
FBA_POINTS = FIXTURES / "espn" / "fba_settings_points.json"
EASTERN = ZoneInfo("America/New_York")

# Week 4 and 5 instants from the fixture (UTC). Oct 1 20:15 ET is Oct 2 00:15 UTC, and so on.
TNF_WEEK_4 = datetime(2026, 10, 2, 0, 15, tzinfo=UTC)  # PIT at CLE
LONDON_WEEK_4 = datetime(2026, 10, 4, 13, 30, tzinfo=UTC)  # IND at WSH, 9:30 a.m. ET
EARLY_WEEK_4 = datetime(2026, 10, 4, 17, 0, tzinfo=UTC)  # 1:00 p.m. ET window, LAR at PHI among them
MNF_WEEK_4 = datetime(2026, 10, 6, 0, 15, tzinfo=UTC)  # ATL at NO
TNF_WEEK_5 = datetime(2026, 10, 9, 0, 15, tzinfo=UTC)  # TB at DAL
MNF_WEEK_5 = datetime(2026, 10, 13, 0, 15, tzinfo=UTC)  # BUF at LAR
TNF_WEEK_4_GAME = 401872964
LAR_AT_PHI_GAME = 401872970
TNF_WEEK_5_GAME = 401872980
MNF_WEEK_5_GAME = 401872994


def team(abbrev: str) -> int:
    """ESPN pro team id by abbreviation, through the id map so the tests read as team names."""
    return next(team_id for team_id, label in FFL.pro_teams.items() if label == abbrev)


def slots(*labels: str) -> frozenset[int]:
    return frozenset(FFL.slot_id(label) for label in labels)


def eastern(at: datetime) -> str:
    return at.astimezone(EASTERN).strftime("%a %H:%M")


def _view(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _set_game(view: dict[str, Any], game_id: int, **fields: Any) -> None:
    """Edit a game in a ``proTeamSchedules_wl`` view under both teams that list it."""
    for pro_team in view["settings"]["proTeams"]:
        for games in pro_team.get("proGamesByScoringPeriod", {}).values():
            for game in games:
                if game["id"] == game_id:
                    game.update(fields)


# --- a ScheduleLike over the recorded view ----------------------------------------------------------------------------


@dataclass(frozen=True)
class FixtureGame:
    """The ``GameLike`` surface: ESPN's field names, ``date`` as an aware UTC datetime."""

    id: int | None
    date: datetime
    home_pro_team_id: int
    away_pro_team_id: int
    start_time_tbd: bool = False
    valid_for_locking: bool = True


class FixtureSchedule:
    """``ScheduleLike`` over a ``proTeamSchedules_wl`` view, mirroring ``fm.espn.models.ProSchedule``'s surface."""

    def __init__(self, view: dict[str, Any]) -> None:
        self._by_team: dict[int, dict[int, list[FixtureGame]]] = {}
        for pro_team in view["settings"]["proTeams"]:
            periods = self._by_team.setdefault(int(pro_team["id"]), {})
            for period, games in pro_team.get("proGamesByScoringPeriod", {}).items():
                periods[int(period)] = [
                    FixtureGame(
                        id=game.get("id"),
                        date=datetime.fromtimestamp(game["date"] / 1000, tz=UTC),
                        home_pro_team_id=game["homeProTeamId"],
                        away_pro_team_id=game["awayProTeamId"],
                        start_time_tbd=game.get("startTimeTBD", False),
                        valid_for_locking=game.get("validForLocking", True),
                    )
                    for game in games
                ]

    @property
    def scoring_periods(self) -> tuple[int, ...]:
        return tuple(sorted({period for periods in self._by_team.values() for period in periods}))

    def games(self, scoring_period: int) -> tuple[FixtureGame, ...]:
        seen: dict[int | None, FixtureGame] = {}
        for periods in self._by_team.values():
            for game in periods.get(scoring_period, ()):
                seen.setdefault(game.id, game)
        return tuple(sorted(seen.values(), key=lambda game: (game.date, game.id or 0)))

    def games_for(self, pro_team_id: int, scoring_period: int) -> tuple[FixtureGame, ...]:
        return tuple(self._by_team.get(pro_team_id, {}).get(scoring_period, ()))

    def idle_teams(self, scoring_period: int) -> tuple[int, ...]:
        return tuple(
            sorted(
                team_id
                for team_id, periods in self._by_team.items()
                if team_id != FREE_AGENT_TEAM and not periods.get(scoring_period)
            )
        )


def load_schedule(view: dict[str, Any] | None = None) -> ScheduleLike:
    schedule: ScheduleLike = FixtureSchedule(view if view is not None else _view(PRO_SCHEDULE))
    return schedule


@pytest.fixture
def schedule() -> ScheduleLike:
    return load_schedule()


@pytest.fixture
def ffl_ppr() -> LeagueSettings:
    return load_league_settings(FFL_PPR)


@pytest.fixture
def fresh() -> DecisionRegistry:
    return DecisionRegistry()


# --- plugin identity and discovery ------------------------------------------------------------------------------------


def test_nfl_plugin_identity() -> None:
    assert isinstance(NFL, SportPlugin) and isinstance(NFL, NflPlugin)
    assert PLUGIN is NFL
    assert NFL.game is Game.FFL and NFL.sport == "nfl"
    assert NFL.period_kind is PeriodKind.WEEK
    assert NFL.ids is FFL
    assert NFL.game_duration == NFL_GAME_DURATION == timedelta(hours=4)
    with pytest.raises(TypeError):
        SportPlugin()  # type: ignore[abstract]


def test_plugin_for_accepts_sport_game_key_or_enum() -> None:
    assert plugin_for("nfl") is NFL
    assert plugin_for("ffl") is NFL
    assert plugin_for(Game.FFL) is NFL
    assert plugin_for("NFL") is NFL
    with pytest.raises(ValueError, match="mlb"):
        plugin_for("mlb")


def test_subclass_with_mismatched_sport_and_game_is_rejected_at_definition() -> None:
    with pytest.raises(TypeError, match="'nba' is not ESPN game 'ffl'"):

        class Mismatched(SportPlugin):
            game = Game.FFL
            sport = "nba"
            period_kind = PeriodKind.WEEK
            game_duration = timedelta(hours=4)

            @property
            def slot_positions(self) -> dict[int, frozenset[str]]:
                return {}


WRONG_GAME_PLUGIN = """
from datetime import timedelta

from fm.espn.ids import Game
from fm.sports.base import PeriodKind, SportPlugin


class Plugin(SportPlugin):
    game = Game.FBA
    sport = "nba"
    period_kind = PeriodKind.DAY
    game_duration = timedelta(hours=3)

    @property
    def slot_positions(self):
        return {}


PLUGIN = Plugin()
"""


@pytest.fixture
def synthetic_package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Callable[[str, str | None], str]]:
    """Create ``<name>`` on ``sys.path`` with an optional ``nfl.py``; the modules are dropped from ``sys.modules``
    afterwards so a later test cannot see a stale one."""
    created: list[str] = []

    def make(name: str, nfl_source: str | None) -> str:
        root = tmp_path / name
        root.mkdir()
        (root / "__init__.py").write_text("", encoding="utf-8")
        if nfl_source is not None:
            (root / "nfl.py").write_text(nfl_source, encoding="utf-8")
        created.append(name)
        return name

    monkeypatch.syspath_prepend(str(tmp_path))
    yield make
    for module_name in list(sys.modules):
        if any(module_name == name or module_name.startswith(f"{name}.") for name in created):
            del sys.modules[module_name]


def test_plugin_for_reports_missing_and_malformed_plugin_modules(
    synthetic_package: Callable[[str, str | None], str],
) -> None:
    with pytest.raises(LookupError, match="fm_t7_absent.nfl does not exist"):
        plugin_for("nfl", package="fm_t7_absent")  # the package itself is missing
    with pytest.raises(LookupError, match="fm_t7_empty.nfl does not exist"):
        plugin_for("nfl", package=synthetic_package("fm_t7_empty", None))
    with pytest.raises(TypeError, match=f"fm_t7_noattr.nfl.{PLUGIN_ATTR} should be a SportPlugin"):
        plugin_for("nfl", package=synthetic_package("fm_t7_noattr", "X = 1\n"))
    with pytest.raises(TypeError, match="should be a SportPlugin instance, got <object"):
        plugin_for("nfl", package=synthetic_package("fm_t7_object", f"{PLUGIN_ATTR} = object()\n"))
    with pytest.raises(TypeError, match="is the fba plugin, not ffl"):
        plugin_for("nfl", package=synthetic_package("fm_t7_wrong", WRONG_GAME_PLUGIN))
    # A plugin whose own import is broken surfaces that error; it is not "no plugin".
    broken = synthetic_package("fm_t7_broken", "import fm_t7_no_such_dependency\n")
    with pytest.raises(ModuleNotFoundError, match="fm_t7_no_such_dependency"):
        plugin_for("nfl", package=broken)


# --- slot eligibility -------------------------------------------------------------------------------------------------


def test_flex_takes_rb_wr_te_and_nothing_else() -> None:
    flex = FFL.slot_id("FLEX")
    assert NFL.positions_for_slot(flex) == frozenset({"RB", "WR", "TE"})
    assert NFL.is_eligible("RB", flex) and NFL.is_eligible("WR", flex) and NFL.is_eligible("TE", flex)
    assert not NFL.is_eligible("QB", flex)
    assert not NFL.is_eligible("K", flex)
    assert not NFL.is_eligible("D/ST", flex)
    assert not NFL.is_eligible("LB", flex)


def test_op_is_the_superflex_slot() -> None:
    op = FFL.slot_id("OP")
    assert NFL.positions_for_slot(op) == frozenset({"QB", "RB", "WR", "TE"})
    assert NFL.is_eligible("QB", op) and not NFL.is_eligible("QB", FFL.slot_id("FLEX"))
    assert NFL.is_eligible("RB", op) and NFL.is_eligible("WR", op) and NFL.is_eligible("TE", op)
    assert not NFL.is_eligible("K", op) and not NFL.is_eligible("D/ST", op)


@pytest.mark.parametrize(
    ("position", "expected"),
    [
        ("QB", ("QB", "OP", "BE", "IR")),
        ("RB", ("RB", "RB/WR", "RB/WR/TE", "OP", "BE", "IR")),
        ("WR", ("WR", "RB/WR", "WR/TE", "RB/WR/TE", "OP", "BE", "IR")),
        ("TE", ("TE", "WR/TE", "RB/WR/TE", "OP", "BE", "IR")),
        ("K", ("K", "BE", "IR")),
        ("P", ("P", "BE", "IR")),
        ("D/ST", ("D/ST", "BE", "IR")),
        ("HC", ("HC", "BE", "IR")),
        ("DT", ("DT", "DL", "DP", "BE", "IR")),
        ("DE", ("DE", "DL", "DP", "BE", "IR")),
        ("LB", ("LB", "DP", "BE", "IR")),
        ("CB", ("CB", "DB", "DP", "BE", "IR")),
        ("S", ("S", "DB", "DP", "BE", "IR")),
    ],
)
def test_eligible_slots_by_position(position: str, expected: tuple[str, ...]) -> None:
    assert NFL.eligible_slots(position) == slots(*expected)
    assert NFL.eligible_slots(position, include_reserve=False) == slots(*expected) - slots("BE", "IR")
    assert NFL.eligible_slots(FFL.position_id(position)) == NFL.eligible_slots(position)  # by ESPN position id


def test_bench_and_ir_take_every_position_and_unknowns_are_handled() -> None:
    every = frozenset(FFL.positions.values())
    assert NFL.positions_for_slot(FFL.bench_slot) == every and NFL.positions_for_slot(FFL.ir_slot) == every
    assert NFL.is_eligible(1, FFL.bench_slot)  # QB by position id
    assert NFL.positions_for_slot(99) == frozenset()
    assert NFL.positions_for_slot(FFL.slot_id("TQB")) == frozenset()  # team QB is not supported
    assert NFL.position_label(2) == "RB" and NFL.position_label("RB") == "RB"
    with pytest.raises(KeyError):
        NFL.eligible_slots("XX")
    with pytest.raises(KeyError):
        NFL.eligible_slots(99)


def test_slot_table_uses_espn_ids_and_is_read_only() -> None:
    assert NFL.slot_positions is NFL_SLOT_POSITIONS
    assert NFL_SLOT_POSITIONS[23] == frozenset({"RB", "WR", "TE"}) and NFL_SLOT_POSITIONS[7] >= {"QB", "TE"}
    assert set(NFL_SLOT_POSITIONS) <= set(FFL.lineup_slots)
    assert FFL.bench_slot not in NFL_SLOT_POSITIONS and FFL.ir_slot not in NFL_SLOT_POSITIONS
    assert all(positions <= set(FFL.positions.values()) for positions in NFL_SLOT_POSITIONS.values())
    table: Any = NFL_SLOT_POSITIONS
    with pytest.raises(TypeError):
        table[23] = frozenset()


def test_eligibility_covers_the_league_fixture_slots(ffl_ppr: LeagueSettings) -> None:
    assert all(NFL.positions_for_slot(slot.slot_id) for slot in ffl_ppr.active_slots)
    rb_slots_in_league = {slot_id for slot_id in NFL.eligible_slots("RB") if ffl_ppr.slot_count(slot_id)}
    assert rb_slots_in_league == slots("RB", "FLEX", "BE", "IR")
    view = _view(FFL_PPR)
    view["settings"]["rosterSettings"]["lineupSlotCounts"][str(FFL.slot_id("OP"))] = 1
    superflex = parse_league_settings(view)
    qb_slots_in_league = {slot_id for slot_id in NFL.eligible_slots("QB") if superflex.slot_count(slot_id)}
    assert qb_slots_in_league == slots("QB", "OP", "BE", "IR")


# --- stat schema ------------------------------------------------------------------------------------------------------


def test_stat_schema_maps_ids_and_abbreviations() -> None:
    schema = NFL.stat_schema
    assert schema.game is Game.FFL and len(schema) == len(FFL.stats) == 235
    assert schema.abbr(53) == "REC" and schema.stat_id("REC") == 53
    assert schema.label("REC") == "Each reception" and schema.label(4) == "TD Pass"
    assert "REC" in schema and 53 in schema
    assert "NOPE" not in schema and 999 not in schema and True not in schema
    assert schema.abbr(999) == "STAT_999" and schema.label(999) == "Stat 999"
    assert schema.stat_id("STAT_999") == 999 and "STAT_999" not in schema
    assert schema.abbreviations[:3] == ("PA", "PC", "INC")
    with pytest.raises(KeyError):
        schema.stat_id("NOPE")
    assert NFL.stat_schema is schema  # cached per plugin
    assert StatSchema.for_game("nfl") == schema and StatSchema.for_game("fba") != schema


def test_stat_schema_converts_espn_stat_dicts() -> None:
    schema = NFL.stat_schema
    line = schema.from_espn({"53": 5, 42: 61.5, "999": 1, "3": None})
    assert line == {"REC": 5.0, "REY": 61.5, "STAT_999": 1.0}
    assert schema.to_espn(line) == {53: 5.0, 42: 61.5, 999: 1.0}
    assert schema.unknown({"REC": 1.0, "foo": 2.0, "STAT_999": 3.0}) == ("foo", "STAT_999")
    assert schema.unknown(line) == ("STAT_999",)
    with pytest.raises(ValueError, match="non-numeric ESPN stat id 'REC'"):
        schema.from_espn({"REC": 1})
    with pytest.raises(ValueError, match="stat 53 has a non-numeric value 'five'"):
        schema.from_espn({"53": "five"})
    with pytest.raises(ValueError, match="non-numeric value True"):
        schema.from_espn({"53": True})
    with pytest.raises(KeyError):
        schema.to_espn({"foo": 1.0})


def test_stat_schema_knows_what_a_league_scores(ffl_ppr: LeagueSettings) -> None:
    scored = NFL.stat_schema.scored_by(ffl_ppr)
    assert "REC" in scored and "PY" in scored and "TK" not in scored
    assert len(scored) == 45 and scored == tuple(item.stat for item in ffl_ppr.scoring_items)
    with pytest.raises(ValueError, match="is fba, not ffl"):
        NFL.stat_schema.scored_by(load_league_settings(FBA_POINTS))


# --- the fixture schedule through the ScheduleLike surface ------------------------------------------------------------


def test_fixture_schedule_has_the_two_weeks_and_their_byes(schedule: ScheduleLike) -> None:
    assert tuple(schedule.scoring_periods) == (4, 5)
    assert len(schedule.games(4)) == 16 and len(schedule.games(5)) == 15 and schedule.games(99) == ()
    assert schedule.idle_teams(4) == () and tuple(schedule.idle_teams(5)) == (team("KC"), team("CAR"))
    assert len(teams_playing(schedule, 4)) == 32 and len(teams_playing(schedule, 5)) == 30
    assert teams_in(schedule, 5) == teams_playing(schedule, 4) and len(teams_in(schedule, 5)) == 32
    assert FREE_AGENT_TEAM not in teams_in(schedule, 4) and teams_in(schedule, 99) == frozenset()
    ids = [game.id for game in schedule.games(4)]
    assert len(set(ids)) == 16  # each game once although ESPN lists it under both teams


def test_game_for_a_team_and_period(schedule: ScheduleLike) -> None:
    game = game_for(schedule, team("PHI"), 4)
    assert game is not None and game.id == LAR_AT_PHI_GAME
    assert game.date == EARLY_WEEK_4 and eastern(game.date) == "Sun 13:00"
    assert (game.home_pro_team_id, game.away_pro_team_id) == (team("PHI"), team("LAR"))
    assert not game.start_time_tbd and game.valid_for_locking and not is_provisional(game)
    assert game_for(schedule, team("LAR"), 4) == game
    assert game_for(schedule, team("CAR"), 5) is None  # bye
    assert game_for(schedule, FREE_AGENT_TEAM, 4) is None
    assert game_for(schedule, team("PHI"), 99) is None
    assert game_for(schedule, 99, 4) is None


def test_kickoff_windows_come_from_the_schedule(schedule: ScheduleLike) -> None:
    windows = start_times(schedule, 4)
    assert len(windows) == 7 and windows[0] == TNF_WEEK_4 and windows[1] == LONDON_WEEK_4 and windows[-1] == MNF_WEEK_4
    assert [eastern(window) for window in windows] == [
        "Thu 20:15",
        "Sun 09:30",
        "Sun 13:00",
        "Sun 16:05",
        "Sun 16:25",
        "Sun 20:20",
        "Mon 20:15",
    ]
    assert first_start(schedule, 4) == TNF_WEEK_4 and last_start(schedule, 4) == MNF_WEEK_4
    assert first_start(schedule, 5) == TNF_WEEK_5 and last_start(schedule, 5) == MNF_WEEK_5
    assert first_start(schedule, 99) is None and last_start(schedule, 99) is None and start_times(schedule, 99) == ()


def test_fixture_is_trimmed_and_scrubbed() -> None:
    text = PRO_SCHEDULE.read_text(encoding="utf-8")
    assert len(text.encode()) < 20_000
    lowered = text.lower()
    assert "espn_s2" not in lowered and "swid" not in lowered and "members" not in lowered
    assert "teamPlayersByPosition" not in text
    view = _view(PRO_SCHEDULE)
    periods = {period for pro_team in view["settings"]["proTeams"] for period in pro_team["proGamesByScoringPeriod"]}
    assert periods == {"4", "5"}
    assert [pro_team["id"] for pro_team in view["settings"]["proTeams"]][:2] == [0, 1]  # FA pseudo-team kept as sent
    copies = [
        game
        for pro_team in view["settings"]["proTeams"]
        for games in pro_team["proGamesByScoringPeriod"].values()
        for game in games
        if game["id"] == LAR_AT_PHI_GAME
    ]
    assert len(copies) == 2 and copies[0] == copies[1]  # listed under both teams, identically


# --- lock times -------------------------------------------------------------------------------------------------------


def test_lock_time_is_each_teams_kickoff_under_individual_game_locking(schedule: ScheduleLike) -> None:
    assert NFL.lock_time(team("PHI"), 4, schedule) == EARLY_WEEK_4
    assert NFL.lock_time(team("PIT"), 4, schedule) == TNF_WEEK_4
    assert NFL.lock_time(team("IND"), 4, schedule) == LONDON_WEEK_4
    assert NFL.lock_time(team("ATL"), 4, schedule) == MNF_WEEK_4
    assert NFL.lock_time(team("BUF"), 5, schedule) == MNF_WEEK_5
    assert NFL.lock_time(team("CAR"), 5, schedule) is None  # bye: nothing locks all week
    assert NFL.lock_time(FREE_AGENT_TEAM, 4, schedule) is None
    assert NFL.lock_time(team("PHI"), 99, schedule) is None
    lock = NFL.lock_time(team("ATL"), 4, schedule)
    assert lock is not None and lock.astimezone(EASTERN).strftime("%a %m-%d %H:%M") == "Mon 10-05 20:15"


def test_is_locked_follows_the_clock(schedule: ScheduleLike) -> None:
    assert not NFL.is_locked(team("PHI"), 4, EARLY_WEEK_4 - timedelta(seconds=1), schedule)
    assert NFL.is_locked(team("PHI"), 4, EARLY_WEEK_4, schedule)
    assert NFL.is_locked(team("PHI"), 4, MNF_WEEK_4 + timedelta(days=1), schedule)
    assert NFL.is_locked(team("PIT"), 4, EARLY_WEEK_4, schedule)  # Thursday's game is over
    assert not NFL.is_locked(team("ATL"), 4, EARLY_WEEK_4, schedule)  # Monday night has not kicked off
    assert not NFL.is_locked(team("CAR"), 5, MNF_WEEK_5 + timedelta(hours=1), schedule)  # bye
    with pytest.raises(ValueError, match="aware"):
        NFL.is_locked(team("PHI"), 4, datetime(2026, 10, 4, 18, 0), schedule)


def test_first_game_lock_locks_everyone_at_the_opener(schedule: ScheduleLike) -> None:
    first = LockType.FIRSTGAME_SCORINGPERIOD
    assert NFL.lock_time(team("CAR"), 5, schedule, lock_type=first) == TNF_WEEK_5  # bye team locks too
    assert NFL.lock_time(team("BUF"), 5, schedule, lock_type=first) == TNF_WEEK_5  # not at its Monday kickoff
    assert NFL.lock_time(team("PHI"), 99, schedule, lock_type=first) is None
    assert NFL.is_locked(team("BUF"), 5, TNF_WEEK_5, schedule, lock_type=first)
    assert not NFL.is_locked(team("BUF"), 5, TNF_WEEK_5 - timedelta(minutes=1), schedule, lock_type=first)
    assert NFL.lock_windows(5, schedule, lock_type=first) == (TNF_WEEK_5,)
    assert NFL.lock_windows(99, schedule, lock_type=first) == ()
    locks = NFL.locks(5, schedule, lock_type=first)
    assert len(locks) == 32 and {lock.at for lock in locks} == {TNF_WEEK_5}
    by_team = {lock.team_id: lock for lock in locks}
    assert by_team[team("CAR")].game_id is None and by_team[team("BUF")].game_id == MNF_WEEK_5_GAME
    assert not any(lock.provisional for lock in locks)
    assert NFL.locks(99, schedule, lock_type=first) == ()


def test_unknown_lock_type_raises_instead_of_guessing(schedule: ScheduleLike) -> None:
    with pytest.raises(ValueError, match="UNKNOWN"):
        NFL.lock_time(team("PHI"), 4, schedule, lock_type=LockType.UNKNOWN)
    with pytest.raises(ValueError, match="UNKNOWN"):
        NFL.is_locked(team("PHI"), 4, EARLY_WEEK_4, schedule, lock_type=LockType.UNKNOWN)
    with pytest.raises(ValueError, match="UNKNOWN"):
        NFL.lock_windows(4, schedule, lock_type=LockType.UNKNOWN)
    with pytest.raises(ValueError, match="UNKNOWN"):
        NFL.locks(4, schedule, lock_type=LockType.UNKNOWN)


def test_lock_windows_and_locks_per_game(schedule: ScheduleLike) -> None:
    assert NFL.lock_windows(4, schedule) == start_times(schedule, 4) and len(NFL.lock_windows(4, schedule)) == 7
    assert NFL.lock_windows(99, schedule) == () and NFL.locks(99, schedule) == ()
    locks = NFL.locks(4, schedule)
    assert len(locks) == 32 and [lock.at for lock in locks] == sorted(lock.at for lock in locks)
    assert locks[0] == LineupLock(team_id=team("CLE"), period=4, at=TNF_WEEK_4, game_id=TNF_WEEK_4_GAME)
    assert (locks[1].team_id, locks[-1].team_id) == (team("PIT"), team("NO"))
    assert all(lock.period == 4 and not lock.provisional for lock in locks)
    for lock in locks:
        game = game_for(schedule, lock.team_id, 4)
        assert game is not None and game.id == lock.game_id and game.date == lock.at
    assert len(NFL.locks(5, schedule)) == 30  # CAR and KC on bye have no lock
    with pytest.raises(ValidationError):
        LineupLock(team_id=1, period=4, at=datetime(2026, 10, 4, 17, 0))  # naive


def test_tbd_games_make_their_locks_provisional() -> None:
    view = _view(PRO_SCHEDULE)
    _set_game(view, MNF_WEEK_5_GAME, startTimeTBD=True, validForLocking=False)
    schedule = load_schedule(view)
    game = game_for(schedule, team("BUF"), 5)
    assert game is not None and game.start_time_tbd and not game.valid_for_locking and is_provisional(game)
    assert NFL.lock_time(team("BUF"), 5, schedule) == MNF_WEEK_5  # the placeholder is still the best estimate
    provisional = {lock.team_id for lock in NFL.locks(5, schedule) if lock.provisional}
    assert provisional == {team("BUF"), team("LAR")}
    # Under a first-game lock only the opener's own status matters.
    assert not any(lock.provisional for lock in NFL.locks(5, schedule, lock_type=LockType.FIRSTGAME_SCORINGPERIOD))
    _set_game(view, TNF_WEEK_5_GAME, startTimeTBD=True)
    opener_tbd = load_schedule(view)
    assert all(lock.provisional for lock in NFL.locks(5, opener_tbd, lock_type=LockType.FIRSTGAME_SCORINGPERIOD))


def test_lock_type_comes_from_league_settings(schedule: ScheduleLike, ffl_ppr: LeagueSettings) -> None:
    assert ffl_ppr.lineup_lock_type is LockType.INDIVIDUAL_GAME
    assert NFL.lock_time(team("ATL"), 4, schedule, lock_type=ffl_ppr.lineup_lock_type) == MNF_WEEK_4
    view = _view(FFL_PPR)
    view["settings"]["rosterSettings"]["lineupLocktimeType"] = "FIRSTGAME_SCORINGPERIOD"  # ESPN's own name for it
    espn_named = parse_league_settings(view)
    assert espn_named.lineup_lock_type is LockType.FIRSTGAME_SCORINGPERIOD
    assert NFL.lock_time(team("ATL"), 4, schedule, lock_type=espn_named.lineup_lock_type) == TNF_WEEK_4
    view["settings"]["rosterSettings"]["lineupLocktimeType"] = "FIRST_GAME_OF_WEEK"  # the old guess: no league has it
    guessed = parse_league_settings(view)
    assert guessed.lineup_lock_type is LockType.UNKNOWN
    with pytest.raises(ValueError, match="UNKNOWN"):
        NFL.lock_time(team("ATL"), 4, schedule, lock_type=guessed.lineup_lock_type)


# --- scoring periods --------------------------------------------------------------------------------------------------


def test_period_window_spans_first_to_last_kickoff(schedule: ScheduleLike) -> None:
    window = NFL.period_window(4, schedule)
    assert window == PeriodWindow(
        period=4, first_start=TNF_WEEK_4, last_start=MNF_WEEK_4, end=MNF_WEEK_4 + NFL_GAME_DURATION
    )
    assert NFL.period_window(99, schedule) is None


def test_scoring_period_rolls_over_when_the_last_game_ends(schedule: ScheduleLike) -> None:
    assert NFL.scoring_period_at(datetime(2026, 9, 1, tzinfo=UTC), schedule) == 4  # before the first known game
    assert NFL.scoring_period_at(TNF_WEEK_4, schedule) == 4
    assert NFL.scoring_period_at(EARLY_WEEK_4 + timedelta(hours=1), schedule) == 4
    assert NFL.scoring_period_at(MNF_WEEK_4 + timedelta(hours=3), schedule) == 4  # Monday night still on
    assert NFL.scoring_period_at(MNF_WEEK_4 + NFL_GAME_DURATION, schedule) == 5
    assert NFL.scoring_period_at(TNF_WEEK_5 - timedelta(days=1), schedule) == 5  # mid-week: next lineup is week 5
    assert NFL.scoring_period_at(MNF_WEEK_5 + timedelta(hours=3), schedule) == 5
    assert NFL.scoring_period_at(MNF_WEEK_5 + NFL_GAME_DURATION, schedule) is None  # past the fixture's last game
    with pytest.raises(ValueError, match="aware"):
        NFL.scoring_period_at(datetime(2026, 10, 4), schedule)


def test_nfl_has_no_transaction_cutoff(schedule: ScheduleLike) -> None:
    assert NFL.transaction_cutoff(4, schedule) is None


# --- decision registry ------------------------------------------------------------------------------------------------


def _lineup() -> str:
    return "lineup"


def _waivers() -> str:
    return "waivers"


def test_register_and_lookup(fresh: DecisionRegistry) -> None:
    registration = fresh.register("nfl", "lineup", _lineup)
    assert registration == Registration(sport="nfl", kind="lineup", fn=_lineup)
    assert registration.key == ("nfl", "lineup") and registration.name == "nfl:lineup"
    assert registration.target == f"{__name__}._lineup"
    assert fresh.lookup("nfl", "lineup") is _lineup
    assert fresh.lookup("ffl", "lineup") is _lineup and fresh.lookup(Game.FFL, "lineup") is _lineup
    assert fresh.get("nfl", "lineup") == registration and fresh.get("nba", "lineup") is None
    assert ("nfl", "lineup") in fresh and ("ffl", "lineup") in fresh and (Game.FFL, "lineup") in fresh
    assert ("nba", "lineup") not in fresh and ("mlb", "lineup") not in fresh
    assert "nfl" not in fresh and ("nfl",) not in fresh and ("nfl", 1) not in fresh
    assert len(fresh) == 1 and list(fresh) == [registration]


def test_decorator_registers_and_returns_the_function(fresh: DecisionRegistry) -> None:
    @fresh.decision("nba", "lineup_daily")
    def daily(period: int) -> int:
        return period

    assert daily(3) == 3
    assert fresh.lookup("fba", "lineup_daily") is daily
    assert fresh.kinds("nba") == ("lineup_daily",) and fresh.kinds("nfl") == ()


def test_duplicate_registration_raises_naming_both_targets(fresh: DecisionRegistry) -> None:
    fresh.register("nfl", "lineup", _lineup)
    with pytest.raises(DuplicateDecisionError, match=rf"nfl:lineup is registered twice, by {__name__}._lineup and by"):
        fresh.register("ffl", "lineup", _waivers)
    assert fresh.lookup("nfl", "lineup") is _lineup  # the first binding stands
    fresh.register("nba", "lineup", _waivers)  # the same kind for the other sport is a different decision
    assert len(fresh) == 2


def test_unknown_decision_names_what_is_registered(fresh: DecisionRegistry) -> None:
    with pytest.raises(UnknownDecisionError, match="no 'lineup' decision is registered for nfl; registered: nothing"):
        fresh.lookup("nfl", "lineup")
    fresh.register("nfl", "lineup", _lineup)
    fresh.register("nfl", "waivers", _waivers)
    with pytest.raises(LookupError, match="no 'trades' decision is registered for nfl; registered: lineup, waivers"):
        fresh.lookup("nfl", "trades")
    with pytest.raises(UnknownDecisionError):
        fresh.unregister("nfl", "trades")


@pytest.mark.parametrize("kind", ["Lineup", "", "add-drop", "1x", " lineup", "lineup daily"])
def test_malformed_kinds_are_rejected(fresh: DecisionRegistry, kind: str) -> None:
    with pytest.raises(ValueError, match="lower-case identifier"):
        fresh.register("nfl", kind, _lineup)
    assert len(fresh) == 0


def test_unsupported_sports_and_non_callables_are_rejected(fresh: DecisionRegistry) -> None:
    with pytest.raises(ValueError, match="mlb"):
        fresh.register("mlb", "lineup", _lineup)
    with pytest.raises(TypeError, match="should be callable"):
        fresh.register("nfl", "lineup", 42)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mlb"):
        fresh.registered("mlb")
    assert normalize_sport("FBA") == "nba" and normalize_sport(Game.FFL) == "nfl" and normalize_sport("nfl") == "nfl"
    assert normalize_kind("lineup_daily") == "lineup_daily"


def test_registered_and_kinds_follow_registration_order(fresh: DecisionRegistry) -> None:
    waivers = fresh.register("nfl", "waivers", _waivers)
    daily = fresh.register("nba", "lineup_daily", _lineup)
    lineup = fresh.register("nfl", "lineup", _lineup)
    assert fresh.registered() == (waivers, daily, lineup)
    assert fresh.registered("nfl") == (waivers, lineup) and fresh.registered("fba") == (daily,)
    assert fresh.kinds("nfl") == ("waivers", "lineup")
    assert fresh.unregister("ffl", "waivers") == waivers
    assert fresh.registered("nfl") == (lineup,) and ("nfl", "waivers") not in fresh
    fresh.clear()
    assert len(fresh) == 0 and fresh.registered() == ()


def test_registration_target_for_callables_without_a_name() -> None:
    class Runner:
        def __call__(self) -> None:
            return None

    runner = Runner()
    registration = Registration(sport="nfl", kind="lineup", fn=runner)
    assert registration.target == repr(runner)


def test_module_level_functions_use_the_process_wide_registry() -> None:
    assert isinstance(decide_registry.registry, DecisionRegistry)
    probe_kind = "t7_probe"
    registered_before = decide_registry.registered()
    try:
        registration = decide_registry.register("nfl", probe_kind, _lineup)
        assert decide_registry.lookup("ffl", probe_kind) is _lineup
        assert decide_registry.get("nfl", probe_kind) == registration
        assert registration in decide_registry.registered("nfl") and registration in decide_registry.registered()
        assert ("nfl", probe_kind) in decide_registry.registry

        @decide_registry.decision("nba", probe_kind)
        def daily() -> str:
            return "daily"

        assert decide_registry.lookup("nba", probe_kind) is daily
        with pytest.raises(DuplicateDecisionError):
            decide_registry.register("nfl", probe_kind, _waivers)
    finally:
        for sport in ("nfl", "nba"):
            if (sport, probe_kind) in decide_registry.registry:
                decide_registry.unregister(sport, probe_kind)
    assert decide_registry.registered() == registered_before
