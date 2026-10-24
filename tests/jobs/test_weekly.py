"""The weekly window and the weekly report job (ROADMAP #62): :mod:`fm.jobs.deadlines`' Monday-morning window over
the fixture pro schedule and a synthetic one, and :mod:`fm.jobs.report`'s per-league writer, strategist inputs and
strategy section against the fixture home. The tick's own weekly run (the strategist, its drafts and the push) is in
``test_tick.py``. No network, no key: the market is a fake that answers nothing."""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from fm import paths
from fm.browser.fakes import FakeEspnApi
from fm.commands.advise import _config, _schedule
from fm.commands.trade import _read_matchups
from fm.config import Config, League, TradeFinder
from fm.decide.strategy import StrategyIdea, StrategyInputs, StrategyProposal, StrategyReport, StrategyResult
from fm.decide.trades import Acceptance, SideImpact, TradeEvaluation, TradeLegality, TradeSpec
from fm.espn.settings import LeagueSettings, load_league_settings
from fm.jobs import report as job
from fm.jobs.deadlines import (
    WEEKLY_AT,
    WEEKLY_DAY,
    RunWindow,
    WindowKind,
    matchup_bounds,
    run_windows,
    weekly_opens_at,
    weekly_window,
    weekly_windows,
)
from fm.sources.base import Fetched
from fm.sports.base import EASTERN, ScheduleLike
from fm.store import LeagueRow, ProposalRow, Store

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ESPN = FIXTURES / "espn"
HOME = FIXTURES / "home"
RECORDED = ESPN / "real" / "ffl"
LEAGUE = League.model_validate({"key": "nfl", "sport": "nfl", "espn_league_id": 1234567, "season": 2026, "team_id": 1})
MONDAY = datetime(2026, 10, 5, 13, 0, tzinfo=UTC)
"""Monday 09:00 ET of the fixture's week 4 (Monday Night Football day)."""
TURN = datetime(2026, 10, 6, 7, 0, tzinfo=UTC)
"""Tuesday 03:00 ET: week 4 turns into week 5."""
NOW = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)


def fixture_settings() -> LeagueSettings:
    return load_league_settings(ESPN / "ffl_settings_ppr.json", game="ffl")


def fixture_schedule() -> ScheduleLike:
    api = FakeEspnApi()
    api.serve_file("proTeamSchedules_wl", ESPN / "ffl_pro_schedule_2026.json")
    return api.client(LEAGUE.game, LEAGUE.espn_league_id, LEAGUE.season, session=None).pro_schedule().data


# --- a synthetic schedule: a week that ends on Monday night, then one that ends on Sunday -----------------------------


@dataclass(frozen=True)
class Game:
    date: datetime
    id: int | None = 1
    home_pro_team_id: int = 1
    away_pro_team_id: int = 2
    start_time_tbd: bool = False
    valid_for_locking: bool = True


class Schedule:
    def __init__(self, periods: dict[int, Sequence[datetime]]) -> None:
        self._periods = {period: tuple(Game(date) for date in dates) for period, dates in periods.items()}

    @property
    def scoring_periods(self) -> Sequence[int]:
        return tuple(sorted(self._periods))

    def games(self, scoring_period: int) -> Sequence[Game]:
        return self._periods.get(scoring_period, ())

    def games_for(self, pro_team_id: int, scoring_period: int) -> Sequence[Game]:
        return self.games(scoring_period)

    def idle_teams(self, scoring_period: int) -> Sequence[int]:
        return ()


def et(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=EASTERN).astimezone(UTC)


# --- the window -------------------------------------------------------------------------------------------------------


def test_weekly_opens_at_is_the_last_monday_morning_before_the_turn() -> None:
    assert (WEEKLY_DAY, WEEKLY_AT.hour) == (0, 9)  # Monday 09:00 ET, never a UTC clock
    # an NFL week: opens Tuesday 03:00 ET, turns the next Tuesday 03:00 ET: Monday Night Football's morning
    assert weekly_opens_at(et(2026, 9, 29, 3), et(2026, 10, 6, 3)) == MONDAY
    # an NBA week: opens Monday 03:00 ET and turns the next Monday 03:00 ET: its own Monday morning
    assert weekly_opens_at(et(2026, 10, 5, 3), et(2026, 10, 12, 3)) == MONDAY
    # two Mondays inside (the All-Star fortnight): the later one, still before the turn
    assert weekly_opens_at(et(2026, 10, 5, 3), et(2026, 10, 19, 3)) == MONDAY + timedelta(days=7)
    # no Monday morning inside (a week from Tuesday 03:00 ET to Monday 03:00 ET): the opening itself
    assert weekly_opens_at(et(2026, 10, 6, 3), et(2026, 10, 12, 3)) == et(2026, 10, 6, 3)


def test_the_fixture_weeks_window_runs_from_monday_morning_until_the_matchup_turns() -> None:
    settings, schedule = fixture_settings(), fixture_schedule()
    bounds = matchup_bounds(settings, schedule, 4)
    assert bounds is not None and bounds[0] == (4,) and bounds[2] == TURN  # the ffl calendar lists scoring periods
    window = weekly_window("nfl", settings, schedule, period=4)
    assert window is not None
    assert (window.kind, window.opens_at, window.closes_at, window.period) == (WindowKind.WEEKLY, MONDAY, TURN, 4)
    assert window.key == "nfl:weekly:2026-10-05T13:00Z" and window.decisions == frozenset()
    assert not window.runs("lineup") and not window.runs("trades")  # the tick's own, never a registered decision
    assert window.describe() == "nfl weekly from Mon 09:00 ET until the matchup turns Tue 03:00 ET"
    assert weekly_window("nfl", settings, schedule, period=3) is None  # no games in the fixture before week 4


def test_a_matchup_period_with_no_monday_morning_opens_its_window_when_the_period_opens() -> None:
    schedule = Schedule(
        {
            6: (et(2026, 10, 8, 20, 15), et(2026, 10, 11, 13), et(2026, 10, 12, 20, 15)),  # Thursday to Monday night
            7: (et(2026, 10, 15, 20, 15), et(2026, 10, 18, 13), et(2026, 10, 18, 20, 20)),  # Thursday to Sunday night
        }
    )
    window = weekly_window("nfl", fixture_settings(), schedule, period=7)
    assert window is not None
    assert window.opens_at == et(2026, 10, 13, 3) and window.closes_at == et(2026, 10, 19, 3)  # the whole period
    assert window.describe() == "nfl weekly from Tue 03:00 ET until the matchup turns Mon 03:00 ET"


def test_weekly_windows_cover_the_previous_ticks_matchup_period_too() -> None:
    settings, schedule = fixture_settings(), fixture_schedule()
    in_week_4, in_week_5 = MONDAY - timedelta(hours=1), TURN + timedelta(hours=1)
    both = weekly_windows("nfl", "nfl", settings, schedule, now=in_week_5, since=in_week_4)
    assert [w.key for w in both] == ["nfl:weekly:2026-10-05T13:00Z", "nfl:weekly:2026-10-12T13:00Z"]
    (same,) = weekly_windows("nfl", "nfl", settings, schedule, now=in_week_4, since=in_week_4 - timedelta(days=2))
    assert same.key == "nfl:weekly:2026-10-05T13:00Z"
    (first,) = weekly_windows("nfl", "nfl", settings, schedule, now=in_week_4)  # no previous tick
    assert first == same


def test_run_windows_adds_the_weekly_window_only_with_the_league_settings() -> None:
    settings, schedule = fixture_settings(), fixture_schedule()
    without = run_windows("nfl", "nfl", (), schedule=schedule, now=NOW)
    assert all(w.kind is not WindowKind.WEEKLY for w in without)
    with_settings = run_windows("nfl", "nfl", (), schedule=schedule, now=NOW, settings=settings)
    weekly = [w for w in with_settings if w.kind is WindowKind.WEEKLY]
    assert len(weekly) == 1 and weekly[0].opens_at == MONDAY
    assert isinstance(weekly[0], RunWindow) and len(with_settings) == len(without) + 1


# --- the report job ---------------------------------------------------------------------------------------------------


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    copy = tmp_path / "home"
    shutil.copytree(HOME, copy, ignore=shutil.ignore_patterns("snapshots", "build.py", "README.md", "__pycache__"))
    monkeypatch.setenv(paths.CONFIG_DIR_ENV, str(copy))
    monkeypatch.setenv(paths.CACHE_DIR_ENV, str(copy / "cache"))
    return copy


def league_row(store: Store) -> LeagueRow:
    row = store.leagues.by_key("nfl")
    assert row is not None
    return row


class NoMarket:
    """A market source that reaches nothing: every value comes back empty, as it would with the API down."""

    def __init__(self) -> None:
        self.calls = 0

    def market_values(self, settings: LeagueSettings, *, rank_type: str = "") -> Fetched[dict[int, Any]]:
        self.calls += 1
        return Fetched({}, NOW, "market", "fantasycalc", "nfl", degraded=True, warnings=("market down",))


def test_report_path_names_a_league_report_and_odds_key_pins_the_season(tmp_path: Path) -> None:
    assert job.report_path(NOW, tmp_path) == tmp_path / "weekly-2026-10-04.md"
    assert job.report_path(NOW, tmp_path, league="nfl") == tmp_path / "weekly-2026-10-04-nfl.md"
    row = LeagueRow(key="nfl", sport="nfl", espn_league_id=1, season=2026, team_id=1, as_of=NOW)
    assert job.odds_key(row) == "nfl-2026"
    assert job.report_body("# Weekly report\n\nAs of now.\n\n## nfl\n") == "As of now.\n\n## nfl"
    assert job.report_body("no title") == "no title"


def test_write_league_report_writes_the_leagues_file_with_the_sections_and_remembers_the_odds(home: Path) -> None:
    config = _config()
    with Store.open(home / "state.db") as store:
        row = league_row(store)
        loaded = _schedule(store, config.league("nfl"), path=RECORDED / "proTeamSchedules_wl.json")
        view, _ = _read_matchups(RECORDED / "mMatchup.json")
        written = job.write_league_report(
            store,
            config,
            row,
            now=NOW,
            schedule=loaded.schedule,
            matchups=view,
            sections=("### Weekly strategy\n\nA quiet week.",),
            notes=("nfl: strategy: no trade ideas",),
        )
    assert written.path == home / "cache" / "reports" / "weekly-2026-10-04-nfl.md" and written.path.is_file()
    markdown = written.path.read_text(encoding="utf-8")
    assert markdown == written.markdown and markdown.startswith("# Weekly report 2026-10-04")
    for heading in ("### Matchup outlook", "### Playoff odds", "### Moves made", "### Upcoming deadlines"):
        assert heading in markdown
    assert markdown.index("### Report card") < markdown.index("### Weekly strategy")  # ours is the last section
    assert "A quiet week." in markdown and "- nfl: strategy: no trade ideas" in markdown
    assert written.title == "Weekly report 2026-10-04 (nfl)" and written.body.startswith("As of ")
    assert written.report.matchup is not None and "Matchup period 4 vs Fixture Team 2" in markdown
    # the fixture home holds two rosters, so the season cannot be simulated: a note, no odds, nothing remembered
    assert written.report.odds is None and any("cannot be simulated" in note for note in written.report.notes)
    assert not (home / "cache" / "reports" / "odds-nfl-2026.json").exists()


def test_strategy_inputs_degrade_to_warnings_when_the_league_cannot_be_valued(tmp_path: Path) -> None:
    config = Config.model_validate({"league": [LEAGUE.model_dump(mode="json")]})
    with Store.open(tmp_path / "state.db") as store:
        row = store.leagues.upsert(
            LeagueRow(key="nfl", sport="nfl", espn_league_id=1234567, season=2026, team_id=1, as_of=NOW)
        )
        inputs = job.strategy_inputs(store, config, row, now=NOW, finder=TradeFinder(enabled=True), label="nfl (nfl)")
    assert inputs.empty and inputs.ideas == () and inputs.team_names == {} and inputs.league_id == row.row_id
    assert inputs.league_label == "nfl (nfl)"
    assert any(w.startswith("no trade context (") for w in inputs.warnings)
    assert "league settings are not synced; no category plan" in inputs.warnings


def test_strategy_inputs_read_the_fixture_home_without_the_network(home: Path) -> None:
    config = _config()
    market = NoMarket()
    with Store.open(home / "state.db") as store:
        row = league_row(store)
        loaded = _schedule(store, config.league("nfl"), path=RECORDED / "proTeamSchedules_wl.json")
        view, _ = _read_matchups(RECORDED / "mMatchup.json")
        common: dict[str, Any] = {
            "now": NOW,
            "label": "nfl",
            "schedule": loaded.schedule,
            "matchups": view,
            "period": 4,
        }
        off = job.strategy_inputs(store, config, row, finder=TradeFinder(enabled=False), market=market, **common)
        reads_with_finder_off = market.calls
        on = job.strategy_inputs(store, config, row, finder=TradeFinder(enabled=True), market=market, **common)
    assert off.ideas == () and off.plan is None  # a points league has no categories to plan, and no warning says so
    assert "the trade finder is off ([league.trade_finder] enabled = false): no trade ideas" in off.warnings
    assert off.team_names[1] == "Fixture Team 1" and reads_with_finder_off == 0  # no deals wanted, no market read
    assert all("category" not in w or "plan" not in w for w in off.warnings)
    assert market.calls == 1 and "market down" in on.warnings  # the finder's market, as the trade finder uses it
    assert all(isinstance(idea, StrategyIdea) and idea.key.startswith("nfl:") for idea in on.ideas)
    assert "the trade finder is off" not in " ".join(on.warnings)


# --- the strategy section ---------------------------------------------------------------------------------------------


def evaluation(team: int, give: int, get: int, score: float | None = 0.03) -> TradeEvaluation:
    spec = TradeSpec(team, (give,), (get,))
    ours = SideImpact(1, "us", spec.give, spec.get, 100.0, 102.4)
    theirs = SideImpact(team, "them", spec.get, spec.give, 90.0, 92.0)
    return TradeEvaluation(
        spec,
        ours,
        theirs,
        TradeLegality((), (), ()),
        Acceptance(0.7, 120.0, 100.0, 0.1, 0.5),
        "a",
        "b",
        score,
        "ros",
        "pts",
    )


def idea(team: int, give: int, get: int) -> StrategyIdea:
    ev = evaluation(team, give, get)
    return StrategyIdea(ev, f"nfl:{ev.spec.key}", {}, "Give.", 4)


def row(row_id: int) -> ProposalRow:
    return ProposalRow(
        id=row_id,
        league_id=1,
        kind="trade_propose",
        status="proposed",
        policy="approve",
        payload={},
        created_by="advisor.strategist",
        created_at=NOW,
    )


def test_render_strategy_lists_the_note_its_source_and_each_targets_fate() -> None:
    inputs = StrategyInputs("nfl", names={10: "Our Back", 20: "Their Back", 11: "Our End"}, team_names={2: "Rivals"})
    first, second, third = idea(2, 10, 20), idea(3, 11, 21), idea(4, 10, 22)
    report = StrategyReport(
        "nfl",
        "A close week.",
        ("Hold the backs.",),
        ("BLK",),
        ("PTS", "REB"),
        (first, second, third),
        {first.key: "Lifts the floor."},
        "claude",
    )
    result = StrategyResult(
        report,
        (
            StrategyProposal(first, row(12)),
            StrategyProposal(second, row(9), existing=True),
            StrategyProposal(third, blocked="one offer a team: team 4 already has one"),
        ),
    )
    text = job.render_strategy(inputs, result)
    lines = text.splitlines()
    assert lines[:5] == [
        "### Weekly strategy",
        "",
        "A close week.",
        "",
        "Note by Claude; the numbers and the targets are the engine's.",
    ]
    assert "- Hold the backs." in lines and "Contest PTS, REB; punt BLK." in lines
    assert (
        "- give Our Back to Rivals for Their Back: our rest-of-season value +2.4 pts; P(accept) 70%: "
        "drafted as proposal #12, awaiting approval. Lifts the floor." in lines
    )
    assert any(line.endswith("already open as proposal #9.") and "Our End" in line for line in lines)
    assert any(line.endswith("not drafted (one offer a team: team 4 already has one).") for line in lines)

    template = job.render_strategy(
        inputs, StrategyResult(StrategyReport("nfl", "Nothing to plan.", (), (), (), (), {}, "template", "no key"), ())
    )
    assert "Template note (no key)." in template and template.rstrip().endswith("Trade targets:\n\nNone.")
    fallback = StrategyReport("nfl", "Nothing.", (), (), (), (), {}, "fallback", "blocked: budget")
    assert "Template note; Claude could not be used (blocked: budget)." in job.render_strategy(
        inputs, StrategyResult(fallback, ())
    )
    dry = StrategyResult(
        fallback, (StrategyProposal(first, dry=True), StrategyProposal(second, blocked="no", dry=True))
    )
    assert "would be drafted." in job.render_strategy(inputs, dry) and "would be refused (no)." in job.render_strategy(
        inputs, dry
    )
