"""Backtest harness: projection MAE by position, lineup efficiency and start/sit regret (ROADMAP #33).

The numbers are worked by hand on a ten-player league over two weeks (every stat is a reception, one point in the
fixture league's PPR scoring), so each metric can be checked against arithmetic rather than against the code. The
shipped replay fixture is checked for shape and invariants, the loaders for their errors, and the command for exit
codes, output and isolation from the network and the real state database.
"""

from __future__ import annotations

import json
import shutil
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from fm.cli import app
from fm.espn.settings import LeagueSettings, load_league_settings
from fm.eval.backtest import (
    BASELINE,
    BacktestData,
    BacktestError,
    BacktestWeek,
    LineupReport,
    blend_source,
    load_fixture,
    load_store_data,
    run_backtest,
    with_blend,
)
from fm.model.projections import BLEND, BlendWeights
from fm.store import LeagueRow, PlayerRow, ProjectionRow, RosterEntryRow, Store, TeamRow

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
BACKTEST_FIXTURES = FIXTURES / "backtest"
NFL_FIXTURE = BACKTEST_FIXTURES / "nfl"
AS_OF = datetime(2026, 9, 1, 12, tzinfo=UTC)
runner = CliRunner()

# slot ids of the fixture league (QB, RB x2, WR x2, TE, FLEX, D/ST, K; bench 20, IR 21)
QB, RB, WR, TE, DST, K, FLEX, BENCH, IR = 0, 2, 4, 6, 16, 17, 23, 20, 21
ELIGIBLE = {
    "QB": [0, 20, 21],
    "RB": [2, 3, 23, 20, 21],
    "WR": [3, 4, 5, 23, 20, 21],
    "TE": [5, 6, 23, 20, 21],
    "K": [17, 20, 21],
    "D/ST": [16, 20, 21],
}
# espn id -> position: q, three running backs, three receivers, a tight end, a kicker and a defense
PLAYERS = {
    1: "QB",
    2: "RB",
    3: "RB",
    4: "RB",
    5: "WR",
    6: "WR",
    7: "WR",
    8: "TE",
    9: "K",
    -16021: "D/ST",
}
Q, R1, R2, R3, W1, W2, W3, T, KK, D = PLAYERS


@pytest.fixture(scope="module")
def settings() -> LeagueSettings:
    return load_league_settings(NFL_FIXTURE / "settings.json")


def player(espn_id: int, position: str) -> PlayerRow:
    return PlayerRow(
        sport="nfl",
        espn_id=espn_id,
        full_name=f"Player {espn_id}",
        default_position_id=None,
        position=position,
        eligible_slot_ids=ELIGIBLE[position],
        as_of=AS_OF,
    )


def line(receptions: float) -> dict[str, float]:
    """A stat line worth ``receptions`` points: the fixture league scores a reception at 1.0."""
    return {"REC": receptions}


def rows(source: str, week: Mapping[int, Mapping[str, float]], period: int) -> list[ProjectionRow]:
    return [
        ProjectionRow(
            sport="nfl",
            espn_id=espn_id,
            source=source,
            season=2026,
            scoring_period_id=period,
            stats=dict(stats),
            as_of=AS_OF,
        )
        for espn_id, stats in week.items()
    ]


# Week 1 actual points: the optimum is QB 1, RB 10 + 3, WR 5 + 4, TE 6, FLEX 2 (R1), D/ST 1, K 1 = 33.
ACTUAL_1 = {Q: 1, R1: 2, R2: 3, R3: 10, W1: 4, W2: 1, W3: 5, T: 6, KK: 1, D: 1}
# The manager benches R3 and starts W3 at FLEX: 1 + 2 + 3 + 4 + 1 + 6 + 5 + 1 + 1 = 24.
LINEUP_1 = {Q: QB, R1: RB, R2: RB, R3: BENCH, W1: WR, W2: WR, W3: FLEX, T: TE, KK: K, D: DST}
# Week 2: R2 is on bye (no actual) but the manager starts him, W3 is on IR having scored 50 (he cannot start), and
# with nine players for nine slots everyone else starts: the optimum is the manager's lineup, 17 points.
ACTUAL_2 = {Q: 1, R1: 2, R3: 4, W1: 3, W2: 3, W3: 50, T: 2, KK: 1, D: 1}
LINEUP_2 = {Q: QB, R1: RB, R2: RB, R3: FLEX, W1: WR, W2: WR, W3: IR, T: TE, KK: K, D: DST}
# "good" projects every line exactly; "bad" has R1 +3, R3 -9 and W2 +8 in week 1 and is exact in week 2
GOOD_1 = {espn_id: points for espn_id, points in ACTUAL_1.items()}
BAD_1 = {**GOOD_1, R1: 5, R3: 1, W2: 9}
GOOD_2 = {espn_id: points for espn_id, points in ACTUAL_2.items() if espn_id != W3} | {R2: 0}


def build(
    settings: LeagueSettings, sources: Mapping[str, tuple[Mapping[int, float], Mapping[int, float]]]
) -> BacktestData:
    """The two hand-worked weeks with each source's (week 1, week 2) projected points."""
    data = BacktestData(
        sport="nfl",
        season=2026,
        settings=settings,
        players={espn_id: player(espn_id, position) for espn_id, position in PLAYERS.items()},
        weeks=(
            BacktestWeek(1, LINEUP_1, {i: line(p) for i, p in ACTUAL_1.items()}),
            BacktestWeek(2, LINEUP_2, {i: line(p) for i, p in ACTUAL_2.items()}),
        ),
    )
    for name, (first, second) in sources.items():
        projected = {
            1: {i: line(p) for i, p in first.items()},
            2: {i: ({} if p == 0 else line(p)) for i, p in second.items()},
        }
        data = data.with_source(name, [row for period, week in projected.items() for row in rows(name, week, period)])
    return data


@pytest.fixture
def hand(settings: LeagueSettings) -> BacktestData:
    return build(settings, {"good": (GOOD_1, GOOD_2), "bad": (BAD_1, GOOD_2)})


# --- the numbers worked by hand ---------------------------------------------------------------------------------------


def test_baseline_lineup_efficiency_and_regret(hand: BacktestData) -> None:
    baseline = run_backtest(hand).baseline
    week_one, week_two = baseline.weeks
    assert baseline.label == BASELINE
    assert (week_one.points, week_one.optimal, week_one.regret) == (24.0, 33.0, 9.0)
    assert week_one.efficiency == pytest.approx(24 / 33)
    assert week_one.swaps == 1  # the optimum starts R3 and not W2: one starter differs
    # W3 scored 50 on IR: he is not eligible for the optimum, so week two's optimum is the manager's own lineup
    assert (week_two.points, week_two.optimal, week_two.regret, week_two.swaps) == (17.0, 17.0, 0.0, 0)
    assert baseline.points == 41.0
    assert baseline.optimal == 50.0
    assert baseline.efficiency == pytest.approx(41 / 50)
    assert baseline.mean_weekly_efficiency == pytest.approx((24 / 33 + 1.0) / 2)
    assert baseline.regret == 9.0
    assert baseline.regret_per_week == 4.5
    assert baseline.swaps == 1
    worst = baseline.worst_week
    assert worst is not None
    assert worst.period == 1


def test_a_source_that_projects_every_line_exactly_sets_the_optimal_lineup(hand: BacktestData) -> None:
    good = run_backtest(hand).source("good")
    assert good.overall.mae == 0.0
    assert good.overall.samples == 18  # ten in week one; eight in week two (R2 on bye: empty line; W3 on IR: no row)
    assert good.lineups.efficiency == pytest.approx(1.0)
    assert good.lineups.regret == 0.0
    assert good.lineups.swaps == 0


def test_projection_mae_by_position_with_bias_and_rmse(hand: BacktestData) -> None:
    bad = run_backtest(hand).source("bad")
    # week one: R1 +3, R3 -9 (running backs), W2 +8 (receivers); everything else exact; week two exact
    rb, wr = bad.by_position["RB"], bad.by_position["WR"]
    assert rb.samples == 5  # R1, R2, R3 in week one and R1, R3 in week two (R2's bye line is skipped)
    assert rb.mae == pytest.approx(12 / rb.samples)
    assert rb.bias == pytest.approx((3 - 9) / rb.samples)
    assert rb.rmse == pytest.approx(((9 + 81) / rb.samples) ** 0.5)
    assert wr.samples == 5  # W1, W2, W3 in week one and W1, W2 in week two (W3 is on IR)
    assert wr.mae == pytest.approx(8 / wr.samples)
    assert bad.by_position["QB"].mae == 0.0
    assert bad.overall.position == "ALL"
    assert bad.overall.samples == 18
    assert bad.overall.mae == pytest.approx(20 / 18)
    assert bad.mae == bad.overall.mae


def test_bad_source_loses_efficiency_and_regret_is_the_gap(hand: BacktestData) -> None:
    report = run_backtest(hand)
    bad = report.source("bad").lineups
    first = bad.weeks[0]
    # bad starts R1, R2 at RB, W2, W3 at WR, TE, FLEX W1: actual 2 + 3 + 1 + 5 + 6 + 4 + 1 + 1 + 1 = 24 against 33
    assert first.points == 24.0
    assert first.optimal == 33.0
    assert first.regret == 9.0
    assert first.swaps == 1
    assert bad.weeks[1].regret == 0.0
    assert bad.efficiency == pytest.approx(41 / 50)
    # every lineup is bounded by the same hindsight optimum
    assert {w.optimal for lineups in (report.baseline, bad) for w in lineups.weeks if w.period == 1} == {33.0}


def test_hindsight_optimum_leaves_ir_players_on_ir(hand: BacktestData) -> None:
    week_two = run_backtest(hand).baseline.weeks[1]
    assert W3 not in week_two.optimal_starters
    assert W3 not in week_two.starters
    assert week_two.optimal == 17.0  # not 67


def test_bye_projection_is_skipped_but_a_did_not_play_counts_as_a_miss(settings: LeagueSettings) -> None:
    """R2 on bye has an empty projected line (skipped); a player projected to play who has no actual scored 0."""
    base = build(settings, {"s": (GOOD_1, GOOD_2)})
    assert run_backtest(base).source("s").overall.samples == 18
    still_projected = {**GOOD_2, R2: 4}  # now R2 is expected to play 4 in week two but has no actual line: a miss of 4
    data = build(settings, {"s": (GOOD_1, still_projected)})
    result = run_backtest(data).source("s")
    assert result.overall.samples == 19
    assert result.by_position["RB"].bias > 0
    assert result.overall.mae == pytest.approx(4 / 19)


def test_restrict_to_common_scores_sources_on_the_same_player_weeks(settings: LeagueSettings) -> None:
    sparse_1 = {i: p for i, p in BAD_1.items() if i not in (R1, W2)}  # no line for two of the players "bad" misses
    data = build(settings, {"sparse": (sparse_1, GOOD_2), "bad": (BAD_1, GOOD_2)})
    own = run_backtest(data)
    assert own.source("sparse").overall.samples < own.source("bad").overall.samples
    assert not own.restricted_to_common
    common = run_backtest(data, restrict_to_common=True)
    assert common.restricted_to_common
    assert common.source("sparse").overall.samples == common.source("bad").overall.samples
    # on the shared player-weeks (16) "bad" keeps only its R3 miss of 9, whoever is scored
    assert common.source("bad").overall.samples == 16
    assert common.source("bad").overall.mae == pytest.approx(9 / 16)


def test_sources_are_pluggable_and_selectable(hand: BacktestData) -> None:
    assert hand.sources == ("good", "bad")
    only = run_backtest(hand, sources=["bad"])
    assert list(only.sources) == ["bad"]
    extra = hand.with_source("flat", rows("ignored", {Q: line(1)}, 1))
    assert extra.sources == ("good", "bad", "flat")
    assert {row.source for row in extra.projections["flat"]} == {"flat"}
    assert extra.without_source("flat").sources == ("good", "bad")
    assert hand.sources == ("good", "bad")  # data is immutable: with_source returned a copy
    with pytest.raises(BacktestError, match="no such projection source: nope"):
        run_backtest(hand, sources=["nope"])
    with pytest.raises(KeyError, match="was not evaluated"):
        only.source("good")


def test_restrict_to_holds_out_periods(hand: BacktestData) -> None:
    held = hand.restrict_to([2])
    assert held.periods == (2,)
    report = run_backtest(held)
    assert report.periods == (2,)
    assert report.source("bad").overall.mae == 0.0  # the bad source is exact in week two
    assert all(row.scoring_period_id == 2 for rows_ in held.projections.values() for row in rows_)


def test_blend_is_a_source_under_given_weights(settings: LeagueSettings) -> None:
    data = build(settings, {"espn": (GOOD_1, GOOD_2), "sleeper": (BAD_1, GOOD_2)})
    weights = BlendWeights.parse("[nfl.default]\nespn = 1.0\nsleeper = 1.0\n")
    blended = with_blend(data, weights)
    assert blended.sources == ("espn", "sleeper", BLEND)
    # an equal-weighted blend of an exact and a biased source is half as far off: R1 1.5, R3 4.5, W2 4.0
    report = run_backtest(blended)
    assert report.source(BLEND).overall.mae == pytest.approx(10 / 18)
    assert report.source("espn").overall.mae == 0.0
    # weight it entirely to the exact source and the blend is that source
    exact = BlendWeights.parse("[nfl.default]\nespn = 1.0\nsleeper = 0.0\n")
    assert run_backtest(with_blend(data, exact)).source(BLEND).overall.mae == 0.0
    renamed = with_blend(data, weights, sources=["espn", "sleeper"], name="blend_plus")
    assert "blend_plus" in renamed.sources
    assert {row.source for row in blend_source(data, weights, name="x")} == {"x"}
    with pytest.raises(BacktestError, match="no such projection source"):
        blend_source(data, weights, sources=["espn", "nope"])


# --- invalid data -----------------------------------------------------------------------------------------------------


def test_category_leagues_are_not_backtested() -> None:
    settings = load_league_settings(FIXTURES / "espn" / "fba_settings_9cat.json")
    with pytest.raises(BacktestError, match="points leagues"):
        BacktestData("nba", 2027, settings, {}, ())


def test_settings_must_match_the_sport(settings: LeagueSettings) -> None:
    with pytest.raises(BacktestError, match="not nba"):
        BacktestData("nba", 2026, settings, {}, ())


def test_data_is_validated(settings: LeagueSettings) -> None:
    players = {Q: player(Q, "QB")}
    with pytest.raises(BacktestError, match="not in the players table"):
        BacktestData("nfl", 2026, settings, players, (BacktestWeek(1, {R1: RB}, {}),))
    with pytest.raises(BacktestError, match="not a league slot"):
        BacktestData("nfl", 2026, settings, players, (BacktestWeek(1, {Q: 99}, {}),))
    with pytest.raises(BacktestError, match="unique and in period order"):
        BacktestData("nfl", 2026, settings, players, (BacktestWeek(2, {Q: QB}, {}), BacktestWeek(1, {Q: QB}, {})))
    base = BacktestData("nfl", 2026, settings, players, (BacktestWeek(1, {Q: QB}, {}),))
    with pytest.raises(BacktestError, match="unknown player"):
        base.with_source("s", rows("s", {R1: line(1)}, 1))
    wrong_season = ProjectionRow(
        sport="nfl", espn_id=Q, source="s", season=2025, scoring_period_id=1, stats=line(1), as_of=AS_OF
    )
    with pytest.raises(BacktestError, match="not nfl 2026 projected"):
        base.with_source("s", [wrong_season])


# --- the replay fixture -----------------------------------------------------------------------------------------------


def test_replay_fixture_loads() -> None:
    data = load_fixture(NFL_FIXTURE)
    assert (data.sport, data.season) == ("nfl", 2026)
    assert data.periods == (1, 2, 3, 4)
    assert set(data.sources) == {"espn", "sleeper"}
    assert len(data.players) == 14
    assert data.settings.is_points
    assert {week.lineup[4361370] for week in data.weeks[:2]} == {21}  # on IR in weeks one and two


def test_replay_fixture_report_invariants() -> None:
    data = load_fixture(NFL_FIXTURE)
    weights = BlendWeights.parse("[nfl.default]\nespn = 1.0\nsleeper = 1.0\n")
    report = run_backtest(with_blend(data, weights))
    assert list(report.sources) == ["espn", "sleeper", BLEND]
    assert report.positions == ("QB", "RB", "WR", "TE", "K", "D/ST")
    baseline = report.baseline
    assert baseline.efficiency is not None
    assert 0.5 < baseline.efficiency < 1.0
    assert baseline.regret > 0  # the fixture lineups include a start of a player on bye
    for name, source in report.sources.items():
        assert source.overall.samples > 0, name
        assert source.overall.mae > 0, name
        assert source.overall.rmse >= source.overall.mae, name
        for lineups in (source.lineups, baseline):
            for week, optimal in zip(lineups.weeks, baseline.weeks, strict=True):
                assert week.optimal == optimal.optimal
                assert week.points <= week.optimal + 1e-9
                assert week.regret == pytest.approx(week.optimal - week.points)
        assert source.lineups.efficiency is not None
        assert 0.5 < source.lineups.efficiency <= 1.0
    # the blend of two sources is no worse than the worse of them on every position
    for position in report.positions:
        errors = [report.source(n).by_position[position].mae for n in ("espn", "sleeper")]
        assert report.source(BLEND).by_position[position].mae <= max(errors) + 1e-9


def test_replay_fixture_bye_week_is_benched_by_projections_not_by_the_manager() -> None:
    data = load_fixture(NFL_FIXTURE)
    report = run_backtest(data)
    bye = 4241416  # Hubbard, on bye in week three, and the manager's FLEX
    week_three = next(week for week in report.baseline.weeks if week.period == 3)
    assert bye in week_three.starters
    for source in report.sources.values():
        assert bye not in next(week for week in source.lineups.weeks if week.period == 3).starters


def _copy_fixture(tmp_path: Path) -> Path:
    target = tmp_path / "nfl"
    shutil.copytree(NFL_FIXTURE, target)
    return target


def _edit(directory: Path, change: Callable[[dict[str, Any]], object]) -> None:
    path = directory / "backtest.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    change(document)
    path.write_text(json.dumps(document), encoding="utf-8")


def test_load_fixture_errors_name_the_problem(tmp_path: Path) -> None:
    with pytest.raises(BacktestError, match="cannot read backtest fixture"):
        load_fixture(tmp_path / "missing")
    directory = _copy_fixture(tmp_path)
    (directory / "backtest.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(BacktestError, match="backtest.json"):
        load_fixture(directory)
    directory = _copy_fixture(tmp_path / "format")
    _edit(directory, lambda document: document.update(format=99))
    with pytest.raises(BacktestError, match="format 99 is not supported"):
        load_fixture(directory)
    directory = _copy_fixture(tmp_path / "position")
    _edit(directory, lambda document: document["players"][0].update(position="XX"))
    with pytest.raises(BacktestError, match="unknown position"):
        load_fixture(directory)
    directory = _copy_fixture(tmp_path / "settings")
    (directory / "settings.json").unlink()
    with pytest.raises(BacktestError, match="settings.json"):
        load_fixture(directory)
    directory = _copy_fixture(tmp_path / "unknown_player")
    _edit(directory, lambda document: document["weeks"][0]["lineup"].update({"999": 20}))
    with pytest.raises(BacktestError, match="not in the players table"):
        load_fixture(directory)


# --- the state database -----------------------------------------------------------------------------------------------


def test_store_data_replays_played_periods(settings: LeagueSettings, hand: BacktestData, tmp_path: Path) -> None:
    with Store.open(tmp_path / "state.db") as store:
        league = store.leagues.upsert(
            LeagueRow(key="nfl", sport="nfl", espn_league_id=1234567, season=2026, team_id=1, as_of=AS_OF)
        )
        store.teams.upsert(TeamRow(league_id=league.row_id, team_id=1, name="Fixture Team 1", as_of=AS_OF))
        store.players.upsert_many(hand.players.values())
        for week in hand.weeks:
            store.rosters.replace(
                league.row_id,
                week.period,
                1,
                [
                    RosterEntryRow(
                        league_id=league.row_id,
                        scoring_period_id=week.period,
                        team_id=1,
                        espn_id=espn_id,
                        lineup_slot_id=slot_id,
                        as_of=AS_OF,
                    )
                    for espn_id, slot_id in week.lineup.items()
                ],
            )
            store.projections.upsert_many(
                ProjectionRow(
                    sport="nfl",
                    espn_id=espn_id,
                    source="espn",
                    kind="actual",
                    season=2026,
                    scoring_period_id=week.period,
                    stats=dict(stats),
                    as_of=AS_OF,
                )
                for espn_id, stats in week.actuals.items()
            )
        for source, source_rows in hand.projections.items():
            store.projections.upsert_many(row.model_copy(update={"source": source}) for row in source_rows)
        # a stored blend is ignored (it is blended fresh), and so is a period nobody has played yet
        store.projections.upsert(
            ProjectionRow(sport="nfl", espn_id=Q, source=BLEND, season=2026, scoring_period_id=1, stats={}, as_of=AS_OF)
        )
        store.rosters.replace(
            league.row_id,
            3,
            1,
            [
                RosterEntryRow(
                    league_id=league.row_id, scoring_period_id=3, team_id=1, espn_id=Q, lineup_slot_id=QB, as_of=AS_OF
                )
            ],
        )
        loaded = load_store_data(store, league, settings)
        assert loaded.periods == (1, 2)
        assert set(loaded.sources) == {"good", "bad"}
        assert load_store_data(store, league, settings, periods=[2]).periods == (2,)
        with pytest.raises(BacktestError, match="no played period"):
            load_store_data(store, league, settings, periods=[3])
    direct, replayed = run_backtest(hand), run_backtest(loaded)

    def summary(report: LineupReport) -> list[tuple[int, float, float, int]]:
        return [(w.period, w.points, w.optimal, w.swaps) for w in report.weeks]

    assert summary(replayed.baseline) == summary(direct.baseline)
    assert summary(replayed.source("bad").lineups) == summary(direct.source("bad").lineups)
    assert replayed.source("bad").by_position == direct.source("bad").by_position


# --- the command ------------------------------------------------------------------------------------------------------


def test_command_prints_the_three_metrics(tmp_path: Path) -> None:
    result = runner.invoke(app, ["backtest", "--sport", "nfl", "--fixtures", str(BACKTEST_FIXTURES)])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "Projection MAE by position" in out
    assert "Lineup efficiency" in out
    assert "start/sit regret" in out
    assert "Your lineup-efficiency baseline:" in out
    for position in ("QB", "RB", "WR", "TE", "K", "D/ST", "ALL"):
        assert position in out
    for name in ("espn", "sleeper", "blend"):
        assert name in out
    assert "%" in out
    assert not (tmp_path / "config" / "state.db").exists()  # --fixtures never opens the state database


def test_command_source_filter_and_common(tmp_path: Path) -> None:
    result = runner.invoke(
        app, ["backtest", "--fixtures", str(BACKTEST_FIXTURES), "--source", "espn", "--source", "blend", "--common"]
    )
    assert result.exit_code == 0, result.output
    assert "sleeper" not in result.output
    assert "every source projects" in result.output


def test_command_reports_missing_sport_and_bad_input() -> None:
    nba = runner.invoke(app, ["backtest", "--sport", "nba", "--fixtures", str(BACKTEST_FIXTURES)])
    assert nba.exit_code == 1
    assert "no nba backtest fixture" in nba.output
    bad = runner.invoke(app, ["backtest", "--sport", "mlb", "--fixtures", str(BACKTEST_FIXTURES)])
    assert bad.exit_code == 1
    assert "--sport must be one of" in bad.output
    unknown = runner.invoke(app, ["backtest", "--fixtures", str(BACKTEST_FIXTURES), "--source", "nope"])
    assert unknown.exit_code == 1
    assert "no such projection source" in unknown.output


def test_command_without_fixtures_needs_a_configured_league() -> None:
    result = runner.invoke(app, ["backtest", "--sport", "nfl"])
    assert result.exit_code == 1
    assert "error:" in result.output
