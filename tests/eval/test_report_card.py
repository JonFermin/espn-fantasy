"""The live report card (ROADMAP #47): lineup efficiency, bench points and pickup value on the decisions we made.

A small hand-made league is stored the way ``fm sync`` stores it (roster snapshots, ESPN actual lines, executions) and
the card is checked against numbers worked out by hand. The league's points come from the fixture league's scoring
items (a reception is 1.0). Offline.
"""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from fm import paths
from fm.commands import report as report_cmd
from fm.espn.settings import LeagueSettings, load_league_settings
from fm.eval.report_card import build_report_card, render_report_card
from fm.store import (
    ExecutionRow,
    LeagueRow,
    PlayerRow,
    ProjectionRow,
    ProposalRow,
    RosterEntryRow,
    Store,
    TeamRow,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SETTINGS = FIXTURES / "backtest" / "nfl" / "settings.json"
AS_OF = datetime(2026, 9, 1, 12, tzinfo=UTC)
NOW = datetime(2026, 10, 4, 15, tzinfo=UTC)
runner = CliRunner()

QB, RB, WR, TE, DST, K, FLEX, BENCH, IR = 0, 2, 4, 6, 16, 17, 23, 20, 21
ELIGIBLE = {
    "QB": [0, 20, 21],
    "RB": [2, 3, 23, 20, 21],
    "WR": [3, 4, 5, 23, 20, 21],
    "TE": [5, 6, 23, 20, 21],
    "K": [17, 20, 21],
    "D/ST": [16, 20, 21],
}
PLAYERS = {1: "QB", 2: "RB", 3: "RB", 4: "RB", 5: "WR", 6: "WR", 7: "WR", 8: "TE", 9: "K", -16021: "D/ST"}
Q, R1, R2, R3, W1, W2, W3, T, KK, D = PLAYERS
WAIVER_RB = 11  # an RB on the wire: dropped for R3's pickup, never on our roster, no players row

# Week 1: optimum is 33 (R3's 10 starts); the manager benched him and started W3 at FLEX for 24, so 9 were left on the
# bench and the bench scored 10. Week 2: nine players for nine slots, so the lineup is the optimum (17) and W3 (50) is
# on IR, not benched.
ACTUAL = {
    1: {Q: 1, R1: 2, R2: 3, R3: 10, W1: 4, W2: 1, W3: 5, T: 6, KK: 1, D: 1, WAIVER_RB: 3},
    2: {Q: 1, R1: 2, R3: 4, W1: 3, W2: 3, W3: 50, T: 2, KK: 1, D: 1, WAIVER_RB: 2},
}
LINEUP = {
    1: {Q: QB, R1: RB, R2: RB, R3: BENCH, W1: WR, W2: WR, W3: FLEX, T: TE, KK: K, D: DST},
    2: {Q: QB, R1: RB, R2: RB, R3: FLEX, W1: WR, W2: WR, W3: IR, T: TE, KK: K, D: DST},
}


@pytest.fixture(scope="module")
def settings() -> LeagueSettings:
    return load_league_settings(SETTINGS)


def seed(store: Store, *, actuals: bool = True) -> LeagueRow:
    league = store.leagues.upsert(
        LeagueRow(key="nfl", sport="nfl", espn_league_id=1234567, season=2026, team_id=1, as_of=AS_OF)
    )
    store.teams.upsert(TeamRow(league_id=league.row_id, team_id=1, name="Fixture Team 1", as_of=AS_OF))
    store.players.upsert_many(
        PlayerRow(
            sport="nfl",
            espn_id=espn_id,
            full_name=f"Player {espn_id}",
            default_position_id=None,
            position=position,
            eligible_slot_ids=ELIGIBLE[position],
            as_of=AS_OF,
        )
        for espn_id, position in PLAYERS.items()
    )
    for period, lineup in LINEUP.items():
        store.rosters.replace(
            league.row_id,
            period,
            1,
            [
                RosterEntryRow(
                    league_id=league.row_id,
                    scoring_period_id=period,
                    team_id=1,
                    espn_id=espn_id,
                    lineup_slot_id=slot_id,
                    as_of=AS_OF,
                )
                for espn_id, slot_id in lineup.items()
            ],
        )
        if actuals:
            store.projections.upsert_many(
                ProjectionRow(
                    sport="nfl",
                    espn_id=espn_id,
                    source="espn",
                    kind="actual",
                    season=2026,
                    scoring_period_id=period,
                    stats={"REC": float(points)},
                    as_of=AS_OF,
                )
                for espn_id, points in ACTUAL[period].items()
            )
    return league


def add_pickup(
    store: Store, league: LeagueRow, *, add: int, drop: int | None, period: int, status: str = "verified"
) -> int:
    proposal = store.proposals.insert(
        ProposalRow(
            league_id=league.row_id,
            kind="add_drop",
            status=status,  # type: ignore[arg-type]
            policy="approve",
            scoring_period_id=period,
            payload={"add_espn_id": add, "drop_espn_id": drop},
            created_by="test",
            created_at=AS_OF,
        )
    )
    assert proposal.row_id is not None
    store.executions.insert(
        ExecutionRow(
            proposal_id=proposal.row_id,
            mode="api",
            status=status,  # type: ignore[arg-type]
            started_at=AS_OF,
            finished_at=AS_OF + timedelta(minutes=1),
        )
    )
    return proposal.row_id


def test_card_scores_the_lineups_we_started_against_the_hindsight_optimum(
    settings: LeagueSettings, tmp_path: Path
) -> None:
    with Store.open(tmp_path / "state.db") as store:
        league = seed(store)
        card = build_report_card(store, league, now=NOW, settings=settings)
    first, second = card.weeks
    assert (first.period, first.points, first.optimal, first.bench, first.left, first.swaps) == (1, 24, 33, 10, 9, 1)
    assert (second.period, second.points, second.optimal, second.bench, second.left, second.swaps) == (
        2,
        17,
        17,
        0,
        0,
        0,
    )
    assert card.efficiency == pytest.approx(41 / 50)
    assert (card.left, card.bench, card.swaps) == (9, 10, 1)
    assert card.pickups == ()
    assert "no verified add or claim to value yet" in card.notes


def test_pickup_value_is_the_add_against_the_player_dropped_in_the_weeks_he_was_ours(
    settings: LeagueSettings, tmp_path: Path
) -> None:
    with Store.open(tmp_path / "state.db") as store:
        league = seed(store)
        add_pickup(store, league, add=R3, drop=WAIVER_RB, period=1)
        add_pickup(store, league, add=R3, drop=None, period=2, status="failed")  # a failed move is not valued
        card = build_report_card(store, league, now=NOW, settings=settings)
    (pickup,) = card.pickups
    assert pickup.periods == (1, 2)
    assert (pickup.gained, pickup.given_up, pickup.net) == (14, 5, 9)
    assert pickup.added == "Player 4" and pickup.dropped == f"player {WAIVER_RB}"
    assert card.pickup_net == 9


def test_a_pickup_with_no_played_week_yet_is_listed_but_not_counted(settings: LeagueSettings, tmp_path: Path) -> None:
    with Store.open(tmp_path / "state.db") as store:
        league = seed(store)
        add_pickup(store, league, add=WAIVER_RB, drop=None, period=3)  # claimed for period 3: not played, not ours yet
        card = build_report_card(store, league, now=NOW, settings=settings)
    (pickup,) = card.pickups
    assert not pickup.settled and card.pickup_net == 0
    text = "\n".join(render_report_card(card))
    assert "no played week yet" in text and "Pickups net" not in text


def test_render_lists_the_efficiency_the_bench_and_the_pickups(settings: LeagueSettings, tmp_path: Path) -> None:
    with Store.open(tmp_path / "state.db") as store:
        league = seed(store)
        add_pickup(store, league, add=R3, drop=WAIVER_RB, period=1)
        text = "\n".join(render_report_card(build_report_card(store, league, now=NOW, settings=settings)))
    assert text.startswith("### Report card")
    assert "Lineup efficiency 82% over 2 played weeks (41.0 of 50.0 points)" in text
    assert "9.0 points left on the bench" in text
    assert "| 1 | 24.0 | 33.0 | 73% | 10.0 | 9.0 | 1 |" in text
    assert "Player 4 for player 11: 14.0 vs 5.0 over periods 1-2, net 9.0" in text
    assert "Pickups net 9.0 points." in text


def test_missing_history_is_a_note_not_a_crash(settings: LeagueSettings, tmp_path: Path) -> None:
    with Store.open(tmp_path / "state.db") as store:
        league = seed(store, actuals=False)
        card = build_report_card(store, league, now=NOW, settings=settings)
        assert card.weeks == () and "no report card: " in card.notes[0] and "actuals" in card.notes[0]
        assert "Not available (see notes)." in render_report_card(card)
        unsynced = build_report_card(store, league, now=NOW)  # and no synced settings either
        assert unsynced.weeks == () and "settings are not synced" in unsynced.notes[0]


def test_a_category_league_is_a_note(tmp_path: Path) -> None:
    categories = load_league_settings(FIXTURES / "backtest" / "nba" / "settings.json")
    with Store.open(tmp_path / "state.db") as store:
        league = seed(store)
        card = build_report_card(store, league, now=NOW, settings=categories)
    assert card.weeks == () and card.notes


# --- the weekly report ---------------------------------------------------------------------------------------------


def cli() -> typer.Typer:
    root = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)
    root.callback()(lambda: None)
    report_cmd.register(root)
    return root


def test_the_weekly_report_gets_a_report_card_section(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    home = tmp_path / "home"
    shutil.copytree(
        FIXTURES / "home", home, ignore=shutil.ignore_patterns("snapshots", "build.py", "README.md", "__pycache__")
    )
    monkeypatch.setenv(paths.CONFIG_DIR_ENV, str(home))
    monkeypatch.setenv(paths.CACHE_DIR_ENV, str(home / "cache"))
    result = runner.invoke(cli(), ["report", "--no-notify", "--as-of", "2026-10-04T15:00Z"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    (path,) = (home / "cache" / "reports").glob("weekly-*.md")
    markdown = path.read_text(encoding="utf-8")
    # the fixture home has no replayable history: the section says so, the report still has its other sections
    assert "### Report card\n\nNot available (see notes)." in markdown
    assert "report card:" in markdown and "### Moves made" in markdown
