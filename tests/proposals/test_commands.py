"""``fm proposals list|approve|reject`` and ``fm pause`` / ``fm resume`` through the CLI.

The commands are registered on a private root instead of importing ``fm.cli`` so a half-written command module from
another task cannot break this file; ``tests/test_cli.py`` covers discovery of the real root. The state database lives
under the per-test ``FM_CONFIG_DIR`` (conftest), which is also what the commands open.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

from fm import paths
from fm.commands import proposals as proposals_cmd
from fm.config import Config, load_config
from fm.proposals import (
    AddDropPayload,
    LineupMove,
    LineupPayload,
    ProposalKind,
    approve,
    begin_execution,
    get_proposal,
    is_paused,
    pause,
    propose,
)
from fm.store import LeagueRow, Store

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
LINEUP = LineupPayload(moves=(LineupMove(espn_id=10, from_slot_id=20, to_slot_id=0),))

runner = CliRunner()


def _root() -> None:
    pass


def cli() -> typer.Typer:
    root = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)
    root.callback()(_root)
    proposals_cmd.register(root)
    return root


def run(*args: str, expect: int = 0) -> str:
    result = runner.invoke(cli(), list(args), catch_exceptions=False)
    assert result.exit_code == expect, result.output
    return result.output


@pytest.fixture
def config() -> Config:
    return load_config(FIXTURES / "config.sample.toml", environ={})


def seed(config: Config, *, deadline: datetime | None = None) -> tuple[int, int, int]:
    """Three proposals in the default state DB: a lineup move, an add/drop and an NBA lineup; returns their ids."""
    with Store.open() as store:
        nfl = store.leagues.upsert(league_row(config, "nfl"))
        nba = store.leagues.upsert(league_row(config, "nba"))
        first = propose(store, config, nfl, ProposalKind.LINEUP, LINEUP, created_by="t", deadline=deadline, now=NOW)
        second = propose(
            store,
            config,
            nfl,
            ProposalKind.ADD_DROP,
            AddDropPayload(add_espn_id=7, drop_espn_id=8),
            created_by="t",
            now=NOW,
        )
        third = propose(store, config, nba, ProposalKind.LINEUP, LINEUP, created_by="t", now=NOW)
    return first.row_id, second.row_id, third.row_id


def league_row(config: Config, key: str) -> LeagueRow:
    league = config.league(key)
    return LeagueRow(
        key=league.key,
        sport=league.sport,
        espn_league_id=league.espn_league_id,
        season=league.season,
        team_id=league.team_id,
        as_of=NOW,
    )


def test_list_exits_zero_on_an_empty_database() -> None:
    assert not paths.state_db().exists()
    assert run("proposals", "list").strip() == "no open proposals (proposed, approved, executing)"
    assert run("proposals", "list", "--all").strip() == "no proposals"
    assert paths.state_db().is_file()


def test_list_shows_open_proposals_with_league_kind_status_and_summary(config: Config) -> None:
    # The list sweeps with the real clock, so the deadline must be one that never arrives.
    deadline = datetime(2099, 1, 1, 17, 0, tzinfo=UTC)
    first, second, third = seed(config, deadline=deadline)
    out = run("proposals", "list")
    lines = out.strip().splitlines()
    assert len(lines) == 3
    assert lines[0].startswith(f"#{first}") and "nfl" in lines[0] and "lineup" in lines[0]
    assert "proposed" in lines[0] and "approve" in lines[0] and "due 2099-01-01 17:00Z" in lines[0]
    assert lines[0].endswith("10: slot 20 -> 0")
    assert lines[1].startswith(f"#{second}") and "add_drop" in lines[1] and lines[1].endswith("add 7, drop 8")
    assert "due -" in lines[1]
    assert lines[2].startswith(f"#{third}") and "nba" in lines[2]

    assert run("proposals", "list", "--league", "nba").strip().startswith(f"#{third}")
    assert run("proposals", "list", "-l", "nfl").count("\n") == 2
    assert "no league 'mlb' in the store; known: nba, nfl" in run("proposals", "list", "--league", "mlb", expect=1)


def test_approve_and_reject_change_status_and_record_who_decided(config: Config) -> None:
    first, second, _ = seed(config)
    assert run("proposals", "approve", str(first)).strip() == f"#{first} approved: lineup 10: slot 20 -> 0"
    assert (
        run("proposals", "reject", str(second), "--by", "jon").strip() == f"#{second} rejected: add_drop add 7, drop 8"
    )
    with Store.open() as store:
        approved, rejected = get_proposal(store, first), get_proposal(store, second)
    assert (approved.status, approved.decided_by) == ("approved", "cli") and approved.execution_token is not None
    assert (rejected.status, rejected.decided_by) == ("rejected", "jon")

    out = run("proposals", "list")
    assert "approved by cli" in out and f"#{second}" not in out  # rejected is no longer open
    everything = run("proposals", "list", "--all")
    assert "rejected by jon" in everything and everything.count("\n") == 3

    assert f"cannot approve proposal #{first}: it was approved by cli" in run(
        "proposals", "approve", str(first), expect=1
    )
    assert f"cannot reject proposal #{second}: it was rejected by jon" in run(
        "proposals", "reject", str(second), expect=1
    )


def test_an_expired_proposal_cannot_be_approved_and_is_listed_as_expired(config: Config) -> None:
    first, _, _ = seed(config, deadline=NOW + timedelta(minutes=1))  # long past by the time the CLI runs
    out = run("proposals", "approve", str(first), expect=1)
    assert f"error: cannot approve proposal #{first}: it expired at 2026-10-04 12:01 UTC" in out
    assert f"#{first}" not in run("proposals", "list")
    assert "expired by expiry" in run("proposals", "list", "--all")


def test_executing_proposals_are_listed_but_cannot_be_decided(config: Config) -> None:
    first, _, _ = seed(config)
    with Store.open() as store:
        approved = approve(store, first, decided_by="cli", now=NOW)
        assert approved.execution_token is not None
        begin_execution(store, first, approved.execution_token, now=NOW)
    assert "executing" in run("proposals", "list")
    assert f"cannot reject proposal #{first}: it is executing" in run("proposals", "reject", str(first), expect=1)


def test_unknown_or_invalid_ids_fail_cleanly() -> None:
    assert "error: no proposal #42" in run("proposals", "approve", "42", expect=1)
    assert "error: no proposal #42" in run("proposals", "reject", "42", expect=1)
    assert run("proposals", "approve", "0", expect=2)  # typer: min=1
    assert run("proposals", "approve", "abc", expect=2)


def test_pause_and_resume(config: Config) -> None:
    assert run("pause", "--reason", "incident").strip().startswith("paused since ")
    assert is_paused()
    assert run("pause").strip().endswith("(incident)")  # idempotent, keeps the first reason
    listing = run("proposals", "list")
    assert listing.startswith("PAUSED: paused since ") and "run fm resume" in listing

    resumed = run("resume").strip()
    assert resumed.startswith("resumed (paused since ") and resumed.endswith("(incident))")
    assert not is_paused()
    assert run("resume").strip() == "resumed"
    assert "PAUSED" not in run("proposals", "list")


def test_decisions_still_work_while_paused(config: Config) -> None:
    first, _, _ = seed(config)
    pause("incident", now=NOW)
    assert "approved" in run("proposals", "approve", str(first))
    run("resume")


def test_help_lists_every_command() -> None:
    out = run("proposals", "--help")
    assert "list" in out and "approve" in out and "reject" in out
    root = run("--help")
    assert "pause" in root and "resume" in root and "proposals" in root
