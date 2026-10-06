"""``fm status``, ``fm lineup`` and ``fm waivers`` against the fixture home (ROADMAP #26).

Each test copies ``tests/fixtures/home`` (config, state DB, captured views; see its README) into its temp dirs, so the
commands may store proposals without touching the committed fixture, and pins the clock with ``--as-of`` so the
output is reproducible: the snapshots under ``tests/fixtures/home/snapshots/`` are compared verbatim
(``FM_UPDATE_SNAPSHOTS=1`` rewrites them). The commands are registered on a private root, as the proposals CLI tests
do; ``tests/test_cli.py`` covers discovery of the real root. No network and no browser: a missing pro schedule reaches
``load_session``, which refuses before any browser starts because the home has no profile.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType

import pytest
import typer
from typer.testing import CliRunner

from fm import paths
from fm.commands import advise
from fm.proposals import pause
from fm.store import Store

FIXTURES = Path(__file__).resolve().parent / "fixtures"
HOME = FIXTURES / "home"
SNAPSHOTS = HOME / "snapshots"
SCHEDULE = FIXTURES / "sports" / "ffl_pro_schedule_2026.json"
AS_OF = "2026-10-04T15:00Z"  # Sunday 11 a.m. ET of week 4: Thursday night played, the early games not started
WEEK_OVER = "2026-10-06T18:00Z"  # Tuesday: every game of week 4 is over and the waiver run has passed
UPDATE = "FM_UPDATE_SNAPSHOTS"

runner = CliRunner()


def _root() -> None:
    pass


def cli() -> typer.Typer:
    root = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)
    root.callback()(_root)
    advise.register(root)
    return root


def run(*args: str, expect: int = 0) -> str:
    result = runner.invoke(cli(), list(args), catch_exceptions=False)
    assert result.exit_code == expect, result.output
    return result.output


def snapshot(name: str, output: str) -> None:
    """Compare ``output`` with ``snapshots/<name>.txt``; with ``FM_UPDATE_SNAPSHOTS=1`` write it instead."""
    path = SNAPSHOTS / f"{name}.txt"
    if os.environ.get(UPDATE):
        SNAPSHOTS.mkdir(exist_ok=True)
        path.write_text(output, encoding="utf-8", newline="\n")
    expected = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    assert output == expected, f"{path} differs; rerun with {UPDATE}=1 to accept the new output"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private copy of the fixture home, with FM_CONFIG_DIR and FM_CACHE_DIR pointing into it."""
    copy = tmp_path / "home"
    shutil.copytree(HOME, copy, ignore=shutil.ignore_patterns("snapshots", "build.py", "README.md", "__pycache__"))
    monkeypatch.setenv(paths.CONFIG_DIR_ENV, str(copy))
    monkeypatch.setenv(paths.CACHE_DIR_ENV, str(copy / "cache"))
    return copy


def proposals(home: Path) -> list[tuple[str, str, str]]:
    with Store.open(home / "state.db") as store:
        return [(row.kind, row.status, row.created_by) for row in store.proposals.find()]


# --- snapshots --------------------------------------------------------------------------------------------------------


def test_status_snapshot(home: Path) -> None:
    output = run("status")
    snapshot("status", output)
    assert "Fixture League (PPR)" in output and "roster for scoring period 4" in output
    assert "pro schedule: captured 2026-10-04 15:00Z, periods 4-5" in output
    assert "open proposals: none" in output


def test_lineup_snapshot_with_an_opponent(home: Path) -> None:
    output = run("lineup", "--as-of", AS_OF, "--opponent", "2")
    snapshot("lineup", output)
    assert "win probability" in output and "expected points" in output
    assert "bench_inactive: proposed as #1 (policy auto)" in output
    assert "lineup: proposed as #2 (policy approve)" in output
    assert "due 2026-10-05 00:20Z" in output  # the lineup draft's deadline: the earliest lock among its moves
    assert "Tyler Allgeier (RB) BE -> RB/WR/TE: 5.1 proj x 1.00 = 5.1 exp" in output
    assert proposals(home) == [
        ("bench_inactive", "proposed", advise.LINEUP_CREATED_BY),
        ("lineup", "proposed", advise.LINEUP_CREATED_BY),
    ]


def test_waivers_snapshot(home: Path) -> None:
    output = run("waivers", "--as-of", AS_OF)
    snapshot("waivers", output)
    assert "waiver: proposed as #1 (policy approve)" in output
    assert "Claim Jaylen Warren (RB) off waivers: +83.7 rest-of-season points from period 5; bid $17" in output
    assert "add_drop: proposed as #2 (policy approve)" in output
    assert "top 10 of 18 ranked pairs" in output and "replacement level: " in output
    assert proposals(home) == [("waiver", "proposed", "decide.waivers"), ("add_drop", "proposed", "decide.waivers")]


def test_status_lists_the_open_proposals_with_their_numbers(home: Path) -> None:
    run("lineup", "--as-of", AS_OF)
    run("waivers", "--as-of", AS_OF)
    output = run("status")
    snapshot("status_with_proposals", output)
    assert "open proposals: 4 (fm proposals list)" in output
    assert "#3 waiver (proposed, policy approve), due 2026-10-05 16:40Z: Claim Jaylen Warren (RB)" in output


# --- verdicts ---------------------------------------------------------------------------------------------------------


def test_a_rerun_finds_the_proposals_already_open(home: Path) -> None:
    run("lineup", "--as-of", AS_OF)
    output = run("lineup", "--as-of", AS_OF)
    assert "bench_inactive: already open as #1 (proposed, policy auto)" in output
    assert "lineup: already open as #2 (proposed, policy approve)" in output
    assert len(proposals(home)) == 2
    run("waivers", "--as-of", AS_OF)
    again = run("waivers", "--as-of", AS_OF)
    assert "no move worth proposing" in again  # both moves are pending, so nothing new is worth it
    assert "warning: nfl: 2 players in open proposals were left out of new moves" in again
    assert len(proposals(home)) == 4


def test_dry_run_shows_the_verdict_and_stores_nothing(home: Path) -> None:
    lineup = run("lineup", "--as-of", AS_OF, "--dry-run")
    assert "bench_inactive: would be proposed (policy auto); --dry-run stored nothing" in lineup
    waivers = run("waivers", "--as-of", AS_OF, "--dry-run")
    assert "waiver: would be proposed (policy approve); --dry-run stored nothing" in waivers
    assert proposals(home) == []


def test_policy_refusals_are_verdicts_not_tracebacks(home: Path) -> None:
    config = home / "config.toml"
    config.write_text(config.read_text(encoding="utf-8").replace('lineup = "approve"', 'lineup = "off"'), "utf-8")
    output = run("lineup", "--as-of", AS_OF)
    assert "lineup: blocked: lineup is off for league 'nfl' ([league.policy] in config.toml)" in output
    assert proposals(home) == [("bench_inactive", "proposed", advise.LINEUP_CREATED_BY)]
    pause("maintenance")
    paused = run("waivers", "--as-of", AS_OF)
    assert "waiver: blocked: paused since" in paused and "(maintenance); run fm resume" in paused
    assert "add_drop: blocked: paused since" in paused
    assert len(proposals(home)) == 1
    assert "PAUSED: paused since" in run("status")


def test_waivers_name_every_skipped_candidate_once_the_week_is_over(home: Path) -> None:
    output = run("waivers", "--as-of", WEEK_OVER, "--dry-run", "--top", "0")
    assert "no move worth proposing" in output
    assert "3 wire candidates skipped: his game this period has started" in output
    assert "1 wire candidates skipped: the waiver run passed after the last sync; run fm sync" in output
    assert proposals(home) == []


def test_lineup_after_the_week_reports_a_locked_lineup(home: Path) -> None:
    output = run("lineup", "--as-of", WEEK_OVER)
    assert "lineup stands: nothing to move" in output
    assert "1 active slot(s) stay empty" in output
    assert output.count("locked") >= 10 and "proposed as" not in output
    assert proposals(home) == []


def test_the_wall_clock_is_the_default(home: Path) -> None:
    before = datetime.now(UTC)
    status = run("status")
    lineup = run("lineup")
    waivers = run("waivers", "--dry-run", "--top", "0")
    assert "Fixture League (PPR)" in status and "Fixture League (PPR)" in lineup and "Fixture League (PPR)" in waivers
    assert f"as of {before:%Y-%m-%d}" in lineup or f"as of {datetime.now(UTC):%Y-%m-%d}" in lineup


# --- the pro schedule and the cache -----------------------------------------------------------------------------------


def test_without_the_cache_the_commands_exit_zero_and_say_why(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv(paths.CACHE_DIR_ENV, str(tmp_path / "empty-cache"))
    lineup = run("lineup", "--as-of", AS_OF)
    assert lineup.startswith("nfl: no lineup planned\n")
    assert "warning: nfl: no pro schedule for ffl 2026 is captured under" in lineup
    assert "(FM_CACHE_DIR) and there is no ESPN session to fetch one (no browser profile yet; run `fm login`)" in lineup
    assert "pass --schedule FILE" in lineup
    waivers = run("waivers", "--as-of", AS_OF, "--dry-run", "--top", "0")
    assert "wire of 0" in waivers and "is gone from the cache; run fm sync" in waivers
    assert "the captured pro schedule" in waivers and "is gone from the cache; fetching it again" in waivers
    assert "pro schedule: none captured" in run("status")
    assert proposals(home) == []


def test_a_schedule_file_stands_in_for_the_cache(home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(paths.CACHE_DIR_ENV, str(tmp_path / "empty-cache"))
    output = run("lineup", "--as-of", AS_OF, "--schedule", str(SCHEDULE), "--dry-run")
    assert "recommended:    76.6 expected points" in output
    assert "no lineup planned" not in output


def test_a_stale_sync_is_reported_not_planned_around(home: Path) -> None:
    output = run("lineup", "--as-of", "2026-10-08T16:00Z", "--dry-run")  # Thursday of week 5
    assert "warning: nfl: roster synced for scoring period 4 but period 5 is current; run fm sync" in output
    assert "scoring period 4 lineup" in output


# --- errors and edge cases --------------------------------------------------------------------------------------------


def test_before_the_first_sync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    shutil.copy(HOME / "config.toml", fresh / "config.toml")
    monkeypatch.setenv(paths.CONFIG_DIR_ENV, str(fresh))
    assert "nfl: NFL 2026, ESPN league 1234567: not synced; run fm sync" in run("status")
    assert run("lineup") == "nfl: not synced; run fm sync\n"
    assert run("waivers") == "nfl: not synced; run fm sync\n"


def test_an_unknown_league_key_fails(home: Path) -> None:
    for command in ("status", "lineup", "waivers"):
        output = run(command, "--league", "nba", expect=1)
        assert "error: no league 'nba' in config.toml; known: nfl" in output


def test_a_bad_as_of_fails(home: Path) -> None:
    assert "error: --as-of 'yesterday' is not an ISO 8601 timestamp" in run("lineup", "--as-of", "yesterday", expect=1)


def test_opponent_needs_one_league(home: Path) -> None:
    config = home / "config.toml"
    extra = '\n[[league]]\nkey = "other"\nsport = "nfl"\nespn_league_id = 7654321\nseason = 2026\nteam_id = 3\n'
    config.write_text(config.read_text(encoding="utf-8") + extra, "utf-8")
    refused = run("lineup", "--opponent", "2", expect=1)
    assert "error: --opponent names one league's opponent; add --league KEY" in refused
    output = run("lineup", "--league", "nfl", "--opponent", "2", "--as-of", AS_OF, "--dry-run")
    assert "opponent: Fixture Team 2" in output
    both = run("lineup", "--as-of", AS_OF, "--dry-run")
    assert "other: not synced; run fm sync" in both and "nfl: Fixture League (PPR)" in both


def test_without_config_the_commands_fail_cleanly(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(paths.CONFIG_DIR_ENV, str(tmp_path / "nothing"))
    assert run("status", expect=1).startswith("error: ")


# --- the committed home matches its builder ---------------------------------------------------------------------------


def _load_builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("fixture_home_build", HOME / "build.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_committed_home_is_what_build_py_makes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``build.py`` into a temp dir renders exactly the committed snapshots, so ``state.db`` has not drifted."""
    rebuilt = tmp_path / "rebuilt"
    _load_builder().build(rebuilt)
    monkeypatch.setenv(paths.CONFIG_DIR_ENV, str(rebuilt))
    monkeypatch.setenv(paths.CACHE_DIR_ENV, str(rebuilt / "cache"))
    skipped = {"snapshots", "__pycache__", "build.py", "README.md"}
    assert sorted(p.relative_to(rebuilt) for p in rebuilt.rglob("*") if p.is_file()) == sorted(
        p.relative_to(HOME) for p in HOME.rglob("*") if p.is_file() and not skipped & set(p.relative_to(HOME).parts)
    )
    snapshot("status", run("status"))
    snapshot("lineup", run("lineup", "--as-of", AS_OF, "--opponent", "2"))
