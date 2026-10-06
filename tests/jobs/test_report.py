"""The weekly report (ROADMAP #46): ``fm report`` and :mod:`fm.jobs.report` against the fixture home.

Each test copies ``tests/fixtures/home`` into its temp dirs (the report writes next to the cache dir) and pins the
clock with ``--as-of``. Inputs are recorded views: ``--fixtures tests/fixtures/espn/real/ffl`` supplies the pro
schedule and ``mMatchup``. The phone is a fake channel; with none configured the real ``open_channel`` refuses before
any network call. No network and no browser.
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
from fm.jobs import report as job
from fm.notify import Message, NotifyError
from fm.store import ExecutionRow, ProposalRow, Store

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
HOME = FIXTURES / "home"
RECORDED = FIXTURES / "espn" / "real" / "ffl"
AS_OF = "2026-10-04T15:00Z"
NOW = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)

runner = CliRunner()


class FakeChannel:
    name = "fake"

    def __init__(self) -> None:
        self.sent: list[Message] = []

    def send(self, message: Message) -> None:
        self.sent.append(message)


def cli() -> typer.Typer:
    root = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)
    root.callback()(lambda: None)
    report_cmd.register(root)
    return root


def run(*args: str, expect: int = 0) -> str:
    result = runner.invoke(cli(), ["report", *args], catch_exceptions=False)
    assert result.exit_code == expect, result.output
    return result.output


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    copy = tmp_path / "home"
    shutil.copytree(HOME, copy, ignore=shutil.ignore_patterns("snapshots", "build.py", "README.md", "__pycache__"))
    monkeypatch.setenv(paths.CONFIG_DIR_ENV, str(copy))
    monkeypatch.setenv(paths.CACHE_DIR_ENV, str(copy / "cache"))
    return copy


def written(home: Path) -> str:
    files = sorted((home / "cache" / "reports").glob("weekly-*.md"))
    assert len(files) == 1, files
    return files[0].read_text(encoding="utf-8")


def add_move(home: Path, *, finished: datetime, status: str = "verified", execution: str = "verified") -> None:
    with Store.open(home / "state.db") as store:
        league = store.leagues.by_key("nfl")
        assert league is not None
        proposal = store.proposals.insert(
            ProposalRow(
                league_id=league.row_id,
                kind="add_drop",
                status=status,  # type: ignore[arg-type]
                policy="approve",
                payload={"add_espn_id": 4361370, "drop_espn_id": 4038815},
                created_by="test",
                created_at=finished - timedelta(hours=1),
            )
        )
        assert proposal.row_id is not None
        store.executions.insert(
            ExecutionRow(
                proposal_id=proposal.row_id,
                mode="api",
                status=execution,  # type: ignore[arg-type]
                started_at=finished - timedelta(minutes=1),
                finished_at=finished,
            )
        )


# --- the command ------------------------------------------------------------------------------------------------------


def test_report_from_recorded_views_writes_all_sections(home: Path) -> None:
    output = run("--no-notify", "--as-of", AS_OF, "--fixtures", str(RECORDED))
    markdown = written(home)
    assert output.startswith("wrote ") and str(home / "cache" / "reports") in output
    assert "phone" not in output
    for heading in ("# Weekly report 2026-10-04", "### Matchup outlook", "### Playoff odds", "### Moves made"):
        assert heading in markdown
    assert "### Upcoming deadlines" in markdown
    assert "Matchup period 4 vs Fixture Team 2" in markdown and "to win" in markdown
    assert "lineup lock" in markdown and "waiver run" in markdown  # from the league's settings, not constants
    assert "Moves made\n\nNone." in markdown


def test_report_degrades_without_matchups(home: Path) -> None:
    run("--no-notify", "--as-of", AS_OF)  # no --fixtures: the captured schedule, no mMatchup, no ESPN session
    markdown = written(home)
    assert "no mMatchup schedule" in markdown and "matchup outlook and playoff odds are skipped" in markdown
    assert "### Matchup outlook\n\nNot available" in markdown
    assert "waiver run" in markdown  # the cached schedule still gives deadlines


def test_out_writes_where_asked(home: Path, tmp_path: Path) -> None:
    target = tmp_path / "elsewhere" / "r.md"
    output = run("--no-notify", "--as-of", AS_OF, "--fixtures", str(RECORDED), "--out", str(target))
    assert str(target) in output and target.read_text(encoding="utf-8").startswith("# Weekly report")
    assert not (home / "cache" / "reports" / "weekly-2026-10-04.md").exists()


def test_unknown_league_is_an_error(home: Path) -> None:
    assert "no league" in run("--league", "nope", "--no-notify", expect=1)


def test_moves_made_lists_finished_executions_in_the_window(home: Path) -> None:
    add_move(home, finished=NOW - timedelta(days=2))
    add_move(home, finished=NOW - timedelta(days=20))  # outside the window
    add_move(home, finished=NOW - timedelta(hours=3), status="failed", execution="failed")
    add_move(home, finished=NOW - timedelta(hours=1), execution="dry_run")  # a dry run is not a move
    run("--no-notify", "--as-of", AS_OF, "--fixtures", str(RECORDED))
    moves = written(home).split("### Moves made")[1].split("###")[0]
    assert moves.count("add_drop") == 2
    assert "add_drop verified" in moves and "add_drop failed" in moves
    assert "add " in moves and "drop " in moves


# --- the phone --------------------------------------------------------------------------------------------------------


def test_report_is_pushed_to_the_phone(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    channel = FakeChannel()
    monkeypatch.setattr(report_cmd, "open_channel", lambda config: channel)
    output = run("--as-of", AS_OF, "--fixtures", str(RECORDED))
    assert "phone: sent via telegram" in output
    (message,) = channel.sent
    assert message.priority == "low" and message.title.startswith("Weekly report 2026-10-04")
    assert message.body == written(home).split("\n", 2)[2].strip()


def test_unconfigured_phone_is_a_note_not_a_failure(home: Path) -> None:
    output = run("--as-of", AS_OF, "--fixtures", str(RECORDED))  # no .env secrets: open_channel refuses
    assert "phone: not sent" in output
    assert written(home).startswith("# Weekly report")


def test_phone_failure_is_a_note(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(config: object) -> FakeChannel:
        raise NotifyError("no chat id")

    monkeypatch.setattr(report_cmd, "open_channel", refuse)
    assert "phone: not sent (no chat id)" in run("--as-of", AS_OF, "--fixtures", str(RECORDED))


def test_odds_history_trends_against_last_week_not_a_rerun_or_a_backdated_run(tmp_path: Path) -> None:
    last_week = job.OddsSnapshot(NOW - timedelta(days=7), 0.5, 0.1, 0.05)
    job.remember_odds(tmp_path, "nfl-2026", last_week)
    job.remember_odds(tmp_path, "nfl-2026", job.OddsSnapshot(NOW - timedelta(days=2), 0.55, 0.1, 0.05))  # mid-week run
    job.remember_odds(tmp_path, "nfl-2026", job.OddsSnapshot(NOW, 0.6, 0.1, 0.05))
    job.remember_odds(tmp_path, "nfl-2026", job.OddsSnapshot(NOW + timedelta(hours=1), 0.61, 0.1, 0.05))  # rerun
    assert job.previous_odds(tmp_path, "nfl-2026", before=NOW + timedelta(hours=1)) == last_week
    job.remember_odds(tmp_path, "nfl-2026", job.OddsSnapshot(NOW - timedelta(days=14), 0.4, 0.1, 0.05))  # backdated
    assert job.previous_odds(tmp_path, "nfl-2026", before=NOW) == last_week  # newer odds untouched
    assert job.previous_odds(tmp_path, "nfl-2027", before=NOW) is None  # a new season starts fresh


# --- the pieces -------------------------------------------------------------------------------------------------------


def test_odds_trend_round_trips_and_renders(tmp_path: Path) -> None:
    assert job.previous_odds(tmp_path, "nfl", before=NOW) is None
    before = job.OddsSnapshot(NOW - timedelta(days=7), 0.5, 0.1, 0.05)
    job.remember_odds(tmp_path, "nfl", before)
    assert job.previous_odds(tmp_path, "nfl", before=NOW) == before
    (tmp_path / "odds-nfl.json").write_text("not json")
    assert job.previous_odds(tmp_path, "nfl", before=NOW) is None
    now = job.OddsSnapshot(NOW, 0.62, 0.1, 0.04)
    league = job.LeagueReport("nfl", "League", "Team", "2-1", odds=now, previous=before)
    text = "\n".join(job.render_league(league))
    assert "Playoffs 62% (+12 pts), bye 10% (unchanged), title 4% (-1 pts)" in text


def test_render_markdown_collects_notes_once() -> None:
    first = job.LeagueReport("nfl", "A", "T", "1-0", notes=("nfl: x", "y"))
    markdown = job.render_markdown([first], as_of=NOW, notes=["nfl: x"])
    assert markdown.count("nfl: x") == 1 and "- nfl: y" in markdown
    assert job.report_title([first], NOW) == "Weekly report 2026-10-04 (nfl)"


def test_report_path_is_under_the_cache_dir(home: Path) -> None:
    assert job.report_path(NOW) == home / "cache" / "reports" / "weekly-2026-10-04.md"
