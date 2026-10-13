"""``fm run`` and ``fm tick`` around the run lock, driven through a private typer root.

``fm run`` is checked with ``run_forever`` replaced (the loop has its own tests in ``test_runner.py``): its options,
its log file, how it reports the stop, and ``SIGTERM`` turned into the clean stop Ctrl+C gives. ``fm tick`` is checked
with its tick body replaced, so only the lock decision runs: it skips with exit 0 while ``fm run`` holds the lock, it
ticks under the parent's lock when it is ``fm run``'s own child, and otherwise it holds the lock itself while it
ticks. Nothing ticks for real and the lock lives in the per-test config dir.
"""

from __future__ import annotations

import logging
import os
import signal
import sqlite3
from datetime import timedelta

import pytest
import typer
from typer.testing import CliRunner

from fm.commands import run as run_cmd
from fm.commands import schedule as schedule_cmd
from fm.jobs.runlock import HELD_BY_ENV, LockHolder, RunLock
from fm.jobs.runner import DEFAULT_TICK_TIMEOUT, RunOptions, RunStats, run_log_path

runner = CliRunner()


def _root() -> None:
    pass


def cli() -> typer.Typer:
    root = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)
    root.callback()(_root)
    schedule_cmd.register(root)
    run_cmd.register(root)
    return root


def invoke(*args: str) -> str:
    result = runner.invoke(cli(), list(args), catch_exceptions=False)
    assert result.exit_code == 0, result.output
    return result.output


TickCall = tuple[list[str] | None, bool, bool, LockHolder | None]


@pytest.fixture
def ticks(monkeypatch: pytest.MonkeyPatch) -> list[TickCall]:
    """The tick body replaced by a recorder of its arguments and of who held the lock while it ran."""
    calls: list[TickCall] = []

    def record(league: list[str] | None, no_execute: bool, no_sync: bool) -> None:
        calls.append((league, no_execute, no_sync, RunLock().holder()))

    monkeypatch.setattr(schedule_cmd, "_tick_once", record)
    monkeypatch.delenv(HELD_BY_ENV, raising=False)
    return calls


# --- fm tick and the lock ---------------------------------------------------------------------------------------------


def test_fm_tick_skips_with_exit_0_while_fm_run_holds_the_lock(ticks: list[TickCall]) -> None:
    with RunLock(command="fm run"):
        output = invoke("tick")
    assert output == f"fm run is active (pid {os.getpid()}); skipping\n"
    assert ticks == []


def test_a_tick_that_meets_another_tick_names_it(ticks: list[TickCall]) -> None:
    with RunLock(command="fm tick"):
        assert invoke("tick") == f"fm tick is active (pid {os.getpid()}); skipping\n"
    assert ticks == []


def test_fm_tick_holds_the_lock_while_it_ticks_and_releases_it(ticks: list[TickCall]) -> None:
    invoke("tick", "--no-execute", "--league", "nfl")
    ((league, no_execute, no_sync, holder),) = ticks
    assert league == ["nfl"] and no_execute and not no_sync
    assert holder is not None and holder.command == "fm tick" and holder.pid == os.getpid()
    assert not RunLock().is_locked()


def test_the_tick_child_of_fm_run_ticks_under_its_parents_lock(
    ticks: list[TickCall], monkeypatch: pytest.MonkeyPatch
) -> None:
    with RunLock(command="fm run"):
        monkeypatch.setenv(HELD_BY_ENV, str(os.getpid()))
        output = invoke("tick", "--no-sync")
        assert "skipping" not in output
        ((_, _, no_sync, holder),) = ticks
        assert no_sync and holder is not None and holder.command == "fm run"  # still the parent's lock
        monkeypatch.setenv(HELD_BY_ENV, str(os.getpid() + 1))  # a marker naming any other pid is not enough
        assert invoke("tick") == f"fm run is active (pid {os.getpid()}); skipping\n"
        assert len(ticks) == 1
    assert not RunLock().is_locked()


# --- fm run -----------------------------------------------------------------------------------------------------------


def test_fm_run_passes_its_options_logs_to_run_log_and_reports_how_it_stopped(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, object] = {}

    def fake_run_forever(*, options: RunOptions) -> RunStats:
        seen["options"] = options
        seen["sigterm"] = signal.getsignal(signal.SIGTERM)
        logging.getLogger("fm.jobs.runner").info("a line from the loop")
        return RunStats(ticks=2, decisions=1, stopped_by="interrupt")

    monkeypatch.setattr(run_cmd, "run_forever", fake_run_forever)
    before = signal.getsignal(signal.SIGTERM)
    output = invoke("run", "--every", "5", "--no-execute", "--tick-timeout", "7")
    assert "fm run: tick every 5 minutes (no execution), each stopped after 7 min" in output
    assert output.rstrip().endswith("stopped (interrupt) after 2 ticks and 1 decision")
    assert seen["options"] == RunOptions(
        interval=timedelta(minutes=5), execute=False, tick_timeout=timedelta(minutes=7)
    )
    assert seen["sigterm"] is run_cmd._interrupt and signal.getsignal(signal.SIGTERM) is before
    assert "a line from the loop" in run_log_path().read_text(encoding="utf-8")


def test_fm_run_defaults_to_a_tick_every_10_minutes_stopped_after_20(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[RunOptions] = []

    def fake_run_forever(*, options: RunOptions) -> RunStats:
        seen.append(options)
        return RunStats(ticks=1, stopped_by="stop")

    monkeypatch.setattr(run_cmd, "run_forever", fake_run_forever)
    output = invoke("run")
    assert seen == [RunOptions()] and seen[0].tick_timeout == DEFAULT_TICK_TIMEOUT == timedelta(minutes=20)
    assert "each stopped after 20 min" in output and "stopped (stop) after 1 tick and 0 decisions" in output


def test_fm_run_interrupted_while_waiting_for_the_lock_or_failing_on_the_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupted(*, options: RunOptions) -> RunStats:
        raise KeyboardInterrupt

    monkeypatch.setattr(run_cmd, "run_forever", interrupted)
    assert "stopped while waiting for the run lock" in invoke("run")

    def broken(*, options: RunOptions) -> RunStats:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(run_cmd, "run_forever", broken)
    result = runner.invoke(cli(), ["run"], catch_exceptions=False)
    assert result.exit_code == 1 and "error: state database: database is locked" in result.output


def test_sigterm_becomes_the_clean_stop_ctrl_c_gives() -> None:
    before = signal.getsignal(signal.SIGTERM)
    with run_cmd._sigterm_stops():
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGTERM)
    assert signal.getsignal(signal.SIGTERM) is before
