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
import sys
import time
from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path
from typing import NoReturn

import pytest
import typer
from typer.testing import CliRunner

from fm import paths
from fm.commands import run as run_cmd
from fm.commands import schedule as schedule_cmd
from fm.config import ConfigError
from fm.jobs import proctree
from fm.jobs.runlock import HELD_BY_ENV, LockHolder, RunLock
from fm.jobs.runner import (
    DEFAULT_TICK_TIMEOUT,
    TICK_CRASHED_EXIT,
    RunOptions,
    RunStats,
    TickResult,
    run_log_path,
    run_tick_child,
)
from fm.jobs.tick import TickOptions, TickState, state_path
from fm.notify import Message, NotifyError

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


@pytest.fixture(autouse=True)
def _no_dashboard(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[int]]:
    """No test here binds a socket: ``fm run`` serves the dashboard unless ``--no-ui``, so every ``run`` here passes
    it, and a bind that slips through is recorded (``fm run`` only logs a dashboard that fails to start) and fails the
    test at teardown. The dashboard's own wiring is tested with a fake server in ``tests/web/test_commands.py``."""
    import fm.web.server as web_server

    binds: list[int] = []

    def refuse(port: int = 0, **_: object) -> NoReturn:
        binds.append(port)
        raise OSError("a unit test must not bind the dashboard's socket; pass --no-ui")

    monkeypatch.setattr(web_server, "bind_dashboard", refuse)
    yield binds
    assert binds == [], "fm run tried to serve the dashboard in a unit test; pass --no-ui"


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
    invoke("tick", "--in-process", "--no-execute", "--league", "nfl")
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


# --- fm tick in a child process ---------------------------------------------------------------------------------------


class FakeChildren:
    """``run_tick_child`` replaced: records ``(options, timeout, lock holder)`` per call and answers from
    ``answers`` (exit 0 once they run out)."""

    def __init__(self) -> None:
        self.calls: list[tuple[TickOptions, timedelta, LockHolder | None]] = []
        self.answers: list[TickResult] = []

    def __call__(self, options: TickOptions, *, timeout: timedelta) -> TickResult:
        self.calls.append((options, timeout, RunLock().holder()))
        return self.answers.pop(0) if self.answers else TickResult(0)


@pytest.fixture
def children(monkeypatch: pytest.MonkeyPatch) -> FakeChildren:
    fake = FakeChildren()
    monkeypatch.setattr(schedule_cmd, "run_tick_child", fake)
    monkeypatch.delenv(HELD_BY_ENV, raising=False)
    return fake


class FakePhone:
    name = "fake"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.sent: list[Message] = []

    def send(self, message: Message) -> None:
        if self.fail:
            raise NotifyError("502 from the channel")
        self.sent.append(message)

    def close(self) -> None:
        pass


def test_fm_tick_runs_the_tick_in_a_child_under_its_own_lock(children: FakeChildren) -> None:
    assert invoke("tick", "--no-execute", "--league", "nfl", "--timeout", "7") == ""
    ((options, timeout, holder),) = children.calls
    assert options == TickOptions(execute=False, sync=True, leagues=("nfl",))
    assert timeout == timedelta(minutes=7)
    assert holder is not None and holder.command == "fm tick" and holder.pid == os.getpid()
    assert not RunLock().is_locked()
    invoke("tick")
    assert children.calls[1][:2] == (TickOptions(), DEFAULT_TICK_TIMEOUT)


def test_fm_tick_exits_1_when_its_child_reports_a_failure_or_crashes(children: FakeChildren) -> None:
    children.answers = [TickResult(1), TickResult(TICK_CRASHED_EXIT)]
    assert runner.invoke(cli(), ["tick"], catch_exceptions=False).exit_code == 1
    assert runner.invoke(cli(), ["tick"], catch_exceptions=False).exit_code == 1
    assert not RunLock().is_locked()


def test_fm_tick_skips_without_a_child_while_fm_run_holds_the_lock(children: FakeChildren) -> None:
    with RunLock(command="fm run"):
        assert invoke("tick") == f"fm run is active (pid {os.getpid()}); skipping\n"
    assert children.calls == []


def test_a_timed_out_tick_exits_1_and_is_alerted_at_most_once_an_hour(
    children: FakeChildren, monkeypatch: pytest.MonkeyPatch
) -> None:
    phone = FakePhone()
    monkeypatch.setattr(schedule_cmd, "load_config", lambda: object())
    monkeypatch.setattr(schedule_cmd, "open_channel", lambda config: phone)
    hung = TickResult(None, pid=4242, notes=("pid 4242 still running 10s after the kill",))
    children.answers = [hung, hung]
    result = runner.invoke(cli(), ["tick"], catch_exceptions=False)
    assert result.exit_code == 1
    assert "error: tick timed out after 20 min and was stopped (pid 4242 and every process under it)" in result.stderr
    assert "note: pid 4242 still running 10s after the kill" in result.stderr
    assert [message.title for message in phone.sent] == ["fm tick: tick timed out"]
    assert TickState.load(state_path()).alerts[schedule_cmd.TIMEOUT_ALERT_KEY] is not None
    assert runner.invoke(cli(), ["tick"], catch_exceptions=False).exit_code == 1
    assert len(phone.sent) == 1  # the second timeout within the hour is not pushed again


def test_a_timeout_alert_that_did_not_go_out_is_tried_again_next_time(
    children: FakeChildren, monkeypatch: pytest.MonkeyPatch
) -> None:
    phone = FakePhone(fail=True)
    monkeypatch.setattr(schedule_cmd, "load_config", lambda: object())
    monkeypatch.setattr(schedule_cmd, "open_channel", lambda config: phone)
    children.answers = [TickResult(None, pid=1), TickResult(None, pid=2)]
    result = runner.invoke(cli(), ["tick"], catch_exceptions=False)
    assert result.exit_code == 1 and "timeout not pushed to the phone (502 from the channel)" in result.stderr
    assert schedule_cmd.TIMEOUT_ALERT_KEY not in TickState.load(state_path()).alerts
    phone.fail = False
    runner.invoke(cli(), ["tick"], catch_exceptions=False)
    assert [message.title for message in phone.sent] == ["fm tick: tick timed out"]


def test_a_real_hung_tick_child_of_fm_tick_is_stopped_with_its_tree(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The scheduled tick's own timeout, for real: a child that hangs with a grandchild (the browser) is killed with
    its whole tree, and the marker in its environment names this process, which holds the lock as fm tick."""
    log = tmp_path / "tick.log"
    seen: list[TickResult] = []

    def hung_child(options: TickOptions, *, timeout: timedelta) -> TickResult:
        result = run_tick_child(options, timeout=timedelta(seconds=1), argv=[*PYTHON, "-c", HANG], log_path=log)
        seen.append(result)
        return result

    monkeypatch.setattr(schedule_cmd, "run_tick_child", hung_child)
    monkeypatch.setattr(schedule_cmd, "load_config", _no_config)
    monkeypatch.delenv(HELD_BY_ENV, raising=False)
    result = runner.invoke(cli(), ["tick"], catch_exceptions=False)
    assert result.exit_code == 1 and "tick timed out after 20 min and was stopped" in result.stderr
    assert "timeout not pushed to the phone (" in result.stderr and "config.toml not found" in result.stderr
    text = log.read_text(encoding="utf-8")
    child = next(line.split() for line in text.splitlines() if line.startswith("child "))
    grandchild = next(line.split() for line in text.splitlines() if line.startswith("grandchild "))
    assert child[3] == str(os.getpid())  # HELD_BY_ENV: this process, the fm tick holding the lock
    pids = {seen[0].pid or 0, int(child[1]), int(child[2]), int(grandchild[1])}
    assert _still_running(pids) == set(), "the scheduled tick's child tree survived its timeout"
    assert not RunLock().is_locked()


def test_the_real_child_of_fm_tick_ticks_under_the_parents_lock(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``python -m fm.jobs.runner tick`` started by fm tick does not skip on its parent's lock: it gets as far as the
    temp config dir's missing config.toml (so nothing reaches ESPN) and its exit code comes back."""
    assert not (paths.config_dir() / "config.toml").exists()
    log = tmp_path / "tick.log"

    def logged_child(options: TickOptions, *, timeout: timedelta) -> TickResult:
        return run_tick_child(options, timeout=timedelta(seconds=120), log_path=log)

    monkeypatch.setattr(schedule_cmd, "run_tick_child", logged_child)
    monkeypatch.delenv(HELD_BY_ENV, raising=False)
    result = runner.invoke(cli(), ["tick", "--no-sync"], catch_exceptions=False)
    text = log.read_text(encoding="utf-8", errors="replace")
    assert result.exit_code == 1, text
    assert "config.toml not found" in text and "skipping" not in text


PYTHON = [sys.executable, "-I", "-S"]
HANG = (
    "import os, subprocess, sys, time\n"
    "grandchild = subprocess.Popen([sys.executable, '-I', '-S', '-c', "
    "'import os, time; print(\"grandchild\", os.getpid(), flush=True); time.sleep(60)'])\n"
    "time.sleep(0.5)\n"
    f"print('child', os.getpid(), grandchild.pid, os.environ.get('{HELD_BY_ENV}', '-'), flush=True)\n"
    "time.sleep(60)\n"
)


def _no_config() -> NoReturn:
    raise ConfigError(f"{paths.config_file()} not found")


def _still_running(pids: set[int], timeout: float = 10.0) -> set[int]:
    deadline = time.monotonic() + timeout
    while True:
        alive = {pid for pid in pids if proctree.is_running(pid)}
        if not alive or time.monotonic() >= deadline:
            return alive
        time.sleep(0.05)


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
    output = invoke("run", "--no-ui", "--every", "5", "--no-execute", "--tick-timeout", "7")
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
    output = invoke("run", "--no-ui")
    assert seen == [RunOptions()] and seen[0].tick_timeout == DEFAULT_TICK_TIMEOUT == timedelta(minutes=20)
    assert "each stopped after 20 min" in output and "stopped (stop) after 1 tick and 0 decisions" in output


def test_fm_run_interrupted_while_waiting_for_the_lock_or_failing_on_the_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupted(*, options: RunOptions) -> RunStats:
        raise KeyboardInterrupt

    monkeypatch.setattr(run_cmd, "run_forever", interrupted)
    assert "stopped while waiting for the run lock" in invoke("run", "--no-ui")

    def broken(*, options: RunOptions) -> RunStats:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(run_cmd, "run_forever", broken)
    result = runner.invoke(cli(), ["run", "--no-ui"], catch_exceptions=False)
    assert result.exit_code == 1 and "error: state database: database is locked" in result.output


def test_sigterm_becomes_the_clean_stop_ctrl_c_gives() -> None:
    before = signal.getsignal(signal.SIGTERM)
    with run_cmd._sigterm_stops():
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGTERM)
    assert signal.getsignal(signal.SIGTERM) is before
