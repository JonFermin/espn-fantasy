"""``fm.jobs.runlock``: one exclusive OS-level lock, a readable holder record, and the tick child's marker.

The lock lives in the per-test config dir (conftest). Two :class:`RunLock` objects in one process conflict like two
processes would (a Windows byte-range lock and a ``flock`` both belong to the open file, not the process), so most
tests hold it in-process; one test holds it from a real child process and shows the OS frees it when that dies.
Blocking waits run in fake time.
"""

from __future__ import annotations

import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fm import paths
from fm.jobs import proctree
from fm.jobs.runlock import (
    HELD_BY_ENV,
    LOCK_FILENAME,
    LockHolder,
    RunLock,
    RunLockHeld,
    held_by_parent,
    lock_path,
)

NOW = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)


def test_the_lock_lives_in_the_config_dir() -> None:
    assert lock_path() == paths.config_dir() / LOCK_FILENAME
    assert RunLock().path == lock_path()


def test_the_holder_record_is_readable_while_held_and_cleared_on_release() -> None:
    lock = RunLock(command="fm run", clock=lambda: NOW)
    with lock.acquire():
        holder = RunLock().holder()
        assert holder == LockHolder(pid=os.getpid(), command="fm run", started_at=NOW)
        assert holder is not None and holder.describe() == f"fm run (pid {os.getpid()}) since 2026-10-04 15:00 UTC"
        assert lock.held and RunLock().is_locked() and lock.is_locked()
    assert not lock.held
    assert not RunLock().is_locked() and RunLock().holder() is None


def test_a_second_holder_is_refused_with_the_first_ones_record() -> None:
    first = RunLock(command="fm run").acquire()
    try:
        with pytest.raises(RunLockHeld) as caught:
            RunLock(command="fm tick").acquire()
        held = caught.value
        assert held.holder is not None and held.holder.pid == os.getpid() and held.holder.command == "fm run"
        assert f"is held by fm run (pid {os.getpid()})" in str(held)
        with pytest.raises(RuntimeError, match="already held by this process"):
            first.acquire()
    finally:
        first.release()
    with RunLock(command="fm tick") as again:  # free again
        assert again.held
    again.release()  # releasing a lock not held is a no-op


def test_a_blocking_acquire_says_whom_it_waits_for_and_takes_the_lock_once_free() -> None:
    other = RunLock(command="fm tick").acquire()
    sleeps: list[float] = []
    waits: list[LockHolder | None] = []

    def sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) == 3:
            other.release()

    lock = RunLock(command="fm run")
    lock.acquire(blocking=True, poll_s=0.5, sleep=sleep, on_wait=waits.append)
    try:
        assert lock.held and sleeps == [0.5, 0.5, 0.5]
        assert [holder.command for holder in waits if holder is not None] == ["fm tick"] and len(waits) == 1
        holder = RunLock().holder()
        assert holder is not None and holder.command == "fm run"
    finally:
        lock.release()


def test_a_blocking_acquire_gives_up_after_its_timeout() -> None:
    other = RunLock(command="fm tick").acquire()
    try:
        with pytest.raises(RunLockHeld):
            RunLock().acquire(blocking=True, timeout=2.0, poll_s=0.5, sleep=lambda _s: None)
    finally:
        other.release()


def test_a_broken_or_naive_record_reads_as_no_holder_or_utc(tmp_path: Path) -> None:
    path = tmp_path / "run.lock"
    lock = RunLock(path)
    assert lock.holder() is None and not lock.is_locked()  # no file yet
    path.write_text("{not json", encoding="utf-8")
    assert lock.holder() is None
    path.write_text('{"pid": 7, "command": "fm run", "started_at": "2026-10-04T15:00:00"}', encoding="utf-8")
    assert lock.holder() == LockHolder(pid=7, command="fm run", started_at=NOW)


def test_held_by_parent_needs_the_marker_to_name_the_holders_pid() -> None:
    holder = LockHolder(pid=4321, command="fm run", started_at=NOW)
    assert held_by_parent(holder, {HELD_BY_ENV: "4321"})
    assert held_by_parent(holder, {HELD_BY_ENV: " 4321 "})
    assert not held_by_parent(holder, {HELD_BY_ENV: "1234"})
    assert not held_by_parent(holder, {HELD_BY_ENV: "not a pid"})
    assert not held_by_parent(holder, {})
    assert not held_by_parent(None, {HELD_BY_ENV: "4321"})


HOLD = (
    "import os, sys, time\n"
    "from pathlib import Path\n"
    "from fm.jobs.runlock import RunLock\n"
    "lock = RunLock(Path(sys.argv[1]), command='fm run').acquire()\n"
    "print('held', os.getpid(), flush=True)\n"
    "time.sleep(60)\n"
)


def test_a_lock_held_by_another_process_is_refused_and_freed_when_that_process_dies(tmp_path: Path) -> None:
    path = tmp_path / "run.lock"
    log = tmp_path / "hold.log"
    output = proctree.open_append(log)
    try:
        tree = proctree.spawn([sys.executable, "-c", HOLD, str(path)], output=output)
    finally:
        os.close(output)
    try:
        holder_pid = int(_wait_for_held(log)[1])
        with pytest.raises(RunLockHeld) as caught:
            RunLock(path, command="fm tick").acquire()
        assert caught.value.holder is not None and caught.value.holder.pid == holder_pid
        assert RunLock(path).is_locked()
    finally:
        tree.kill()
    lock = RunLock(path, command="fm tick").acquire(blocking=True, timeout=10.0, poll_s=0.1)
    try:
        assert lock.held
    finally:
        lock.release()


def _wait_for_held(log: Path, timeout: float = 60.0) -> list[str]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
        for line in text.splitlines():
            if line.startswith("held "):
                return line.split()
        time.sleep(0.05)
    raise AssertionError(f"the child never took the lock: {log.read_text(encoding='utf-8', errors='replace')}")
