"""``fm.jobs.proctree``: a child started as the root of a process tree, and the whole tree stopped.

The module is mostly its platform switch, so the core test starts real, tiny, offline Python processes: a child that
starts a grandchild and sleeps, stopped through :meth:`ProcessTree.kill`, with nothing of it left running (on Windows
the venv's ``python.exe`` is a launcher, so the tree is launcher -> interpreter -> launcher -> interpreter). The
helper commands are rendered and parsed through fake runners. Every script sleeps a bounded time, so a failed test
leaves nothing behind for long, and each test kills what it started in a ``finally``.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import pytest

from fm.jobs import proctree
from fm.jobs.proctree import ProcessTree, is_running, open_append, ps_argv, quiet_runner, spawn, taskkill_argv
from fm.jobs.scheduler_base import RunResult

PYTHON = [sys.executable, "-I", "-S"]
"""A bare interpreter (no site, no user environment): starts in well under a second, even through the launcher."""
TREE = (
    "import os, subprocess, sys, time\n"
    "grandchild = subprocess.Popen([sys.executable, '-I', '-S', '-c', "
    "'import os, time; print(\"grandchild\", os.getpid(), flush=True); time.sleep(60)'])\n"
    "print('child', os.getpid(), grandchild.pid, os.environ.get('FM_TEST_MARK', '-'), flush=True)\n"
    "time.sleep(60)\n"
)
"""A child that starts a grandchild and both sleep: ``child <pid> <grandchild root pid> <mark>`` and ``grandchild
<pid>`` land in the output file."""


def wait_for_lines(path: Path, *names: str, timeout: float = 30.0) -> dict[str, list[str]]:
    """The first line starting with each of ``names`` in ``path``, split into words, once they are all there."""
    deadline = time.monotonic() + timeout
    found: dict[str, list[str]] = {}
    while time.monotonic() < deadline:
        text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
        for line in text.splitlines():
            words = line.split()
            if words and words[0] in names:
                found.setdefault(words[0], words)
        if all(name in found for name in names):
            return found
        time.sleep(0.05)
    raise AssertionError(f"{names} not all in {path} after {timeout}s: {found}")


def still_running(pids: set[int], timeout: float = 10.0) -> set[int]:
    """The pids still alive once ``timeout`` has passed (empty as soon as all are gone)."""
    deadline = time.monotonic() + timeout
    while True:
        alive = {pid for pid in pids if is_running(pid)}
        if not alive or time.monotonic() >= deadline:
            return alive
        time.sleep(0.05)


def test_kill_stops_the_child_and_the_grandchild_it_started(tmp_path: Path) -> None:
    log = tmp_path / "logs" / "tree.log"
    output = open_append(log)
    try:
        tree = spawn([*PYTHON, "-c", TREE], env={**os.environ, "FM_TEST_MARK": "marked"}, output=output)
    finally:
        os.close(output)
    try:
        lines = wait_for_lines(log, "child", "grandchild")
        assert tree.poll() is None
        assert lines["child"][3] == "marked"
        pids = {tree.pid, int(lines["child"][1]), int(lines["child"][2]), int(lines["grandchild"][1])}
        assert still_running(pids, timeout=0) == pids

        assert tree.kill() == []
        assert tree.poll() is not None
        assert still_running(pids) == set()
        assert tree.kill() == []  # a tree already stopped is not an error
    finally:
        tree.kill()


def test_a_child_that_exits_reports_its_code_and_its_output(tmp_path: Path) -> None:
    log = tmp_path / "out.log"
    output = open_append(log)
    try:
        tree = spawn([*PYTHON, "-c", "import sys; print('hello from the child'); sys.exit(3)"], output=output)
    finally:
        os.close(output)
    try:
        assert tree.wait(30) == 3
    finally:
        tree.kill()
    assert "hello from the child" in log.read_text(encoding="utf-8")
    assert not is_running(tree.pid)


def test_a_wait_past_its_timeout_raises_and_leaves_the_child_running(tmp_path: Path) -> None:
    output = open_append(tmp_path / "x.log")
    try:
        tree = spawn([*PYTHON, "-c", "import time; time.sleep(60)"], output=output)
    finally:
        os.close(output)
    try:
        with pytest.raises(subprocess.TimeoutExpired):
            tree.wait(0.2)
        assert tree.poll() is None
    finally:
        tree.kill()
    assert tree.poll() is not None


def test_appends_from_two_writers_never_overwrite_each_other(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "run.log"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"start\n")
    fd = open_append(path)
    try:
        os.write(fd, b"child 1\n")
        with path.open("ab") as parent:  # what the parent's log handler does in between
            parent.write(b"parent\n")
        os.write(fd, b"child 2\n")
    finally:
        os.close(fd)
    assert path.read_bytes() == b"start\nchild 1\nparent\nchild 2\n"


def test_is_running_knows_this_process_and_an_exited_one() -> None:
    assert is_running(os.getpid())
    done = subprocess.Popen([*PYTHON, "-c", "pass"])
    assert done.wait(30) == 0
    assert not is_running(done.pid)


# --- the helper commands ----------------------------------------------------------------------------------------------


class ScriptedRunner:
    def __init__(self, *results: RunResult) -> None:
        self.results = list(results)
        self.calls: list[list[str]] = []

    def __call__(self, argv: Sequence[str]) -> RunResult:
        self.calls.append([str(arg) for arg in argv])
        return self.results.pop(0)


def test_the_helper_commands_are_rendered_without_a_shell() -> None:
    assert taskkill_argv(4242) == ["taskkill", "/PID", "4242", "/T", "/F"]
    assert ps_argv() == ["ps", "-A", "-o", "pid=", "-o", "ppid="]


def test_descendants_are_read_from_ps_breadth_first_and_a_failed_ps_finds_none() -> None:
    listing = "    1     0\n   10     1\n   11    10\n   12    11\n   13    10\n   20     1\n  bad line\n"
    runner = ScriptedRunner(RunResult(tuple(ps_argv()), 0, stdout=listing))
    assert proctree._descendants(10, runner) == [11, 13, 12]
    assert runner.calls == [ps_argv()]
    assert proctree._descendants(10, ScriptedRunner(RunResult(tuple(ps_argv()), 1, stderr="no ps"))) == []


def test_the_quiet_runner_reports_a_missing_executable_instead_of_raising() -> None:
    result = quiet_runner(["fm-no-such-helper-executable", "--version"])
    assert result.returncode == -1 and not result.ok
    assert "FileNotFoundError" in result.stderr


class FakeProcess:
    """A ``Popen`` stand-in that never was a process, for the Windows fallback path."""

    pid = 4242

    def __init__(self) -> None:
        self.returncode: int | None = None
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = 1

    def wait(self, timeout: float | None = None) -> int | None:
        return self.returncode


@pytest.mark.skipif(sys.platform != "win32", reason="the taskkill path is the Windows branch")
def test_on_windows_kill_runs_taskkill_on_the_tree_and_falls_back_to_the_child_alone() -> None:
    process = FakeProcess()
    ok = ScriptedRunner(RunResult(tuple(taskkill_argv(4242)), 0, stdout="SUCCESS"))
    assert ProcessTree(cast(subprocess.Popen[bytes], process)).kill(runner=ok, wait_s=0) == []
    assert ok.calls == [taskkill_argv(4242)] and not process.killed

    process = FakeProcess()
    failing = ScriptedRunner(RunResult(tuple(taskkill_argv(4242)), -1, stderr="FileNotFoundError: taskkill"))
    (note,) = ProcessTree(cast(subprocess.Popen[bytes], process)).kill(runner=failing, wait_s=0)
    assert process.killed
    assert note.startswith("taskkill failed (FileNotFoundError: taskkill); terminated pid 4242 alone")
