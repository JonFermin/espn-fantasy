"""Start a child as the root of a process tree, and stop the whole tree (PLUG_AND_PLAY section 4).

Why this exists: on a real machine a scheduled ``fm tick`` hung for five days inside a headless Playwright session
(``python`` -> ``node.exe`` -> ``msedge.exe``), and the Task Scheduler's two-hour limit killed only the wrapper at the
top, orphaning the rest while no tick completed. ``fm run`` therefore runs each tick as a child process
(:func:`fm.jobs.runner.child_tick`) with a hard timeout, and on timeout stops everything the tick started, not only
the process it spawned: Playwright's driver and the browser are grandchildren, and on Windows the venv's
``python.exe`` is itself a launcher whose child is the real interpreter.

This module is the second platform branch outside the scheduler backends (CLAUDE.md; the first is
:mod:`fm.jobs.runlock`), kept to the functions under the ``sys.platform`` switch:

- **Windows.** The child starts with ``CREATE_NO_WINDOW``: ``fm run`` under ``uvw.exe`` has no console, and a console
  child of a process without one would open a window. :meth:`ProcessTree.kill` runs ``taskkill /PID <pid> /T /F``,
  which walks the tree by parent pid and terminates all of it; when ``taskkill`` itself fails the child alone is
  terminated and the failure is reported.
- **macOS (POSIX).** The child starts a new session (``start_new_session=True``), so it and what it starts share a
  process group. :meth:`ProcessTree.kill` snapshots the descendants by parent pid (``ps``) first, then sends
  ``SIGKILL`` to the group and to each descendant, which also reaches a process that moved to a group of its own
  (Playwright starts the browser detached).

Helper commands (``taskkill``, ``ps``) go through an injectable :class:`fm.jobs.scheduler_base.Runner`, like the
scheduler backends; the default, :func:`quiet_runner`, opens no console window and never raises, because a kill must
go on whatever a helper does. :func:`open_append` opens the log a child writes to with appends the OS makes atomic
(``FILE_APPEND_DATA`` without write access on Windows, ``O_APPEND`` elsewhere), so the child's output and the parent's
own log lines interleave and never overwrite each other. :func:`is_running` says whether a pid is alive; the test that
proves a kill uses it. :func:`kill_pid_tree` stops a tree known only by its root's pid (the same ``taskkill /T`` or
descendants-then-``SIGKILL``), which is how ``fm run`` stops a scheduled ``fm tick`` stuck holding the run lock.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

from fm.jobs.scheduler_base import Runner, RunResult

KILL_WAIT_S = 10.0
"""How long :meth:`ProcessTree.kill` waits for the killed child to be reaped."""
HELPER_TIMEOUT_S = 30.0
"""The longest a helper command (``taskkill``, ``ps``) may take."""
TASKKILL = "taskkill"
PS = "ps"

if sys.platform == "win32":
    import _winapi
    import ctypes
    import msvcrt
    from ctypes import wintypes

    _NO_WINDOW = subprocess.CREATE_NO_WINDOW
    _FILE_APPEND_DATA = 0x0004
    _SYNCHRONIZE = 0x0010_0000
    _SHARE_ALL = 0x1 | 0x2 | 0x4  # FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE
    _OPEN_ALWAYS = 4
    _FILE_ATTRIBUTE_NORMAL = 0x80
    _PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    _STILL_ACTIVE = 259
    _ERROR_ACCESS_DENIED = 5

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _kernel32.OpenProcess.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    _kernel32.OpenProcess.restype = wintypes.HANDLE
    _kernel32.GetExitCodeProcess.argtypes = (wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD))
    _kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    _kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    _kernel32.CloseHandle.restype = wintypes.BOOL

    def _popen(
        argv: list[str], *, env: dict[str, str] | None, output: int | None, cwd: Path | None
    ) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT if output is not None else None,
            env=env,
            cwd=cwd,
            creationflags=_NO_WINDOW,
        )

    def _kill_tree(process: subprocess.Popen[bytes], runner: Runner) -> list[str]:
        result = runner(taskkill_argv(process.pid))
        if result.ok or process.poll() is not None:
            return []
        lines = result.output.splitlines()
        problem = lines[0] if lines else f"exit {result.returncode}"
        with contextlib.suppress(OSError):
            process.kill()
        return [f"taskkill failed ({problem}); terminated pid {process.pid} alone, its children may still run"]

    def _kill_pid_tree(pid: int, runner: Runner) -> list[str]:
        result = runner(taskkill_argv(pid))
        if result.ok or not is_running(pid):
            return []
        lines = result.output.splitlines()
        return [f"taskkill /PID {pid} failed ({lines[0] if lines else f'exit {result.returncode}'})"]

    def open_append(path: Path) -> int:
        """A write-only descriptor on ``path`` (created when missing) whose every write the OS appends at the end,
        shared for read, write and delete so the parent's log rotation can still rename the file. Close it with
        ``os.close`` once the child has it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = _winapi.CreateFile(
            str(path), _FILE_APPEND_DATA | _SYNCHRONIZE, _SHARE_ALL, 0, _OPEN_ALWAYS, _FILE_ATTRIBUTE_NORMAL, 0
        )
        return msvcrt.open_osfhandle(handle, os.O_WRONLY)

    def is_running(pid: int) -> bool:
        """Whether a process with ``pid`` exists and has not exited."""
        handle = _kernel32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return ctypes.get_last_error() == _ERROR_ACCESS_DENIED  # exists, but belongs to someone else
        try:
            code = wintypes.DWORD()
            if not _kernel32.GetExitCodeProcess(handle, ctypes.byref(code)):
                return True
            return code.value == _STILL_ACTIVE
        finally:
            _kernel32.CloseHandle(handle)

else:
    import signal

    _NO_WINDOW = 0

    def _popen(
        argv: list[str], *, env: dict[str, str] | None, output: int | None, cwd: Path | None
    ) -> subprocess.Popen[bytes]:
        return subprocess.Popen(
            argv,
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT if output is not None else None,
            env=env,
            cwd=cwd,
            start_new_session=True,
        )

    def _kill_tree(process: subprocess.Popen[bytes], runner: Runner) -> list[str]:
        if process.poll() is not None:
            return []  # reaped: its pid, and so its group id, may belong to someone else by now
        notes: list[str] = []
        descendants = _descendants(process.pid, runner)
        try:
            os.killpg(process.pid, signal.SIGKILL)  # the session leader's pid is its group id
        except ProcessLookupError:
            pass
        except PermissionError as exc:
            notes.append(f"could not signal process group {process.pid}: {exc}")
        for pid in descendants:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                os.kill(pid, signal.SIGKILL)
        with contextlib.suppress(OSError):
            process.kill()
        return notes

    def _kill_pid_tree(pid: int, runner: Runner) -> list[str]:
        notes: list[str] = []
        descendants = _descendants(pid, runner)
        with contextlib.suppress(ProcessLookupError, PermissionError):
            if os.getpgid(pid) == pid:  # a group leader (a session we started): its group goes too
                os.killpg(pid, signal.SIGKILL)
        for target in (pid, *descendants):
            try:
                os.kill(target, signal.SIGKILL)
            except ProcessLookupError:
                pass
            except PermissionError as exc:
                notes.append(f"could not signal pid {target}: {exc}")
        return notes

    def open_append(path: Path) -> int:
        """A write-only ``O_APPEND`` descriptor on ``path`` (created when missing). Close it with ``os.close`` once
        the child has it."""
        path.parent.mkdir(parents=True, exist_ok=True)
        return os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)

    def is_running(pid: int) -> bool:
        """Whether a process with ``pid`` exists (a zombie counts until its parent reaps it)."""
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True  # exists, but belongs to someone else
        return True


def quiet_runner(argv: Sequence[str], *, timeout: float = HELPER_TIMEOUT_S) -> RunResult:
    """Run a helper command for real: never through a shell, without a console window, with stdin closed. A missing
    executable or a hang is a failed :class:`RunResult` (return code -1), never an exception."""
    args = [str(arg) for arg in argv]
    try:
        proc = subprocess.run(
            args,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
            creationflags=_NO_WINDOW,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return RunResult(argv=tuple(args), returncode=-1, stderr=f"{type(exc).__name__}: {exc}")
    return RunResult(argv=tuple(args), returncode=proc.returncode, stdout=proc.stdout or "", stderr=proc.stderr or "")


def taskkill_argv(pid: int) -> list[str]:
    """``taskkill /PID <pid> /T /F``: ``pid`` and every process under it, forcibly (Windows)."""
    return [TASKKILL, "/PID", str(pid), "/T", "/F"]


def ps_argv() -> list[str]:
    """``ps -A -o pid= -o ppid=``: every process with its parent, no header (macOS and Linux alike)."""
    return [PS, "-A", "-o", "pid=", "-o", "ppid="]


def _descendants(root: int, runner: Runner) -> list[int]:
    """Every process under ``root`` by parent pid, parents before children; empty when ``ps`` fails."""
    result = runner(ps_argv())
    if not result.ok:
        return []
    children: dict[int, list[int]] = {}
    for line in result.stdout.splitlines():
        fields = line.split()
        if len(fields) == 2 and fields[0].isdigit() and fields[1].isdigit():
            children.setdefault(int(fields[1]), []).append(int(fields[0]))
    found: list[int] = []
    queue = [root]
    while queue:
        for child in children.get(queue.pop(0), ()):
            if child != root and child not in found:
                found.append(child)
                queue.append(child)
    return found


def kill_pid_tree(pid: int, *, runner: Runner = quiet_runner) -> list[str]:
    """Stop the process ``pid`` and every process under it, by pid alone: for a tree this process did not start
    itself, such as a scheduled ``fm tick`` that has held the run lock far past any tick's timeout (``fm run`` stops
    it so the loop can take over). Windows: ``taskkill /PID <pid> /T /F``. macOS: the descendants by parent pid, then
    ``SIGKILL`` to each and to ``pid`` (and to its group when it leads one). Returns what did not go to plan; a pid
    that is already gone is not an error. Never pass this process's own pid."""
    if pid == os.getpid():
        raise ValueError("kill_pid_tree refuses to stop this process's own tree")
    return _kill_pid_tree(pid, runner)


class ProcessTree:
    """A child started by :func:`spawn`, and everything it starts."""

    def __init__(self, process: subprocess.Popen[bytes]) -> None:
        self.process = process

    @property
    def pid(self) -> int:
        """The child's pid (on Windows in a venv, the launcher's; the interpreter is its child)."""
        return self.process.pid

    def poll(self) -> int | None:
        return self.process.poll()

    def wait(self, timeout: float | None = None) -> int:
        """The child's exit code once it exits; ``subprocess.TimeoutExpired`` after ``timeout`` seconds. Waits for
        the child only, never for a pipe a grandchild might hold open (there is none: output goes to a file)."""
        return self.process.wait(timeout)

    def kill(self, *, runner: Runner = quiet_runner, wait_s: float = KILL_WAIT_S) -> list[str]:
        """Stop the child and every process under it, then reap the child. Returns what did not go to plan (empty
        when the tree is gone). A child that already exited is not an error."""
        notes = _kill_tree(self.process, runner)
        try:
            self.process.wait(wait_s)
        except subprocess.TimeoutExpired:
            notes.append(f"pid {self.pid} still running {wait_s:.0f}s after the kill")
        return notes


def spawn(
    argv: Sequence[str],
    *,
    env: Mapping[str, str] | None = None,
    output: int | None = None,
    cwd: Path | None = None,
) -> ProcessTree:
    """Start ``argv`` (never through a shell) as the root of a tree :meth:`ProcessTree.kill` can stop whole: stdin
    from ``os.devnull``, stdout and stderr both to the descriptor ``output`` (this process's own when ``None``),
    ``env`` as its whole environment (this process's when ``None``)."""
    return ProcessTree(
        _popen([str(arg) for arg in argv], env=None if env is None else dict(env), output=output, cwd=cwd)
    )


__all__ = [
    "HELPER_TIMEOUT_S",
    "KILL_WAIT_S",
    "PS",
    "TASKKILL",
    "ProcessTree",
    "is_running",
    "kill_pid_tree",
    "open_append",
    "ps_argv",
    "quiet_runner",
    "spawn",
    "taskkill_argv",
]
