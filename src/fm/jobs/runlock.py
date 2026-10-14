"""The run lock (DESIGN section 13.2): one process at a time ticks and executes.

``fm run`` holds ``run.lock`` under the config dir for its lifetime and ``fm tick`` holds it for one tick. A scheduled
tick that fires while the loop is up therefore skips (``fm run is active (pid N); skipping``, exit 0) instead of
running a second tick against the same store and browser profile, and the loop waits when a scheduled tick is
mid-run. That is what lets the tick task from ``fm schedule install`` stay installed purely as the Windows wake
timer: it wakes the PC, finds the lock held, exits, and the woken loop catches up on its own
(:mod:`fm.jobs.runner`).

The lock is an OS-level exclusive lock on the file, so a process that dies releases it with no stale pid file to
clean up: ``msvcrt.locking`` on Windows, ``fcntl.flock`` elsewhere. This is one of the two platform branches outside
the scheduler backends (CLAUDE.md; the other is :mod:`fm.jobs.proctree`), kept to :func:`_lock` and :func:`_unlock`.
The holder writes ``{pid, command, started_at}`` as JSON at the start of the file and :meth:`RunLock.holder` reads it
from any process. On Windows a byte-range lock also blocks other processes' reads of the locked range, so the lock is
taken on one byte at :data:`LOCK_OFFSET`, far past the holder record, and a read of the record never touches it
(``fcntl.flock`` locks the whole file but never blocks reads).

Both lock calls are non-blocking; a blocking :meth:`RunLock.acquire` retries them on a timer with an injectable
``sleep``, so a test waits in fake time and ``fm run`` can say what it is waiting for (``on_wait``).

``fm run`` runs each tick as a child process (``fm tick`` under a hard timeout, :func:`fm.jobs.runner.child_tick`)
while it keeps holding the lock itself, so that child must not skip. The parent puts its pid in
:data:`HELD_BY_ENV` in the child's environment, and :func:`held_by_parent` lets ``fm tick`` run without the lock only
when that pid is the one the lock record names: a stale or copied variable never lets a second tick in beside a
running ``fm run``.
"""

from __future__ import annotations

import errno
import json
import os
import sys
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Self

from fm import paths
from fm.store import utc_now

LOCK_FILENAME = "run.lock"
LOCK_OFFSET = 1 << 20
"""The byte locked on Windows: 1 MiB in, far beyond the holder record, so reading the record never hits the lock."""
LOCK_POLL_S = 0.5
"""How often a blocking acquire retries the non-blocking lock."""
HELD_BY_ENV = "FM_RUN_LOCK_HELD_BY"
"""Set by ``fm run`` in its tick child's environment: the pid holding the lock on the child's behalf."""
_HELD_ERRNOS = frozenset({errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK, errno.EDEADLK})
"""``msvcrt.locking`` answers a held region with ``EACCES`` (``EDEADLK`` after its own retries); ``flock`` with
``EWOULDBLOCK``/``EAGAIN``. Any other error is a real failure and propagates."""

if sys.platform == "win32":
    import msvcrt

    def _lock(fd: int) -> None:
        os.lseek(fd, LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)

    def _unlock(fd: int) -> None:
        os.lseek(fd, LOCK_OFFSET, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)

else:
    import fcntl

    def _lock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


def lock_path() -> Path:
    """``$FM_CONFIG_DIR/run.lock``."""
    return paths.config_dir() / LOCK_FILENAME


@dataclass(frozen=True, slots=True)
class LockHolder:
    """What the process holding the lock wrote about itself."""

    pid: int
    command: str
    started_at: datetime

    def describe(self) -> str:
        """``fm run (pid 1234) since 2026-10-04 15:00 UTC``."""
        return f"{self.command} (pid {self.pid}) since {self.started_at.astimezone(UTC):%Y-%m-%d %H:%M} UTC"


class RunLockHeld(RuntimeError):
    """The lock is held by another process; ``holder`` is its record when it could be read."""

    def __init__(self, path: Path, holder: LockHolder | None) -> None:
        self.path = path
        self.holder = holder
        super().__init__(f"{path} is held by {holder.describe() if holder is not None else 'another process'}")


class RunLock:
    """The exclusive run lock at ``path`` (default :func:`lock_path`), taken on behalf of ``command``.

    ``with RunLock().acquire(): ...`` or ``with RunLock(): ...`` take it non-blocking and release it on exit; a
    blocking acquire waits for the other holder. One object holds the lock at most once at a time.
    """

    def __init__(
        self, path: Path | None = None, *, command: str = "fm run", clock: Callable[[], datetime] = utc_now
    ) -> None:
        self.path = path if path is not None else lock_path()
        self.command = command
        self._clock = clock
        self._fd: int | None = None

    @property
    def held(self) -> bool:
        """This object holds the lock."""
        return self._fd is not None

    def acquire(
        self,
        *,
        blocking: bool = False,
        timeout: float | None = None,
        poll_s: float = LOCK_POLL_S,
        sleep: Callable[[float], object] = time.sleep,
        on_wait: Callable[[LockHolder | None], object] | None = None,
    ) -> Self:
        """Take the lock and write the holder record. Non-blocking, a held lock raises :class:`RunLockHeld`;
        blocking, the lock is retried every ``poll_s`` seconds until it is free or ``timeout`` seconds have passed
        (``RunLockHeld`` then). ``on_wait`` is called once, with the holder's record, when waiting starts."""
        if self._fd is not None:
            raise RuntimeError(f"{self.path} is already held by this process")
        fd = _open(self.path)
        waited = 0.0
        waiting = False
        try:
            while True:
                try:
                    _lock(fd)
                except OSError as exc:
                    if exc.errno not in _HELD_ERRNOS:
                        raise
                    holder = self.holder()
                    if not blocking or (timeout is not None and waited >= timeout):
                        raise RunLockHeld(self.path, holder) from None
                    if not waiting:
                        waiting = True
                        if on_wait is not None:
                            on_wait(holder)
                    sleep(poll_s)
                    waited += poll_s
                else:
                    break
            self._fd = fd
            self._write_holder(fd)
        except BaseException:
            if self._fd is fd:
                self._fd = None
                _unlock(fd)
            os.close(fd)
            raise
        return self

    def release(self) -> None:
        """Clear the holder record and let the lock go. Releasing a lock not held is a no-op."""
        fd = self._fd
        if fd is None:
            return
        self._fd = None
        try:
            os.ftruncate(fd, 0)  # a reader never sees a stale holder while the lock is free
        finally:
            try:
                _unlock(fd)
            finally:
                os.close(fd)

    def holder(self) -> LockHolder | None:
        """The record the current holder wrote, or ``None`` when the file is missing, empty or unreadable (a crashed
        holder leaves its record behind; :meth:`is_locked` says whether the lock is actually held)."""
        try:
            raw = self.path.read_bytes()
        except OSError:
            return None
        text = raw.decode("utf-8", errors="replace").strip()
        if not text:
            return None
        try:
            data = json.loads(text)
            started_at = datetime.fromisoformat(str(data["started_at"]))
            if started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=UTC)
            return LockHolder(pid=int(data["pid"]), command=str(data.get("command", "")), started_at=started_at)
        except (ValueError, TypeError, KeyError, AttributeError):
            return None

    def is_locked(self) -> bool:
        """Whether some process holds the lock right now, without keeping it (the probe takes and drops it)."""
        if self._fd is not None:
            return True
        try:
            fd = _open(self.path)
        except OSError:
            return False
        try:
            try:
                _lock(fd)
            except OSError as exc:
                if exc.errno in _HELD_ERRNOS:
                    return True
                raise
            _unlock(fd)
            return False
        finally:
            os.close(fd)

    def __enter__(self) -> Self:
        if self._fd is None:
            self.acquire()
        return self

    def __exit__(self, *exc: object) -> None:
        self.release()

    def _write_holder(self, fd: int) -> None:
        record = {"pid": os.getpid(), "command": self.command, "started_at": self._clock().isoformat()}
        payload = json.dumps(record).encode("utf-8")
        os.lseek(fd, 0, os.SEEK_SET)
        os.ftruncate(fd, 0)
        os.write(fd, payload)


@contextmanager
def exclusive_lock(
    path: Path,
    *,
    command: str,
    timeout: float,
    poll_s: float = LOCK_POLL_S,
    sleep: Callable[[float], object] = time.sleep,
) -> Iterator[RunLock]:
    """Hold the same OS-level exclusive lock as the run lock, on any file ``path`` (created when missing), for the
    duration of the ``with`` block. Waits at most ``timeout`` seconds for another holder and raises
    :class:`RunLockHeld` (naming it) when the lock does not come free. ``fm.config_edit`` serialises its
    read-modify-write edits of ``config.toml`` with it, so the platform branch stays in this module."""
    lock = RunLock(path, command=command)
    lock.acquire(blocking=True, timeout=timeout, poll_s=poll_s, sleep=sleep)
    try:
        yield lock
    finally:
        lock.release()


def held_by_parent(holder: LockHolder | None, environ: Mapping[str, str] | None = None) -> bool:
    """Whether the lock is held on this process's behalf: ``environ`` (default ``os.environ``) names, in
    :data:`HELD_BY_ENV`, the very pid ``holder`` (the current lock record) belongs to. ``fm tick`` then runs as the
    tick child of that ``fm run`` instead of skipping."""
    env = os.environ if environ is None else environ
    value = env.get(HELD_BY_ENV, "").strip()
    if holder is None or not value.isdigit():
        return False
    return int(value) == holder.pid


def _open(path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_BINARY", 0)
    return os.open(path, flags, 0o644)


__all__ = [
    "HELD_BY_ENV",
    "LOCK_FILENAME",
    "LOCK_OFFSET",
    "LOCK_POLL_S",
    "LockHolder",
    "RunLock",
    "RunLockHeld",
    "exclusive_lock",
    "held_by_parent",
    "lock_path",
]
