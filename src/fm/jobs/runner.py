"""The ``fm run`` loop (PLUG_AND_PLAY section 4): ticks on a wall-clock timer and approvals in between, one process.

:func:`run_forever` replaces the scheduled ``fm tick`` plus a separately started ``fm bot``:

- **Ticks.** A tick is due by wall clock (``next_tick_at``), never by counting sleeps, so the first iteration after
  the machine slept ticks at once and reports what was missed. The config is reloaded for every tick, so an edit
  from ``fm setup``, the dashboard or ``fm config set-preset`` applies without a restart; a config that will not load
  is logged and alerted, the tick is skipped, and the loop keeps polling with the channel it has.
- **Each tick is a child process with a hard timeout.** On a real machine a scheduled ``fm tick`` hung for five days
  inside a headless browser session and no tick completed after it. :func:`child_tick`, the default tick runner,
  starts ``python -m fm.jobs.runner tick`` (:func:`tick_child_main`, the ``fm tick`` command), appends its stdout and
  stderr to ``logs/run.log``, and waits at most :attr:`RunOptions.tick_timeout` (20 minutes). A tick that overstays
  is stopped with its whole process tree, browser included (:mod:`fm.jobs.proctree`), alerted as "tick timed out
  after N min and was stopped" at most once an hour, and the loop carries on. The child would otherwise skip because
  this process holds the run lock; its environment names this pid (:data:`fm.jobs.runlock.HELD_BY_ENV`), and ``fm
  tick`` runs under the parent's lock only when that pid is the lock's holder.
- **Approvals are recorded here and executed by a tick.** Between ticks the loop polls the phone channel. A poll waits
  server-side (Telegram's long poll, the ntfy subscription) when the next tick is at least
  :attr:`RunOptions.poll_wait_s` away, otherwise it asks for what is waiting and sleeps a little, so a tick is never
  late by more than the channel's own wait. Each press goes through :func:`fm.notify.handle_reply` (the trust
  boundary: nonce, then ``fm.proposals``). Nothing executes in this process: an approval whose deadline is within
  :data:`fm.jobs.tick.NEAR_DEADLINE` brings the next tick forward to now, and that tick child executes it through the
  one write path under the tick's timeout and kill-tree, so a browser that hangs mid-execution can never stall the
  loop (an execution in this process would have no timeout at all).
- **No channel.** When ``config.toml`` names no usable channel the loop just sleeps to the next tick, in short slices
  so ``stop`` and Ctrl+C are answered quickly.
- **Errors never end the loop.** A tick, a poll, a press or a config load that raises is logged with its traceback
  and alerted at most once per :attr:`RunOptions.alert_every` per kind (``tick``, ``timeout``, ``config``, ``poll``,
  ``approval``, ``stuck``). An alert counts as sent only once the channel took it: one that could not be pushed (no
  channel, a channel error) is logged and the next occurrence tries again. Channel errors back off like
  :func:`fm.notify.listen` (2 s doubling to 60 s). Only ``KeyboardInterrupt``, ``SystemExit`` and ``MemoryError`` get
  out, and the first is a clean stop (a tick child that is running then is stopped with its tree).
- **The run lock.** It is held for the whole run and released on the way out, whatever happened. At start the loop
  waits for a holder (a scheduled ``fm tick`` mid-run), but not forever: a holder whose record says ``fm tick`` and
  that started more than :attr:`RunOptions.tick_timeout` plus :data:`STUCK_TICK_GRACE` ago is a tick that hung past
  every timeout (it predates the scheduled tick's own child-process timeout, or its parent hung), so its whole tree
  is stopped (:func:`fm.jobs.proctree.kill_pid_tree`), alerted once ("a stuck fm tick (pid N, started ...) was
  stopped") when the channel opens, and the lock taken.
- **Exit.** Ctrl+C, the ``stop`` event or ``max_ticks`` ends the loop; ``stop`` is checked between steps, so a tick
  that is running finishes or times out first.

Every collaborator is injectable (config loader, store opener, channel opener, tick runner, reply handler, tree
killer, clock, sleep, stop event, lock), so the whole loop runs in a test with fake time, a fake phone and a fake
tick; :func:`in_process_tick` runs :func:`fm.jobs.tick.tick` in this process for tests and the offline journey,
without a timeout. :func:`run_tick_child` is the tick-child machinery itself, shared with the scheduled ``fm tick``,
which runs its work the same way. :func:`run_logging` is the log setup ``fm run`` uses: ``logs/run.log`` under the
config dir, rotating 5 x 1 MB, plus the console when one is attached (``uvw.exe`` starts the loop without one).
"""

from __future__ import annotations

import logging
import os
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Protocol, TextIO

from fm import paths
from fm.browser.flows import FlowRegistry
from fm.config import Config, ConfigError, load_config
from fm.executor import ExecutorOptions, RuntimeOpener
from fm.jobs import proctree
from fm.jobs.runlock import HELD_BY_ENV, LOCK_POLL_S, LockHolder, RunLock, RunLockHeld
from fm.jobs.scheduler_base import DEFAULT_INTERVAL_MINUTES, LOGS_DIRNAME, RUN_LOG_FILENAME, Runner
from fm.jobs.tick import NEAR_DEADLINE, TickOptions, TickReport, tick
from fm.notify import DecisionResult, NotifyChannel, NotifyError, Reply, handle_reply, open_channel, send_alert
from fm.notify.bot import FIRST_BACKOFF_S, MAX_BACKOFF_S
from fm.proposals import pause_state
from fm.store import Store, utc_now

logger = logging.getLogger(__name__)

RUN_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
RUN_LOG_MAX_BYTES = 1_000_000
RUN_LOG_FILES = 5
"""``run.log`` plus its rotated copies: 5 files of at most 1 MB each."""
DEFAULT_TICK_TIMEOUT = timedelta(minutes=20)
"""How long a tick child may run before it is stopped with its process tree (a normal tick takes seconds to a few
minutes: a sync, the due decisions, an execution or two through the browser)."""
TICK_CHILD_MODULE = "fm.jobs.runner"
"""What the tick child runs: ``python -m fm.jobs.runner tick [...]`` (:func:`tick_child_main`)."""
TICK_WAIT_SLICE_S = 1.0
"""The parent waits for a tick child in slices this long, so Ctrl+C is answered within a second on every platform."""
TICK_CRASHED_EXIT = 70
"""The tick child's exit code when ``fm tick`` raised (``EX_SOFTWARE``), told apart from exit 1, a tick that ran and
reported a failure it has already alerted."""
STUCK_TICK_GRACE = timedelta(minutes=2)
"""A ``fm tick`` holding the run lock longer than the tick timeout plus this is stuck, and ``fm run`` stops it."""
_CRITICAL = (KeyboardInterrupt, SystemExit, MemoryError)

type TreeKiller = Callable[[int], Sequence[str]]
"""Stops a process and everything under it by pid, returning what did not go to plan (:func:`fm.jobs.proctree.
kill_pid_tree` by default)."""


class StopSignal(Protocol):
    """What the loop checks between steps: a ``threading.Event`` or anything with ``is_set()``."""

    def is_set(self) -> bool: ...


class TickFunction(Protocol):
    """:func:`fm.jobs.tick.tick` narrowed to how :func:`in_process_tick` calls it."""

    def __call__(
        self,
        store: Store,
        config: Config,
        *,
        channel: NotifyChannel | None,
        opener: RuntimeOpener | None,
        flows: FlowRegistry | None,
        options: TickOptions,
        clock: Callable[[], datetime],
    ) -> TickReport: ...


@dataclass(frozen=True, slots=True)
class TickCall:
    """One tick as the loop asks for it: the loop's store, the config just loaded, the channel it opened, the tick's
    options and the timeout. :func:`child_tick` passes only the options on (the child opens its own store and
    channel); :func:`in_process_tick` uses all of it."""

    store: Store
    config: Config
    channel: NotifyChannel | None
    options: TickOptions
    timeout: timedelta
    clock: Callable[[], datetime]


@dataclass(frozen=True, slots=True)
class TickResult:
    """What one tick came back with."""

    returncode: int | None
    """``0``: the tick ran and everything verified. ``1``: it ran and reported a failure (a decision raised, an
    execution did not verify; the tick alerted it) or refused to start. Anything else: it crashed. ``None``: it
    overran its timeout and was stopped."""
    elapsed: timedelta = timedelta(0)
    lines: tuple[str, ...] = ()
    """Report lines for the loop to log (an in-process tick); a child's output is already in ``run.log``."""
    pid: int | None = None
    """The tick child's pid."""
    notes: tuple[str, ...] = ()
    """What went wrong stopping a child that timed out (a ``taskkill`` failure, a process that would not die)."""

    @property
    def timed_out(self) -> bool:
        return self.returncode is None

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    @property
    def crashed(self) -> bool:
        return self.returncode is not None and self.returncode not in (0, 1)


class TickRunner(Protocol):
    """Runs one tick for the loop: :func:`child_tick` (the default), :func:`in_process_tick`, or a test's fake."""

    def __call__(self, call: TickCall, /) -> TickResult: ...


class ReplyHandler(Protocol):
    """:func:`fm.notify.handle_reply` narrowed to how the loop calls it."""

    def __call__(
        self, store: Store, channel: NotifyChannel, reply: Reply, *, now: datetime | None
    ) -> DecisionResult | None: ...


@dataclass(frozen=True, slots=True)
class RunOptions:
    interval: timedelta = timedelta(minutes=DEFAULT_INTERVAL_MINUTES)
    """Time between ticks (measured from the end of one to the start of the next)."""
    execute: bool = True
    """Execute approved proposals in the ticks (a near-deadline approval brings the next tick forward); off, nothing
    is written and the tick reports what it would have executed."""
    sync: bool = True
    """Let the tick sync a league whose stored state is stale."""
    executor: ExecutorOptions | None = None
    """For :func:`in_process_tick`; a tick child uses the executor's defaults."""
    tick_timeout: timedelta = DEFAULT_TICK_TIMEOUT
    """A tick that runs longer is stopped with its process tree."""
    poll_wait_s: float = 25.0
    """A poll waits server-side only when the next tick is at least this far away (Telegram holds a long poll open
    for 25 s)."""
    poll_gap_s: float = 1.0
    """The pause after a waiting poll, so a channel that answers at once is not polled in a tight loop."""
    idle_poll_s: float = 5.0
    """Closer to a tick than ``poll_wait_s``, polls are non-waiting and this far apart."""
    sleep_slice_s: float = 1.0
    """The longest single sleep: how quickly ``stop`` is noticed."""
    alert_every: timedelta = timedelta(hours=1)
    """A kind of failure (``tick``, ``timeout``, ``config``, ``poll``, ``approval``, ``stuck``) is alerted at most this
    often."""
    max_ticks: int | None = None
    """Stop after this many tick attempts (tests and the offline journey); ``None`` runs until stopped."""


@dataclass(slots=True)
class RunStats:
    """What a run did, returned when the loop ends."""

    ticks: int = 0
    """Ticks that ran to the end (exit 0, or exit 1 with a failure the tick reported itself)."""
    tick_failures: int = 0
    """Ticks that raised or crashed."""
    tick_timeouts: int = 0
    """Ticks stopped at the timeout."""
    config_failures: int = 0
    """Ticks skipped because ``config.toml`` would not load."""
    decisions: int = 0
    """Presses recorded or refused (ignored presses do not count)."""
    expedited: int = 0
    """Near-deadline approvals that brought the next tick forward so it executes them now."""
    stuck_ticks_stopped: int = 0
    """Scheduled ``fm tick`` processes stopped because they held the run lock past every timeout."""
    poll_failures: int = 0
    stopped_by: str = ""
    """``stop``, ``interrupt`` or ``max_ticks``."""
    alerts: list[str] = field(default_factory=list[str])
    """Alert titles the channel took (an alert that could not be pushed is only logged)."""

    @property
    def attempts(self) -> int:
        """Every tick the loop started or skipped: what ``max_ticks`` counts."""
        return self.ticks + self.tick_failures + self.tick_timeouts + self.config_failures


def run_forever(
    *,
    options: RunOptions | None = None,
    stop: StopSignal | None = None,
    lock: RunLock | None = None,
    clock: Callable[[], datetime] = utc_now,
    sleep: Callable[[float], object] = time.sleep,
    config_loader: Callable[[], Config] = load_config,
    store_opener: Callable[[], Store] = Store.open,
    channel_opener: Callable[[Config], NotifyChannel] = open_channel,
    tick_fn: TickRunner | None = None,
    reply_handler: ReplyHandler = handle_reply,
    tree_killer: TreeKiller = proctree.kill_pid_tree,
    log: logging.Logger = logger,
) -> RunStats:
    """Run ticks and approvals until ``stop`` is set, Ctrl+C, or ``options.max_ticks`` tick attempts (the module docs
    give the loop). Takes the run lock first, waiting for a tick that is mid-run (and stopping one that is stuck,
    through ``tree_killer``), and releases it on any exit. Every collaborator defaults to the live one:
    ``config.toml``, ``state.db``, the configured channel, the tick as a child process (:func:`child_tick`). Nothing
    executes in this process; the tick children do."""
    opts = options if options is not None else RunOptions()
    stop_signal = stop if stop is not None else threading.Event()
    run_lock = lock if lock is not None else RunLock(command="fm run")
    stats = RunStats()
    pending = _take_run_lock(
        run_lock, opts, stop=stop_signal, clock=clock, sleep=sleep, tree_killer=tree_killer, stats=stats, log=log
    )
    if pending is None:
        stats.stopped_by = "stop"
        log.info("fm run: stopped while waiting for the run lock")
        return stats
    try:
        with store_opener() as store:
            loop = _Loop(
                store,
                opts,
                stats=stats,
                pending_alerts=pending,
                stop=stop_signal,
                clock=clock,
                sleep=sleep,
                config_loader=config_loader,
                channel_opener=channel_opener,
                tick_fn=tick_fn if tick_fn is not None else child_tick,
                reply_handler=reply_handler,
                log=log,
            )
            return loop.run()
    finally:
        run_lock.release()


type _Alert = tuple[str, str, str]
"""``(kind, title, body)`` of an alert raised before the loop has a channel to push it on."""


def _take_run_lock(
    lock: RunLock,
    options: RunOptions,
    *,
    stop: StopSignal,
    clock: Callable[[], datetime],
    sleep: Callable[[float], object],
    tree_killer: TreeKiller,
    stats: RunStats,
    log: logging.Logger,
) -> list[_Alert] | None:
    """Take the run lock, waiting for its holder in :data:`fm.jobs.runlock.LOCK_POLL_S` steps. A holder that is a
    stuck ``fm tick`` (:func:`_stuck`) is stopped with its tree, once, and the wait goes on until the OS frees the
    lock. Returns the alerts to push once the channel is open, or ``None`` when ``stop`` was set while waiting."""
    alerts: list[_Alert] = []
    stopped: set[int] = set()
    waiting = False
    while True:
        try:
            lock.acquire()
            return alerts
        except RunLockHeld as exc:
            holder = exc.holder
        if not waiting:
            waiting = True
            log.info(
                "waiting for the run lock held by %s", holder.describe() if holder is not None else "another process"
            )
        if holder is not None and holder.pid not in stopped and _stuck(holder, options, clock()):
            stopped.add(holder.pid)
            alerts.append(_stop_stuck_tick(holder, options, tree_killer, stats, log))
            continue  # the kill usually frees the lock at once
        if stop.is_set():
            return None
        sleep(LOCK_POLL_S)


def _stuck(holder: LockHolder, options: RunOptions, now: datetime) -> bool:
    """A scheduled ``fm tick`` that has held the lock longer than any tick may run. Never this process itself, and
    never a holder of another kind (``fm run`` elsewhere, ``fm doctor``, a config edit)."""
    if holder.command != "fm tick" or holder.pid == os.getpid():
        return False
    return now - holder.started_at > options.tick_timeout + STUCK_TICK_GRACE


def _stop_stuck_tick(
    holder: LockHolder, options: RunOptions, tree_killer: TreeKiller, stats: RunStats, log: logging.Logger
) -> _Alert:
    started = f"{holder.started_at.astimezone(UTC):%Y-%m-%d %H:%M} UTC"
    what = f"a stuck fm tick (pid {holder.pid}, started {started}) was stopped"
    log.error(
        "fm run: %s: it held the run lock past the %s tick timeout; stopping it and every process under it",
        what,
        _span(options.tick_timeout),
    )
    try:
        notes = list(tree_killer(holder.pid))
    except _CRITICAL:
        raise
    except Exception as exc:  # a kill helper that blew up must not keep the loop from starting
        notes = [f"{type(exc).__name__}: {exc}"]
    for note in notes:
        log.warning("fm run: %s", note)
    stats.stuck_ticks_stopped += 1
    body = (
        f"{what}: it held the run lock for more than {_span(options.tick_timeout + STUCK_TICK_GRACE)}, so its "
        f"processes (browser included) were killed and fm run took over. See logs/{RUN_LOG_FILENAME}."
    )
    return ("stuck", "fm run: stuck tick stopped", body)


class _Loop:
    def __init__(
        self,
        store: Store,
        options: RunOptions,
        *,
        stats: RunStats,
        pending_alerts: list[_Alert],
        stop: StopSignal,
        clock: Callable[[], datetime],
        sleep: Callable[[float], object],
        config_loader: Callable[[], Config],
        channel_opener: Callable[[Config], NotifyChannel],
        tick_fn: TickRunner,
        reply_handler: ReplyHandler,
        log: logging.Logger,
    ) -> None:
        self.store = store
        self.options = options
        self.stop = stop
        self.clock = clock
        self.sleep = sleep
        self.config_loader = config_loader
        self.channel_opener = channel_opener
        self.tick_fn = tick_fn
        self.reply_handler = reply_handler
        self.log = log
        self.stats = stats
        self.pending_alerts = pending_alerts
        self.channel: NotifyChannel | None = None
        self.channel_key: tuple[object, object] | None = None
        self.channel_problem: str | None = None
        self.alerted: dict[str, datetime] = {}
        self.next_tick_at = clock()

    # --- the loop

    def run(self) -> RunStats:
        self.log.info(
            "fm run: tick every %s (timeout %s)%s",
            _describe(self.options.interval),
            _span(self.options.tick_timeout),
            "" if self.options.execute else " (no execution)",
        )
        self.next_tick_at = self.clock()
        backoff = FIRST_BACKOFF_S
        try:
            while not self.stop.is_set():
                now = self.clock()
                if now >= self.next_tick_at:
                    self._tick()
                    self.next_tick_at = self.clock() + self.options.interval
                    if self.options.max_ticks is not None and self.stats.attempts >= self.options.max_ticks:
                        self.stats.stopped_by = "max_ticks"
                        break
                    continue
                if self.channel is None:
                    self._pause(min(self._until(self.next_tick_at), self.options.sleep_slice_s))
                    continue
                wait = self._until(self.next_tick_at) >= self.options.poll_wait_s
                try:
                    replies = self.channel.poll(wait=wait)
                except NotifyError as exc:
                    self._poll_failed(f"{self.channel.name}: {exc}", backoff)
                    self._pause(min(backoff, self._until(self.next_tick_at)))
                    backoff = min(backoff * 2, MAX_BACKOFF_S)
                    continue
                except _CRITICAL:
                    raise
                except Exception as exc:
                    self.log.exception("fm run: polling %s failed", self.channel.name)
                    self._poll_failed(f"{self.channel.name}: {type(exc).__name__}: {exc}", backoff)
                    self._pause(min(backoff, self._until(self.next_tick_at)))
                    backoff = min(backoff * 2, MAX_BACKOFF_S)
                    continue
                backoff = FIRST_BACKOFF_S
                for reply in replies:
                    self._handle(reply)
                gap = self.options.poll_gap_s if wait else self.options.idle_poll_s
                self._pause(min(gap, self._until(self.next_tick_at)))
            else:
                self.stats.stopped_by = "stop"
        except KeyboardInterrupt:
            self.stats.stopped_by = "interrupt"
        finally:
            self._close_channel()
        self.log.info(
            "fm run: stopped (%s) after %d ticks and %d decisions",
            self.stats.stopped_by,
            self.stats.ticks,
            self.stats.decisions,
        )
        return self.stats

    def _until(self, moment: datetime) -> float:
        return max((moment - self.clock()).total_seconds(), 0.0)

    def _pause(self, seconds: float) -> None:
        """Sleep ``seconds`` in slices of at most ``sleep_slice_s``, giving up as soon as ``stop`` is set."""
        left = seconds
        while left > 0 and not self.stop.is_set():
            step = min(left, self.options.sleep_slice_s)
            self.sleep(step)
            left -= step

    # --- ticks

    def _tick(self) -> None:
        config = self._load_config()
        if config is None:
            return
        self._refresh_channel(config)
        pending, self.pending_alerts = self.pending_alerts, []
        for kind, title, body in pending:  # raised before there was a channel (a stuck tick stopped at start)
            self._alert(kind, title, body)
        call = TickCall(
            store=self.store,
            config=config,
            channel=self.channel,
            options=TickOptions(execute=self.options.execute, sync=self.options.sync, executor=self.options.executor),
            timeout=self.options.tick_timeout,
            clock=self.clock,
        )
        try:
            result = self.tick_fn(call)
        except _CRITICAL:
            raise
        except Exception as exc:
            self.stats.tick_failures += 1
            self.log.exception("fm run: tick failed")
            self._alert(
                "tick",
                "fm run: tick failed",
                f"{type(exc).__name__}: {exc}\nThe loop continues; see logs/{RUN_LOG_FILENAME}.",
            )
            return
        for line in result.lines:
            self.log.info("%s", line)
        for note in result.notes:
            self.log.warning("fm run: %s", note)
        took = _span(result.elapsed)
        if result.timed_out:
            self.stats.tick_timeouts += 1
            what = f"tick timed out after {_span(call.timeout)} and was stopped"
            self.log.error("fm run: %s (pid %s and every process under it killed)", what, result.pid)
            self._alert(
                "timeout",
                "fm run: tick timed out",
                f"{what}; its processes (browser included) were killed. The loop continues and the next tick runs on "
                f"time; see logs/{RUN_LOG_FILENAME}.",
            )
        elif result.crashed:
            self.stats.tick_failures += 1
            self.log.error("fm run: tick crashed (exit %s) after %s", result.returncode, took)
            self._alert(
                "tick",
                "fm run: tick failed",
                f"The tick exited with code {result.returncode}. The loop continues; see logs/{RUN_LOG_FILENAME}.",
            )
        elif result.ok:
            self.stats.ticks += 1
            self.log.info("fm run: tick done in %s", took)
        else:
            self.stats.ticks += 1  # it ran; what failed in it was reported (and alerted) by the tick
            self.log.warning("fm run: tick finished with a failure (exit %s) in %s", result.returncode, took)

    def _load_config(self) -> Config | None:
        try:
            return self.config_loader()
        except _CRITICAL:
            raise
        except Exception as exc:
            detail = str(exc) if isinstance(exc, ConfigError) else f"{type(exc).__name__}: {exc}"
            self.stats.config_failures += 1
            self.log.error("fm run: config not loaded, tick skipped: %s", detail)
            self._alert(
                "config", "fm run: config.toml not loaded", f"{detail}\nFix it; the loop retries at the next tick."
            )
            return None

    def _refresh_channel(self, config: Config) -> None:
        """Open the channel ``config`` names, or reopen it when the notify settings or secrets changed since."""
        key = (config.notify, config.secrets)
        if key == self.channel_key and (self.channel is not None or self.channel_problem is not None):
            return
        self._close_channel()
        self.channel_key = key
        try:
            self.channel = self.channel_opener(config)
        except _CRITICAL:
            raise
        except Exception as exc:
            problem = str(exc) if isinstance(exc, NotifyError) else f"{type(exc).__name__}: {exc}"
            if problem != self.channel_problem:
                self.log.warning(
                    "fm run: no phone channel (%s); proposals wait in the store for fm proposals or the dashboard, "
                    "and alerts stay in this log",
                    problem,
                )
            self.channel_problem = problem
            return
        self.channel_problem = None
        self.log.info("fm run: listening for approvals on %s", self.channel.name)

    def _close_channel(self) -> None:
        channel, self.channel = self.channel, None
        if channel is None:
            return
        try:
            channel.close()
        except Exception as exc:  # closing is best effort
            self.log.warning("fm run: closing the %s channel failed: %s", channel.name, exc)

    # --- approvals

    def _handle(self, reply: Reply) -> None:
        if self.channel is None:
            return
        try:
            result = self.reply_handler(self.store, self.channel, reply, now=self.clock())
        except sqlite3.Error as exc:
            self.log.error(
                "fm run: could not record the press on proposal #%d (press again): %s", reply.proposal_id, exc
            )
            return
        except _CRITICAL:
            raise
        except Exception as exc:
            self.log.exception("fm run: the press on proposal #%d failed", reply.proposal_id)
            self._alert(
                "approval",
                f"fm run: press on #{reply.proposal_id} not recorded",
                f"{type(exc).__name__}: {exc}\nPress again, or decide with fm proposals.",
            )
            return
        if result is None:
            return
        self.stats.decisions += 1
        self.log.info("fm run: %s", result.describe())
        self._expedite(result)

    def _expedite(self, result: DecisionResult) -> None:
        """Bring the next tick forward to now for an approval whose deadline is within
        :data:`fm.jobs.tick.NEAR_DEADLINE`, so the tick child executes it at once under the tick timeout. The same
        conditions as :func:`fm.jobs.tick.execute_on_approval`: approved, tokened, deadline ahead, not paused."""
        if not self.options.execute or not result.ok or result.decision != "approve":
            return
        row = self.store.proposals.get(result.proposal_id)
        if row is None or row.status != "approved" or row.execution_token is None or row.deadline is None:
            return
        now = self.clock()
        if row.deadline <= now or row.deadline - now > NEAR_DEADLINE or pause_state() is not None:
            return
        if self.next_tick_at > now:
            self.next_tick_at = now
            self.stats.expedited += 1
            self.log.info(
                "fm run: #%d is due in %s: ticking now so it executes under the tick timeout",
                row.row_id,
                _span(row.deadline - now),
            )

    # --- failures

    def _poll_failed(self, detail: str, backoff: float) -> None:
        self.stats.poll_failures += 1
        self.log.warning("fm run: poll failed: %s; retrying in %.0fs", detail, backoff)
        self._alert(
            "poll", "fm run: phone channel failing", f"{detail}\nRetrying with backoff; presses wait on the server."
        )

    def _alert(self, kind: str, title: str, body: str) -> None:
        """Push an alert for ``kind`` at most once per ``alert_every``. The time is recorded only once the channel
        took the push: an alert that could not go out (no channel, a channel error) is logged, and the next
        occurrence tries again rather than being silenced for an hour."""
        now = self.clock()
        last = self.alerted.get(kind)
        if last is not None and now - last < self.options.alert_every:
            return
        if self.channel is None:
            self.log.warning("fm run: alert not pushed (no phone channel): %s: %s", title, " ".join(body.split()))
            return
        try:
            send_alert(self.channel, title, body)
        except NotifyError as exc:
            self.log.warning("fm run: alert not pushed (%s): %s", title, exc)
            return
        except _CRITICAL:
            raise
        except Exception:
            self.log.exception("fm run: alert not pushed (%s)", title)
            return
        self.alerted[kind] = now
        self.stats.alerts.append(title)


def _describe(interval: timedelta) -> str:
    seconds = int(interval.total_seconds())
    if seconds % 60 == 0:
        minutes = seconds // 60
        return f"{minutes} minute{'' if minutes == 1 else 's'}"
    return f"{seconds} s"


def _span(span: timedelta) -> str:
    """``20 min``, ``2 min 30 s``, ``41 s``, ``0.5 s``: a timeout or a duration in a log line or an alert."""
    total = span.total_seconds()
    if total < 10 and total != int(total):
        return f"{total:.1f} s"
    seconds = round(total)
    minutes, rest = divmod(seconds, 60)
    if not minutes:
        return f"{rest} s"
    return f"{minutes} min" if not rest else f"{minutes} min {rest} s"


# --- the tick runners -------------------------------------------------------------------------------------------------


def tick_child_argv(options: TickOptions, *, python: str | None = None) -> list[str]:
    """``<python> -m fm.jobs.runner tick [--no-execute] [--no-sync] [--league K ...]``: ``fm tick`` with
    ``options``, run by this interpreter (the venv's, so the child imports the same ``fm``)."""
    argv = [python if python is not None else sys.executable, "-m", TICK_CHILD_MODULE, "tick"]
    if not options.execute:
        argv.append("--no-execute")
    if not options.sync:
        argv.append("--no-sync")
    for key in options.leagues or ():
        argv += ["--league", key]
    return argv


def child_tick(
    call: TickCall,
    /,
    *,
    argv: Sequence[str] | None = None,
    log_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    runner: Runner = proctree.quiet_runner,
    log: logging.Logger = logger,
) -> TickResult:
    """The live tick runner of ``fm run``: :func:`run_tick_child` with ``call``'s options and timeout, the child's
    output appended to ``log_path`` (default ``logs/run.log``)."""
    return run_tick_child(
        call.options,
        timeout=call.timeout,
        argv=argv,
        log_path=log_path if log_path is not None else run_log_path(),
        environ=environ,
        runner=runner,
        log=log,
    )


def run_tick_child(
    options: TickOptions,
    *,
    timeout: timedelta,
    argv: Sequence[str] | None = None,
    log_path: Path | None = None,
    environ: Mapping[str, str] | None = None,
    runner: Runner = proctree.quiet_runner,
    log: logging.Logger = logger,
) -> TickResult:
    """``fm tick`` as a child process (``argv``, default :func:`tick_child_argv` of ``options``), stopped with its
    whole process tree when it runs past ``timeout`` (:mod:`fm.jobs.proctree`; ``runner`` runs the kill helpers): the
    one way both ``fm run`` and the scheduled ``fm tick`` run a tick. Its stdout and stderr are appended to
    ``log_path``, or go to this process's own when it is ``None`` (the scheduled tick's wrapper log). The child's
    environment is ``environ`` (default this process's) plus :data:`fm.jobs.runlock.HELD_BY_ENV` naming this pid, so
    the tick runs under the run lock this process holds instead of skipping, and unbuffered UTF-8 output, so nothing
    is lost when it is killed. Waits on the process only, never on a pipe a grandchild could hold open. Ctrl+C while
    it runs stops the tree before the interrupt goes on."""
    command = list(argv) if argv is not None else tick_child_argv(options)
    env = dict(os.environ if environ is None else environ)
    env[HELD_BY_ENV] = str(os.getpid())
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    log.info("tick child starting (stopped after %s)", _span(timeout))
    started = time.monotonic()
    output = proctree.open_append(log_path) if log_path is not None else None
    try:
        tree = proctree.spawn(command, env=env, output=output)
    finally:
        if output is not None:
            os.close(output)
    deadline = started + max(timeout.total_seconds(), 0.0)
    notes: list[str] = []
    returncode: int | None = None
    try:
        while True:
            # In slices: on Windows a single long wait cannot be interrupted, and Ctrl+C would wait for the timeout.
            try:
                returncode = tree.wait(min(TICK_WAIT_SLICE_S, max(deadline - time.monotonic(), 0.0)))
                break
            except subprocess.TimeoutExpired:
                if time.monotonic() >= deadline:
                    notes = tree.kill(runner=runner)
                    break
    except BaseException:
        tree.kill(runner=runner)  # Ctrl+C (or SIGTERM) while the tick runs: never leave its tree behind
        raise
    return TickResult(
        returncode=returncode,
        elapsed=timedelta(seconds=time.monotonic() - started),
        pid=tree.pid,
        notes=tuple(notes),
    )


def in_process_tick(
    *, tick_fn: TickFunction = tick, opener: RuntimeOpener | None = None, flows: FlowRegistry | None = None
) -> TickRunner:
    """A tick runner that calls ``tick_fn`` (default :func:`fm.jobs.tick.tick`) in this process with the loop's
    store, config and channel, ``opener`` and ``flows``: what tests and the offline journey use. It has no timeout
    (nothing can stop a call in the same process), which is why ``fm run`` never uses it."""

    def run(call: TickCall, /) -> TickResult:
        started = time.monotonic()
        report = tick_fn(
            call.store,
            call.config,
            channel=call.channel,
            opener=opener,
            flows=flows,
            options=call.options,
            clock=call.clock,
        )
        return TickResult(
            returncode=0 if report.ok else 1,
            elapsed=timedelta(seconds=time.monotonic() - started),
            lines=tuple(report.lines()),
        )

    return run


def tick_child_main(args: Sequence[str] | None = None) -> int:
    """The tick child's entry point (``python -m fm.jobs.runner tick [...]``): the ``fm`` CLI with ``args`` (default
    ``sys.argv[1:]``; :func:`child_tick` always passes ``tick`` and its options), ``fm`` log lines at INFO on stderr
    in the ``run.log`` format, and the exit code returned. An exception the CLI lets through is printed and becomes
    :data:`TICK_CRASHED_EXIT`, so the loop tells a crash from a tick that reported a failure (exit 1)."""
    from fm.cli import app  # deferred: fm.cli imports fm.commands.run, which imports this module

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(RUN_LOG_FORMAT))
    fm_logger = logging.getLogger("fm")
    previous = fm_logger.level
    fm_logger.addHandler(handler)
    if previous == logging.NOTSET or previous > logging.INFO:
        fm_logger.setLevel(logging.INFO)
    try:
        app(args=list(sys.argv[1:] if args is None else args), prog_name="fm", standalone_mode=True)
    except SystemExit as exc:
        return _exit_code(exc.code)
    except Exception:
        traceback.print_exc()
        return TICK_CRASHED_EXIT
    finally:
        fm_logger.removeHandler(handler)
        fm_logger.setLevel(previous)
    return 0


def _exit_code(code: object) -> int:
    if code is None:
        return 0
    if isinstance(code, int):
        return code
    print(code, file=sys.stderr)  # sys.exit("message") prints it and exits 1
    return 1


# --- logging ----------------------------------------------------------------------------------------------------------


def run_log_path() -> Path:
    """``$FM_CONFIG_DIR/logs/run.log``."""
    return paths.config_dir() / LOGS_DIRNAME / RUN_LOG_FILENAME


@contextmanager
def run_logging(
    path: Path | None = None, *, console: TextIO | None = None, level: int = logging.INFO
) -> Iterator[list[logging.Handler]]:
    """Send every ``fm`` log line to ``path`` (default :func:`run_log_path`), rotating at :data:`RUN_LOG_MAX_BYTES`
    into :data:`RUN_LOG_FILES` files, and to ``console`` when one is given (``None`` under ``uvw.exe`` or
    ``pythonw``, which have no streams). The handlers are removed and closed on exit."""
    target = path if path is not None else run_log_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    formatter = logging.Formatter(RUN_LOG_FORMAT)
    handlers: list[logging.Handler] = [
        RotatingFileHandler(target, maxBytes=RUN_LOG_MAX_BYTES, backupCount=RUN_LOG_FILES - 1, encoding="utf-8")
    ]
    if console is not None:
        handlers.append(logging.StreamHandler(console))
    root = logging.getLogger("fm")
    previous = root.level
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(level)
        root.addHandler(handler)
    try:
        yield handlers
    finally:
        for handler in handlers:
            root.removeHandler(handler)
            handler.close()
        root.setLevel(previous)


__all__ = [
    "DEFAULT_TICK_TIMEOUT",
    "RUN_LOG_FILES",
    "RUN_LOG_FORMAT",
    "RUN_LOG_MAX_BYTES",
    "STUCK_TICK_GRACE",
    "TICK_CHILD_MODULE",
    "TICK_CRASHED_EXIT",
    "TICK_WAIT_SLICE_S",
    "ReplyHandler",
    "RunOptions",
    "RunStats",
    "StopSignal",
    "TickCall",
    "TickFunction",
    "TickResult",
    "TickRunner",
    "TreeKiller",
    "child_tick",
    "in_process_tick",
    "run_forever",
    "run_log_path",
    "run_logging",
    "run_tick_child",
    "tick_child_argv",
    "tick_child_main",
]


if __name__ == "__main__":
    raise SystemExit(tick_child_main())
