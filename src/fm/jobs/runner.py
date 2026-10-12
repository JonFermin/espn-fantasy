"""The ``fm run`` loop (PLUG_AND_PLAY section 4): ticks on a wall-clock timer and approvals in between, one process.

:func:`run_forever` replaces the scheduled ``fm tick`` plus a separately started ``fm bot``:

- **Ticks.** A tick is due by wall clock (``next_tick_at``), never by counting sleeps, so the first iteration after
  the machine slept ticks at once and reports what was missed. The config is reloaded for every tick, so an edit
  from ``fm setup``, the dashboard or ``fm config set-preset`` applies without a restart; a config that will not load
  is logged and alerted, and the loop keeps polling with the channel it has.
- **Approvals.** Between ticks the loop polls the phone channel. A poll waits server-side (Telegram's long poll, the
  ntfy subscription) when the next tick is at least :attr:`RunOptions.poll_wait_s` away, otherwise it asks for what is
  waiting and sleeps a little, so a tick is never late by more than the channel's own wait. Each press goes through
  :func:`fm.notify.handle_reply` (the trust boundary: nonce, then ``fm.proposals``) and then
  :func:`fm.jobs.tick.execute_on_approval`, so an approval close to its deadline executes immediately through the
  one write path instead of waiting for the next tick. ``fm bot`` never wired that hook in.
- **No channel.** When ``config.toml`` names no usable channel the loop just sleeps to the next tick, in short slices
  so ``stop`` and Ctrl+C are answered quickly.
- **Errors never end the loop.** A tick or a poll that raises is logged with its traceback and alerted at most once
  per :attr:`RunOptions.alert_every` per kind (``tick``, ``config``, ``poll``, ``approval``); channel errors back off
  like :func:`fm.notify.listen` (2 s doubling to 60 s). Only ``KeyboardInterrupt``, ``SystemExit`` and
  ``MemoryError`` get out, and the first is a clean stop.
- **Exit.** Ctrl+C, the ``stop`` event or ``max_ticks`` ends the loop; the run lock (:mod:`fm.jobs.runlock`) is held
  for the whole run and released on the way out, whatever happened.

Every collaborator is injectable (config loader, store opener, channel opener, tick function, approval hook factory,
reply handler, runtime opener, clock, sleep, stop event, lock), so the whole loop runs in a test with fake time and
a fake phone. :func:`run_logging` is the log setup ``fm run`` uses: ``logs/run.log`` under the config dir, rotating
5 x 1 MB, plus the console when one is attached (``uvw.exe`` starts the loop without one).
"""

from __future__ import annotations

import logging
import sqlite3
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Protocol, TextIO

from fm import paths
from fm.browser.flows import FlowRegistry
from fm.config import Config, ConfigError, load_config
from fm.executor import ExecutorOptions, RuntimeOpener, live_opener
from fm.jobs.runlock import LockHolder, RunLock
from fm.jobs.scheduler_base import DEFAULT_INTERVAL_MINUTES, LOGS_DIRNAME, RUN_LOG_FILENAME
from fm.jobs.tick import ExecutionOutcome, TickOptions, TickReport, execute_on_approval, tick
from fm.notify import DecisionResult, Message, NotifyChannel, NotifyError, Reply, handle_reply, open_channel, send_alert
from fm.notify.bot import FIRST_BACKOFF_S, MAX_BACKOFF_S
from fm.notify.messages import proposal_name
from fm.store import Store, utc_now

logger = logging.getLogger(__name__)

RUN_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
RUN_LOG_MAX_BYTES = 1_000_000
RUN_LOG_FILES = 5
"""``run.log`` plus its rotated copies: 5 files of at most 1 MB each."""
_CRITICAL = (KeyboardInterrupt, SystemExit, MemoryError)


class StopSignal(Protocol):
    """What the loop checks between steps: a ``threading.Event`` or anything with ``is_set()``."""

    def is_set(self) -> bool: ...


class TickRunner(Protocol):
    """:func:`fm.jobs.tick.tick` narrowed to how the loop calls it."""

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


class ApprovalHookFactory(Protocol):
    """:func:`fm.jobs.tick.execute_on_approval` narrowed to how the loop builds the hook."""

    def __call__(
        self,
        store: Store,
        *,
        opener: RuntimeOpener,
        flows: FlowRegistry | None,
        options: ExecutorOptions | None,
        clock: Callable[[], datetime],
        on_outcome: Callable[[ExecutionOutcome], object] | None,
    ) -> Callable[[DecisionResult], object]: ...


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
    """Execute approved proposals (in ticks and at once after a near-deadline approval); off, nothing is written and
    the tick reports what it would have executed."""
    sync: bool = True
    """Let the tick sync a league whose stored state is stale."""
    executor: ExecutorOptions | None = None
    poll_wait_s: float = 25.0
    """A poll waits server-side only when the next tick is at least this far away (Telegram holds a long poll open
    for 25 s)."""
    idle_poll_s: float = 5.0
    """Closer to a tick than ``poll_wait_s``, polls are non-waiting and this far apart."""
    sleep_slice_s: float = 1.0
    """The longest single sleep: how quickly ``stop`` is noticed."""
    alert_every: timedelta = timedelta(hours=1)
    """A kind of failure (``tick``, ``config``, ``poll``, ``approval``) is alerted at most this often."""
    max_ticks: int | None = None
    """Stop after this many ticks (tests and the offline journey); ``None`` runs until stopped."""


@dataclass(slots=True)
class RunStats:
    """What a run did, returned when the loop ends."""

    ticks: int = 0
    """Ticks that completed (a report came back)."""
    tick_failures: int = 0
    """Ticks that raised."""
    config_failures: int = 0
    """Ticks skipped because ``config.toml`` would not load."""
    decisions: int = 0
    """Presses recorded or refused (ignored presses do not count)."""
    executions: int = 0
    """Executions run at once after an approval."""
    poll_failures: int = 0
    stopped_by: str = ""
    """``stop``, ``interrupt`` or ``max_ticks``."""
    alerts: list[str] = field(default_factory=list[str])
    """Alert titles pushed (or that would have been, without a channel)."""


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
    tick_fn: TickRunner = tick,
    approval_hook: ApprovalHookFactory = execute_on_approval,
    reply_handler: ReplyHandler = handle_reply,
    opener: RuntimeOpener | None = None,
    flows: FlowRegistry | None = None,
    log: logging.Logger = logger,
) -> RunStats:
    """Run ticks and approvals until ``stop`` is set, Ctrl+C, or ``options.max_ticks`` ticks (the module docs give
    the loop). Takes the run lock first, waiting for a tick that is mid-run, and releases it on any exit. Every
    collaborator defaults to the live one: ``config.toml``, ``state.db``, the configured channel, the real tick,
    :func:`fm.executor.live_opener` (the browser opens only when something executes)."""
    opts = options if options is not None else RunOptions()
    stop_signal = stop if stop is not None else threading.Event()
    run_lock = lock if lock is not None else RunLock(command="fm run")
    run_lock.acquire(blocking=True, sleep=sleep, on_wait=lambda holder: _log_wait(log, holder))
    try:
        with store_opener() as store:
            loop = _Loop(
                store,
                opts,
                stop=stop_signal,
                clock=clock,
                sleep=sleep,
                config_loader=config_loader,
                channel_opener=channel_opener,
                tick_fn=tick_fn,
                approval_hook=approval_hook,
                reply_handler=reply_handler,
                opener=opener if opener is not None else live_opener(),
                flows=flows,
                log=log,
            )
            return loop.run()
    finally:
        run_lock.release()


def _log_wait(log: logging.Logger, holder: LockHolder | None) -> None:
    log.info("waiting for the run lock held by %s", holder.describe() if holder is not None else "another process")


class _Loop:
    def __init__(
        self,
        store: Store,
        options: RunOptions,
        *,
        stop: StopSignal,
        clock: Callable[[], datetime],
        sleep: Callable[[float], object],
        config_loader: Callable[[], Config],
        channel_opener: Callable[[Config], NotifyChannel],
        tick_fn: TickRunner,
        approval_hook: ApprovalHookFactory,
        reply_handler: ReplyHandler,
        opener: RuntimeOpener,
        flows: FlowRegistry | None,
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
        self.opener = opener
        self.flows = flows
        self.log = log
        self.stats = RunStats()
        self.channel: NotifyChannel | None = None
        self.channel_key: tuple[object, object] | None = None
        self.channel_problem: str | None = None
        self.alerted: dict[str, datetime] = {}
        self.on_result: Callable[[DecisionResult], object] | None = None
        if options.execute:
            self.on_result = approval_hook(
                store, opener=opener, flows=flows, options=options.executor, clock=clock, on_outcome=self._outcome
            )

    # --- the loop

    def run(self) -> RunStats:
        self.log.info(
            "fm run: tick every %s%s",
            _describe(self.options.interval),
            "" if self.options.execute else " (no execution)",
        )
        next_tick_at = self.clock()
        backoff = FIRST_BACKOFF_S
        try:
            while not self.stop.is_set():
                now = self.clock()
                if now >= next_tick_at:
                    self._tick(now)
                    next_tick_at = self.clock() + self.options.interval
                    done = self.stats.ticks + self.stats.tick_failures + self.stats.config_failures
                    if self.options.max_ticks is not None and done >= self.options.max_ticks:
                        self.stats.stopped_by = "max_ticks"
                        break
                    continue
                remaining = (next_tick_at - now).total_seconds()
                if self.channel is None:
                    self._pause(min(remaining, self.options.sleep_slice_s))
                    continue
                wait = remaining >= self.options.poll_wait_s
                try:
                    replies = self.channel.poll(wait=wait)
                except NotifyError as exc:
                    self._poll_failed(f"{self.channel.name}: {exc}", backoff)
                    self._pause(backoff)
                    backoff = min(backoff * 2, MAX_BACKOFF_S)
                    continue
                except _CRITICAL:
                    raise
                except Exception as exc:
                    self.log.exception("fm run: polling %s failed", self.channel.name)
                    self._poll_failed(f"{self.channel.name}: {type(exc).__name__}: {exc}", backoff)
                    self._pause(backoff)
                    backoff = min(backoff * 2, MAX_BACKOFF_S)
                    continue
                backoff = FIRST_BACKOFF_S
                for reply in replies:
                    self._handle(reply)
                if not wait:
                    self._pause(min(remaining, self.options.idle_poll_s))
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

    def _pause(self, seconds: float) -> None:
        """Sleep ``seconds`` in slices of at most ``sleep_slice_s``, giving up as soon as ``stop`` is set."""
        left = seconds
        while left > 0 and not self.stop.is_set():
            step = min(left, self.options.sleep_slice_s)
            self.sleep(step)
            left -= step

    # --- ticks

    def _tick(self, now: datetime) -> None:
        try:
            config = self.config_loader()
        except ConfigError as exc:
            self.stats.config_failures += 1
            self.log.error("fm run: config not loaded, tick skipped: %s", exc)
            self._alert(
                "config", "fm run: config.toml not loaded", f"{exc}\nFix it; the loop retries at the next tick."
            )
            return
        self._refresh_channel(config)
        options = TickOptions(execute=self.options.execute, sync=self.options.sync, executor=self.options.executor)
        try:
            report = self.tick_fn(
                self.store,
                config,
                channel=self.channel,
                opener=self.opener,
                flows=self.flows,
                options=options,
                clock=self.clock,
            )
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
        self.stats.ticks += 1
        for line in report.lines():
            self.log.info("%s", line)

    def _refresh_channel(self, config: Config) -> None:
        """Open the channel ``config`` names, or reopen it when the notify settings or secrets changed since."""
        key = (config.notify, config.secrets)
        if key == self.channel_key and (self.channel is not None or self.channel_problem is not None):
            return
        self._close_channel()
        self.channel_key = key
        try:
            self.channel = self.channel_opener(config)
        except NotifyError as exc:
            problem = str(exc)
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
        if result is None:
            return
        self.stats.decisions += 1
        self.log.info("fm run: %s", result.describe())
        if self.on_result is None:
            return
        try:
            self.on_result(result)
        except _CRITICAL:
            raise
        except Exception as exc:
            self.log.exception("fm run: executing after the approval of #%d failed", result.proposal_id)
            self._alert(
                "approval",
                f"fm run: execution of #{result.proposal_id} failed",
                f"{type(exc).__name__}: {exc}\nThe tick retries while the deadline holds.",
            )

    def _outcome(self, outcome: ExecutionOutcome) -> None:
        self.stats.executions += 1
        self.log.info("fm run: executed %s", outcome.describe())
        if self.channel is None:
            return
        name = proposal_name(outcome.league_key, outcome.kind, outcome.proposal_id)
        try:
            if outcome.ok:
                self.channel.send(Message(title=f"✅ Done: {name}", body=f"{outcome.summary}\nVerified on ESPN."))
            else:
                send_alert(self.channel, f"❌ Failed: {name}", f"{outcome.summary}\n{outcome.detail}")
        except NotifyError as exc:
            self.log.warning("fm run: execution of #%d not reported: %s", outcome.proposal_id, exc)

    # --- failures

    def _poll_failed(self, detail: str, backoff: float) -> None:
        self.stats.poll_failures += 1
        self.log.warning("fm run: poll failed: %s; retrying in %.0fs", detail, backoff)
        self._alert(
            "poll", "fm run: phone channel failing", f"{detail}\nRetrying with backoff; presses wait on the server."
        )

    def _alert(self, kind: str, title: str, body: str) -> None:
        """Push an alert for ``kind`` at most once per ``alert_every``; a channel that cannot deliver is logged."""
        now = self.clock()
        last = self.alerted.get(kind)
        if last is not None and now - last < self.options.alert_every:
            return
        self.alerted[kind] = now
        self.stats.alerts.append(title)
        if self.channel is None:
            return
        try:
            send_alert(self.channel, title, body)
        except NotifyError as exc:
            self.log.warning("fm run: alert not pushed (%s): %s", title, exc)
        except _CRITICAL:
            raise
        except Exception:
            self.log.exception("fm run: alert not pushed (%s)", title)


def _describe(interval: timedelta) -> str:
    seconds = int(interval.total_seconds())
    if seconds % 60 == 0:
        minutes = seconds // 60
        return f"{minutes} minute{'' if minutes == 1 else 's'}"
    return f"{seconds} s"


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
    "RUN_LOG_FILES",
    "RUN_LOG_FORMAT",
    "RUN_LOG_MAX_BYTES",
    "ApprovalHookFactory",
    "ReplyHandler",
    "RunOptions",
    "RunStats",
    "StopSignal",
    "TickRunner",
    "run_forever",
    "run_log_path",
    "run_logging",
]
