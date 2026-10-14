"""``fm run``: ticks and phone approvals in one long-running foreground process (DESIGN section 13.1).

It is what ``fm autostart install`` starts at logon, and what replaces the scheduled ``fm tick`` plus a separately
started ``fm bot``: :func:`fm.jobs.runner.run_forever` ticks every ``--every`` minutes by wall clock (so a laptop
that slept ticks as soon as it wakes), reloads ``config.toml`` for every tick, polls the phone channel in between
and ticks at once after a near-deadline approval, so it executes right away. Each tick runs as a child process that
is stopped, browser and all, once it has run ``--tick-timeout`` minutes, so a tick that hangs can never stall the
loop; nothing executes in the loop's own process. It holds the run lock (:mod:`fm.jobs.runlock`) for its lifetime,
waiting for a scheduled tick that is mid-run (and stopping one stuck past the tick timeout); the tick task from ``fm
schedule install`` can stay installed as the wake timer, since a tick that finds the lock held exits at once.

Everything it does goes to ``logs/run.log`` under the config dir (rotating, 5 x 1 MB; the tick children append their
output to it too) and to the console when the process has one; under ``uvw.exe`` there is none. ``--no-execute``
runs the loop without the writes: ticks report what they would have executed and an approval is recorded but not
acted on. Ctrl+C stops it cleanly, and so does ``SIGTERM`` (what ``launchctl bootout`` sends), which is turned into
the same clean stop so a running tick child is stopped with its tree instead of being orphaned.

It also serves the dashboard (``fm ui``, :mod:`fm.web`) on ``127.0.0.1:--port`` (8711) in a daemon thread for as long
as the loop runs; ``--no-ui`` leaves it out. ``fm ui`` then finds it through ``<config dir>/dashboard.url`` and opens
it signed in, rather than serving a second one. The dashboard only records decisions and edits settings, which the next
tick picks up (the config is reloaded per tick). uvicorn installs signal handlers only in the main thread, so Ctrl+C and
``SIGTERM`` still reach the loop. A port that is taken, or a dashboard that will not start, is logged and printed and
the loop runs without it: the dashboard is a convenience, the ticks are the job.
"""

from __future__ import annotations

import logging
import signal
import sqlite3
import sys
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timedelta
from types import FrameType
from typing import Annotated, NoReturn

import typer

from fm.jobs.runner import DEFAULT_TICK_TIMEOUT, RunOptions, run_forever, run_log_path, run_logging
from fm.jobs.scheduler_base import DEFAULT_INTERVAL_MINUTES, MAX_INTERVAL_MINUTES
from fm.web import DEFAULT_PORT, LOOPBACK

logger = logging.getLogger(__name__)

EveryOption = Annotated[
    int,
    typer.Option(
        "--every",
        min=1,
        max=MAX_INTERVAL_MINUTES,
        help="Minutes between ticks (DESIGN section 13: one cheap tick every 10 minutes).",
    ),
]
NoExecuteOption = Annotated[
    bool,
    typer.Option("--no-execute", help="Never write to ESPN: ticks list what they would have executed."),
]
DEFAULT_TICK_TIMEOUT_MINUTES = int(DEFAULT_TICK_TIMEOUT.total_seconds() // 60)
TickTimeoutOption = Annotated[
    int,
    typer.Option(
        "--tick-timeout",
        min=1,
        max=MAX_INTERVAL_MINUTES,
        help="Minutes a tick may run before it is stopped with every process it started (the browser included).",
    ),
]
UiOption = Annotated[
    bool,
    typer.Option("--ui/--no-ui", help="Serve the dashboard (fm ui) on 127.0.0.1 while the loop runs."),
]
PortOption = Annotated[
    int,
    typer.Option("--port", min=1, max=65535, help="Port on 127.0.0.1 for the dashboard."),
]


def run_(
    every: EveryOption = DEFAULT_INTERVAL_MINUTES,
    no_execute: NoExecuteOption = False,
    tick_timeout: TickTimeoutOption = DEFAULT_TICK_TIMEOUT_MINUTES,
    ui: UiOption = True,
    port: PortOption = DEFAULT_PORT,
) -> None:
    """Tick every N minutes and listen for phone approvals in between, until Ctrl+C; serves the dashboard beside the
    loop unless --no-ui. Started at logon by `fm autostart install`; the `fm schedule` tick task stays as the wake
    timer."""
    log_path = run_log_path()
    typer.echo(
        f"fm run: tick every {every} minute{'' if every == 1 else 's'}{' (no execution)' if no_execute else ''}, "
        f"each stopped after {tick_timeout} min; log at {log_path}; Ctrl+C stops"
    )
    options = RunOptions(
        interval=timedelta(minutes=every), execute=not no_execute, tick_timeout=timedelta(minutes=tick_timeout)
    )
    with run_logging(log_path, console=sys.stderr), _sigterm_stops(), _dashboard(ui, port):
        try:
            stats = run_forever(options=options)
        except sqlite3.Error as exc:
            _fail(f"state database: {exc}")
        except KeyboardInterrupt:  # while waiting for the run lock, before the loop took over Ctrl+C
            typer.echo("stopped while waiting for the run lock")
            return
    typer.echo(
        f"stopped ({stats.stopped_by}) after {stats.ticks} tick{'' if stats.ticks == 1 else 's'} and "
        f"{stats.decisions} decision{'' if stats.decisions == 1 else 's'}"
    )


@contextmanager
def _dashboard(enabled: bool, port: int) -> Iterator[str | None]:
    """Serve the dashboard in a daemon thread for the duration and yield its URL; ``None`` when ``enabled`` is off or
    it could not start (a taken port, anything else), which is logged (``run.log``, and the console when there is one)
    but never stops the loop."""
    if not enabled:
        yield None
        return
    try:
        from fm.web.server import bind_dashboard  # Starlette, uvicorn and the views load only when serving

        dashboard = bind_dashboard(port)
        dashboard.start()
    except OSError as exc:
        logger.warning(
            "fm run: dashboard not started: %s:%d is unavailable (%s); fm run --port N picks another",
            LOOPBACK,
            port,
            exc.strerror or exc,
        )
        yield None
        return
    except Exception:
        logger.exception("fm run: dashboard not started; the loop runs without it")
        yield None
        return
    logger.info("fm run: dashboard at %s (fm ui opens it signed in)", dashboard.url)
    try:
        yield dashboard.url
    finally:
        dashboard.stop()


@contextmanager
def _sigterm_stops() -> Iterator[None]:
    """Turn ``SIGTERM`` into the ``KeyboardInterrupt`` Ctrl+C raises, for the duration. Python's default ends the
    process on the spot, which would orphan a running tick child; signal handlers can only be set from the main
    thread, so elsewhere this does nothing."""
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.signal(signal.SIGTERM, _interrupt)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, previous)


def _interrupt(signum: int, frame: FrameType | None) -> None:
    raise KeyboardInterrupt(f"signal {signum}")


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)


def register(root: typer.Typer) -> None:
    root.command("run")(run_)
