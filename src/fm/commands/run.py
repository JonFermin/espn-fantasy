"""``fm run``: ticks and phone approvals in one long-running foreground process (PLUG_AND_PLAY section 4).

It is what ``fm autostart install`` starts at logon, and what replaces the scheduled ``fm tick`` plus a separately
started ``fm bot``: :func:`fm.jobs.runner.run_forever` ticks every ``--every`` minutes by wall clock (so a laptop
that slept ticks as soon as it wakes), reloads ``config.toml`` for every tick, polls the phone channel in between
and executes a near-deadline approval at once. It holds the run lock (:mod:`fm.jobs.runlock`) for its lifetime,
waiting for a scheduled tick that is mid-run; the tick task from ``fm schedule install`` can stay installed as the
wake timer, since a tick that finds the lock held exits at once.

Everything it does goes to ``logs/run.log`` under the config dir (rotating, 5 x 1 MB) and to the console when the
process has one; under ``uvw.exe`` there is none. ``--no-execute`` runs the loop without the writes: ticks report
what they would have executed and an approval is recorded but not acted on. Ctrl+C stops it cleanly.
"""

from __future__ import annotations

import sqlite3
import sys
from datetime import timedelta
from typing import Annotated, NoReturn

import typer

from fm.jobs.runner import RunOptions, run_forever, run_log_path, run_logging
from fm.jobs.scheduler_base import DEFAULT_INTERVAL_MINUTES, MAX_INTERVAL_MINUTES

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


def run_(every: EveryOption = DEFAULT_INTERVAL_MINUTES, no_execute: NoExecuteOption = False) -> None:
    """Tick every N minutes and listen for phone approvals in between, until Ctrl+C. Started at logon by
    `fm autostart install`; the `fm schedule` tick task stays as the wake timer."""
    log_path = run_log_path()
    typer.echo(
        f"fm run: tick every {every} minute{'' if every == 1 else 's'}{' (no execution)' if no_execute else ''}; "
        f"log at {log_path}; Ctrl+C stops"
    )
    options = RunOptions(interval=timedelta(minutes=every), execute=not no_execute)
    with run_logging(log_path, console=sys.stderr):
        try:
            stats = run_forever(options=options)
        except sqlite3.Error as exc:
            _fail(f"state database: {exc}")
    typer.echo(
        f"stopped ({stats.stopped_by}) after {stats.ticks} tick{'' if stats.ticks == 1 else 's'} and "
        f"{stats.decisions} decision{'' if stats.decisions == 1 else 's'}"
    )


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)


def register(root: typer.Typer) -> None:
    root.command("run")(run_)
