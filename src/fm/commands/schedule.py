"""``fm tick`` and ``fm schedule install|uninstall|show`` (DESIGN sections 12 and 13).

``fm tick`` runs :func:`fm.jobs.tick.tick` once with the live collaborators (the browser profile's session, ESPN,
the sources, the phone channel from ``config.toml`` when its secrets are set, the executor's live opener) and prints
the report. ``--no-execute`` runs everything but the writes and lists what would have executed; ``--no-sync`` decides
on the stored state. The exit code is 1 when a decision failed or an execution did not verify, so the scheduler's log
shows it.

``fm schedule`` registers that tick with the platform's scheduler (:func:`backend`): on Windows the Task Scheduler
through :mod:`fm.jobs.scheduler_windows` (a ``.cmd`` wrapper, ``schtasks /Create`` every ``--every`` minutes, and the
power settings that let the task wake the PC and run on battery); on macOS launchd through
:mod:`fm.jobs.scheduler_macos` (a ``.sh`` wrapper and a per-user LaunchAgent). ``install --dry-run`` prints the
wrapper and the commands without running anything, ``show`` queries the job (and never fails when it is not
installed), ``uninstall`` removes it.
"""

from __future__ import annotations

import sys
from typing import Annotated, NoReturn

import typer

from fm.config import ConfigError, load_config
from fm.jobs.scheduler_base import (
    DEFAULT_INTERVAL_MINUTES,
    MAX_INTERVAL_MINUTES,
    Backend,
    SchedulerError,
    default_runner,
)
from fm.jobs.scheduler_macos import MACOS
from fm.jobs.scheduler_windows import WINDOWS
from fm.jobs.tick import TickOptions, tick
from fm.notify import NotifyError, open_channel
from fm.store import Store

app = typer.Typer(
    help="Install the tick as a scheduled job (Windows Task Scheduler or macOS launchd), or show and remove it.",
    no_args_is_help=True,
)

BACKENDS: dict[str, Backend] = {"win32": WINDOWS, "darwin": MACOS}
"""The scheduler backend per ``sys.platform``."""


def backend(platform: str | None = None) -> Backend:
    """The scheduler backend for ``platform`` (default: this machine). Raises :class:`SchedulerError` elsewhere."""
    key = sys.platform if platform is None else platform
    try:
        return BACKENDS[key]
    except KeyError:
        raise SchedulerError(f"fm schedule supports Windows and macOS, not {key!r}") from None


EveryOption = Annotated[
    int,
    typer.Option(
        "--every",
        min=1,
        max=MAX_INTERVAL_MINUTES,
        help="Minutes between ticks (DESIGN section 13: one cheap tick every 10 minutes).",
    ),
]
DryRunOption = Annotated[
    bool, typer.Option("--dry-run", help="Print the wrapper and the scheduler commands without running anything.")
]
LeagueOption = Annotated[
    list[str] | None,
    typer.Option("--league", "-l", help="Only these league keys from config.toml (repeatable). Default: every league."),
]
NoExecuteOption = Annotated[
    bool, typer.Option("--no-execute", help="Do everything but the writes; list what would have executed.")
]
NoSyncOption = Annotated[bool, typer.Option("--no-sync", help="Decide on the stored state without syncing ESPN.")]


# --- fm tick ----------------------------------------------------------------------------------------------------------


def tick_(league: LeagueOption = None, no_execute: NoExecuteOption = False, no_sync: NoSyncOption = False) -> None:
    """Run one tick: reconcile, expire, check the session, sync what is stale, run the decisions that are due,
    push new proposals, auto-approve at T-15 and execute what is approved."""
    try:
        config = load_config()
    except ConfigError as exc:
        _fail(str(exc))
    for key in league or ():
        try:
            config.league(key)
        except KeyError as exc:
            _fail(exc.args[0])
    try:
        channel = open_channel(config)
    except NotifyError as exc:
        channel = None
        typer.echo(f"note: no phone channel ({exc}); proposals and alerts stay in the store and this log")
    options = TickOptions(execute=not no_execute, sync=not no_sync, leagues=None if league is None else tuple(league))
    try:
        with Store.open() as store:
            report = tick(store, config, channel=channel, options=options)
    finally:
        if channel is not None:
            channel.close()
    for line in report.lines():
        typer.echo(line)
    if not report.ok:
        raise typer.Exit(1)


# --- fm schedule ------------------------------------------------------------------------------------------------------


@app.command("install")
def install_(every: EveryOption = DEFAULT_INTERVAL_MINUTES, dry_run: DryRunOption = False) -> None:
    """Write the wrapper and register the scheduled job (replacing one that exists)."""
    try:
        chosen = backend()
        spec = chosen.build_spec(interval_minutes=every)
    except SchedulerError as exc:
        _fail(str(exc))
    if dry_run:
        typer.echo("dry run: nothing is written or installed")
        for line in chosen.plan(spec):
            typer.echo(line)
        return
    try:
        report = chosen.install(spec, default_runner)
    except SchedulerError as exc:
        _fail(str(exc))
    for line in report.lines():
        typer.echo(line)
    typer.echo(f"installed: fm tick every {every} minutes ({chosen.scheduler}); log at {spec.log_path}")


@app.command("uninstall")
def uninstall_() -> None:
    """Remove the scheduled job (a missing one is not an error) and its wrapper."""
    try:
        chosen = backend()
        spec = chosen.build_spec(require_uv=False)
        report = chosen.uninstall(spec, default_runner)
    except SchedulerError as exc:
        _fail(str(exc))
    for line in report.lines():
        typer.echo(line)


@app.command("show")
def show_() -> None:
    """Show the scheduled job as the platform scheduler reports it, and what the wrapper runs."""
    try:
        chosen = backend()
        spec = chosen.build_spec(require_uv=False)
    except SchedulerError as exc:
        _fail(str(exc))
    typer.echo(chosen.show(spec, default_runner))
    typer.echo(f"Runs:                                 {' '.join(spec.argv())}")
    typer.echo(f"Log:                                  {spec.log_path}")


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)


def register(root: typer.Typer) -> None:
    root.command("tick")(tick_)
    root.add_typer(app, name="schedule")
