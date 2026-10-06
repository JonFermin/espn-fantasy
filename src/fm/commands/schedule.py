"""``fm tick`` and ``fm schedule install|uninstall|show`` (DESIGN sections 12 and 13).

``fm tick`` runs :func:`fm.jobs.tick.tick` once with the live collaborators (the browser profile's session, ESPN,
the sources, the phone channel from ``config.toml`` when its secrets are set, the executor's live opener) and prints
the report. ``--no-execute`` runs everything but the writes and lists what would have executed; ``--no-sync`` decides
on the stored state. The exit code is 1 when a decision failed or an execution did not verify, so the scheduler's log
shows it.

``fm schedule`` registers that tick with the Windows Task Scheduler through :mod:`fm.jobs.scheduler_windows`: a
``.cmd`` wrapper under the config dir, ``schtasks /Create`` every ``--every`` minutes, and the power settings that let
the task wake the PC and run on battery. ``install --dry-run`` prints the wrapper and the commands without running
anything, ``show`` queries the task (and never fails when it is not installed), ``uninstall`` deletes it.
"""

from __future__ import annotations

from typing import Annotated, NoReturn

import typer

from fm.config import ConfigError, load_config
from fm.jobs import scheduler_windows as scheduler
from fm.jobs.tick import TickOptions, tick
from fm.notify import NotifyError, open_channel
from fm.store import Store

app = typer.Typer(help="Install the tick as a Windows scheduled task, or show and remove it.", no_args_is_help=True)

EveryOption = Annotated[
    int,
    typer.Option(
        "--every",
        min=1,
        max=scheduler.MAX_INTERVAL_MINUTES,
        help="Minutes between ticks (DESIGN section 13: one cheap tick every 10 minutes).",
    ),
]
DryRunOption = Annotated[
    bool, typer.Option("--dry-run", help="Print the wrapper and the schtasks commands without running anything.")
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
    options = TickOptions(
        execute=not no_execute, sync=not no_sync, leagues=None if league is None else tuple(league)
    )
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
def install_(every: EveryOption = scheduler.DEFAULT_INTERVAL_MINUTES, dry_run: DryRunOption = False) -> None:
    """Write the .cmd wrapper and create the scheduled task (replacing one that exists), then allow wake and battery."""
    try:
        spec = scheduler.ScheduleSpec.build(interval_minutes=every)
    except scheduler.SchedulerError as exc:
        _fail(str(exc))
    if dry_run:
        typer.echo("dry run: nothing is written or installed")
        for line in scheduler.render_plan(spec):
            typer.echo(line)
        return
    try:
        report = scheduler.install(spec, scheduler.default_runner)
    except scheduler.SchedulerError as exc:
        _fail(str(exc))
    for line in report.lines():
        typer.echo(line)
    typer.echo(f"installed: fm tick every {every} minutes; log at {spec.log_path}")


@app.command("uninstall")
def uninstall_() -> None:
    """Delete the scheduled task (a missing one is not an error) and remove the wrapper."""
    try:
        spec = scheduler.ScheduleSpec.build(require_uv=False)
        report = scheduler.uninstall(spec, scheduler.default_runner)
    except scheduler.SchedulerError as exc:
        _fail(str(exc))
    for line in report.lines():
        typer.echo(line)


@app.command("show")
def show_() -> None:
    """Show the scheduled task as Task Scheduler reports it, and what the wrapper runs."""
    try:
        spec = scheduler.ScheduleSpec.build(require_uv=False)
    except scheduler.SchedulerError as exc:
        _fail(str(exc))
    typer.echo(scheduler.show(spec, scheduler.default_runner))
    typer.echo(f"Runs:                                 {' '.join(spec.argv())}")
    typer.echo(f"Log:                                  {spec.log_path}")


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)


def register(root: typer.Typer) -> None:
    root.command("tick")(tick_)
    root.add_typer(app, name="schedule")
