"""``fm autostart install|uninstall|show``: start ``fm run`` at logon (PLUG_AND_PLAY section 4).

The logon job goes through the platform's scheduler backend (:func:`backend`): on Windows a Task Scheduler task with
an at-logon trigger for the current user, restart on failure and no run-time limit, whose action is ``uvw.exe`` (uv's
windowless launcher) so no console window stays open (:mod:`fm.jobs.scheduler_windows`); on macOS a ``RunAtLoad``
plus ``KeepAlive`` LaunchAgent in the user's GUI domain (:mod:`fm.jobs.scheduler_macos`). Neither needs admin.

The tick job from ``fm schedule install`` is left installed on purpose: on Windows it is the wake timer (launchd
never wakes a Mac). A tick that fires while ``fm run`` holds the run lock prints ``fm run is active; skipping`` and
exits 0, and the woken loop ticks on its own because its next tick is due by wall clock.

``install`` also starts ``fm run`` at once (``Start-ScheduledTask`` on Windows, where an at-logon trigger would wait
for the next logon; launchd's ``RunAtLoad`` on macOS); ``--no-start`` registers only, on Windows.
``install --dry-run`` prints the files and the commands without running anything, ``show`` reports the job and
whether ``fm run`` holds the lock right now (and never fails), ``uninstall`` removes the job. Every command runs
through :func:`fm.jobs.scheduler_base.default_runner`, which tests replace with a fake.
"""

from __future__ import annotations

import sys
from typing import Annotated, NoReturn

import typer

from fm.jobs.runlock import RunLock
from fm.jobs.scheduler_base import AutostartBackend, SchedulerError, default_runner
from fm.jobs.scheduler_macos import MACOS_AUTOSTART
from fm.jobs.scheduler_windows import WINDOWS_AUTOSTART, WINDOWS_AUTOSTART_NO_START

app = typer.Typer(
    help="Start `fm run` at logon (Windows Task Scheduler or macOS launchd), or show and remove that job.",
    no_args_is_help=True,
)

BACKENDS: dict[str, AutostartBackend] = {"win32": WINDOWS_AUTOSTART, "darwin": MACOS_AUTOSTART}
"""The logon-job backend per ``sys.platform``: each starts ``fm run`` at once as well as at every logon."""
NO_START_BACKENDS: dict[str, AutostartBackend] = {"win32": WINDOWS_AUTOSTART_NO_START}
"""``install --no-start`` per ``sys.platform``. launchd starts a loaded agent at once (``RunAtLoad``), so macOS has
no register-only variant and ``--no-start`` is a note there."""

DryRunOption = Annotated[
    bool, typer.Option("--dry-run", help="Print the files and the scheduler commands without running anything.")
]
NoStartOption = Annotated[
    bool,
    typer.Option("--no-start", help="Register only; fm run first starts at the next logon (Windows)."),
]


def backend(platform: str | None = None, *, start: bool = True) -> AutostartBackend:
    """The logon-job backend for ``platform`` (default: this machine), the register-only one when ``start`` is off
    and the platform has one. Raises :class:`SchedulerError` elsewhere."""
    key = sys.platform if platform is None else platform
    if not start and key in NO_START_BACKENDS:
        return NO_START_BACKENDS[key]
    try:
        return BACKENDS[key]
    except KeyError:
        raise SchedulerError(f"fm autostart supports Windows and macOS, not {key!r}") from None


@app.command("install")
def install_(dry_run: DryRunOption = False, no_start: NoStartOption = False) -> None:
    """Register `fm run` to start at logon (replacing the job if it exists) and start it now unless --no-start. The
    `fm schedule` tick task stays installed as the wake timer."""
    try:
        chosen = backend(start=not no_start)
        spec = chosen.build_spec()
    except SchedulerError as exc:
        _fail(str(exc))
    if no_start and chosen not in NO_START_BACKENDS.values():
        typer.echo(f"note: --no-start does not apply to {chosen.scheduler}: it starts fm run as soon as it is loaded")
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
    typer.echo(f"installed: fm run at logon ({chosen.scheduler}); log at {spec.log_path}")


@app.command("uninstall")
def uninstall_() -> None:
    """Remove the logon job (a missing one is not an error) and its files. `fm schedule` is untouched."""
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
    """Show the logon job as the scheduler reports it, what it runs, and whether `fm run` is active now."""
    try:
        chosen = backend()
        spec = chosen.build_spec(require_uv=False)
    except SchedulerError as exc:
        _fail(str(exc))
    typer.echo(chosen.show(spec, default_runner))
    typer.echo(f"Runs:                                 {' '.join(spec.argv())}")
    typer.echo(f"Log:                                  {spec.log_path}")
    typer.echo(f"Active now:                           {_active()}")


def _active() -> str:
    lock = RunLock()
    if not lock.is_locked():
        return "no (the run lock is free)"
    holder = lock.holder()
    return f"yes, {holder.describe()}" if holder is not None else "yes (holder unknown)"


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)


def register(root: typer.Typer) -> None:
    root.add_typer(app, name="autostart")
