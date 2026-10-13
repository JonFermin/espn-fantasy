"""``fm tick`` and ``fm schedule install|uninstall|show`` (DESIGN sections 12 and 13).

``fm tick`` runs :func:`fm.jobs.tick.tick` once with the live collaborators (the browser profile's session, ESPN,
the sources, the phone channel from ``config.toml`` when its secrets are set, the executor's live opener) and prints
the report. ``--no-execute`` runs everything but the writes and lists what would have executed; ``--no-sync`` decides
on the stored state. The exit code is 1 when a decision failed or an execution did not verify, so the scheduler's log
shows it. The tick holds the run lock (:mod:`fm.jobs.runlock`) while it works; when ``fm run`` holds it, the tick
prints ``fm run is active (pid N); skipping`` and exits 0, which is what lets the scheduled task stay installed as
the Windows wake timer beside ``fm autostart``.

The work itself runs in a child process, exactly as ``fm run`` runs its ticks (:func:`fm.jobs.runner.run_tick_child`:
``python -m fm.jobs.runner tick``), stopped with its whole process tree after ``--timeout`` minutes: a scheduled tick
once hung for five days in a headless browser, and the Task Scheduler's own limit kills only the wrapper on top. The
child's output goes to this process's (the scheduler wrapper's log); a timeout is reported, alerted on the phone at
most once an hour (:data:`TIMEOUT_ALERT_EVERY`, remembered in ``tick-state.json``) and exits 1. The child's
environment names this process's pid (:data:`fm.jobs.runlock.HELD_BY_ENV`), and when that matches the lock record it
runs under the parent's lock instead of skipping (:func:`fm.jobs.runlock.held_by_parent`), the same rule as for the
tick children of ``fm run``. ``--in-process`` runs the tick in this process with no timeout, for debugging.

``fm schedule`` registers that tick with the platform's scheduler (:func:`backend`): on Windows the Task Scheduler
through :mod:`fm.jobs.scheduler_windows` (a ``.cmd`` wrapper, ``schtasks /Create`` every ``--every`` minutes, and the
power settings that let the task wake the PC and run on battery); on macOS launchd through
:mod:`fm.jobs.scheduler_macos` (a ``.sh`` wrapper and a per-user LaunchAgent). ``install --dry-run`` prints the
wrapper and the commands without running anything, ``show`` queries the job (and never fails when it is not
installed), ``uninstall`` removes it.
"""

from __future__ import annotations

import sys
from datetime import timedelta
from typing import Annotated, NoReturn

import typer

from fm.config import ConfigError, load_config
from fm.jobs.runlock import RunLock, RunLockHeld, held_by_parent
from fm.jobs.runner import DEFAULT_TICK_TIMEOUT, TickResult, run_tick_child
from fm.jobs.scheduler_base import (
    DEFAULT_INTERVAL_MINUTES,
    MAX_INTERVAL_MINUTES,
    Backend,
    SchedulerError,
    default_runner,
)
from fm.jobs.scheduler_macos import MACOS
from fm.jobs.scheduler_windows import WINDOWS
from fm.jobs.tick import TickOptions, TickState, state_path, tick
from fm.notify import NotifyError, open_channel, send_alert
from fm.store import Store, utc_now

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
DEFAULT_TICK_TIMEOUT_MINUTES = int(DEFAULT_TICK_TIMEOUT.total_seconds() // 60)
TimeoutOption = Annotated[
    int,
    typer.Option(
        "--timeout",
        min=1,
        max=MAX_INTERVAL_MINUTES,
        help="Minutes the tick may run before it is stopped with every process it started (the browser included).",
    ),
]
InProcessOption = Annotated[
    bool,
    typer.Option("--in-process", help="Debugging: run the tick in this process, with no timeout and no child."),
]
TIMEOUT_ALERT_KEY = "fm-tick:timeout"
"""Where ``tick-state.json`` remembers the last timeout alert of a scheduled tick."""
TIMEOUT_ALERT_EVERY = timedelta(hours=1)
"""A scheduled tick that keeps timing out is alerted at most this often, as ``fm run`` alerts its own timeouts."""


# --- fm tick ----------------------------------------------------------------------------------------------------------


def tick_(
    league: LeagueOption = None,
    no_execute: NoExecuteOption = False,
    no_sync: NoSyncOption = False,
    timeout: TimeoutOption = DEFAULT_TICK_TIMEOUT_MINUTES,
    in_process: InProcessOption = False,
) -> None:
    """Run one tick: reconcile, expire, check the session, sync what is stale, run the decisions that are due,
    push new proposals, auto-approve at T-15 and execute what is approved. Skips (exit 0) while `fm run` is active.
    The tick runs in a child process that is stopped, browser and all, after --timeout minutes."""
    lock = RunLock(command="fm tick")
    try:
        lock.acquire()
    except RunLockHeld as exc:
        holder = exc.holder
        if held_by_parent(holder):  # the tick child of fm run or of a scheduled fm tick, which holds the lock for it
            _tick_once(league, no_execute, no_sync)
            return
        who = holder.command if holder is not None and holder.command else "fm run"
        pid = str(holder.pid) if holder is not None else "unknown"
        typer.echo(f"{who} is active (pid {pid}); skipping")
        return
    try:
        if in_process:
            _tick_once(league, no_execute, no_sync)
        else:
            _tick_in_child(league, no_execute, no_sync, timedelta(minutes=timeout))
    finally:
        lock.release()


def _tick_in_child(league: list[str] | None, no_execute: bool, no_sync: bool, timeout: timedelta) -> None:
    """The tick as a child process under this process's lock (:func:`fm.jobs.runner.run_tick_child`). The child loads
    the config and checks ``--league`` itself and prints its own errors; this process only reports how it ended."""
    options = TickOptions(execute=not no_execute, sync=not no_sync, leagues=None if league is None else tuple(league))
    result: TickResult = run_tick_child(options, timeout=timeout)
    for note in result.notes:
        typer.echo(f"note: {note}", err=True)
    if result.timed_out:
        minutes = int(timeout.total_seconds() // 60)
        what = f"tick timed out after {minutes} min and was stopped (pid {result.pid} and every process under it)"
        typer.echo(f"error: {what}", err=True)
        _alert_timeout(what)
        raise typer.Exit(1)
    if not result.ok:
        raise typer.Exit(1)  # the child printed what failed (a decision, an execution, the config)


def _alert_timeout(what: str) -> None:
    """Push the timeout to the phone at most once per :data:`TIMEOUT_ALERT_EVERY`. Recorded only after the push went
    out; no config, no channel or a failed push leaves the line above in the scheduler's log as the only report."""
    now = utc_now()
    path = state_path()
    state = TickState.load(path)
    last = state.alerts.get(TIMEOUT_ALERT_KEY)
    if last is not None and now - last < TIMEOUT_ALERT_EVERY:
        return
    try:
        channel = open_channel(load_config())
    except (ConfigError, NotifyError) as exc:
        typer.echo(f"note: timeout not pushed to the phone ({exc})", err=True)
        return
    try:
        send_alert(
            channel,
            "fm tick: tick timed out",
            f"The scheduled {what}. The next scheduled tick runs as usual; see the tick log (fm schedule show).",
        )
    except NotifyError as exc:
        typer.echo(f"note: timeout not pushed to the phone ({exc})", err=True)
        return
    finally:
        channel.close()
    state.alerts[TIMEOUT_ALERT_KEY] = now  # the child is gone and this process holds the lock: no other writer
    state.save(path)


def _tick_once(league: list[str] | None, no_execute: bool, no_sync: bool) -> None:
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
