"""``fm setup``: sign in, find your leagues, pick presets, add keys, pair the phone, write the config, sync, and start
``fm run`` at logon, in one command (docs/DESIGN.md section 12.2).

The steps live in :mod:`fm.setup.wizard`; this module holds the terminal prompts (:class:`TerminalPrompts`) and
:func:`live_deps`, the real collaborators: the browser profile's session and the ``fm login`` flow, ESPN discovery
and league confirmation, :mod:`fm.notify.setup`, :func:`fm.jobs.sync.sync`, the ``fm autostart`` and ``fm schedule``
scheduler backends, a process listing for a running ``fm bot`` (reported, never stopped; while it runs the logon job
is registered without starting ``fm run``) and ``fm doctor``'s checks. ``--yes`` answers every question with its
default and skips every secret, which never adds a newly discovered league to an existing ``config.toml`` or changes a
league's team; ``--league-url`` names a league to add, and is the fallback when discovery fails; ``--preset`` and
``--channel`` answer those questions for every league. Re-running changes nothing when nothing is new.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import click
import typer

from fm.commands import autostart as autostart_cmd
from fm.commands import login as login_cmd
from fm.commands import schedule as schedule_cmd
from fm.commands.doctor import check_lines
from fm.commands.sync import render as render_sync
from fm.config import Config
from fm.espn.auth import AuthError, EspnSession, load_session
from fm.espn.client import EspnClientError
from fm.espn.discover import confirm_league, discover_leagues
from fm.health import job_installed, run_checks
from fm.jobs.proctree import quiet_runner
from fm.jobs.scheduler_base import DEFAULT_INTERVAL_MINUTES, SchedulerError, default_runner
from fm.jobs.scheduler_windows import POWERSHELL
from fm.jobs.sync import SyncError, sync
from fm.model.ids import CrosswalkError
from fm.setup import (
    AcceptDefaults,
    Job,
    Prompts,
    SetupDeps,
    SetupError,
    SetupOptions,
    StepError,
    notify_setup,
    parse_bot_pids,
    run_setup,
)
from fm.sources.base import SourceError
from fm.store import Store

PROCESS_LISTING: dict[str, list[str]] = {
    "win32": [
        POWERSHELL,
        "-NoProfile",
        "-NonInteractive",
        "-Command",
        "Get-CimInstance Win32_Process -Filter \"CommandLine like '%bot%'\" | "
        'ForEach-Object { "$($_.ProcessId) $($_.CommandLine)" }',
    ],
    "darwin": ["ps", "-A", "-o", "pid=", "-o", "command="],
    "linux": ["ps", "-A", "-o", "pid=", "-o", "command="],
}
"""Per ``sys.platform``, the command that lists processes as ``<pid> <command line>`` lines for the ``fm bot`` probe
(the Windows one filters to command lines mentioning ``bot`` first: listing every process through CIM is slow)."""
PLACEHOLDER_UV = Path("uv")
"""A scheduler ``show`` needs only the job's name and files; a placeholder uv skips the PATH search for it."""


class PresetChoice(StrEnum):
    hands_off = "hands_off"
    balanced = "balanced"
    ask_first = "ask_first"


class PhoneChoice(StrEnum):
    ntfy = "ntfy"
    telegram = "telegram"
    none = "none"


def setup(
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Accept every default without asking; secrets you have not set are skipped. An existing config.toml "
            "gets no newly discovered league (pass --league-url) and keeps its teams.",
        ),
    ] = False,
    league_url: Annotated[
        list[str] | None,
        typer.Option(
            "--league-url",
            help="A league to set up when discovery fails: your team page URL, or any league page with leagueId= "
            "(repeatable).",
        ),
    ] = None,
    preset: Annotated[
        PresetChoice | None,
        typer.Option("--preset", help="Preset for every new league; default: ask (balanced with --yes)."),
    ] = None,
    channel: Annotated[
        PhoneChoice | None,
        typer.Option("--channel", help="Phone channel; default: ask (the configured one, else ntfy, with --yes)."),
    ] = None,
    no_sync: Annotated[bool, typer.Option("--no-sync", help="Skip the first sync of new leagues.")] = False,
    no_autostart: Annotated[
        bool, typer.Option("--no-autostart", help="Do not install fm run at logon or the wake timer.")
    ] = False,
) -> None:
    """Set everything up: ESPN sign-in, leagues, presets, API keys, phone, config.toml, first sync, autostart.

    Safe to run again: leagues already in config.toml are never rewritten, and nothing changes when nothing is new.
    """
    options = SetupOptions(
        league_urls=tuple(league_url or ()),
        preset=None if preset is None else preset.value,
        channel=None if channel is None else channel.value,
        sync=not no_sync,
        autostart=not no_autostart,
    )
    prompts: Prompts = AcceptDefaults() if yes else TerminalPrompts()
    try:
        run_setup(options, live_deps(), prompts=prompts, echo=typer.echo)
    except SetupError as exc:
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(1) from None


class TerminalPrompts:
    """The wizard's questions on the terminal; secrets are typed hidden and an empty answer skips."""

    def confirm(self, question: str, *, default: bool) -> bool:
        return typer.confirm(question, default=default)

    def choose(self, question: str, choices: Sequence[str], *, default: str) -> str:
        return str(typer.prompt(question, default=default, type=click.Choice(list(choices)), show_choices=True))

    def text(self, question: str, *, default: str = "") -> str:
        return str(typer.prompt(question, default=default, show_default=bool(default)))

    def secret(self, question: str) -> str:
        return str(typer.prompt(question, default="", hide_input=True, show_default=False))


def live_deps() -> SetupDeps:
    """The real collaborators. Tests replace this function with one returning fakes."""
    return SetupDeps(
        load_session=load_session,
        login=_login,
        discover=discover_leagues,
        confirm_league=confirm_league,
        setup_channel=notify_setup,
        sync=_sync,
        autostart=Job(
            name="fm run at logon",
            command="fm autostart install",
            installed=_autostart_installed,
            install=_install_autostart,
            install_no_start=_register_autostart if autostart_cmd.NO_START_BACKENDS.get(sys.platform) else None,
        ),
        wake_timer=Job(
            name="the wake timer (fm tick)",
            command="fm schedule install",
            installed=_wake_timer_installed,
            install=_install_wake_timer,
        ),
        bot_pids=_bot_pids,
        doctor=_doctor,
    )


def _login() -> EspnSession:
    """``fm login``'s own flow (a headed window for the manual sign-in), then the session it saved."""
    try:
        login_cmd.login(check=False, channel=None, timeout=login_cmd.DEFAULT_LOGIN_TIMEOUT_S)
    except typer.Exit as exc:  # fm login printed why
        raise AuthError("the sign-in window closed or timed out") from exc
    return load_session()


def _sync(config: Config, keys: Sequence[str], session: EspnSession) -> list[str]:
    try:
        with Store.open() as store:
            report = sync(store, config, session=session, leagues=keys)
    except (SyncError, EspnClientError, SourceError, CrosswalkError) as exc:
        raise StepError(str(exc)) from exc
    lines = render_sync(report)
    for failed in report.failed:
        if failed.gate is not None and failed.gate.error is not None:
            lines.append(f"error: {failed.key}: {failed.gate.error}")
    return lines


def _autostart_installed() -> bool | None:
    try:
        chosen = autostart_cmd.backend()
        spec = chosen.build_spec(require_uv=False, uv_path=PLACEHOLDER_UV)
    except SchedulerError:
        return None
    return job_installed(chosen.show(spec, default_runner))


def _install_autostart(*, start: bool = True) -> list[str]:
    try:
        chosen = autostart_cmd.backend(start=start)
        spec = chosen.build_spec()
        report = chosen.install(spec, default_runner)
    except SchedulerError as exc:
        raise StepError(str(exc)) from exc
    started = "" if start else " (registered only: fm run starts at the next logon or with fm autostart install)"
    return [*report.lines(), f"installed: fm run at logon ({chosen.scheduler}); log at {spec.log_path}{started}"]


def _register_autostart() -> list[str]:
    """``fm autostart install --no-start``: the logon job without starting ``fm run`` now, for while ``fm bot`` runs.
    Only where the scheduler has such a mode (:data:`fm.commands.autostart.NO_START_BACKENDS`)."""
    return _install_autostart(start=False)


def _wake_timer_installed() -> bool | None:
    try:
        chosen = schedule_cmd.backend()
        spec = chosen.build_spec(require_uv=False, uv_path=PLACEHOLDER_UV)
    except SchedulerError:
        return None
    return job_installed(chosen.show(spec, default_runner))


def _install_wake_timer() -> list[str]:
    try:
        chosen = schedule_cmd.backend()
        spec = chosen.build_spec(interval_minutes=DEFAULT_INTERVAL_MINUTES)
        report = chosen.install(spec, default_runner)
    except SchedulerError as exc:
        raise StepError(str(exc)) from exc
    every = f"every {DEFAULT_INTERVAL_MINUTES} minutes"
    return [*report.lines(), f"installed: fm tick {every} ({chosen.scheduler}); log at {spec.log_path}"]


def _bot_pids() -> list[int]:
    """Running ``fm bot`` processes, from a process listing; none when the listing fails (the probe is advisory)."""
    argv = PROCESS_LISTING.get(sys.platform)
    if argv is None:
        return []
    result = quiet_runner(argv)
    return parse_bot_pids(result.stdout, exclude=(os.getpid(),)) if result.ok else []


def _doctor() -> list[str]:
    return check_lines(run_checks(launch_browser=True))


def register(root: typer.Typer) -> None:
    root.command("setup")(setup)
