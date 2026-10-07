"""``fm drill``: walk each UI-mode flow on ESPN's live pages up to, but never including, its final save (DESIGN 6.3).

It keeps the UI fallback known-good for the day API mode breaks. For each league it reads the current views, picks a
target for the lineup, add/drop and waiver-claim flows from them, runs each flow's click-through on a live page and
stops where the flow would save: the control must show, and is never clicked. Then it navigates away and closes the
page. The browser context is a dry run (every non-GET to ESPN is aborted, as in ``fm execute --dry-run`` and ``fm
canary``), the driver and the page refuse the save too, and nothing is proposed or stored (see
:mod:`fm.browser.drills` for the four layers).

It prints one line per flow and one per finding, and exits 1 when anything is wrong. A failing run, or one that could
not start (the session expired, the browser would not open), also pushes one alert to the phone channel
(``[notify].channel``); ``--no-alert`` keeps it quiet. There is no ``--fixtures`` mode: a drill is only worth running
against the live site.

Run it weekly, on a day with no games (a player whose game has started is locked, so the drill has nothing to move
and skips that flow). ``fm schedule`` installs the tick only, so a weekly run is a second scheduled task that runs ``fm
drill`` (the wrapper :mod:`fm.jobs.scheduler_windows` or :mod:`fm.jobs.scheduler_macos` writes for the tick is the
model), or a call to :func:`run_live`
and :func:`send_alert_for` from the tick's health checks once the last drill is a week old.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated, NoReturn

import typer

from fm.browser.drills import (
    DrillError,
    DrillReport,
    DrillResult,
    DrillStatus,
    could_not_run_alert,
    drill_alert,
    drillable_flows,
    run_drills,
    unsaved_league_row,
)
from fm.browser.flows import discover, flow_registry
from fm.browser.session import BrowserError, Channel, LaunchOptions
from fm.config import Config, ConfigError, League, load_config
from fm.espn.auth import AuthError
from fm.executor import live_opener
from fm.jobs.sync import SyncError, select_leagues
from fm.notify import Message, NotifyError, open_channel, send_alert

LeagueOption = Annotated[
    list[str] | None,
    typer.Option("--league", "-l", help="Only these league keys from config.toml (repeatable). Default: every league."),
]
FlowOption = Annotated[
    list[str] | None,
    typer.Option(
        "--flow", "-f", help="Only these flows (set_lineup, add_drop, claim_waiver; repeatable). Default: all."
    ),
]
AlertOption = Annotated[
    bool, typer.Option("--alert/--no-alert", help="Push an alert to the phone when something is wrong.")
]
HeadedOption = Annotated[bool, typer.Option("--headed", help="Show the browser window.")]
ChannelOption = Annotated[
    Channel | None, typer.Option(help="Browser to drive. Default: the first one installed, Edge before Chrome.")
]


def drill(
    league: LeagueOption = None,
    flow: FlowOption = None,
    alert: AlertOption = True,
    headed: HeadedOption = False,
    channel: ChannelOption = None,
) -> None:
    """Dry-run each UI-mode flow on the live site, stopping before the final save; alert on failure."""
    try:
        config = load_config()
        leagues = select_leagues(config, league)
    except (ConfigError, SyncError) as exc:
        _fail(str(exc))
    if not leagues:
        _fail("no leagues in config.toml")
    _check_flow_names(leagues, flow)
    options = LaunchOptions(headless=not headed, channel=None if channel is None else channel.value)
    try:
        reports = [run_live(item, options, flow) for item in leagues]
    except (AuthError, BrowserError) as exc:
        typer.echo(f"error: {exc}", err=True)
        if alert:
            send_message(config, could_not_run_alert(str(exc)))
        raise typer.Exit(1) from None
    except DrillError as exc:
        _fail(str(exc))
    for report in reports:
        for line in report_lines(report):
            typer.echo(line)
    problem = any(not report.ok for report in reports)
    if problem and alert:
        send_alert_for(config, reports)
    if problem:
        raise typer.Exit(1)


def report_lines(report: DrillReport) -> list[str]:
    """What ``fm drill`` prints for one league: one line per flow, then one per finding."""
    lines = [f"{report.league}: {_result_line(result)}" for result in report.results]
    lines.extend(f"{report.league}: FAILED {finding.line()}" for finding in report.findings)
    if report.ok:
        lines.append(f"{report.league}: ok")
    return lines


def _result_line(result: DrillResult) -> str:
    if result.status is DrillStatus.SKIPPED:
        return f"{result.flow}: skipped ({result.skipped})"
    steps = f"{len(result.pages)} pages, {len(result.clicks)} clicks"
    if result.status is DrillStatus.PASSED:
        return f"{result.flow}: ok, {result.target}: {steps}, stopped before {result.reached!r}"
    return f"{result.flow}: failed, {result.target or 'no target'}: {steps}"


# --- running ----------------------------------------------------------------------------------------------------------


def run_live(league: League, options: LaunchOptions, flows: list[str] | None = None) -> DrillReport:
    """One league on the real browser profile. ``dry_run=True`` makes the context abort every request that is not a GET
    to ESPN and gives the runtime a transport that refuses every send; the drill refuses any runtime that lacks it."""
    with live_opener(options)(unsaved_league_row(league, datetime.now(UTC)), dry_run=True) as runtime:
        return run_drills(league, runtime=runtime, flows=flows)


def send_alert_for(config: Config, reports: list[DrillReport]) -> None:
    """Push the one alert for a run's failing reports, if any."""
    message = drill_alert(reports)
    if message is not None:
        send_message(config, message)


def send_message(config: Config, message: Message) -> None:
    try:
        channel = open_channel(config)
        try:
            send_alert(channel, message.title, message.body, link=message.link)
        finally:
            channel.close()
    except NotifyError as exc:
        typer.echo(f"error: could not send the alert: {exc}", err=True)
        return
    typer.echo(f"alert sent: {message.title}")


def _check_flow_names(leagues: tuple[League, ...], names: list[str] | None) -> None:
    """Refuse an unknown ``--flow`` before a browser opens: each name must be a flow with a UI mode in every league."""
    if not names:
        return
    discover()
    for item in leagues:
        known = [found.name for found in drillable_flows(flow_registry, item.sport)]
        unknown = sorted(set(names) - set(known))
        if unknown:
            _fail(f"no UI flow {', '.join(unknown)} for {item.key} ({item.sport}); with a UI mode: {', '.join(known)}")


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)


def register(root: typer.Typer) -> None:
    root.command("drill")(drill)
