"""``fm canary``: check that ESPN's pages and read views still look the way the executor expects (DESIGN 6.3).

Read-only. For each league it opens the pages the selector registry covers and counts every selector on them (no
click, nothing is typed, and the browser context aborts anything but a GET to ESPN, as in a dry run), then reads every
view through the API and parses it. It prints what it found, one line per finding, and exits 1 when anything is wrong.
A live run that finds drift also pushes one alert to the phone channel (``[notify].channel``); ``--no-alert`` keeps it
quiet. Run it daily (the tick's health checks do) so ESPN's drift is known before a real move depends on it.

``--fixtures DIR`` reads the views from recorded files (``tests/fixtures/espn/real/ffl``) instead of ESPN, as
``fm execute --dry-run --fixtures`` does. It opens no browser, so the selectors are not checked, and it never alerts.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, NoReturn

import httpx
import typer

from fm.browser.canary import CanaryReport, drift_alert, run_canary
from fm.browser.session import BrowserError, Channel, LaunchOptions
from fm.commands.execute import RecordedViews
from fm.config import Config, ConfigError, League, load_config
from fm.espn.auth import AuthError
from fm.espn.client import EspnClient
from fm.espn.ids import Game
from fm.executor import live_opener
from fm.jobs.sync import SyncError, select_leagues
from fm.notify import NotifyError, open_channel, send_alert
from fm.store import LeagueRow

LeagueOption = Annotated[
    list[str] | None,
    typer.Option("--league", "-l", help="Only these league keys from config.toml (repeatable). Default: every league."),
]
FixturesOption = Annotated[
    Path | None,
    typer.Option(
        "--fixtures",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="Read the views from recorded files in this folder (<view>.json, such as tests/fixtures/espn/real/ffl) "
        "instead of ESPN. Opens no browser, so selectors are not checked, and never sends an alert.",
    ),
]
AlertOption = Annotated[
    bool, typer.Option("--alert/--no-alert", help="Push an alert to the phone when something is wrong (live runs).")
]
HeadedOption = Annotated[bool, typer.Option("--headed", help="Show the browser window.")]
ChannelOption = Annotated[
    Channel | None, typer.Option(help="Browser to drive. Default: the first one installed, Edge before Chrome.")
]


def canary(
    league: LeagueOption = None,
    fixtures: FixturesOption = None,
    alert: AlertOption = True,
    headed: HeadedOption = False,
    channel: ChannelOption = None,
) -> None:
    """Assert that every registered selector resolves and every read view parses; alert on drift. Read-only."""
    try:
        config = load_config()
        leagues = select_leagues(config, league)
    except (ConfigError, SyncError) as exc:
        _fail(str(exc))
    if not leagues:
        _fail("no leagues in config.toml")
    if fixtures is not None:
        typer.echo(f"note: reading recorded views from {fixtures}; no browser, so selectors are not checked")
        reports = [_run_fixture(item, fixtures) for item in leagues]
    else:
        options = LaunchOptions(headless=not headed, channel=None if channel is None else channel.value)
        try:
            reports = [_run_live(item, options) for item in leagues]
        except (AuthError, BrowserError) as exc:
            _fail(str(exc))
    for report in reports:
        for line in report_lines(report):
            typer.echo(line)
    problem = any(not report.ok for report in reports)
    if problem and alert and fixtures is None:
        _send_alert(config, reports)
    if problem:
        raise typer.Exit(1)


def report_lines(report: CanaryReport) -> list[str]:
    """What ``fm canary`` prints for one league: what was checked, then one line per finding."""
    lines: list[str] = []
    checks = report.selector_check
    if checks is None:
        lines.append(f"{report.league}: selectors not checked (no browser)")
    else:
        lines.append(
            f"{report.league}: {len(checks.resolved)} selectors resolved, {len(checks.absent_sometimes)} "
            f"sometimes-selectors absent (fine), {len(checks.skipped)} skipped"
        )
    views = report.view_check
    if views is not None:
        lines.append(f"{report.league}: {len(views.parsed)} views parsed, {len(views.skipped)} skipped")
    lines.extend(
        f"{report.league}: {'DRIFT' if finding.is_drift else 'unchecked'} {finding.line()}"
        for finding in report.findings
    )
    if report.ok:
        lines.append(f"{report.league}: ok")
    return lines


# --- running ----------------------------------------------------------------------------------------------------------


def _league_row(league: League) -> LeagueRow:
    """An unsaved row with the fields a runtime reads; the canary does not need the store."""
    return LeagueRow(
        key=league.key,
        sport=league.sport,
        espn_league_id=league.espn_league_id,
        season=league.season,
        team_id=league.team_id,
        as_of=datetime.now(UTC),
    )


def _run_live(league: League, options: LaunchOptions) -> CanaryReport:
    """One league on the real browser profile. ``dry_run=True`` makes the context abort every request that is not a GET
    to ESPN and gives the runtime a transport that refuses every send."""
    with live_opener(options)(_league_row(league), dry_run=True) as runtime:
        page = runtime.browser.new_page()
        try:
            return run_canary(league, reader=runtime.reader, page=page)
        finally:
            page.close()


def _run_fixture(league: League, directory: Path) -> CanaryReport:
    with _recorded_reader(league, directory) as reader:
        return run_canary(league, reader=reader, page=None)


@contextmanager
def _recorded_reader(league: League, directory: Path) -> Iterator[EspnClient]:
    views = RecordedViews(directory)
    with httpx.Client(transport=httpx.MockTransport(views.handle)) as http:
        yield EspnClient(
            Game.coerce(league.game),
            league.espn_league_id,
            league.season,
            None,
            client=http,
            capture=False,
            min_interval_s=0.0,
            max_attempts=1,
            sleep=lambda _seconds: None,
        )


def _send_alert(config: Config, reports: list[CanaryReport]) -> None:
    message = drift_alert(reports)
    if message is None:
        return
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


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)


def register(root: typer.Typer) -> None:
    root.command("canary")(canary)
