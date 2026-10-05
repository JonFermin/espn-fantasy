"""``fm sync``: pull league state and the sources into the store (DESIGN sections 12, 13).

The command reads ``config.toml``, harvests the ESPN session from the browser profile (``fm login`` fills it), runs
:func:`fm.jobs.sync.sync` and prints one line per league and per source dataset, with each source's freshness state
(fresh, cached, stale, degraded) and warnings, then one line per gate. An unmapped rostered player fails the command:
the players are listed on stderr and the exit code is 1, while everything else that was read stays in the store.
"""

from __future__ import annotations

from typing import Annotated, NoReturn

import typer

from fm.browser.session import BrowserError
from fm.config import ConfigError, load_config
from fm.espn.auth import AuthError, describe, load_session
from fm.espn.client import EspnClientError
from fm.jobs.sync import DEFAULT_POOL_SIZE, SyncError, SyncReport, select_leagues, sync
from fm.model.ids import CrosswalkError
from fm.sources.base import SourceError
from fm.store import Store

LeagueOption = Annotated[
    list[str] | None,
    typer.Option("--league", "-l", help="Only these league keys from config.toml (repeatable). Default: every league."),
]
ForceOption = Annotated[bool, typer.Option("--force", help="Refresh every source, ignoring its cache TTL.")]
PoolOption = Annotated[
    int,
    typer.Option("--pool", min=0, help="Free agents (and players on waivers) to pull per league, most-owned first."),
]


def sync_(league: LeagueOption = None, force: ForceOption = False, pool: PoolOption = DEFAULT_POOL_SIZE) -> None:
    """Pull league state (settings, teams, rosters, player pool, projections) and the sources into the store.

    Runs the unmapped-rostered-player gate: exits 1 naming every rostered player the id crosswalk lacks.
    """
    try:
        config = load_config()
        select_leagues(config, league)  # a mistyped key fails here, before a browser starts or the store opens
        session = load_session()
    except (ConfigError, SyncError, AuthError, BrowserError) as exc:
        _fail(str(exc))
    typer.echo(describe(session))
    try:
        with Store.open() as store:
            report = sync(store, config, session=session, leagues=league, pool_size=pool, force=force)
    except (SyncError, EspnClientError, SourceError, CrosswalkError) as exc:
        _fail(str(exc))
    for line in render(report):
        typer.echo(line)
    if not report.ok:
        for failed in report.failed:
            if failed.gate is not None and failed.gate.error is not None:
                typer.echo(f"error: {failed.key}: {failed.gate.error}", err=True)
        raise typer.Exit(1)


def render(report: SyncReport) -> list[str]:
    """The report as lines: leagues, then sources (each with its warnings indented), then the gates."""
    lines: list[str] = []
    for league in report.leagues:
        lines.append(league.describe())
        lines.extend(f"  warning: {warning}" for warning in league.warnings)
    for source in report.sources:
        lines.append(source.describe())
        lines.extend(f"  warning: {warning}" for warning in source.warnings)
    for league in report.leagues:
        gate = league.gate
        if gate is None:
            lines.append(f"gate {league.key}: not checked")
        elif gate.passed:
            lines.append(f"gate {league.key}: {gate.checked} rostered players mapped")
        else:
            lines.append(f"gate {league.key}: FAILED, {len(gate.unmapped)} of {gate.checked} rostered players unmapped")
    return lines


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)


def register(root: typer.Typer) -> None:
    root.command("sync")(sync_)
