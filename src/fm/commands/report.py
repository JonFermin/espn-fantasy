"""``fm report``: the weekly report (DESIGN section 9.5, ROADMAP #46): matchup outlook, playoff odds, moves made and
upcoming deadlines per league, as a markdown file and a quiet phone push.

Read-only: nothing here has a write path to ESPN. The markdown goes to ``--out FILE``, else
``<cache dir>/reports/weekly-YYYY-MM-DD.md`` (``paths.cache_dir()``), and the same text goes to the phone through
``fm.notify.send_report`` (long reports are split by the channel) unless ``--no-notify``. A phone that is not set up
or cannot be reached is a note, never a failure: the file is the report of record.

Inputs, as for ``fm lineup``: ``--schedule FILE`` or the captured pro schedule; ``--matchups FILE`` or the captured
``mMatchup`` or a live read. ``--fixtures DIR`` takes both from recorded views in a folder (``proTeamSchedules_wl.json``
and ``mMatchup.json``, such as ``tests/fixtures/espn/real/ffl``) and reads nothing from ESPN. A missing input degrades
its sections with a note. ``--as-of`` pins the clock.

Fixture home: ``FM_CONFIG_DIR=tests/fixtures/home FM_CACHE_DIR=tests/fixtures/home/cache uv run fm report --no-notify
--as-of 2026-10-04T15:00Z --fixtures tests/fixtures/espn/real/ffl --out report.md`` (copy the home first; the report
writes next to it otherwise).
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path
from typing import Annotated

import httpx
import typer

from fm.commands.advise import (
    AsOfOption,
    LeagueOption,
    ScheduleOption,
    _config,
    _fail,
    _parse_as_of,
    _schedule,
    _select,
)
from fm.commands.trade import _matchups
from fm.config import Config
from fm.espn.client import View
from fm.espn.models import MatchupsView
from fm.eval.report_card import attach_report_card
from fm.jobs.report import (
    DEFAULT_WINDOW_DAYS,
    LeagueReport,
    build_league_report,
    previous_odds,
    remember_odds,
    render_markdown,
    report_path,
    report_title,
    reports_dir,
    write_report,
)
from fm.notify import NotifyError, open_channel, send_report
from fm.store import LeagueRow, Store

MatchupsOption = Annotated[
    Path | None,
    typer.Option(
        "--matchups",
        exists=True,
        dir_okay=False,
        resolve_path=True,
        help="A recorded mMatchup view to use as the league schedule instead of the cache or ESPN. One league only.",
    ),
]
FixturesOption = Annotated[
    Path | None,
    typer.Option(
        "--fixtures",
        exists=True,
        file_okay=False,
        resolve_path=True,
        help="Read the pro schedule and mMatchup from recorded views in this folder (<view>.json, such as "
        "tests/fixtures/espn/real/ffl) instead of the cache or ESPN.",
    ),
]
OutOption = Annotated[
    Path | None,
    typer.Option(
        "--out", dir_okay=False, resolve_path=True, help="Write the markdown here instead of the reports dir."
    ),
]
DaysOption = Annotated[int, typer.Option("--days", min=1, help="Days of moves made, and of deadlines ahead.")]
NotifyOption = Annotated[
    bool, typer.Option("--notify/--no-notify", help="Push the report to the phone channel. A failure is a note.")
]


def report(
    league: LeagueOption = None,
    as_of: AsOfOption = None,
    schedule: ScheduleOption = None,
    matchups: MatchupsOption = None,
    fixtures: FixturesOption = None,
    out: OutOption = None,
    days: DaysOption = DEFAULT_WINDOW_DAYS,
    notify: NotifyOption = True,
) -> None:
    """Write the weekly report (matchup outlook, playoff odds, moves made, deadlines) and push it to the phone."""
    config = _config()
    at = _parse_as_of(as_of)
    chosen = _select(config, league)
    if matchups is not None and len(chosen) != 1:
        _fail("--matchups names one league's schedule; add --league KEY")
    if fixtures is not None:
        schedule = schedule or _recorded(fixtures, View.PRO_SCHEDULES)
        if matchups is None and len(chosen) == 1:
            matchups = _recorded(fixtures, View.MATCHUP)
    notes: list[str] = []
    reports: list[LeagueReport] = []
    directory = reports_dir()  # odds history stays here even with --out
    with Store.open() as store:
        for configured in chosen:
            row = store.leagues.by_key(configured.key)
            if row is None:
                notes.append(f"{configured.key}: not synced; run fm sync")
                continue
            loaded = _schedule(store, configured, path=schedule)
            view: MatchupsView | None = None
            if fixtures is not None and matchups is None:
                notes.append(f"{configured.key}: no mMatchup.json in {fixtures}")
            else:
                view, note = _matchups(store, configured, row, path=matchups)
                notes.extend(filter(None, [note]))
            notes.extend(loaded.warnings)
            if loaded.schedule is None:
                notes.append(loaded.note)
            built = build_league_report(
                store,
                config,
                row,
                now=at,
                schedule=loaded.schedule,
                matchups=view,
                previous=previous_odds(directory, _odds_key(row), before=at),
                window=timedelta(days=days),
            )
            built = attach_report_card(built, store, row, now=at, schedule=loaded.schedule)
            reports.append(built)
            if built.odds is not None:
                remember_odds(directory, _odds_key(row), built.odds)
    markdown = render_markdown(reports, as_of=at, notes=notes)
    path = write_report(markdown, out if out is not None else report_path(at, directory))
    typer.echo(f"wrote {path}")
    if notify:
        typer.echo(_push(config, report_title(reports, at), markdown))


def _odds_key(row: LeagueRow) -> str:
    """The odds history of one league season: a new season never trends against the last one."""
    return f"{row.key}-{row.season}"


def _recorded(directory: Path, view: View) -> Path | None:
    path = directory / f"{view.value}.json"
    return path if path.is_file() else None


def _push(config: Config, title: str, markdown: str) -> str:
    """Send the report to the phone; the outcome as one line (a failure never fails the command)."""
    body = markdown.split("\n", 2)[2] if markdown.startswith("# ") else markdown
    try:
        channel = open_channel(config)
        send_report(channel, title, body.strip())
    except NotifyError as exc:
        return f"phone: not sent ({exc})"
    except httpx.HTTPError as exc:
        return f"phone: not sent ({type(exc).__name__}: {exc})"
    return f"phone: sent via {config.notify.channel}"


def register(root: typer.Typer) -> None:
    root.command("report")(report)
