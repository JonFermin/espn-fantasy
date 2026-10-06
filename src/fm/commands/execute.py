"""``fm execute <proposal> [--dry-run]``: carry out one approved proposal on ESPN (DESIGN sections 6.3 and 12).

This command and the tick are the ways into :func:`fm.executor.execute`, the one write path to ESPN (CLAUDE.md). It
reads the proposal's single-use execution token from the store (``fm proposals approve`` minted it) and hands it to
the executor, which checks preconditions through the API, sends the web app's own request (UI click-through as the
fallback) and verifies the change by re-reading the league. A second run is refused: the token is spent.

``--dry-run`` takes a proposed or approved proposal and sends nothing: it checks the preconditions, then builds and
saves the request, or with ``--mode ui`` walks the page up to its final confirm, and stops. During development, use
nothing else (CLAUDE.md: no live ESPN writes). Every run prints the audit folder with its request, response,
verification, screenshots and trace, and explains any ESPN error code in the answer (``fm.browser.transactions``).

``--dry-run --fixtures DIR`` reads the league from recorded views instead of ESPN, so a dry run works offline and
without ``fm login``. ``DIR`` holds one ``<view>.json`` per read (``mRoster.json``, ``mSettings.json``,
``mTeam+mStandings.json``), as ``tests/fixtures/espn/real/ffl`` and ``.../fba`` do. They are served through an
``httpx.MockTransport`` whatever the request's filter or scoring period, so the proposal must be for the period the
files were recorded in. No browser opens and the write transport refuses every send. The memberId shown is our
team's owner in the recorded ``mTeam+mStandings.json``, when the folder has one. Recorded views never back a real
write, so ``--fixtures`` without ``--dry-run`` is refused.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from pathlib import Path
from typing import Annotated, Any, NoReturn

import httpx
import typer

from fm.browser.flows import Mode, ModeUnavailableError, PageLike
from fm.browser.session import BrowserError, Channel, LaunchOptions
from fm.browser.transactions import error_code
from fm.espn.auth import AuthError
from fm.espn.client import READS_HOST, EspnClient, EspnClientError
from fm.espn.ids import Game
from fm.executor import (
    ExecutionResult,
    ExecutorError,
    RefusingTransport,
    Runtime,
    RuntimeOpener,
    execute,
    live_opener,
)
from fm.proposals import ProposalError, get_proposal, parse_payload, pause_state
from fm.store import ExecutionRow, LeagueRow, Store

ProposalId = Annotated[int, typer.Argument(min=1, help="Proposal id, as shown by fm proposals list.")]
DryRunOption = Annotated[
    bool,
    typer.Option(
        "--dry-run",
        help="Check preconditions and build the request (or walk the UI to its final confirm), then stop. "
        "Sends nothing; works on proposed proposals too.",
    ),
]
ModeOption = Annotated[
    Mode | None,
    typer.Option("--mode", case_sensitive=False, help="Run only this mode instead of API first with UI as fallback."),
]
HeadedOption = Annotated[bool, typer.Option("--headed", help="Show the browser window, to watch a UI walk.")]
ChannelOption = Annotated[
    Channel | None, typer.Option(help="Browser to drive. Default: the first one installed, Edge before Chrome.")
]
FixturesOption = Annotated[
    Path | None,
    typer.Option(
        "--fixtures",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
        help="With --dry-run: read the league from recorded views in this folder (<view>.json files, such as "
        "tests/fixtures/espn/real/ffl) instead of ESPN. Opens no browser and needs no fm login.",
    ),
]

FIXTURE_DRY_RUN = "fixture dry run"
_VIEW_KEY = re.compile(r"[A-Za-z0-9_]+(?:\+[A-Za-z0-9_]+)*")
"""What a recorded view's file stem may be: view names joined by ``+``, nothing that walks out of the folder."""


def execute_(
    proposal_id: ProposalId,
    dry_run: DryRunOption = False,
    mode: ModeOption = None,
    headed: HeadedOption = False,
    channel: ChannelOption = None,
    fixtures: FixturesOption = None,
) -> None:
    """Carry out an approved proposal: preconditions, one write, a re-read to verify it. --dry-run sends nothing."""
    if fixtures is not None:
        if not dry_run:
            _fail("--fixtures works only with --dry-run: recorded views never back a real write")
        if mode is Mode.UI:
            _fail("--fixtures opens no browser, so it cannot walk the UI; drop --mode ui")
        typer.echo(f"note: reading recorded views from {fixtures}; no browser, and nothing can be sent")
        opener = fixture_opener(fixtures)
    else:
        opener = live_opener(LaunchOptions(headless=not headed, channel=None if channel is None else channel.value))
    with Store.open() as store:
        try:
            row = get_proposal(store, proposal_id)
            paused = pause_state()
            if dry_run and paused is not None:
                typer.echo(f"note: {paused.describe()}; a dry run sends nothing, but nothing executes until fm resume")
            result = execute(
                store,
                proposal_id,
                token=None if dry_run else row.execution_token,
                dry_run=dry_run,
                mode=mode,
                opener=opener,
            )
        except (ProposalError, ExecutorError, AuthError, BrowserError) as exc:
            _fail(str(exc))
        league = store.leagues.get(result.proposal.league_id)
        key = league.key if league is not None else f"league {result.proposal.league_id}"
    for line in result_lines(result, key):
        typer.echo(line)
    if not result.ok:
        raise typer.Exit(1)


def result_lines(result: ExecutionResult, league_key: str) -> list[str]:
    """What ``fm execute`` prints for a run: the move, preconditions, each attempt, the outcome, the audit folder."""
    proposal = result.proposal
    summary = parse_payload(proposal).summary()
    lines = [f"#{proposal.row_id} {proposal.kind} in {league_key} via {result.flow}: {summary}"]
    pre = result.preconditions
    lines.append("preconditions: ok" if pre.ok else "preconditions: FAILED: " + "; ".join(pre.failures))
    for attempt in result.attempts:
        lines.extend(_attempt_lines(attempt, dry_run=result.dry_run))
    if result.dry_run:
        verdict = "it would have been sent" if result.ok else "it would not have been sent"
        lines.append(f"dry run: nothing was sent; {verdict}")
    else:
        lines.append(f"proposal #{proposal.row_id}: {proposal.status}")
    lines.append(f"audit: {result.audit_dir}")
    return lines


def _attempt_lines(row: ExecutionRow, *, dry_run: bool) -> list[str]:
    head = f"{row.mode}: {'dry run' if row.status == 'dry_run' else row.status}"
    lines = [head + (f": {row.error}" if row.error else "")]
    request: dict[str, Any] = row.request or {}
    if "url" in request and row.mode == "api":
        lines.append(f"  request: {request.get('method', 'POST')} {request['url']}")
        if dry_run and "body" in request:
            lines.append("  body: " + json.dumps(request["body"], sort_keys=True, separators=(",", ":")))
    if request.get("stopped_before"):
        lines.append(f"  stopped before: {request['stopped_before']}")
    elif request.get("confirms"):
        lines.append("  confirmed: " + ", ".join(str(step) for step in request["confirms"]))
    response: dict[str, Any] = row.response or {}
    if "status" in response:
        codes = [str(code) for code in response.get("error_codes") or ()]
        lines.append(f"  response: HTTP {response['status']}" + (f" {', '.join(codes)}" if codes else ""))
        lines.extend(f"  ESPN {error_code(code).describe()}" for code in codes)
    if row.espn_transaction_id:
        lines.append(f"  ESPN transaction: {row.espn_transaction_id}")
    verification: dict[str, Any] = row.verification or {}
    if verification:
        matched = verification.get("matched")
        state = "shows the change" if matched else "unreadable" if matched is None else "does not show the change"
        detail = verification.get("detail") or ""
        lines.append(f"  re-read ({verification.get('reads', 0)}x): {state}" + (f": {detail}" if detail else ""))
    return lines


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)


# --- recorded views (--fixtures) --------------------------------------------------------------------------------------


class RecordedViews:
    """ESPN's read host answered from a folder of recorded views, for an ``httpx.MockTransport``.

    A GET whose ``view=`` params join to ``<view>`` (``mTeam+mStandings``) gets ``<directory>/<view>.json``, whatever
    its filter or scoring period. A view the folder lacks is a 404 in ESPN's error shape; anything but a GET to the
    read host is a 405, so nothing reaches ESPN.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.served: list[str] = []
        """The view of every request answered, in order."""

    def handle(self, request: httpx.Request) -> httpx.Response:
        if request.method != "GET" or request.url.host != READS_HOST:
            return _espn_error(405, f"recorded views answer GETs to {READS_HOST} only, not {request.method}")
        view = "+".join(request.url.params.get_list("view"))
        path = self.directory / f"{view}.json"
        if not _VIEW_KEY.fullmatch(view) or not path.is_file():
            return _espn_error(404, f"no recorded {view or 'view'} in {self.directory} ({view or 'view'}.json)")
        self.served.append(view)
        return httpx.Response(200, content=path.read_bytes(), headers={"Content-Type": "application/json"})


class NoBrowser:
    """The browser of a fixture dry run: there is none, so UI mode is unavailable (``fm.executor.BrowserLike``)."""

    def new_page(self) -> PageLike:
        raise ModeUnavailableError(f"a {FIXTURE_DRY_RUN} has no browser; run without --fixtures to walk the UI")

    def start_trace(self) -> None:
        return None

    def stop_trace(self, path: Path) -> None:
        return None


def fixture_opener(directory: Path) -> RuntimeOpener:
    """A ``fm.executor.RuntimeOpener`` over recorded views in ``directory``, for dry runs only."""

    def opener(league: LeagueRow, *, dry_run: bool) -> AbstractContextManager[Runtime]:
        return open_fixture_runtime(directory, league, dry_run=dry_run)

    return opener


@contextmanager
def open_fixture_runtime(directory: Path, league: LeagueRow, *, dry_run: bool) -> Iterator[Runtime]:
    """A runtime whose reads come from ``directory`` (:class:`RecordedViews`), whose transport refuses every send and
    which has no browser. Refuses (``ExecutorError``) to serve anything but a dry run."""
    if not dry_run:
        raise ExecutorError("recorded views back dry runs only; a real write reads the live league")
    views = RecordedViews(directory)
    with httpx.Client(transport=httpx.MockTransport(views.handle)) as http:
        reader = EspnClient(
            Game.from_sport(league.sport),
            league.espn_league_id,
            league.season,
            None,
            client=http,
            capture=False,
            min_interval_s=0.0,
            max_attempts=1,
            sleep=lambda _seconds: None,
        )
        yield Runtime(
            reader=reader,
            transport=RefusingTransport(FIXTURE_DRY_RUN),
            browser=NoBrowser(),
            member_id=_recorded_owner(reader, league),
            dry_run=True,
        )


def _recorded_owner(reader: EspnClient, league: LeagueRow) -> str | None:
    """Our team's owner (a SWID) in the recorded ``mTeam+mStandings``, the envelope's ``memberId``; ``None`` without
    one."""
    try:
        return reader.teams().data.team(league.team_id).primary_owner
    except (EspnClientError, KeyError):
        return None


def _espn_error(status: int, message: str) -> httpx.Response:
    return httpx.Response(status, json={"messages": [message], "details": []})


def register(root: typer.Typer) -> None:
    root.command("execute")(execute_)
