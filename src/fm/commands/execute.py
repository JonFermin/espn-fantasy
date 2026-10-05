"""``fm execute <proposal> [--dry-run]``: carry out one approved proposal on ESPN (DESIGN sections 6.3 and 12).

This command and the tick are the ways into :func:`fm.executor.execute`, the one write path to ESPN (CLAUDE.md). It
reads the proposal's single-use execution token from the store (``fm proposals approve`` minted it) and hands it to
the executor, which checks preconditions through the API, sends the web app's own request (UI click-through as the
fallback) and verifies the change by re-reading the league. A second run is refused: the token is spent.

``--dry-run`` takes a proposed or approved proposal and sends nothing: it checks the preconditions, then builds and
saves the request, or with ``--mode ui`` walks the page up to its final confirm, and stops. During development, use
nothing else (CLAUDE.md: no live ESPN writes). Every run prints the audit folder with its request, response,
verification, screenshots and trace.
"""

from __future__ import annotations

import json
from typing import Annotated, Any, NoReturn

import typer

from fm.browser.flows import Mode
from fm.browser.session import BrowserError, Channel, LaunchOptions
from fm.espn.auth import AuthError
from fm.executor import ExecutionResult, ExecutorError, execute, live_opener
from fm.proposals import ProposalError, get_proposal, parse_payload, pause_state
from fm.store import ExecutionRow, Store

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


def execute_(
    proposal_id: ProposalId,
    dry_run: DryRunOption = False,
    mode: ModeOption = None,
    headed: HeadedOption = False,
    channel: ChannelOption = None,
) -> None:
    """Carry out an approved proposal: preconditions, one write, a re-read to verify it. --dry-run sends nothing."""
    launch = LaunchOptions(headless=not headed, channel=None if channel is None else channel.value)
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
                opener=live_opener(launch),
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
        codes = ", ".join(str(code) for code in response.get("error_codes") or ())
        lines.append(f"  response: HTTP {response['status']}" + (f" {codes}" if codes else ""))
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


def register(root: typer.Typer) -> None:
    root.command("execute")(execute_)
