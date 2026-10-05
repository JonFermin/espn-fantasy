"""``fm proposals list|approve|reject`` and the ``fm pause`` / ``fm resume`` kill switch (DESIGN sections 11, 12).

The CLI is the first approval channel; the phone channels (ROADMAP #18) call the same ``fm.proposals`` functions.
Listing sweeps expired proposals first so what is shown as open really is. No command here needs ``config.toml``:
decisions act on proposals that already cleared policy, and the list works against an empty state database.
"""

from __future__ import annotations

from typing import Annotated, NoReturn

import typer

from fm.proposals import (
    ProposalError,
    approve,
    expire_due,
    is_open,
    parse_payload,
    pause,
    pause_state,
    reject,
    resume,
)
from fm.store import OPEN_PROPOSAL_STATUSES, ProposalRow, Store

app = typer.Typer(help="Review and decide proposals.", no_args_is_help=True)
OPEN = ", ".join(OPEN_PROPOSAL_STATUSES)

ProposalId = Annotated[int, typer.Argument(min=1, help="Proposal id, as shown by fm proposals list.")]
ByOption = Annotated[str, typer.Option("--by", help="Who is deciding; recorded as decided_by.")]


@app.command("list")
def list_(
    league: Annotated[str | None, typer.Option("--league", "-l", help="Only this league key from config.toml.")] = None,
    show_all: Annotated[bool, typer.Option("--all", "-a", help="Include decided and finished proposals.")] = False,
) -> None:
    """Show open proposals (proposed, approved, executing); --all shows every proposal."""
    with Store.open() as store:
        expire_due(store)
        keys = {row.row_id: row.key for row in store.leagues.all()}
        league_id = None
        if league is not None:
            league_id = next((row_id for row_id, key in keys.items() if key == league), None)
            if league_id is None:
                _fail(f"no league {league!r} in the store; known: {', '.join(sorted(keys.values())) or 'none'}")
        rows = store.proposals.find(league_id=league_id)
        if not show_all:
            rows = [row for row in rows if is_open(row)]
        paused = pause_state()
        if paused is not None:
            typer.echo(f"PAUSED: {paused.describe()}; run fm resume")
        if not rows:
            typer.echo("no proposals" if show_all else f"no open proposals ({OPEN})")
            return
        for row in rows:
            typer.echo(_line(row, keys.get(row.league_id, f"league {row.league_id}")))


@app.command("approve")
def approve_(proposal_id: ProposalId, by: ByOption = "cli") -> None:
    """Approve a proposal; it runs at its time (fm execute, or the tick). Void once its deadline passes."""
    with Store.open() as store:
        try:
            row = approve(store, proposal_id, decided_by=by)
        except ProposalError as exc:
            _fail(str(exc))
    typer.echo(f"#{row.row_id} approved: {row.kind} {parse_payload(row).summary()}")


@app.command("reject")
def reject_(proposal_id: ProposalId, by: ByOption = "cli") -> None:
    """Reject a proposal, or withdraw an approval that has not started executing."""
    with Store.open() as store:
        try:
            row = reject(store, proposal_id, decided_by=by)
        except ProposalError as exc:
            _fail(str(exc))
    typer.echo(f"#{row.row_id} rejected: {row.kind} {parse_payload(row).summary()}")


ReasonOption = Annotated[str | None, typer.Option("--reason", "-r", help="Why; shown wherever the pause is reported.")]


def pause_(reason: ReasonOption = None) -> None:
    """Kill switch: stop new proposals, auto-approvals and every execution until fm resume."""
    state = pause(reason)
    typer.echo(state.describe())


def resume_() -> None:
    """Lift fm pause."""
    state = resume()
    typer.echo("resumed" if state is None else f"resumed ({state.describe()})")


def _line(row: ProposalRow, league_key: str) -> str:
    deadline = "-" if row.deadline is None else f"{row.deadline:%Y-%m-%d %H:%M}Z"
    decided = f" by {row.decided_by}" if row.decided_by and row.status not in ("proposed", "executing") else ""
    return (
        f"#{row.row_id:<4} {league_key:<6} {row.kind:<14} {row.status + decided:<22} {row.policy:<8} "
        f"due {deadline:<17} {parse_payload(row).summary()}"
    )


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)


def register(root: typer.Typer) -> None:
    root.add_typer(app, name="proposals")
    root.command("pause")(pause_)
    root.command("resume")(resume_)
