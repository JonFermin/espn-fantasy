"""``fm doctor [--online] [--json]``: what is broken and how to fix it, from :func:`fm.health.run_checks`.

One line per check (``FAIL  session  no browser profile yet ...  → fix: fm login``), or a JSON list with ``--json``
for scripts and the dashboard. The exit code is 1 when any check is a FAIL, so ``fm setup`` and shell scripts can
gate on it. ``--online`` adds the ESPN read per league and the phone-channel ping; without it nothing leaves the
machine. Unlike the dashboard and the MCP tool, ``fm doctor`` may launch the browser to read the ESPN session
(``launch_browser=True``), but only when the last tick's session record is stale and the run lock is free, so it never
starts a browser beside a running tick.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Annotated

import typer

from fm.health import Check, has_failure, run_checks

OnlineOption = Annotated[
    bool,
    typer.Option("--online", help="Also read each league's mStatus from ESPN and ping the phone channel."),
]
JsonOption = Annotated[bool, typer.Option("--json", help="Print the checks as a JSON list instead of lines.")]


def doctor(online: OnlineOption = False, as_json: JsonOption = False) -> None:
    """Check config, ESPN session, browser, phone channel, ticks, syncs, pause, proposals and Claude spend.

    Exits 1 when any check fails; each failing line names the command that fixes it.
    """
    checks = run_checks(online=online, launch_browser=True)
    if as_json:
        typer.echo(json.dumps([check.as_dict() for check in checks], indent=2))
    else:
        for line in check_lines(checks):
            typer.echo(line)
    if has_failure(checks):
        raise typer.Exit(code=1)


def check_lines(checks: Sequence[Check]) -> list[str]:
    """``STATUS  name  detail`` per check, the detail folded onto one line, plus ``→ fix: <command>`` when set."""
    width = max((len(check.name) for check in checks), default=0)
    lines: list[str] = []
    for check in checks:
        detail = " ".join(check.detail.split())
        line = f"{check.status.value.upper():<4}  {check.name:<{width}}  {detail}"
        if check.fix:
            line += f"  → fix: {check.fix}"
        lines.append(line)
    return lines


def register(root: typer.Typer) -> None:
    root.command("doctor")(doctor)
