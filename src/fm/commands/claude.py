"""``fm claude install|uninstall|show``: connect Claude Code and Claude Desktop to ``fm`` (PLUG_AND_PLAY section 8).

``install`` registers the MCP server ``espn-fantasy`` (``<uv> run --project <repo> fm mcp``) with Claude Code through
``claude mcp add-json --scope user`` and copies the skill and the slash commands (``/espn-fantasy:week``, ``:lineup``,
``:waivers``, ``:trade``) into ``~/.claude``; it merges the same server into Claude Desktop's
``claude_desktop_config.json`` (backed up first, every other entry kept). ``--no-code`` / ``--no-desktop`` skip a
client, ``--dry-run`` prints what would happen and changes nothing. ``uninstall`` removes exactly those pieces.
``show`` reports what is installed where, from the files alone, and never fails.

The logic is :mod:`fm.claude.install`; this module only wires it to the real home folder, ``claude`` on PATH and
:func:`fm.jobs.scheduler_base.default_runner`, through the hooks below that tests replace. Exits 1 when a part failed.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from typing import Annotated, NoReturn

import typer

from fm.claude import (
    ClaudeError,
    ClaudeHome,
    ClaudeReport,
    McpCommand,
    claude_status,
    install_claude,
    uninstall_claude,
)
from fm.jobs.scheduler_base import RunResult, default_runner

app = typer.Typer(
    help="Connect Claude Code and Claude Desktop to fm: the espn-fantasy MCP server, a skill and slash commands.",
    no_args_is_help=True,
)

CodeOption = Annotated[
    bool,
    typer.Option(
        "--code/--no-code",
        help="Claude Code: the MCP server (claude mcp add-json, user scope) plus the skill and slash commands in "
        "~/.claude.",
    ),
]
DesktopOption = Annotated[
    bool,
    typer.Option(
        "--desktop/--no-desktop",
        help="Claude Desktop: the MCP server in claude_desktop_config.json (backed up first; other entries kept).",
    ),
]
DryRunOption = Annotated[bool, typer.Option("--dry-run", help="Print what would change without changing anything.")]


# --- hooks (tests replace these) --------------------------------------------------------------------------------------


def _claude_home() -> ClaudeHome:
    return ClaudeHome.current()


def _mcp_command(*, require_uv: bool = True) -> McpCommand:
    return McpCommand.build(require_uv=require_uv)


def _runner(argv: Sequence[str]) -> RunResult:
    return default_runner(argv)


def _which(name: str) -> str | None:
    return shutil.which(name)


# --- commands ---------------------------------------------------------------------------------------------------------


@app.command("install")
def install_(code: CodeOption = True, desktop: DesktopOption = True, dry_run: DryRunOption = False) -> None:
    """Register the espn-fantasy MCP server with Claude Code and Claude Desktop, and install the skill and slash
    commands for Claude Code. Re-running refreshes everything."""
    try:
        command = _mcp_command()
    except ClaudeError as exc:
        _fail(str(exc))
    typer.echo(f"server: {command.display()}")
    _finish(
        install_claude(
            _claude_home(), command, code=code, desktop=desktop, dry_run=dry_run, runner=_runner, which=_which
        )
    )


@app.command("uninstall")
def uninstall_(code: CodeOption = True, desktop: DesktopOption = True, dry_run: DryRunOption = False) -> None:
    """Remove what install added (the MCP server registrations, the skill and the slash commands) and nothing else."""
    _finish(uninstall_claude(_claude_home(), code=code, desktop=desktop, dry_run=dry_run, runner=_runner, which=_which))


@app.command("show")
def show_() -> None:
    """Show what is installed where: the Claude Code server, skill and commands, and each Claude Desktop config."""
    status = claude_status(_claude_home(), _mcp_command(require_uv=False), which=_which)
    for line in status.lines():
        typer.echo(line)


def _finish(report: ClaudeReport) -> None:
    for line in report.lines:
        typer.echo(line)
    if report.failed:
        raise typer.Exit(1)


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)


def register(root: typer.Typer) -> None:
    root.add_typer(app, name="claude")
