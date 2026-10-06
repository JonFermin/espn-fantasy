"""``fm mcp``: run the MCP server over stdio (DESIGN section 12, ROADMAP #40).

Point Claude Code or Claude Desktop at it, for example ``claude mcp add espn-fantasy -- uv run --directory <repo> fm
mcp``. The server's tools read the store ``fm sync`` filled and can store a ``proposed`` proposal; it has no way to
approve or execute anything (see :mod:`fm.mcp_server`). stdout carries the protocol, so nothing else may print there.
"""

from __future__ import annotations

import typer


def mcp() -> None:
    """Serve the read tools and create_proposal to an MCP client over stdio (stdout is the protocol channel)."""
    from fm.mcp_server import build_server  # fastmcp is slow to import; keep it out of every other command

    build_server().run(transport="stdio", show_banner=False)


def register(root: typer.Typer) -> None:
    root.command("mcp")(mcp)
