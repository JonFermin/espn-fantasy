"""``fm ui [--port 8711] [--no-open]``: serve the local dashboard and open it (PLUG_AND_PLAY section 7, ROADMAP #54).

The dashboard (:mod:`fm.web`) shows the leagues, the proposal queue with Approve and Reject, lineup and waiver plans,
the reports, settings and health. It is served on ``127.0.0.1`` only and records decisions without executing them:
``fm run`` or the scheduled tick does that. ``fm run`` serves the same dashboard beside its loop unless ``--no-ui``, so
this command is for when the loop is not running; on a port ``fm run`` already serves it says so and exits 1.
"""

from __future__ import annotations

import webbrowser
from typing import Annotated, NoReturn

import typer

from fm.web import DEFAULT_PORT, LOOPBACK

PortOption = Annotated[
    int,
    typer.Option("--port", min=1, max=65535, help="Port on 127.0.0.1 to serve the dashboard on."),
]
NoOpenOption = Annotated[bool, typer.Option("--no-open", help="Do not open the dashboard in the browser.")]


def ui(port: PortOption = DEFAULT_PORT, no_open: NoOpenOption = False) -> None:
    """Serve the local dashboard on 127.0.0.1 and open it in the browser; Ctrl+C stops."""
    from fm.web.server import bind_dashboard  # Starlette, uvicorn and the views load only for this command

    try:
        dashboard = bind_dashboard(port)
    except OSError as exc:
        _fail(
            f"cannot serve on {LOOPBACK}:{port}: {exc.strerror or exc}. If fm run is running, it serves the dashboard "
            f"at http://{LOOPBACK}:{port}/ already; otherwise pick another port with --port"
        )
    typer.echo(f"dashboard at {dashboard.url} (Ctrl+C stops)")
    if not no_open:
        webbrowser.open(dashboard.url)
    try:
        dashboard.serve()
    except KeyboardInterrupt:
        pass
    typer.echo("stopped")


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)


def register(root: typer.Typer) -> None:
    root.command("ui")(ui)
