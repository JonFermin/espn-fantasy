"""``fm ui [--port 8711] [--no-open]``: open the local dashboard, serving it first when nothing does yet (DESIGN
section 12.4, ROADMAP #54).

The dashboard (:mod:`fm.web`) shows the leagues, the proposal queue with Approve and Reject, lineup and waiver plans,
the reports, settings and health. It is served on ``127.0.0.1`` only, records decisions without executing them (``fm
run`` or the scheduled tick does that), and answers only a browser signed in with a one-time link that the serving
process writes to ``<config dir>/dashboard.url`` (:mod:`fm.web.login`), a file only this user can read.

So ``fm ui`` first looks for a dashboard that is already served, by ``fm run`` (which serves one beside its loop
unless ``--no-ui``) or another ``fm ui``: the URL file names its port, the port answers, and that server puts a fresh
link in the file. It opens that link and exits. Otherwise it serves the dashboard itself on ``--port`` until Ctrl+C
and opens its link. ``--no-open`` prints the link instead (it works once, for 10 minutes).
"""

from __future__ import annotations

import webbrowser
from typing import Annotated, NoReturn

import typer

from fm.web import DEFAULT_PORT, LOOPBACK

PortOption = Annotated[
    int,
    typer.Option("--port", min=1, max=65535, help="Port on 127.0.0.1 to serve the dashboard on, if none is served."),
]
NoOpenOption = Annotated[
    bool, typer.Option("--no-open", help="Print the one-time sign-in link instead of opening the browser.")
]


def ui(port: PortOption = DEFAULT_PORT, no_open: NoOpenOption = False) -> None:
    """Open the local dashboard in the browser, signed in; serves it on 127.0.0.1 first unless fm run already does
    (Ctrl+C stops a dashboard this command serves)."""
    from fm.web import login  # standard library only: finding a running dashboard loads no web stack

    running = login.find_running()
    if running is not None:
        served = login.port_of(running)
        typer.echo(f"the dashboard is already served at http://{LOOPBACK}:{served}/ (by fm run or another fm ui)")
        _open(running, no_open)
        return

    from fm.web.server import bind_dashboard  # Starlette, uvicorn and the views load only to serve

    try:
        dashboard = bind_dashboard(port)
    except OSError as exc:
        _fail(
            f"cannot serve on {LOOPBACK}:{port}: {exc.strerror or exc}. Another program has the port, or a dashboard "
            "from another config dir does; pick another port with --port"
        )
    typer.echo(f"dashboard at {dashboard.url} (Ctrl+C stops)")
    _open(dashboard.login_url(), no_open)
    try:
        dashboard.serve()
    except KeyboardInterrupt:
        pass
    typer.echo("stopped")


def _open(link: str, no_open: bool) -> None:
    """Open the sign-in link, or print it when asked not to open the browser or the browser will not open."""
    if not no_open and webbrowser.open(link):
        return
    typer.echo(f"sign in with this one-time link (it works once, for 10 minutes): {link}")


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)


def register(root: typer.Typer) -> None:
    root.command("ui")(ui)
