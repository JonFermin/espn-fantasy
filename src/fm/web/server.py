"""Serving the dashboard (DESIGN section 12.4): one socket on ``127.0.0.1`` and uvicorn on it.

:func:`bind_dashboard` binds the socket itself, before uvicorn starts, for three reasons:

- A port that is taken fails right there with ``OSError``, in the caller's thread. ``fm ui`` reports it and exits 1;
  ``fm run`` logs it and runs its loop without the dashboard. uvicorn would instead log the error and call
  ``sys.exit`` from inside its event loop.
- The app's ``Host`` check needs the port actually bound (``port=0`` picks a free one).
- The browser can be opened at once: connections queue on the bound socket until uvicorn accepts them. So the app's
  one-time sign-in link is written to ``<config dir>/dashboard.url`` right away (:mod:`fm.web.login`), and removed
  when the dashboard stops.

On Windows the socket takes the port exclusively (``SO_EXCLUSIVEADDRUSE``, so no other process can bind it with
``SO_REUSEADDR`` and receive the dashboard's requests); elsewhere it sets ``SO_REUSEADDR``, which there only lets a
restart reuse a port still in ``TIME_WAIT``. Both are feature checks on the ``socket`` module, not platform branches.

:class:`Dashboard` then serves in the foreground (:meth:`Dashboard.serve`, ``fm ui``: uvicorn installs its Ctrl+C and
SIGTERM handlers because it runs in the main thread) or in a daemon thread (:meth:`Dashboard.start`, ``fm run``:
uvicorn installs signal handlers only in the main thread, so the loop's own Ctrl+C and SIGTERM handling is untouched,
and :meth:`Dashboard.stop` ends it when the loop ends). uvicorn's logging config is left alone (``log_config=None``):
its warnings propagate to the root logger, and the app logs its own errors under ``fm.web.app``, which ``fm run`` sends
to ``logs/run.log``.
"""

from __future__ import annotations

import logging
import socket
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import uvicorn
from starlette.applications import Starlette

from fm.web import DEFAULT_PORT, LOOPBACK
from fm.web.app import create_app
from fm.web.login import LoginLinks, url_file_path

logger = logging.getLogger(__name__)

STOP_TIMEOUT_S: Final = 5.0
"""How long :meth:`Dashboard.stop` waits for the server thread to finish."""
GRACEFUL_SHUTDOWN_S: Final = 2
"""How long uvicorn lets open requests finish once asked to stop."""


@dataclass(slots=True)
class Dashboard:
    """A dashboard bound to its port and ready to serve. Built by :func:`bind_dashboard`."""

    port: int
    app: Starlette
    sock: socket.socket
    server: uvicorn.Server
    thread: threading.Thread | None = field(default=None)

    @property
    def url(self) -> str:
        """``http://127.0.0.1:<port>/``: the dashboard's address (a browser that is not signed in gets told to run
        ``fm ui`` there)."""
        return f"http://{LOOPBACK}:{self.port}/"

    def login_url(self) -> str:
        """The current one-time sign-in link (:class:`fm.web.login.LoginLinks`): what ``fm ui`` opens. Secret: never
        logged."""
        links: LoginLinks = self.app.state.login
        return links.url()

    def serve(self) -> None:
        """Serve in this thread until stopped. In the main thread uvicorn handles Ctrl+C itself: it shuts down, then
        re-raises the signal, so the caller sees ``KeyboardInterrupt`` once the server is down. The URL file goes
        with the server."""
        try:
            self.server.run(sockets=[self.sock])
        finally:
            self.release()

    def release(self) -> None:
        """Remove ``dashboard.url`` if it still holds this dashboard's link (idempotent)."""
        links: LoginLinks = self.app.state.login
        links.close()

    def start(self) -> None:
        """Serve in a daemon thread (``fm-dashboard``); an error there is logged, never raised into the caller."""
        thread = threading.Thread(target=self._serve_logged, name="fm-dashboard", daemon=True)
        self.thread = thread
        thread.start()

    def stop(self, timeout: float = STOP_TIMEOUT_S) -> None:
        """Ask the server to stop and wait up to ``timeout`` seconds for its thread; closes the socket if it never
        served."""
        self.server.should_exit = True
        if self.thread is not None:
            self.thread.join(timeout)
            if self.thread.is_alive():
                logger.warning("dashboard: the server thread did not stop within %.0f s", timeout)
        elif not self.server.started:
            self.sock.close()
        self.release()

    def _serve_logged(self) -> None:
        try:
            self.serve()
        except (Exception, SystemExit):  # uvicorn ends a failed start with sys.exit
            logger.exception("dashboard: the server at %s stopped with an error", self.url)


def bind_dashboard(port: int = DEFAULT_PORT, *, url_file: Path | None = None) -> Dashboard:
    """Bind ``127.0.0.1:<port>`` and build the app and the uvicorn server for it; nothing is served yet, but the
    one-time sign-in link is already in ``url_file`` (default ``<config dir>/dashboard.url``,
    :func:`fm.web.login.url_file_path`), since connections queue on the bound socket.

    Raises ``OSError`` when the port is taken or cannot be bound. ``port=0`` binds a free port (``Dashboard.port`` says
    which).
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        exclusive = getattr(socket, "SO_EXCLUSIVEADDRUSE", None)
        if exclusive is not None:
            sock.setsockopt(socket.SOL_SOCKET, exclusive, 1)
        else:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((LOOPBACK, port))
        sock.listen()
    except OSError:
        sock.close()
        raise
    bound = int(sock.getsockname()[1])
    try:
        app = create_app(port=bound, url_file=url_file if url_file is not None else url_file_path())
    except BaseException:
        sock.close()
        raise
    config = uvicorn.Config(
        app,
        host=LOOPBACK,
        port=bound,
        interface="asgi3",
        http="h11",
        ws="none",
        loop="asyncio",
        lifespan="off",
        log_config=None,
        log_level="warning",
        access_log=False,
        proxy_headers=False,
        server_header=False,
        timeout_graceful_shutdown=GRACEFUL_SHUTDOWN_S,
    )
    return Dashboard(port=bound, app=app, sock=sock, server=uvicorn.Server(config))
