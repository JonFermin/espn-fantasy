"""``fm ui`` and ``fm run --ui/--no-ui`` (ROADMAP #54), with the server replaced by a fake: no socket is bound here.

``fm ui`` binds, prints the URL, opens the browser unless ``--no-open``, serves until Ctrl+C, and exits 1 with a hint
when the port is taken. ``fm run`` starts the dashboard in a thread before its loop and stops it after; a port that is
taken, or a dashboard that fails to start for any other reason, is logged and printed and the loop runs anyway.
:class:`fm.web.server.Dashboard` itself is checked for the two behaviours ``fm run`` relies on, without serving: a
failure in its thread is logged rather than raised, and stopping a dashboard that never served releases its socket.
"""

from __future__ import annotations

import errno
import logging
import socket
from dataclasses import dataclass, field

import pytest
import typer
import uvicorn
from typer.testing import CliRunner

import fm.web.server as web_server
from fm.cli import build_app
from fm.commands import run as run_cmd
from fm.commands import ui as ui_cmd
from fm.jobs.runner import RunOptions, RunStats, run_log_path
from fm.web import DEFAULT_PORT
from fm.web.app import create_app

runner = CliRunner()


def _root() -> None:
    pass


def cli() -> typer.Typer:
    root = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)
    root.callback()(_root)
    ui_cmd.register(root)
    run_cmd.register(root)
    return root


@dataclass
class FakeDashboard:
    """What ``bind_dashboard`` returns, recording what is done with it; ``serve`` ends as Ctrl+C would."""

    port: int
    events: list[str] = field(default_factory=list[str])

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def serve(self) -> None:
        self.events.append("serve")
        raise KeyboardInterrupt

    def start(self) -> None:
        self.events.append("start")

    def stop(self, timeout: float = 5.0) -> None:
        self.events.append("stop")


@pytest.fixture
def events(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """``bind_dashboard`` replaced by a fake, and the browser by a recorder; both log into the returned list."""
    log: list[str] = []

    def bind(port: int = DEFAULT_PORT, *, token: str | None = None) -> FakeDashboard:
        log.append(f"bind {port}")
        return FakeDashboard(port, log)

    monkeypatch.setattr(web_server, "bind_dashboard", bind)
    monkeypatch.setattr(ui_cmd.webbrowser, "open", lambda url, *args, **kwargs: log.append(f"open {url}") or True)
    return log


def taken(port: int = DEFAULT_PORT, *, token: str | None = None) -> FakeDashboard:
    raise OSError(errno.EADDRINUSE, "Only one usage of each socket address is normally permitted")


# --- fm ui ------------------------------------------------------------------------------------------------------------


def test_fm_ui_serves_opens_the_browser_and_stops_on_ctrl_c(events: list[str]) -> None:
    result = runner.invoke(cli(), ["ui"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert events == ["bind 8711", "open http://127.0.0.1:8711/", "serve"]
    assert "dashboard at http://127.0.0.1:8711/ (Ctrl+C stops)" in result.output
    assert result.output.rstrip().endswith("stopped")


def test_fm_ui_no_open_serves_on_the_port_given_without_the_browser(events: list[str]) -> None:
    result = runner.invoke(cli(), ["ui", "--no-open", "--port", "8799"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert events == ["bind 8799", "serve"]


def test_fm_ui_on_a_taken_port_exits_1_and_says_what_to_do(events: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web_server, "bind_dashboard", taken)
    result = runner.invoke(cli(), ["ui", "--no-open"], catch_exceptions=False)
    assert result.exit_code == 1
    assert "error: cannot serve on 127.0.0.1:8711" in result.output
    assert "fm run" in result.output and "--port" in result.output
    assert events == []


def test_fm_ui_is_discovered_by_the_cli() -> None:
    result = runner.invoke(build_app(), ["ui", "--help"], catch_exceptions=False)
    assert result.exit_code == 0 and "--no-open" in result.output and "--port" in result.output
    help_run = runner.invoke(build_app(), ["run", "--help"], catch_exceptions=False)
    assert "--ui" in help_run.output and "--no-ui" in help_run.output


# --- fm run --ui ------------------------------------------------------------------------------------------------------


@pytest.fixture
def loop(monkeypatch: pytest.MonkeyPatch, events: list[str]) -> list[RunOptions]:
    """``run_forever`` replaced: it records its options and that it ran, between the dashboard's start and stop."""
    seen: list[RunOptions] = []

    def fake_run_forever(*, options: RunOptions) -> RunStats:
        seen.append(options)
        events.append("loop")
        return RunStats(ticks=1, stopped_by="stop")

    monkeypatch.setattr(run_cmd, "run_forever", fake_run_forever)
    return seen


def test_fm_run_serves_the_dashboard_beside_the_loop_by_default(loop: list[RunOptions], events: list[str]) -> None:
    result = runner.invoke(cli(), ["run"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert events == ["bind 8711", "start", "loop", "stop"]
    assert "dashboard at http://127.0.0.1:8711/" in result.output and loop == [RunOptions()]
    assert "fm run: dashboard at http://127.0.0.1:8711/" in run_log_path().read_text(encoding="utf-8")


def test_fm_run_takes_the_dashboard_port_or_leaves_the_dashboard_out(loop: list[RunOptions], events: list[str]) -> None:
    assert runner.invoke(cli(), ["run", "--port", "8800"], catch_exceptions=False).exit_code == 0
    assert events == ["bind 8800", "start", "loop", "stop"]
    events.clear()
    result = runner.invoke(cli(), ["run", "--no-ui"], catch_exceptions=False)
    assert result.exit_code == 0 and events == ["loop"] and "dashboard" not in result.output


def test_fm_run_on_a_taken_port_runs_the_loop_without_the_dashboard(
    loop: list[RunOptions], events: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(web_server, "bind_dashboard", taken)
    result = runner.invoke(cli(), ["run"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert events == ["loop"] and loop == [RunOptions()]
    assert "fm run: dashboard not started: 127.0.0.1:8711 is unavailable" in result.output  # the console log line
    assert "fm run --port N" in result.output
    log = run_log_path().read_text(encoding="utf-8")
    assert "WARNING fm.commands.run fm run: dashboard not started: 127.0.0.1:8711 is unavailable" in log


def test_fm_run_with_a_dashboard_that_fails_to_start_still_runs_the_loop(
    loop: list[RunOptions], events: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(port: int = DEFAULT_PORT, *, token: str | None = None) -> FakeDashboard:
        raise RuntimeError("fm/web/static/index.html is missing")

    monkeypatch.setattr(web_server, "bind_dashboard", broken)
    result = runner.invoke(cli(), ["run"], catch_exceptions=False)
    assert result.exit_code == 0 and events == ["loop"]
    assert "fm run: dashboard not started; the loop runs without it" in result.output
    assert "index.html is missing" in run_log_path().read_text(encoding="utf-8")  # with its traceback


# --- the Dashboard object ---------------------------------------------------------------------------------------------


def _unserved() -> web_server.Dashboard:
    """A Dashboard around an unbound socket (no network): what ``bind_dashboard`` builds, minus the bind."""
    app = create_app(port=DEFAULT_PORT)
    server = uvicorn.Server(uvicorn.Config(app, log_config=None))
    return web_server.Dashboard(port=DEFAULT_PORT, app=app, sock=socket.socket(), server=server)


def test_a_dashboard_that_fails_in_its_thread_logs_it_and_raises_nothing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    dashboard = _unserved()

    def failing(sockets: list[socket.socket] | None = None) -> None:
        raise SystemExit(3)  # what uvicorn does when its start-up fails

    monkeypatch.setattr(dashboard.server, "run", failing)
    with caplog.at_level(logging.ERROR, logger="fm.web.server"):
        dashboard.start()
        dashboard.stop()
    assert dashboard.thread is not None and not dashboard.thread.is_alive()
    assert "the server at http://127.0.0.1:8711/ stopped with an error" in caplog.text
    dashboard.sock.close()


def test_stopping_a_dashboard_that_never_served_releases_its_socket() -> None:
    dashboard = _unserved()
    assert dashboard.url == "http://127.0.0.1:8711/"
    dashboard.stop()
    assert dashboard.server.should_exit is True and dashboard.sock.fileno() == -1
