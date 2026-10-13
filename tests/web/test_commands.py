"""``fm ui`` and ``fm run --ui/--no-ui`` (ROADMAP #54), with the server replaced by a fake: no socket is bound here.

``fm ui`` first looks for a dashboard already served (``fm run --ui``) through ``<config dir>/dashboard.url`` and a ping
of its port (faked here): when one answers it opens that server's fresh one-time link and serves nothing. Otherwise it
binds, prints the URL, opens its own sign-in link unless ``--no-open`` (which prints the link), serves until Ctrl+C,
and exits 1 with a hint when the port is taken. ``fm run`` starts the dashboard in a thread before its loop and stops
it after; a port that is taken, or a dashboard that fails to start for any other reason, is logged and printed and the
loop runs anyway.
:class:`fm.web.server.Dashboard` itself is checked for the two behaviours ``fm run`` relies on, without serving: a
failure in its thread is logged rather than raised, and stopping a dashboard that never served releases its socket.
"""

from __future__ import annotations

import errno
import logging
import socket
from dataclasses import dataclass, field
from pathlib import Path

import pytest
import typer
import uvicorn
from typer.testing import CliRunner

import fm.web.login as web_login
import fm.web.server as web_server
from fm import paths
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

    def login_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/auth?t=one-time-secret-{self.port}"

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

    def bind(port: int = DEFAULT_PORT, *, url_file: Path | None = None) -> FakeDashboard:
        log.append(f"bind {port}")
        return FakeDashboard(port, log)

    def ping(port: int, *, timeout: float = 2.0) -> bool:
        log.append(f"ping {port}")
        return False  # nothing answers unless a test says so

    monkeypatch.setattr(web_server, "bind_dashboard", bind)
    monkeypatch.setattr(web_login, "ping", ping)
    monkeypatch.setattr(ui_cmd.webbrowser, "open", lambda url, *args, **kwargs: log.append(f"open {url}") or True)
    return log


def taken(port: int = DEFAULT_PORT, *, url_file: Path | None = None) -> FakeDashboard:
    raise OSError(errno.EADDRINUSE, "Only one usage of each socket address is normally permitted")


# --- fm ui ------------------------------------------------------------------------------------------------------------


def test_fm_ui_serves_opens_the_signed_in_link_and_stops_on_ctrl_c(events: list[str]) -> None:
    result = runner.invoke(cli(), ["ui"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert events == ["bind 8711", "open http://127.0.0.1:8711/auth?t=one-time-secret-8711", "serve"]
    assert "dashboard at http://127.0.0.1:8711/ (Ctrl+C stops)" in result.output
    assert "one-time-secret" not in result.output  # opened, not printed
    assert result.output.rstrip().endswith("stopped")


def test_fm_ui_no_open_serves_on_the_port_given_and_prints_the_link(events: list[str]) -> None:
    result = runner.invoke(cli(), ["ui", "--no-open", "--port", "8799"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert events == ["bind 8799", "serve"]
    assert "sign in with this one-time link (it works once, for 10 minutes): http://127.0.0.1:8799/auth?t=" in (
        result.output
    )


def test_fm_ui_prints_the_link_when_no_browser_opens(events: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ui_cmd.webbrowser, "open", lambda url, *args, **kwargs: False)
    result = runner.invoke(cli(), ["ui"], catch_exceptions=False)
    assert "sign in with this one-time link" in result.output and events == ["bind 8711", "serve"]


def test_fm_ui_on_a_taken_port_exits_1_and_says_what_to_do(events: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(web_server, "bind_dashboard", taken)
    result = runner.invoke(cli(), ["ui", "--no-open"], catch_exceptions=False)
    assert result.exit_code == 1
    assert "error: cannot serve on 127.0.0.1:8711" in result.output and "--port" in result.output
    assert events == []


def test_fm_ui_opens_the_dashboard_fm_run_serves_instead_of_serving_another(
    events: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Review fix: with fm run --ui serving, fm ui asks it for a fresh link through dashboard.url (the ping makes the
    server rotate it) and opens that; it binds nothing."""
    url_file = paths.config_dir() / "dashboard.url"
    url_file.parent.mkdir(parents=True, exist_ok=True)
    url_file.write_text("http://127.0.0.1:8800/auth?t=old-link\n", encoding="utf-8")

    def answering(port: int, *, timeout: float = 2.0) -> bool:
        events.append(f"ping {port}")
        url_file.write_text("http://127.0.0.1:8800/auth?t=fresh-link\n", encoding="utf-8")  # the server's rotation
        return True

    monkeypatch.setattr(web_login, "ping", answering)
    result = runner.invoke(cli(), ["ui"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert events == ["ping 8800", "open http://127.0.0.1:8800/auth?t=fresh-link"]  # no bind, no serve
    assert "already served at http://127.0.0.1:8800/ (by fm run or another fm ui)" in result.output


def test_fm_ui_serves_its_own_when_the_url_files_dashboard_is_gone(events: list[str]) -> None:
    url_file = paths.config_dir() / "dashboard.url"
    url_file.parent.mkdir(parents=True, exist_ok=True)
    url_file.write_text("http://127.0.0.1:8800/auth?t=left-by-a-crashed-run\n", encoding="utf-8")
    result = runner.invoke(cli(), ["ui"], catch_exceptions=False)
    assert result.exit_code == 0, result.output
    assert events == ["ping 8800", "bind 8711", "open http://127.0.0.1:8711/auth?t=one-time-secret-8711", "serve"]


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
    def broken(port: int = DEFAULT_PORT, *, url_file: Path | None = None) -> FakeDashboard:
        raise RuntimeError("fm/web/static/index.html is missing")

    monkeypatch.setattr(web_server, "bind_dashboard", broken)
    result = runner.invoke(cli(), ["run"], catch_exceptions=False)
    assert result.exit_code == 0 and events == ["loop"]
    assert "fm run: dashboard not started; the loop runs without it" in result.output
    assert "index.html is missing" in run_log_path().read_text(encoding="utf-8")  # with its traceback


# --- the Dashboard object ---------------------------------------------------------------------------------------------


def _unserved() -> web_server.Dashboard:
    """A Dashboard around an unbound socket (no network): what ``bind_dashboard`` builds, minus the bind (and minus
    the URL file)."""
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


def test_a_dashboard_removes_its_url_file_when_it_stops_but_not_another_ones(tmp_path: Path) -> None:
    url_file = tmp_path / "dashboard.url"
    app = create_app(port=DEFAULT_PORT, url_file=url_file)
    server = uvicorn.Server(uvicorn.Config(app, log_config=None))
    dashboard = web_server.Dashboard(port=DEFAULT_PORT, app=app, sock=socket.socket(), server=server)
    assert url_file.read_text(encoding="utf-8").strip() == dashboard.login_url()
    dashboard.stop()
    assert not url_file.exists()

    other = create_app(port=DEFAULT_PORT, url_file=url_file)
    url_file.write_text("http://127.0.0.1:8799/auth?t=another-dashboards-link\n", encoding="utf-8")
    other.state.login.close()  # the file now names another dashboard: left alone
    assert url_file.read_text(encoding="utf-8").startswith("http://127.0.0.1:8799/")
