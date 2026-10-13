"""The dashboard's sign-in plumbing (:mod:`fm.web.login`, review fix for "the dashboard answers every local account").

One unused login secret at a time, good once and for ten minutes, its link kept in an owner-only ``dashboard.url``;
``fm ui`` finds a running dashboard through that file and a ping of its port. The ping is checked against a fake
``http.client`` connection: no socket is opened here.
"""

from __future__ import annotations

import http.client
import json
import os
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

import fm.web.login as login
from fm import paths
from fm.web.login import LoginLinks, find_running, port_of, read_url_file, url_file_path, write_url_file


class Clock:
    def __init__(self) -> None:
        self.now = 5000.0

    def __call__(self) -> float:
        return self.now


def secret(link: str) -> str:
    return link.rsplit("t=", 1)[1]


def test_the_url_file_lives_in_the_config_dir_and_is_owner_only(tmp_path: Path) -> None:
    assert url_file_path() == paths.config_dir() / "dashboard.url"
    target = tmp_path / "config" / "dashboard.url"
    write_url_file(target, "http://127.0.0.1:8711/auth?t=abc")
    assert read_url_file(target) == "http://127.0.0.1:8711/auth?t=abc"
    assert [path.name for path in target.parent.iterdir()] == ["dashboard.url"]  # no temp file left behind
    if sys.platform != "win32":  # on Windows the profile's ACL guards the config dir instead of a file mode
        assert stat.S_IMODE(os.stat(target).st_mode) & 0o077 == 0
    assert read_url_file(tmp_path / "missing.url") is None


def test_a_secret_works_once_and_the_file_always_holds_the_one_that_works(tmp_path: Path) -> None:
    clock = Clock()
    target = tmp_path / "dashboard.url"
    links = LoginLinks(8711, url_file=target, clock=clock)
    first = links.url()
    assert read_url_file(target) == first and first.startswith("http://127.0.0.1:8711/auth?t=")
    assert links.url() == first  # stable until used or expired
    assert not links.consume("not-it") and read_url_file(target) == first  # a guess burns nothing
    assert links.consume(secret(first))
    second = read_url_file(target)
    assert second is not None and second != first and second == links.url()
    assert not links.consume(secret(first))  # once only


def test_a_secret_expires_after_ten_minutes_and_a_ping_keeps_five_minutes_on_it(tmp_path: Path) -> None:
    clock = Clock()
    links = LoginLinks(8711, url_file=tmp_path / "dashboard.url", clock=clock)
    first = links.url()
    clock.now += 4 * 60
    links.refresh()
    assert links.url() == first  # six minutes left: kept
    clock.now += 2 * 60
    links.refresh()
    second = links.url()
    assert second != first  # four minutes left: rotated for the link fm ui is about to open
    clock.now += 10 * 60
    assert not links.consume(secret(second)) and links.url() != second  # expired unused, replaced


def test_a_link_without_a_file_writes_nothing(tmp_path: Path) -> None:
    links = LoginLinks(8711)
    assert links.consume(secret(links.url()))
    links.close()
    assert not url_file_path().exists()


@pytest.mark.parametrize(
    ("url", "port"),
    [
        ("http://127.0.0.1:8711/auth?t=abc", 8711),
        ("http://127.0.0.1:9000/auth?t=abc", 9000),
        ("http://localhost:8711/auth?t=abc", None),
        ("http://evil.example:8711/auth?t=abc", None),
        ("https://127.0.0.1:8711/auth?t=abc", None),
        ("http://127.0.0.1:8711/?t=abc", None),
        ("http://127.0.0.1:8711/auth", None),
        ("http://127.0.0.1/auth?t=abc", None),
        ("http://127.0.0.1:99999/auth?t=abc", None),
        ("not a url", None),
    ],
)
def test_only_a_loopback_login_link_names_a_port(url: str, port: int | None) -> None:
    assert port_of(url) == port


def test_find_running_needs_the_file_a_dashboard_link_and_an_answer(tmp_path: Path) -> None:
    target = tmp_path / "dashboard.url"
    asked: list[int] = []

    def answers(port: int) -> bool:
        asked.append(port)
        write_url_file(target, f"http://127.0.0.1:{port}/auth?t=refreshed")  # what the server's ping handler does
        return True

    assert find_running(target, probe=answers) is None and asked == []  # no file: nothing probed
    write_url_file(target, "http://evil.example:80/auth?t=x")
    assert find_running(target, probe=answers) is None and asked == []  # not a dashboard link
    write_url_file(target, "http://127.0.0.1:8711/auth?t=stale")
    assert find_running(target, probe=lambda port: False) is None  # nothing answers: a stopped dashboard's file
    assert find_running(target, probe=answers) == "http://127.0.0.1:8711/auth?t=refreshed" and asked == [8711]


class FakeConnection:
    """``http.client.HTTPConnection`` answering one canned response (or raising), recording the request."""

    answer: tuple[int, bytes] | Exception = (200, b'{"app": "espn-fantasy"}')
    requests: list[tuple[str, int, str, dict[str, str]]] = []

    def __init__(self, host: str, port: int, *, timeout: float) -> None:
        self.host, self.port = host, port

    def request(self, method: str, path: str, headers: dict[str, str]) -> None:
        FakeConnection.requests.append((self.host, self.port, f"{method} {path}", headers))
        if isinstance(self.answer, Exception):
            raise self.answer

    def getresponse(self) -> FakeResponse:
        assert not isinstance(self.answer, Exception)
        return FakeResponse(*self.answer)

    def close(self) -> None:
        pass


@dataclass(frozen=True)
class FakeResponse:
    status: int
    body: bytes

    def read(self, amount: int) -> bytes:
        return self.body[:amount]


@pytest.mark.parametrize(
    ("answer", "expected"),
    [
        ((200, json.dumps({"app": "espn-fantasy"}).encode()), True),
        ((200, b'{"app": "something-else"}'), False),
        ((200, b"<html>not json</html>"), False),
        ((404, b'{"app": "espn-fantasy"}'), False),
        (ConnectionRefusedError(10061, "refused"), False),
        (http.client.RemoteDisconnected("closed"), False),
    ],
)
def test_ping_knows_a_dashboard_from_anything_else_on_the_port(
    monkeypatch: pytest.MonkeyPatch, answer: tuple[int, bytes] | Exception, expected: bool
) -> None:
    monkeypatch.setattr(FakeConnection, "answer", answer)
    monkeypatch.setattr(FakeConnection, "requests", [])
    monkeypatch.setattr(login.http.client, "HTTPConnection", FakeConnection)
    assert login.ping(8711) is expected
    ((host, port, line, headers),) = FakeConnection.requests
    assert (host, port, line) == ("127.0.0.1", 8711, "GET /auth/ping") and headers["Host"] == "127.0.0.1:8711"
