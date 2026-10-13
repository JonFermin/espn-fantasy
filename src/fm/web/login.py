"""Signing in to the dashboard (PLUG_AND_PLAY section 7): one-time login links, the URL file, and finding a dashboard
that is already served. Standard library only, so ``fm ui`` can look for a running dashboard without loading Starlette.

Why a sign-in at all: the dashboard binds 127.0.0.1, which every account on the machine (and every process) can reach.
Only the person who can read this user's config dir may use it:

- The server keeps one unused **login secret** at a time (:class:`LoginLinks`) and writes its link,
  ``http://127.0.0.1:<port>/auth?t=<secret>``, to ``<config dir>/dashboard.url`` (:func:`url_file_path`). The file is
  written through a temp file that ``tempfile.mkstemp`` creates owner-only (mode 0600) where the OS has POSIX modes; on
  Windows the config dir sits in the user's profile, whose ACL already keeps other accounts out (``~/.config`` under
  ``%USERPROFILE%``, never ``%APPDATA%``).
- A secret is good for one sign-in and :data:`LOGIN_TTL` (10 minutes). Using it rotates it: a new one replaces it in
  the file at once, so the file always holds a link that works. The server exchanges a used link for a session cookie
  (``fm.web.app``).
- ``fm ui`` asks a running dashboard for a fresh link through the file (:func:`find_running`): it requests
  :data:`PING_PATH`, which makes the server rotate the secret when less than :data:`FRESH_FOR` of it is left, then
  reads the file. Nothing secret crosses the socket; the secret only travels through the owner-only file and the
  browser's address bar.

Usage::

    from fm.web.login import find_running
    url = find_running()  # the login link of the dashboard fm run serves, or None when none answers
"""

from __future__ import annotations

import contextlib
import http.client
import json
import os
import secrets
import tempfile
import threading
import time
from collections.abc import Callable
from datetime import timedelta
from pathlib import Path
from typing import Final
from urllib.parse import parse_qs, urlsplit

from fm import paths
from fm.web import LOOPBACK

URL_FILENAME: Final = "dashboard.url"
"""The file under the config dir that holds the running dashboard's one-time login link."""
LOGIN_PATH: Final = "/auth"
"""``GET /auth?t=<secret>`` exchanges a login secret for a session cookie."""
LOGIN_PARAM: Final = "t"
"""The query parameter :data:`LOGIN_PATH` reads the secret from."""
PING_PATH: Final = "/auth/ping"
"""``GET /auth/ping``: answers :data:`APP_ID` and makes sure the link in the URL file has :data:`FRESH_FOR` left."""
APP_ID: Final = "espn-fantasy"
"""What :data:`PING_PATH` answers as ``{"app": ...}``, so a probe knows the port is a dashboard, not something else."""
LOGIN_TTL: Final = timedelta(minutes=10)
"""How long a login secret is good for, unused."""
FRESH_FOR: Final = timedelta(minutes=5)
"""A ping rotates the secret when less than this is left, so the link ``fm ui`` opens has at least this long."""
PROBE_TIMEOUT_S: Final = 2.0
"""How long ``fm ui`` waits for a dashboard on the URL file's port to answer."""


def url_file_path() -> Path:
    """``<config dir>/dashboard.url``."""
    return paths.config_dir() / URL_FILENAME


def login_url(port: int, secret: str) -> str:
    """``http://127.0.0.1:<port>/auth?t=<secret>``."""
    return f"http://{LOOPBACK}:{port}{LOGIN_PATH}?{LOGIN_PARAM}={secret}"


class LoginLinks:
    """The server's one unused login secret, written to ``url_file`` (none when ``None``, as in tests) as a link.

    Thread-safe: the dashboard serves from its own thread while ``fm run`` or ``fm ui`` reads :meth:`url`. ``clock``
    is monotonic seconds (tests move it)."""

    def __init__(
        self,
        port: int,
        *,
        url_file: Path | None = None,
        ttl: timedelta = LOGIN_TTL,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.port = port
        self.url_file = url_file
        self.ttl_s = ttl.total_seconds()
        self.clock = clock
        self._lock = threading.Lock()
        self._secret = ""
        self._expires = 0.0
        self._written: str | None = None
        with self._lock:
            self._mint()

    def url(self) -> str:
        """The current login link (a new one when the last expired unused)."""
        with self._lock:
            if self.clock() >= self._expires:
                self._mint()
            return login_url(self.port, self._secret)

    def refresh(self, min_left: timedelta = FRESH_FOR) -> None:
        """Rotate the secret unless it has at least ``min_left`` to go: what a ping asks for before ``fm ui`` reads the
        URL file."""
        with self._lock:
            if self._expires - self.clock() < min_left.total_seconds():
                self._mint()

    def consume(self, offered: str) -> bool:
        """Spend ``offered`` if it is the current, unexpired secret: true once, and the file gets a new link at once."""
        with self._lock:
            valid = self.clock() < self._expires and secrets.compare_digest(
                offered.encode("utf-8", errors="replace"), self._secret.encode("ascii")
            )
            if valid or self.clock() >= self._expires:
                self._mint()
            return valid

    def close(self) -> None:
        """Remove the URL file if it still holds this server's link (another dashboard may have replaced it)."""
        with self._lock:
            written, self._written = self._written, None
            if self.url_file is None or written is None or read_url_file(self.url_file) != written:
                return
            with contextlib.suppress(OSError):
                self.url_file.unlink()

    def _mint(self) -> None:
        self._secret = secrets.token_urlsafe(32)
        self._expires = self.clock() + self.ttl_s
        if self.url_file is not None:
            link = login_url(self.port, self._secret)
            write_url_file(self.url_file, link)
            self._written = link


def write_url_file(path: Path, url: str) -> None:
    """Replace ``path`` with ``url`` atomically, through a temp file ``mkstemp`` creates owner-only (0600 on POSIX; on
    Windows the file takes the config dir's ACL, the user profile's)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temp = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(url + "\n")
        os.replace(temp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            temp.unlink(missing_ok=True)
        raise


def read_url_file(path: Path | None = None) -> str | None:
    """The link in the URL file (:func:`url_file_path` by default), ``None`` when there is none to read."""
    target = path if path is not None else url_file_path()
    try:
        text = target.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return text or None


def port_of(url: str) -> int | None:
    """The port of a login link that points at this machine's dashboard, ``None`` for anything else."""
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    if parts.scheme != "http" or parts.hostname != LOOPBACK or parts.path != LOGIN_PATH or port is None:
        return None
    return port if parse_qs(parts.query).get(LOGIN_PARAM) else None


def ping(port: int, *, timeout: float = PROBE_TIMEOUT_S) -> bool:
    """Whether a dashboard answers on ``127.0.0.1:<port>`` (``GET /auth/ping`` says :data:`APP_ID`). The request also
    makes that dashboard refresh the link in its URL file."""
    connection = http.client.HTTPConnection(LOOPBACK, port, timeout=timeout)
    try:
        connection.request("GET", PING_PATH, headers={"Host": f"{LOOPBACK}:{port}", "Accept": "application/json"})
        response = connection.getresponse()
        body = response.read(4096)
    except (OSError, http.client.HTTPException):
        return False
    finally:
        connection.close()
    if response.status != 200:
        return False
    try:
        answer = json.loads(body)
    except ValueError:
        return False
    return isinstance(answer, dict) and answer.get("app") == APP_ID


def find_running(path: Path | None = None, *, probe: Callable[[int], bool] | None = None) -> str | None:
    """The login link of a dashboard already served for this config dir: the URL file names its port, the port answers
    the ping (``probe``, :func:`ping` by default; which refreshes the link), and the file is read again. ``None`` when
    there is no file, the file is not a dashboard link, or nothing answers (a dashboard that stopped without cleaning
    up)."""
    target = path if path is not None else url_file_path()
    url = read_url_file(target)
    port = port_of(url) if url is not None else None
    if port is None or not (probe if probe is not None else ping)(port):
        return None
    fresh = read_url_file(target)
    return fresh if fresh is not None and port_of(fresh) == port else None


__all__ = [
    "APP_ID",
    "FRESH_FOR",
    "LOGIN_PARAM",
    "LOGIN_PATH",
    "LOGIN_TTL",
    "PING_PATH",
    "PROBE_TIMEOUT_S",
    "URL_FILENAME",
    "LoginLinks",
    "find_running",
    "login_url",
    "ping",
    "port_of",
    "read_url_file",
    "url_file_path",
    "write_url_file",
]
