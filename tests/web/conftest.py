"""Shared setup for the dashboard tests: a private copy of ``tests/fixtures/home`` and a Starlette ``TestClient`` on it.

``FM_CONFIG_DIR`` and ``FM_CACHE_DIR`` point into the copy, as in ``tests/test_views.py``, so every request reads and
writes only there. The client talks to the app in process (no socket) and sends ``Host: 127.0.0.1:<PORT>``, the
``Host`` a browser sends to the dashboard served on that port.

``client`` is signed in the way ``fm ui`` signs a browser in: it follows the app's one-time login link
(``app.state.login.url()``) and keeps the session cookie. ``anonymous`` is a browser that never did. ``token`` is the
signed-in session's CSRF token, from ``GET /api/session`` as the page reads it.
"""

from __future__ import annotations

import shutil
from collections.abc import Iterator
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

from fm import paths
from fm.web.app import create_app

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
HOME = FIXTURES / "home"
PORT = 8711
BASE_URL = f"http://127.0.0.1:{PORT}"


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A private copy of the fixture home, with FM_CONFIG_DIR and FM_CACHE_DIR pointing into it."""
    copy = tmp_path / "home"
    shutil.copytree(HOME, copy, ignore=shutil.ignore_patterns("snapshots", "build.py", "README.md", "__pycache__"))
    monkeypatch.setenv(paths.CONFIG_DIR_ENV, str(copy))
    monkeypatch.setenv(paths.CACHE_DIR_ENV, str(copy / "cache"))
    return copy


@pytest.fixture
def app(home: Path) -> Starlette:
    return create_app(port=PORT)


def sign_in(client: TestClient, app: Starlette) -> None:
    """Follow the app's current one-time login link, as the browser ``fm ui`` opens does: a 303 to ``/`` with the
    session cookie, which the client keeps."""
    response = client.get(app.state.login.url(), follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/", response.text


@pytest.fixture
def anonymous(app: Starlette) -> Iterator[TestClient]:
    """A browser that never signed in."""
    with TestClient(app, base_url=BASE_URL) as test_client:
        yield test_client


@pytest.fixture
def client(app: Starlette) -> Iterator[TestClient]:
    """A signed-in browser."""
    with TestClient(app, base_url=BASE_URL) as test_client:
        sign_in(test_client, app)
        yield test_client


@pytest.fixture
def port() -> int:
    """The port the app under test believes it is served on."""
    return PORT


@pytest.fixture
def token(client: TestClient) -> str:
    """The signed-in session's CSRF token, which the page reads from ``GET /api/session`` and sends as
    ``X-FM-Token``."""
    response = client.get("/api/session")
    assert response.status_code == 200, response.text
    return str(response.json()["csrf"])
