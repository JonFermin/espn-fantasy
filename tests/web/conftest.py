"""Shared setup for the dashboard tests: a private copy of ``tests/fixtures/home`` and a Starlette ``TestClient`` on it.

``FM_CONFIG_DIR`` and ``FM_CACHE_DIR`` point into the copy, as in ``tests/test_views.py``, so every request reads and
writes only there. The client talks to the app in process (no socket) and sends ``Host: 127.0.0.1:<PORT>``, the
``Host`` a browser sends to the dashboard served on that port.
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


@pytest.fixture
def client(app: Starlette) -> Iterator[TestClient]:
    with TestClient(app, base_url=BASE_URL) as test_client:
        yield test_client


@pytest.fixture
def port() -> int:
    """The port the app under test believes it is served on."""
    return PORT


@pytest.fixture
def token(app: Starlette) -> str:
    """The app's per-process token, which the page carries in ``<meta name="fm-token">``."""
    return str(app.state.token)
