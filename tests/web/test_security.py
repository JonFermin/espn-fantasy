"""The dashboard's guard (ROADMAP #54, PLUG_AND_PLAY section 7): what a request needs before it reaches a route.

- A ``Host`` other than ``127.0.0.1:<port>`` / ``localhost:<port>`` is refused with 400 on every path, so a page
  that rebinds its own name to 127.0.0.1 reads nothing.
- Every POST needs the per-process token as ``X-FM-Token`` (403 without it, and nothing changes) and, when an
  ``Origin`` is sent, the dashboard's own origin.
- No response carries a CORS header; every response carries the security headers.
- The shipped page is static and offline: no inline script, no external URL, and the script never parses data as HTML.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any, Protocol

import pytest
from starlette.testclient import TestClient

import fm.web.app as web_app
from fm.config import load_config
from fm.proposals import AddDropPayload, pause_state, propose
from fm.store import Store

DOWDLE = 4038815
STATIC = resources.files("fm.web") / "static"


class Answer(Protocol):
    """What the ``TestClient`` returns (an ``httpx2.Response``, since starlette 1.x moved its test client to httpx2),
    narrowed to what these tests read."""

    @property
    def status_code(self) -> int: ...

    @property
    def headers(self) -> Mapping[str, str]: ...

    @property
    def text(self) -> str: ...

    def json(self, **kwargs: Any) -> Any: ...


@pytest.fixture
def proposal(home: Path) -> int:
    with Store.open(home / "state.db") as store:
        row = store.leagues.by_key("nfl")
        assert row is not None
        made = propose(store, load_config(), row, "add_drop", AddDropPayload(add_espn_id=DOWDLE), created_by="test")
        assert made.row_id is not None
        return made.row_id


def status_of(home: Path, proposal_id: int) -> str:
    with Store.open(home / "state.db") as store:
        row = store.proposals.get(proposal_id)
        assert row is not None
        return row.status


def error(response: Answer, status: int) -> str:
    assert response.status_code == status, response.text
    assert response.headers["content-type"].startswith("application/json")
    return str(response.json()["error"])


# --- Host -------------------------------------------------------------------------------------------------------------

FOREIGN_HOSTS = (
    "evil.example",
    "evil.example:8711",
    "127.0.0.1",
    "127.0.0.1:9999",
    "localhost",
    "[::1]:8711",
    "0.0.0.0:8711",
    "127.0.0.1.evil.example:8711",
    "",
)


@pytest.mark.parametrize("host", FOREIGN_HOSTS)
@pytest.mark.parametrize("path", ["/", "/static/app.js", "/api/overview", "/api/settings", "/api/nothing"])
def test_a_foreign_host_is_refused_on_every_path(client: TestClient, host: str, path: str) -> None:
    assert "answers requests for" in error(client.get(path, headers={"Host": host}), 400)


@pytest.mark.parametrize("host", FOREIGN_HOSTS)
def test_a_foreign_host_cannot_decide_even_with_the_token(
    client: TestClient, home: Path, token: str, proposal: int, host: str
) -> None:
    response = client.post(f"/api/proposals/{proposal}/approve", headers={"Host": host, "X-FM-Token": token})
    error(response, 400)
    assert status_of(home, proposal) == "proposed"


@pytest.mark.parametrize("host", ["127.0.0.1:8711", "localhost:8711", "LOCALHOST:8711"])
def test_the_dashboards_own_host_names_are_answered(client: TestClient, host: str) -> None:
    assert client.get("/api/overview", headers={"Host": host}).status_code == 200


# --- the token and the Origin -----------------------------------------------------------------------------------------

POSTS: tuple[tuple[str, dict[str, Any] | None], ...] = (
    ("/api/proposals/{id}/approve", None),
    ("/api/proposals/{id}/reject", None),
    ("/api/pause", {"reason": "test"}),
    ("/api/resume", None),
    ("/api/settings", {"league": "nfl", "preset": "balanced"}),
)


@pytest.mark.parametrize(("path", "body"), POSTS)
@pytest.mark.parametrize("supplied", [None, "", "wrong-token", "token-in-the-query-string"])
def test_a_post_without_the_token_is_403_and_changes_nothing(
    client: TestClient,
    home: Path,
    token: str,
    proposal: int,
    path: str,
    body: dict[str, Any] | None,
    supplied: str | None,
) -> None:
    config = (home / "config.toml").read_bytes()
    headers = {} if supplied is None else {"X-FM-Token": supplied}
    url = path.format(id=proposal)
    if supplied == "token-in-the-query-string":
        headers, url = {}, f"{url}?token={token}"
    assert web_app.TOKEN_HEADER in error(client.post(url, headers=headers, json=body), 403)
    assert status_of(home, proposal) == "proposed"
    assert pause_state() is None
    assert (home / "config.toml").read_bytes() == config


@pytest.mark.parametrize(
    "origin",
    ["http://evil.example", "http://127.0.0.1:9999", "https://127.0.0.1:8711", "null", "http://localhost:8711"],
)
def test_a_post_from_another_origin_is_403_even_with_the_token(
    client: TestClient, home: Path, token: str, proposal: int, origin: str
) -> None:
    headers = {"X-FM-Token": token, "Origin": origin}  # the Host is 127.0.0.1:8711, so localhost is another origin
    assert "cross-origin" in error(client.post(f"/api/proposals/{proposal}/approve", headers=headers), 403)
    assert status_of(home, proposal) == "proposed"


def test_a_same_origin_post_with_the_token_goes_through(
    client: TestClient, home: Path, token: str, proposal: int
) -> None:
    headers = {"X-FM-Token": token, "Origin": "http://127.0.0.1:8711"}
    assert client.post(f"/api/proposals/{proposal}/approve", headers=headers).status_code == 200
    assert status_of(home, proposal) == "approved"


def test_each_app_draws_its_own_token(home: Path) -> None:
    first, second = web_app.create_app(port=8711), web_app.create_app(port=8711)
    assert first.state.token != second.state.token and len(first.state.token) >= 40
    assert web_app.create_app(port=8711, token="fixed").state.token == "fixed"


# --- headers ----------------------------------------------------------------------------------------------------------


def test_no_response_carries_cors_headers_and_every_one_the_security_headers(client: TestClient, token: str) -> None:
    origin = {"Origin": "http://evil.example"}
    responses = [
        client.get("/", headers=origin),
        client.get("/api/overview", headers=origin),
        client.get("/api/overview", headers={"Host": "evil.example"}),
        client.post("/api/pause", headers=origin),
        client.options("/api/pause", headers={**origin, "Access-Control-Request-Method": "POST"}),
        client.get("/api/missing"),
    ]
    for response in responses:
        assert not [name for name in response.headers if name.lower().startswith("access-control-")]
        for name, value in web_app.SECURITY_HEADERS.items():
            assert response.headers[name] == value
    assert "frame-ancestors 'none'" in web_app.CONTENT_SECURITY_POLICY
    assert "script-src 'self'" in web_app.CONTENT_SECURITY_POLICY
    assert responses[4].status_code == 403  # a preflight needs the token like any other non-GET: no CORS, ever


# --- the shipped page -------------------------------------------------------------------------------------------------


def test_the_page_is_static_offline_and_never_parses_data_as_html() -> None:
    page = (STATIC / "index.html").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    style = (STATIC / "app.css").read_text(encoding="utf-8")
    assert page.count(web_app.TOKEN_PLACEHOLDER) == 1
    assert re.findall(r"<script(?![^>]*\bsrc=)[^>]*>", page) == []  # no inline script: the CSP would block it
    assert " style=" not in page and "<style" not in page
    for text in (page, script, style):
        assert not re.search(r"""(https?:)?//[a-z0-9.-]+\.[a-z]{2,}""", text.replace("http://<", "")), "no CDN"
        assert "@import" not in text
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert sink not in script, sink
    assert "X-FM-Token" in script and 'meta[name="fm-token"]' in script
