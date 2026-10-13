"""The dashboard's guard (ROADMAP #54, PLUG_AND_PLAY section 7): what a request needs before it reaches a route.

- A ``Host`` other than ``127.0.0.1:<port>`` / ``localhost:<port>`` is refused with 400 on every path, so a page
  that rebinds its own name to 127.0.0.1 reads nothing.
- Only a signed-in browser gets in (review fix: 127.0.0.1 is open to every local account). A one-time login link
  (``/auth?t=<secret>``, single use, 10 minutes, kept in the owner-only ``dashboard.url``) sets an HttpOnly,
  ``SameSite=Strict`` session cookie; every ``/api/*`` call without it is a 401 and ``/`` says to run ``fm ui``.
- Every POST needs its session's CSRF token (from ``GET /api/session``) as ``X-FM-Token`` (403 without it, and
  nothing changes) and, when an ``Origin`` is sent, the dashboard's own origin.
- No response carries a CORS header; every response carries the security headers.
- The shipped pages are static and offline: nothing secret in the HTML, no inline script, no external URL, and the
  script never parses data as HTML.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from importlib import resources
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import parse_qs, urlsplit

import pytest
from starlette.applications import Starlette
from starlette.testclient import TestClient

import fm.web.app as web_app
from fm.config import load_config
from fm.proposals import AddDropPayload, pause_state, propose
from fm.store import Store

DOWDLE = 4038815
STATIC = resources.files("fm.web") / "static"
BASE_URL = "http://127.0.0.1:8711"  # the conftest's app's port


def sign_in(browser: TestClient, app: Starlette) -> None:
    """Follow the app's current one-time login link, as the browser ``fm ui`` opens does (``conftest.sign_in``)."""
    assert browser.get(app.state.login.url(), follow_redirects=False).status_code == 303


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


def test_each_session_has_its_own_csrf_token_and_each_app_its_own_login_secret(
    app: Starlette, client: TestClient, token: str, home: Path, proposal: int
) -> None:
    first, second = web_app.create_app(port=8711), web_app.create_app(port=8711)
    assert first.state.login.url() != second.state.login.url()
    with TestClient(app, base_url=BASE_URL) as other:  # another browser, signed in with its own one-time link
        sign_in(other, app)
        other_token = other.get("/api/session").json()["csrf"]
        assert other_token != token and len(token) >= 40
        headers = {"X-FM-Token": other_token}  # this browser's cookie with the other session's token
        assert web_app.TOKEN_HEADER in error(client.post(f"/api/proposals/{proposal}/approve", headers=headers), 403)
    assert status_of(home, proposal) == "proposed"


# --- signing in -------------------------------------------------------------------------------------------------------


def secret_of(link: str) -> str:
    return parse_qs(urlsplit(link).query)["t"][0]


API_PATHS = ("/api/session", "/api/overview", "/api/proposals", "/api/settings", "/api/health", "/api/nothing")


@pytest.mark.parametrize("path", API_PATHS)
def test_without_a_session_every_api_call_is_401(anonymous: TestClient, path: str) -> None:
    """Review fix: the dashboard answered every local account. Now /api/* needs the session cookie fm ui's link sets."""
    assert "run fm ui" in error(anonymous.get(path), 401)


@pytest.mark.parametrize(("path", "body"), POSTS)
def test_without_a_session_no_post_goes_through_even_with_a_token(
    anonymous: TestClient, home: Path, proposal: int, path: str, body: dict[str, Any] | None
) -> None:
    config = (home / "config.toml").read_bytes()
    response = anonymous.post(path.format(id=proposal), headers={"X-FM-Token": "guessed"}, json=body)
    assert "not signed in" in error(response, 401)
    assert status_of(home, proposal) == "proposed" and pause_state() is None
    assert (home / "config.toml").read_bytes() == config


def test_the_page_without_a_session_says_to_run_fm_ui_and_carries_nothing(anonymous: TestClient) -> None:
    page = anonymous.get("/")
    assert page.status_code == 401 and page.headers["content-type"].startswith("text/html")
    assert "fm ui" in page.text and '<section id="overview"' not in page.text
    for name, value in web_app.SECURITY_HEADERS.items():
        assert page.headers[name] == value
    assert anonymous.get("/static/app.css").status_code == 200  # the sign-in page's style is public, like the script


def test_a_login_link_signs_in_once_and_sets_a_strict_httponly_session_cookie(
    app: Starlette, anonymous: TestClient
) -> None:
    link = app.state.login.url()
    response = anonymous.get(link, follow_redirects=False)
    assert response.status_code == 303 and response.headers["location"] == "/"
    cookie = response.headers["set-cookie"]
    assert cookie.startswith("fm_session_8711=")
    attributes = {part.strip().lower() for part in cookie.split(";")[1:]}
    assert {"httponly", "samesite=strict", "path=/"} <= attributes and "secure" not in attributes
    assert "max-age" not in cookie.lower() and "expires" not in cookie.lower()  # gone when the browser closes
    assert anonymous.get("/").status_code == 200 and anonymous.get("/api/session").status_code == 200
    assert app.state.login.url() != link  # spent: a fresh link replaces it at once

    with TestClient(app, base_url=BASE_URL) as stranger:  # the same link again, from another browser
        again = stranger.get(link, follow_redirects=False)
        assert again.status_code == 403 and "fm ui" in again.text and "set-cookie" not in again.headers
        assert "run fm ui" in error(stranger.get("/api/overview"), 401)
    reloaded = anonymous.get(link, follow_redirects=False)  # the signed-in browser reloading its own old link
    assert reloaded.status_code == 303 and "set-cookie" not in reloaded.headers


@pytest.mark.parametrize("offered", ["", "wrong", "a" * 43])
def test_a_wrong_login_secret_signs_nobody_in_and_keeps_the_real_one(
    app: Starlette, anonymous: TestClient, offered: str
) -> None:
    link = app.state.login.url()
    response = anonymous.get(f"/auth?t={offered}", follow_redirects=False)
    assert response.status_code == 403 and "set-cookie" not in response.headers
    assert app.state.login.url() == link  # a guess does not burn the link the owner is about to open
    assert anonymous.get(link, follow_redirects=False).status_code == 303


def test_a_login_link_expires_after_ten_minutes(home: Path) -> None:
    now = [1000.0]
    app = web_app.create_app(port=8711, clock=lambda: now[0])
    link = app.state.login.url()
    now[0] += 10 * 60
    with TestClient(app, base_url=BASE_URL) as browser:
        assert browser.get(link, follow_redirects=False).status_code == 403
        fresh = app.state.login.url()
        assert fresh != link and browser.get(fresh, follow_redirects=False).status_code == 303


def test_the_server_keeps_its_current_link_in_the_url_file_and_rotates_it(home: Path, tmp_path: Path) -> None:
    url_file = tmp_path / "config" / "dashboard.url"
    now = [1000.0]
    app = web_app.create_app(port=8711, url_file=url_file, clock=lambda: now[0])
    first = url_file.read_text(encoding="utf-8").strip()
    assert first == app.state.login.url() and first.startswith("http://127.0.0.1:8711/auth?t=")
    with TestClient(app, base_url=BASE_URL) as browser:
        browser.get(first, follow_redirects=False)
        second = url_file.read_text(encoding="utf-8").strip()
        assert second != first and second == app.state.login.url()  # rotated after the use
        ping = browser.get("/auth/ping")
        assert ping.status_code == 200 and ping.json() == {"app": "espn-fantasy"}
        assert url_file.read_text(encoding="utf-8").strip() == second  # plenty of time left: unchanged
        now[0] += 6 * 60
        assert browser.get("/auth/ping").json() == {"app": "espn-fantasy"}
        third = url_file.read_text(encoding="utf-8").strip()
        assert third != second and secret_of(third) not in ping.text  # fresh again, and only through the file
    app.state.login.close()
    assert not url_file.exists()


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
    signin = (STATIC / "signin.html").read_text(encoding="utf-8")
    script = (STATIC / "app.js").read_text(encoding="utf-8")
    style = (STATIC / "app.css").read_text(encoding="utf-8")
    assert "fm-token" not in page and "__FM_TOKEN__" not in page  # nothing secret is ever put into the HTML
    for html in (page, signin):
        assert re.findall(r"<script(?![^>]*\bsrc=)[^>]*>", html) == []  # no inline script: the CSP would block it
        assert " style=" not in html and "<style" not in html
    assert "<script" not in signin  # the signed-out page is text only
    for text in (page, signin, script, style):
        assert not re.search(r"""(https?:)?//[a-z0-9.-]+\.[a-z]{2,}""", text.replace("http://<", "")), "no CDN"
        assert "@import" not in text
    for sink in ("innerHTML", "outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert sink not in script, sink
    assert "X-FM-Token" in script and '"/api/session"' in script and "document.cookie" not in script
