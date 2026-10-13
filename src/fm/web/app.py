"""The dashboard's Starlette app (PLUG_AND_PLAY section 7, ROADMAP #54): a JSON API and one static page.

:func:`create_app` builds it for the port it will be served on. Every endpoint reads through the same modules the CLI
and the MCP server use, per request, so the page shows what ``fm sync`` and the ticks stored since it opened:

=====================================================  ===============================================================
endpoint                                               serves
=====================================================  ===============================================================
``GET /``, ``GET /static/{app.js,app.css}``            the page once signed in (else a page saying to run ``fm ui``)
``GET /auth?t=<secret>``, ``GET /auth/ping``           sign in with a one-time link; ``fm ui``'s probe (below)
``GET /api/session``                                   this session's CSRF token, for the page's POSTs
``GET /api/overview[?as_of=]``                         per league the team, record, standing, sync age, the next
                                                       deadline of each kind and the open proposals; the session
                                                       check, the health counts and the kill switch
``GET /api/proposals[?all=1&league=]``                 :func:`fm.views.list_proposals`, each with its moves in player
                                                       names (:func:`fm.notify.messages.moves_text`)
``POST /api/proposals/{id}/approve``, ``.../reject``   :func:`fm.proposals.approve` / :func:`fm.proposals.reject` with
                                                       ``decided_by="dashboard"``
``POST /api/pause`` (``{"reason"}``), ``/api/resume``  the kill switch
``GET /api/lineup/{league}``, ``/api/waivers/{l}``     :func:`fm.views.lineup` / :func:`fm.views.waivers` (``as_of``,
                                                       ``top``); nothing is stored
``GET /api/reports``, ``GET /api/reports/{name}``      the ``fm report`` files and one file's markdown, as JSON
``GET /api/health``                                    :func:`fm.health.run_checks`, offline, never launching a browser
``GET /api/settings``, ``POST /api/settings``          presets and their overrides, ``trade_finder.enabled``, the
                                                       notify channel; edits go through :mod:`fm.config_edit`
=====================================================  ===============================================================

Errors come back as ``{"error": "..."}`` with a 4xx status: 400 for a bad argument, a ``config.toml`` that does not load
or an edit it refuses (:class:`fm.views.ViewError`, ``ConfigError``), 403 for a refused request, 404 for an unknown
league, proposal, report or path, 409 for a decision the proposal's status does not allow. Anything unexpected is
logged with its traceback under ``fm.web.app`` (``logs/run.log`` under ``fm run``) and answered with a 500.

Security
--------
The dashboard binds ``127.0.0.1`` only (:mod:`fm.web.server`), which every account and process on the machine can
reach, so it also needs a sign-in that only this user can complete. Every request passes :class:`_Guard` first:

- ``Host`` must be ``127.0.0.1:<port>`` or ``localhost:<port>``, else 400. A page on another site that rebinds its
  own name to 127.0.0.1 (DNS rebinding) still sends its own name as the ``Host``, so it can read nothing.
- **Sign-in.** The server keeps one unused login secret (:class:`fm.web.login.LoginLinks`) and writes its link to
  ``<config dir>/dashboard.url``, which only this user can read. ``GET /auth?t=<secret>`` spends it (once, within
  10 minutes; the file gets a new link at once) and answers with a session cookie, ``fm_session_<port>``: HttpOnly
  (no script reads it), ``SameSite=Strict`` (no other site's page sends it) and ``Path=/``, then redirects to ``/``.
  ``fm ui`` opens that link (:mod:`fm.commands.ui`). Every ``/api/*`` request, and every request other than GET and
  HEAD, needs a live session: 401 otherwise. ``/`` without one serves a static page that says to run ``fm ui``.
- **CSRF.** Every request other than GET and HEAD also needs ``X-FM-Token`` equal to its session's CSRF token and,
  when the browser sends an ``Origin``, an ``Origin`` equal to ``http://<the Host>``; anything else is a 403. The page
  gets the token from ``GET /api/session``, which itself needs the cookie, and no other origin can read that answer:
  no response carries a CORS header. A double-submit cookie would also work; a token delivered by an authenticated GET
  keeps every cookie HttpOnly and ties the token to the session server-side, with nothing put into the HTML.
- Responses carry ``nosniff``, ``no-store``, ``frame-ancestors 'none'`` and a Content-Security-Policy that allows
  this origin's script and style only. The page is static: no data is ever put into HTML on the server, and the
  script builds the DOM with ``textContent``.
- No secret leaves the machine's files: settings say only whether each ``.env`` variable is set.

No write path
-------------
Workers propose, the executor acts (CLAUDE.md). The writes here are the human decisions only: approve, reject, pause,
resume and ``config.toml`` edits. This package imports nothing from ``fm.executor`` or ``fm.browser`` (the read
views bring in ``fm.browser.session``, the cookie reader, through ``fm.espn.auth``, as they do for the MCP server), no
propose or execute function, and never launches the browser: :func:`fm.health.run_checks` keeps its default
``launch_browser=False``. Approving records the decision and ``fm run`` or the scheduled tick executes it.
"""

from __future__ import annotations

import functools
import json
import logging
import re
import secrets
import sqlite3
import threading
import time
from collections import OrderedDict
from collections.abc import Awaitable, Callable, Mapping, MutableMapping
from datetime import UTC, datetime, timedelta
from importlib import resources
from pathlib import Path
from typing import Any, Final, Literal, get_args

from starlette import routing
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers, MutableHeaders
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.requests import Request, cookie_parser
from starlette.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

import fm.views as views
from fm import paths
from fm.config import SECRET_ENV_VARS, Channel, Config, ConfigError, League, load_config
from fm.config_edit import preset_overrides, set_notify_channel, set_values
from fm.health import CheckStatus, has_failure, run_checks
from fm.jobs.deadlines import upcoming
from fm.jobs.report import REPORTS_DIRNAME
from fm.notify.messages import moves_text
from fm.presets import PRESET_DESCRIPTIONS, PRESET_NAMES
from fm.proposals import TRANSITIONS, PauseState, PolicyError, ProposalError, approve, kind_spec, pause, reject, resume
from fm.proposals.pause import pause_state
from fm.proposals.policy import stored_settings
from fm.store import LeagueRow, Store
from fm.web.login import APP_ID, LOGIN_PARAM, LOGIN_PATH, LOGIN_TTL, PING_PATH, LoginLinks

logger = logging.getLogger(__name__)

TOKEN_HEADER: Final = "X-FM-Token"
"""The request header every POST carries its session's CSRF token in (from ``GET /api/session``)."""
SESSION_COOKIE_PREFIX: Final = "fm_session_"
"""The session cookie is ``fm_session_<port>``: cookies are per host, not per port, so a dashboard on another port
keeps its own."""
MAX_SESSIONS: Final = 32
"""Signed-in browsers a dashboard remembers; the oldest session is forgotten beyond this."""
DECIDED_BY: Final = "dashboard"
"""``decided_by`` on a proposal approved or rejected here."""
SAFE_METHODS: Final = frozenset({"GET", "HEAD"})
"""Requests that change nothing and need no token. Every other method needs it."""
MAX_BODY_BYTES: Final = 16_384
"""The largest request body accepted; every body here is a small JSON object."""
MAX_REPORT_BYTES: Final = 2_000_000
"""The largest report served; ``fm report`` writes a few kilobytes."""
PROPOSALS_LIMIT: Final = 200
"""How many proposals ``GET /api/proposals`` lists, newest first (``count`` is the total)."""
DEADLINE_HORIZON: Final = timedelta(days=7)
"""How far ahead the overview looks for deadlines."""
CONTENT_SECURITY_POLICY: Final = (
    "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; "
    "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
)
SECURITY_HEADERS: Final[Mapping[str, str]] = {
    "content-security-policy": CONTENT_SECURITY_POLICY,
    "x-content-type-options": "nosniff",
    "x-frame-options": "DENY",
    "referrer-policy": "no-referrer",
    "cache-control": "no-store",
}
"""Added to every response, the refusals included."""
STATIC_FILES: Final[Mapping[str, str]] = {
    "app.js": "text/javascript; charset=utf-8",
    "app.css": "text/css; charset=utf-8",
}
"""The files under ``static/`` served at ``/static/<name>``, with their media types. ``index.html`` is served at
``/`` to a signed-in browser and ``signin.html`` to anyone else."""
TRADE_DEADLINE: Final = "trade_deadline"
"""The overview's deadline kind for the league's trade deadline, beside :class:`fm.jobs.deadlines.DeadlineKind`."""
_REPORT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}\.md$")
_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"", "0", "false", "no", "off"})
_SETTINGS_KEYS = frozenset({"league", "preset", "reset_overrides", "trade_finder_enabled", "notify_channel"})
_CHANNELS: tuple[str, ...] = get_args(Channel)

type _Decision = Literal["approve", "reject"]
type _Json = dict[str, Any]


class _ApiError(Exception):
    """A request the API refuses, with the status and the message for the page."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


def create_app(
    *,
    port: int,
    url_file: Path | None = None,
    login_ttl: timedelta = LOGIN_TTL,
    clock: Callable[[], float] = time.monotonic,
) -> Starlette:
    """The dashboard for ``http://127.0.0.1:<port>/``: the requests it accepts name that port in their ``Host``.

    ``app.state.login`` holds the one-time login links (:class:`fm.web.login.LoginLinks`, each good for ``login_ttl``
    by ``clock``), written to ``url_file`` when one is given (:func:`fm.web.server.bind_dashboard` gives
    ``<config dir>/dashboard.url``; tests read ``app.state.login.url()``). ``app.state.sessions`` are the signed-in
    browsers. The static files are read once, here, from the installed package (``importlib.resources``), so a
    missing file fails at start rather than on the first request.
    """
    static = resources.files("fm.web") / "static"
    hosts = frozenset({f"127.0.0.1:{port}", f"localhost:{port}"})
    sessions = Sessions(port)
    app = Starlette(
        routes=[
            routing.Route("/", _index, methods=["GET"]),
            routing.Route(LOGIN_PATH, _auth, methods=["GET"]),
            routing.Route(PING_PATH, _ping, methods=["GET"]),
            routing.Route("/static/{name}", _static, methods=["GET"]),
            routing.Route("/api/session", _get_session, methods=["GET"]),
            routing.Route("/api/overview", _get_overview, methods=["GET"]),
            routing.Route("/api/proposals", _get_proposals, methods=["GET"]),
            routing.Route("/api/proposals/{proposal_id:int}/approve", _post_approve, methods=["POST"]),
            routing.Route("/api/proposals/{proposal_id:int}/reject", _post_reject, methods=["POST"]),
            routing.Route("/api/pause", _post_pause, methods=["POST"]),
            routing.Route("/api/resume", _post_resume, methods=["POST"]),
            routing.Route("/api/lineup/{league}", _get_lineup, methods=["GET"]),
            routing.Route("/api/waivers/{league}", _get_waivers, methods=["GET"]),
            routing.Route("/api/reports", _get_reports, methods=["GET"]),
            routing.Route("/api/reports/{name}", _get_report, methods=["GET"]),
            routing.Route("/api/health", _get_health, methods=["GET"]),
            routing.Route("/api/settings", _get_settings, methods=["GET"]),
            routing.Route("/api/settings", _post_settings, methods=["POST"]),
        ],
        middleware=[Middleware(_Guard, hosts=hosts, sessions=sessions)],
        exception_handlers={HTTPException: _http_error},
    )
    app.state.sessions = sessions
    app.state.index_html = (static / "index.html").read_text(encoding="utf-8")
    app.state.signin_html = (static / "signin.html").read_text(encoding="utf-8")
    app.state.static = {name: (static / name).read_bytes() for name in STATIC_FILES}
    app.state.login = LoginLinks(port, url_file=url_file, ttl=login_ttl, clock=clock)
    return app


# --- sessions ---------------------------------------------------------------------------------------------------------


class Sessions:
    """The browsers signed in to one dashboard: session id (the cookie's value) -> CSRF token, newest last, at most
    :data:`MAX_SESSIONS`. Kept in memory only, so a restarted dashboard asks every browser to sign in again (``fm
    ui``). Thread-safe: the dashboard serves from its own thread."""

    def __init__(self, port: int) -> None:
        self.cookie = f"{SESSION_COOKIE_PREFIX}{port}"
        self._by_id: OrderedDict[str, str] = OrderedDict()
        self._lock = threading.Lock()

    def open(self) -> str:
        """A new session; returns its id, for the cookie."""
        session_id, csrf = secrets.token_urlsafe(32), secrets.token_urlsafe(32)
        with self._lock:
            self._by_id[session_id] = csrf
            while len(self._by_id) > MAX_SESSIONS:
                self._by_id.popitem(last=False)
        return session_id

    def csrf(self, headers: Headers) -> str | None:
        """The CSRF token of the session the request's cookie names; ``None`` when it names none (not signed in)."""
        session_id = cookie_parser(headers.get("cookie", "")).get(self.cookie)
        if not session_id:
            return None
        with self._lock:
            return self._by_id.get(session_id)


# --- the guard --------------------------------------------------------------------------------------------------------


class _Guard:
    """The checks every request passes before it reaches a route (the module docstring's Security section)."""

    def __init__(self, app: ASGIApp, *, hosts: frozenset[str], sessions: Sessions) -> None:
        self.app = app
        self.hosts = hosts
        self.sessions = sessions

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] == "lifespan":
            await self.app(scope, receive, send)
            return
        if scope["type"] != "http":  # no websockets: refuse the handshake
            await send({"type": "websocket.close", "code": 1008})
            return

        async def send_secured(message: MutableMapping[str, Any]) -> None:
            if message["type"] == "http.response.start":
                headers = MutableHeaders(scope=message)
                for name, value in SECURITY_HEADERS.items():
                    headers[name] = value
            await send(message)

        refusal = self._refusal(Headers(scope=scope), str(scope["method"]), str(scope["path"]))
        if refusal is not None:
            await refusal(scope, receive, send_secured)
            return
        await self.app(scope, receive, send_secured)

    def _refusal(self, headers: Headers, method: str, path: str) -> Response | None:
        host = headers.get("host", "").strip().lower()
        if host not in self.hosts:
            expected = " or ".join(sorted(self.hosts))
            return _error(400, f"this dashboard answers requests for {expected} only")
        if method in SAFE_METHODS and not path.startswith("/api/"):
            return None  # the page, the static files, the sign-in: each route decides
        csrf = self.sessions.csrf(headers)
        if csrf is None:
            return _error(401, "not signed in: run fm ui to open the dashboard")
        if method in SAFE_METHODS:
            return None
        origin = headers.get("origin")
        if origin is not None and origin.strip().lower() != f"http://{host}":
            return _error(403, "cross-origin requests are refused")
        supplied = headers.get(TOKEN_HEADER, "").encode("utf-8", errors="replace")
        if not secrets.compare_digest(supplied, csrf.encode("ascii")):
            return _error(403, f"missing or wrong {TOKEN_HEADER}; reload the dashboard page")
        return None


def _error(status: int, message: str, headers: Mapping[str, str] | None = None) -> JSONResponse:
    return JSONResponse({"error": message}, status_code=status, headers=headers)


async def _http_error(request: Request, exc: Exception) -> Response:
    """Starlette's own refusals (no such route, a method a route does not take, with its ``Allow``) as JSON too."""
    if isinstance(exc, HTTPException):
        return _error(exc.status_code, exc.detail, exc.headers)
    return _error(500, "internal error")


# --- endpoint plumbing ------------------------------------------------------------------------------------------------


def _api(handler: Callable[[Request], Awaitable[_Json]]) -> Callable[[Request], Awaitable[Response]]:
    """Wrap an API handler: its dict becomes a JSON response and every error a ``{"error"}`` with its status."""

    @functools.wraps(handler)
    async def endpoint(request: Request) -> Response:
        try:
            return JSONResponse(await handler(request))
        except _ApiError as exc:
            return _error(exc.status, exc.message)
        except (views.ViewError, ConfigError) as exc:
            return _error(400, str(exc))
        except ProposalError as exc:
            return _error(409, str(exc))
        except sqlite3.Error as exc:
            logger.exception("dashboard: %s %s", request.method, request.url.path)
            return _error(503, f"state database: {exc}")
        except Exception:
            logger.exception("dashboard: %s %s", request.method, request.url.path)
            return _error(500, "internal error; the log has the details")

    return endpoint


async def _body(request: Request) -> _Json:
    """The request's JSON object; an empty body is ``{}``."""
    raw = await request.body()
    if len(raw) > MAX_BODY_BYTES:
        raise _ApiError(413, f"request body over {MAX_BODY_BYTES} bytes")
    if not raw.strip():
        return {}
    try:
        data = json.loads(raw)
    except ValueError:
        raise _ApiError(400, "the request body is not JSON") from None
    if not isinstance(data, dict):
        raise _ApiError(400, "the request body must be a JSON object")
    return data


def _flag(request: Request, name: str) -> bool:
    value = request.query_params.get(name, "").strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise _ApiError(400, f"{name} must be 0 or 1, got {value!r}")


def _count(request: Request, name: str, default: int) -> int:
    value = request.query_params.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        raise _ApiError(400, f"{name} must be a whole number, got {value!r}") from None


def _time(value: datetime | None) -> str | None:
    return None if value is None else value.astimezone(UTC).isoformat()


# --- the page ---------------------------------------------------------------------------------------------------------


async def _index(request: Request) -> Response:
    """The dashboard to a signed-in browser; to anyone else the static page that says to run ``fm ui`` (401)."""
    if _sessions(request).csrf(request.headers) is None:
        return HTMLResponse(request.app.state.signin_html, status_code=401)
    return HTMLResponse(request.app.state.index_html)


async def _auth(request: Request) -> Response:
    """Spend a one-time login secret: a session cookie and a redirect to ``/``. A used, expired or wrong one gets the
    sign-in page (403), unless this browser is signed in already (a reloaded link), which just goes to ``/``."""
    sessions = _sessions(request)
    links: LoginLinks = request.app.state.login
    offered = request.query_params.get(LOGIN_PARAM, "")
    if offered and links.consume(offered):
        response = RedirectResponse("/", status_code=303)
        response.set_cookie(sessions.cookie, sessions.open(), path="/", httponly=True, samesite="strict")
        return response
    if sessions.csrf(request.headers) is not None:
        return RedirectResponse("/", status_code=303)
    return HTMLResponse(request.app.state.signin_html, status_code=403)


async def _ping(request: Request) -> Response:
    """``fm ui``'s probe: says this is a dashboard and makes sure the link in the URL file has a while to go. It
    answers nothing secret; the link travels only through the owner-only file."""
    links: LoginLinks = request.app.state.login
    links.refresh()
    return JSONResponse({"app": APP_ID})


def _sessions(request: Request) -> Sessions:
    return request.app.state.sessions


async def _static(request: Request) -> Response:
    name = request.path_params["name"]
    if name not in STATIC_FILES:
        return _error(404, f"no static file {name!r}")
    return Response(request.app.state.static[name], media_type=STATIC_FILES[name])


# --- the session ------------------------------------------------------------------------------------------------------


@_api
async def _get_session(request: Request) -> _Json:
    """This session's CSRF token, which every POST sends as ``X-FM-Token``. The guard let the request in, so the cookie
    names a session."""
    csrf = _sessions(request).csrf(request.headers)
    if csrf is None:  # forgotten between the guard and here (MAX_SESSIONS newer sign-ins)
        raise _ApiError(401, "not signed in: run fm ui to open the dashboard")
    return {"csrf": csrf}


# --- overview ---------------------------------------------------------------------------------------------------------


@_api
async def _get_overview(request: Request) -> _Json:
    return await run_in_threadpool(_overview, request.query_params.get("as_of"))


def _overview(as_of: str | None) -> _Json:
    """Every league at a glance plus what needs attention: the session, failing checks and the kill switch."""
    at = views.parse_as_of(as_of)
    checks = run_checks(now=at)
    session = next((check.as_dict() for check in checks if check.name == "session"), None)
    out: _Json = {
        "as_of": _time(at),
        "paused": _pause_dict(pause_state()),
        "session": session,
        "health": {
            "fail": sum(check.status is CheckStatus.FAIL for check in checks),
            "warn": sum(check.status is CheckStatus.WARN for check in checks),
        },
        "config_error": None,
        "leagues": [],
    }
    try:
        config = load_config()
        status = views.status()
    except (ConfigError, views.ViewError) as exc:
        out["config_error"] = str(exc)
        return out
    configured = {league.key: league for league in config.leagues}
    with Store.open() as store:
        for block in status["leagues"]:
            league = configured.get(block["league"])
            if league is not None:  # a config edited between the two reads: the next refresh shows it
                out["leagues"].append(_league_overview(store, league, block, at))
    return out


def _league_overview(store: Store, configured: League, block: _Json, at: datetime) -> _Json:
    """:func:`fm.views.status`'s block for one league, plus the preset, standing, sync age and next deadlines."""
    out = {**block, "preset": configured.preset, "standing": None, "sync_age_s": None, "deadlines": [], "notes": []}
    row = store.leagues.by_key(configured.key)
    if row is None:
        return out
    out["sync_age_s"] = max(0, round((at - row.as_of).total_seconds()))
    team = store.teams.get(row.row_id, row.team_id)
    teams = store.teams.for_league(row.row_id)
    if team is not None:
        out["standing"] = {"playoff_seed": team.playoff_seed, "teams": len(teams)}
    out["deadlines"], out["notes"] = _next_deadlines(store, configured, row, at)
    out["open_proposals"] = [_with_moves(store, item) for item in block.get("open_proposals", [])]
    return out


def _next_deadlines(store: Store, configured: League, row: LeagueRow, at: datetime) -> tuple[list[_Json], list[str]]:
    """The next deadline of each kind within :data:`DEADLINE_HORIZON`, soonest first, from the league's synced
    settings and the captured pro schedule (no guess when either is missing: a note says why)."""
    settings = stored_settings(store, row)
    if settings is None:
        return [], [f"{configured.key}: league settings are not synced; no deadlines"]
    notes: list[str] = []
    found: dict[str, _Json] = {}
    # fm.views reads the captured schedule for the lineup and status views; the overview reads the same capture.
    schedule, _, problem = views._cached_schedule(store, configured)
    if schedule is None:
        notes.append(problem)
    else:
        deadlines, warnings = upcoming(
            configured.key, configured.sport, settings, schedule, now=at, horizon=DEADLINE_HORIZON
        )
        notes.extend(warnings)
        for deadline in deadlines:
            if deadline.at >= at:
                found.setdefault(
                    deadline.kind.value,
                    {
                        "kind": deadline.kind.value,
                        "at": _time(deadline.at),
                        "period": deadline.period,
                        "provisional": deadline.provisional,
                        "text": deadline.describe(),
                    },
                )
    trade = settings.trade.deadline
    if trade is not None and at <= trade <= at + DEADLINE_HORIZON:
        found[TRADE_DEADLINE] = {
            "kind": TRADE_DEADLINE,
            "at": _time(trade),
            "period": None,
            "provisional": False,
            "text": f"{configured.key}: trade deadline",
        }
    return sorted(found.values(), key=lambda item: item["at"]), notes


def _pause_dict(state: PauseState | None) -> _Json:
    if state is None:
        return {"on": False, "since": None, "reason": None, "detail": None}
    return {"on": True, "since": _time(state.since), "reason": state.reason, "detail": state.describe()}


# --- proposals --------------------------------------------------------------------------------------------------------


@_api
async def _get_proposals(request: Request) -> _Json:
    include_closed = _flag(request, "all")
    return await run_in_threadpool(_proposals, request.query_params.get("league"), include_closed)


def _proposals(league: str | None, include_closed: bool) -> _Json:
    listed = views.list_proposals(league, include_closed=include_closed, limit=PROPOSALS_LIMIT)
    with Store.open() as store:
        listed["proposals"] = [_with_moves(store, item) for item in listed["proposals"]]
    return listed


def _with_moves(store: Store, item: _Json) -> _Json:
    """A :func:`fm.views.proposal_dict` plus what the page shows and offers: the kind's label, the moves in player
    names and slot labels, and whether it can still be approved or rejected."""
    row = store.proposals.get(int(item["id"]))
    league = store.leagues.get(row.league_id) if row is not None else None
    exits = TRANSITIONS.get(item["status"], frozenset())
    return {
        **item,
        "label": _kind_label(str(item["kind"])),
        "moves": moves_text(store, league, row) if row is not None else item["summary"],
        "can_approve": "approved" in exits,
        "can_reject": "rejected" in exits,
    }


def _kind_label(kind: str) -> str:
    try:
        return kind_spec(kind).label
    except PolicyError:
        return kind


@_api
async def _post_approve(request: Request) -> _Json:
    return await run_in_threadpool(_decide, request.path_params["proposal_id"], "approve")


@_api
async def _post_reject(request: Request) -> _Json:
    return await run_in_threadpool(_decide, request.path_params["proposal_id"], "reject")


def _decide(proposal_id: int, decision: _Decision) -> _Json:
    """Record a human decision. Approving issues the execution token inside the store and executes nothing; a decision
    the proposal's status does not allow (decided already, past its deadline) is a ``ProposalError``: 409."""
    with Store.open() as store:
        if store.proposals.get(proposal_id) is None:
            raise _ApiError(404, f"no proposal #{proposal_id}")
        decide = approve if decision == "approve" else reject
        row = decide(store, proposal_id, decided_by=DECIDED_BY)
        league = store.leagues.get(row.league_id)
        return {"proposal": _with_moves(store, views.proposal_dict(row, league.key if league is not None else None))}


# --- the kill switch --------------------------------------------------------------------------------------------------


@_api
async def _post_pause(request: Request) -> _Json:
    reason = (await _body(request)).get("reason")
    if reason is not None and not isinstance(reason, str):
        raise _ApiError(400, "reason must be text")
    state = await run_in_threadpool(pause, reason)
    return {"paused": _pause_dict(state)}


@_api
async def _post_resume(request: Request) -> _Json:
    await run_in_threadpool(resume)
    return {"paused": _pause_dict(None)}


# --- lineup and waivers -----------------------------------------------------------------------------------------------


@_api
async def _get_lineup(request: Request) -> _Json:
    league = request.path_params["league"]
    as_of = request.query_params.get("as_of")
    return await run_in_threadpool(_lineup, league, as_of)


def _lineup(league: str, as_of: str | None) -> _Json:
    _known_league(league)
    return views.lineup(league, as_of=as_of)


@_api
async def _get_waivers(request: Request) -> _Json:
    league = request.path_params["league"]
    as_of = request.query_params.get("as_of")
    top = _count(request, "top", 10)
    return await run_in_threadpool(_waivers, league, as_of, top)


def _waivers(league: str, as_of: str | None, top: int) -> _Json:
    _known_league(league)
    return views.waivers(league, as_of=as_of, top=top)


def _known_league(key: str) -> Config:
    config = load_config()
    if key not in {league.key for league in config.leagues}:
        known = ", ".join(league.key for league in config.leagues)
        raise _ApiError(404, f"no league {key!r} in config.toml; known: {known}")
    return config


# --- reports ----------------------------------------------------------------------------------------------------------


def _reports_dir() -> Path:
    """Where ``fm report`` writes (:func:`fm.jobs.report.reports_dir`), without creating it."""
    return paths.cache_dir() / REPORTS_DIRNAME


@_api
async def _get_reports(request: Request) -> _Json:
    return await run_in_threadpool(_reports)


def _reports() -> _Json:
    """The markdown reports, newest first. Odds sidecars and anything else in the directory are not listed."""
    directory = _reports_dir()
    found: list[_Json] = []
    if directory.is_dir():
        for path in directory.iterdir():
            if not path.is_file() or not _REPORT_NAME.fullmatch(path.name):
                continue
            stat = path.stat()
            found.append(
                {
                    "name": path.name,
                    "size": stat.st_size,
                    "modified": _time(datetime.fromtimestamp(stat.st_mtime, tz=UTC)),
                }
            )
    found.sort(key=lambda item: (item["modified"], item["name"]), reverse=True)
    return {"directory": str(directory), "reports": found}


@_api
async def _get_report(request: Request) -> _Json:
    return await run_in_threadpool(_report, request.path_params["name"])


def _report(name: str) -> _Json:
    """One report's markdown. Only a plain file name from the listing is served: no path ever reaches the disk."""
    if not _REPORT_NAME.fullmatch(name):
        raise _ApiError(404, f"no report {name!r}")
    directory = _reports_dir()
    path = directory / name
    if not path.is_file() or path.resolve().parent != directory.resolve():
        raise _ApiError(404, f"no report {name!r}")
    if path.stat().st_size > MAX_REPORT_BYTES:
        raise _ApiError(413, f"report {name!r} is over {MAX_REPORT_BYTES} bytes; open it from {directory}")
    return {"name": name, "markdown": path.read_text(encoding="utf-8", errors="replace")}


# --- health -----------------------------------------------------------------------------------------------------------


@_api
async def _get_health(request: Request) -> _Json:
    return await run_in_threadpool(_health)


def _health() -> _Json:
    """:func:`fm.health.run_checks` offline, as ``fm doctor --json`` shows it. It never launches the browser: the
    session verdict comes from the last tick's record (``fm doctor`` reads the cookie when that is stale)."""
    checks = run_checks()
    return {
        "as_of": _time(datetime.now(UTC)),
        "has_failure": has_failure(checks),
        "checks": [check.as_dict() for check in checks],
    }


# --- settings ---------------------------------------------------------------------------------------------------------


@_api
async def _get_settings(request: Request) -> _Json:
    return await run_in_threadpool(_settings)


def _settings() -> _Json:
    """What the settings form edits: per league the preset, the written keys that override it and whether the trade
    finder drafts offers; the notify channel; and whether each ``.env`` variable is set (never its value)."""
    path = paths.config_file()
    out: _Json = {
        "config_path": str(path),
        "presets": [{"name": name, "description": PRESET_DESCRIPTIONS[name]} for name in PRESET_NAMES],
        "channels": list(_CHANNELS),
        "config_error": None,
        "leagues": [],
        "notify": None,
        "secrets": {},
        "missing_secrets": [],
    }
    try:
        config = load_config()
    except ConfigError as exc:
        out["config_error"] = str(exc)
        return out
    out["leagues"] = [
        {
            "league": league.key,
            "sport": league.sport,
            "season": league.season,
            "preset": league.preset,
            "overrides": preset_overrides(path, league.key),
            "trade_finder_enabled": league.trade_finder.enabled,
            "policy": league.policy.model_dump(mode="json"),
        }
        for league in config.leagues
    ]
    out["notify"] = {"channel": config.notify.channel}
    out["secrets"] = {name: config.secrets.is_set(name) for name in SECRET_ENV_VARS}
    out["missing_secrets"] = list(config.missing_secrets())
    return out


@_api
async def _post_settings(request: Request) -> _Json:
    body = await _body(request)
    return await run_in_threadpool(_update_settings, body)


def _update_settings(body: _Json) -> _Json:
    """Apply one form's changes through :mod:`fm.config_edit` (validated before the file is replaced; an edit
    ``config.toml`` would reject is a 400 and leaves the file as it was) and return the settings as they now stand.

    ``{"league", "preset", "reset_overrides", "trade_finder_enabled"}`` edits one league in one write: ``preset`` is a
    preset name or ``null`` (no preset), ``reset_overrides`` removes the written keys that override the preset (the new
    one when ``preset`` is given). ``{"notify_channel"}`` sets ``[notify] channel``.
    """
    unknown = sorted(set(body) - _SETTINGS_KEYS)
    if unknown:
        raise _ApiError(400, f"unknown settings {', '.join(unknown)}; known: {', '.join(sorted(_SETTINGS_KEYS))}")
    path = paths.config_file()
    changes = _league_changes(path, body)
    channel = body.get("notify_channel")
    if "notify_channel" in body and channel not in _CHANNELS:
        raise _ApiError(400, f"notify_channel must be one of {', '.join(_CHANNELS)}")
    if not changes and channel is None:
        raise _ApiError(400, "nothing to change: send preset, reset_overrides, trade_finder_enabled or notify_channel")
    if changes:
        league = body.get("league")
        if not isinstance(league, str):
            raise _ApiError(400, "league must name the league to change")
        set_values(path, league, changes)
    if channel is not None:
        set_notify_channel(path, channel)
    return _settings()


def _league_changes(path: Path, body: _Json) -> dict[str, object]:
    """The dotted keys :func:`fm.config_edit.set_values` writes for one league's form (``None`` removes a key)."""
    changes: dict[str, object] = {}
    if "preset" in body:
        preset = body["preset"]
        if preset is not None and preset not in PRESET_NAMES:
            raise _ApiError(400, f"preset must be one of {', '.join(PRESET_NAMES)} or null")
        changes["preset"] = preset
    if "trade_finder_enabled" in body:
        enabled = body["trade_finder_enabled"]
        if not isinstance(enabled, bool):
            raise _ApiError(400, "trade_finder_enabled must be true or false")
        changes["trade_finder.enabled"] = enabled
    reset = body.get("reset_overrides", False)
    if not isinstance(reset, bool):
        raise _ApiError(400, "reset_overrides must be true or false")
    if reset:
        league = body.get("league")
        if not isinstance(league, str):
            raise _ApiError(400, "league must name the league to reset")
        preset = changes["preset"] if "preset" in changes else _written_preset(path, league)
        if preset is None:
            raise _ApiError(400, f"league {league!r} has no preset, so nothing overrides one")
        for key in preset_overrides(path, league, preset=str(preset)):
            changes.setdefault(key, None)  # a key this same form sets explicitly stays as sent
    return changes


def _written_preset(path: Path, league: str) -> str | None:
    config = load_config(path)
    try:
        return config.league(league).preset
    except KeyError:
        raise _ApiError(404, f"no league {league!r} in config.toml") from None
