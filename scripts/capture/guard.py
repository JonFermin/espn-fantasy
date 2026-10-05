"""Write guard for the real-league capture (ROADMAP #14, DESIGN section 6.3 "Spike method").

Nothing the capture drives may change a real ESPN league. :meth:`WriteGuard.install` runs before any ESPN page opens
and adds one context-wide route over every request that aborts:

- every request to ``lm-api-writes.fantasy.espn.com`` (the host of ESPN's transaction endpoint),
- every non-GET request to ``espn.com`` or any of its subdomains,
- every request whose URL contains ``transactions``,

recording each aborted request (method, URL, JSON body, resource type, a few non-secret headers). The route fails
closed: a request it cannot classify is aborted too. WebSockets to ESPN hosts are closed before they reach a server,
and GETs to ESPN's JSON APIs are paced at least :data:`MIN_READ_INTERVAL_S` apart, the app's own page loads included.

Two independent checks back the route. A response to anything the route should have aborted is a leak: the guard
records it and :meth:`WriteGuard.check` raises :class:`GuardLeakError`, which stops a run. And :meth:`WriteGuard.prove`
fires probes from the page that would be harmless even if they got through (a POST for league 0, which does not exist,
with an empty body and no credentials; a POST to another ESPN host; a GET whose URL contains ``transactions``) and
raises :class:`GuardProofError` unless each one was aborted and recorded. The capture proves the guard on
``about:blank`` before the first ESPN page and again on that page before it drives any flow.

Playwright answers CORS preflights itself while routing is on (they never reach a server) and routes requests that
service workers make through the context's routes (Chromium), so the route sees every request a write could use.
Cookies and authorization headers are never recorded.

The ``transactions`` marker is matched as written, lowercase: it is the write endpoint's path segment. The web app's
own reads of ``view=mTransactions2`` and ``view=mPendingTransactions`` are GETs to the read host and pass (the team page
needs them); a non-GET to ``.../teams/{id}/pendingTransactions`` is caught by the host and method rules instead.

**Sign-in mode** (``WriteGuard(sign_in=True)``, only for ``capture.py web-login``, where a person signs in by hand)
lets non-GET requests to ESPN hosts outside ``fantasy.espn.com`` through, in case the Disney sign-in needs one. It
still aborts the write host, every URL containing ``transactions`` and every non-GET to ``fantasy.espn.com`` or its
subdomains (where every league API lives), and records each request it let through in :attr:`WriteGuard.passed`.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlsplit

from playwright.sync_api import BrowserContext, Page, Request, Response, Route, WebSocketRoute

WRITE_HOST = "lm-api-writes.fantasy.espn.com"
READ_HOST = "lm-api-reads.fantasy.espn.com"
ESPN_DOMAIN = "espn.com"
FANTASY_DOMAIN = "fantasy.espn.com"
TRANSACTIONS_MARKER = "transactions"
MIN_READ_INTERVAL_S = 1.0
ABORT_ERROR = "blockedbyclient"
KEPT_HEADERS: tuple[str, ...] = (
    "content-type",
    "accept",
    "x-fantasy-filter",
    "x-fantasy-platform",
    "x-fantasy-source",
    "x-fantasy-pref",
)
"""The only request headers a record keeps; cookies and authorization never are."""

PROBE_SEASON = 2026
PROBE_LEAGUE = 0
"""Probes address league 0, which does not exist, so a probe that got through could not touch a real league."""


class GuardError(RuntimeError):
    """The write guard could not be trusted; the run must stop."""


class GuardProofError(GuardError):
    """A probe was not aborted by the guard."""


class GuardLeakError(GuardError):
    """A request the guard should have aborted got a response."""


def _within(host: str | None, domain: str) -> bool:
    if not host:
        return False
    name = host.lower().rstrip(".")
    return name == domain or name.endswith("." + domain)


def is_espn_host(host: str | None) -> bool:
    """``espn.com`` or any subdomain of it."""
    return _within(host, ESPN_DOMAIN)


def is_fantasy_host(host: str | None) -> bool:
    """``fantasy.espn.com`` or any subdomain of it: the web app and every league API (reads, writes, messages)."""
    return _within(host, FANTASY_DOMAIN)


def block_reason(method: str, url: str, *, sign_in: bool = False) -> str | None:
    """Why the guard aborts this request, or ``None`` when it may go out (``sign_in``: see the module docstring)."""
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    verb = method.upper()
    if host == WRITE_HOST:
        return "write host"
    if verb != "GET" and is_espn_host(host) and (not sign_in or is_fantasy_host(host)):
        return f"{verb} to {'a fantasy.espn.com' if sign_in else 'an espn.com'} host"
    if TRANSACTIONS_MARKER in url:
        return f"URL contains {TRANSACTIONS_MARKER!r}"
    return None


def is_api_read(method: str, url: str) -> bool:
    """A GET to one of ESPN's JSON APIs: the fantasy read host, ``fantasy.espn.com/apis/...`` or ``*.api.espn.com``."""
    if method.upper() != "GET":
        return False
    parts = urlsplit(url)
    host = (parts.hostname or "").lower()
    if not is_espn_host(host):
        return False
    return host == READ_HOST or host.endswith(".api.espn.com") or parts.path.startswith("/apis/")


def _parse_body(raw: bytes | None) -> Any:
    if not raw:
        return None
    text = raw.decode("utf-8", errors="replace")
    try:
        return json.loads(text)
    except ValueError:
        return text


def _kept_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {name: value for name, value in headers.items() if name.lower() in KEPT_HEADERS}


@dataclass(frozen=True, slots=True)
class BlockedRequest:
    """A request the guard aborted: the payload capture of a write flow, or a probe."""

    method: str
    url: str
    reason: str
    resource_type: str
    headers: dict[str, str]
    body: Any
    at: datetime
    probe: bool = False

    def as_json(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "url": self.url,
            "reason": self.reason,
            "resource_type": self.resource_type,
            "headers": self.headers,
            "body": self.body,
            "at": self.at.isoformat(),
            "probe": self.probe,
        }


@dataclass(slots=True)
class ApiRead:
    """A GET the page made to an ESPN API (which views the web app uses, with its ``X-Fantasy-Filter``)."""

    url: str
    filter: Any
    resource_type: str
    at: datetime
    status: int | None = None

    def as_json(self) -> dict[str, Any]:
        return {
            "url": self.url,
            "filter": self.filter,
            "resource_type": self.resource_type,
            "at": self.at.isoformat(),
            "status": self.status,
        }


@dataclass
class WriteGuard:
    """The route, its records, and the checks around it. One guard per browser context."""

    min_read_interval_s: float = MIN_READ_INTERVAL_S
    pace_reads: bool = True
    sign_in: bool = False
    """Sign-in mode (module docstring): non-GET requests to ESPN hosts outside ``fantasy.espn.com`` go out."""
    sleep: Callable[[float], object] = time.sleep
    monotonic: Callable[[], float] = time.monotonic
    blocked: list[BlockedRequest] = field(default_factory=list)
    passed: list[str] = field(default_factory=list)
    """Non-GET requests to ESPN hosts that sign-in mode let through (method and URL)."""
    reads: list[ApiRead] = field(default_factory=list)
    websockets: list[str] = field(default_factory=list)
    leaks: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    proven: bool = False
    _last_read: float | None = field(default=None, init=False, repr=False)
    _probe_urls: frozenset[str] = field(default=frozenset(), init=False, repr=False)

    def install(self, context: BrowserContext) -> None:
        """Route every request and WebSocket of ``context`` through the guard. Call before opening any ESPN page."""
        context.route("**/*", self._on_route)
        context.route_web_socket(lambda url: is_espn_host(urlsplit(url).hostname), self._on_websocket)
        context.on("response", self._on_response)

    # --- route ------------------------------------------------------------------------------------------------------

    def _on_route(self, route: Route, request: Request) -> None:
        try:
            method, url = request.method, request.url
            reason = block_reason(method, url, sign_in=self.sign_in)
        except Exception as exc:  # fail closed: a request the guard cannot classify never goes out
            self.errors.append(f"classify failed: {exc!r}")
            route.abort(ABORT_ERROR)
            return
        if reason is not None:
            try:
                self._record_blocked(request, reason)
            finally:
                route.abort(ABORT_ERROR)
            return
        if method.upper() != "GET" and is_espn_host(urlsplit(url).hostname):
            self.passed.append(f"{method.upper()} {url}")  # only sign-in mode gets here
        if is_api_read(method, url):
            if self.pace_reads:
                self._pace()
            self._record_read(request)
        route.continue_()

    def _record_blocked(self, request: Request, reason: str) -> None:
        try:
            body = _parse_body(request.post_data_buffer)
        except Exception as exc:  # a body we cannot read is still a blocked request worth recording
            body = f"<unreadable body: {exc!r}>"
        self.blocked.append(
            BlockedRequest(
                method=request.method.upper(),
                url=request.url,
                reason=reason,
                resource_type=request.resource_type,
                headers=_kept_headers(request.headers),
                body=body,
                at=datetime.now(UTC),
                probe=request.url in self._probe_urls,
            )
        )

    def _record_read(self, request: Request) -> None:
        raw_filter = request.headers.get("x-fantasy-filter")
        parsed: Any = None
        if raw_filter:
            try:
                parsed = json.loads(raw_filter)
            except ValueError:
                parsed = raw_filter
        self.reads.append(ApiRead(request.url, parsed, request.resource_type, datetime.now(UTC)))

    def _pace(self) -> None:
        now = self.monotonic()
        if self._last_read is not None:
            wait = self.min_read_interval_s - (now - self._last_read)
            if wait > 0:
                self.sleep(wait)
        self._last_read = self.monotonic()

    def _on_websocket(self, ws: WebSocketRoute) -> None:
        self.websockets.append(ws.url)
        ws.close(code=1000, reason="blocked by the capture write guard")

    def _on_response(self, response: Response) -> None:
        request = response.request
        if request.method.upper() == "OPTIONS":
            return  # CORS preflights: Playwright answers them itself while routing is on
        reason = block_reason(request.method, request.url, sign_in=self.sign_in)
        if reason is not None:
            self.leaks.append(f"{request.method} {request.url} -> HTTP {response.status} ({reason})")
            return
        for read in reversed(self.reads):
            if read.url == request.url and read.status is None:
                read.status = response.status
                break

    # --- checks -----------------------------------------------------------------------------------------------------

    def check(self) -> None:
        """Raise :class:`GuardLeakError` if anything that should have been aborted got a response."""
        if self.leaks:
            raise GuardLeakError("write guard leak: " + "; ".join(self.leaks))
        if self.errors:
            raise GuardError("write guard errors: " + "; ".join(self.errors))

    def prove(self, page: Page, *, season: int = PROBE_SEASON) -> list[BlockedRequest]:
        """Fire the probes from ``page`` and confirm each was aborted and recorded; returns their records."""
        league = f"/apis/v3/games/ffl/seasons/{season}/segments/0/leagues/{PROBE_LEAGUE}"
        probes = (
            ("POST", f"https://{WRITE_HOST}{league}/{TRANSACTIONS_MARKER}/"),
            ("POST", f"https://fantasy.espn.com{league}/fm-guard-probe"),
            ("GET", f"https://{READ_HOST}{league}/{TRANSACTIONS_MARKER}/"),
        )
        # Only these URLs count as probes, so an app request aborted during a proof is still a capture.
        self._probe_urls = frozenset(url for _, url in probes)
        records: list[BlockedRequest] = []
        for method, url in probes:
            before = len(self.blocked)
            outcome = page.evaluate(_PROBE_JS, [method, url])
            page.wait_for_timeout(250)
            recorded = [r for r in self.blocked[before:] if r.url == url and r.method == method]
            if outcome.get("sent") or not recorded:
                raise GuardProofError(
                    f"probe {method} {url} was not blocked (page saw {outcome!r}, recorded {len(recorded)})"
                )
            records.extend(recorded)
        self.check()
        self.proven = True
        return records

    @property
    def captures(self) -> list[BlockedRequest]:
        """Aborted requests other than the probes: what the driven flows tried to send."""
        return [record for record in self.blocked if not record.probe]


_PROBE_JS = """
async ([method, url]) => {
  const init = { method, credentials: 'omit', cache: 'no-store' };
  if (method !== 'GET') {
    init.body = '{}';
    init.headers = { 'Content-Type': 'application/json' };
  }
  try {
    const response = await fetch(url, init);
    return { sent: true, status: response.status };
  } catch (error) {
    return { sent: false, error: String(error) };
  }
}
"""
