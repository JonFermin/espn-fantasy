"""ESPN session cookies from the browser profile, plus expiry detection (DESIGN section 6.2).

ESPN authenticates private-league reads and the web app's writes with two cookies: ``espn_s2`` (the login session) and
``SWID`` (the account id, braces included). Both live in the persistent browser profile that ``fm login`` fills by
hand, and this module reads them from an open browser context, so there is one source of truth and nothing
ESPN-related in ``.env``. Expiry is detected two ways: ahead of time from the ``espn_s2`` cookie's expiry date, so a
tick can warn early, and after the fact from a 401 ``AUTH_MISSING_CREDENTIALS`` response.

Renewal is a new sign-in, not a re-read: ``fm login`` forgets ESPN's cookies before opening the login page, so the
form shows instead of an already-signed-in home page, and then waits for an ``espn_s2`` that differs from the one it
forgot. A sign-in always mints a new value, so that is what tells a renewal apart from the stale session (still
dated in the future, possibly revoked server-side) that was already in the jar.

Cookie values are secrets. ``EspnSession`` keeps them out of ``repr`` and nothing here prints or stores them.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from re import Pattern
from typing import Any, Protocol
from urllib.parse import unquote

from playwright.sync_api import Error as PlaywrightError

from fm.browser.session import LaunchOptions, open_browser, profile_exists

LOGIN_URL = "https://www.espn.com/login"
"""Where ``fm login`` sends the headed browser. The Disney sign-in is completed by hand."""

COOKIE_DOMAIN = "espn.com"
COOKIE_DOMAIN_PATTERN = re.compile(rf"(^|\.){re.escape(COOKIE_DOMAIN)}$")
"""``espn.com`` and its subdomains as a cookie's ``domain`` field (``.espn.com``, ``fantasy.espn.com``), for
``BrowserContext.clear_cookies``. Matches exactly the cookies ``find_session`` would read."""
S2_COOKIE = "espn_s2"
SWID_COOKIE = "SWID"
AUTH_FAILURE_CODE = "AUTH_MISSING_CREDENTIALS"
DEFAULT_WARN_BEFORE = timedelta(days=7)
DEFAULT_LOGIN_TIMEOUT_S = 600.0
DEFAULT_POLL_S = 1.0


class AuthError(RuntimeError):
    """A problem with the ESPN session."""


class NotLoggedInError(AuthError):
    """The browser profile holds no ESPN session."""


class LoginTimeoutError(AuthError):
    """The manual sign-in did not finish in time."""


class LoginAbortedError(AuthError):
    """The browser window closed before the sign-in finished."""


class SessionStatus(StrEnum):
    """Health of a saved session, judged from its cookie expiry date alone.

    A profile with no session is not a status: ``harvest_session`` and ``load_session`` raise ``NotLoggedInError``
    for it, and callers that need a single verdict (the tick's session check) map that exception themselves.
    """

    OK = "ok"
    EXPIRING = "expiring"
    EXPIRED = "expired"


class CookieSource(Protocol):
    """Anything with a Playwright-shaped cookie jar: a ``BrowserContext``, or a fake in tests."""

    def cookies(self) -> Sequence[Mapping[str, Any]]: ...


class CookieJar(CookieSource, Protocol):
    """A ``CookieSource`` whose cookies can also be cleared by domain: a ``BrowserContext``, or a fake in tests."""

    def clear_cookies(self, *, domain: str | Pattern[str] | None = None) -> None: ...


@dataclass(frozen=True)
class EspnSession:
    """The two ESPN cookies and the login cookie's expiry (UTC; ``None`` for a cookie without an expiry date)."""

    espn_s2: str = field(repr=False)
    swid: str = field(repr=False)
    expires_at: datetime | None = None

    def status(self, now: datetime | None = None, *, warn_before: timedelta = DEFAULT_WARN_BEFORE) -> SessionStatus:
        """``EXPIRED`` at or past the expiry, ``EXPIRING`` within ``warn_before`` of it, otherwise ``OK``."""
        if self.expires_at is None:
            return SessionStatus.OK
        current = now if now is not None else datetime.now(UTC)
        if self.expires_at <= current:
            return SessionStatus.EXPIRED
        if self.expires_at - current <= warn_before:
            return SessionStatus.EXPIRING
        return SessionStatus.OK

    def as_cookies(self) -> dict[str, str]:
        """Cookie dict for an ``httpx`` client: ``{"espn_s2": ..., "SWID": ...}``."""
        return {S2_COOKIE: self.espn_s2, SWID_COOKIE: self.swid}

    def cookie_header(self) -> str:
        """The same two cookies as a ``Cookie`` header value."""
        return "; ".join(f"{name}={value}" for name, value in self.as_cookies().items())


def normalize_swid(value: str) -> str:
    """ESPN's API wants the SWID with braces; the jar may hold it URL-encoded (``%7B...%7D``) or bare."""
    swid = unquote(value).strip()
    if not swid.startswith("{"):
        swid = "{" + swid
    if not swid.endswith("}"):
        swid = swid + "}"
    return swid


def cookie_expiry(cookie: Mapping[str, Any]) -> datetime | None:
    """The cookie's ``expires`` (Unix seconds) as an aware UTC datetime; ``None`` for session cookies (``-1``)."""
    seconds = _expires_seconds(cookie)
    if seconds == float("-inf"):
        return None
    try:
        return datetime.fromtimestamp(seconds, tz=UTC)
    except (OverflowError, OSError, ValueError):
        return None


def find_session(jar: Iterable[Mapping[str, Any]]) -> EspnSession | None:
    """Pick ``espn_s2`` and ``SWID`` for ``espn.com`` out of a cookie jar, or ``None`` when either is missing.

    An anonymous visitor already has a ``SWID``; only a signed-in one has ``espn_s2``. When a name appears more than
    once, the persistent cookie with the latest expiry wins over session cookies.
    """
    cookies = list(jar)
    s2 = _best(cookies, S2_COOKIE)
    swid = _best(cookies, SWID_COOKIE)
    if s2 is None or swid is None:
        return None
    return EspnSession(espn_s2=str(s2["value"]), swid=normalize_swid(str(swid["value"])), expires_at=cookie_expiry(s2))


def harvest_session(source: CookieSource) -> EspnSession:
    """Read the ESPN session out of a browser context's cookie jar; ``NotLoggedInError`` when there is none."""
    session = find_session(source.cookies())
    if session is None:
        raise NotLoggedInError("no ESPN session in the browser profile; run `fm login`")
    return session


def clear_session(source: CookieJar) -> EspnSession | None:
    """Forget every ``espn.com`` cookie in the jar and return the session that was there, or ``None``.

    ``fm login`` calls this before opening the login page: with a live ``espn_s2`` still in the jar ESPN skips the
    form, so the command could never mint a new session. The returned snapshot is what ``wait_for_login`` compares
    against. Raises ``LoginAbortedError`` when the window is already gone.
    """
    try:
        previous = find_session(source.cookies())
        source.clear_cookies(domain=COOKIE_DOMAIN_PATTERN)
    except PlaywrightError as exc:
        raise LoginAbortedError("the browser window was closed before the sign-in finished") from exc
    return previous


def wait_for_login(
    source: CookieSource,
    *,
    previous: EspnSession | None = None,
    timeout_s: float = DEFAULT_LOGIN_TIMEOUT_S,
    poll_s: float = DEFAULT_POLL_S,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> EspnSession:
    """Poll the jar until a sign-in lands both cookies with an ``espn_s2`` other than ``previous``'s.

    ``previous`` is the session saved before the login page opened (see ``clear_session``). A sign-in always mints a
    new ``espn_s2``, so an unchanged value is the old session still sitting in the jar, not a renewal, and the wait
    goes on. Raises ``LoginTimeoutError`` or ``LoginAbortedError``.
    """
    deadline = clock() + timeout_s
    while True:
        try:
            jar = source.cookies()
        except PlaywrightError as exc:
            raise LoginAbortedError("the browser window was closed before the sign-in finished") from exc
        session = find_session(jar)
        if session is not None and (previous is None or session.espn_s2 != previous.espn_s2):
            return session
        if clock() >= deadline:
            raise LoginTimeoutError(f"no ESPN sign-in after {timeout_s:.0f}s; run `fm login` again and sign in")
        sleep(poll_s)


def load_session(options: LaunchOptions | None = None) -> EspnSession:
    """Open the persistent profile (headless by default) and harvest the session; the runtime entry point.

    Raises ``NotLoggedInError`` before touching a browser when no profile exists yet, and ``BrowserError`` when the
    browser cannot start.
    """
    opts = options if options is not None else LaunchOptions()
    if not profile_exists(opts.resolved_profile_dir()):
        raise NotLoggedInError("no browser profile yet; run `fm login`")
    with open_browser(opts) as browser:
        return harvest_session(browser.context)


def is_auth_failure(status_code: int, body: str | bytes | None = None) -> bool:
    """True for an ESPN API response that means the session is gone: a 401, or ``AUTH_MISSING_CREDENTIALS`` in it."""
    if status_code == 401:
        return True
    if body is None:
        return False
    text = body.decode(errors="replace") if isinstance(body, bytes) else body
    return AUTH_FAILURE_CODE in text


def describe(session: EspnSession, now: datetime | None = None) -> str:
    """One line for the CLI and notifications. Never includes cookie values."""
    current = now if now is not None else datetime.now(UTC)
    if session.expires_at is None:
        return "ESPN session saved; the cookie carries no expiry date."
    day = session.expires_at.date().isoformat()
    days_left = max((session.expires_at - current).days, 0)
    match session.status(current):
        case SessionStatus.EXPIRED:
            return f"ESPN session expired on {day}; run `fm login`."
        case SessionStatus.EXPIRING:
            return f"ESPN session expires soon: {day} ({days_left} days left); run `fm login` to renew."
        case _:
            return f"ESPN session OK until {day} ({days_left} days left)."


def _on_espn(domain: object) -> bool:
    if not isinstance(domain, str):
        return False
    host = domain.lstrip(".").lower()
    return host == COOKIE_DOMAIN or host.endswith("." + COOKIE_DOMAIN)


def _expires_seconds(cookie: Mapping[str, Any]) -> float:
    raw = cookie.get("expires")
    if isinstance(raw, int | float) and not isinstance(raw, bool) and raw > 0:
        return float(raw)
    return float("-inf")


def _best(cookies: Sequence[Mapping[str, Any]], name: str) -> Mapping[str, Any] | None:
    matches = [c for c in cookies if c.get("name") == name and c.get("value") and _on_espn(c.get("domain"))]
    if not matches:
        return None
    return max(matches, key=_expires_seconds)
