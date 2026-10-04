"""Cookie harvest from a fake browser context, session expiry detection, the login wait loop, and session renewal.

Cookie values here are obviously fake. No browser starts and nothing touches ESPN: the autouse guard in
``tests/browser/conftest.py`` turns any launch that slipped past a monkeypatch into a test failure.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from re import Pattern
from types import SimpleNamespace
from typing import Any

import pytest
from playwright.sync_api import Error as PlaywrightError

from fm.browser.session import LaunchOptions
from fm.espn import auth
from fm.espn.auth import (
    COOKIE_DOMAIN_PATTERN,
    EspnSession,
    LoginAbortedError,
    LoginTimeoutError,
    NotLoggedInError,
    SessionStatus,
    clear_session,
    describe,
    find_session,
    harvest_session,
    is_auth_failure,
    load_session,
    normalize_swid,
    wait_for_login,
)

NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
S2 = "AEBfake-espn_s2-value-not-a-real-session"
RENEWED_S2 = "AEBfake-espn_s2-value-minted-by-a-second-sign-in"
SWID = "{FAKE0000-0000-4000-8000-000000000001}"


def cookie(name: str, value: str, *, domain: str = ".espn.com", expires: float = -1, path: str = "/") -> dict[str, Any]:
    """A cookie in the shape ``BrowserContext.cookies()`` returns; ``expires=-1`` is a session cookie."""
    return {
        "name": name,
        "value": value,
        "domain": domain,
        "path": path,
        "expires": expires,
        "httpOnly": True,
        "secure": True,
        "sameSite": "Lax",
    }


def in_days(days: float) -> float:
    return (NOW + timedelta(days=days)).timestamp()


# ESPN sets SWID for anonymous visitors too; only a signed-in visitor has espn_s2.
ANONYMOUS = [cookie("SWID", SWID, expires=in_days(3650)), cookie("_dcf", "1")]
SIGNED_IN = [*ANONYMOUS, cookie("espn_s2", S2, expires=in_days(100))]
RENEWED = [*ANONYMOUS, cookie("espn_s2", RENEWED_S2, expires=in_days(400))]


class FakeContext:
    """Cookie jar that changes between polls: each ``cookies()`` call moves to the next jar (or raises it)."""

    def __init__(self, *jars: list[dict[str, Any]] | Exception) -> None:
        self.jars = list(jars)
        self.calls = 0
        self.cleared: list[str | Pattern[str] | None] = []

    def cookies(self) -> list[dict[str, Any]]:
        jar = self.jars[min(self.calls, len(self.jars) - 1)]
        self.calls += 1
        if isinstance(jar, Exception):
            raise jar
        return list(jar)

    def clear_cookies(
        self,
        *,
        name: str | Pattern[str] | None = None,
        domain: str | Pattern[str] | None = None,
        path: str | Pattern[str] | None = None,
    ) -> None:
        self.cleared.append(domain)


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


# --- harvest -----------------------------------------------------------------------------------------------------


def test_harvest_reads_both_cookies_and_the_login_expiry() -> None:
    session = harvest_session(FakeContext(SIGNED_IN))
    assert session.espn_s2 == S2
    assert session.swid == SWID
    assert session.expires_at == NOW + timedelta(days=100)
    assert session.status(NOW) is SessionStatus.OK


def test_anonymous_visitor_has_a_swid_but_no_session() -> None:
    with pytest.raises(NotLoggedInError, match="no ESPN session in the browser profile; run `fm login`"):
        harvest_session(FakeContext(ANONYMOUS))
    assert find_session([]) is None


def test_cookies_from_other_domains_do_not_count() -> None:
    assert find_session([cookie("espn_s2", S2, domain=".example.com"), cookie("SWID", SWID, domain=".go.com")]) is None
    assert find_session([cookie("espn_s2", S2, domain=".notespn.com"), cookie("SWID", SWID)]) is None
    on_hosts = [cookie("espn_s2", S2, domain="fantasy.espn.com"), cookie("SWID", SWID, domain="www.espn.com")]
    assert find_session(on_hosts) is not None


def test_empty_values_are_treated_as_missing() -> None:
    assert find_session([cookie("espn_s2", ""), cookie("SWID", SWID)]) is None


def test_swid_is_url_decoded_and_braced() -> None:
    assert normalize_swid("%7BFAKE0000-0000-4000-8000-000000000001%7D") == SWID
    assert normalize_swid("FAKE0000-0000-4000-8000-000000000001") == SWID
    assert normalize_swid(SWID) == SWID
    session = find_session([cookie("espn_s2", S2), cookie("SWID", "%7BFAKE0000-0000-4000-8000-000000000001%7D")])
    assert session is not None
    assert session.swid == SWID


def test_persistent_cookie_with_the_latest_expiry_wins_over_duplicates() -> None:
    jar = [
        cookie("espn_s2", "stale", expires=in_days(1)),
        cookie("espn_s2", "session-only"),
        cookie("espn_s2", "fresh", expires=in_days(200), domain="www.espn.com"),
        cookie("SWID", SWID),
    ]
    session = find_session(jar)
    assert session is not None
    assert session.espn_s2 == "fresh"
    assert session.expires_at == NOW + timedelta(days=200)


def test_session_cookie_without_an_expiry_date() -> None:
    session = find_session([cookie("espn_s2", S2), cookie("SWID", SWID)])
    assert session is not None
    assert session.expires_at is None
    assert session.status(NOW) is SessionStatus.OK
    assert describe(session, NOW) == "ESPN session saved; the cookie carries no expiry date."


def test_unparseable_expiry_is_treated_as_no_date() -> None:
    session = find_session([cookie("espn_s2", S2, expires=1e300), cookie("SWID", SWID)])
    assert session is not None
    assert session.expires_at is None


# --- expiry ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("days", "expected"),
    [
        (365, SessionStatus.OK),
        (7.5, SessionStatus.OK),
        (7, SessionStatus.EXPIRING),
        (0.5, SessionStatus.EXPIRING),
        (0, SessionStatus.EXPIRED),
        (-30, SessionStatus.EXPIRED),
    ],
)
def test_status_from_the_cookie_expiry(days: float, expected: SessionStatus) -> None:
    session = EspnSession(espn_s2=S2, swid=SWID, expires_at=NOW + timedelta(days=days))
    assert session.status(NOW) is expected


def test_warn_window_is_configurable() -> None:
    session = EspnSession(espn_s2=S2, swid=SWID, expires_at=NOW + timedelta(days=20))
    assert session.status(NOW) is SessionStatus.OK
    assert session.status(NOW, warn_before=timedelta(days=30)) is SessionStatus.EXPIRING


def test_status_defaults_to_the_current_time() -> None:
    assert EspnSession(S2, SWID, datetime(2000, 1, 1, tzinfo=UTC)).status() is SessionStatus.EXPIRED
    assert EspnSession(S2, SWID, datetime(2999, 1, 1, tzinfo=UTC)).status() is SessionStatus.OK


def test_describe_lines() -> None:
    ok = EspnSession(S2, SWID, NOW + timedelta(days=100, hours=3))
    assert describe(ok, NOW) == "ESPN session OK until 2027-01-12 (100 days left)."
    soon = EspnSession(S2, SWID, NOW + timedelta(days=5, hours=1))
    assert describe(soon, NOW) == "ESPN session expires soon: 2026-10-09 (5 days left); run `fm login` to renew."
    gone = EspnSession(S2, SWID, NOW - timedelta(days=2))
    assert describe(gone, NOW) == "ESPN session expired on 2026-10-02; run `fm login`."


def test_repr_str_and_describe_never_show_cookie_values() -> None:
    session = EspnSession(espn_s2=S2, swid=SWID, expires_at=NOW)
    for text in (repr(session), str(session), describe(session, NOW)):
        assert S2 not in text
        assert SWID not in text
    assert "expires_at" in repr(session)


def test_cookies_for_the_api_client() -> None:
    session = EspnSession(espn_s2=S2, swid=SWID)
    assert session.as_cookies() == {"espn_s2": S2, "SWID": SWID}
    assert session.cookie_header() == f"espn_s2={S2}; SWID={SWID}"


@pytest.mark.parametrize(
    ("status_code", "body", "expected"),
    [
        (401, None, True),
        (401, b"", True),
        (200, '{"teams": []}', False),
        (400, '{"messages": ["AUTH_MISSING_CREDENTIALS"]}', True),
        (403, b'{"messages": ["AUTH_MISSING_CREDENTIALS"]}', True),
        (403, "forbidden", False),
        (500, None, False),
    ],
)
def test_is_auth_failure_from_an_api_response(status_code: int, body: str | bytes | None, expected: bool) -> None:
    assert is_auth_failure(status_code, body) is expected


# --- waiting for the manual sign-in ------------------------------------------------------------------------------


def test_wait_for_login_polls_until_the_sign_in_lands_both_cookies() -> None:
    context = FakeContext([], ANONYMOUS, ANONYMOUS, SIGNED_IN)
    clock = Clock()
    session = wait_for_login(context, timeout_s=600, poll_s=2, clock=clock.monotonic, sleep=clock.sleep)
    assert session.espn_s2 == S2
    assert context.calls == 4
    assert clock.sleeps == [2, 2, 2]


def test_wait_for_login_times_out() -> None:
    context = FakeContext(ANONYMOUS)
    clock = Clock()
    with pytest.raises(LoginTimeoutError, match="no ESPN sign-in after 10s"):
        wait_for_login(context, timeout_s=10, poll_s=4, clock=clock.monotonic, sleep=clock.sleep)
    assert clock.sleeps == [4, 4, 4]  # polled at t=0, 4, 8 and 12; the last one is past the deadline
    assert context.calls == 4


def test_wait_for_login_aborts_when_the_window_is_closed() -> None:
    closed = PlaywrightError("Target page, context or browser has been closed")
    context = FakeContext(ANONYMOUS, closed)
    clock = Clock()
    with pytest.raises(LoginAbortedError, match="closed before the sign-in finished") as info:
        wait_for_login(context, timeout_s=600, poll_s=1, clock=clock.monotonic, sleep=clock.sleep)
    assert info.value.__cause__ is closed


# --- renewing a saved session ------------------------------------------------------------------------------------


def test_clear_session_forgets_the_espn_cookies_and_returns_what_was_saved() -> None:
    context = FakeContext(SIGNED_IN)
    previous = clear_session(context)
    assert previous is not None
    assert previous.espn_s2 == S2
    assert previous.expires_at == NOW + timedelta(days=100)
    assert context.cleared == [COOKIE_DOMAIN_PATTERN]
    assert context.calls == 1  # the snapshot is taken before the clear


def test_clear_session_with_nothing_saved_still_clears_and_returns_none() -> None:
    for jar in (ANONYMOUS, []):
        context = FakeContext(jar)
        assert clear_session(context) is None
        assert context.cleared == [COOKIE_DOMAIN_PATTERN]


def test_clear_session_aborts_when_the_window_is_already_closed() -> None:
    closed = PlaywrightError("Target page, context or browser has been closed")
    context = FakeContext(closed)
    with pytest.raises(LoginAbortedError, match="closed before the sign-in finished") as info:
        clear_session(context)
    assert info.value.__cause__ is closed
    assert context.cleared == []


@pytest.mark.parametrize(
    ("domain", "espn"),
    [
        (".espn.com", True),
        ("espn.com", True),
        ("www.espn.com", True),
        ("fantasy.espn.com", True),
        (".go.com", False),
        ("registerdisney.go.com", False),
        (".notespn.com", False),
        ("espn.com.example", False),
    ],
)
def test_clear_pattern_covers_exactly_the_domains_find_session_reads(domain: str, espn: bool) -> None:
    assert (COOKIE_DOMAIN_PATTERN.search(domain) is not None) is espn
    jar = [cookie("espn_s2", S2, domain=domain), cookie("SWID", SWID, domain=domain)]
    assert (find_session(jar) is not None) is espn


def test_wait_for_login_with_a_saved_session_waits_for_a_new_espn_s2() -> None:
    previous = find_session(SIGNED_IN)
    context = FakeContext(SIGNED_IN, SIGNED_IN, RENEWED)  # the old cookie is still there on the first two polls
    clock = Clock()
    session = wait_for_login(
        context, previous=previous, timeout_s=600, poll_s=1, clock=clock.monotonic, sleep=clock.sleep
    )
    assert session.espn_s2 == RENEWED_S2
    assert session.expires_at == NOW + timedelta(days=400)
    assert context.calls == 3
    assert clock.sleeps == [1, 1]


def test_wait_for_login_times_out_when_the_saved_session_never_changes() -> None:
    # The 401 case: a server-revoked espn_s2 whose expiry date is still far out looks fine from the jar alone, so an
    # unchanged value must never count as a sign-in.
    previous = find_session(SIGNED_IN)
    context = FakeContext(SIGNED_IN)
    clock = Clock()
    with pytest.raises(LoginTimeoutError, match="no ESPN sign-in after 10s"):
        wait_for_login(context, previous=previous, timeout_s=10, poll_s=5, clock=clock.monotonic, sleep=clock.sleep)
    assert context.calls == 3


def test_wait_for_login_without_a_saved_session_takes_the_first_sign_in() -> None:
    context = FakeContext(SIGNED_IN)
    clock = Clock()
    session = wait_for_login(context, previous=None, timeout_s=600, poll_s=1, clock=clock.monotonic, sleep=clock.sleep)
    assert session.espn_s2 == S2
    assert context.calls == 1


# --- runtime entry point -----------------------------------------------------------------------------------------


def _used_profile(tmp_path: Path) -> Path:
    profile = tmp_path / "config" / "browser-profile"
    profile.mkdir(parents=True)
    (profile / "Local State").write_text("{}", encoding="utf-8")
    return profile


def test_load_session_opens_the_profile_headless_and_harvests(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile = _used_profile(tmp_path)
    seen: list[LaunchOptions | None] = []

    @contextmanager
    def open_browser(options: LaunchOptions | None = None) -> Iterator[SimpleNamespace]:
        seen.append(options)
        yield SimpleNamespace(context=FakeContext(SIGNED_IN), channel="msedge", profile_dir=profile, headless=True)

    monkeypatch.setattr(auth, "open_browser", open_browser)
    session = load_session()
    assert session.espn_s2 == S2
    assert seen == [LaunchOptions()]  # headless, default channel order, config-dir profile


def test_load_session_with_an_anonymous_profile(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    profile = _used_profile(tmp_path)

    @contextmanager
    def open_browser(options: LaunchOptions | None = None) -> Iterator[SimpleNamespace]:
        yield SimpleNamespace(context=FakeContext(ANONYMOUS), channel="msedge", profile_dir=profile, headless=True)

    monkeypatch.setattr(auth, "open_browser", open_browser)
    with pytest.raises(NotLoggedInError, match="no ESPN session in the browser profile"):
        load_session(LaunchOptions(channel="chrome"))


def test_load_session_without_a_profile_never_starts_a_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    def open_browser(options: LaunchOptions | None = None) -> None:
        raise AssertionError("a browser must not be launched when no profile exists")

    monkeypatch.setattr(auth, "open_browser", open_browser)
    with pytest.raises(NotLoggedInError, match="no browser profile yet; run `fm login`"):
        load_session()
