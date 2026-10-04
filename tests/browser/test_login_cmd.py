"""``fm login``: the manual sign-in flow, renewal of a saved session, and ``--check``, driven through the CLI.

``open_browser`` is replaced in both modules that import it, so no browser starts and nothing touches ESPN. An autouse
guard makes a launch that slipped past the stub fail loudly, and the CLI is invoked without exception capture so a
crash inside the command is a traceback here, not a silent exit code.
"""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from pathlib import Path
from re import Pattern
from typing import Any, NoReturn

import pytest
from playwright.sync_api import Error as PlaywrightError
from typer.testing import CliRunner, Result

from fm import paths
from fm.browser import session as browser_session
from fm.browser.session import BrowserUnavailableError, LaunchOptions
from fm.cli import app
from fm.commands import login as login_cmd
from fm.espn import auth
from fm.espn.auth import COOKIE_DOMAIN_PATTERN, LOGIN_URL

runner = CliRunner()

S2 = "AEBfake-espn_s2-value-not-a-real-session"
RENEWED_S2 = "AEBfake-espn_s2-value-minted-by-a-second-sign-in"
SWID = "{FAKE0000-0000-4000-8000-000000000002}"


def cookie(name: str, value: str, *, expires: float = -1) -> dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "domain": ".espn.com",
        "path": "/",
        "expires": expires,
        "httpOnly": True,
        "secure": True,
        "sameSite": "Lax",
    }


def in_days(days: float) -> float:
    return (datetime.now(UTC) + timedelta(days=days)).timestamp()


def expiry_day(jar: list[dict[str, Any]]) -> str:
    """The ``espn_s2`` expiry date in a jar, as ``describe`` prints it."""
    (s2,) = (c for c in jar if c["name"] == "espn_s2")
    return datetime.fromtimestamp(s2["expires"], tz=UTC).date().isoformat()


ANONYMOUS = [cookie("SWID", SWID, expires=in_days(3650))]


def signed_in(days: float = 100) -> list[dict[str, Any]]:
    return [*ANONYMOUS, cookie("espn_s2", S2, expires=in_days(days))]


def renewed(days: float = 400) -> list[dict[str, Any]]:
    """The jar after a second sign-in: a different ``espn_s2``, as every real sign-in mints one."""
    return [*ANONYMOUS, cookie("espn_s2", RENEWED_S2, expires=in_days(days))]


@pytest.fixture(autouse=True)
def _no_real_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missed monkeypatch must fail the test, not start a headed Edge on the profile and visit ESPN."""

    def refuse() -> NoReturn:
        raise AssertionError("real browser launch in a unit test; install the `browser` fixture")

    monkeypatch.setattr(browser_session, "sync_playwright", refuse)


class Clock:
    """Fake monotonic clock for the login wait: ``sleep`` advances time instead of waiting."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class FakePage:
    def __init__(self, events: list[str], goto_error: Exception | None = None) -> None:
        self.events = events
        self.visited: list[tuple[str, dict[str, Any]]] = []
        self.goto_error = goto_error

    def goto(self, url: str, **kwargs: Any) -> None:
        self.events.append("goto")
        self.visited.append((url, kwargs))
        if self.goto_error is not None:
            raise self.goto_error


class FakeContext:
    """Scripted cookie jar plus an ordered log of ``cookies`` / ``clear`` / ``goto`` calls."""

    def __init__(self, jars: list[list[dict[str, Any]] | Exception], goto_error: Exception | None = None) -> None:
        self.jars = jars
        self.calls = 0
        self.events: list[str] = []
        self.cleared: list[str | Pattern[str] | None] = []
        self.pages = [FakePage(self.events, goto_error)]

    def new_page(self) -> FakePage:
        page = FakePage(self.events)
        self.pages.append(page)
        return page

    def cookies(self) -> list[dict[str, Any]]:
        self.events.append("cookies")
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
        self.events.append("clear")
        self.cleared.append(domain)


@dataclass
class FakeBrowser:
    context: FakeContext
    channel: str
    profile_dir: Path
    headless: bool


@dataclass
class BrowserStub:
    """Replaces ``open_browser``; records every launch request and the contexts handed out."""

    jars: list[list[dict[str, Any]] | Exception]
    error: Exception | None = None
    goto_error: Exception | None = None
    clock: Clock = field(default_factory=Clock)
    options: list[LaunchOptions | None] = field(default_factory=list)
    contexts: list[FakeContext] = field(default_factory=list)

    @contextmanager
    def open_browser(self, options: LaunchOptions | None = None) -> Iterator[FakeBrowser]:
        self.options.append(options)
        if self.error is not None:
            raise self.error
        opts = options if options is not None else LaunchOptions()
        context = FakeContext(self.jars, self.goto_error)
        self.contexts.append(context)
        yield FakeBrowser(
            context=context, channel="msedge", profile_dir=opts.resolved_profile_dir(), headless=opts.headless
        )


@pytest.fixture
def browser(monkeypatch: pytest.MonkeyPatch) -> Callable[..., BrowserStub]:
    """``browser(*jars, error=, goto_error=, profile=True)`` installs the fake; a used profile exists by default.

    The login wait runs on the stub's fake clock, so a jar that never yields a sign-in times out instantly instead
    of spinning for ``--timeout`` real seconds.
    """

    def install(
        *jars: list[dict[str, Any]] | Exception,
        error: Exception | None = None,
        goto_error: Exception | None = None,
        profile: bool = True,
    ) -> BrowserStub:
        stub = BrowserStub(list(jars) or [[]], error=error, goto_error=goto_error)
        monkeypatch.setattr(login_cmd, "open_browser", stub.open_browser)
        monkeypatch.setattr(auth, "open_browser", stub.open_browser)
        timed = functools.partial(auth.wait_for_login, clock=stub.clock.monotonic, sleep=stub.clock.sleep)
        monkeypatch.setattr(login_cmd, "wait_for_login", timed)
        if profile:
            directory = paths.browser_profile_dir()
            directory.mkdir(parents=True)
            (directory / "Local State").write_text("{}", encoding="utf-8")
        return stub

    return install


def invoke(*args: str) -> Result:
    return runner.invoke(app, ["login", *args], catch_exceptions=False)


def test_help_exits_zero() -> None:
    result = invoke("--help")
    assert result.exit_code == 0, result.output
    assert "--check" in result.output
    assert "--channel" in result.output
    assert "msedge" in result.output and "chrome" in result.output  # the enum's choices are shown
    assert "--timeout" in result.output


def test_login_opens_a_headed_window_on_espn_and_waits_for_the_cookies(browser: Callable[..., BrowserStub]) -> None:
    stub = browser(ANONYMOUS, ANONYMOUS, signed_in(100))
    result = invoke()
    assert result.exit_code == 0, result.output
    assert stub.options == [LaunchOptions(headless=False, channel=None)]
    (context,) = stub.contexts
    # Snapshot and clear the jar before the login page opens, then poll until the sign-in lands.
    assert context.events == ["cookies", "clear", "goto", "cookies", "cookies"]
    assert context.cleared == [COOKIE_DOMAIN_PATTERN]
    assert context.pages[0].visited == [(LOGIN_URL, {"wait_until": "domcontentloaded"})]
    assert stub.clock.sleeps == [login_cmd.POLL_SECONDS]
    assert "Opened msedge. Sign in to ESPN in that window" in result.output
    assert "Replacing the saved ESPN session" not in result.output
    assert "Signed in. ESPN session OK until" in result.output
    assert S2 not in result.output
    assert SWID not in result.output


def test_login_with_an_explicit_channel_and_timeout(browser: Callable[..., BrowserStub]) -> None:
    stub = browser(ANONYMOUS, signed_in())
    result = invoke("--channel", "chrome", "--timeout", "30")
    assert result.exit_code == 0, result.output
    assert stub.options == [LaunchOptions(headless=False, channel="chrome")]
    (options,) = stub.options
    assert options is not None and type(options.channel) is str  # a plain string reaches Playwright, not the enum


def test_login_rejects_an_unknown_channel_before_any_launch(browser: Callable[..., BrowserStub]) -> None:
    stub = browser(ANONYMOUS, signed_in())
    result = invoke("--channel", "firefox")
    assert result.exit_code == 2, result.output
    assert "Invalid value for '--channel'" in result.output
    assert "firefox" in result.output
    assert stub.options == []


def test_login_fails_cleanly_when_the_window_is_closed_early(browser: Callable[..., BrowserStub]) -> None:
    browser(ANONYMOUS, PlaywrightError("Target page, context or browser has been closed"))
    result = invoke()
    assert result.exit_code == 1
    assert "error: the browser window was closed before the sign-in finished" in result.output


def test_login_reports_a_timeout(browser: Callable[..., BrowserStub]) -> None:
    stub = browser(ANONYMOUS)
    result = invoke("--timeout", "60")
    assert result.exit_code == 1
    assert "error: no ESPN sign-in after 60s" in result.output
    assert stub.clock.sleeps == [login_cmd.POLL_SECONDS] * 60  # polled once a second until the deadline


def test_login_without_an_installed_browser(browser: Callable[..., BrowserStub]) -> None:
    message = "no installed browser for channels msedge, chrome; install Microsoft Edge or Google Chrome"
    browser(error=BrowserUnavailableError(message))
    result = invoke()
    assert result.exit_code == 1
    assert f"error: {message}" in result.output


def test_login_reports_a_page_that_will_not_open(browser: Callable[..., BrowserStub]) -> None:
    stub = browser(ANONYMOUS, goto_error=PlaywrightError("Page.goto: net::ERR_NAME_NOT_RESOLVED"))
    result = invoke()
    assert result.exit_code == 1
    assert f"error: could not open {LOGIN_URL}: Page.goto: net::ERR_NAME_NOT_RESOLVED" in result.output
    assert stub.contexts[0].events == ["cookies", "clear", "goto"]


def test_timeout_must_be_at_least_one_second(browser: Callable[..., BrowserStub]) -> None:
    stub = browser(ANONYMOUS, signed_in())
    result = invoke("--timeout", "0")
    assert result.exit_code == 2, result.output
    assert "--timeout" in result.output
    assert stub.options == []


# --- renewing a saved session ------------------------------------------------------------------------------------


@pytest.mark.parametrize("old_days", [3, 200], ids=["expiring-soon", "revoked-server-side-but-dated-far-out"])
def test_login_again_replaces_the_saved_session(browser: Callable[..., BrowserStub], old_days: float) -> None:
    # The stale cookie is still in the jar on the first poll after the page opens; only a new espn_s2 counts.
    old, new = signed_in(old_days), renewed(400)
    stub = browser(old, old, new)
    result = invoke()
    assert result.exit_code == 0, result.output
    (context,) = stub.contexts
    assert context.events == ["cookies", "clear", "goto", "cookies", "cookies"]
    assert context.cleared == [COOKIE_DOMAIN_PATTERN]
    assert "Replacing the saved ESPN session; sign in again to renew it." in result.output
    assert f"Signed in. ESPN session OK until {expiry_day(new)}" in result.output
    assert expiry_day(old) not in result.output
    assert "run `fm login`" not in result.output
    assert S2 not in result.output and RENEWED_S2 not in result.output


def test_login_again_times_out_when_no_new_sign_in_happens(browser: Callable[..., BrowserStub]) -> None:
    stub = browser(signed_in(200))
    result = invoke("--timeout", "60")
    assert result.exit_code == 1
    assert "Replacing the saved ESPN session" in result.output
    assert "error: no ESPN sign-in after 60s" in result.output
    assert "Signed in" not in result.output
    assert stub.contexts[0].cleared == [COOKIE_DOMAIN_PATTERN]


# --- --check -----------------------------------------------------------------------------------------------------


def test_check_reports_a_healthy_session_headless_without_visiting_espn(browser: Callable[..., BrowserStub]) -> None:
    stub = browser(signed_in(100))
    result = invoke("--check")
    assert result.exit_code == 0, result.output
    assert stub.options == [LaunchOptions(headless=True, channel=None)]
    (context,) = stub.contexts
    assert context.events == ["cookies"]  # read only: no navigation and nothing cleared
    assert context.pages[0].visited == []
    assert "ESPN session OK until" in result.output
    assert S2 not in result.output


def test_check_with_an_expiring_session_still_exits_zero(browser: Callable[..., BrowserStub]) -> None:
    browser(signed_in(3))
    result = invoke("--check")
    assert result.exit_code == 0, result.output
    assert "ESPN session expires soon" in result.output
    assert "run `fm login` to renew" in result.output


def test_check_with_an_expired_session_exits_one(browser: Callable[..., BrowserStub]) -> None:
    browser(signed_in(-1))
    result = invoke("--check")
    assert result.exit_code == 1
    assert "ESPN session expired on" in result.output


def test_check_with_an_anonymous_profile_exits_one(browser: Callable[..., BrowserStub]) -> None:
    browser(ANONYMOUS)
    result = invoke("--check")
    assert result.exit_code == 1
    assert "error: no ESPN session in the browser profile; run `fm login`" in result.output


def test_check_without_a_profile_never_launches_a_browser(browser: Callable[..., BrowserStub]) -> None:
    stub = browser(signed_in(), profile=False)
    result = invoke("--check")
    assert result.exit_code == 1
    assert "error: no browser profile yet; run `fm login`" in result.output
    assert stub.options == []
