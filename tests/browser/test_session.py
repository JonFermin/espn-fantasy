"""Persistent profile launch: profile location, channel fallback, launch errors, and the open/close wrapper.

No browser is ever started: a fake launcher stands in for ``playwright.sync_api.BrowserType``.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from playwright.sync_api import Error as PlaywrightError

from fm.browser import session as browser_session
from fm.browser.session import (
    CHANNELS,
    BrowserLaunchError,
    BrowserUnavailableError,
    Channel,
    LaunchOptions,
    is_missing_channel,
    launch_persistent,
    open_browser,
    profile_exists,
)

# The Playwright driver's wording when a channel's executable is absent (playwright 1.63, server registry).
NOT_FOUND = (
    "BrowserType.launch_persistent_context: Chromium distribution '{channel}' is not found at "
    'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe\nRun "playwright install {channel}"'
)
IN_USE = "BrowserType.launch_persistent_context: Failed to launch: the profile appears to be in use"


class FakeContext:
    def __init__(self) -> None:
        self.closed = 0
        self.close_error: PlaywrightError | None = None

    def close(self) -> None:
        self.closed += 1
        if self.close_error is not None:
            raise self.close_error


@dataclass
class FakeLauncher:
    """``installed`` channels launch; ``failures`` raise their message; anything else is "not found"."""

    installed: set[str]
    failures: dict[str, str] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)
    created: list[FakeContext] = field(default_factory=list)

    def launch_persistent_context(
        self,
        user_data_dir: str | Path,
        *,
        channel: str | None = None,
        headless: bool | None = None,
        timeout: float | None = None,
    ) -> FakeContext:
        self.calls.append(
            {"user_data_dir": user_data_dir, "channel": channel, "headless": headless, "timeout": timeout}
        )
        if channel in self.failures:
            raise PlaywrightError(self.failures[channel])
        if channel not in self.installed:
            raise PlaywrightError(NOT_FOUND.format(channel=channel))
        context = FakeContext()
        self.created.append(context)
        return context

    def channels_tried(self) -> list[str | None]:
        return [call["channel"] for call in self.calls]


def test_default_profile_dir_is_the_config_dir_profile(tmp_path: Path) -> None:
    assert LaunchOptions().resolved_profile_dir() == tmp_path / "config" / "browser-profile"
    assert LaunchOptions(profile_dir=tmp_path / "elsewhere").resolved_profile_dir() == tmp_path / "elsewhere"


def test_defaults_are_headless_and_try_edge_then_chrome() -> None:
    options = LaunchOptions()
    assert options.headless is True
    assert options.channels() == CHANNELS == ("msedge", "chrome")
    assert LaunchOptions(channel="chrome").channels() == ("chrome",)


def test_channel_enum_is_the_single_source_of_the_fallback_order() -> None:
    assert [member.value for member in Channel] == ["msedge", "chrome"]
    assert CHANNELS == tuple(member.value for member in Channel)
    assert all(type(name) is str for name in CHANNELS)  # plain strings go to Playwright, not enum members
    assert Channel("chrome") is Channel.CHROME
    with pytest.raises(ValueError):
        Channel("firefox")


def test_launch_passes_profile_headless_and_timeout(tmp_path: Path) -> None:
    launcher = FakeLauncher(installed={"msedge"})
    context, channel = launch_persistent(launcher, LaunchOptions())
    assert isinstance(context, FakeContext)
    assert channel == "msedge"
    profile = tmp_path / "config" / "browser-profile"
    assert launcher.calls == [{"user_data_dir": profile, "channel": "msedge", "headless": True, "timeout": 30_000}]
    assert profile.is_dir()


def test_headed_launch_for_the_manual_sign_in() -> None:
    launcher = FakeLauncher(installed={"msedge"})
    launch_persistent(launcher, LaunchOptions(headless=False))
    assert launcher.calls[0]["headless"] is False


def test_falls_back_to_chrome_when_edge_is_not_installed() -> None:
    launcher = FakeLauncher(installed={"chrome"})
    _, channel = launch_persistent(launcher, LaunchOptions())
    assert channel == "chrome"
    assert launcher.channels_tried() == ["msedge", "chrome"]


def test_no_installed_browser_is_a_clear_error() -> None:
    launcher = FakeLauncher(installed=set())
    with pytest.raises(BrowserUnavailableError, match=r"msedge, chrome; install Microsoft Edge or Google Chrome"):
        launch_persistent(launcher, LaunchOptions())
    assert launcher.channels_tried() == ["msedge", "chrome"]


def test_explicit_channel_is_the_only_one_tried() -> None:
    launcher = FakeLauncher(installed={"msedge"})
    with pytest.raises(BrowserUnavailableError, match="channels chrome;"):
        launch_persistent(launcher, LaunchOptions(channel="chrome"))
    assert launcher.channels_tried() == ["chrome"]


def test_other_launch_failures_do_not_fall_through_to_the_next_channel(tmp_path: Path) -> None:
    launcher = FakeLauncher(installed={"msedge", "chrome"}, failures={"msedge": IN_USE})
    expected = r"msedge failed to launch with profile .*appears to be in use"
    with pytest.raises(BrowserLaunchError, match=expected) as info:
        launch_persistent(launcher, LaunchOptions(profile_dir=tmp_path / "profile"))
    assert isinstance(info.value.__cause__, PlaywrightError)
    assert launcher.channels_tried() == ["msedge"]


@pytest.mark.parametrize(
    ("message", "missing"),
    [
        (NOT_FOUND.format(channel="msedge"), True),
        ("Chromium distribution 'msedge' is not supported on freebsd", True),
        ("Executable doesn't exist at C:\\x\\chrome.exe", True),
        (IN_USE, False),
        ("BrowserType.launch_persistent_context: Timeout 30000ms exceeded.", False),
    ],
)
def test_is_missing_channel(message: str, missing: bool) -> None:
    assert is_missing_channel(message) is missing


def test_profile_exists_only_after_a_browser_wrote_into_it(tmp_path: Path) -> None:
    profile = tmp_path / "config" / "browser-profile"
    assert profile_exists() is False
    profile.mkdir(parents=True)
    assert profile_exists() is False  # created but never launched
    (profile / "Local State").write_text("{}", encoding="utf-8")
    assert profile_exists() is True
    assert profile_exists(tmp_path / "elsewhere") is False


@contextmanager
def _fake_playwright(launcher: FakeLauncher) -> Iterator[SimpleNamespace]:
    yield SimpleNamespace(chromium=launcher)


def test_open_browser_yields_the_session_and_closes_the_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    launcher = FakeLauncher(installed={"chrome"})
    monkeypatch.setattr(browser_session, "sync_playwright", lambda: _fake_playwright(launcher))
    with open_browser() as browser:
        context = browser.context
        assert isinstance(context, FakeContext)
        assert browser.channel == "chrome"
        assert browser.headless is True
        assert browser.profile_dir == tmp_path / "config" / "browser-profile"
        assert context.closed == 0
    assert context.closed == 1


def test_open_browser_closes_the_context_when_the_block_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = FakeLauncher(installed={"msedge"})
    monkeypatch.setattr(browser_session, "sync_playwright", lambda: _fake_playwright(launcher))
    with pytest.raises(KeyError), open_browser(LaunchOptions(headless=False)):
        raise KeyError("boom")
    (context,) = launcher.created
    assert context.closed == 1


def test_open_browser_tolerates_a_window_the_user_already_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    launcher = FakeLauncher(installed={"msedge"})
    monkeypatch.setattr(browser_session, "sync_playwright", lambda: _fake_playwright(launcher))
    with open_browser(LaunchOptions(headless=False)) as browser:
        context = browser.context
        assert isinstance(context, FakeContext)
        context.close_error = PlaywrightError("Target page, context or browser has been closed")
    assert context.closed == 1
