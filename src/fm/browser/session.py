"""Persistent Playwright browser session over the installed Edge or Chrome (DESIGN section 6.2).

The ESPN login lives in a dedicated Chromium profile under the config dir (``fm.paths.browser_profile_dir``). ``fm
login`` fills it once by hand; every later run reuses it, headless by default. The browser is the installed Microsoft
Edge or Google Chrome (``channel="msedge"`` / ``"chrome"``), never Playwright's bundled Chromium: a real browser build
on a real profile is what keeps the site from challenging the session.

This module knows nothing about ESPN cookies or URLs. ``fm.espn.auth`` harvests cookies from the context opened here,
and the executor flows (later tasks) send their requests through it. Only one process can hold a Chromium profile at
a time, so a launch while another ``fm`` process has it open fails with ``BrowserLaunchError``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Protocol

from playwright.sync_api import BrowserContext, sync_playwright
from playwright.sync_api import Error as PlaywrightError

from fm import paths


class Channel(StrEnum):
    """Installed browsers Playwright can drive as a channel, in fallback order. Edge first: it ships with
    Windows. On macOS either may be absent; a channel that is not installed is skipped.

    The CLI types its ``--channel`` option with this enum so an unknown browser is rejected before any launch.
    """

    MSEDGE = "msedge"
    CHROME = "chrome"


CHANNELS: tuple[str, ...] = tuple(channel.value for channel in Channel)
"""The channel names to try when none is given, in ``Channel`` order."""

DEFAULT_LAUNCH_TIMEOUT_S = 30.0

MISSING_CHANNEL_MARKERS: tuple[str, ...] = ("is not found", "is not supported on", "Executable doesn't exist")
"""Fragments of the Playwright driver's message when a channel's executable is absent (vs. a real launch failure)."""


class BrowserError(RuntimeError):
    """The browser could not be started."""


class BrowserUnavailableError(BrowserError):
    """None of the requested channels is installed."""


class BrowserLaunchError(BrowserError):
    """An installed channel failed to launch: profile held by another process, crash, or timeout."""


@dataclass(frozen=True)
class LaunchOptions:
    """How to open the persistent profile. Defaults: headless, first installed channel, the config-dir profile."""

    headless: bool = True
    channel: str | None = None
    profile_dir: Path | None = None
    timeout_s: float = DEFAULT_LAUNCH_TIMEOUT_S

    def channels(self) -> tuple[str, ...]:
        """The channels to try: the explicit one, or ``CHANNELS`` in order."""
        return (self.channel,) if self.channel else CHANNELS

    def resolved_profile_dir(self) -> Path:
        """The profile directory: ``profile_dir`` or ``fm.paths.browser_profile_dir()``."""
        return self.profile_dir if self.profile_dir is not None else paths.browser_profile_dir()


class PersistentLauncher[ContextT](Protocol):
    """The slice of ``playwright.sync_api.BrowserType`` this module uses. Tests pass a fake."""

    def launch_persistent_context(
        self,
        user_data_dir: str | Path,
        *,
        channel: str | None = None,
        headless: bool | None = None,
        timeout: float | None = None,
    ) -> ContextT: ...


@dataclass(frozen=True)
class BrowserSession:
    """An open persistent context plus how it was opened."""

    context: BrowserContext
    channel: str
    profile_dir: Path
    headless: bool


def is_missing_channel(message: str) -> bool:
    """True when a Playwright launch error means the channel is not installed (so the next channel is worth trying)."""
    return any(marker in message for marker in MISSING_CHANNEL_MARKERS)


def profile_exists(profile_dir: Path | None = None) -> bool:
    """True once a browser has written into the profile, i.e. after the first ``fm login``."""
    path = profile_dir if profile_dir is not None else paths.browser_profile_dir()
    return path.is_dir() and any(path.iterdir())


def launch_persistent[ContextT](launcher: PersistentLauncher[ContextT], options: LaunchOptions) -> tuple[ContextT, str]:
    """Open the persistent profile on the first installed channel. Returns ``(context, channel)``.

    A channel that is not installed is skipped; any other launch failure is raised at once as ``BrowserLaunchError``,
    because retrying on another channel would not help and could mask a profile that is already in use.
    """
    profile = options.resolved_profile_dir()
    profile.mkdir(parents=True, exist_ok=True)
    tried: list[str] = []
    for channel in options.channels():
        tried.append(channel)
        try:
            context = launcher.launch_persistent_context(
                profile, channel=channel, headless=options.headless, timeout=options.timeout_s * 1000
            )
        except PlaywrightError as exc:
            if is_missing_channel(exc.message):
                continue
            raise BrowserLaunchError(f"{channel} failed to launch with profile {profile}: {exc.message}") from exc
        return context, channel
    raise BrowserUnavailableError(
        f"no installed browser for channels {', '.join(tried)}; install Microsoft Edge or Google Chrome"
    )


@contextmanager
def open_browser(options: LaunchOptions | None = None) -> Iterator[BrowserSession]:
    """Open the persistent profile for the duration of the block and close it afterwards."""
    opts = options if options is not None else LaunchOptions()
    with sync_playwright() as playwright:
        context, channel = launch_persistent(playwright.chromium, opts)
        try:
            yield BrowserSession(
                context=context, channel=channel, profile_dir=opts.resolved_profile_dir(), headless=opts.headless
            )
        finally:
            # The user may have closed a headed window already; closing twice is not an error worth surfacing.
            with contextlib.suppress(PlaywrightError):
                context.close()
