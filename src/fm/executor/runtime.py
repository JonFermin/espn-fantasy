"""What an execution runs against: the read client, the write transport and the browser (DESIGN 6.2, 6.3).

:class:`Runtime` bundles the three, plus the account's SWID for the transaction envelope. The executor never opens one
itself: it is handed a :class:`RuntimeOpener` and calls it only after the cheap refusals (pause, proposal state, token,
flow lookup) have passed, so a refused run never starts a browser.

:func:`live_opener` is the real thing: it opens the persistent browser profile (``fm.browser.session``), harvests the
ESPN session from it (``fm.espn.auth``), builds an :class:`fm.espn.client.EspnClient` with those cookies for the reads,
and a :class:`fm.executor.transport.PlaywrightTransport` over the same browser context for the writes. A missing or
expired session is refused before any read. For a dry run the transport refuses every send and the context aborts
every request to ESPN's write host, so not even a UI walk that clicked too far could reach it. Tests hand the executor
the fakes in ``fm.browser.fakes`` instead.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from playwright.sync_api import BrowserContext, Page, Route

from fm.browser.flows import WRITES_HOST, PageLike, WriteTransport
from fm.browser.session import LaunchOptions, open_browser, profile_exists
from fm.espn.auth import AuthError, NotLoggedInError, SessionStatus, describe, harvest_session
from fm.espn.client import EspnClient
from fm.espn.ids import Game
from fm.executor.transport import PlaywrightTransport, RefusingTransport
from fm.store import LeagueRow

logger = logging.getLogger(__name__)

WRITES_ROUTE = f"https://{WRITES_HOST}/**"
"""Every URL on ESPN's write host, as a Playwright route pattern; a dry run aborts all of them."""


class BrowserLike(Protocol):
    """The browser surface UI mode needs: fresh pages and a trace. ``fm.browser.fakes.FakeBrowser`` has this shape."""

    def new_page(self) -> PageLike: ...
    def start_trace(self) -> None: ...
    def stop_trace(self, path: Path) -> None: ...


@dataclass(frozen=True, slots=True)
class Runtime:
    """One league's execution environment."""

    reader: EspnClient
    """Reads for preconditions and verification."""
    transport: WriteTransport
    """API-mode writes. The executor sends at most one request per attempt through it."""
    browser: BrowserLike
    """UI mode: pages for the click-through and the Playwright trace."""
    member_id: str | None = None
    """The account's SWID, the envelope's ``memberId``."""


class RuntimeOpener(Protocol):
    """Opens a :class:`Runtime` for a league; the executor calls it once per run."""

    def __call__(self, league: LeagueRow, *, dry_run: bool) -> AbstractContextManager[Runtime]: ...


class LiveBrowser:
    """:class:`BrowserLike` over a Playwright browser context."""

    def __init__(self, context: BrowserContext) -> None:
        self.context = context

    def new_page(self) -> Page:
        return self.context.new_page()

    def start_trace(self) -> None:
        self.context.tracing.start(screenshots=True, snapshots=True)

    def stop_trace(self, path: Path) -> None:
        self.context.tracing.stop(path=path)


def _abort_write(route: Route) -> None:
    logger.warning("executor: dry run aborted a request to %s", route.request.url)
    route.abort()


@contextmanager
def open_live_runtime(league: LeagueRow, *, dry_run: bool, launch: LaunchOptions | None = None) -> Iterator[Runtime]:
    """Open the browser profile and the ESPN session for ``league`` for the duration of the block.

    Raises ``NotLoggedInError`` without starting a browser when there is no profile yet, ``AuthError`` when the saved
    session has expired, and ``fm.browser.session.BrowserError`` when the browser cannot start.
    """
    options = launch if launch is not None else LaunchOptions()
    if not profile_exists(options.resolved_profile_dir()):
        raise NotLoggedInError("no browser profile yet; run `fm login`")
    with open_browser(options) as browser:
        session = harvest_session(browser.context)
        status = session.status()
        if status is SessionStatus.EXPIRED:
            raise AuthError(describe(session))
        if status is SessionStatus.EXPIRING:
            logger.warning("executor: %s", describe(session))
        if dry_run:
            browser.context.route(WRITES_ROUTE, _abort_write)
            transport: WriteTransport = RefusingTransport("dry run")
        else:
            transport = PlaywrightTransport(browser.context.request)
        game = Game.from_sport(league.sport)
        with EspnClient(game, league.espn_league_id, league.season, session) as reader:
            yield Runtime(
                reader=reader, transport=transport, browser=LiveBrowser(browser.context), member_id=session.swid
            )


def live_opener(launch: LaunchOptions | None = None) -> RuntimeOpener:
    """A :class:`RuntimeOpener` over the real browser profile; ``launch`` picks headless/headed and the channel."""

    def opener(league: LeagueRow, *, dry_run: bool) -> AbstractContextManager[Runtime]:
        return open_live_runtime(league, dry_run=dry_run, launch=launch)

    return opener
