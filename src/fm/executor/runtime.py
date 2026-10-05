"""What an execution runs against: the read client, the write transport and the browser (DESIGN 6.2, 6.3).

:class:`Runtime` bundles the three, plus the account's SWID for the transaction envelope. The executor never opens one
itself: it is handed a :class:`RuntimeOpener` and calls it only after the cheap refusals (pause, proposal state, token,
flow lookup) have passed, so a refused run never starts a browser.

:func:`live_opener` is the real thing: it opens the persistent browser profile (``fm.browser.session``), harvests the
ESPN session from it (``fm.espn.auth``), builds an :class:`fm.espn.client.EspnClient` with those cookies for the reads,
and a :class:`fm.executor.transport.PlaywrightTransport` over the same browser context for the writes. A missing or
expired session is refused before any read. For a dry run the transport refuses every send, and the context routes
every request through :func:`dry_run_block_reason`, aborting anything sent to ESPN's write host and every request
other than a GET to ``espn.com`` or its subdomains (league messages on ``lm-api-communication`` included), so not even
a UI walk that clicked past ``UiDriver.confirm`` could save anything. Tests hand the executor the fakes in
``fm.browser.fakes`` instead.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
from urllib.parse import urlsplit

from playwright.sync_api import BrowserContext, Page, Route

from fm.browser.flows import WRITES_HOST, PageLike, WriteTransport
from fm.browser.session import LaunchOptions, open_browser, profile_exists
from fm.espn.auth import AuthError, NotLoggedInError, SessionStatus, describe, harvest_session
from fm.espn.client import EspnClient
from fm.espn.ids import Game
from fm.executor.transport import PlaywrightTransport, RefusingTransport
from fm.store import LeagueRow

logger = logging.getLogger(__name__)

ESPN_DOMAIN = "espn.com"
DRY_RUN_ROUTE = "**/*"
"""A dry run routes every request of the browser context through :func:`dry_run_block_reason`."""
DRY_RUN_ABORT = "blockedbyclient"
"""The network error a request the dry run aborted fails with in the page."""


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


def dry_run_block_reason(method: str, url: str) -> str | None:
    """Why a dry run aborts this request, or ``None`` when it may go out.

    Aborted: anything sent to ESPN's write host (:data:`fm.browser.flows.WRITES_HOST`), and every request other than a
    GET to ``espn.com`` or a subdomain of it, which covers every league API (transactions, league messages on
    ``lm-api-communication``) and the web app itself. This is the capture's write guard (``scripts/capture/guard.py``)
    without its ``transactions`` URL marker, which only adds GETs. Page loads, API reads and the Disney sign-in
    (``registerdisney.go.com``) go through.
    """
    host = (urlsplit(url).hostname or "").lower().rstrip(".")
    if host == WRITES_HOST:
        return "ESPN's write host"
    verb = method.upper()
    if verb != "GET" and (host == ESPN_DOMAIN or host.endswith(f".{ESPN_DOMAIN}")):
        return f"a {verb} to an {ESPN_DOMAIN} host"
    return None


def _dry_run_route(route: Route) -> None:
    """The dry run's route over every request: abort what :func:`dry_run_block_reason` names, let the rest through. A
    request it cannot classify is aborted too."""
    try:
        request = route.request
        method, url = request.method, request.url
        reason = dry_run_block_reason(method, url)
    except Exception as exc:  # fail closed
        logger.warning("executor: dry run aborted a request it could not classify: %r", exc)
        route.abort(DRY_RUN_ABORT)
        return
    if reason is None:
        route.continue_()
        return
    logger.warning("executor: dry run aborted %s %s (%s)", method, url, reason)
    route.abort(DRY_RUN_ABORT)


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
            browser.context.route(DRY_RUN_ROUTE, _dry_run_route)
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
