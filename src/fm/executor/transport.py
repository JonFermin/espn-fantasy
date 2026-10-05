"""API-mode write transports and the check every write request passes first (DESIGN 6.3 write safety).

:class:`PlaywrightTransport` sends a :class:`fm.browser.flows.WriteRequest` from inside the logged-in browser session
(``BrowserContext.request``: the same cookies and client as the web app) exactly once: no retries, no redirects, and a
hard timeout over the whole exchange. A timeout becomes :class:`fm.browser.flows.WriteTimeoutError` and any other
transport failure :class:`fm.browser.flows.WriteUncertainError`, because the request may have reached ESPN either way;
the executor then marks the attempt ``unknown`` and re-reads. Every HTTP answer, error statuses included, comes back
as a :class:`fm.browser.flows.WriteResponse`.

:class:`RefusingTransport` is what a dry run holds: it refuses every send. :func:`check_write_request` is the
executor's gate in front of any transport: a POST of a JSON object to this league's endpoint on ESPN's write host, as
our own team, never as league manager, with no credentials in the headers.
"""

from __future__ import annotations

import contextlib
import json
import time
from collections.abc import Mapping
from urllib.parse import urlsplit

from playwright.sync_api import APIRequestContext
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError

from fm.browser.flows import (
    CREDENTIAL_HEADERS,
    WRITES_HOST,
    WriteRefusedError,
    WriteRequest,
    WriteResponse,
    WriteTimeoutError,
    WriteUncertainError,
    league_write_root,
)
from fm.espn.ids import Game
from fm.store import LeagueRow


def check_write_request(request: WriteRequest, league: LeagueRow) -> None:
    """Refuse (``WriteRefusedError``) a request that is not a POST of a JSON object to ``league``'s endpoint on ESPN's
    write host, acting as our team with ``isLeagueManager: false`` and without credential headers. Lists every
    problem at once."""
    problems: list[str] = []
    if request.method.upper() != "POST":
        problems.append(f"writes are POSTs, not {request.method}")
    parts = urlsplit(request.url)
    root = league_write_root(Game.from_sport(league.sport), league.season, league.espn_league_id)
    if parts.scheme != "https" or parts.hostname != WRITES_HOST:
        problems.append(f"writes go only to https://{WRITES_HOST}, not {parts.scheme}://{parts.hostname}")
    elif not (request.url == root or request.url.startswith(root + "/")):
        problems.append(
            f"{request.url} is not ESPN {league.sport} league {league.espn_league_id} season {league.season}"
        )
    credentials = sorted(name for name in request.headers if name.lower() in CREDENTIAL_HEADERS)
    if credentials:
        problems.append(f"credential headers {', '.join(credentials)}: the browser session supplies those")
    body = request.body
    if not isinstance(body, Mapping):
        problems.append("the body must be a JSON object")
    else:
        if body.get("isLeagueManager") is not False:
            problems.append("isLeagueManager must be false: league-manager powers are never used")
        team = body.get("teamId")
        if team != league.team_id or isinstance(team, bool):
            problems.append(f"teamId must be our team {league.team_id}, not {team!r}")
        try:
            json.dumps(body, allow_nan=False)
        except (TypeError, ValueError) as exc:
            problems.append(f"the body is not plain JSON ({exc})")
    if problems:
        raise WriteRefusedError("write refused: " + "; ".join(problems))


class PlaywrightTransport:
    """Sends API-mode writes through a browser context's request client (``BrowserContext.request``)."""

    def __init__(self, requests: APIRequestContext) -> None:
        self._requests = requests

    def send(self, request: WriteRequest, *, timeout_s: float) -> WriteResponse:
        """POST once with a hard timeout; never retries and never follows a redirect."""
        if timeout_s <= 0:
            raise WriteRefusedError(f"a write needs a positive timeout, got {timeout_s!r}")
        headers = {"Content-Type": "application/json", "Accept": "application/json", **dict(request.headers)}
        started = time.monotonic()
        try:
            response = self._requests.post(
                request.url,
                data=json.dumps(request.body, separators=(",", ":")),
                headers=headers,
                timeout=timeout_s * 1000,
                fail_on_status_code=False,
                max_redirects=0,
                max_retries=0,
            )
        except PlaywrightTimeoutError as exc:
            raise WriteTimeoutError(f"no answer within {timeout_s:g} s: {exc.message}") from exc
        except PlaywrightError as exc:
            raise WriteUncertainError(f"the request may have reached ESPN: {exc.message}") from exc
        elapsed = round(time.monotonic() - started, 3)
        try:
            text = response.text()
        except PlaywrightError:
            text = ""  # the status already answers the question; the body is for the audit trail
        status = response.status
        with contextlib.suppress(PlaywrightError):
            response.dispose()
        return WriteResponse.from_text(status, text, elapsed_s=elapsed)


class RefusingTransport:
    """A transport that sends nothing: what a dry run holds, so a slip can never reach ESPN."""

    def __init__(self, reason: str = "dry run") -> None:
        self.reason = reason

    def send(self, request: WriteRequest, *, timeout_s: float) -> WriteResponse:
        raise WriteRefusedError(f"{self.reason}: nothing is sent to {request.url}")
