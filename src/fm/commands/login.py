"""``fm login``: sign in to ESPN by hand in the persistent browser profile; ``--check`` reports the saved session.

The sign-in itself is manual on purpose (DESIGN section 6.2): a headed window opens on ESPN's login page, you complete
the Disney sign-in including any one-time code, and the command closes the window as soon as ``espn_s2`` and ``SWID``
land in the profile's cookie jar. Nothing ESPN-related is written anywhere but that profile.

Running it again renews the session: ESPN's cookies are forgotten before the page opens so the form shows, and the
command only returns once a sign-in has minted an ``espn_s2`` different from the one it forgot. That is also the way
out of a session the server has revoked while its cookie still carries a future date (a 401 on every API call).
"""

from __future__ import annotations

from typing import Annotated, NoReturn

import typer
from playwright.sync_api import Error as PlaywrightError

from fm.browser.session import BrowserError, Channel, LaunchOptions, open_browser
from fm.espn.auth import (
    DEFAULT_LOGIN_TIMEOUT_S,
    DEFAULT_POLL_S,
    LOGIN_URL,
    AuthError,
    SessionStatus,
    clear_session,
    describe,
    load_session,
    wait_for_login,
)

POLL_SECONDS = DEFAULT_POLL_S


def login(
    check: Annotated[
        bool,
        typer.Option(
            "--check", help="Only report the saved session (no window opens); exit 1 when it is missing or expired."
        ),
    ] = False,
    channel: Annotated[
        Channel | None, typer.Option(help="Browser to drive. Default: the first one installed, Edge before Chrome.")
    ] = None,
    timeout: Annotated[
        float, typer.Option(min=1, help="Seconds to wait for the sign-in to finish.")
    ] = DEFAULT_LOGIN_TIMEOUT_S,
) -> None:
    """Open a browser window for a manual ESPN sign-in; run it again to renew. The session persists in the profile."""
    options = LaunchOptions(headless=check, channel=None if channel is None else channel.value)
    try:
        if check:
            _check(options)
        else:
            _login(options, timeout)
    except (AuthError, BrowserError) as exc:
        _fail(str(exc))


def _login(options: LaunchOptions, timeout_s: float) -> None:
    with open_browser(options) as browser:
        previous = clear_session(browser.context)
        if previous is not None:
            typer.echo("Replacing the saved ESPN session; sign in again to renew it.")
        typer.echo(
            f"Opened {browser.channel}. Sign in to ESPN in that window (click Log In if the page does not ask); "
            "it closes on its own once the session is saved."
        )
        page = browser.context.pages[0] if browser.context.pages else browser.context.new_page()
        try:
            page.goto(LOGIN_URL, wait_until="domcontentloaded")
        except PlaywrightError as exc:
            raise BrowserError(f"could not open {LOGIN_URL}: {exc.message}") from exc
        session = wait_for_login(browser.context, previous=previous, timeout_s=timeout_s, poll_s=POLL_SECONDS)
    typer.echo(f"Signed in. {describe(session)}")


def _check(options: LaunchOptions) -> None:
    session = load_session(options)
    typer.echo(describe(session))
    if session.status() is SessionStatus.EXPIRED:
        raise typer.Exit(1)


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)


def register(root: typer.Typer) -> None:
    root.command("login")(login)
