"""``fm bot``: listen for Approve/Reject presses from the phone and record them (DESIGN section 13).

Started at logon and left running, it long-polls Telegram or subscribes to the ntfy reply topic (no inbound port
either way) and hands every press to :func:`fm.notify.handle_reply`, which checks the single-use nonce and calls
``fm.proposals.approve`` / ``reject``. ``--once`` records the presses already waiting and exits. Ignored presses
(another chat, an unknown or used button) are logged to stderr.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from contextlib import closing, contextmanager
from typing import Annotated, NoReturn

import typer

from fm.config import ConfigError, load_config
from fm.notify import DecisionResult, NotifyError, listen, open_channel
from fm.store import Store

LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def bot(
    once: Annotated[bool, typer.Option("--once", help="Record the presses waiting now, then exit.")] = False,
) -> None:
    """Listen for Approve/Reject presses from the phone and record the decisions. Ctrl+C stops it."""
    try:
        config = load_config()
    except ConfigError as exc:
        _fail(str(exc))
    try:
        channel = open_channel(config)
    except NotifyError as exc:
        _fail(str(exc))
    with _console_log(), closing(channel), Store.open() as store:
        if not once:
            typer.echo(f"listening on {channel.name}; Ctrl+C stops")
        try:
            handled = listen(store, channel, once=once, on_result=_show)
        except KeyboardInterrupt:
            typer.echo("stopped")
            return
        except NotifyError as exc:
            _fail(str(exc))
    typer.echo(f"recorded {handled} decision{'' if handled == 1 else 's'}")


def _show(result: DecisionResult) -> None:
    typer.echo(result.describe())


@contextmanager
def _console_log() -> Iterator[None]:
    """Warnings from the listener (ignored presses, channel errors) go to stderr while the command runs."""
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.WARNING)
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logger = logging.getLogger("fm.notify")
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)


def register(root: typer.Typer) -> None:
    root.command("bot")(bot)
