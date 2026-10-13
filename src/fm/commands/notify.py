"""``fm notify setup``: capture the Telegram chat id, or generate the ntfy topics, into ``.env`` (DESIGN section 11).

Telegram: create a bot with @BotFather (``/newbot``) and put its token in ``.env`` as ``TELEGRAM_BOT_TOKEN``. This
command prints a one-time code and a ``t.me`` link that sends it; the private chat that sends the code is saved as
``TELEGRAM_CHAT_ID``, the only chat the bot will ever answer. ntfy: two random topic names are generated and saved
(``NTFY_TOPIC`` for the phone, ``NTFY_REPLY_TOPIC`` for the PC); subscribe to the first one in the ntfy app. Either
way a test push follows, so the command doubles as a check that the channel works. Run again with ``.env`` already
filled in, it only sends the test push; ``--replace`` captures or generates afresh. A config with ``channel = "none"``
(no phone) has nothing to set up, so the command then needs ``--channel`` to say which one to add.

The setup itself lives in :mod:`fm.notify.setup`, which the ``fm setup`` wizard shares; this command reads the current
secrets from ``config.toml``'s ``.env`` and hands them over.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, NoReturn

import typer

from fm.config import ConfigError, config_paths, load_config
from fm.notify import NotifyError
from fm.notify.setup import DEFAULT_WAIT_S, saved_token, saved_topics, setup_code, setup_ntfy, setup_telegram

app = typer.Typer(help="Phone notifications and approvals: Telegram or ntfy.", no_args_is_help=True)


class ChannelChoice(StrEnum):
    telegram = "telegram"
    ntfy = "ntfy"


@app.command("setup")
def setup(
    channel: Annotated[
        ChannelChoice | None,
        typer.Option("--channel", "-c", help="Channel to set up; default: the notify channel in config.toml."),
    ] = None,
    timeout: Annotated[
        float, typer.Option("--timeout", min=1, help="Seconds to wait for the code to reach the Telegram bot.")
    ] = DEFAULT_WAIT_S,
    replace: Annotated[
        bool, typer.Option("--replace", help="Capture the chat id or generate the topics again, replacing .env values.")
    ] = False,
) -> None:
    """Set up the phone channel, save what it needs to .env, and send a test push.

    Telegram needs TELEGRAM_BOT_TOKEN in .env first (a bot from @BotFather); then send the bot the code this prints.
    ntfy needs nothing: subscribe to the generated topic in the app.
    """
    config_path, env_path = config_paths()
    try:
        config = load_config(config_path, env_path=env_path)
    except ConfigError as exc:
        _fail(str(exc))
    name = config.notify.channel if channel is None else channel.value
    if name == "none":
        _fail(
            f'{config_path} says channel = "none" (no phone): pass --channel telegram or --channel ntfy to set one up, '
            "then set that channel under [notify]"
        )
    found = config.secrets
    try:
        if name == "telegram":
            setup_telegram(
                env_path,
                token=saved_token(found),
                chat_id=found.telegram_chat_id,
                echo=typer.echo,
                timeout_s=timeout,
                replace=replace,
                code=setup_code,  # looked up at call time, so a test can pin the code
            )
        else:
            setup_ntfy(env_path, topics=saved_topics(found), echo=typer.echo, replace=replace)
    except NotifyError as exc:
        _fail(str(exc))
    if name != config.notify.channel:
        typer.echo(f'Set channel = "{name}" under [notify] in {config_path} so fm run listens on it.')


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)


def register(root: typer.Typer) -> None:
    root.add_typer(app, name="notify")
