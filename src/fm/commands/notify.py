"""``fm notify setup``: capture the Telegram chat id, or generate the ntfy topics, into ``.env`` (DESIGN section 11).

Telegram: create a bot with @BotFather (``/newbot``) and put its token in ``.env`` as ``TELEGRAM_BOT_TOKEN``. This
command prints a one-time code and a ``t.me`` link that sends it; the private chat that sends the code is saved as
``TELEGRAM_CHAT_ID``, the only chat the bot will ever answer. ntfy: two random topic names are generated and saved
(``NTFY_TOPIC`` for the phone, ``NTFY_REPLY_TOPIC`` for the PC); subscribe to the first one in the ntfy app. Either
way a test push follows, so the command doubles as a check that the channel works. Run again with ``.env`` already
filled in, it only sends the test push; ``--replace`` captures or generates afresh.
"""

from __future__ import annotations

import secrets
from enum import StrEnum
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from dotenv import set_key

from fm import paths
from fm.config import Config, ConfigError, config_paths, load_config
from fm.notify import Message, NotifyError, NtfyChannel, TelegramChannel, generate_topic
from fm.notify.ntfy import DEFAULT_SERVER, REPLY_TOPIC_ENV, SINCE_FILE, TOPIC_ENV
from fm.notify.telegram import CHAT_ID_ENV, TOKEN_ENV

app = typer.Typer(help="Phone notifications and approvals: Telegram or ntfy.", no_args_is_help=True)

DEFAULT_WAIT_S = 180.0
TEST_MESSAGE = Message(
    title="espn-fantasy: notifications are set up",
    body="Proposals arrive here with Approve and Reject buttons; fm bot records what you press.",
)


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
    try:
        if name == "telegram":
            _setup_telegram(config, env_path, timeout, replace)
        else:
            _setup_ntfy(config, env_path, replace)
    except NotifyError as exc:
        _fail(str(exc))
    if name != config.notify.channel:
        typer.echo(f'Set channel = "{name}" under [notify] in {config_path} so fm bot listens on it.')


def setup_code() -> str:
    """The one-time code the Telegram chat must send: six random digits."""
    return f"{secrets.randbelow(1_000_000):06d}"


def _setup_telegram(config: Config, env_path: Path, timeout_s: float, replace: bool) -> None:
    token = config.secrets.telegram_bot_token
    if token is None:
        raise NotifyError(
            f"{TOKEN_ENV} is not set in {env_path}: create a bot with @BotFather (/newbot) and put its token there"
        )
    chat_id = None if replace else config.secrets.telegram_chat_id
    with TelegramChannel(token.get_secret_value(), chat_id) as telegram:
        username = telegram.username()
        typer.echo(f"Bot token works: @{username}.")
        if telegram.chat_id is None:
            code = setup_code()
            typer.echo(f"On your phone, open https://t.me/{username}?start={code} and tap Start,")
            typer.echo(f"or send @{username} the code {code}. Waiting {timeout_s:.0f}s...")
            capture = telegram.wait_for_chat(code, timeout_s=timeout_s)
            if capture is None:
                raise NotifyError("the code did not reach the bot in time; run fm notify setup again")
            telegram.chat_id = capture.chat_id
            set_key(env_path, CHAT_ID_ENV, str(capture.chat_id), quote_mode="never")
            who = f" with {capture.name}" if capture.name else ""
            typer.echo(f"Saved {CHAT_ID_ENV} (your private chat{who}) to {env_path}.")
        else:
            typer.echo(f"{CHAT_ID_ENV} is already set; --replace captures it again.")
        telegram.send(TEST_MESSAGE)
    typer.echo("Sent a test message to that chat. Run fm bot to record what you press.")


def _setup_ntfy(config: Config, env_path: Path, replace: bool) -> None:
    topic, reply_topic = config.secrets.ntfy_topic, config.secrets.ntfy_reply_topic
    if replace or topic is None or reply_topic is None:
        names = (generate_topic(), generate_topic())
        set_key(env_path, TOPIC_ENV, names[0], quote_mode="never")
        set_key(env_path, REPLY_TOPIC_ENV, names[1], quote_mode="never")
        (paths.config_dir() / SINCE_FILE).unlink(missing_ok=True)  # it points into the old reply topic
        typer.echo(f"Generated {TOPIC_ENV} and {REPLY_TOPIC_ENV} and saved them to {env_path}.")
    else:
        names = (topic.get_secret_value(), reply_topic.get_secret_value())
        typer.echo(f"{TOPIC_ENV} and {REPLY_TOPIC_ENV} are already set; --replace generates new ones.")
    typer.echo(f"In the ntfy app, subscribe to this topic on {DEFAULT_SERVER} (the name is the secret; share it with")
    typer.echo("nobody, and never subscribe to the reply topic, which is for the PC):")
    typer.echo(f"  {names[0]}")
    with NtfyChannel(*names) as ntfy:
        ntfy.send(TEST_MESSAGE)
    typer.echo("Sent a test notification (subscribed later? run fm notify setup again for a fresh one).")
    typer.echo("Run fm bot to record what you press.")


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)


def register(root: typer.Typer) -> None:
    root.add_typer(app, name="notify")
