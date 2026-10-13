"""Phone channel setup, shared by ``fm notify setup`` and the ``fm setup`` wizard (PLUG_AND_PLAY section 6).

Telegram: a bot from @BotFather (``/newbot``) whose token is ``TELEGRAM_BOT_TOKEN``. :func:`setup_telegram` proves
the token with ``getMe``, prints a one-time code and a ``t.me`` link that sends it, and saves the private chat that
sends the code as ``TELEGRAM_CHAT_ID``, the only chat the bot will ever answer. ntfy: :func:`setup_ntfy` generates two
random topic names (``NTFY_TOPIC`` for the phone, ``NTFY_REPLY_TOPIC`` for the PC) and tells you which one to subscribe
to. Either way a test push follows, so setting up doubles as a check that the channel works, and a channel already
set up only gets the test push unless ``replace`` asks for a fresh capture or fresh topics.

Both functions take the current secrets as arguments rather than a loaded ``Config``, because the wizard sets the
phone up before ``config.toml`` exists, and their console output goes through an injected ``echo`` so the wizard and
the command print the same lines. Values are written to ``.env`` with :func:`dotenv.set_key` and never printed, except
the phone topic, which the user has to type into the ntfy app.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from pathlib import Path

import httpx
from dotenv import set_key

from fm import paths
from fm.config import Secrets
from fm.notify.base import Message, NotifyError
from fm.notify.ntfy import DEFAULT_SERVER, REPLY_TOPIC_ENV, SINCE_FILE, TOPIC_ENV, NtfyChannel, generate_topic
from fm.notify.telegram import CHAT_ID_ENV, TOKEN_ENV, TelegramChannel

DEFAULT_WAIT_S = 180.0
"""How long :func:`setup_telegram` waits for the code to reach the bot."""
TEST_MESSAGE = Message(
    title="espn-fantasy: notifications are set up",
    body="Proposals arrive here with Approve and Reject buttons; fm run records what you press.",
)
BOTFATHER_HINT = "create a bot with @BotFather (/newbot)"
"""How to get a ``TELEGRAM_BOT_TOKEN``; ``fm notify setup`` says it when the token is missing."""

Echo = Callable[[str], object]
"""Where a setup step's console lines go: ``typer.echo`` in the CLI, a list's ``append`` in tests."""


def setup_code() -> str:
    """The one-time code the Telegram chat must send: six random digits."""
    return f"{secrets.randbelow(1_000_000):06d}"


def saved_token(found: Secrets) -> str | None:
    """The ``TELEGRAM_BOT_TOKEN`` in ``found`` (the secrets ``.env`` holds now), or ``None``."""
    token = found.telegram_bot_token
    return None if token is None else token.get_secret_value()


def saved_topics(found: Secrets) -> tuple[str, str] | None:
    """``(NTFY_TOPIC, NTFY_REPLY_TOPIC)`` from ``found`` when both are set, else ``None`` (new ones are needed)."""
    topic, reply_topic = found.ntfy_topic, found.ntfy_reply_topic
    if topic is None or reply_topic is None:
        return None
    return topic.get_secret_value(), reply_topic.get_secret_value()


def setup_telegram(
    env_path: Path,
    *,
    token: str | None,
    chat_id: int | None,
    echo: Echo,
    timeout_s: float = DEFAULT_WAIT_S,
    replace: bool = False,
    ask_token: Callable[[], str] | None = None,
    code: Callable[[], str] = setup_code,
    client: httpx.Client | None = None,
) -> int:
    """Make sure the bot token works, capture the private chat (unless ``chat_id`` is known and ``replace`` is off),
    send the test message, and return the chat id.

    ``token`` and ``chat_id`` are what ``.env`` holds now. Without a token, ``ask_token`` (the wizard's hidden prompt)
    asks for one, which is saved to ``.env`` only once ``getMe`` accepts it; without ``ask_token`` (``fm notify
    setup``) a missing token is a :class:`NotifyError` saying where to get one. A new token always captures the chat
    afresh: the user has to start the new bot before it may message them. ``code`` makes the one-time code and
    ``client`` is an injected ``httpx.Client`` (tests). Raises :class:`NotifyError`; nothing is saved then.
    """
    new_token = False
    if token is None:
        if ask_token is None:
            raise NotifyError(f"{TOKEN_ENV} is not set in {env_path}: {BOTFATHER_HINT} and put its token there")
        echo("Telegram: in the Telegram app, message @BotFather /newbot, pick a name, and paste the token it gives.")
        token = ask_token().strip()
        if not token:
            raise NotifyError(f"no bot token given: {BOTFATHER_HINT}, then put its token in {env_path} as {TOKEN_ENV}")
        new_token = True
    known_chat = None if replace or new_token else chat_id
    with TelegramChannel(token, known_chat, client=client) as telegram:
        username = telegram.username()
        echo(f"Bot token works: @{username}.")
        if new_token:
            set_key(env_path, TOKEN_ENV, token, quote_mode="never")
            echo(f"Saved {TOKEN_ENV} to {env_path}.")
        if telegram.chat_id is None:
            one_time = code()
            echo(f"On your phone, open https://t.me/{username}?start={one_time} and tap Start,")
            echo(f"or send @{username} the code {one_time}. Waiting {timeout_s:.0f}s...")
            capture = telegram.wait_for_chat(one_time, timeout_s=timeout_s)
            if capture is None:
                raise NotifyError("the code did not reach the bot in time; run fm notify setup again")
            telegram.chat_id = capture.chat_id
            set_key(env_path, CHAT_ID_ENV, str(capture.chat_id), quote_mode="never")
            who = f" with {capture.name}" if capture.name else ""
            echo(f"Saved {CHAT_ID_ENV} (your private chat{who}) to {env_path}.")
        else:
            echo(f"{CHAT_ID_ENV} is already set; --replace captures it again.")
        telegram.send(TEST_MESSAGE)
        captured = telegram.chat_id
    echo("Sent a test message to that chat. fm run (or fm bot) records what you press.")
    return captured


def setup_ntfy(
    env_path: Path,
    *,
    topics: tuple[str, str] | None,
    echo: Echo,
    replace: bool = False,
    client: httpx.Client | None = None,
) -> tuple[str, str]:
    """Generate the phone and reply topics (unless ``topics`` holds both and ``replace`` is off), say which one to
    subscribe to, send a test push, and return ``(topic, reply_topic)``.

    New topics are saved to ``.env`` before the push, so a push that fails (no network) leaves the channel set up and
    a re-run only retries the push. The saved resume point is removed with the old reply topic it pointed into.
    ``client`` is an injected ``httpx.Client`` (tests). Raises :class:`NotifyError` when the push fails.
    """
    if replace or topics is None:
        names = (generate_topic(), generate_topic())
        set_key(env_path, TOPIC_ENV, names[0], quote_mode="never")
        set_key(env_path, REPLY_TOPIC_ENV, names[1], quote_mode="never")
        (paths.config_dir() / SINCE_FILE).unlink(missing_ok=True)  # it points into the old reply topic
        echo(f"Generated {TOPIC_ENV} and {REPLY_TOPIC_ENV} and saved them to {env_path}.")
    else:
        names = topics
        echo(f"{TOPIC_ENV} and {REPLY_TOPIC_ENV} are already set; --replace generates new ones.")
    echo(f"In the ntfy app, subscribe to this topic on {DEFAULT_SERVER} (the name is the secret; share it with")
    echo("nobody, and never subscribe to the reply topic, which is for the PC):")
    echo(f"  {names[0]}")
    with NtfyChannel(*names, client=client) as ntfy:
        ntfy.send(TEST_MESSAGE)
    echo("Sent a test notification (subscribed later? run fm notify setup again for a fresh one).")
    echo("fm run (or fm bot) records what you press.")
    return names


__all__ = [
    "BOTFATHER_HINT",
    "DEFAULT_WAIT_S",
    "TEST_MESSAGE",
    "Echo",
    "saved_token",
    "saved_topics",
    "setup_code",
    "setup_ntfy",
    "setup_telegram",
]
