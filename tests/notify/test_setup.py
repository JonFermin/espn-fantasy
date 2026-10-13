"""``fm.notify.setup``: the phone setup ``fm notify setup`` and the ``fm setup`` wizard share, over respx (no network).

The command's own behaviour is pinned by the ``fm notify setup`` tests in ``test_telegram.py`` and ``test_ntfy.py``;
these cover what the wizard adds: a Telegram token asked for (and saved only once ``getMe`` accepts it), the secrets
passed in rather than read from a ``Config``, and output through an injected ``echo``. The token, chat id and topics
are made up.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from dotenv import dotenv_values

from fm.config import Secrets
from fm.notify import NotifyError
from fm.notify.setup import TEST_MESSAGE, saved_token, saved_topics, setup_ntfy, setup_telegram
from fm.notify.telegram import BOT_API_ROOT
from fm.setup import AcceptDefaults, notify_setup

TOKEN = "123456:TEST-token_for_setup-tests"
API = f"{BOT_API_ROOT}/bot{TOKEN}"
CHAT_ID = 4242
CODE = "135790"
TOPIC = "espn-fantasy-PhoneTopicForTests_0123456"
REPLY = "espn-fantasy-ReplyTopicForTests_6543210"


class FakeBot:
    """The Bot API for one token: ``getMe`` answers (or refuses), the user's ``/start <code>`` waits in the updates."""

    def __init__(self, router: respx.MockRouter, *, accepts: bool = True) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.accepts = accepts
        router.post(url__regex=rf"^{re.escape(API)}/(?P<method>\w+)$").mock(side_effect=self._handle)

    def _handle(self, request: httpx.Request, method: str) -> httpx.Response:
        params: dict[str, Any] = json.loads(request.content or b"{}")
        self.calls.append((method, params))
        if not self.accepts:
            return httpx.Response(401, json={"ok": False, "error_code": 401, "description": "Unauthorized"})
        if method == "getMe":
            return _ok({"id": 123456, "is_bot": True, "first_name": "Fantasy", "username": "fm_setup_bot"})
        if method == "getUpdates":
            if "offset" in params:
                return _ok([])
            chat = {"id": CHAT_ID, "type": "private"}
            sender = {"id": CHAT_ID, "is_bot": False, "first_name": "Sam"}
            message = {"message_id": 1, "date": 0, "chat": chat, "from": sender, "text": f"/start {CODE}"}
            return _ok([{"update_id": 900, "message": message}])
        if method == "sendMessage":
            return _ok({"message_id": 2, "date": 0, "chat": {"id": params["chat_id"], "type": "private"}})
        return _ok(True)

    def methods(self) -> list[str]:
        return [method for method, _ in self.calls]


def _ok(result: object) -> httpx.Response:
    return httpx.Response(200, json={"ok": True, "result": result})


@pytest.fixture
def router() -> Iterator[respx.MockRouter]:
    with respx.mock(assert_all_called=False) as mocked:
        yield mocked


@pytest.fixture
def env_path(tmp_path: Path) -> Path:
    """Where ``.env`` would go, in a config dir that exists (as ``fm.paths`` makes it) but holds no ``.env`` yet."""
    directory = tmp_path / "config"
    directory.mkdir()
    return directory / ".env"


def test_an_asked_for_token_is_saved_once_getme_accepts_it_then_the_chat_is_captured(
    router: respx.MockRouter, env_path: Path
) -> None:
    bot = FakeBot(router)
    lines: list[str] = []
    chat_id = setup_telegram(
        env_path, token=None, chat_id=None, echo=lines.append, ask_token=lambda: f"  {TOKEN} ", code=lambda: CODE
    )
    assert chat_id == CHAT_ID
    assert dotenv_values(env_path) == {"TELEGRAM_BOT_TOKEN": TOKEN, "TELEGRAM_CHAT_ID": str(CHAT_ID)}
    assert bot.methods() == ["getMe", "getUpdates", "getUpdates", "sendMessage"]
    assert bot.calls[-1][1]["text"].startswith(TEST_MESSAGE.title)
    output = "\n".join(lines)
    assert "@BotFather" in output and f"https://t.me/fm_setup_bot?start={CODE}" in output
    assert TOKEN not in output


def test_a_new_token_captures_the_chat_again_even_when_one_is_saved(router: respx.MockRouter, env_path: Path) -> None:
    bot = FakeBot(router)
    setup_telegram(env_path, token=None, chat_id=9090, echo=print, ask_token=lambda: TOKEN, code=lambda: CODE)
    assert "getUpdates" in bot.methods()  # the new bot may not message the user before they start it
    assert dotenv_values(env_path)["TELEGRAM_CHAT_ID"] == str(CHAT_ID)


def test_a_refused_or_missing_token_saves_nothing(router: respx.MockRouter, env_path: Path) -> None:
    FakeBot(router, accepts=False)
    with pytest.raises(NotifyError, match="getMe: HTTP 401"):
        setup_telegram(env_path, token=None, chat_id=None, echo=print, ask_token=lambda: TOKEN)
    assert not env_path.exists()
    with pytest.raises(NotifyError, match="no bot token given"):
        setup_telegram(env_path, token=None, chat_id=None, echo=print, ask_token=lambda: "  ")
    with pytest.raises(NotifyError, match="TELEGRAM_BOT_TOKEN is not set"):
        setup_telegram(env_path, token=None, chat_id=None, echo=print)  # fm notify setup never asks
    assert not env_path.exists()


def test_a_saved_token_and_chat_only_get_the_test_message(router: respx.MockRouter, env_path: Path) -> None:
    bot = FakeBot(router)
    lines: list[str] = []
    assert setup_telegram(env_path, token=TOKEN, chat_id=CHAT_ID, echo=lines.append) == CHAT_ID
    assert bot.methods() == ["getMe", "sendMessage"]
    assert any("already set" in line for line in lines)
    assert not env_path.exists()  # nothing new to save


def test_ntfy_generates_topics_or_keeps_the_saved_ones(router: respx.MockRouter, env_path: Path) -> None:
    published: list[dict[str, Any]] = []

    def publish(request: httpx.Request) -> httpx.Response:
        published.append(json.loads(request.content))
        return httpx.Response(200, json={"id": "msg0001", "time": 1, "event": "message"})

    router.post("https://ntfy.sh/").mock(side_effect=publish)
    lines: list[str] = []
    topic, reply = setup_ntfy(env_path, topics=None, echo=lines.append)
    assert dotenv_values(env_path) == {"NTFY_TOPIC": topic, "NTFY_REPLY_TOPIC": reply}
    assert topic in "\n".join(lines) and reply not in "\n".join(lines)
    assert setup_ntfy(env_path, topics=(TOPIC, REPLY), echo=lines.append) == (TOPIC, REPLY)
    assert [push["topic"] for push in published] == [topic, TOPIC]
    assert dotenv_values(env_path) == {"NTFY_TOPIC": topic, "NTFY_REPLY_TOPIC": reply}  # kept, not replaced


def test_saved_secrets_are_read_without_their_values_leaking() -> None:
    found = Secrets.model_validate({"TELEGRAM_BOT_TOKEN": TOKEN, "NTFY_TOPIC": TOPIC})
    assert saved_token(found) == TOKEN and saved_topics(found) is None  # one topic is not a channel
    both = Secrets.model_validate({"NTFY_TOPIC": TOPIC, "NTFY_REPLY_TOPIC": REPLY})
    assert saved_token(both) is None and saved_topics(both) == (TOPIC, REPLY)


def test_the_wizard_asks_for_the_telegram_token_through_its_hidden_prompt(
    router: respx.MockRouter, env_path: Path
) -> None:
    FakeBot(router)

    class Secret(AcceptDefaults):
        def __init__(self) -> None:
            self.asked: list[str] = []

        def secret(self, question: str) -> str:
            self.asked.append(question)
            return TOKEN

    prompts = Secret()
    with pytest.raises(NotifyError, match="did not reach the bot"):
        # the code the bot sees is not the random one this run makes, so the capture times out at once
        notify_setup("telegram", env_path, Secrets(), prompts=prompts, echo=print, timeout_s=0)
    assert prompts.asked == ["Telegram bot token from @BotFather (hidden)"]
    assert dotenv_values(env_path) == {"TELEGRAM_BOT_TOKEN": TOKEN}  # getMe accepted it before the wait
