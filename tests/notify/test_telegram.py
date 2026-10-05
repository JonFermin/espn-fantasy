"""Telegram over a mocked Bot API (respx; no network): Approve/Reject buttons rendered, presses accepted from our chat
only, decisions recorded through ``fm.proposals``, confirmations, ``fm notify setup``'s chat capture and
``fm bot --once``.

Leagues come from ``fixtures/config.sample.toml``; the bot token and chat ids are made up. Clocks are pinned except in
the CLI tests, whose proposals have no deadline.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
import typer
from dotenv import dotenv_values
from typer.testing import CliRunner

from fm import paths
from fm.commands import bot as bot_cmd
from fm.commands import notify as notify_cmd
from fm.config import Config, load_config
from fm.notify import (
    INVALID_PRESS,
    ChatCapture,
    DecisionResult,
    Message,
    NotifyError,
    ProposalNotice,
    TelegramChannel,
    alert,
    decode_callback,
    listen,
    notify_proposal,
    report,
)
from fm.notify.base import Callback
from fm.notify.telegram import BOT_API_ROOT
from fm.proposals import LineupMove, LineupPayload, ProposalKind, expire_due, get_proposal, propose
from fm.store import LeagueRow, PlayerRow, ProposalRow, Store

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
TOKEN = "123456:TEST-token_for_unit-tests"
API = f"{BOT_API_ROOT}/bot{TOKEN}"
CHAT_ID = 4242
STRANGER = 9090
NONCE = "AbCdEfGhIjKlMnOpQrStUv_-"
NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
DEADLINE = NOW + timedelta(hours=2)
LINEUP = LineupPayload(moves=(LineupMove(espn_id=10, from_slot_id=20, to_slot_id=0),))
SUMMARY = "nfl lineup change: Sample Quarterback: BE -> QB"

runner = CliRunner()


class FakeTelegram:
    """One bot's Bot API over respx: records every call and serves queued updates the way ``getUpdates`` does."""

    def __init__(self, router: respx.MockRouter) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.updates: list[dict[str, Any]] = []
        self.messages: list[dict[str, Any]] = []
        self.failures: dict[str, tuple[int, str]] = {}
        self.errors: dict[str, str] = {}
        self._next_update = 900
        router.post(url__regex=rf"^{re.escape(API)}/(?P<method>\w+)$").mock(side_effect=self._handle)

    def _handle(self, request: httpx.Request, method: str) -> httpx.Response:
        params: dict[str, Any] = json.loads(request.content or b"{}")
        self.calls.append((method, params))
        if method in self.errors:
            raise httpx.ConnectError(self.errors[method], request=request)
        if method in self.failures:
            status, description = self.failures[method]
            return httpx.Response(status, json={"ok": False, "error_code": status, "description": description})
        return httpx.Response(200, json={"ok": True, "result": self._result(method, params)})

    def _result(self, method: str, params: dict[str, Any]) -> Any:
        if method == "getMe":
            return {"id": 123456, "is_bot": True, "first_name": "Fantasy", "username": "fm_test_bot"}
        if method == "getUpdates":
            offset = params.get("offset")
            if isinstance(offset, int):
                self.updates = [update for update in self.updates if update["update_id"] >= offset]
            allowed = params.get("allowed_updates") or ["message", "callback_query"]
            return [update for update in self.updates if any(kind in update for kind in allowed)][: params["limit"]]
        if method == "sendMessage":
            message = {
                "message_id": 100 + len(self.messages),
                "date": 0,
                "chat": {"id": params["chat_id"], "type": "private"},
                "text": params["text"],
            }
            if "reply_markup" in params:
                message["reply_markup"] = params["reply_markup"]
            self.messages.append(message)
            return message
        return True

    def called(self, method: str) -> list[dict[str, Any]]:
        return [params for name, params in self.calls if name == method]

    def press(self, message: dict[str, Any], label: str, *, user_id: int = CHAT_ID, chat_id: int = CHAT_ID) -> str:
        """Tap a button of a message the bot sent; returns the callback query id."""
        button = next(button for button in message["reply_markup"]["inline_keyboard"][0] if button["text"] == label)
        return self.callback(message, button["callback_data"], user_id=user_id, chat_id=chat_id)

    def callback(self, message: dict[str, Any], data: str, *, user_id: int = CHAT_ID, chat_id: int = CHAT_ID) -> str:
        self._next_update += 1
        query_id = f"query-{self._next_update}"
        pressed = {"message_id": message["message_id"], "date": 0, "chat": {"id": chat_id, "type": "private"}}
        self.updates.append(
            {
                "update_id": self._next_update,
                "callback_query": {
                    "id": query_id,
                    "from": {"id": user_id, "is_bot": False, "first_name": "Sam"},
                    "message": {**pressed, "text": message["text"]},
                    "chat_instance": "instance",
                    "data": data,
                },
            }
        )
        return query_id

    def say(self, text: str, *, chat_id: int = CHAT_ID, chat_type: str = "private") -> None:
        """Someone sends the bot a message."""
        self._next_update += 1
        sender = {"id": chat_id, "is_bot": False, "first_name": "Sam", "last_name": "Sample"}
        chat = {"id": chat_id, "type": chat_type}
        message = {"message_id": self._next_update, "date": 0, "chat": chat, "from": sender, "text": text}
        self.updates.append({"update_id": self._next_update, "message": message})


@pytest.fixture
def api() -> Iterator[FakeTelegram]:
    with respx.mock(assert_all_called=False) as router:
        yield FakeTelegram(router)


@pytest.fixture
def channel(api: FakeTelegram) -> Iterator[TelegramChannel]:
    with TelegramChannel(TOKEN, CHAT_ID) as telegram:
        yield telegram


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        yield opened


@pytest.fixture
def config() -> Config:
    return load_config(FIXTURES / "config.sample.toml", environ={})


@pytest.fixture
def league(store: Store, config: Config) -> LeagueRow:
    return seed_league(store, config)


def seed_league(store: Store, config: Config) -> LeagueRow:
    nfl = config.league("nfl")
    row = store.leagues.upsert(
        LeagueRow(
            key=nfl.key,
            sport=nfl.sport,
            espn_league_id=nfl.espn_league_id,
            season=nfl.season,
            team_id=nfl.team_id,
            as_of=NOW,
        )
    )
    store.players.upsert(PlayerRow(sport="nfl", espn_id=10, full_name="Sample Quarterback", as_of=NOW))
    return row


def lineup(store: Store, config: Config, league: LeagueRow) -> ProposalRow:
    return propose(store, config, league, ProposalKind.LINEUP, LINEUP, created_by="test", deadline=DEADLINE, now=NOW)


def notice(proposal_id: int = 12, body: str = "Sample Quarterback: BE -> QB") -> ProposalNotice:
    return ProposalNotice(proposal_id, NONCE, Message(f"nfl: lineup change #{proposal_id}", body, priority="high"))


def at(minutes: int) -> datetime:
    return NOW + timedelta(minutes=minutes)


# --- the adapter ------------------------------------------------------------------------------------------------------


def test_a_proposal_goes_out_with_inline_approve_and_reject_buttons(
    api: FakeTelegram, channel: TelegramChannel
) -> None:
    channel.send_proposal(notice())
    (sent,) = api.called("sendMessage")
    assert (sent["chat_id"], sent["text"]) == (CHAT_ID, "nfl: lineup change #12\nSample Quarterback: BE -> QB")
    buttons = sent["reply_markup"]["inline_keyboard"][0]
    assert [button["text"] for button in buttons] == ["Approve", "Reject"]
    assert [decode_callback(button["callback_data"]) for button in buttons] == [
        Callback("approve", 12, NONCE),
        Callback("reject", 12, NONCE),
    ]
    assert all(len(button["callback_data"].encode()) <= 64 for button in buttons)
    assert "parse_mode" not in sent  # plain text: names with markup characters are safe


def test_a_long_proposal_is_split_with_the_buttons_on_its_last_message(
    api: FakeTelegram, channel: TelegramChannel
) -> None:
    channel.send_proposal(notice(body="\n".join(f"move {index}: " + "x" * 90 for index in range(80))))
    *first, last = api.called("sendMessage")
    assert len(first) == 2  # about 8 KB of moves: three messages
    assert all("reply_markup" not in params for params in first) and "inline_keyboard" in last["reply_markup"]
    assert all(len(params["text"].encode()) <= 3800 for params in [*first, last])


def test_an_alert_ends_with_its_link_and_a_report_is_silent(api: FakeTelegram, channel: TelegramChannel) -> None:
    channel.send(alert("Lineup not set", "start A over B", link="https://fantasy.espn.com/football/team"))
    channel.send(report("Weekly report", "all good"))
    first, second = api.called("sendMessage")
    assert first["text"] == "Lineup not set\nstart A over B\nhttps://fantasy.espn.com/football/team"
    assert (first["disable_notification"], second["disable_notification"]) == (False, True)
    assert "reply_markup" not in first and "reply_markup" not in second  # a bad link can never sink the alert


def test_poll_long_polls_for_button_presses_and_acknowledges_them(api: FakeTelegram, channel: TelegramChannel) -> None:
    channel.send_proposal(notice())
    query = api.press(api.messages[0], "Approve")
    (reply,) = channel.poll()
    assert (reply.channel, reply.proposal_id, reply.decision, reply.nonce, reply.sender) == (
        "telegram",
        12,
        "approve",
        NONCE,
        str(CHAT_ID),
    )
    assert reply.ref == {"callback_query_id": query, "message_id": 100, "text": api.messages[0]["text"]}
    first = api.called("getUpdates")[0]
    assert (first["timeout"], first["allowed_updates"], "offset" in first) == (25, ["callback_query"], False)
    assert channel.poll(wait=False) == []
    second = api.called("getUpdates")[1]
    assert (second["timeout"], second["offset"]) == (0, 901 + 1)


def test_presses_from_anyone_but_our_chat_are_dropped_unanswered(api: FakeTelegram, channel: TelegramChannel) -> None:
    channel.send_proposal(notice())
    message = api.messages[0]
    api.press(message, "Approve", user_id=STRANGER, chat_id=STRANGER)  # another chat
    api.press(message, "Approve", user_id=STRANGER)  # another user in a chat with our id
    api.press(message, "Approve", chat_id=STRANGER)  # our user, but not our chat
    api.callback(message, "approve:12:not-a-nonce")  # not a decision
    api.say("approve")  # a message, not a press
    assert channel.poll() == []
    assert api.called("answerCallbackQuery") == []


def test_confirm_answers_the_press_and_edits_out_the_buttons(api: FakeTelegram, channel: TelegramChannel) -> None:
    channel.send_proposal(notice())
    query = api.press(api.messages[0], "Reject")
    (reply,) = channel.poll()
    channel.confirm(reply, DecisionResult(12, "reject", ok=True, detail=SUMMARY))
    assert api.called("answerCallbackQuery") == [{"callback_query_id": query, "text": f"#12 rejected: {SUMMARY}"}]
    (edit,) = api.called("editMessageText")
    assert (edit["chat_id"], edit["message_id"]) == (CHAT_ID, 100)
    assert edit["text"] == "nfl: lineup change #12\nSample Quarterback: BE -> QB\n\nRejected via Telegram."
    assert "reply_markup" not in edit  # editing without a keyboard removes the buttons
    assert len(api.called("sendMessage")) == 1  # no extra message when the toast worked


def test_a_press_too_old_to_answer_gets_a_reply_message(api: FakeTelegram, channel: TelegramChannel) -> None:
    channel.send_proposal(notice())
    api.press(api.messages[0], "Approve")
    (reply,) = channel.poll()
    api.failures["answerCallbackQuery"] = (400, "Bad Request: query is too old and response timeout expired")
    channel.confirm(reply, DecisionResult(12, "approve", ok=False, detail="it expired"))
    reply_message = api.called("sendMessage")[-1]
    assert reply_message["text"] == "#12 not approved: it expired"
    assert reply_message["reply_parameters"] == {"message_id": 100, "allow_sending_without_reply": True}
    assert api.called("editMessageText")[0]["text"].endswith("\n\nNot approved: it expired")


def test_confirm_fails_only_when_nothing_reached_the_phone(api: FakeTelegram, channel: TelegramChannel) -> None:
    channel.send_proposal(notice())
    api.press(api.messages[0], "Approve")
    (reply,) = channel.poll()
    api.failures["editMessageText"] = (400, "Bad Request: message can't be edited")
    channel.confirm(reply, DecisionResult(12, "approve", ok=True, detail=SUMMARY))  # the toast landed
    api.failures["answerCallbackQuery"] = (400, "Bad Request: query is too old")
    api.failures["sendMessage"] = (403, "Forbidden: bot was blocked by the user")
    with pytest.raises(NotifyError, match="sendMessage: HTTP 403"):
        channel.confirm(reply, DecisionResult(12, "approve", ok=True, detail=SUMMARY))


def test_dismiss_answers_an_ignored_press_with_the_reason(api: FakeTelegram, channel: TelegramChannel) -> None:
    channel.send_proposal(notice())
    query = api.press(api.messages[0], "Approve")
    (reply,) = channel.poll()
    channel.dismiss(reply, INVALID_PRESS)
    assert api.called("answerCallbackQuery") == [{"callback_query_id": query, "text": INVALID_PRESS}]


def test_errors_never_echo_the_token(api: FakeTelegram, channel: TelegramChannel) -> None:
    api.failures["getUpdates"] = (401, f"Unauthorized: bad token {TOKEN}")
    with pytest.raises(NotifyError) as unauthorized:
        channel.poll()
    api.errors["sendMessage"] = f"cannot reach {API}/sendMessage"
    with pytest.raises(NotifyError) as unreachable:
        channel.send(Message("hello"))
    for error in (unauthorized.value, unreachable.value):
        assert TOKEN not in str(error) and "123456:" not in str(error)
    assert unauthorized.value.__context__ is None
    assert unreachable.value.__cause__ is None and unreachable.value.__suppress_context__  # httpx's error stays out
    assert str(unauthorized.value) == "telegram getUpdates: HTTP 401: Unauthorized: bad token ***"


def test_request_log_lines_hide_the_token(
    api: FakeTelegram, channel: TelegramChannel, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="httpx")
    channel.send(Message("hello"))
    assert "sendMessage" in caplog.text
    assert TOKEN not in caplog.text and f"{BOT_API_ROOT}/bot***/sendMessage" in caplog.text


def test_a_malformed_token_is_refused_without_echoing_it() -> None:
    with pytest.raises(NotifyError) as error:
        TelegramChannel("not a token/../x", CHAT_ID)
    assert "not a token" not in str(error.value)


def test_from_config_names_the_missing_secrets() -> None:
    sample = FIXTURES / "config.sample.toml"
    with pytest.raises(NotifyError, match="TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID missing"):
        TelegramChannel.from_config(load_config(sample, environ={}))
    with pytest.raises(NotifyError, match="TELEGRAM_CHAT_ID missing"):
        TelegramChannel.from_config(load_config(sample, environ={"TELEGRAM_BOT_TOKEN": TOKEN}))
    ready = load_config(sample, environ={"TELEGRAM_BOT_TOKEN": TOKEN, "TELEGRAM_CHAT_ID": str(CHAT_ID)})
    assert TelegramChannel.from_config(ready).chat_id == CHAT_ID


def test_wait_for_chat_captures_only_the_private_chat_that_sends_the_code(api: FakeTelegram) -> None:
    api.say("hi", chat_id=STRANGER)
    api.say("/start 246810", chat_id=-1001, chat_type="group")
    api.say("/start 135799", chat_id=STRANGER)
    api.say("/start 246810")
    with TelegramChannel(TOKEN, None) as telegram:
        capture = telegram.wait_for_chat("246810", timeout_s=60, sleep=lambda _: None)
    assert capture == ChatCapture(chat_id=CHAT_ID, name="Sam Sample")
    polls = api.called("getUpdates")
    assert polls[0]["allowed_updates"] == ["message"]
    assert polls[-1]["offset"] == 905  # everything seen is acknowledged
    assert api.updates == []


def test_wait_for_chat_gives_up_after_the_timeout(api: FakeTelegram) -> None:
    api.say("no code here")
    clock = iter(range(0, 1000, 20))  # each look at the clock is 20 s later
    with TelegramChannel(TOKEN, None) as telegram:
        assert (
            telegram.wait_for_chat("246810", timeout_s=60, sleep=lambda _: None, monotonic=lambda: next(clock)) is None
        )
    polls = api.called("getUpdates")
    assert [poll["timeout"] for poll in polls[:2]] == [25, 20]  # never past the deadline
    assert polls[-1] == {"offset": 902, "limit": 1, "timeout": 0}  # what it saw is acknowledged


# --- decisions through fm.proposals -----------------------------------------------------------------------------------


def test_a_press_from_our_chat_records_the_approval(
    api: FakeTelegram, channel: TelegramChannel, store: Store, config: Config, league: LeagueRow
) -> None:
    row = lineup(store, config, league)
    notify_proposal(channel, store, row.row_id, now=NOW, tz=UTC)
    (message,) = api.messages
    assert message["text"].startswith(f"nfl: lineup change #{row.row_id}\nSample Quarterback: BE -> QB\n")
    query = api.press(message, "Approve")
    results: list[DecisionResult] = []
    assert listen(store, channel, once=True, clock=lambda: at(1), on_result=results.append) == 1
    approved = get_proposal(store, row.row_id)
    assert (approved.status, approved.decided_by, approved.decided_at) == ("approved", "telegram", at(1))
    assert approved.execution_token is not None
    assert approved.execution_token not in json.dumps(api.calls)  # no button or message ever carried it
    assert results == [DecisionResult(row.row_id, "approve", ok=True, detail=SUMMARY)]
    assert api.called("answerCallbackQuery") == [
        {"callback_query_id": query, "text": f"#{row.row_id} approved: {SUMMARY}"}
    ]
    assert api.called("editMessageText")[0]["text"].endswith("\n\nApproved via Telegram.")


def test_a_reject_press_records_the_rejection(
    api: FakeTelegram, channel: TelegramChannel, store: Store, config: Config, league: LeagueRow
) -> None:
    row = lineup(store, config, league)
    notify_proposal(channel, store, row.row_id, now=NOW)
    api.press(api.messages[0], "Reject")
    assert listen(store, channel, once=True, clock=lambda: at(1)) == 1
    rejected = get_proposal(store, row.row_id)
    assert (rejected.status, rejected.decided_by, rejected.execution_token) == ("rejected", "telegram", None)


def test_another_chat_cannot_decide_even_with_the_real_button_data(
    api: FakeTelegram, channel: TelegramChannel, store: Store, config: Config, league: LeagueRow
) -> None:
    row = lineup(store, config, league)
    notify_proposal(channel, store, row.row_id, now=NOW)
    (message,) = api.messages
    api.press(message, "Approve", user_id=STRANGER, chat_id=STRANGER)
    assert listen(store, channel, once=True, clock=lambda: at(1)) == 0
    assert get_proposal(store, row.row_id).status == "proposed"
    api.press(message, "Approve")  # the stranger did not burn our button
    assert listen(store, channel, once=True, clock=lambda: at(2)) == 1
    assert get_proposal(store, row.row_id).status == "approved"


def test_a_forged_or_reused_button_is_ignored(
    api: FakeTelegram, channel: TelegramChannel, store: Store, config: Config, league: LeagueRow
) -> None:
    row = lineup(store, config, league)
    notify_proposal(channel, store, row.row_id, now=NOW)
    (message,) = api.messages
    forged = api.callback(message, f"approve:{row.row_id}:{NONCE}")  # well formed, never issued
    assert listen(store, channel, once=True, clock=lambda: at(1)) == 0
    assert get_proposal(store, row.row_id).status == "proposed"
    assert api.called("answerCallbackQuery") == [{"callback_query_id": forged, "text": INVALID_PRESS}]
    api.press(message, "Reject")
    second = api.press(message, "Approve")  # a second tap on the same notification
    assert listen(store, channel, once=True, clock=lambda: at(2)) == 1
    decided = get_proposal(store, row.row_id)
    assert (decided.status, decided.execution_token) == ("rejected", None)
    assert api.called("answerCallbackQuery")[-1] == {"callback_query_id": second, "text": INVALID_PRESS}


def test_an_expired_proposal_cannot_be_approved(
    api: FakeTelegram, channel: TelegramChannel, store: Store, config: Config, league: LeagueRow
) -> None:
    row = lineup(store, config, league)
    notify_proposal(channel, store, row.row_id, now=NOW)
    api.press(api.messages[0], "Approve")
    late = DEADLINE + timedelta(minutes=1)
    results: list[DecisionResult] = []
    assert listen(store, channel, once=True, clock=lambda: late, on_result=results.append) == 1
    expired = get_proposal(store, row.row_id)
    assert (expired.status, expired.decided_by, expired.execution_token) == ("expired", "expiry", None)
    (result,) = results
    assert not result.ok and "expired at 2026-10-04 14:00 UTC" in result.detail
    assert api.called("answerCallbackQuery")[0]["text"].startswith(f"#{row.row_id} not approved: ")
    assert "Not approved: " in api.called("editMessageText")[0]["text"]


def test_an_already_swept_proposal_cannot_be_approved_either(
    api: FakeTelegram, channel: TelegramChannel, store: Store, config: Config, league: LeagueRow
) -> None:
    row = lineup(store, config, league)
    notify_proposal(channel, store, row.row_id, now=NOW)
    late = DEADLINE + timedelta(hours=3)
    assert [swept.row_id for swept in expire_due(store, now=late)] == [row.row_id]
    api.press(api.messages[0], "Approve")
    assert listen(store, channel, once=True, clock=lambda: late) == 1
    assert get_proposal(store, row.row_id).status == "expired"


def test_the_buttons_of_every_notification_die_with_the_decision(
    api: FakeTelegram, channel: TelegramChannel, store: Store, config: Config, league: LeagueRow
) -> None:
    row = lineup(store, config, league)
    notify_proposal(channel, store, row.row_id, now=NOW)
    notify_proposal(channel, store, row.row_id, now=at(30))  # a reminder, with a fresh nonce
    first, reminder = api.messages
    assert first["reply_markup"] != reminder["reply_markup"]
    api.press(reminder, "Approve")
    assert listen(store, channel, once=True, clock=lambda: at(31)) == 1
    stale = api.press(first, "Reject")
    assert listen(store, channel, once=True, clock=lambda: at(32)) == 0
    assert get_proposal(store, row.row_id).status == "approved"
    assert api.called("answerCallbackQuery")[-1] == {"callback_query_id": stale, "text": INVALID_PRESS}


# --- fm notify setup / fm bot -----------------------------------------------------------------------------------------


def _root() -> None:
    pass


def cli() -> typer.Typer:
    root = typer.Typer(no_args_is_help=True, add_completion=False, pretty_exceptions_enable=False)
    root.callback()(_root)
    notify_cmd.register(root)
    bot_cmd.register(root)
    return root


def run(*args: str, expect: int = 0) -> str:
    result = runner.invoke(cli(), list(args), catch_exceptions=False)
    assert result.exit_code == expect, result.output
    return result.output


def write_config(env: dict[str, str]) -> Path:
    """The sample config.toml and a .env with ``env`` in the per-test config dir; returns the .env path."""
    directory = paths.config_dir()
    (directory / "config.toml").write_text((FIXTURES / "config.sample.toml").read_text(encoding="utf-8"), "utf-8")
    env_path = directory / ".env"
    env_path.write_text("".join(f"{name}={value}\n" for name, value in env.items()), encoding="utf-8")
    return env_path


def test_setup_saves_the_private_chat_that_sends_the_code(api: FakeTelegram, monkeypatch: pytest.MonkeyPatch) -> None:
    env_path = write_config({"TELEGRAM_BOT_TOKEN": TOKEN})
    monkeypatch.setattr(notify_cmd, "setup_code", lambda: "246810")
    api.say("hello?", chat_id=STRANGER)  # a stranger messaging the public bot is not captured
    api.say("/start 246810")
    output = run("notify", "setup")
    assert "https://t.me/fm_test_bot?start=246810" in output and "246810" in output
    assert dotenv_values(env_path) == {"TELEGRAM_BOT_TOKEN": TOKEN, "TELEGRAM_CHAT_ID": str(CHAT_ID)}
    (test_message,) = api.called("sendMessage")
    assert test_message["chat_id"] == CHAT_ID
    assert test_message["text"].startswith("espn-fantasy: notifications are set up")
    assert TOKEN not in output


def test_setup_with_the_chat_already_saved_only_sends_the_test_message(api: FakeTelegram) -> None:
    write_config({"TELEGRAM_BOT_TOKEN": TOKEN, "TELEGRAM_CHAT_ID": str(CHAT_ID)})
    output = run("notify", "setup")
    assert "already set" in output
    assert [method for method, _ in api.calls] == ["getMe", "sendMessage"]


def test_setup_without_a_token_says_how_to_get_one(api: FakeTelegram) -> None:
    write_config({})
    output = run("notify", "setup", expect=1)
    assert "TELEGRAM_BOT_TOKEN is not set" in output and "@BotFather" in output
    assert api.calls == []


def test_setup_that_never_hears_the_code_fails_without_saving(api: FakeTelegram) -> None:
    env_path = write_config({"TELEGRAM_BOT_TOKEN": TOKEN})
    output = run("notify", "setup", "--timeout", "1", expect=1)  # waits one real second
    assert "did not reach the bot in time" in output
    assert dotenv_values(env_path) == {"TELEGRAM_BOT_TOKEN": TOKEN}
    assert api.called("sendMessage") == []


def test_fm_bot_once_records_the_presses_waiting(api: FakeTelegram, config: Config) -> None:
    write_config({"TELEGRAM_BOT_TOKEN": TOKEN, "TELEGRAM_CHAT_ID": str(CHAT_ID)})
    with Store.open() as store:
        league = seed_league(store, config)
        row = propose(store, config, league, ProposalKind.LINEUP, LINEUP, created_by="test")
        with TelegramChannel(TOKEN, CHAT_ID) as telegram:
            notify_proposal(telegram, store, row.row_id)
    api.press(api.messages[0], "Approve")
    output = run("bot", "--once")
    assert f"#{row.row_id} approved: {SUMMARY}" in output
    assert "recorded 1 decision" in output
    with Store.open() as store:
        assert get_proposal(store, row.row_id).decided_by == "telegram"


def test_fm_bot_without_a_channel_set_up_fails_cleanly(api: FakeTelegram) -> None:
    write_config({})
    assert "run fm notify setup" in run("bot", "--once", expect=1)
    assert api.calls == []
