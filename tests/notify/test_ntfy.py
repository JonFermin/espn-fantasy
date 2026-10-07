"""ntfy over a mocked server (respx; no network): proposals published with ``http`` action buttons that post the
single-use nonce to a private reply topic, the reply topic subscribed to without any inbound port, a confirmation push
after each decision, forged and replayed posts ignored, ``fm notify setup``'s topics and ``fm bot --once``.

A tap is simulated by sending the published action's own request (``url``, ``method``, ``body``) through the mock, so
the test proves the button itself, not a hand-built reply. Topic names are made up.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from collections import defaultdict
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
    DecisionResult,
    Message,
    NotifyError,
    NtfyChannel,
    ProposalNotice,
    alert,
    decode_callback,
    generate_topic,
    listen,
    notify_proposal,
    report,
)
from fm.notify.base import Callback
from fm.notify.ntfy import TOPIC_PATTERN
from fm.proposals import LineupMove, LineupPayload, ProposalKind, get_proposal, propose
from fm.store import LeagueRow, PlayerRow, ProposalRow, Store

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SERVER = "https://ntfy.sh"
TOPIC = "espn-fantasy-PhoneTopicForTests_0123456"
REPLY = "espn-fantasy-ReplyTopicForTests_6543210"
NONCE = "AbCdEfGhIjKlMnOpQrStUv_-"
NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
DEADLINE = NOW + timedelta(hours=2)
LINEUP = LineupPayload(moves=(LineupMove(espn_id=10, from_slot_id=20, to_slot_id=0),))
SUMMARY = "nfl lineup change: Sample Quarterback: BE -> QB"

runner = CliRunner()


class FakeNtfy:
    """An ntfy server over respx: JSON publishes, plain posts (what an ``http`` action sends) and the ``/json``
    subscription, streamed (``open``, cached messages, ``keepalive``) or polled (``poll=1``)."""

    def __init__(self, router: respx.MockRouter) -> None:
        self.published: list[dict[str, Any]] = []
        self.topics: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
        self.subscriptions: list[httpx.Request] = []
        self.publish_status = 200
        self.subscribe_status = 200
        self._next = 0
        base = re.escape(SERVER)
        router.post(f"{SERVER}/").mock(side_effect=self._publish)
        router.post(url__regex=rf"^{base}/(?P<topic>[\w-]+)$").mock(side_effect=self._post)
        router.get(url__regex=rf"^{base}/(?P<topic>[\w-]+)/json(\?.*)?$").mock(side_effect=self._subscribe)

    def _cache(self, topic: str, message: str, title: str | None = None) -> dict[str, Any]:
        self._next += 1
        event: dict[str, Any] = {
            "id": f"msg{self._next:04d}",
            "time": 1_790_000_000 + self._next,
            "event": "message",
            "topic": topic,
            "message": message,
        }
        if title:
            event["title"] = title
        self.topics[topic].append(event)
        return event

    def _publish(self, request: httpx.Request) -> httpx.Response:
        if self.publish_status != 200:
            return httpx.Response(self.publish_status, json={"code": 50001, "error": "internal error"})
        body: dict[str, Any] = json.loads(request.content)
        self.published.append(body)
        return httpx.Response(200, json=self._cache(body["topic"], body["message"], body.get("title")))

    def _post(self, request: httpx.Request, topic: str) -> httpx.Response:
        return httpx.Response(200, json=self._cache(topic, request.content.decode()))

    def _subscribe(self, request: httpx.Request, topic: str) -> httpx.Response:
        self.subscriptions.append(request)
        if self.subscribe_status != 200:
            return httpx.Response(self.subscribe_status, json={"code": 40301, "error": "forbidden"})
        params = request.url.params
        events = self.topics[topic]
        ids = [event["id"] for event in events]
        since = params.get("since", "all")
        if since in ids:
            events = events[ids.index(since) + 1 :]
        streaming = params.get("poll") != "1"
        lines: list[dict[str, Any]] = []
        if streaming:
            lines.append({"id": "open", "time": 0, "event": "open", "topic": topic})
        lines.extend(events)
        if streaming:
            lines.append({"id": "keepalive", "time": 0, "event": "keepalive", "topic": topic})
        return httpx.Response(200, content="".join(json.dumps(line) + "\n" for line in lines).encode())

    def phone(self) -> list[dict[str, Any]]:
        """What the phone's topic received, in order."""
        return [body for body in self.published if body["topic"] == TOPIC]

    def tap(self, published: dict[str, Any], label: str) -> None:
        """Press a button the way the ntfy app does: send the action's own request."""
        action = next(action for action in published["actions"] if action["label"] == label)
        assert (action["action"], action["method"]) == ("http", "POST")
        httpx.request(action["method"], action["url"], content=action["body"])

    def post(self, topic: str, body: str) -> None:
        """Anyone who knows a topic's name can post to it."""
        httpx.post(f"{SERVER}/{topic}", content=body)


@pytest.fixture
def server() -> Iterator[FakeNtfy]:
    with respx.mock(assert_all_called=False) as router:
        yield FakeNtfy(router)


@pytest.fixture
def channel(server: FakeNtfy) -> Iterator[NtfyChannel]:
    with NtfyChannel(TOPIC, REPLY) as ntfy:
        yield ntfy


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


def notice(proposal_id: int = 12) -> ProposalNotice:
    title = f"nfl: lineup change #{proposal_id}"
    return ProposalNotice(proposal_id, NONCE, Message(title, "Sample Quarterback: BE -> QB", priority="high"))


def at(minutes: int) -> datetime:
    return NOW + timedelta(minutes=minutes)


# --- the adapter ------------------------------------------------------------------------------------------------------


def test_a_proposal_is_published_with_http_buttons_that_post_its_nonce_to_the_reply_topic(
    server: FakeNtfy, channel: NtfyChannel
) -> None:
    channel.send_proposal(notice())
    (published,) = server.published
    assert published["topic"] == TOPIC
    assert (published["title"], published["message"], published["priority"]) == (
        "nfl: lineup change #12",
        "Sample Quarterback: BE -> QB",
        4,
    )
    actions = published["actions"]
    assert [action["label"] for action in actions] == ["Approve", "Reject"]
    for action in actions:
        assert (action["action"], action["method"], action["url"], action["clear"]) == (
            "http",
            "POST",
            f"{SERVER}/{REPLY}",
            True,
        )
    assert [decode_callback(action["body"]) for action in actions] == [
        Callback("approve", 12, NONCE),
        Callback("reject", 12, NONCE),
    ]


def test_a_long_proposal_is_split_with_the_buttons_on_its_last_push(server: FakeNtfy, channel: NtfyChannel) -> None:
    body = "\n".join(f"move {index}: " + "x" * 90 for index in range(50))
    channel.send_proposal(ProposalNotice(12, NONCE, Message("nfl: lineup change #12", body, priority="high")))
    first, last = server.published
    assert (first["title"], last["title"]) == ("nfl: lineup change #12 (1/2)", "nfl: lineup change #12 (2/2)")
    assert "actions" not in first and [action["label"] for action in last["actions"]] == ["Approve", "Reject"]
    assert f"{first['message']}\n{last['message']}" == body


def test_an_alert_opens_its_link_and_a_long_report_is_split_and_quiet(server: FakeNtfy, channel: NtfyChannel) -> None:
    channel.send(alert("Lineup not set", "start A over B", link="https://fantasy.espn.com/football/team"))
    channel.send(report("Weekly report", "\n".join("line " + "x" * 100 for _ in range(60))))
    first, *pieces = server.published
    assert (first["title"], first["message"], first["priority"], first["tags"], first["click"]) == (
        "Lineup not set",
        "start A over B",
        4,
        ["warning"],
        "https://fantasy.espn.com/football/team",
    )
    assert [piece["title"] for piece in pieces] == ["Weekly report (1/2)", "Weekly report (2/2)"]
    assert {piece["priority"] for piece in pieces} == {2}
    assert all(len(piece["message"].encode()) <= 3800 for piece in pieces)


def test_a_push_without_a_body_shows_its_title_once(server: FakeNtfy, channel: NtfyChannel) -> None:
    channel.send(Message("Session expired; run fm login"))
    (published,) = server.published
    assert published["message"] == "Session expired; run fm login" and "title" not in published


def test_waiting_subscribes_to_the_reply_stream_and_returns_the_first_press(
    server: FakeNtfy, channel: NtfyChannel
) -> None:
    channel.send_proposal(notice())
    server.tap(server.published[0], "Approve")
    server.tap(server.published[0], "Reject")
    (first,) = channel.poll()
    assert (first.channel, first.proposal_id, first.decision, first.nonce) == ("ntfy", 12, "approve", NONCE)
    (second,) = channel.poll()
    assert second.decision == "reject"
    assert channel.poll() == []  # only the keepalive is left
    sinces = [request.url.params["since"] for request in server.subscriptions]
    assert sinces == ["all", first.sender, second.sender]
    assert all("poll" not in request.url.params for request in server.subscriptions)
    assert all(request.url.path == f"/{REPLY}/json" for request in server.subscriptions)


def test_posts_that_are_not_decisions_are_skipped(server: FakeNtfy, channel: NtfyChannel) -> None:
    server.post(REPLY, "hello")
    server.post(REPLY, "approve:12:not-a-nonce")
    server.post(REPLY, f"approve:12:{NONCE}")
    (reply,) = channel.poll()
    assert reply.nonce == NONCE


def test_not_waiting_asks_once_with_poll_and_returns_every_press(server: FakeNtfy, channel: NtfyChannel) -> None:
    channel.send_proposal(notice())
    server.tap(server.published[0], "Approve")
    server.post(REPLY, "junk")
    server.tap(server.published[0], "Reject")
    assert [reply.decision for reply in channel.poll(wait=False)] == ["approve", "reject"]
    assert server.subscriptions[0].url.params["poll"] == "1"
    assert channel.poll(wait=False) == []


def test_a_quiet_stream_that_times_out_is_no_error() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(url__regex=rf"^{re.escape(SERVER)}/{REPLY}/json").mock(side_effect=httpx.ReadTimeout("quiet"))
        with NtfyChannel(TOPIC, REPLY) as ntfy:
            assert ntfy.poll() == []


def test_confirmation_goes_to_the_phone_topic_never_the_reply_topic(server: FakeNtfy, channel: NtfyChannel) -> None:
    channel.send_proposal(notice())
    server.tap(server.published[0], "Approve")
    (reply,) = channel.poll()
    channel.confirm(reply, DecisionResult(12, "approve", ok=True, detail=SUMMARY))
    confirmation = server.published[-1]
    assert (confirmation["topic"], confirmation["title"], confirmation["message"]) == (TOPIC, "#12 approved", SUMMARY)
    assert confirmation["tags"] == ["white_check_mark"] and "actions" not in confirmation
    channel.dismiss(reply, "no longer valid")  # an ignored post gets no push: anyone with the topic could spam
    assert len(server.published) == 2


def test_errors_never_name_the_topics(server: FakeNtfy, channel: NtfyChannel) -> None:
    server.publish_status = 500
    with pytest.raises(NotifyError) as publish:
        channel.send(Message("hello"))
    server.subscribe_status = 403
    with pytest.raises(NotifyError) as subscribe:
        channel.poll()
    with pytest.raises(NotifyError) as poll:
        channel.poll(wait=False)
    assert [str(error.value) for error in (publish, subscribe, poll)] == [
        "ntfy publish: HTTP 500",
        "ntfy subscribe: HTTP 403",
        "ntfy poll: HTTP 403",
    ]


def test_a_connection_error_never_names_the_topics() -> None:
    with respx.mock(assert_all_called=False) as router:
        router.post(f"{SERVER}/").mock(side_effect=httpx.ConnectError(f"cannot reach {SERVER}/{TOPIC}"))
        router.get(url__regex=rf"^{re.escape(SERVER)}/").mock(side_effect=httpx.ConnectError(f"no route to {REPLY}"))
        with NtfyChannel(TOPIC, REPLY) as ntfy:
            with pytest.raises(NotifyError) as publish:
                ntfy.send(Message("hello"))
            with pytest.raises(NotifyError) as subscribe:
                ntfy.poll()
    for error in (publish.value, subscribe.value):
        assert TOPIC not in str(error) and REPLY not in str(error) and error.__suppress_context__
    assert str(publish.value) == f"ntfy publish: ConnectError: cannot reach {SERVER}/***"


def test_request_log_lines_hide_the_topics(
    server: FakeNtfy, channel: NtfyChannel, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO, logger="httpx")
    channel.poll(wait=False)
    channel.send(Message("hello"))
    assert "HTTP Request" in caplog.text and f"{SERVER}/***/json" in caplog.text
    assert TOPIC not in caplog.text and REPLY not in caplog.text


@pytest.mark.parametrize(
    ("topic", "reply_topic"),
    [(TOPIC, TOPIC), ("", REPLY), (TOPIC, "has space"), ("x" * 65, REPLY), (TOPIC, "slash/ed")],
)
def test_topics_must_be_valid_names_and_distinct(topic: str, reply_topic: str) -> None:
    with pytest.raises(NotifyError) as error:
        NtfyChannel(topic, reply_topic)
    assert TOPIC not in str(error.value)


def test_generated_topics_are_long_random_valid_names() -> None:
    topics = {generate_topic() for _ in range(20)}
    assert len(topics) == 20
    assert all(
        TOPIC_PATTERN.match(topic) and topic.startswith("espn-fantasy-") and len(topic) == 37 for topic in topics
    )


def test_from_config_names_the_missing_secrets() -> None:
    sample = FIXTURES / "config.sample.toml"
    with pytest.raises(NotifyError, match="NTFY_TOPIC, NTFY_REPLY_TOPIC missing"):
        NtfyChannel.from_config(load_config(sample, environ={}))
    ready = load_config(sample, environ={"NTFY_TOPIC": TOPIC, "NTFY_REPLY_TOPIC": REPLY})
    assert NtfyChannel.from_config(ready).name == "ntfy"


# --- decisions through fm.proposals -----------------------------------------------------------------------------------


def test_a_tapped_approve_records_the_approval_and_a_confirmation_push_follows(
    server: FakeNtfy, channel: NtfyChannel, store: Store, config: Config, league: LeagueRow
) -> None:
    row = lineup(store, config, league)
    notify_proposal(channel, store, row.row_id, now=NOW, tz=UTC)
    (pushed,) = server.phone()
    assert pushed["message"].startswith("Start: Sample Quarterback (QB)\n⏰ Decide by Sun 04 Oct 14:00 UTC")
    server.tap(pushed, "Approve")
    assert listen(store, channel, once=True, clock=lambda: at(1)) == 1
    approved = get_proposal(store, row.row_id)
    assert (approved.status, approved.decided_by) == ("approved", "ntfy")
    assert approved.execution_token is not None
    assert approved.execution_token not in json.dumps(server.published)  # buttons carry the nonce, never the token
    confirmation = server.phone()[-1]
    assert (confirmation["title"], confirmation["message"]) == (f"#{row.row_id} approved", SUMMARY)


def test_a_post_with_a_forged_token_is_ignored_without_a_push(
    server: FakeNtfy, channel: NtfyChannel, store: Store, config: Config, league: LeagueRow
) -> None:
    row = lineup(store, config, league)
    notify_proposal(channel, store, row.row_id, now=NOW)
    server.post(REPLY, f"approve:{row.row_id}:{NONCE}")  # knows the reply topic, not the nonce
    assert listen(store, channel, once=True, clock=lambda: at(1)) == 0
    assert get_proposal(store, row.row_id).status == "proposed"
    assert len(server.phone()) == 1  # no confirmation, no reply to the forger


def test_replies_replayed_after_a_restart_are_ignored(
    server: FakeNtfy, channel: NtfyChannel, store: Store, config: Config, league: LeagueRow
) -> None:
    row = lineup(store, config, league)
    notify_proposal(channel, store, row.row_id, now=NOW)
    server.tap(server.phone()[0], "Reject")
    assert listen(store, channel, once=True, clock=lambda: at(1)) == 1
    with NtfyChannel(TOPIC, REPLY) as restarted:  # no saved resume point: everything the server still caches
        assert listen(store, restarted, once=True, clock=lambda: at(5)) == 0
    rejected = get_proposal(store, row.row_id)
    assert (rejected.status, rejected.decided_by, rejected.decided_at) == ("rejected", "ntfy", at(1))


def test_a_restart_resumes_after_the_last_reply_seen(server: FakeNtfy, tmp_path: Path) -> None:
    since_file = tmp_path / "ntfy-since"
    server.post(REPLY, f"approve:12:{NONCE}")
    with NtfyChannel(TOPIC, REPLY, since_file=since_file) as first:
        (reply,) = first.poll(wait=False)
    assert since_file.read_text(encoding="utf-8") == reply.sender
    server.post(REPLY, f"reject:12:{NONCE}")
    with NtfyChannel(TOPIC, REPLY, since_file=since_file) as restarted:
        assert [later.decision for later in restarted.poll(wait=False)] == ["reject"]  # no replay of the approve
    assert [request.url.params["since"] for request in server.subscriptions] == ["all", reply.sender]


@pytest.mark.parametrize(("saved", "age_s"), [("", 0), ("not an id!", 0), ("x" * 40, 0), ("msg0001", 7 * 3600)])
def test_a_bad_or_stale_resume_point_starts_from_the_cache(
    server: FakeNtfy, tmp_path: Path, saved: str, age_s: int
) -> None:
    since_file = tmp_path / "ntfy-since"
    since_file.write_text(saved, encoding="utf-8")
    saved_at = time.time() - age_s  # older than RESUME_WINDOW_S: the id may have left the server's cache
    os.utime(since_file, (saved_at, saved_at))
    with NtfyChannel(TOPIC, REPLY, since_file=since_file) as ntfy:
        ntfy.poll(wait=False)
    assert server.subscriptions[0].url.params["since"] == "all"


def test_the_configured_channel_keeps_its_resume_point_in_the_config_dir(server: FakeNtfy) -> None:
    sample = FIXTURES / "config.sample.toml"
    server.post(REPLY, f"approve:12:{NONCE}")
    with NtfyChannel.from_config(load_config(sample, environ={"NTFY_TOPIC": TOPIC, "NTFY_REPLY_TOPIC": REPLY})) as ntfy:
        (reply,) = ntfy.poll(wait=False)
    assert (paths.config_dir() / "notify" / "ntfy-since").read_text(encoding="utf-8") == reply.sender


def test_an_expired_proposal_cannot_be_approved(
    server: FakeNtfy, channel: NtfyChannel, store: Store, config: Config, league: LeagueRow
) -> None:
    row = lineup(store, config, league)
    notify_proposal(channel, store, row.row_id, now=NOW)
    server.tap(server.phone()[0], "Approve")
    assert listen(store, channel, once=True, clock=lambda: DEADLINE + timedelta(minutes=10)) == 1
    expired = get_proposal(store, row.row_id)
    assert (expired.status, expired.execution_token) == ("expired", None)
    confirmation = server.phone()[-1]
    assert confirmation["title"] == f"#{row.row_id} not approved"
    assert "expired at 2026-10-04 14:00 UTC" in confirmation["message"]


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


def write_config(env: dict[str, str], *, channel: str = "ntfy") -> Path:
    """The sample config.toml with ``[notify].channel`` set, and a .env with ``env``; returns the .env path."""
    directory = paths.config_dir()
    sample = (FIXTURES / "config.sample.toml").read_text(encoding="utf-8")
    text = sample.replace('channel = "telegram"', f'channel = "{channel}"')
    assert f'channel = "{channel}"' in text
    (directory / "config.toml").write_text(text, encoding="utf-8")
    env_path = directory / ".env"
    env_path.write_text("".join(f"{name}={value}\n" for name, value in env.items()), encoding="utf-8")
    return env_path


def test_setup_generates_both_topics_and_sends_a_test_push(server: FakeNtfy) -> None:
    env_path = write_config({"ANTHROPIC_API_KEY": "placeholder"})
    output = run("notify", "setup")
    saved = dotenv_values(env_path)
    topic, reply_topic = saved["NTFY_TOPIC"], saved["NTFY_REPLY_TOPIC"]
    assert topic and reply_topic and topic != reply_topic
    assert saved["ANTHROPIC_API_KEY"] == "placeholder"  # the rest of .env is kept
    assert topic in output and reply_topic not in output  # the phone subscribes to one; the other stays hidden
    (test_push,) = server.published
    assert (test_push["topic"], test_push["title"]) == (topic, "espn-fantasy: notifications are set up")


def test_setup_keeps_existing_topics_unless_told_to_replace_them(server: FakeNtfy) -> None:
    env_path = write_config({"NTFY_TOPIC": TOPIC, "NTFY_REPLY_TOPIC": REPLY})
    resume_point = paths.config_dir() / "notify" / "ntfy-since"
    resume_point.parent.mkdir(parents=True, exist_ok=True)
    resume_point.write_text("msg0042", encoding="utf-8")
    assert "already set" in run("notify", "setup")
    assert dotenv_values(env_path) == {"NTFY_TOPIC": TOPIC, "NTFY_REPLY_TOPIC": REPLY}
    assert resume_point.exists()
    run("notify", "setup", "--replace")
    replaced = dotenv_values(env_path)
    assert replaced["NTFY_TOPIC"] not in (TOPIC, None) and replaced["NTFY_REPLY_TOPIC"] not in (REPLY, None)
    assert [push["topic"] for push in server.published] == [TOPIC, replaced["NTFY_TOPIC"]]
    assert not resume_point.exists()  # it pointed into the old reply topic


def test_setup_of_another_channel_than_configured_says_to_switch(server: FakeNtfy) -> None:
    write_config({}, channel="telegram")
    assert 'Set channel = "ntfy" under [notify]' in run("notify", "setup", "--channel", "ntfy")


def test_fm_bot_once_records_a_tapped_button(server: FakeNtfy, config: Config) -> None:
    write_config({"NTFY_TOPIC": TOPIC, "NTFY_REPLY_TOPIC": REPLY})
    with Store.open() as store:
        league = seed_league(store, config)
        row = propose(store, config, league, ProposalKind.LINEUP, LINEUP, created_by="test")
        with NtfyChannel(TOPIC, REPLY) as ntfy:
            notify_proposal(ntfy, store, row.row_id)
    server.tap(server.phone()[0], "Approve")
    output = run("bot", "--once")
    assert f"#{row.row_id} approved: {SUMMARY}" in output and "recorded 1 decision" in output
    with Store.open() as store:
        assert get_proposal(store, row.row_id).decided_by == "ntfy"
    assert server.phone()[-1]["title"] == f"#{row.row_id} approved"
