"""The listener and the senders (``fm.notify.bot``, ``fm.notify.send``) over an in-memory channel: the trust boundary
in ``handle_reply``, ``listen``'s error handling, ``notify_proposal``'s guards, and the configured channel.

The HTTP adapters have their own tests (test_telegram.py, test_ntfy.py); here the channel is a fake that records what
it was asked to do. Leagues come from ``fixtures/config.sample.toml``; clocks are pinned.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fm.config import Config, Notify, load_config
from fm.notify import (
    INVALID_PRESS,
    Decision,
    DecisionResult,
    Message,
    NotifyChannel,
    NotifyError,
    NtfyChannel,
    ProposalNotice,
    Reply,
    TelegramChannel,
    handle_reply,
    listen,
    nonces,
    notify_proposal,
    open_channel,
    send_alert,
    send_report,
)
from fm.notify import bot as bot_module
from fm.notify.nonces import LATE_GRACE
from fm.proposals import (
    LifecycleError,
    LineupMove,
    LineupPayload,
    ProposalKind,
    get_proposal,
    pause,
    propose,
    reject,
)
from fm.store import LeagueRow, PlayerRow, ProposalRow, Store

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
DEADLINE = NOW + timedelta(hours=2)
LINEUP = LineupPayload(moves=(LineupMove(espn_id=10, from_slot_id=20, to_slot_id=0),))
SUMMARY = "nfl lineup change: Sample Quarterback: BE -> QB"


class FakeChannel:
    """A channel that records what it was asked to do; ``inbox`` is what each poll returns, or raises."""

    name = "fake"

    def __init__(self) -> None:
        self.sent: list[Message] = []
        self.notices: list[ProposalNotice] = []
        self.inbox: list[list[Reply] | Exception] = []
        self.polls: list[bool] = []
        self.confirmed: list[DecisionResult] = []
        self.dismissed: list[tuple[Reply, str]] = []
        self.fail_push = False
        self.fail_confirm = False

    def send(self, message: Message) -> None:
        self.sent.append(message)

    def send_proposal(self, notice: ProposalNotice) -> None:
        if self.fail_push:
            raise NotifyError("push failed")
        self.notices.append(notice)

    def poll(self, *, wait: bool = True) -> list[Reply]:
        self.polls.append(wait)
        item = self.inbox.pop(0) if self.inbox else []
        if isinstance(item, Exception):
            raise item
        return item

    def confirm(self, reply: Reply, result: DecisionResult) -> None:
        if self.fail_confirm:
            raise NotifyError("confirmation failed")
        self.confirmed.append(result)

    def dismiss(self, reply: Reply, reason: str) -> None:
        self.dismissed.append((reply, reason))

    def close(self) -> None:
        pass


def press(notice: ProposalNotice, decision: Decision = "approve", *, proposal_id: int | None = None) -> Reply:
    """What the phone sends back for one of ``notice``'s buttons (optionally tampered with)."""
    target = notice.proposal_id if proposal_id is None else proposal_id
    return Reply(channel="fake", proposal_id=target, decision=decision, nonce=notice.nonce, sender="phone")


@pytest.fixture
def channel() -> FakeChannel:
    return FakeChannel()


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        yield opened


@pytest.fixture
def config() -> Config:
    return load_config(FIXTURES / "config.sample.toml", environ={})


@pytest.fixture
def league(store: Store, config: Config) -> LeagueRow:
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


def lineup(store: Store, config: Config, league: LeagueRow, *, deadline: datetime | None = DEADLINE) -> ProposalRow:
    return propose(store, config, league, ProposalKind.LINEUP, LINEUP, created_by="test", deadline=deadline, now=NOW)


def at(minutes: int) -> datetime:
    return NOW + timedelta(minutes=minutes)


def test_the_fake_is_a_notify_channel(channel: FakeChannel) -> None:
    satisfied: NotifyChannel = channel
    assert satisfied.name == "fake"


# --- handle_reply: the trust boundary ---------------------------------------------------------------------------------


class TestHandleReply:
    def test_an_accepted_press_decides_through_fm_proposals_and_is_confirmed(
        self, channel: FakeChannel, store: Store, config: Config, league: LeagueRow
    ) -> None:
        row = lineup(store, config, league)
        notice = notify_proposal(channel, store, row.row_id, now=NOW)
        result = handle_reply(store, channel, press(notice), now=at(1))
        assert result == DecisionResult(row.row_id, "approve", ok=True, detail=SUMMARY)
        approved = get_proposal(store, row.row_id)
        assert (approved.status, approved.decided_by, approved.decided_at) == ("approved", "fake", at(1))
        assert channel.confirmed == [result] and channel.dismissed == []
        assert nonces.live(row.row_id, now=at(1)) == 0

    def test_a_reject_press_rejects(
        self, channel: FakeChannel, store: Store, config: Config, league: LeagueRow
    ) -> None:
        row = lineup(store, config, league)
        notice = notify_proposal(channel, store, row.row_id, now=NOW)
        result = handle_reply(store, channel, press(notice, "reject"), now=at(1))
        assert result is not None and (result.ok, result.headline) == (True, f"#{row.row_id} rejected")
        assert get_proposal(store, row.row_id).status == "rejected"

    def test_an_unknown_or_used_nonce_is_ignored_and_dismissed(
        self, channel: FakeChannel, store: Store, config: Config, league: LeagueRow
    ) -> None:
        row = lineup(store, config, league)
        notice = notify_proposal(channel, store, row.row_id, now=NOW)
        forged = Reply("fake", row.row_id, "approve", "Z" * 24, sender="phone")
        assert handle_reply(store, channel, forged, now=at(1)) is None
        assert get_proposal(store, row.row_id).status == "proposed"
        assert handle_reply(store, channel, press(notice, "reject"), now=at(2)) is not None
        assert handle_reply(store, channel, press(notice, "approve"), now=at(3)) is None  # the same button again
        assert get_proposal(store, row.row_id).status == "rejected"
        assert [reason for _, reason in channel.dismissed] == [INVALID_PRESS, INVALID_PRESS]

    def test_a_button_cannot_decide_another_proposal(
        self, channel: FakeChannel, store: Store, config: Config, league: LeagueRow
    ) -> None:
        first = lineup(store, config, league)
        second = propose(
            store,
            config,
            league,
            ProposalKind.LINEUP,
            LineupPayload(moves=(LineupMove(espn_id=11, from_slot_id=20, to_slot_id=2),)),
            created_by="test",
            deadline=DEADLINE,
            now=NOW,
        )
        notice = notify_proposal(channel, store, first.row_id, now=NOW)
        assert handle_reply(store, channel, press(notice, proposal_id=second.row_id), now=at(1)) is None
        assert get_proposal(store, second.row_id).status == "proposed"
        assert handle_reply(store, channel, press(notice), now=at(2)) is None  # the tampered-with nonce is burnt
        assert get_proposal(store, first.row_id).status == "proposed"

    def test_a_decision_fm_proposals_refuses_is_reported_back(
        self, channel: FakeChannel, store: Store, config: Config, league: LeagueRow
    ) -> None:
        row = lineup(store, config, league)
        notice = notify_proposal(channel, store, row.row_id, now=NOW)
        reject(store, row.row_id, decided_by="cli", now=at(1))
        result = handle_reply(store, channel, press(notice), now=at(2))
        assert result is not None and not result.ok
        assert result.detail == f"cannot approve proposal #{row.row_id}: it was rejected by cli"
        assert channel.confirmed == [result]
        assert get_proposal(store, row.row_id).decided_by == "cli"

    def test_an_expired_proposal_is_expired_on_the_way_and_never_approved(
        self, channel: FakeChannel, store: Store, config: Config, league: LeagueRow
    ) -> None:
        row = lineup(store, config, league)
        notice = notify_proposal(channel, store, row.row_id, now=NOW)
        result = handle_reply(store, channel, press(notice), now=DEADLINE)
        assert result is not None and not result.ok
        assert result.detail == f"cannot approve proposal #{row.row_id}: it expired at 2026-10-04 14:00 UTC"
        expired = get_proposal(store, row.row_id)
        assert (expired.status, expired.decided_by, expired.execution_token) == ("expired", "expiry", None)

    def test_a_press_after_the_grace_period_is_ignored(
        self, channel: FakeChannel, store: Store, config: Config, league: LeagueRow
    ) -> None:
        row = lineup(store, config, league)
        notice = notify_proposal(channel, store, row.row_id, now=NOW)
        assert handle_reply(store, channel, press(notice), now=DEADLINE + LATE_GRACE) is None
        assert get_proposal(store, row.row_id).status == "proposed"  # left for the tick's expire_due sweep

    def test_a_failed_confirmation_does_not_undo_the_decision(
        self, channel: FakeChannel, store: Store, config: Config, league: LeagueRow
    ) -> None:
        row = lineup(store, config, league)
        notice = notify_proposal(channel, store, row.row_id, now=NOW)
        channel.fail_confirm = True
        result = handle_reply(store, channel, press(notice), now=at(1))
        assert result is not None and result.ok
        assert get_proposal(store, row.row_id).status == "approved"

    def test_an_unexpected_failure_puts_the_nonce_back(
        self, channel: FakeChannel, store: Store, config: Config, league: LeagueRow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        row = lineup(store, config, league)
        notice = notify_proposal(channel, store, row.row_id, now=NOW)

        def locked(*args: object, **kwargs: object) -> ProposalRow:
            raise sqlite3.OperationalError("database is locked")

        with monkeypatch.context() as patch:
            patch.setattr(bot_module, "approve", locked)
            with pytest.raises(sqlite3.OperationalError):
                handle_reply(store, channel, press(notice), now=at(1))
        assert get_proposal(store, row.row_id).status == "proposed"
        assert channel.confirmed == [] and channel.dismissed == []
        result = handle_reply(store, channel, press(notice), now=at(2))  # pressing again works
        assert result is not None and result.ok

    def test_an_approval_while_paused_is_recorded_and_says_nothing_runs(
        self, channel: FakeChannel, store: Store, config: Config, league: LeagueRow
    ) -> None:
        row = lineup(store, config, league)
        notice = notify_proposal(channel, store, row.row_id, now=NOW)
        pause("vacation", now=at(1))
        result = handle_reply(store, channel, press(notice), now=at(2))
        assert result is not None and result.ok
        assert result.detail == (
            f"{SUMMARY} (paused since 2026-10-04 12:01 UTC (vacation): nothing runs until fm resume)"
        )
        assert get_proposal(store, row.row_id).status == "approved"


# --- listen -----------------------------------------------------------------------------------------------------------


class TestListen:
    def test_once_records_what_is_waiting_without_waiting(
        self, channel: FakeChannel, store: Store, config: Config, league: LeagueRow
    ) -> None:
        row = lineup(store, config, league)
        notice = notify_proposal(channel, store, row.row_id, now=NOW)
        forged = Reply("fake", row.row_id, "reject", "Z" * 24, sender="phone")
        channel.inbox.append([forged, press(notice)])
        results: list[DecisionResult] = []
        assert listen(store, channel, once=True, clock=lambda: at(1), on_result=results.append) == 1
        assert channel.polls == [False]
        assert [result.headline for result in results] == [f"#{row.row_id} approved"]

    def test_once_raises_a_channel_error(self, channel: FakeChannel, store: Store) -> None:
        channel.inbox.append(NotifyError("telegram getUpdates: HTTP 409: Conflict"))
        with pytest.raises(NotifyError, match="409"):
            listen(store, channel, once=True)

    def test_channel_errors_back_off_and_the_listener_keeps_going(
        self, channel: FakeChannel, store: Store, config: Config, league: LeagueRow
    ) -> None:
        row = lineup(store, config, league)
        notice = notify_proposal(channel, store, row.row_id, now=NOW)
        channel.inbox.extend([NotifyError("down"), NotifyError("still down"), [], [press(notice)]])
        sleeps: list[float] = []
        handled = listen(
            store,
            channel,
            stop=lambda: len(channel.polls) >= 4,
            sleep=sleeps.append,
            clock=lambda: at(1),
            interval_s=0.5,
        )
        assert handled == 1 and channel.polls == [True, True, True, True]
        assert sleeps == [2.0, 4.0, 0.5]  # backoff doubles, then the normal pause once the channel answers
        assert get_proposal(store, row.row_id).status == "approved"

    def test_a_database_error_loses_no_press(
        self,
        channel: FakeChannel,
        store: Store,
        config: Config,
        league: LeagueRow,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        row = lineup(store, config, league)
        notice = notify_proposal(channel, store, row.row_id, now=NOW)
        real_approve = bot_module.approve
        calls: list[int] = []

        def flaky(store: Store, proposal_id: int, *, decided_by: str, now: datetime | None = None) -> ProposalRow:
            calls.append(proposal_id)
            if len(calls) == 1:
                raise sqlite3.OperationalError("database is locked")
            return real_approve(store, proposal_id, decided_by=decided_by, now=now)

        monkeypatch.setattr(bot_module, "approve", flaky)
        channel.inbox.extend([[press(notice)], [press(notice)]])
        handled = listen(
            store, channel, stop=lambda: len(channel.polls) >= 2, sleep=lambda _: None, clock=lambda: at(1)
        )
        assert handled == 1 and calls == [row.row_id, row.row_id]
        assert get_proposal(store, row.row_id).status == "approved"

    def test_expired_nonces_are_swept_when_listening_starts(self, channel: FakeChannel, store: Store) -> None:
        nonces.issue(41, deadline=NOW, now=NOW - timedelta(hours=1))
        live = nonces.issue(42, now=NOW)
        listen(store, channel, once=True, clock=lambda: NOW + LATE_GRACE)
        assert sorted(path.suffix for path in nonces.nonce_dir().iterdir()) == [".json"]
        assert nonces.live(42, now=NOW + LATE_GRACE) == 1 and live.proposal_id == 42


# --- notify_proposal, alerts, reports, the configured channel ---------------------------------------------------------


class TestNotifyProposal:
    def test_a_proposed_proposal_goes_out_with_a_fresh_nonce_bound_to_it(
        self, channel: FakeChannel, store: Store, config: Config, league: LeagueRow
    ) -> None:
        row = lineup(store, config, league)
        notice = notify_proposal(channel, store, row.row_id, now=NOW, tz=UTC)
        assert channel.notices == [notice]
        assert notice.proposal_id == row.row_id and notice.message.title == f"🏈 NFL: Lineup change (#{row.row_id})"
        assert notice.message.priority == "high"
        assert nonces.live(row.row_id, now=DEADLINE + LATE_GRACE - timedelta(seconds=1)) == 1
        assert nonces.live(row.row_id, now=DEADLINE + LATE_GRACE) == 0
        again = notify_proposal(channel, store, row.row_id, now=at(5))
        assert again.nonce != notice.nonce and nonces.live(row.row_id, now=at(5)) == 2

    def test_without_a_deadline_the_buttons_last_a_week(
        self, channel: FakeChannel, store: Store, config: Config, league: LeagueRow
    ) -> None:
        row = lineup(store, config, league, deadline=None)
        notify_proposal(channel, store, row.row_id, now=NOW)
        assert nonces.live(row.row_id, now=NOW + timedelta(days=7) - timedelta(seconds=1)) == 1
        assert nonces.live(row.row_id, now=NOW + timedelta(days=7)) == 0

    def test_what_cannot_be_decided_is_not_sent(
        self, channel: FakeChannel, store: Store, config: Config, league: LeagueRow
    ) -> None:
        decided = lineup(store, config, league)
        reject(store, decided.row_id, decided_by="cli", now=NOW)
        due = lineup(store, config, league)
        with pytest.raises(NotifyError, match="no proposal #999"):
            notify_proposal(channel, store, 999, now=NOW)
        with pytest.raises(NotifyError, match="is rejected; only a proposed one"):
            notify_proposal(channel, store, decided.row_id, now=NOW)
        with pytest.raises(NotifyError, match="deadline 2026-10-04 14:00 UTC has passed"):
            notify_proposal(channel, store, due.row_id, now=DEADLINE)
        assert channel.notices == [] and nonces.live(now=NOW) == 0

    def test_a_failed_push_leaves_no_live_button(
        self, channel: FakeChannel, store: Store, config: Config, league: LeagueRow
    ) -> None:
        row = lineup(store, config, league)
        channel.fail_push = True
        with pytest.raises(NotifyError, match="push failed"):
            notify_proposal(channel, store, row.row_id, now=NOW)
        assert nonces.live(now=NOW) == 0

    def test_old_nonces_are_swept_before_a_new_one_is_issued(
        self, channel: FakeChannel, store: Store, config: Config, league: LeagueRow
    ) -> None:
        nonces.issue(77, deadline=NOW - timedelta(days=2), now=NOW - timedelta(days=3))
        row = lineup(store, config, league)
        notify_proposal(channel, store, row.row_id, now=NOW)
        assert len(list(nonces.nonce_dir().glob("*.json"))) == 1


def test_alerts_and_reports_are_pushed_and_returned(channel: FakeChannel) -> None:
    sent_alert = send_alert(channel, "Lineup not set", "start A over B", link="https://fantasy.espn.com/")
    sent_report = send_report(channel, "Weekly report", "body")
    assert channel.sent == [sent_alert, sent_report]
    assert (sent_alert.priority, sent_alert.link, sent_report.priority) == ("high", "https://fantasy.espn.com/", "low")


def test_the_configured_channel_is_opened_from_the_env_secrets() -> None:
    sample = FIXTURES / "config.sample.toml"
    telegram = load_config(sample, environ={"TELEGRAM_BOT_TOKEN": "123:abc", "TELEGRAM_CHAT_ID": "42"})
    assert isinstance(open_channel(telegram), TelegramChannel)
    ntfy = load_config(sample, environ={"NTFY_TOPIC": "phone-topic-x", "NTFY_REPLY_TOPIC": "reply-topic-y"})
    assert isinstance(open_channel(ntfy.model_copy(update={"notify": Notify(channel="ntfy")})), NtfyChannel)
    with pytest.raises(NotifyError, match="telegram is not set up"):
        open_channel(ntfy)


def test_a_decision_on_an_unknown_proposal_is_a_lifecycle_refusal(channel: FakeChannel, store: Store) -> None:
    issued = nonces.issue(404, now=NOW)
    result = handle_reply(store, channel, Reply("fake", 404, "approve", issued.nonce, sender="phone"), now=NOW)
    assert result is not None and not result.ok and result.detail == "no proposal #404"
    with pytest.raises(LifecycleError):
        get_proposal(store, 404)
