"""The proposal lifecycle: propose, approve, reject, expiry at the deadline, T-15 auto-approval, execution tokens.

Clocks are pinned with ``now=`` so deadline arithmetic is exact. Leagues come from ``fixtures/config.sample.toml``.
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from fm.config import Config, League, load_config
from fm.proposals import (
    AUTO_LEAD,
    DECIDED_BY_AUTO,
    DECIDED_BY_EXPIRY,
    TRANSITIONS,
    AddDropPayload,
    LifecycleError,
    LineupMove,
    LineupPayload,
    PausedError,
    PolicyError,
    ProposalError,
    ProposalKind,
    TradeResponsePayload,
    approve,
    auto_approve_due,
    begin_execution,
    expire_due,
    finish_execution,
    get_proposal,
    is_open,
    pause,
    propose,
    reject,
    resume,
)
from fm.store import LeagueRow, ProposalRow, ProposalStatus, Store

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
NOW = datetime(2026, 10, 4, 12, 0, tzinfo=UTC)
DEADLINE = NOW + timedelta(hours=2)
LINEUP = LineupPayload(moves=(LineupMove(espn_id=10, from_slot_id=20, to_slot_id=0),))


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        yield opened


@pytest.fixture
def config() -> Config:
    return load_config(FIXTURES / "config.sample.toml", environ={})


@pytest.fixture
def league(store: Store, config: Config) -> LeagueRow:
    return seed_league(store, config.league("nfl"))


def seed_league(store: Store, league: League) -> LeagueRow:
    return store.leagues.upsert(
        LeagueRow(
            key=league.key,
            sport=league.sport,
            espn_league_id=league.espn_league_id,
            season=league.season,
            team_id=league.team_id,
            as_of=NOW,
        )
    )


def lineup(
    store: Store,
    config: Config,
    league: LeagueRow,
    *,
    kind: ProposalKind = ProposalKind.LINEUP,
    deadline: datetime | None = DEADLINE,
    now: datetime = NOW,
    dedupe_key: str | None = None,
) -> ProposalRow:
    return propose(
        store, config, league, kind, LINEUP, created_by="test", deadline=deadline, now=now, dedupe_key=dedupe_key
    )


def status_of(store: Store, row: ProposalRow) -> ProposalStatus:
    return get_proposal(store, row.row_id).status


class TestPropose:
    def test_stores_the_row_under_the_setting_it_was_made_under(
        self, store: Store, config: Config, league: LeagueRow
    ) -> None:
        row = propose(
            store,
            config,
            league,
            "lineup",
            LINEUP,
            created_by="decide.lineup",
            scoring_period_id=4,
            engine_numbers={"delta_points": 3.2},
            rationale="X has no game",
            deadline=DEADLINE,
            dedupe_key="lineup:4:10",
            now=NOW,
        )
        assert row.id == 1 and is_open(row)
        assert (row.kind, row.status, row.policy) == ("lineup", "proposed", "approve")
        assert (row.league_id, row.scoring_period_id, row.created_by, row.created_at) == (
            league.row_id,
            4,
            "decide.lineup",
            NOW,
        )
        assert (row.payload, row.engine_numbers, row.rationale) == (
            LINEUP.model_dump(mode="json"),
            {"delta_points": 3.2},
            "X has no game",
        )
        assert (row.deadline, row.dedupe_key) == (DEADLINE, "lineup:4:10")
        assert row.decided_by is None and row.decided_at is None and row.execution_token is None
        assert store.proposals.get(1) == row

        auto = lineup(store, config, league, kind=ProposalKind.BENCH_INACTIVE)
        assert (auto.policy, auto.status) == ("auto", "proposed")  # auto still waits for T-15
        decline = TradeResponsePayload(other_team_id=2, get_espn_ids=(9,), espn_transaction_id="t1")
        trade = propose(store, config, league, ProposalKind.TRADE_DECLINE, decline, created_by="test", now=NOW)
        assert trade.policy == "approve"

    def test_blocked_proposals_are_not_stored(self, store: Store, config: Config, league: LeagueRow) -> None:
        with pytest.raises(PolicyError, match="lineup blocked: deadline 2026-10-04 12:00 UTC has passed"):
            lineup(store, config, league, deadline=NOW)
        assert store.proposals.find() == []
        assert issubclass(PolicyError, ProposalError) and issubclass(LifecycleError, ProposalError)

    def test_dedupe_key_returns_the_open_proposal(self, store: Store, config: Config, league: LeagueRow) -> None:
        first = lineup(store, config, league, dedupe_key="k")
        assert lineup(store, config, league, dedupe_key="k", now=NOW + timedelta(minutes=10)) == first
        approved = approve(store, first.row_id, decided_by="cli", now=NOW)
        assert lineup(store, config, league, dedupe_key="k") == approved  # approved is still open
        assert lineup(store, config, league, dedupe_key="other").id == 2
        assert len(store.proposals.find()) == 2

    def test_dedupe_ignores_closed_proposals_and_other_leagues(
        self, store: Store, config: Config, league: LeagueRow
    ) -> None:
        first = lineup(store, config, league, dedupe_key="k")
        reject(store, first.row_id, decided_by="cli", now=NOW)
        again = lineup(store, config, league, dedupe_key="k")
        assert again.id == 2 and again.status == "proposed"
        nba = seed_league(store, config.league("nba"))
        assert lineup(store, config, nba, dedupe_key="k").id == 3

    def test_a_stale_duplicate_is_expired_and_replaced(self, store: Store, config: Config, league: LeagueRow) -> None:
        first = lineup(store, config, league, dedupe_key="k")  # due at DEADLINE, not yet swept
        later = DEADLINE + timedelta(hours=1)
        next_deadline = later + timedelta(hours=1)
        again = lineup(store, config, league, dedupe_key="k", deadline=next_deadline, now=later)
        assert (again.id, again.status, again.deadline) == (2, "proposed", next_deadline)
        stale = get_proposal(store, first.row_id)
        assert (stale.status, stale.decided_by, stale.decided_at) == ("expired", DECIDED_BY_EXPIRY, later)
        assert lineup(store, config, league, dedupe_key="k", deadline=next_deadline, now=later) == again

    def test_an_add_drop_payload_goes_through_the_store_as_json(
        self, store: Store, config: Config, league: LeagueRow
    ) -> None:
        row = propose(
            store,
            config,
            league,
            ProposalKind.ADD_DROP,
            AddDropPayload(add_espn_id=1, drop_espn_id=2),
            created_by="t",
            now=NOW,
        )
        assert row.payload == {"add_espn_id": 1, "drop_espn_id": 2}


class TestDecisions:
    def test_approve_issues_a_single_use_token(self, store: Store, config: Config, league: LeagueRow) -> None:
        row = lineup(store, config, league)
        at = NOW + timedelta(minutes=5)
        approved = approve(store, row.row_id, decided_by="telegram", now=at)
        assert (approved.status, approved.decided_by, approved.decided_at) == ("approved", "telegram", at)
        assert approved.execution_token is not None and len(approved.execution_token) >= 32
        assert approved.token_consumed_at is None
        assert store.proposals.get(row.row_id) == approved
        other = approve(store, lineup(store, config, league).row_id, decided_by="cli", now=at)
        assert other.execution_token != approved.execution_token

    def test_approve_and_reject_refuse_a_decided_proposal(
        self, store: Store, config: Config, league: LeagueRow
    ) -> None:
        row = lineup(store, config, league)
        approve(store, row.row_id, decided_by="cli", now=NOW)
        with pytest.raises(LifecycleError, match="cannot approve proposal #1: it was approved by cli"):
            approve(store, row.row_id, decided_by="telegram", now=NOW)
        rejected = reject(store, lineup(store, config, league).row_id, decided_by="cli", now=NOW)
        assert rejected.status == "rejected" and rejected.execution_token is None
        with pytest.raises(LifecycleError, match="cannot approve proposal #2: it was rejected by cli"):
            approve(store, rejected.row_id, decided_by="cli", now=NOW)
        with pytest.raises(LifecycleError, match="cannot reject proposal #2: it was rejected by cli"):
            reject(store, rejected.row_id, decided_by="cli", now=NOW)

    def test_rejecting_an_approval_withdraws_it(self, store: Store, config: Config, league: LeagueRow) -> None:
        row = approve(store, lineup(store, config, league).row_id, decided_by="cli", now=NOW)
        later = NOW + timedelta(minutes=1)
        withdrawn = reject(store, row.row_id, decided_by="cli", now=later)
        assert (withdrawn.status, withdrawn.decided_at) == ("rejected", later)
        assert row.execution_token is not None
        with pytest.raises(LifecycleError, match="cannot execute proposal #1: it was rejected by cli"):
            begin_execution(store, row.row_id, row.execution_token, now=later)

    def test_unknown_proposal(self, store: Store) -> None:
        with pytest.raises(LifecycleError, match="no proposal #42"):
            approve(store, 42, decided_by="cli", now=NOW)
        with pytest.raises(LifecycleError, match="no proposal #42"):
            get_proposal(store, 42)

    def test_human_decisions_work_while_paused(self, store: Store, config: Config, league: LeagueRow) -> None:
        row = lineup(store, config, league)
        pause(now=NOW)
        try:
            assert approve(store, row.row_id, decided_by="cli", now=NOW).status == "approved"
            assert reject(store, row.row_id, decided_by="cli", now=NOW).status == "rejected"
        finally:
            resume()


class TestExpiry:
    def test_an_expired_proposal_cannot_be_approved(self, store: Store, config: Config, league: LeagueRow) -> None:
        row = lineup(store, config, league)
        with pytest.raises(LifecycleError, match="cannot approve proposal #1: it expired at 2026-10-04 14:00 UTC"):
            approve(store, row.row_id, decided_by="telegram", now=DEADLINE)
        expired = get_proposal(store, row.row_id)
        assert (expired.status, expired.decided_by, expired.decided_at) == ("expired", DECIDED_BY_EXPIRY, DEADLINE)
        assert not is_open(expired)
        with pytest.raises(LifecycleError, match="cannot reject proposal #1: it expired"):
            reject(store, row.row_id, decided_by="cli", now=DEADLINE)

    def test_an_approval_is_void_after_its_deadline(self, store: Store, config: Config, league: LeagueRow) -> None:
        row = approve(store, lineup(store, config, league).row_id, decided_by="cli", now=NOW)
        assert row.execution_token is not None
        with pytest.raises(LifecycleError, match="cannot execute proposal #1: it expired at 2026-10-04 14:00 UTC"):
            begin_execution(store, row.row_id, row.execution_token, now=DEADLINE + timedelta(minutes=1))
        assert status_of(store, row) == "expired"

    def test_expire_due_sweeps_proposed_and_approved_rows(
        self, store: Store, config: Config, league: LeagueRow
    ) -> None:
        due = lineup(store, config, league)
        due_approved = approve(store, lineup(store, config, league).row_id, decided_by="cli", now=NOW)
        later = lineup(store, config, league, deadline=DEADLINE + timedelta(hours=1))
        open_ended = lineup(store, config, league, deadline=None)
        executing = approve(store, lineup(store, config, league).row_id, decided_by="cli", now=NOW)
        assert executing.execution_token is not None
        begin_execution(store, executing.row_id, executing.execution_token, now=NOW)
        rejected = reject(store, lineup(store, config, league).row_id, decided_by="cli", now=NOW)

        assert expire_due(store, now=NOW) == []
        swept = expire_due(store, now=DEADLINE)
        assert [row.row_id for row in swept] == [due.row_id, due_approved.row_id]
        assert all(row.status == "expired" and row.decided_by == DECIDED_BY_EXPIRY for row in swept)
        assert status_of(store, later) == "proposed" and status_of(store, open_ended) == "proposed"
        assert status_of(store, executing) == "executing" and status_of(store, rejected) == "rejected"
        assert expire_due(store, now=DEADLINE) == []  # idempotent
        assert [row.row_id for row in expire_due(store, now=DEADLINE + timedelta(hours=1))] == [later.row_id]

    def test_expire_due_can_be_scoped_to_a_league(self, store: Store, config: Config, league: LeagueRow) -> None:
        nfl = lineup(store, config, league)
        nba = lineup(store, config, seed_league(store, config.league("nba")))
        assert [row.row_id for row in expire_due(store, now=DEADLINE, league_id=nba.league_id)] == [nba.row_id]
        assert status_of(store, nfl) == "proposed"


class TestAutoApproval:
    def test_auto_fires_only_inside_the_lead_and_before_the_deadline(
        self, store: Store, config: Config, league: LeagueRow
    ) -> None:
        auto = lineup(store, config, league, kind=ProposalKind.BENCH_INACTIVE)
        manual = lineup(store, config, league)  # policy approve: never auto
        assert auto.policy == "auto" and manual.policy == "approve"
        assert AUTO_LEAD == timedelta(minutes=15)

        assert auto_approve_due(store, now=DEADLINE - AUTO_LEAD - timedelta(seconds=1)) == []
        fired = auto_approve_due(store, now=DEADLINE - AUTO_LEAD)
        assert [row.row_id for row in fired] == [auto.row_id]
        (row,) = fired
        assert (row.status, row.decided_by, row.decided_at) == ("approved", DECIDED_BY_AUTO, DEADLINE - AUTO_LEAD)
        assert row.execution_token is not None
        assert status_of(store, manual) == "proposed"
        assert auto_approve_due(store, now=DEADLINE - timedelta(minutes=1)) == []  # already approved

    def test_a_missed_window_expires_instead_of_firing(self, store: Store, config: Config, league: LeagueRow) -> None:
        auto = lineup(store, config, league, kind=ProposalKind.BENCH_INACTIVE)
        assert auto_approve_due(store, now=DEADLINE) == []
        assert expire_due(store, now=DEADLINE) == [get_proposal(store, auto.row_id)]
        assert status_of(store, auto) == "expired"

    def test_a_human_answer_beats_auto(self, store: Store, config: Config, league: LeagueRow) -> None:
        auto = lineup(store, config, league, kind=ProposalKind.BENCH_INACTIVE)
        reject(store, auto.row_id, decided_by="telegram", now=NOW)
        assert auto_approve_due(store, now=DEADLINE - timedelta(minutes=5)) == []
        assert status_of(store, auto) == "rejected"

    def test_nothing_fires_while_paused(self, store: Store, config: Config, league: LeagueRow) -> None:
        auto = lineup(store, config, league, kind=ProposalKind.BENCH_INACTIVE)
        pause(now=NOW)
        try:
            assert auto_approve_due(store, now=DEADLINE - timedelta(minutes=5)) == []
            assert status_of(store, auto) == "proposed"
        finally:
            resume()
        assert len(auto_approve_due(store, now=DEADLINE - timedelta(minutes=5))) == 1

    def test_lead_is_adjustable(self, store: Store, config: Config, league: LeagueRow) -> None:
        lineup(store, config, league, kind=ProposalKind.BENCH_INACTIVE)
        assert auto_approve_due(store, now=DEADLINE - timedelta(minutes=30), lead=timedelta(minutes=20)) == []
        assert len(auto_approve_due(store, now=DEADLINE - timedelta(minutes=30), lead=timedelta(minutes=30))) == 1


class TestExecution:
    def test_token_is_consumed_exactly_once(self, store: Store, config: Config, league: LeagueRow) -> None:
        approved = approve(store, lineup(store, config, league).row_id, decided_by="cli", now=NOW)
        token = approved.execution_token
        assert token is not None
        at = NOW + timedelta(minutes=1)
        with pytest.raises(LifecycleError, match="execution token does not match or was already used"):
            begin_execution(store, approved.row_id, "not-the-token", now=at)
        assert status_of(store, approved) == "approved"

        executing = begin_execution(store, approved.row_id, token, now=at)
        assert (executing.status, executing.token_consumed_at, executing.execution_token) == ("executing", at, token)
        with pytest.raises(LifecycleError, match="cannot execute proposal #1: it is executing"):
            begin_execution(store, approved.row_id, token, now=at)
        assert store.proposals.consume_execution_token(approved.row_id, token, at) is False

    def test_only_approved_proposals_execute(self, store: Store, config: Config, league: LeagueRow) -> None:
        row = lineup(store, config, league)
        with pytest.raises(LifecycleError, match="cannot execute proposal #1: it is proposed"):
            begin_execution(store, row.row_id, "anything", now=NOW)

    def test_execution_is_refused_while_paused(self, store: Store, config: Config, league: LeagueRow) -> None:
        approved = approve(store, lineup(store, config, league).row_id, decided_by="cli", now=NOW)
        assert approved.execution_token is not None
        pause("incident", now=NOW)
        try:
            with pytest.raises(
                PausedError, match=r"cannot execute proposal #1: paused since 2026-10-04 12:00 UTC \(incident\)"
            ):
                begin_execution(store, approved.row_id, approved.execution_token, now=NOW)
        finally:
            resume()
        refreshed = get_proposal(store, approved.row_id)
        assert refreshed.status == "approved" and refreshed.token_consumed_at is None  # nothing consumed
        assert begin_execution(store, approved.row_id, approved.execution_token, now=NOW).status == "executing"

    def test_finish_records_the_outcome(self, store: Store, config: Config, league: LeagueRow) -> None:
        for outcome in ("verified", "failed"):
            approved = approve(store, lineup(store, config, league).row_id, decided_by="cli", now=NOW)
            assert approved.execution_token is not None
            begin_execution(store, approved.row_id, approved.execution_token, now=NOW)
            done = finish_execution(store, approved.row_id, outcome)
            assert done.status == outcome and not is_open(done)
            with pytest.raises(LifecycleError, match=f"cannot mark verified proposal #{done.row_id}: it is {outcome}"):
                finish_execution(store, approved.row_id, "verified")
        with pytest.raises(LifecycleError, match="cannot mark failed proposal #3: it is proposed"):
            finish_execution(store, lineup(store, config, league).row_id, "failed")


def test_transition_table_is_the_design_lifecycle() -> None:
    """proposed -> approved | rejected | expired -> executing -> verified | failed, plus withdrawing an approval."""
    assert set(TRANSITIONS) == {"proposed", "approved", "rejected", "expired", "executing", "verified", "failed"}
    assert TRANSITIONS["proposed"] == {"approved", "rejected", "expired"}
    assert TRANSITIONS["approved"] == {"executing", "rejected", "expired"}
    assert TRANSITIONS["executing"] == {"verified", "failed"}
    for terminal in ("rejected", "expired", "verified", "failed"):
        assert TRANSITIONS[terminal] == frozenset()
