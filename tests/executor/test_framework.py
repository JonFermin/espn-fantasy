"""Executor framework (ROADMAP #19): flow registry, preconditions, one write, verification by re-read, audit, dry run.

Everything runs offline on ``fm.browser.fakes``: ESPN's read views come from ``tests/fixtures/espn`` through a fake read
API, writes go to a fake transport that records them, and UI mode clicks through a fake page. :class:`SwapFlow` is a
minimal lineup flow (the real one is ROADMAP #25): in the week-4 roster fixture our team 1 swaps Chuba Hubbard (FLEX)
with Tyler Allgeier (bench), and Jahmyr Gibbs is locked since Thursday night. ``open_live_runtime`` itself (what a dry
run installs in the browser) runs against a recording browser context in place of Playwright's.
"""

from __future__ import annotations

import functools
import importlib
import json
import sys
import types
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, NoReturn, cast

import httpx
import pytest
import typer
from playwright.sync_api import APIRequestContext
from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
from typer.testing import CliRunner

from fm import paths
from fm.browser import session as browser_session
from fm.browser.fakes import (
    FAKE_SWID,
    FakeBrowser,
    FakeElement,
    FakeEspnApi,
    FakeOpener,
    FakePage,
    FakeTransport,
    Reply,
    fake_runtime,
)
from fm.browser.flows import (
    DuplicateFlowError,
    Flow,
    FlowContext,
    FlowRegistry,
    Mode,
    ModeUnavailableError,
    Preconditions,
    UiDriver,
    UnknownFlowError,
    Verification,
    WriteOutcome,
    WriteRefusedError,
    WriteRequest,
    WriteResponse,
    WriteTimeoutError,
    WriteUncertainError,
    discover,
    rejection_allows_ui,
    transaction_id_from,
    transactions_url,
)
from fm.commands import execute as execute_cmd
from fm.config import load_config
from fm.espn.auth import AuthError, EspnSession, NotLoggedInError
from fm.executor import (
    STALE_EXECUTION,
    ExecutionResult,
    ExecutorOptions,
    NoFlowError,
    PlaywrightTransport,
    PreconditionReadError,
    RefusingTransport,
    dry_run_block_reason,
    execute,
    open_live_runtime,
    reconcile_executions,
)
from fm.executor import runtime as executor_runtime
from fm.proposals import (
    LifecycleError,
    LineupMove,
    LineupPayload,
    PausedError,
    ProposalKind,
    approve,
    begin_execution,
    get_proposal,
    pause,
    propose,
    resume,
)
from fm.store import ExecutionRow, LeagueRow, ProposalRow, Store

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ROSTERS = FIXTURES / "espn" / "ffl_rosters_week4.json"
NOW = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)
DEADLINE = NOW + timedelta(hours=2)
TEAM, WEEK = 1, 4
HUBBARD, ALLGEIER, GIBBS = 4241416, 4373626, 4429795
RB, BENCH, FLEX = 2, 20, 23
SWAP = LineupPayload(
    moves=(
        LineupMove(espn_id=HUBBARD, from_slot_id=FLEX, to_slot_id=BENCH),
        LineupMove(espn_id=ALLGEIER, from_slot_id=BENCH, to_slot_id=FLEX),
    )
)
LOCKED = LineupPayload(moves=(LineupMove(espn_id=GIBBS, from_slot_id=RB, to_slot_id=BENCH),))
TRANSACTIONS = (
    "https://lm-api-writes.fantasy.espn.com/apis/v3/games/ffl/seasons/2026/segments/0/leagues/1234567/transactions/"
)
ROSTER_PAGE = "https://fantasy.espn.com/football/team?leagueId=1234567&teamId=1"
SLOT_OPTIONS = ("0", "2", "4", "6", "16", "17", "20", "21", "23")
FAST = ExecutorOptions(
    write_timeout_s=5.0, ui_timeout_s=5.0, verify_attempts=2, verify_interval_s=0.0, sleep=lambda _s: None
)
FROM_ROW = "<the proposal's own token>"

runner = CliRunner()


@pytest.fixture(autouse=True)
def _no_real_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    """A slip past the fakes must fail the test, not start Edge on ESPN."""

    def refuse() -> NoReturn:
        raise AssertionError("real browser launch in an executor test; use fm.browser.fakes")

    monkeypatch.setattr(browser_session, "sync_playwright", refuse)


# --- a minimal lineup flow --------------------------------------------------------------------------------------------


class SwapFlow(Flow[LineupPayload]):
    """Lineup moves over ``mRoster``: what #25's ``set_lineup`` does, cut down to what the framework tests need."""

    name = "test_lineup"
    kinds = (ProposalKind.LINEUP, ProposalKind.BENCH_INACTIVE)
    payload_type = LineupPayload
    modes = (Mode.API, Mode.UI)

    def _slots(self, ctx: FlowContext[LineupPayload]) -> dict[int, tuple[int, bool]]:
        roster = ctx.reader.rosters(ctx.scoring_period_id).data.roster(ctx.team_id)
        return {entry.player_id: (entry.lineup_slot_id, entry.lineup_locked) for entry in roster.entries}

    def check(self, ctx: FlowContext[LineupPayload]) -> Preconditions:
        slots = self._slots(ctx)
        failures: list[str] = []
        for move in ctx.payload.moves:
            if move.espn_id not in slots:
                failures.append(f"player {move.espn_id} is not on team {ctx.team_id}")
                continue
            slot, locked = slots[move.espn_id]
            if locked:
                failures.append(f"player {move.espn_id} is locked")
            if slot != move.from_slot_id:
                failures.append(f"player {move.espn_id} is in slot {slot}, not {move.from_slot_id}")
        return Preconditions(tuple(failures), {"slots": {str(pid): slot for pid, (slot, _) in slots.items()}})

    def build_request(self, ctx: FlowContext[LineupPayload], pre: Preconditions) -> WriteRequest:
        return WriteRequest(url=ctx.transactions_url, body=envelope(ctx))

    def run_ui(self, ctx: FlowContext[LineupPayload], ui: UiDriver, pre: Preconditions) -> None:
        ui.page.goto(ROSTER_PAGE)
        for move in ctx.payload.moves:
            row = ui.page.get_by_role("row").filter(has_text=f"player {move.espn_id}")
            row.get_by_role("button", name="Move").click()
            ui.page.get_by_label(f"Slot for {move.espn_id}", exact=True).select_option(str(move.to_slot_id))
        ui.screenshot("lineup-set")
        ui.confirm(ui.page.get_by_role("button", name="Save lineup"), what="save lineup")

    def verify(self, ctx: FlowContext[LineupPayload], pre: Preconditions) -> Verification:
        slots = self._slots(ctx)
        expected = {str(move.espn_id): move.to_slot_id for move in ctx.payload.moves}
        observed = {str(move.espn_id): slots.get(move.espn_id, (None, False))[0] for move in ctx.payload.moves}
        detail = "slots match" if observed == expected else f"slots are {observed}, expected {expected}"
        return Verification(observed == expected, detail, expected, observed)


def envelope(ctx: FlowContext[LineupPayload]) -> dict[str, Any]:
    """The DESIGN 6.3 ``ROSTER`` envelope."""
    return {
        "isLeagueManager": False,
        "teamId": ctx.team_id,
        "type": "ROSTER",
        "memberId": ctx.member_id,
        "scoringPeriodId": ctx.scoring_period_id,
        "executionType": "EXECUTE",
        "items": [
            {
                "playerId": m.espn_id,
                "type": "LINEUP",
                "fromLineupSlotId": m.from_slot_id,
                "toLineupSlotId": m.to_slot_id,
            }
            for m in ctx.payload.moves
        ],
    }


class ApiOnlyFlow(SwapFlow):
    name = "api_only"
    modes = (Mode.API,)


class BadRequestFlow(SwapFlow):
    """Builds a request outside the write envelope the executor allows."""

    name = "bad_request"

    def __init__(self, *, url: str | None = None, headers: Mapping[str, str] | None = None, **body: Any) -> None:
        self.url = url
        self.headers = dict(headers or {})
        self.body = body

    def build_request(self, ctx: FlowContext[LineupPayload], pre: Preconditions) -> WriteRequest:
        good = super().build_request(ctx, pre)
        return WriteRequest(url=self.url or good.url, body={**good.body, **self.body}, headers=self.headers)


class UncapturedFlow(SwapFlow):
    """API mode is declared but this move's request is not captured yet, so only the UI can do it."""

    name = "uncaptured"

    def build_request(self, ctx: FlowContext[LineupPayload], pre: Preconditions) -> WriteRequest:
        raise ModeUnavailableError("this lineup request is not captured yet (#14)")


class InterruptedFlow(SwapFlow):
    """Ctrl+C lands while the flow is still building its request: nothing has been sent."""

    name = "interrupted"

    def build_request(self, ctx: FlowContext[LineupPayload], pre: Preconditions) -> WriteRequest:
        raise KeyboardInterrupt


class ExitingFlow(SwapFlow):
    """The process exits while the executor re-reads the league after ESPN accepted the write."""

    name = "exiting"

    def verify(self, ctx: FlowContext[LineupPayload], pre: Preconditions) -> Verification:
        raise SystemExit(3)


class RacingFlow(SwapFlow):
    """Another process spends the execution token while this run is still reading the preconditions."""

    name = "racing"

    def __init__(self, store: Store, token: str) -> None:
        self.store = store
        self.token = token

    def check(self, ctx: FlowContext[LineupPayload]) -> Preconditions:
        begin_execution(self.store, ctx.proposal.row_id, self.token, now=NOW)
        return super().check(ctx)


# --- the fake league --------------------------------------------------------------------------------------------------


class Harness:
    """The NFL fixture league with a fake read API, write transport, roster page and browser, plus a store."""

    def __init__(self, store: Store) -> None:
        self.store = store
        self.config = load_config(FIXTURES / "config.sample.toml", environ={})
        nfl = self.config.league("nfl")
        self.league = store.leagues.upsert(
            LeagueRow(
                key=nfl.key,
                sport=nfl.sport,
                espn_league_id=nfl.espn_league_id,
                season=nfl.season,
                team_id=TEAM,
                as_of=NOW,
            )
        )
        self.rosters: dict[str, Any] = json.loads(ROSTERS.read_text(encoding="utf-8"))
        self.statuses_at_read: list[tuple[str, ...]] = []
        self.api = FakeEspnApi()
        self.api.serve("mRoster", self._serve_rosters)
        self.transport = FakeTransport(on_send=self.apply_request)
        self.page, self.save = self._roster_page()
        self.browser = FakeBrowser(self.page)
        self.registry = FlowRegistry()
        self.registry.register(SwapFlow())
        self.opener = FakeOpener(fake_runtime(self.api, self.league, transport=self.transport, browser=self.browser))

    def propose(
        self,
        payload: LineupPayload = SWAP,
        *,
        kind: ProposalKind = ProposalKind.LINEUP,
        approved: bool = True,
        dedupe_key: str | None = None,
    ) -> ProposalRow:
        row = propose(
            self.store,
            self.config,
            self.league,
            kind,
            payload,
            created_by="test",
            scoring_period_id=WEEK,
            deadline=DEADLINE,
            dedupe_key=dedupe_key,
            now=NOW,
        )
        return approve(self.store, row.row_id, decided_by="test", now=NOW) if approved else row

    def run(
        self,
        proposal: ProposalRow,
        token: str | None = FROM_ROW,
        *,
        dry_run: bool = False,
        mode: Mode | None = None,
        registry: FlowRegistry | None = None,
    ) -> ExecutionResult:
        return execute(
            self.store,
            proposal.row_id,
            token=proposal.execution_token if token == FROM_ROW else token,
            dry_run=dry_run,
            mode=mode,
            opener=self.opener,
            registry=registry if registry is not None else self.registry,
            options=FAST,
            clock=lambda: NOW,
        )

    def slot(self, espn_id: int) -> int:
        return next(entry["lineupSlotId"] for entry in self._entries() if entry["playerId"] == espn_id)

    def apply_request(self, request: WriteRequest) -> None:
        """ESPN applying an API-mode write that lands."""
        self._move((item["playerId"], item["toLineupSlotId"]) for item in request.body["items"])

    def _entries(self) -> list[dict[str, Any]]:
        team = next(team for team in self.rosters["teams"] if team["id"] == TEAM)
        return team["roster"]["entries"]

    def _move(self, moves: Any) -> None:
        wanted = dict(moves)
        for entry in self._entries():
            if entry["playerId"] in wanted:
                entry["lineupSlotId"] = wanted[entry["playerId"]]

    def _serve_rosters(self, request: httpx.Request) -> dict[str, Any]:
        executions = [e for p in self.store.proposals.find() for e in self.store.executions.for_proposal(p.row_id)]
        self.statuses_at_read.append(tuple(execution.status for execution in executions))
        return self.rosters

    def _roster_page(self) -> tuple[FakePage, FakeElement]:
        """The roster page: a row with a Move button and a slot picker per player, and one Save button. Every visit
        renders fresh rows and pickers, as a reload would; Save stays the same control so tests can rig and count it."""
        selects: dict[int, FakeElement] = {}

        def save_selected(page: FakePage) -> None:  # the ESPN page applying the slots picked on this visit
            self._move((pid, int(select.value)) for pid, select in selects.items() if select.value)

        save = FakeElement(role="button", name="Save lineup", on_click=save_selected)

        def render() -> list[FakeElement]:
            selects.clear()
            rows: list[FakeElement] = []
            for entry in self._entries():
                pid = entry["playerId"]
                rows.append(FakeElement(role="row", text=f"player {pid}").add(FakeElement(role="button", name="Move")))
                selects[pid] = FakeElement(role="combobox", label=f"Slot for {pid}", options=SLOT_OPTIONS)
            return [FakeElement(role="table").add(*rows), *selects.values(), save]

        return FakePage(screens={ROSTER_PAGE: render}), save


@pytest.fixture
def h() -> Iterator[Harness]:
    with Store.open() as store:  # paths.state_db() in the per-test config dir, the one fm execute opens too
        yield Harness(store)


def artifact_names(row: Any) -> set[str]:
    return {Path(relative).name for relative in row.artifacts}


# --- the happy path ---------------------------------------------------------------------------------------------------


def test_api_mode_writes_once_and_is_verified_by_a_re_read(h: Harness) -> None:
    proposal = h.propose()
    result = h.run(proposal)

    assert result.ok and result.status == "verified" and result.flow == "test_lineup"
    assert result.proposal.status == "verified" and result.proposal.token_consumed_at == NOW
    (request,) = h.transport.sent
    assert request.url == TRANSACTIONS and h.transport.timeouts == [FAST.write_timeout_s]
    assert request.body == {
        "isLeagueManager": False,
        "teamId": TEAM,
        "type": "ROSTER",
        "memberId": FAKE_SWID,
        "scoringPeriodId": WEEK,
        "executionType": "EXECUTE",
        "items": [
            {"playerId": HUBBARD, "type": "LINEUP", "fromLineupSlotId": FLEX, "toLineupSlotId": BENCH},
            {"playerId": ALLGEIER, "type": "LINEUP", "fromLineupSlotId": BENCH, "toLineupSlotId": FLEX},
        ],
    }
    (attempt,) = result.attempts
    assert (attempt.mode, attempt.status, attempt.error) == ("api", "verified", None)
    assert attempt.request is not None and attempt.request["body"] == request.body
    assert attempt.response is not None and attempt.response["status"] == 200
    assert attempt.verification is not None and attempt.verification["matched"] is True
    assert attempt.espn_transaction_id == "fake-transaction-1"
    assert h.store.executions.for_proposal(proposal.row_id) == [attempt]
    assert (h.slot(HUBBARD), h.slot(ALLGEIER)) == (BENCH, FLEX)
    assert h.api.reads("mRoster") == 2  # the preconditions, then one re-read that matched
    assert h.browser.opened == [] and h.opener.calls == [(h.league.row_id, False)]


def test_every_attempt_leaves_its_evidence_in_the_audit_folder(h: Harness) -> None:
    proposal = h.propose()
    result = h.run(proposal)

    (attempt,) = result.attempts
    assert result.audit_dir.parent.parent == paths.audit_dir()
    assert {"preconditions.json", "api-request.json", "api-response.json", "api-verification.json"} <= artifact_names(
        attempt
    )
    for relative in attempt.artifacts:
        assert (paths.audit_dir() / relative).is_file(), relative
    saved = json.loads((result.audit_dir / "api-request.json").read_text(encoding="utf-8"))
    assert saved == {"method": "POST", "url": TRANSACTIONS, "headers": {}, "body": h.transport.sent[0].body}
    context = json.loads((result.audit_dir / "preconditions.json").read_text(encoding="utf-8"))
    assert context["flow"] == "test_lineup" and context["preconditions"]["ok"] is True
    assert context["proposal"]["payload"] == SWAP.model_dump(mode="json")
    token = proposal.execution_token
    assert token is not None
    for file in result.audit_dir.iterdir():  # the token authorizes a write; it never leaves the proposal row
        assert token not in file.read_text(encoding="utf-8", errors="replace")


# --- AC: a failed precondition blocks the run -------------------------------------------------------------------------


def test_a_failed_precondition_blocks_the_run(h: Harness) -> None:
    proposal = h.propose(LOCKED)
    result = h.run(proposal)

    assert not result.ok and not result.preconditions.ok
    assert h.transport.sent == [] and h.browser.opened == []  # nothing written, not even a page opened
    (attempt,) = result.attempts
    assert attempt.status == "failed" and attempt.request is None and attempt.response is None
    assert attempt.error == f"preconditions failed, nothing was sent: player {GIBBS} is locked"
    assert result.proposal.status == "failed" and result.proposal.token_consumed_at is not None
    assert h.slot(GIBBS) == RB
    assert h.api.reads("mRoster") == 1  # the preconditions only: there was no write to verify


def test_a_dry_run_reports_failed_preconditions_and_changes_nothing(h: Harness) -> None:
    proposal = h.propose(LOCKED)
    result = h.run(proposal, None, dry_run=True)

    assert not result.ok and result.status == "dry_run"
    assert result.last is not None and "is locked" in (result.last.error or "")
    assert h.transport.sent == [] and get_proposal(h.store, proposal.row_id) == proposal


# --- AC: the execution token can't be reused --------------------------------------------------------------------------


def test_the_execution_token_cannot_be_reused(h: Harness) -> None:
    proposal = h.propose()
    assert h.run(proposal).ok

    with pytest.raises(LifecycleError, match=r"cannot execute proposal #1: it is verified"):
        h.run(proposal, proposal.execution_token)
    assert len(h.transport.sent) == 1
    assert len(h.store.executions.for_proposal(proposal.row_id)) == 1
    assert h.opener.calls == [(h.league.row_id, False)]  # the second run never got as far as the browser


@pytest.mark.parametrize("token", ["not-the-token", "", None])
def test_a_wrong_or_missing_token_is_refused_before_anything_happens(h: Harness, token: str | None) -> None:
    proposal = h.propose()
    with pytest.raises(LifecycleError, match="execution token does not match or was already used"):
        h.run(proposal, token)

    assert h.opener.calls == [] and h.api.reads() == 0 and h.transport.sent == []
    row = get_proposal(h.store, proposal.row_id)
    assert row.status == "approved" and row.token_consumed_at is None
    assert h.run(proposal).ok  # the real token still works


def test_a_token_spent_by_another_process_mid_run_loses_at_begin_execution(h: Harness) -> None:
    proposal = h.propose()
    token = proposal.execution_token
    assert token is not None
    racing = FlowRegistry()
    racing.register(RacingFlow(h.store, token))

    with pytest.raises(LifecycleError):
        h.run(proposal, token, registry=racing)
    assert h.transport.sent == [] and h.store.executions.for_proposal(proposal.row_id) == []
    assert get_proposal(h.store, proposal.row_id).status == "executing"  # the other process owns it now


# --- AC: a timeout is UNKNOWN, with no retry --------------------------------------------------------------------------


def test_a_write_timeout_is_unknown_and_never_retried(h: Harness) -> None:
    h.transport.replies.append(Reply.timed_out())
    proposal = h.propose()
    result = h.run(proposal)

    assert len(h.transport.sent) == 1 and h.transport.timeouts == [FAST.write_timeout_s]
    assert h.browser.opened == []  # no UI fallback after an unknown: that would be a second write
    (attempt,) = result.attempts
    assert attempt.status == "unknown"
    assert attempt.error is not None and "timed out after 5 s" in attempt.error and "check the league" in attempt.error
    assert attempt.verification is not None
    assert (attempt.verification["after"], attempt.verification["matched"]) == ("unknown", False)
    assert attempt.verification["reads"] == FAST.verify_attempts
    # The forced re-read happened, and the row already said unknown when it did.
    assert h.api.reads("mRoster") == 1 + FAST.verify_attempts
    assert h.statuses_at_read[1:] == [("unknown",)] * FAST.verify_attempts
    assert result.proposal.status == "failed" and not result.ok  # unverified means failed


def test_a_timed_out_write_that_landed_is_verified_by_the_forced_re_read(h: Harness) -> None:
    h.transport.replies.append(Reply.timed_out(applies=True))
    result = h.run(h.propose())

    assert result.ok and result.proposal.status == "verified" and len(h.transport.sent) == 1
    (attempt,) = result.attempts
    assert attempt.status == "verified"
    assert attempt.error is not None and "timed out" in attempt.error and "went through" in attempt.error


@pytest.mark.parametrize(
    "reply",
    [Reply(status=503), Reply(error=WriteUncertainError("connection reset after the request went out"))],
    ids=["5xx", "lost-connection"],
)
def test_a_5xx_or_lost_connection_is_unknown_too(h: Harness, reply: Reply) -> None:
    h.transport.replies.append(reply)
    result = h.run(h.propose())

    (attempt,) = result.attempts
    assert attempt.status == "unknown" and len(h.transport.sent) == 1 and h.browser.opened == []
    assert result.proposal.status == "failed"


# --- AC: a verify mismatch fails --------------------------------------------------------------------------------------


def test_a_write_the_re_read_does_not_show_fails(h: Harness) -> None:
    h.transport.replies.append(Reply(applies=False))  # ESPN answers 200, but the roster never changes
    result = h.run(h.propose())

    (attempt,) = result.attempts
    assert attempt.status == "failed"
    assert attempt.error is not None and attempt.error.startswith("the re-read does not show the change")
    assert attempt.verification is not None and attempt.verification["matched"] is False
    assert attempt.verification["reads"] == FAST.verify_attempts
    assert attempt.verification["observed"] == {str(HUBBARD): FLEX, str(ALLGEIER): BENCH}
    assert result.proposal.status == "failed" and len(h.transport.sent) == 1 and h.browser.opened == []


def test_a_write_that_cannot_be_re_read_is_unknown(h: Harness) -> None:
    h.transport.on_send = lambda request: h.api.fail(503, view="mRoster", times=FAST.verify_attempts)
    result = h.run(h.propose())

    (attempt,) = result.attempts
    assert attempt.status == "unknown" and attempt.error is not None and "could not re-read" in attempt.error
    assert attempt.verification is not None and attempt.verification["matched"] is None
    assert result.proposal.status == "failed"


# --- AC: a dry run sends nothing --------------------------------------------------------------------------------------


def test_a_dry_run_builds_and_saves_the_request_but_sends_nothing(h: Harness) -> None:
    proposal = h.propose(approved=False)  # a dry run needs no approval and no token
    result = h.run(proposal, None, dry_run=True)

    assert result.ok and result.dry_run and result.status == "dry_run"
    assert h.transport.sent == [] and h.browser.opened == []
    (attempt,) = result.attempts
    assert (attempt.mode, attempt.status, attempt.error) == ("api", "dry_run", None)
    assert attempt.request is not None and attempt.request["url"] == TRANSACTIONS
    assert attempt.request["body"]["items"][0] == {
        "playerId": HUBBARD,
        "type": "LINEUP",
        "fromLineupSlotId": FLEX,
        "toLineupSlotId": BENCH,
    }
    assert attempt.response is None and attempt.verification is None
    assert "api-request.json" in artifact_names(attempt) and result.audit_dir.name.endswith("-dry")
    assert get_proposal(h.store, proposal.row_id) == proposal
    assert h.slot(HUBBARD) == FLEX and h.api.reads("mRoster") == 1
    assert h.opener.calls == [(h.league.row_id, True)]
    (runtime,) = h.opener.opened  # as live, the dry run held a transport that refuses to send
    assert isinstance(runtime.transport, RefusingTransport) and runtime.reader is h.opener.runtime.reader


def test_a_dry_run_leaves_an_approved_proposal_ready_to_execute(h: Harness) -> None:
    proposal = h.propose()
    assert h.run(proposal, None, dry_run=True).ok

    row = get_proposal(h.store, proposal.row_id)
    assert row.status == "approved" and row.token_consumed_at is None
    assert h.transport.sent == []
    assert h.run(proposal).ok and len(h.transport.sent) == 1


def test_a_ui_dry_run_walks_up_to_the_final_confirm_and_stops(h: Harness) -> None:
    result = h.run(h.propose(), None, dry_run=True, mode=Mode.UI)

    assert result.ok
    (attempt,) = result.attempts
    assert (attempt.mode, attempt.status, attempt.error) == ("ui", "dry_run", None)
    assert attempt.request == {"mode": "ui", "url": ROSTER_PAGE, "confirms": [], "stopped_before": "save lineup"}
    assert h.save.clicks == 0 and h.transport.sent == [] and h.slot(HUBBARD) == FLEX
    assert (
        h.page.did("click").count("role=row name=None >> has_text='player 4241416' >> role=button name='Move'") == 1
    )  # the walk itself happened
    assert h.page.did("select") == [
        "label='Slot for 4241416' = '20'",
        "label='Slot for 4373626' = '23'",
    ]
    names = artifact_names(attempt)
    assert {"ui-01-lineup-set.png", "ui-02-before-save-lineup.png", "ui-trace.zip"} <= names
    assert all((paths.audit_dir() / relative).is_file() for relative in attempt.artifacts)
    assert h.page.closed and not h.browser.tracing


# --- UI mode as the fallback ------------------------------------------------------------------------------------------


def test_a_broken_api_path_falls_back_to_the_ui(h: Harness) -> None:
    h.transport.replies.append(Reply(status=400, body={"messages": ["Bad request"]}))
    result = h.run(h.propose())

    api, ui = result.attempts
    assert (api.mode, api.status) == ("api", "failed")
    assert api.error is not None and api.error.startswith("ESPN rejected the request: HTTP 400: Bad request")
    assert api.verification is not None and api.verification["after"] == "rejection"  # re-read before the fallback
    assert api.verification["matched"] is False and "api-reread.json" in artifact_names(api)
    assert (ui.mode, ui.status, ui.error) == ("ui", "verified", None)
    assert ui.request is not None and ui.request["confirms"] == ["save lineup"]
    assert {"ui-02-before-save-lineup.png", "ui-03-after-save-lineup.png", "ui-trace.zip"} <= artifact_names(ui)
    assert result.proposal.status == "verified" and result.ok
    assert len(h.transport.sent) == 1 and h.save.clicks == 1
    assert (h.slot(HUBBARD), h.slot(ALLGEIER)) == (BENCH, FLEX)
    assert h.page.timeout_ms == FAST.ui_timeout_s * 1000  # every UI step has the hard timeout


def test_a_move_api_mode_cannot_send_goes_straight_to_the_ui(h: Harness) -> None:
    registry = FlowRegistry()
    registry.register(UncapturedFlow())

    dry = h.run(h.propose(), None, dry_run=True, registry=registry)
    assert dry.ok and [(row.mode, row.status) for row in dry.attempts] == [("api", "dry_run"), ("ui", "dry_run")]
    assert dry.attempts[0].error == "API mode unavailable: this lineup request is not captured yet (#14)"
    assert dry.attempts[1].error is None and h.save.clicks == 0

    result = h.run(h.propose(LineupPayload(moves=SWAP.moves[:1])), registry=registry)
    api, ui = result.attempts
    assert (api.status, ui.status) == ("failed", "verified") and api.request is None
    assert result.proposal.status == "verified" and h.transport.sent == [] and h.save.clicks == 1
    assert (h.slot(HUBBARD), h.slot(ALLGEIER)) == (BENCH, BENCH)  # the dry run's picks did not carry over


@pytest.mark.parametrize(
    "reply",
    [
        Reply.espn_error(409, "TRAN_LINEUP_LOCKED", "Lineup is locked"),
        Reply.espn_error(401, "AUTH_MISSING_CREDENTIALS"),
    ],
    ids=["league-rule", "auth"],
)
def test_a_league_rule_or_auth_rejection_does_not_fall_back(h: Harness, reply: Reply) -> None:
    h.transport.replies.append(reply)
    result = h.run(h.propose())

    (attempt,) = result.attempts
    assert (
        attempt.status == "failed"
        and attempt.error is not None
        and str(reply.body["details"][0]["type"]) in attempt.error
    )
    assert h.browser.opened == [] and h.save.clicks == 0
    assert result.proposal.status == "failed"


def test_a_ui_confirm_click_that_times_out_is_unknown(h: Harness) -> None:
    h.save.error = PlaywrightTimeoutError("Timeout 5000ms exceeded.")
    result = h.run(h.propose(), mode=Mode.UI)

    (attempt,) = result.attempts
    assert attempt.mode == "ui" and attempt.status == "unknown"
    assert attempt.error is not None and "failed after a confirm click" in attempt.error
    assert {"ui-02-before-save-lineup.png", "ui-03-error.png", "ui-trace.zip"} <= artifact_names(attempt)
    assert result.proposal.status == "failed" and h.transport.sent == []


def test_a_missing_ui_control_fails_before_anything_is_clicked(h: Harness) -> None:
    h.save.visible = False  # the page drifted: the Save button is gone
    result = h.run(h.propose(), mode=Mode.UI)

    (attempt,) = result.attempts
    assert attempt.status == "failed"  # definite: nothing was clicked, so nothing can have changed
    assert attempt.error is not None and "before any confirm" in attempt.error and "TimeoutError" in attempt.error
    assert h.save.clicks == 0 and result.proposal.status == "failed"


# --- refusals: nothing read, nothing sent, the proposal unchanged -----------------------------------------------------


def test_pause_refuses_execution_but_not_a_dry_run(h: Harness) -> None:
    proposal = h.propose()
    pause("vacation")

    with pytest.raises(PausedError, match="paused since .* \\(vacation\\); run fm resume"):
        h.run(proposal)
    assert h.opener.calls == [] and h.transport.sent == []
    assert get_proposal(h.store, proposal.row_id).status == "approved"
    assert h.run(proposal, None, dry_run=True).ok and h.transport.sent == []

    resume()
    assert h.run(proposal).ok


def test_an_unreadable_league_refuses_the_run_without_spending_the_token(h: Harness) -> None:
    proposal = h.propose()
    h.api.fail(503, view="mRoster")
    with pytest.raises(PreconditionReadError, match="nothing was sent"):
        h.run(proposal)
    h.api.fail(401, view="mRoster")
    with pytest.raises(AuthError, match="fm login"):
        h.run(proposal)

    row = get_proposal(h.store, proposal.row_id)
    assert row.status == "approved" and row.token_consumed_at is None
    assert h.transport.sent == [] and h.store.executions.for_proposal(proposal.row_id) == []
    assert h.run(proposal).ok  # once ESPN answers again it runs


def test_a_kind_or_mode_without_a_flow_is_refused(h: Harness) -> None:
    proposal = h.propose()
    with pytest.raises(NoFlowError, match="no executor flow for lineup proposals in nfl; registered: none"):
        h.run(proposal, registry=FlowRegistry())
    api_only = FlowRegistry()
    api_only.register(ApiOnlyFlow())
    with pytest.raises(NoFlowError, match="api_only has no ui mode; it runs in api"):
        execute(
            h.store,
            proposal.row_id,
            token=proposal.execution_token,
            mode="ui",
            opener=h.opener,
            registry=api_only,
            clock=lambda: NOW,
        )

    assert h.opener.calls == [] and get_proposal(h.store, proposal.row_id).status == "approved"


def test_proposals_that_are_not_approved_or_past_their_deadline_are_refused(h: Harness) -> None:
    proposed = h.propose(approved=False)
    with pytest.raises(LifecycleError, match=r"it is proposed; approve it first \(fm proposals approve 1\)"):
        h.run(proposed, "anything")

    late = h.propose(LOCKED)
    with pytest.raises(LifecycleError, match="expired at 2026-10-04 17:00 UTC"):
        execute(h.store, late.row_id, token=late.execution_token, opener=h.opener, clock=lambda: DEADLINE)
    assert get_proposal(h.store, late.row_id).status == "expired"
    with pytest.raises(LifecycleError, match="cannot dry-run proposal #2: it is expired"):
        h.run(late, None, dry_run=True)
    assert h.opener.calls == []


# --- the write envelope every request must fit ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flow", "problem"),
    [
        (BadRequestFlow(isLeagueManager=True), "isLeagueManager must be false"),
        (BadRequestFlow(teamId=2), "teamId must be our team 1, not 2"),
        (
            BadRequestFlow(url=TRANSACTIONS.replace("lm-api-writes", "lm-api-reads")),
            "writes go only to https://lm-api-writes.fantasy.espn.com",
        ),
        (BadRequestFlow(url=transactions_url("nfl", 2026, 7654321)), "is not ESPN nfl league 1234567 season 2026"),
        (BadRequestFlow(headers={"Cookie": "espn_s2=secret"}), "credential headers Cookie"),
    ],
    ids=["league-manager", "other-team", "reads-host", "other-league", "cookie"],
)
def test_requests_outside_the_write_envelope_are_never_sent(h: Harness, flow: BadRequestFlow, problem: str) -> None:
    registry = FlowRegistry()
    registry.register(flow)
    result = h.run(h.propose(), registry=registry)

    (attempt,) = result.attempts  # refused outright: no UI fallback for a request that should never exist
    assert attempt.status == "failed" and attempt.error is not None and problem in attempt.error
    assert h.transport.sent == [] and h.browser.opened == [] and result.proposal.status == "failed"
    saved = (result.audit_dir / "api-request.json").read_text(encoding="utf-8")
    assert "secret" not in saved and "secret" not in json.dumps(attempt.request)  # credential headers are masked


# --- registry ---------------------------------------------------------------------------------------------------------


def test_the_registry_checks_flows_and_refuses_double_claims() -> None:
    registry = FlowRegistry()
    flow = registry.register(SwapFlow())

    assert registry.flow_for("lineup", "ffl") is flow and registry.flow_for(ProposalKind.BENCH_INACTIVE, "nba") is flow
    assert (ProposalKind.LINEUP, "nfl") in registry and ("add_drop", "nfl") not in registry and len(registry) == 4
    with pytest.raises(DuplicateFlowError, match="nfl:lineup is served by SwapFlow\\(test_lineup\\) already"):
        registry.register(ApiOnlyFlow())
    with pytest.raises(UnknownFlowError, match="registered: nfl:lineup -> test_lineup, nfl:bench_inactive"):
        registry.flow_for("add_drop", "nfl")

    class WrongPayload(SwapFlow):
        name = "wrong_payload"
        kinds = (ProposalKind.ADD_DROP,)

    with pytest.raises(TypeError, match="takes LineupPayload, but add_drop proposals carry AddDropPayload"):
        registry.register(WrongPayload())

    class ClaimsUi(Flow[LineupPayload]):
        name = "claims_ui"
        kinds = (ProposalKind.LINEUP,)
        payload_type = LineupPayload

        def check(self, ctx: FlowContext[LineupPayload]) -> Preconditions:
            return Preconditions()

        def build_request(self, ctx: FlowContext[LineupPayload], pre: Preconditions) -> WriteRequest:
            return WriteRequest(url=ctx.transactions_url, body={})

        def verify(self, ctx: FlowContext[LineupPayload], pre: Preconditions) -> Verification:
            return Verification(True)

    with pytest.raises(TypeError, match="claims_ui lists UI mode but does not implement run_ui"):
        FlowRegistry().register(ClaimsUi())
    assert len(registry) == 4  # failed registrations leave nothing behind
    assert registry.unregister(flow) == 4 and len(registry) == 0


def test_discover_imports_each_public_flow_module_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    package = tmp_path / "fake_flows"
    package.mkdir()
    (package / "__init__.py").write_text('"""Synthetic flow package."""\n', encoding="utf-8")
    (package / "lineup.py").write_text("LOADED = True\n", encoding="utf-8")
    (package / "_helpers.py").write_text("raise AssertionError('private modules are not flows')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module("fake_flows")
    try:
        assert discover(module) == ("lineup",)
        assert "fake_flows.lineup" in sys.modules and "fake_flows._helpers" not in sys.modules
        assert discover(module) == ("lineup",)  # cached imports: nothing registers twice
    finally:
        for name in ("fake_flows", "fake_flows.lineup"):
            sys.modules.pop(name, None)
    assert isinstance(discover(), tuple)  # the real package imports cleanly


# --- write answers and the real transport -----------------------------------------------------------------------------


def test_write_answers_are_classified_from_status_and_espn_codes() -> None:
    locked = WriteResponse.from_text(
        409, '{"messages":["Lineup is locked"],"details":[{"type":"TRAN_LINEUP_LOCKED","message":"Lineup is locked"}]}'
    )
    assert locked.outcome is WriteOutcome.REJECTED and locked.error_codes == ("TRAN_LINEUP_LOCKED",)
    assert locked.describe() == "HTTP 409 TRAN_LINEUP_LOCKED: Lineup is locked" and not rejection_allows_ui(locked)
    assert rejection_allows_ui(WriteResponse(400, {"messages": ["Bad request"]}))
    assert not rejection_allows_ui(WriteResponse(401, None, "AUTH_MISSING_CREDENTIALS"))
    assert WriteResponse(502, None, "<html>bad gateway</html>").outcome is WriteOutcome.UNKNOWN
    assert not rejection_allows_ui(WriteResponse(502))  # an unknown is never followed
    accepted = WriteResponse.from_text(200, '{"id":"abc-123","status":"EXECUTED"}')
    assert accepted.outcome is WriteOutcome.ACCEPTED and transaction_id_from(accepted) == "abc-123"
    assert transaction_id_from(WriteResponse(200, {"transaction": {"transactionId": 42}})) == "42"
    assert WriteResponse.from_text(200, "not json").body is None


class StubRequests:
    """The slice of Playwright's ``APIRequestContext`` that ``PlaywrightTransport`` calls."""

    def __init__(self, outcome: Any) -> None:
        self.outcome = outcome
        self.calls: list[dict[str, Any]] = []

    def post(self, url: str, **options: Any) -> Any:
        self.calls.append({"url": url, **options})
        if isinstance(self.outcome, BaseException):
            raise self.outcome
        status, text = self.outcome
        return types.SimpleNamespace(status=status, text=lambda: text, dispose=lambda: None)


def test_the_browser_session_transport_sends_once_with_a_hard_timeout() -> None:
    request = WriteRequest(url=TRANSACTIONS, body={"isLeagueManager": False, "teamId": TEAM}, headers={"X-Test": "1"})
    stub = StubRequests((409, '{"details":[{"type":"TRAN_LINEUP_LOCKED"}]}'))
    response = PlaywrightTransport(cast(APIRequestContext, stub)).send(request, timeout_s=7.5)

    assert response.status == 409 and response.error_codes == ("TRAN_LINEUP_LOCKED",)
    (call,) = stub.calls
    assert call["url"] == TRANSACTIONS and json.loads(call["data"]) == request.body
    assert (call["timeout"], call["max_retries"], call["max_redirects"], call["fail_on_status_code"]) == (
        7500.0,
        0,
        0,
        False,
    )
    assert call["headers"]["Content-Type"] == "application/json" and call["headers"]["X-Test"] == "1"

    with pytest.raises(WriteTimeoutError, match="no answer within 7.5 s"):
        PlaywrightTransport(cast(APIRequestContext, StubRequests(PlaywrightTimeoutError("Timeout")))).send(
            request, timeout_s=7.5
        )
    with pytest.raises(WriteUncertainError, match="may have reached ESPN"):
        PlaywrightTransport(cast(APIRequestContext, StubRequests(PlaywrightError("socket hang up")))).send(
            request, timeout_s=7.5
        )


def test_the_live_runtime_refuses_without_a_browser_profile(h: Harness) -> None:
    with pytest.raises(NotLoggedInError, match="run `fm login`"), open_live_runtime(h.league, dry_run=True):
        pass  # never reached: no profile, so no browser starts


# --- the live runtime: a dry run's backstop in the browser ------------------------------------------------------------


class RecordingContext:
    """The slice of a Playwright ``BrowserContext`` that ``open_live_runtime`` uses: routes and the request client."""

    def __init__(self) -> None:
        self.routes: list[tuple[str, Callable[..., object]]] = []
        self.request = StubRequests((200, '{"id": "live-transaction-1"}'))

    def route(self, url: str, handler: Callable[..., object]) -> None:
        self.routes.append((url, handler))


class RecordingRoute:
    """The slice of a Playwright ``Route`` the dry-run handler uses; records whether it aborted or continued."""

    def __init__(self, method: str, url: str, *, readable: bool = True) -> None:
        self.method, self.url, self.readable = method, url, readable
        self.outcome: tuple[str, str | None] | None = None

    @property
    def request(self) -> types.SimpleNamespace:
        if not self.readable:
            raise RuntimeError("the request's frame is gone")
        return types.SimpleNamespace(method=self.method, url=self.url)

    def abort(self, error_code: str | None = None) -> None:
        self.outcome = ("abort", error_code)

    def continue_(self) -> None:
        self.outcome = ("continue", None)


def live_browser(monkeypatch: pytest.MonkeyPatch, *, expires_at: datetime | None = None) -> RecordingContext:
    """Stand in for what ``open_live_runtime`` opens (the profile, the browser, its cookie jar), not for the runtime
    itself; returns the browser context it gets."""
    context = RecordingContext()

    @contextmanager
    def open_browser(options: Any) -> Iterator[types.SimpleNamespace]:
        yield types.SimpleNamespace(context=context)

    session = EspnSession(espn_s2="fake-s2", swid=FAKE_SWID, expires_at=expires_at)
    monkeypatch.setattr(executor_runtime, "profile_exists", lambda _profile: True)
    monkeypatch.setattr(executor_runtime, "open_browser", open_browser)
    monkeypatch.setattr(executor_runtime, "harvest_session", lambda _context: session)
    return context


LEAGUE_PATH = "apis/v3/games/ffl/seasons/2026/segments/0/leagues/1234567"
DRY_RUN_REQUESTS = [
    ("POST", TRANSACTIONS, True),
    ("GET", "https://lm-api-writes.fantasy.espn.com/apis/v3/games/ffl/seasons/2026", True),  # the write host: any verb
    ("PUT", "https://lm-api-writes.fantasy.espn.com./apis/v3/x", True),  # trailing dot
    ("PUT", "https://lm-api-communication.fantasy.espn.com/x", True),  # league messages
    ("POST", f"https://fantasy.espn.com/{LEAGUE_PATH}/teams/1/pendingTransactions", True),  # reorder our claims
    ("DELETE", f"https://lm-api-reads.fantasy.espn.com/{LEAGUE_PATH}", True),
    ("post", "https://ESPN.com/x", True),
    ("POST", "https://sw88.espn.com/b/ss/x", True),  # any non-GET to an espn.com host, analytics too
    ("GET", f"https://lm-api-reads.fantasy.espn.com/{LEAGUE_PATH}?view=mRoster", False),
    ("GET", ROSTER_PAGE, False),
    ("POST", "https://registerdisney.go.com/jgc/v8/client/ESPN-ONESITE.WEB-PROD/guest/login", False),  # the sign-in
    ("POST", "https://espn.com.example.net/x", False),  # not an ESPN host
    ("POST", "https://notespn.com/x", False),
]


@pytest.mark.parametrize(("method", "url", "aborted"), DRY_RUN_REQUESTS)
def test_the_dry_run_rule_blocks_every_league_write(method: str, url: str, aborted: bool) -> None:
    assert (dry_run_block_reason(method, url) is not None) is aborted


def test_a_live_dry_run_aborts_writes_in_the_browser_and_refuses_to_send(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = live_browser(monkeypatch)
    with open_live_runtime(h.league, dry_run=True) as runtime:
        assert isinstance(runtime.transport, RefusingTransport) and runtime.member_id == FAKE_SWID
        ((pattern, handler),) = context.routes
        assert pattern == "**/*"  # every request goes through the rule, not only the write host's
        for method, url, aborted in DRY_RUN_REQUESTS:
            route = RecordingRoute(method, url)
            handler(route)
            assert route.outcome == (("abort", "blockedbyclient") if aborted else ("continue", None)), (method, url)
        unreadable = RecordingRoute("GET", ROSTER_PAGE, readable=False)
        handler(unreadable)
        assert unreadable.outcome == ("abort", "blockedbyclient")  # fails closed
        with pytest.raises(WriteRefusedError, match="dry run: nothing is sent"):
            runtime.transport.send(WriteRequest(url=TRANSACTIONS, body={"teamId": TEAM}), timeout_s=1.0)
    assert context.request.calls == []


def test_a_live_run_installs_no_route_and_writes_through_the_browser_session(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = live_browser(monkeypatch)
    with open_live_runtime(h.league, dry_run=False) as runtime:
        assert isinstance(runtime.transport, PlaywrightTransport)
        response = runtime.transport.send(WriteRequest(url=TRANSACTIONS, body={"teamId": TEAM}), timeout_s=1.0)
    assert context.routes == []
    assert response.status == 200 and [call["url"] for call in context.request.calls] == [TRANSACTIONS]


def test_the_live_runtime_refuses_an_expired_session_before_routing_anything(
    h: Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = live_browser(monkeypatch, expires_at=datetime.now(UTC) - timedelta(days=1))
    with pytest.raises(AuthError, match="expired .* run `fm login`"), open_live_runtime(h.league, dry_run=True):
        pass
    assert context.routes == [] and context.request.calls == []


# --- interruptions: a run that something escapes still ends -----------------------------------------------------------


def test_an_interrupt_after_the_write_left_still_ends_the_run(h: Harness) -> None:
    h.transport.replies.append(Reply(error=KeyboardInterrupt(), applies=True))  # the write lands, then Ctrl+C
    proposal = h.propose(dedupe_key="hubbard-allgeier")
    with pytest.raises(KeyboardInterrupt):
        h.run(proposal)

    row = get_proposal(h.store, proposal.row_id)
    assert row.status == "failed" and row.token_consumed_at == NOW  # not left executing for good
    (attempt,) = h.store.executions.for_proposal(proposal.row_id)
    assert attempt.status == "unknown" and attempt.finished_at == NOW
    assert attempt.error == (
        "interrupted after the write started (KeyboardInterrupt); check the league before acting again"
    )
    assert h.slot(HUBBARD) == BENCH  # it did land, which is why the attempt is unknown and not failed
    with pytest.raises(LifecycleError, match="cannot execute proposal #1: it is failed"):
        h.run(proposal)
    assert len(h.transport.sent) == 1
    # Left executing, the open proposal would have been handed back for its dedupe key forever.
    again = h.propose(dedupe_key="hubbard-allgeier", approved=False)
    assert again.row_id != proposal.row_id and again.status == "proposed"


def test_an_interrupt_before_anything_was_sent_fails_the_attempt(h: Harness) -> None:
    registry = FlowRegistry()
    registry.register(InterruptedFlow())
    proposal = h.propose()
    with pytest.raises(KeyboardInterrupt):
        h.run(proposal, registry=registry)

    (attempt,) = h.store.executions.for_proposal(proposal.row_id)
    assert attempt.status == "failed" and attempt.error == "interrupted before anything was written (KeyboardInterrupt)"
    assert get_proposal(h.store, proposal.row_id).status == "failed"
    assert h.transport.sent == [] and h.slot(HUBBARD) == FLEX


def test_an_exit_during_the_re_read_leaves_the_write_unknown(h: Harness) -> None:
    registry = FlowRegistry()
    registry.register(ExitingFlow())
    proposal = h.propose()
    with pytest.raises(SystemExit):
        h.run(proposal, registry=registry)

    (attempt,) = h.store.executions.for_proposal(proposal.row_id)
    assert attempt.status == "unknown" and attempt.response is not None and attempt.response["status"] == 200
    assert attempt.error is not None and attempt.error.startswith("interrupted after the write started (SystemExit: 3)")
    assert get_proposal(h.store, proposal.row_id).status == "failed" and len(h.transport.sent) == 1


def test_an_interrupted_dry_run_closes_its_attempt_and_leaves_the_proposal_be(h: Harness) -> None:
    registry = FlowRegistry()
    registry.register(InterruptedFlow())
    proposal = h.propose()
    with pytest.raises(KeyboardInterrupt):
        h.run(proposal, None, dry_run=True, registry=registry)

    (attempt,) = h.store.executions.for_proposal(proposal.row_id)
    assert attempt.status == "dry_run" and attempt.error == "interrupted (KeyboardInterrupt); nothing was sent"
    assert get_proposal(h.store, proposal.row_id) == proposal  # still approved, its token unspent
    assert h.run(proposal).ok


def test_reconcile_ends_the_runs_a_killed_process_left_executing(h: Harness) -> None:
    stuck, bare, live = (h.propose() for _ in range(3))
    for proposal, spent in ((stuck, NOW), (bare, NOW), (live, NOW + timedelta(minutes=45))):
        assert proposal.execution_token is not None
        begin_execution(h.store, proposal.row_id, proposal.execution_token, now=spent)
    # stuck died mid-attempt; bare between spending its token and its first attempt; live is running elsewhere now.
    h.store.executions.insert(ExecutionRow(proposal_id=stuck.row_id, mode="api", status="running", started_at=NOW))
    later = NOW + timedelta(hours=1)

    failed = reconcile_executions(h.store, now=later)
    assert [(row.row_id, row.status) for row in failed] == [(stuck.row_id, "failed"), (bare.row_id, "failed")]
    (attempt,) = h.store.executions.for_proposal(stuck.row_id)
    assert attempt.status == "unknown" and attempt.finished_at == later
    assert attempt.error is not None and "check the league before acting again" in attempt.error
    assert get_proposal(h.store, live.row_id).status == "executing"  # 15 minutes in: it may still be going
    assert reconcile_executions(h.store, now=later) == []
    (stale,) = reconcile_executions(h.store, now=NOW + timedelta(minutes=45) + STALE_EXECUTION)
    assert stale.row_id == live.row_id and stale.status == "failed"


# --- the fake page behaves like Playwright where flows depend on it ---------------------------------------------------


def test_the_fake_page_follows_playwright_locator_rules() -> None:
    first = FakeElement(role="button", name="Add")
    second = FakeElement(role="button", name="Add player")
    hidden = FakeElement(role="button", name="Drop", visible=False)
    page = FakePage(FakeElement(role="row", text="Josh Allen").add(first), second, hidden)

    with pytest.raises(PlaywrightError, match="strict mode violation"):
        page.get_by_role("button", name="Add").click()  # substring match finds both
    page.get_by_role("button", name="Add", exact=True).click()
    page.get_by_role("row").filter(has_text="josh allen").get_by_role("button").click()
    assert first.clicks == 2 and second.clicks == 0
    with pytest.raises(PlaywrightTimeoutError):
        page.get_by_role("button", name="Drop").click()  # hidden elements are not in the accessibility tree
    with pytest.raises(PlaywrightTimeoutError):
        page.get_by_text("nowhere").wait_for()
    assert page.get_by_role("button").count() == 2 and page.get_by_role("button").last.inner_text() == "Add player"
    page.close()
    with pytest.raises(PlaywrightError, match="closed"):
        page.get_by_role("button", name="Add player").click()


# --- the command ------------------------------------------------------------------------------------------------------


def cli() -> typer.Typer:
    root = typer.Typer()
    root.callback()(lambda: None)  # keep it a group, as fm.cli does
    execute_cmd.register(root)
    return root


@pytest.fixture
def wired(h: Harness, monkeypatch: pytest.MonkeyPatch) -> Harness:
    """``fm execute`` running against the harness instead of the real browser profile and flows."""
    monkeypatch.setattr(execute_cmd, "live_opener", lambda launch: h.opener)
    monkeypatch.setattr(
        execute_cmd, "execute", functools.partial(execute, registry=h.registry, options=FAST, clock=lambda: NOW)
    )
    return h


def test_execute_help_exits_zero() -> None:
    result = runner.invoke(cli(), ["execute", "--help"])
    assert result.exit_code == 0, result.output
    assert "--dry-run" in result.output and "--mode" in result.output


def test_the_command_dry_run_prints_the_request_and_sends_nothing(wired: Harness) -> None:
    proposal = wired.propose(approved=False)
    result = runner.invoke(cli(), ["execute", str(proposal.row_id), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert f"#1 lineup in nfl via test_lineup: {HUBBARD}: slot 23 -> 20; {ALLGEIER}: slot 20 -> 23" in result.output
    assert "preconditions: ok" in result.output and "api: dry run" in result.output
    assert f"request: POST {TRANSACTIONS}" in result.output and '"executionType":"EXECUTE"' in result.output
    assert "dry run: nothing was sent; it would have been sent" in result.output
    assert wired.transport.sent == []


def test_the_command_executes_once_and_reports_refusals_and_failures(wired: Harness) -> None:
    proposal = wired.propose()
    first = runner.invoke(cli(), ["execute", str(proposal.row_id)])
    assert first.exit_code == 0, first.output
    assert "api: verified" in first.output and "proposal #1: verified" in first.output
    assert "ESPN transaction: fake-transaction-1" in first.output and "re-read (1x): shows the change" in first.output

    again = runner.invoke(cli(), ["execute", str(proposal.row_id)])
    assert again.exit_code == 1 and "error: cannot execute proposal #1: it is verified" in again.output
    missing = runner.invoke(cli(), ["execute", "99"])
    assert missing.exit_code == 1 and "error: no proposal #99" in missing.output

    locked = wired.propose(LOCKED)
    failed = runner.invoke(cli(), ["execute", str(locked.row_id)])
    assert failed.exit_code == 1
    assert "preconditions: FAILED: player 4429795 is locked" in failed.output and "proposal #2: failed" in failed.output
    assert len(wired.transport.sent) == 1
