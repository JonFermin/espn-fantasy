"""``propose_trade``, ``respond_trade`` and ``cancel_trade`` (ROADMAP #44): trade offers, our answers to incoming ones
and the withdrawal of our own, approval-only, in API mode with the builder walk as the UI mode.

Everything runs offline over the real-league fixtures (``tests/fixtures/espn/real``). The recorded rosters hold only our
team and a stub of one other, so :class:`League` adds the team each capture traded with and its players. The proposal
body is held to the capture body for body: ``ffl/write_TRADE_PROPOSAL_1.json`` (observed under the write guard, aborted
at Send) and ``fba/write_TRADE_PROPOSAL_derived.json`` (derived from the saved web-client code), with the clock pinned
so ``expirationDate`` is the captured one (send time plus 48 hours). Responses and cancels have no capture; their
envelopes are held to the documented shapes. The fake write transport records what lands: an offer joins
``mTransactions2``, an accept moves the players and marks the offer executed, a decline closes it, a cancel adds its
``CANCEL`` record (the offer stays listed ``PENDING``, as an expired one does). :class:`Site` is the trade builder
with the names the trade-review capture saw.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from fm.browser import selectors
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
from fm.browser.flows import FlowRegistry, Mode, WriteRequest, flow_for
from fm.browser.flows.trade import (
    CANCEL_TRADE,
    EXPIRY_DAYS,
    PROPOSE_TRADE,
    RESPOND_TRADE,
    CancelTrade,
    ProposeTrade,
    RespondTrade,
)
from fm.browser.transactions import WEB_CLIENT_HEADERS
from fm.config import Config, Policy, Sport
from fm.espn.ids import Game, ids_for
from fm.executor import ExecutionResult, ExecutorOptions, NoFlowError, execute
from fm.proposals import (
    KINDS,
    TRADE_KINDS,
    Payload,
    PolicyError,
    ProposalKind,
    TradePayload,
    TradeResponsePayload,
    TransactionCancelPayload,
    approve,
    auto_approve_due,
    effective_setting,
    get_proposal,
    propose,
    validate_setting,
)
from fm.store import LeagueRow, ProposalRow, Store

REAL = Path(__file__).resolve().parents[1] / "fixtures" / "espn" / "real"
TEAM = 1
LEAGUE_IDS = {"ffl": 1010101, "fba": 2020202}
SEASONS = {"ffl": 2026, "fba": 2027}
SPORTS: dict[str, Sport] = {"ffl": "nfl", "fba": "nba"}
FAST = ExecutorOptions(
    write_timeout_s=5.0, ui_timeout_s=5.0, verify_attempts=2, verify_interval_s=0.0, sleep=lambda _s: None
)
CONFIG = Config.model_validate(
    {
        "league": [
            {
                "key": SPORTS[game],
                "sport": SPORTS[game],
                "espn_league_id": LEAGUE_IDS[game],
                "season": SEASONS[game],
                "team_id": TEAM,
            }
            for game in ("ffl", "fba")
        ]
    }
)

# What each capture traded: the scrubbed ids of ffl/write_TRADE_PROPOSAL_1.json and
# fba/write_TRADE_PROPOSAL_derived.json.
DEALS: dict[str, dict[str, Any]] = {
    "ffl": {
        "other": 5,
        "period": 5,
        "mine": (15847, 3040151),  # Travis Kelce and George Kittle, on our recorded roster
        "theirs": {3126486: "Rival Receiver", 4432665: "Rival Back"},
        "sent_at": datetime(2026, 10, 6, 18, 10, 21, 78000, tzinfo=UTC),  # the capture's own send time
        "capture": "write_TRADE_PROPOSAL_1.json",
    },
    "fba": {
        "other": 3,
        "period": 1,
        "mine": (3059319,),  # Andrew Wiggins
        "theirs": {2595516: "Norman Powell"},
        "sent_at": datetime(2026, 10, 6, 18, 30, 7, 428000, tzinfo=UTC),
        "capture": "write_TRADE_PROPOSAL_derived.json",
    },
}


def load(game: str, name: str) -> Any:
    return json.loads((REAL / game / name).read_text(encoding="utf-8"))


def transactions_url(game: str) -> str:
    return (
        f"https://lm-api-writes.fantasy.espn.com/apis/v3/games/{game}/seasons/{SEASONS[game]}/segments/0/leagues/"
        f"{LEAGUE_IDS[game]}/transactions/"
    )


def ms(at: datetime) -> int:
    return int(at.timestamp() * 1000)


class ProposeTradeWithUi(ProposeTrade):
    """The proposal flow as it will be once the UI drill covers it: both modes. The shipped flow lists API only."""

    modes = (Mode.API, Mode.UI)


# --- the fake league and the builder page -----------------------------------------------------------------------------


class League:
    """One real-fixture league plus the team it traded with: the fake read API, a write transport that records what
    lands, the trade builder, and a private flow registry with the three trade flows."""

    def __init__(self, store: Store, game: str, *, with_ui: bool = False) -> None:
        deal = DEALS[game]
        self.store = store
        self.game = Game(game)
        self.ids = ids_for(self.game)
        self.other: int = deal["other"]
        self.period: int = deal["period"]
        self.mine: tuple[int, ...] = deal["mine"]
        self.theirs: dict[int, str] = deal["theirs"]
        self.clock: datetime = deal["sent_at"]
        self.hold_accepts = False
        self.row = store.leagues.upsert(
            LeagueRow(
                key=SPORTS[game],
                sport=SPORTS[game],
                espn_league_id=LEAGUE_IDS[game],
                season=SEASONS[game],
                team_id=TEAM,
                as_of=self.clock,
            )
        )
        self.rosters: dict[str, Any] = load(game, "mRoster.json")
        self.settings: dict[str, Any] = load(game, "mSettings.json")
        self.schedule: dict[str, Any] = load(game, "proTeamSchedules_wl.json")
        self.pending: dict[str, Any] = load(game, "mPendingTransactions.json")
        self.pending.setdefault("pendingTransactions", [])
        self.transactions: dict[str, Any] = load(game, "mTransactions2_waiver_trade.json")
        self.transactions["transactions"] = [
            record for record in self.transactions["transactions"] if not record["type"].startswith("TRADE")
        ]
        self.rosters["status"]["latestScoringPeriod"] = self.period
        self.rosters["scoringPeriodId"] = self.period
        for team in self.rosters["teams"]:
            for entry in team["roster"]["entries"]:
                pool = entry["playerPoolEntry"]
                pool["rosterLocked"] = pool["tradeLocked"] = pool["lineupLocked"] = False
        self.add_team(self.other, self.theirs)
        self.api = FakeEspnApi()
        self.api.serve("mRoster", lambda _request: self.rosters)
        self.api.serve("mSettings", self.settings)
        self.api.serve("proTeamSchedules_wl", self.schedule)
        self.api.serve("mPendingTransactions", lambda _request: self.pending)
        self.api.serve("mTransactions2", lambda _request: self.transactions)
        self.transport = FakeTransport(on_send=self.apply)
        self.site = Site(self)
        self.browser = FakeBrowser(self.site.page)
        self.registry = FlowRegistry()
        self.registry.register(ProposeTradeWithUi() if with_ui else ProposeTrade())
        self.registry.register(RespondTrade())
        self.registry.register(CancelTrade())
        self.opener = FakeOpener(fake_runtime(self.api, self.row, transport=self.transport, browser=self.browser))
        self.counter = 0

    # --- proposals and runs ---

    def propose(
        self, payload: Payload, kind: ProposalKind, *, approved: bool = True, deadline: datetime | None = None
    ) -> ProposalRow:
        row = propose(
            self.store,
            CONFIG,
            self.row,
            kind,
            payload,
            created_by="test",
            scoring_period_id=self.period,
            deadline=deadline,
            now=self.clock,
        )
        return approve(self.store, row.row_id, decided_by="test", now=self.clock) if approved else row

    def offer(self, *, give: tuple[int, ...] | None = None, get: tuple[int, ...] | None = None) -> ProposalRow:
        """A ``trade_propose`` of the capture's deal (or a variation), approved."""
        payload = TradePayload(
            other_team_id=self.other,
            give_espn_ids=self.mine if give is None else give,
            get_espn_ids=tuple(self.theirs) if get is None else get,
        )
        return self.propose(payload, ProposalKind.TRADE_PROPOSE)

    def answer(self, offer_id: str, kind: ProposalKind, *, approved: bool = True) -> ProposalRow:
        payload = TradeResponsePayload(
            other_team_id=self.other,
            give_espn_ids=self.mine,
            get_espn_ids=tuple(self.theirs),
            espn_transaction_id=offer_id,
        )
        return self.propose(payload, kind, approved=approved)

    def cancel(self, offer_id: str, *, approved: bool = True) -> ProposalRow:
        payload = TransactionCancelPayload(espn_transaction_id=offer_id)
        return self.propose(payload, ProposalKind.TRADE_CANCEL, approved=approved)

    def run(self, proposal: ProposalRow, *, dry_run: bool = False, mode: Mode | None = None) -> ExecutionResult:
        return execute(
            self.store,
            proposal.row_id,
            token=None if dry_run else proposal.execution_token,
            dry_run=dry_run,
            mode=mode,
            opener=self.opener,
            registry=self.registry,
            options=FAST,
            clock=lambda: self.clock,
        )

    # --- the league's state ---

    def team(self, team_id: int) -> dict[str, Any]:
        return next(team for team in self.rosters["teams"] if team["id"] == team_id)

    def entries(self, team_id: int) -> list[dict[str, Any]]:
        return self.team(team_id)["roster"]["entries"]

    def entry(self, espn_id: int) -> dict[str, Any]:
        for team in self.rosters["teams"]:
            for entry in team["roster"]["entries"]:
                if entry["playerId"] == espn_id:
                    return entry
        raise KeyError(espn_id)

    def holder(self, espn_id: int) -> int:
        return next(
            team["id"]
            for team in self.rosters["teams"]
            if espn_id in [entry["playerId"] for entry in team["roster"]["entries"]]
        )

    def add_team(self, team_id: int, players: dict[int, str]) -> None:
        """Another team with ``players`` on its bench, cut from one of our recorded entries."""
        template = self.entries(TEAM)[0]
        bench = self.ids.bench_slot
        entries: list[dict[str, Any]] = []
        for espn_id, name in players.items():
            entry = copy.deepcopy(template)
            entry.update(playerId=espn_id, lineupSlotId=bench)
            pool = entry["playerPoolEntry"]
            pool.update(id=espn_id, onTeamId=team_id)
            pool["player"].update(id=espn_id, fullName=name)
            entries.append(entry)
        self.rosters["teams"].append({"id": team_id, "roster": {"entries": entries}})

    def open_offer(
        self,
        offer_id: str,
        *,
        proposer: int,
        give: tuple[int, ...],
        get: tuple[int, ...],
        other: int | None = None,
        expires: datetime | None = None,
    ) -> dict[str, Any]:
        """A pending offer in ``mTransactions2``: ``give`` are our players it takes, ``get`` the ones it brings."""
        partner = self.other if other is None else other
        items = [
            {"playerId": espn_id, "type": "TRADE", "fromTeamId": TEAM, "toTeamId": partner, "toLineupSlotId": -1}
            for espn_id in give
        ] + [
            {"playerId": espn_id, "type": "TRADE", "fromTeamId": partner, "toTeamId": TEAM, "toLineupSlotId": -1}
            for espn_id in get
        ]
        record = {
            "id": offer_id,
            "type": "TRADE_PROPOSAL",
            "status": "PENDING",
            "executionType": "EXECUTE",
            "teamId": proposer,
            "scoringPeriodId": self.period,
            "proposedDate": ms(self.clock - timedelta(hours=1)),
            "expirationDate": ms(expires if expires is not None else self.clock + timedelta(hours=47)),
            "isPending": True,
            "items": items,
        }
        self.transactions["transactions"].append(record)
        return record

    def incoming(self, offer_id: str = "offer-in") -> dict[str, Any]:
        """The capture's deal as an offer from the other team to us."""
        return self.open_offer(offer_id, proposer=self.other, give=self.mine, get=tuple(self.theirs))

    def record(self, offer_id: str) -> dict[str, Any]:
        return next(record for record in self.transactions["transactions"] if record["id"] == offer_id)

    def land(self, body: dict[str, Any]) -> None:
        """ESPN recording a trade transaction that lands."""
        kind, execution = body["type"], body["executionType"]
        self.counter += 1
        if kind == "TRADE_PROPOSAL" and execution == "EXECUTE":
            expires = datetime.fromisoformat(body["expirationDate"].replace("Z", "+00:00"))
            self.transactions["transactions"].append(
                {
                    "id": f"offer-{self.counter}",
                    "type": "TRADE_PROPOSAL",
                    "status": "PENDING",
                    "executionType": "EXECUTE",
                    "teamId": body["teamId"],
                    "scoringPeriodId": body.get("scoringPeriodId"),
                    "proposedDate": ms(self.clock),
                    "expirationDate": ms(expires),
                    "isPending": True,
                    "items": body["items"],
                }
            )
        elif kind == "TRADE_PROPOSAL":
            self.transactions["transactions"].append(
                {
                    "id": f"cancel-{self.counter}",
                    "type": "TRADE_PROPOSAL",
                    "status": "CANCELED",
                    "executionType": "CANCEL",
                    "teamId": body["teamId"],
                    "relatedTransactionId": body["relatedTransactionId"],
                    "isPending": True,
                }
            )
        elif kind == "TRADE_ACCEPT" and not self.hold_accepts:
            offer = self.record(body["relatedTransactionId"])
            for item in offer["items"]:
                entry = self.entry(item["playerId"])
                self.entries(item["fromTeamId"]).remove(entry)
                self.entries(item["toTeamId"]).append(entry)
                entry["playerPoolEntry"]["onTeamId"] = item["toTeamId"]
            offer["status"] = "EXECUTED"
        elif kind == "TRADE_DECLINE":
            self.record(body["relatedTransactionId"])["status"] = "DECLINED"

    def apply(self, request: WriteRequest) -> None:
        self.land(dict(request.body))

    def builder_url(self) -> str:
        return selectors.trade_builder_url(
            self.game, LEAGUE_IDS[self.game.value], SEASONS[self.game.value], self.other, TEAM
        )


class Site:
    """The trade builder over :class:`League` with the names the trade-review capture saw: a ``Trade <Player>``
    checkbox per rostered player, ``Continue``, then the review's expiry select, message box and ``Send Trade
    Proposal``, which records the offer. ``signed_in=False`` shows Log in Required instead."""

    def __init__(self, league: League) -> None:
        self.league = league
        self.signed_in = True
        self.boxes: dict[int, FakeElement] = {}
        self.expiry: FakeElement | None = None
        self.saves: list[tuple[tuple[int, ...], str]] = []
        self.page = FakePage(screens={league.builder_url(): self.render})

    def render(self) -> list[FakeElement]:
        if not self.signed_in:
            return [FakeElement(role="heading", name="Log in Required")]
        self.boxes = {}
        self.expiry = None
        elements = [FakeElement(role="heading", name="Propose Trade Rival Team")]
        for team_id in (TEAM, self.league.other):
            for entry in self.league.entries(team_id):
                name = entry["playerPoolEntry"]["player"]["fullName"]
                box = FakeElement(role="checkbox", name=f"Trade {name}")
                self.boxes[entry["playerId"]] = box
                elements.append(box)
        elements += [
            FakeElement(role="button", name="Continue", on_click=self._review),
            FakeElement(role="button", name="Cancel Trade"),
        ]
        return elements

    def _review(self, page: FakePage) -> None:
        self.expiry = FakeElement(role="combobox", options=tuple(str(day) for day in range(1, 8)), value="2")
        page.add(
            self.expiry,
            FakeElement(role="textbox"),
            FakeElement(role="button", name="Send Trade Proposal", on_click=self._send),
        )

    def _send(self, page: FakePage) -> None:
        assert self.expiry is not None
        picked = tuple(player for player, box in self.boxes.items() if box.checked)
        self.saves.append((picked, self.expiry.value))
        items = [
            {"playerId": player, "type": "TRADE", "fromTeamId": self.league.holder(player), "toTeamId": other}
            for player in picked
            for other in [self.league.other if self.league.holder(player) == TEAM else TEAM]
        ]
        expires = self.league.clock + timedelta(days=int(self.expiry.value))
        self.league.land(
            {
                "type": "TRADE_PROPOSAL",
                "executionType": "EXECUTE",
                "teamId": TEAM,
                "items": items,
                "expirationDate": expires.isoformat().replace("+00:00", "Z"),
            }
        )
        page.show(FakeElement(role="heading", name="Trade proposal sent"))


@pytest.fixture
def nfl() -> Iterator[League]:
    with Store.open() as store:  # paths.state_db() in the per-test config dir
        yield League(store, "ffl")


@pytest.fixture
def nba() -> Iterator[League]:
    with Store.open() as store:
        yield League(store, "fba")


@pytest.fixture
def nfl_ui() -> Iterator[League]:
    with Store.open() as store:
        yield League(store, "ffl", with_ui=True)


def captured_body(game: str) -> dict[str, Any]:
    """The capture's body as our run sends it: the fakes' SWID in place of the scrubbed one."""
    return {**load(game, DEALS[game]["capture"])["body"], "memberId": FAKE_SWID}


def artifact_names(result: ExecutionResult) -> set[str]:
    return {Path(relative).name for attempt in result.attempts for relative in attempt.artifacts}


# --- registration and policy ------------------------------------------------------------------------------------------


def test_the_flows_serve_every_trade_kind_in_both_sports() -> None:
    for sport in ("nfl", "nba"):
        assert flow_for(ProposalKind.TRADE_PROPOSE, sport) is PROPOSE_TRADE
        assert flow_for(ProposalKind.TRADE_ACCEPT, sport) is RESPOND_TRADE
        assert flow_for(ProposalKind.TRADE_DECLINE, sport) is RESPOND_TRADE
        assert flow_for(ProposalKind.TRADE_CANCEL, sport) is CANCEL_TRADE
    assert (PROPOSE_TRADE.name, RESPOND_TRADE.name, CANCEL_TRADE.name) == (
        "propose_trade",
        "respond_trade",
        "cancel_trade",
    )
    # The builder walk is not a registered UI mode until the UI drill covers it (module docs); every other trade
    # flow has no observed page at all.
    assert (
        PROPOSE_TRADE.modes == (Mode.API,) and RESPOND_TRADE.modes == (Mode.API,) and CANCEL_TRADE.modes == (Mode.API,)
    )


def test_auto_can_never_apply_to_a_trade_kind(nfl: League) -> None:
    assert set(TRADE_KINDS) == {kind for kind in ProposalKind if kind.value.startswith("trade")}
    for kind in TRADE_KINDS:
        spec = KINDS[kind]
        assert spec.policy_field is None and spec.allowed == ("approve",), kind  # no field to set, nothing but approve
        assert effective_setting(kind, Policy()) == "approve"
        for setting in ("auto", "off"):
            with pytest.raises(PolicyError, match="approval-only and not configurable"):
                validate_setting(kind, setting)
        with pytest.raises(ValueError, match="approval-only and not configurable"):
            Policy.model_validate({kind.value: "auto"})  # config.toml cannot even name one

    row = nfl.offer()
    assert row.policy == "approve"
    # a producer's ceiling can lower a setting, never raise one: asking for auto through max_setting changes nothing
    capped = propose(
        nfl.store,
        CONFIG,
        nfl.row,
        ProposalKind.TRADE_PROPOSE,
        TradePayload(other_team_id=nfl.other, give_espn_ids=(15847,), get_espn_ids=(3126486,)),
        created_by="test",
        deadline=nfl.clock + timedelta(minutes=10),
        max_setting="auto",
        now=nfl.clock,
    )
    assert capped.policy == "approve"
    # and the T-15 sweep, the only way auto ever acts, leaves a trade unanswered however close its deadline is
    assert auto_approve_due(nfl.store, now=nfl.clock + timedelta(minutes=5)) == []
    assert get_proposal(nfl.store, capped.row_id).status == "proposed"


def test_a_trade_proposal_that_ran_under_auto_or_was_approved_by_auto_is_refused_by_every_flow(nfl: League) -> None:
    """Defence in depth: even a row that somehow carries ``auto`` (a hand-edited database) sends nothing."""
    nfl.incoming()
    nfl.open_offer("mine", proposer=TEAM, give=(15847,), get=(), other=7)
    cases = (
        (nfl.offer(), "propose_trade"),
        (nfl.answer("offer-in", ProposalKind.TRADE_ACCEPT), "respond_trade"),
        (nfl.answer("offer-in", ProposalKind.TRADE_DECLINE), "respond_trade"),
        (nfl.cancel("mine"), "cancel_trade"),
    )
    for row, flow in cases:
        nfl.store.proposals.update(row.model_copy(update={"policy": "auto", "decided_by": "auto"}))
        result = nfl.run(get_proposal(nfl.store, row.row_id))
        assert result.flow == flow and result.proposal.status == "failed"
        assert result.preconditions.failures[:2] == (
            "trades are approval-only, but this proposal ran under policy 'auto'; nothing is sent without a person's "
            "approval",
            "trades are approval-only, but this proposal was approved by auto, not by a person",
        )
    assert nfl.transport.sent == []


# --- the proposal, API mode -------------------------------------------------------------------------------------------


@pytest.mark.parametrize("game", ["ffl", "fba"])
def test_a_proposal_sends_the_captured_body_and_is_verified_open(game: str) -> None:
    with Store.open() as store:
        league = League(store, game)
        result = league.run(league.offer())

        assert result.ok and result.status == "verified" and result.flow == "propose_trade"
        (request,) = league.transport.sent
        captured = load(game, DEALS[game]["capture"])
        assert request.url == transactions_url(game) == captured["url"].split("?")[0]
        assert (
            {key.lower(): value for key, value in request.headers.items()}
            == {key: value for key, value in captured["headers"].items() if key.startswith("x-fantasy")}
            == {key.lower(): value for key, value in WEB_CLIENT_HEADERS.items()}
        )
        assert request.body == captured_body(game)
        assert list(request.body) == list(captured["body"])  # key for key, in the serializer's order
        assert request.body["expirationDate"] == (league.clock + timedelta(days=EXPIRY_DAYS)).isoformat(
            timespec="milliseconds"
        ).replace("+00:00", "Z")
        assert request.body["comment"] == "" and isinstance(request.body["expirationDate"], str)
        (attempt,) = result.attempts
        assert attempt.mode == "api" and attempt.verification is not None
        assert attempt.verification["detail"].endswith("is open")
        assert league.browser.opened == []
        observed = result.preconditions.observed
        assert observed["scoring_period_id"] == DEALS[game]["period"]
        assert [piece["espn_id"] for piece in observed["give"]] == list(league.mine)


def test_the_expiry_is_a_string_even_though_espns_records_carry_milliseconds(nfl: League) -> None:
    nfl.run(nfl.offer())
    (request,) = nfl.transport.sent
    assert request.body["expirationDate"] == "2026-10-08T18:10:21.078Z"
    recorded = nfl.transactions["transactions"][-1]
    assert recorded["expirationDate"] == ms(datetime(2026, 10, 8, 18, 10, 21, 78000, tzinfo=UTC))


def test_a_dry_run_builds_the_proposal_and_sends_nothing(nfl: League) -> None:
    proposal = nfl.offer()
    row = nfl.store.proposals.get(proposal.row_id)
    assert row is not None
    result = nfl.run(
        nfl.propose(TradePayload.model_validate(row.payload), ProposalKind.TRADE_PROPOSE, approved=False), dry_run=True
    )

    assert result.ok and result.status == "dry_run" and nfl.transport.sent == []
    (attempt,) = result.attempts
    assert attempt.request is not None and attempt.request["body"] == captured_body("ffl")
    assert {"preconditions.json", "api-request.json"} <= artifact_names(result)
    assert not [r for r in nfl.transactions["transactions"] if r["type"] == "TRADE_PROPOSAL"]


# --- the proposal, preconditions --------------------------------------------------------------------------------------


def test_a_second_offer_to_the_same_team_or_with_a_player_already_offered_is_blocked(nfl: League) -> None:
    nfl.open_offer("ours", proposer=TEAM, give=(3040151,), get=(4432665,))  # one open offer of ours to team 5
    same_team = nfl.run(nfl.offer(give=(15847,), get=(3126486,)))
    assert same_team.preconditions.failures == ("an offer of ours to team 5 is already open (ours)",)

    nfl2_player = nfl.run(nfl.offer())  # the capture's deal reuses two of the offered players
    assert nfl2_player.preconditions.failures == (
        "an offer of ours to team 5 is already open (ours)",
        "players [3040151, 4432665] are already in the open offer ours",
    )
    assert nfl.transport.sent == [] and nfl2_player.proposal.status == "failed"


def test_an_incoming_offer_does_not_block_ours_but_a_shared_player_does(nfl: League) -> None:
    nfl.open_offer("theirs", proposer=7, give=(), get=(), other=7)  # an offer that does not involve us at all
    nfl.open_offer("incoming", proposer=5, give=(15847,), get=(3126486,))
    blocked = nfl.run(nfl.offer())
    assert blocked.preconditions.failures == ("players [15847, 3126486] are already in the open offer incoming",)

    nfl.transactions["transactions"] = [r for r in nfl.transactions["transactions"] if r["id"] != "incoming"]
    nfl.open_offer("incoming", proposer=5, give=(), get=(), expires=nfl.clock - timedelta(minutes=1))  # expired
    assert nfl.run(nfl.offer()).ok


def test_an_expired_offer_and_a_cancelled_one_are_not_open_so_they_do_not_block(nfl: League) -> None:
    nfl.open_offer("expired", proposer=TEAM, give=(15847,), get=(3126486,), expires=nfl.clock - timedelta(seconds=1))
    nfl.open_offer("cancelled", proposer=TEAM, give=(3040151,), get=(4432665,))
    nfl.land({"type": "TRADE_PROPOSAL", "executionType": "CANCEL", "teamId": TEAM, "relatedTransactionId": "cancelled"})
    result = nfl.run(nfl.offer())

    assert result.ok and result.preconditions.observed["open_offers"] == []


def test_a_locked_player_on_either_side_blocks_the_offer(nfl: League) -> None:
    nfl.entry(15847)["playerPoolEntry"]["tradeLocked"] = True  # ours, ESPN's own flag
    nfl.entry(3126486)["playerPoolEntry"]["rosterLocked"] = True  # theirs, his game has started
    result = nfl.run(nfl.offer())

    assert result.preconditions.failures == (
        "ESPN marks Travis Kelce (15847) trade-locked",
        "ESPN marks Rival Receiver (3126486) roster-locked: his game has started",
    )
    assert result.proposal.status == "failed" and nfl.transport.sent == []


def test_a_closed_roster_lock_blocks_every_player_in_the_league_s_own_terms(nba: League) -> None:
    nba.clock = datetime(2026, 10, 20, 19, 30, tzinfo=UTC)  # day 1's first tip (19:00) has passed
    result = nba.run(nba.offer())

    assert result.preconditions.failures == (
        "trading Andrew Wiggins (3059319) closed at 2026-10-20 19:00 UTC: adds and drops close for everyone at the "
        "period's first game",
        "trading Norman Powell (2595516) closed at 2026-10-20 19:00 UTC: adds and drops close for everyone at the "
        "period's first game",
    )
    assert nba.transport.sent == []


def test_an_unmapped_roster_lock_type_refuses_instead_of_guessing(nba: League) -> None:
    nba.settings["settings"]["rosterSettings"]["rosterLocktimeType"] = "INDIVIDUAL_FIRSTGAME_WEEKLY"
    result = nba.run(nba.offer())
    assert len(result.preconditions.failures) == 1
    assert result.preconditions.failures[0].startswith("the league's roster lock type INDIVIDUAL_FIRSTGAME_WEEKLY")


def test_the_trade_deadline_is_the_leagues_own_setting(nfl: League) -> None:
    deadline = nfl.clock - timedelta(hours=1)
    nfl.settings["settings"]["tradeSettings"]["deadlineDate"] = ms(deadline)
    result = nfl.run(nfl.offer())
    assert result.preconditions.failures == (f"the league's trade deadline {deadline:%Y-%m-%d %H:%M} UTC has passed",)

    nfl.settings["settings"]["tradeSettings"]["deadlineDate"] = ms(nfl.clock + timedelta(hours=1))
    assert nfl.run(nfl.offer()).ok  # nothing hardcoded: the same deal goes through before the deadline


def test_preconditions_name_every_problem_at_once(nfl: League) -> None:
    nfl.settings["settings"]["tradeSettings"]["deadlineDate"] = ms(nfl.clock - timedelta(hours=1))
    nfl.entry(15847)["playerPoolEntry"]["tradeLocked"] = True
    payload = TradePayload(other_team_id=TEAM, give_espn_ids=(15847, 999), get_espn_ids=(3126486,))
    result = nfl.run(nfl.propose(payload, ProposalKind.TRADE_PROPOSE))

    failures = result.preconditions.failures
    assert failures[0] == "team 1 is our own team: a trade is with another team"
    assert any("trade deadline" in failure for failure in failures)
    assert "player 999 is not on team 1 (ours) in scoring period 5" in failures
    assert "player 3126486 is not on team 1 (theirs) in scoring period 5" in failures
    assert "ESPN marks Travis Kelce (15847) trade-locked" in failures
    assert nfl.transport.sent == []


def test_a_deal_that_leaves_us_over_the_roster_size_needs_a_drop_the_payload_cannot_carry(nfl: League) -> None:
    result = nfl.run(nfl.offer(give=(15847,), get=tuple(nfl.theirs)))  # one for two, on a full roster
    assert result.preconditions.failures == (
        "the trade would leave us 17 players and the league allows 16: a trade proposal here carries no drop, so make "
        "room first or change the deal",
    )
    notes = nfl.run(nfl.offer(give=tuple(nfl.mine), get=(3126486,)))  # two for one: they would overflow, we would not
    assert notes.ok and notes.preconditions.observed["notes"] == []
    assert nfl.transport.sent  # the 2-for-1 went out


def test_a_timeout_is_never_retried_and_the_re_read_decides(nfl: League) -> None:
    nfl.transport.replies.append(Reply.timed_out(applies=True))
    result = nfl.run(nfl.offer())

    assert len(nfl.transport.sent) == 1
    assert result.status == "verified" and [a.mode for a in result.attempts] == ["api"]  # the re-read found it
    assert result.last is not None and result.last.error is not None and "the write timed out" in result.last.error


def test_a_timeout_that_did_not_land_is_not_a_success_and_is_not_retried(nfl: League) -> None:
    nfl.transport.replies.append(Reply.timed_out(applies=False))
    result = nfl.run(nfl.offer())
    assert result.status == "unknown" and result.proposal.status == "failed"  # state unknown: never a success
    assert len(nfl.transport.sent) == 1 and nfl.browser.opened == []  # no retry, no UI fallback after an unknown


def test_an_accepted_request_the_re_read_does_not_list_is_a_failure(nfl: League) -> None:
    nfl.transport.replies.append(Reply(status=200, applies=False))
    result = nfl.run(nfl.offer())

    assert result.proposal.status == "failed"
    assert result.last is not None and result.last.error == (
        "the re-read does not show the change: no open offer of ours to team 5 for this deal"
    )
    assert len(nfl.transport.sent) == 1


# --- the proposal, UI mode --------------------------------------------------------------------------------------------


def test_the_builder_walk_picks_the_players_reviews_and_sends_through_the_one_confirm(nfl_ui: League) -> None:
    league = nfl_ui
    result = league.run(league.offer(), mode=Mode.UI)

    assert result.ok and result.status == "verified"
    (attempt,) = result.attempts
    assert attempt.mode == "ui" and attempt.request is not None
    assert attempt.request["confirms"] == [
        "send the trade proposal to team 5: Travis Kelce, George Kittle, Rival Receiver, Rival Back"
    ]
    assert league.site.saves == [((3040151, 15847, 3126486, 4432665), "2")]  # picked in roster order
    page = league.site.page
    assert page.did("goto") == [league.builder_url()]
    assert len(page.did("click")) == 2 and len(page.did("check")) == 4  # Continue, then the confirm
    assert {"ui-trace.zip", "ui-01-trade-builder.png", "ui-02-trade-review.png"} <= artifact_names(result)
    assert league.transport.sent == []


def test_an_api_rejection_falls_back_to_the_builder_when_the_flow_lists_the_ui(nfl_ui: League) -> None:
    nfl_ui.transport.replies.append(Reply(status=400, body={"messages": ["Bad request"], "details": []}))
    result = nfl_ui.run(nfl_ui.offer())

    assert result.ok and [(a.mode, a.status) for a in result.attempts] == [("api", "failed"), ("ui", "verified")]
    assert len(nfl_ui.site.saves) == 1


def test_a_ui_dry_run_stops_before_send_trade_proposal(nfl_ui: League) -> None:
    proposal = nfl_ui.offer()
    row = nfl_ui.store.proposals.get(proposal.row_id)
    assert row is not None
    pending = nfl_ui.propose(TradePayload.model_validate(row.payload), ProposalKind.TRADE_PROPOSE, approved=False)
    result = nfl_ui.run(pending, dry_run=True, mode=Mode.UI)

    assert result.ok and result.status == "dry_run"
    assert result.last is not None and result.last.request is not None
    assert result.last.request["stopped_before"].startswith("send the trade proposal to team 5")
    assert nfl_ui.site.saves == [] and nfl_ui.transport.sent == []


def test_the_shipped_flow_has_no_ui_mode_and_the_builder_needs_a_web_sign_in(nfl: League, nfl_ui: League) -> None:
    with pytest.raises(NoFlowError, match="propose_trade has no ui mode; it runs in api"):
        nfl.run(nfl.offer(), mode=Mode.UI)

    nfl_ui.site.signed_in = False
    result = nfl_ui.run(nfl_ui.offer(), mode=Mode.UI)
    assert result.proposal.status == "failed"
    assert result.last is not None and result.last.error is not None
    assert "the trade builder says Log in Required" in result.last.error
    assert nfl_ui.site.page.did("click") == []


def test_the_walk_stops_when_the_builder_lacks_a_player(nfl_ui: League) -> None:
    proposal = nfl_ui.offer()
    nfl_ui.site.page.screens[nfl_ui.builder_url()] = lambda: [
        FakeElement(role="heading", name="Propose Trade Rival Team"),
        FakeElement(role="button", name="Continue"),
    ]
    missing = nfl_ui.run(proposal, mode=Mode.UI)

    assert missing.proposal.status == "failed"
    assert missing.last is not None and missing.last.error is not None
    assert "the trade builder shows no checkbox named 'Trade Travis Kelce'" in missing.last.error
    assert nfl_ui.site.saves == [] and nfl_ui.transport.sent == []


def test_the_walk_re_reads_the_league_and_stops_when_a_precondition_stopped_holding(nfl_ui: League) -> None:
    proposal = nfl_ui.offer()
    nfl_ui.entry(15847)["playerPoolEntry"]["tradeLocked"] = True  # locked after the proposal, before the run
    result = nfl_ui.run(proposal, mode=Mode.UI)

    assert result.preconditions.failures == ("ESPN marks Travis Kelce (15847) trade-locked",)
    assert nfl_ui.site.page.did("goto") == [] and nfl_ui.site.saves == []


# --- accept and decline -----------------------------------------------------------------------------------------------


def test_an_accept_sends_the_documented_envelope_and_is_verified_by_the_players_moving(nfl: League) -> None:
    nfl.incoming()
    result = nfl.run(nfl.answer("offer-in", ProposalKind.TRADE_ACCEPT))

    assert result.ok and result.flow == "respond_trade"
    (request,) = nfl.transport.sent
    assert request.url == transactions_url("ffl") and dict(request.headers) == dict(WEB_CLIENT_HEADERS)
    assert request.body == {
        "isLeagueManager": False,
        "teamId": TEAM,
        "type": "TRADE_ACCEPT",
        "memberId": FAKE_SWID,
        "scoringPeriodId": 5,
        "executionType": "EXECUTE",
        "relatedTransactionId": "offer-in",
    }
    assert nfl.holder(3126486) == TEAM and nfl.holder(15847) == nfl.other
    assert result.last is not None and result.last.verification is not None
    assert result.last.verification["observed"]["players_moved"] is True


def test_a_decline_carries_a_comment_and_is_verified_when_the_offer_closes(nfl: League) -> None:
    nfl.incoming()
    result = nfl.run(nfl.answer("offer-in", ProposalKind.TRADE_DECLINE))

    assert result.ok
    (request,) = nfl.transport.sent
    assert request.body == {
        "isLeagueManager": False,
        "teamId": TEAM,
        "type": "TRADE_DECLINE",
        "memberId": FAKE_SWID,
        "scoringPeriodId": 5,
        "executionType": "EXECUTE",
        "comment": "",
        "relatedTransactionId": "offer-in",
    }
    assert nfl.holder(15847) == TEAM  # nothing moved
    assert result.last is not None and result.last.verification is not None
    assert result.last.verification["detail"] == "offer offer-in is no longer open (status DECLINED)"


def test_an_accept_checks_the_locks_deadline_and_room_but_a_decline_does_not(nfl: League) -> None:
    nfl.incoming()
    nfl.entry(3040151)["playerPoolEntry"]["tradeLocked"] = True  # ours
    nfl.entry(4432665)["playerPoolEntry"]["rosterLocked"] = True  # theirs
    nfl.settings["settings"]["tradeSettings"]["deadlineDate"] = ms(nfl.clock - timedelta(hours=1))
    accept = nfl.run(nfl.answer("offer-in", ProposalKind.TRADE_ACCEPT))
    assert accept.preconditions.failures == (
        f"the league's trade deadline {nfl.clock - timedelta(hours=1):%Y-%m-%d %H:%M} UTC has passed",
        "ESPN marks George Kittle (3040151) trade-locked",
        "ESPN marks Rival Back (4432665) roster-locked: his game has started",
    )
    assert nfl.transport.sent == []

    decline = nfl.run(nfl.answer("offer-in", ProposalKind.TRADE_DECLINE))  # declining stays allowed
    assert decline.ok and nfl.record("offer-in")["status"] == "DECLINED"


def test_a_response_needs_its_offer_open_incoming_and_still_the_evaluated_deal(nfl: League) -> None:
    gone = nfl.run(nfl.answer("nope", ProposalKind.TRADE_DECLINE))
    assert gone.preconditions.failures == (
        "transaction nope is not an open offer (answered, cancelled or expired already); run fm sync",
    )

    nfl.open_offer("ours", proposer=TEAM, give=nfl.mine, get=tuple(nfl.theirs))
    ours = nfl.run(nfl.answer("ours", ProposalKind.TRADE_DECLINE))
    assert ours.preconditions.failures == ("offer ours is ours: withdraw it with a cancel, it cannot be answered",)

    nfl.open_offer("changed", proposer=nfl.other, give=(15847,), get=(3126486, 4432665))
    changed = nfl.run(nfl.answer("changed", ProposalKind.TRADE_ACCEPT))
    assert changed.preconditions.failures[0].startswith("offer changed no longer exchanges what was evaluated")

    nfl.open_offer("late", proposer=nfl.other, give=nfl.mine, get=tuple(nfl.theirs), expires=nfl.clock - timedelta(1))
    late = nfl.run(nfl.answer("late", ProposalKind.TRADE_DECLINE))
    assert late.preconditions.failures[0].startswith("transaction late is not an open offer")
    assert nfl.transport.sent == []


def test_an_accept_the_league_holds_for_review_is_not_verified(nfl: League) -> None:
    nfl.incoming()
    nfl.hold_accepts = True  # ESPN took it but the offer stays pending (a league that reviews trades)
    result = nfl.run(nfl.answer("offer-in", ProposalKind.TRADE_ACCEPT))

    assert result.proposal.status == "failed"
    assert result.last is not None and result.last.error is not None
    assert "offer offer-in is still open: ESPN may be holding the accepted trade (league review)" in result.last.error


def test_an_accept_that_needs_a_drop_is_refused(nfl: League) -> None:
    nfl.open_offer("big", proposer=nfl.other, give=(15847,), get=(3126486, 4432665))
    payload = TradeResponsePayload(
        other_team_id=nfl.other, give_espn_ids=(15847,), get_espn_ids=(3126486, 4432665), espn_transaction_id="big"
    )
    result = nfl.run(nfl.propose(payload, ProposalKind.TRADE_ACCEPT))
    assert result.preconditions.failures == (
        "accepting would leave us 17 players and the league allows 16: a trade response here carries no drop, so "
        "make room first or answer it by hand",
    )


def test_responses_have_no_ui_mode(nfl: League) -> None:
    nfl.incoming()
    proposal = nfl.answer("offer-in", ProposalKind.TRADE_DECLINE)
    with pytest.raises(NoFlowError, match="respond_trade has no ui mode; it runs in api"):
        nfl.run(proposal, mode=Mode.UI)
    assert get_proposal(nfl.store, proposal.row_id).status == "approved"


# --- the cancel -------------------------------------------------------------------------------------------------------


def test_a_cancel_sends_the_documented_envelope_and_is_verified_when_the_offer_closes(nfl: League) -> None:
    nfl.open_offer("mine", proposer=TEAM, give=nfl.mine, get=tuple(nfl.theirs))
    result = nfl.run(nfl.cancel("mine"))

    assert result.ok and result.flow == "cancel_trade"
    (request,) = nfl.transport.sent
    assert request.body == {
        "isLeagueManager": False,
        "teamId": TEAM,
        "type": "TRADE_PROPOSAL",
        "memberId": FAKE_SWID,
        "scoringPeriodId": 5,
        "executionType": "CANCEL",
        "relatedTransactionId": "mine",
    }
    assert "expirationDate" not in request.body and "comment" not in request.body
    assert nfl.record("mine")["status"] == "PENDING"  # still listed PENDING: the CANCEL record is what closes it
    assert result.last is not None and result.last.verification is not None
    assert result.last.verification["detail"] == "offer mine is no longer open"


def test_a_cancel_needs_an_open_offer_of_ours(nfl: League) -> None:
    nfl.incoming()
    nfl.pending["pendingTransactions"].append(
        {
            "id": "claim-1",
            "type": "WAIVER",
            "status": "PENDING",
            "teamId": TEAM,
            "items": [],
            "executionType": "EXECUTE",
        }
    )
    expected = {
        "gone": "transaction gone is not an open offer (answered, cancelled or expired already); run fm sync",
        "offer-in": "offer offer-in is team 5's, not team 1's (ours)",
        "claim-1": "transaction claim-1 is a WAIVER, not a trade offer",
    }
    for offer_id, failure in expected.items():
        result = nfl.run(nfl.cancel(offer_id))
        assert result.preconditions.failures == (failure,), offer_id
        assert result.proposal.status == "failed"
    assert nfl.transport.sent == []


def test_a_cancelled_offer_cannot_be_cancelled_again_and_a_cancel_that_leaves_it_open_fails(nfl: League) -> None:
    nfl.open_offer("mine", proposer=TEAM, give=nfl.mine, get=tuple(nfl.theirs))
    nfl.transport.replies.append(Reply(status=200, applies=False))
    stuck = nfl.run(nfl.cancel("mine"))
    assert stuck.proposal.status == "failed"
    assert (
        stuck.last is not None and stuck.last.error == "the re-read does not show the change: offer mine is still open"
    )

    assert nfl.run(nfl.cancel("mine")).ok  # now the transport lands it
    again = nfl.run(nfl.cancel("mine"))
    assert again.preconditions.failures == (
        "transaction mine is not an open offer (answered, cancelled or expired already); run fm sync",
    )


def test_a_cancel_dry_run_sends_nothing_and_has_no_ui_mode(nfl: League) -> None:
    nfl.open_offer("mine", proposer=TEAM, give=nfl.mine, get=tuple(nfl.theirs))
    dry = nfl.run(nfl.cancel("mine", approved=False), dry_run=True)
    assert dry.ok and dry.status == "dry_run" and nfl.transport.sent == []

    proposal = nfl.cancel("mine")
    with pytest.raises(NoFlowError, match="cancel_trade has no ui mode; it runs in api"):
        nfl.run(proposal, mode=Mode.UI)
