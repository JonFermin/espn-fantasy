"""``claim_waiver`` and ``cancel_waiver`` (ROADMAP #27): waiver claims with their bids, their cancellation, the UI
fallback on a fake page, and the dry run.

Everything runs offline over the real-league fixtures of #14 (``tests/fixtures/espn/real``): the NFL league on the
Monday of week 4, whose recorded pool is all on waivers until Wednesday's run (``waiverProcessDate``), and the NBA
league on day 1 (nobody on waivers; the tests put a player there). ``fm.browser.fakes`` serves ``mRoster``,
``mSettings``, ``proTeamSchedules_wl``, ``mPendingTransactions``, ``mTransactions2`` (the recorded claims and offers)
and a ``kona_playercard`` cut from the recorded pool to the real read client, and the fake write transport records a
``WAIVER`` claim that lands as a pending transaction and a cancel as a ``CANCEL`` record. :class:`Site` models the
roster-fix page (``type=claim``) with the names the capture saw. The claim body is held to the captured NFL claim
(``ffl/write_WAIVER_1.json``).
"""

from __future__ import annotations

import functools
import json
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
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
from fm.browser.flows.waiver import CANCEL_WAIVER, CLAIM_WAIVER, CancelWaiverClaim, ClaimWaiver
from fm.browser.selectors import RosterFixType
from fm.browser.transactions import WEB_CLIENT_HEADERS
from fm.config import Config, Sport
from fm.espn.client import FILTER_HEADER
from fm.espn.ids import Game, ids_for
from fm.espn.settings import parse_league_settings
from fm.executor import ExecutionResult, ExecutorOptions, NoFlowError, execute
from fm.proposals import Payload, ProposalKind, TransactionCancelPayload, WaiverPayload, approve, get_proposal, propose
from fm.store import LeagueRow, LeagueSettingsRow, ProposalRow, Store

REAL = Path(__file__).resolve().parents[1] / "fixtures" / "espn" / "real"
NOW = datetime(2026, 10, 5, 20, 0, tzinfo=UTC)  # the Monday of NFL week 4, 4 p.m. ET; two weeks before NBA day 1
DEADLINE = NOW + timedelta(hours=3)
WEDNESDAY_RUN = datetime(2026, 10, 7, 7, 0, tzinfo=UTC)  # the recorded pool's waiverProcessDate: Wed 3 a.m. ET
FIRST_TIP = datetime(2026, 10, 20, 19, 0, tzinfo=UTC)  # NBA day 1's first tip in the recorded schedule
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
                "policy": {"add_drop": "approve", "waiver": "approve", "max_transactions_per_week": 10},
            }
            for game in ("ffl", "fba")
        ]
    }
)

# NFL: the captured claim (ffl/write_WAIVER_1.json) was Colston Loveland for Alec Pierce; both are in the fixtures
LOVELAND, COKER, SMITH = 4723086, 4695883, 4241478  # on waivers in the recorded pool
PIERCE, BIJAN, HAMPTON = 4360078, 4430807, 4685382  # our roster (ffl/mRoster.json, team 1)
NFL_IR = 21
# NBA: the recorded free agents and our roster (fba/kona_player_info.json, fba/mRoster.json)
SIMONS, DRAYMOND = 4351851, 6589
BEY, CADE = 4397136, 4432166
NBA_IR = 13

CLAIM_LOVELAND_FOR_PIERCE = WaiverPayload(add_espn_id=LOVELAND, drop_espn_id=PIERCE)
ROSTER_FIX_CONFIRM = "claim Colston Loveland and drop Alec Pierce"


def load(game: str, name: str) -> Any:
    return json.loads((REAL / game / name).read_text(encoding="utf-8"))


def transactions_url(game: str) -> str:
    return (
        f"https://lm-api-writes.fantasy.espn.com/apis/v3/games/{game}/seasons/{SEASONS[game]}/segments/0/leagues/"
        f"{LEAGUE_IDS[game]}/transactions/"
    )


def recorded_league(game: str) -> LeagueRow:
    sport = SPORTS[game]
    return LeagueRow(
        key=sport, sport=sport, espn_league_id=LEAGUE_IDS[game], season=SEASONS[game], team_id=TEAM, as_of=NOW
    )


def ms(at: datetime) -> int:
    return int(at.timestamp() * 1000)


# --- the fake league and its pages ------------------------------------------------------------------------------------


class League:
    """One real-fixture league: a store row, the fake read API over its roster, settings, schedule, pool, pending and
    recorded transactions, a write transport that records what lands, the fake site and a browser showing it, and a
    private flow registry with the two waiver flows."""

    def __init__(self, store: Store, game: str) -> None:
        self.store = store
        self.game = Game(game)
        self.ids = ids_for(self.game)
        self.row = store.leagues.upsert(recorded_league(game))
        self.rosters: dict[str, Any] = load(game, "mRoster.json")
        self.settings: dict[str, Any] = load(game, "mSettings.json")
        self.schedule: dict[str, Any] = load(game, "proTeamSchedules_wl.json")
        self.teams: dict[str, Any] = load(game, "mTeam+mStandings.json")
        self.pending: dict[str, Any] = load(game, "mPendingTransactions.json")
        self.pending.setdefault("pendingTransactions", [])
        self.transactions: dict[str, Any] = load(game, "mTransactions2_waiver_trade.json")
        pool = load(game, "kona_player_info.json")["players"]
        self.cards: dict[int, dict[str, Any]] = {card["id"]: card for card in pool}
        for entry in self.entries():
            self.cards[entry["playerId"]] = entry["playerPoolEntry"]
        self.echo_period = True
        self.clock = NOW
        self.api = FakeEspnApi()
        self.api.serve("mRoster", self._serve_rosters)
        self.api.serve("mSettings", self.settings)
        self.api.serve("proTeamSchedules_wl", self.schedule)
        self.api.serve("mTeam+mStandings", self.teams)
        self.api.serve("kona_playercard", self._serve_cards)
        self.api.serve("mPendingTransactions", lambda _request: self.pending)
        self.api.serve("mTransactions2", lambda _request: self.transactions)
        self.transport = FakeTransport(on_send=self.apply)
        self.site = Site(self)
        self.browser = FakeBrowser(self.site.page)
        self.registry = FlowRegistry()
        self.registry.register(ClaimWaiver())
        self.registry.register(CancelWaiverClaim())
        self.opener = FakeOpener(fake_runtime(self.api, self.row, transport=self.transport, browser=self.browser))

    # --- proposals and runs ---

    def propose(
        self,
        payload: Payload,
        *,
        period: int | None = None,
        kind: ProposalKind = ProposalKind.WAIVER,
        approved: bool = True,
        deadline: datetime | None = DEADLINE,
    ) -> ProposalRow:
        row = propose(
            self.store,
            CONFIG,
            self.row,
            kind,
            payload,
            created_by="test",
            scoring_period_id=period,
            deadline=deadline,
            now=NOW,
        )
        return approve(self.store, row.row_id, decided_by="test", now=NOW) if approved else row

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

    def entries(self) -> list[dict[str, Any]]:
        team = next(team for team in self.rosters["teams"] if team["id"] == TEAM)
        return team["roster"]["entries"]

    def entry(self, espn_id: int) -> dict[str, Any]:
        return next(entry for entry in self.entries() if entry["playerId"] == espn_id)

    def name_of(self, espn_id: int) -> str:
        return self.cards[espn_id]["player"]["fullName"]

    def on_waivers(self, espn_id: int, *, clears: datetime) -> dict[str, Any]:
        """Put a recorded pool player on waivers until ``clears``."""
        card = self.cards[espn_id]
        card["status"] = "WAIVERS"
        card["onTeamId"] = 0
        card["waiverProcessDate"] = ms(clears)
        return card

    def use_faab(self, *, budget: int, minimum_bid: int, spent: int) -> None:
        """Turn the recorded league into a FAAB league, in ESPN's views and in the synced settings the policy reads."""
        acquisition = self.settings["settings"]["acquisitionSettings"]
        acquisition["isUsingAcquisitionBudget"] = True
        acquisition["acquisitionBudget"] = budget
        acquisition["minimumBid"] = minimum_bid
        team = next(team for team in self.teams["teams"] if team["id"] == TEAM)
        team["transactionCounter"]["acquisitionBudgetSpent"] = spent
        parsed = parse_league_settings(self.settings, game=self.game)
        self.store.settings.upsert(
            LeagueSettingsRow(league_id=self.row.row_id, settings=parsed.model_dump(mode="json"), as_of=NOW)
        )

    def open_claim(
        self, claim_id: str, add: int, *, drop: int | None = None, team: int = TEAM, bid: int | None = None
    ) -> dict[str, Any]:
        """A pending claim in the pending view, as ESPN lists a team's own claims before the run."""
        items = [{"playerId": add, "type": "ADD", "fromTeamId": 0, "toTeamId": team}]
        if drop is not None:
            items.append({"playerId": drop, "type": "DROP", "fromTeamId": team, "toTeamId": 0})
        record = {
            "id": claim_id,
            "type": "WAIVER",
            "status": "PENDING",
            "executionType": "EXECUTE",
            "teamId": team,
            "scoringPeriodId": self.rosters["scoringPeriodId"],
            "bidAmount": bid,
            "proposedDate": ms(NOW - timedelta(hours=1)),
            "isPending": True,
            "items": items,
        }
        self.pending["pendingTransactions"].append(record)
        return record

    def open_offer(self, offer_id: str, *, expires: datetime) -> dict[str, Any]:
        """A pending trade offer of ours in the recorded transactions."""
        record = {
            "id": offer_id,
            "type": "TRADE_PROPOSAL",
            "status": "PENDING",
            "executionType": "EXECUTE",
            "teamId": TEAM,
            "scoringPeriodId": self.rosters["scoringPeriodId"],
            "proposedDate": ms(NOW - timedelta(hours=1)),
            "expirationDate": ms(expires),
            "isPending": True,
            "items": [{"playerId": CADE, "type": "TRADE", "fromTeamId": TEAM, "toTeamId": 4}],
        }
        self.transactions["transactions"].append(record)
        return record

    def open_ids(self) -> list[str]:
        return [record["id"] for record in self.pending["pendingTransactions"]]

    def kickoff(self, espn_id: int, period: int) -> datetime:
        """When the player's pro team starts in ``period`` in the recorded schedule."""
        pro_team = self.cards[espn_id]["player"]["proTeamId"]
        team = next(team for team in self.schedule["settings"]["proTeams"] if team["id"] == pro_team)
        game = team["proGamesByScoringPeriod"][str(period)][0]
        return datetime.fromtimestamp(game["date"] / 1000, UTC)

    def roster_fix_url(self, player_id: int, kind: RosterFixType) -> str:
        return selectors.roster_fix_url(self.game, self.row.espn_league_id, self.row.season, TEAM, player_id, kind)

    def land(self, body: dict[str, Any]) -> None:
        """ESPN recording a ``WAIVER`` transaction that lands: a claim joins the pending view, a cancel adds its
        ``CANCEL`` record to the transactions on file (the claim itself stays listed, as an expired offer does)."""
        if body["executionType"] == "CANCEL":
            self.transactions["transactions"].append(
                {
                    "id": f"cancel-{len(self.transactions['transactions'])}",
                    "type": "WAIVER",
                    "status": "CANCELED",
                    "executionType": "CANCEL",
                    "teamId": body["teamId"],
                    "relatedTransactionId": body["relatedTransactionId"],
                    "isPending": True,
                }
            )
            return
        self.open_claim(
            f"claim-{len(self.pending['pendingTransactions'])}",
            next(item["playerId"] for item in body["items"] if item["type"] == "ADD"),
            drop=next((item["playerId"] for item in body["items"] if item["type"] == "DROP"), None),
            team=body["teamId"],
            bid=body.get("bidAmount"),
        )

    def apply(self, request: WriteRequest) -> None:
        self.land(dict(request.body))

    def _serve_rosters(self, request: httpx.Request) -> dict[str, Any]:
        period = request.url.params.get("scoringPeriodId")
        if self.echo_period and period is not None:
            return {**self.rosters, "scoringPeriodId": int(period)}
        return self.rosters

    def _serve_cards(self, request: httpx.Request) -> dict[str, Any]:
        header = json.loads(request.headers.get(FILTER_HEADER, "{}"))
        ids = header.get("players", {}).get("filterIds", {}).get("value", [])
        return {"players": [self.cards[player_id] for player_id in ids if player_id in self.cards]}


class Site:
    """The roster-fix page (``type=claim``) over :class:`League`, with the names the #14 capture saw: ``Drop Player
    <Name>``, ``Continue to claim <Name> and drop <Name>`` and the dialog's ``Confirm claim <Name> and drop <Name>``,
    which records the claim. ``signed_in=False`` shows Log in Required instead."""

    def __init__(self, league: League) -> None:
        self.league = league
        self.signed_in = True
        self.picked: int | None = None
        self.saves: list[tuple[str, int, int | None]] = []
        screens: dict[str, Any] = {}
        for player_id in list(league.cards):
            for kind in RosterFixType:
                screens[league.roster_fix_url(player_id, kind)] = functools.partial(self.render, player_id, kind)
        self.page = FakePage(screens=screens)

    def render(self, player_id: int, kind: RosterFixType) -> list[FakeElement]:
        if not self.signed_in:
            return [FakeElement(role="heading", name="Log in Required")]
        elements: list[FakeElement] = []
        for entry in self.league.entries():
            name = entry["playerPoolEntry"]["player"]["fullName"]
            if entry["playerPoolEntry"]["player"].get("droppable", True):
                elements.append(
                    FakeElement(
                        role="button",
                        name=f"Drop Player {name}",
                        text="DROP",
                        on_click=functools.partial(self._pick, entry["playerId"], player_id, kind),
                    )
                )
            else:
                elements.append(FakeElement(text=f"Can't drop {name}"))
        elements.append(FakeElement(role="button", name="Cancel", text="Cancel"))
        if self.picked is None:
            elements.append(FakeElement(role="button", name="Continue", text="Continue"))
        else:
            label = self._transaction(player_id, kind)
            elements.append(
                FakeElement(
                    role="button",
                    name=f"Continue to {label}",
                    text="Continue",
                    on_click=functools.partial(self._open_dialog, player_id, kind),
                )
            )
        return elements

    def _transaction(self, player_id: int, kind: RosterFixType) -> str:
        assert self.picked is not None
        return f"{kind.value} {self.league.name_of(player_id)} and drop {self.league.name_of(self.picked)}"

    def _pick(self, drop_id: int, player_id: int, kind: RosterFixType, page: FakePage) -> None:
        self.picked = drop_id
        page.show(*self.render(player_id, kind))

    def _open_dialog(self, player_id: int, kind: RosterFixType, page: FakePage) -> None:
        label = self._transaction(player_id, kind)
        page.add(
            FakeElement(role="dialog", name="Confirm Transaction").add(
                FakeElement(
                    role="button",
                    name=f"Confirm {label}",
                    text="Confirm",
                    on_click=functools.partial(self._confirm, player_id, kind),
                ),
                FakeElement(role="button", name="Cancel", text="Cancel"),
            )
        )

    def _confirm(self, player_id: int, kind: RosterFixType, page: FakePage) -> None:
        drop = self.picked
        self.saves.append((kind.value, player_id, drop))
        self.picked = None
        items = [{"playerId": player_id, "type": "ADD", "toTeamId": TEAM}]
        if drop is not None:
            items.append({"playerId": drop, "type": "DROP", "fromTeamId": TEAM})
        self.league.land({"executionType": "EXECUTE", "teamId": TEAM, "items": items, "bidAmount": None})
        page.show(FakeElement(role="heading", name="Transaction complete"))


@pytest.fixture
def nfl() -> Iterator[League]:
    with Store.open() as store:  # paths.state_db() in the per-test config dir
        yield League(store, "ffl")


@pytest.fixture
def nba() -> Iterator[League]:
    with Store.open() as store:
        yield League(store, "fba")


def artifact_names(result: ExecutionResult) -> set[str]:
    return {Path(relative).name for attempt in result.attempts for relative in attempt.artifacts}


def captured_claim_body(*, scoring_period_id: int) -> dict[str, Any]:
    """The captured claim's body as our run sends it: the fakes' SWID and ESPN's current period at run time."""
    body = load("ffl", "write_WAIVER_1.json")["body"]
    return {**body, "memberId": FAKE_SWID, "scoringPeriodId": scoring_period_id}


# --- registration -----------------------------------------------------------------------------------------------------


def test_the_two_flows_serve_claims_and_cancels_in_both_sports() -> None:
    for sport in ("nfl", "nba"):
        assert flow_for(ProposalKind.WAIVER, sport) is CLAIM_WAIVER
        assert flow_for(ProposalKind.WAIVER_CANCEL, sport) is CANCEL_WAIVER
    assert CLAIM_WAIVER.name == "claim_waiver" and CLAIM_WAIVER.modes == (Mode.API, Mode.UI)
    assert CANCEL_WAIVER.name == "cancel_waiver" and CANCEL_WAIVER.modes == (Mode.API,)


# --- the claim, API mode ----------------------------------------------------------------------------------------------


def test_a_claim_with_a_drop_sends_the_captured_body_and_is_verified_pending(nfl: League) -> None:
    result = nfl.run(nfl.propose(CLAIM_LOVELAND_FOR_PIERCE, period=5))  # the run lands in week 5

    assert result.ok and result.status == "verified" and result.flow == "claim_waiver"
    (request,) = nfl.transport.sent
    assert request.url == transactions_url("ffl") and dict(request.headers) == dict(WEB_CLIENT_HEADERS)
    assert request.body == captured_claim_body(scoring_period_id=4)  # scoringPeriodId is ESPN's current period
    assert request.body["bidAmount"] is None and "bidAmount" in request.body  # a league without FAAB: null
    (attempt,) = result.attempts
    assert attempt.mode == "api" and attempt.verification is not None
    assert attempt.verification["detail"] == "claim claim-0 is pending"
    assert attempt.verification["observed"]["open_claims"][0]["adds"] == [LOVELAND]
    observed = result.preconditions.observed
    assert (observed["scoring_period_id"], observed["latest_scoring_period"]) == (5, 4)
    assert (
        observed["add"]["status"] == "WAIVERS" and observed["add"]["waiver_process_date"] == WEDNESDAY_RUN.isoformat()
    )
    assert observed["drop"]["name"] == "Alec Pierce" and observed["bidding"] == {"uses_faab": False, "bid": None}
    assert observed["open_claims"] == [] and nfl.browser.opened == []
    assert nfl.entry(PIERCE)  # the drop happens at the run, not now


def test_a_claim_for_the_current_period_holds_the_drop_to_the_roster_lock(nfl: League) -> None:
    kickoff = nfl.kickoff(PIERCE, 4)
    assert kickoff < NOW and nfl.cards[PIERCE]["rosterLocked"]  # Pierce has played this week
    result = nfl.run(nfl.propose(CLAIM_LOVELAND_FOR_PIERCE, period=4))

    assert result.preconditions.failures == (
        f"dropping Alec Pierce ({PIERCE}) closed at {kickoff:%Y-%m-%d %H:%M} UTC: adds and drops close at each "
        "player's own game",
    )
    assert result.proposal.status == "failed" and nfl.transport.sent == []


def test_preconditions_name_every_problem_at_once(nfl: League) -> None:
    nfl.cards[COKER]["status"] = "FREEAGENT"
    nfl.cards[COKER]["waiverProcessDate"] = None
    nfl.open_claim("claim-earlier", COKER)
    nfl.use_faab(budget=100, minimum_bid=1, spent=0)  # policy lets a bid through only in a FAAB league...
    proposal = nfl.propose(WaiverPayload(add_espn_id=COKER, drop_espn_id=999, bid_amount=5), period=3)
    nfl.settings["settings"]["acquisitionSettings"]["isUsingAcquisitionBudget"] = False  # ...which ESPN says it is not
    result = nfl.run(proposal)

    assert result.preconditions.failures == (
        "scoring period 3 is over: ESPN is on scoring period 4, so the waiver run the claim was timed to has passed",
        f"Jalen Coker ({COKER}) has cleared waivers: add him with an add_drop proposal instead of claiming him",
        f"a claim for Jalen Coker ({COKER}) is already pending (transaction claim-earlier); cancel it first or leave "
        "it",
        "player 999 is not on team 1 in scoring period 3",
        "this league does not bid FAAB (claims go by priority), and the claim carries a bid of $5",
        "the roster is full (16 of 16): the claim needs a drop, or it fails at the run (FAILED_ROSTERLIMIT)",
    )
    assert result.proposal.status == "failed" and nfl.transport.sent == []
    assert result.preconditions.observed["open_claims"][0]["id"] == "claim-earlier"


def test_a_run_that_has_passed_a_rostered_player_and_an_unknown_one_are_refused(nfl: League) -> None:
    nfl.cards[LOVELAND]["waiverProcessDate"] = ms(NOW - timedelta(hours=2))
    passed = nfl.run(nfl.propose(CLAIM_LOVELAND_FOR_PIERCE, period=5))
    assert passed.preconditions.failures == (
        f"Colston Loveland ({LOVELAND})'s waiver run (2026-10-05 18:00 UTC) has passed; run fm sync and decide again",
    )

    rostered = nfl.run(nfl.propose(WaiverPayload(add_espn_id=HAMPTON, drop_espn_id=PIERCE), period=5))
    assert rostered.preconditions.failures == (f"Omarion Hampton ({HAMPTON}) is already on our team, not on waivers",)

    unknown = nfl.run(nfl.propose(WaiverPayload(add_espn_id=424242, drop_espn_id=PIERCE), period=5))
    assert unknown.preconditions.failures == ("ESPN has no player 424242",)
    assert nfl.transport.sent == []


def test_a_claim_without_a_drop_needs_roster_room(nfl: League) -> None:
    full = nfl.run(nfl.propose(WaiverPayload(add_espn_id=LOVELAND), period=5))
    assert full.preconditions.failures == (
        "the roster is full (16 of 16): the claim needs a drop, or it fails at the run (FAILED_ROSTERLIMIT)",
    )

    nfl.entry(PIERCE)["lineupSlotId"] = NFL_IR  # room now
    result = nfl.run(nfl.propose(WaiverPayload(add_espn_id=LOVELAND), period=5))
    assert result.ok
    (request,) = nfl.transport.sent
    assert request.body["items"] == [{"playerId": LOVELAND, "type": "ADD", "toTeamId": TEAM}]


def test_the_drop_of_a_claim_landing_later_is_not_held_to_todays_lock(nba: League) -> None:
    nba.on_waivers(SIMONS, clears=datetime(2026, 10, 21, 8, 0, tzinfo=UTC))  # the run after day 1
    nba.clock = FIRST_TIP + timedelta(minutes=30)  # the day's first tip has passed: drops close for everyone
    today = nba.run(nba.propose(WaiverPayload(add_espn_id=SIMONS, drop_espn_id=BEY), period=1, deadline=None))
    assert today.preconditions.failures == (
        f"dropping Saddiq Bey ({BEY}) closed at 2026-10-20 19:00 UTC: adds and drops close for everyone at the "
        "period's first game",
    )

    tomorrow = nba.run(nba.propose(WaiverPayload(add_espn_id=SIMONS, drop_espn_id=BEY), period=2, deadline=None))
    assert tomorrow.ok and nba.transport.sent[-1].body["scoringPeriodId"] == 1


def test_an_unmapped_roster_lock_type_matters_only_to_a_drop_in_the_current_period(nba: League) -> None:
    nba.on_waivers(SIMONS, clears=datetime(2026, 10, 21, 8, 0, tzinfo=UTC))
    nba.settings["settings"]["rosterSettings"]["rosterLocktimeType"] = "INDIVIDUAL_FIRSTGAME_WEEKLY"
    today = nba.run(nba.propose(WaiverPayload(add_espn_id=SIMONS, drop_espn_id=BEY), period=1))
    assert today.preconditions.failures == (
        "the league's roster lock type INDIVIDUAL_FIRSTGAME_WEEKLY is not mapped, so when adds and drops close is "
        "unknown; the move is refused rather than timed by a guess",
    )

    later = nba.run(nba.propose(WaiverPayload(add_espn_id=SIMONS, drop_espn_id=BEY), period=2))
    assert later.ok and later.preconditions.observed["drop"]["cutoff"] is None


def test_a_faab_league_needs_a_bid_within_the_minimum_and_the_budget_left(nfl: League) -> None:
    nfl.use_faab(budget=100, minimum_bid=2, spent=80)
    cases = {
        None: "this league bids FAAB for claims: the claim needs a bid",
        1: "the bid $1 is below the league's minimum bid $2",
        30: "the bid $30 is more than the $20 FAAB left ($100 budget, $80 spent)",
    }
    for bid, failure in cases.items():
        payload = WaiverPayload(add_espn_id=LOVELAND, drop_espn_id=PIERCE, bid_amount=bid)
        result = nfl.run(nfl.propose(payload, period=5))
        assert result.preconditions.failures == (failure,), bid
        bidding = result.preconditions.observed["bidding"]
        assert (bidding["uses_faab"], bidding["minimum_bid"], bidding["budget"]) == (True, 2, 100)
        assert bid is None or (bidding["spent"], bidding["left"]) == (80, 20)
    assert nfl.transport.sent == []

    result = nfl.run(nfl.propose(WaiverPayload(add_espn_id=LOVELAND, drop_espn_id=PIERCE, bid_amount=15), period=5))
    assert result.ok
    (request,) = nfl.transport.sent
    assert request.body["bidAmount"] == 15
    assert result.last is not None and result.last.verification is not None
    assert result.last.verification["observed"]["open_claims"][0]["bid_amount"] == 15


def test_an_accepted_claim_the_re_read_does_not_list_is_a_failure(nfl: League) -> None:
    nfl.transport.replies.append(Reply(status=200, applies=False))
    result = nfl.run(nfl.propose(CLAIM_LOVELAND_FOR_PIERCE, period=5))

    assert result.proposal.status == "failed" and result.status == "failed"
    assert result.last is not None and result.last.error == (
        f"the re-read does not show the change: no open claim of team 1 for player {LOVELAND}"
    )
    assert len(nfl.transport.sent) == 1 and nfl.browser.opened == []


def test_a_dry_run_builds_the_claim_and_sends_nothing(nfl: League) -> None:
    proposal = nfl.propose(CLAIM_LOVELAND_FOR_PIERCE, period=5, approved=False)
    result = nfl.run(proposal, dry_run=True)

    assert result.ok and result.status == "dry_run"
    assert nfl.transport.sent == [] and nfl.open_ids() == []
    (attempt,) = result.attempts
    assert attempt.request is not None and attempt.request["body"] == captured_claim_body(scoring_period_id=4)
    assert {"preconditions.json", "api-request.json"} <= artifact_names(result)
    assert get_proposal(nfl.store, proposal.row_id).status == "proposed"


# --- the claim, UI mode -----------------------------------------------------------------------------------------------


def test_api_failure_falls_back_to_the_roster_fix_page_as_a_claim(nfl: League) -> None:
    nfl.transport.replies.append(Reply(status=400, body={"messages": ["Bad request"], "details": []}))
    result = nfl.run(nfl.propose(CLAIM_LOVELAND_FOR_PIERCE, period=5))

    assert result.ok and result.proposal.status == "verified"
    assert [(attempt.mode, attempt.status) for attempt in result.attempts] == [("api", "failed"), ("ui", "verified")]
    ui = result.attempts[1]
    assert ui.request is not None and ui.request["confirms"] == [ROSTER_FIX_CONFIRM]
    assert nfl.site.saves == [("claim", LOVELAND, PIERCE)] and nfl.open_ids() == ["claim-0"]
    page = nfl.site.page
    assert page.did("goto") == [nfl.roster_fix_url(LOVELAND, RosterFixType.CLAIM)]
    assert len(page.did("click")) == 3 and {"ui-trace.zip", "ui-01-rosterfix.png"} <= artifact_names(result)


def test_a_ui_dry_run_stops_before_the_confirm(nfl: League) -> None:
    result = nfl.run(nfl.propose(CLAIM_LOVELAND_FOR_PIERCE, period=5, approved=False), dry_run=True, mode=Mode.UI)

    assert result.ok and result.status == "dry_run"
    assert result.last is not None and result.last.request is not None
    assert result.last.request["stopped_before"] == ROSTER_FIX_CONFIRM
    assert nfl.site.saves == [] and nfl.open_ids() == []


def test_the_ui_leaves_a_claim_without_a_drop_or_with_a_bid_to_api_mode(nfl: League) -> None:
    nfl.entry(PIERCE)["lineupSlotId"] = NFL_IR
    no_drop = nfl.run(nfl.propose(WaiverPayload(add_espn_id=LOVELAND), period=5), mode=Mode.UI)
    assert no_drop.proposal.status == "failed"
    assert no_drop.last is not None and no_drop.last.error is not None
    assert "a claim without a drop" in no_drop.last.error

    nfl.use_faab(budget=100, minimum_bid=1, spent=0)
    bid = nfl.run(
        nfl.propose(WaiverPayload(add_espn_id=LOVELAND, drop_espn_id=PIERCE, bid_amount=7), period=5), mode=Mode.UI
    )
    assert bid.proposal.status == "failed"
    assert bid.last is not None and bid.last.error is not None
    assert "a claim with a FAAB bid" in bid.last.error
    assert nfl.site.page.did("goto") == [] and nfl.site.saves == []


def test_ui_mode_needs_a_web_sign_in(nfl: League) -> None:
    nfl.site.signed_in = False
    result = nfl.run(nfl.propose(CLAIM_LOVELAND_FOR_PIERCE, period=5), mode=Mode.UI)

    assert result.proposal.status == "failed"
    assert result.last is not None and result.last.error is not None
    assert "the roster-fix page says Log in Required" in result.last.error
    assert nfl.site.page.did("click") == []


# --- the cancel -------------------------------------------------------------------------------------------------------


def test_a_cancel_sends_the_documented_envelope_and_is_verified_when_the_claim_closes(nfl: League) -> None:
    nfl.open_claim("c-1", LOVELAND, drop=PIERCE)
    proposal = nfl.propose(TransactionCancelPayload(espn_transaction_id="c-1"), kind=ProposalKind.WAIVER_CANCEL)
    result = nfl.run(proposal)

    assert result.ok and result.flow == "cancel_waiver"
    (request,) = nfl.transport.sent
    assert request.url == transactions_url("ffl")
    assert request.body == {
        "isLeagueManager": False,
        "teamId": TEAM,
        "type": "WAIVER",
        "memberId": FAKE_SWID,
        "scoringPeriodId": 4,
        "executionType": "CANCEL",
        "relatedTransactionId": "c-1",
    }
    assert "bidAmount" not in request.body
    assert result.preconditions.observed["claim"]["adds"] == [LOVELAND]
    assert result.last is not None and result.last.verification is not None
    assert result.last.verification["detail"] == "claim c-1 is no longer open"
    assert nfl.open_ids() == ["c-1"]  # still listed, like an expired offer: the CANCEL record is what closes it


def test_a_cancel_needs_an_open_claim_of_ours(nfl: League) -> None:
    nfl.open_claim("theirs", COKER, team=8)
    nfl.open_offer("offer-1", expires=NOW + timedelta(days=1))
    expected = {
        "gone": "transaction gone is not an open claim or offer (processed, cancelled or expired already); run fm sync",
        "theirs": "transaction theirs is team 8's, not team 1's (ours)",
        "offer-1": "transaction offer-1 is a TRADE_PROPOSAL, not a waiver claim",
    }
    for transaction_id, failure in expected.items():
        proposal = nfl.propose(
            TransactionCancelPayload(espn_transaction_id=transaction_id), kind=ProposalKind.WAIVER_CANCEL
        )
        result = nfl.run(proposal)
        assert result.preconditions.failures == (failure,), transaction_id
        assert result.proposal.status == "failed"
    assert nfl.transport.sent == []


def test_a_cancelled_claim_cannot_be_cancelled_again(nfl: League) -> None:
    nfl.open_claim("c-1", LOVELAND)
    nfl.land({"executionType": "CANCEL", "teamId": TEAM, "relatedTransactionId": "c-1"})
    proposal = nfl.propose(TransactionCancelPayload(espn_transaction_id="c-1"), kind=ProposalKind.WAIVER_CANCEL)
    result = nfl.run(proposal)

    assert result.preconditions.failures == (
        "transaction c-1 is not an open claim or offer (processed, cancelled or expired already); run fm sync",
    )


def test_an_accepted_cancel_that_leaves_the_claim_open_is_a_failure_and_a_dry_run_sends_nothing(nfl: League) -> None:
    nfl.open_claim("c-1", LOVELAND)
    nfl.transport.replies.append(Reply(status=200, applies=False))
    result = nfl.run(nfl.propose(TransactionCancelPayload(espn_transaction_id="c-1"), kind=ProposalKind.WAIVER_CANCEL))
    assert result.proposal.status == "failed"
    assert (
        result.last is not None and result.last.error == "the re-read does not show the change: claim c-1 is still open"
    )

    proposal = nfl.propose(
        TransactionCancelPayload(espn_transaction_id="c-1"), kind=ProposalKind.WAIVER_CANCEL, approved=False
    )
    dry = nfl.run(proposal, dry_run=True)
    assert dry.ok and dry.status == "dry_run" and len(nfl.transport.sent) == 1
    assert dry.last is not None and dry.last.request is not None
    assert dry.last.request["body"]["relatedTransactionId"] == "c-1"


def test_a_cancel_has_no_ui_mode(nfl: League) -> None:
    nfl.open_claim("c-1", LOVELAND)
    proposal = nfl.propose(TransactionCancelPayload(espn_transaction_id="c-1"), kind=ProposalKind.WAIVER_CANCEL)
    with pytest.raises(NoFlowError, match="cancel_waiver has no ui mode; it runs in api"):
        nfl.run(proposal, mode=Mode.UI)
    assert get_proposal(nfl.store, proposal.row_id).status == "approved"
