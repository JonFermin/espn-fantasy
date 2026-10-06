"""``add_drop`` (ROADMAP #27): free-agent adds and drops in API mode, the UI fallback on a fake page, and the dry run.

Everything runs offline over the real-league fixtures of #14 (``tests/fixtures/espn/real``): the NBA league on day 1
(Tue Oct 20 2026, first tip 19:00 UTC in the recorded schedule, rosters locking for everyone at that tip under
``FIRSTGAME_SCORINGPERIOD``) and the NFL league on the Monday of week 4 (rosters locking per game under
``INDIVIDUAL_GAME``; everyone but Bijan Robinson, who plays that night, has played). ``fm.browser.fakes`` serves their
``mRoster``, ``mSettings``, ``proTeamSchedules_wl`` and a ``kona_playercard`` cut from the recorded pool to the real
read client, and the fake write transport applies a ``FREEAGENT`` or ``ROSTER`` transaction that lands to the roster
it serves. :class:`Site` models the player list and the roster-fix page with the accessible names the capture saw. The
API bodies are held to the two captured NBA adds (``fba/write_FREEAGENT_{1,2}.json``).
"""

from __future__ import annotations

import functools
import json
from collections.abc import Callable, Iterator
from dataclasses import replace
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
from fm.browser.flows.add_drop import ADD_DROP, AddDrop
from fm.browser.selectors import RosterFixType
from fm.browser.transactions import WEB_CLIENT_HEADERS
from fm.config import Config, Sport
from fm.espn.client import FILTER_HEADER
from fm.espn.ids import Game, ids_for
from fm.executor import ExecutionResult, ExecutorOptions, PreconditionReadError, execute
from fm.proposals import AddDropPayload, ProposalKind, approve, get_proposal, propose
from fm.store import LeagueRow, ProposalRow, Store

REAL = Path(__file__).resolve().parents[1] / "fixtures" / "espn" / "real"
NOW = datetime(2026, 10, 5, 20, 0, tzinfo=UTC)  # the Monday of NFL week 4, 4 p.m. ET; two weeks before NBA day 1
DEADLINE = NOW + timedelta(hours=3)
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
CAPTURED_SWID = "{00000000-0000-0000-0000-000000000001}"
"""The scrubbed SWID in the captures; the fakes sign in as ``FAKE_SWID``."""

# NBA: the recorded free agents (fba/kona_player_info.json) and our roster (fba/mRoster.json, team 1)
DRAYMOND, SIMONS, POOLE = 6589, 4351851, 4277956  # Draymond Green (GSW, no game on day 1), Simons, Poole
CADE, BEY, WIGGINS, JAQUEZ = 4432166, 4397136, 3059319, 4432848
NBA_BENCH, NBA_IR = 12, 13
# NFL: our roster (ffl/mRoster.json, team 1) and a recorded waiver player made a free agent by the tests
BIJAN, HAMPTON, PIERCE = 4430807, 4685382, 4360078
COKER = 4695883  # Jalen Coker, on waivers in the recorded pool
MONDAY_NIGHT_TEAM = 18  # plays the Monday night game of week 4 (after NOW) in the recorded schedule

ADD_BEY_FOR_DRAYMOND = AddDropPayload(add_espn_id=DRAYMOND, drop_espn_id=BEY)
ROSTER_FIX_CONFIRM = "add Draymond Green and drop Saddiq Bey"


def load(game: str, name: str) -> Any:
    return json.loads((REAL / game / name).read_text(encoding="utf-8"))


def transactions_url(game: str) -> str:
    return (
        f"https://lm-api-writes.fantasy.espn.com/apis/v3/games/{game}/seasons/{SEASONS[game]}/segments/0/leagues/"
        f"{LEAGUE_IDS[game]}/transactions/"
    )


def captured_body(game: str, name: str) -> dict[str, Any]:
    """A captured request's body with the fakes' SWID in place of the scrubbed one."""
    body = load(game, name)["body"]
    if "memberId" in body:
        body["memberId"] = FAKE_SWID
    return body


def recorded_league(game: str) -> LeagueRow:
    sport = SPORTS[game]
    return LeagueRow(
        key=sport, sport=sport, espn_league_id=LEAGUE_IDS[game], season=SEASONS[game], team_id=TEAM, as_of=NOW
    )


def ms(at: datetime) -> int:
    return int(at.timestamp() * 1000)


# --- the fake league and its pages ------------------------------------------------------------------------------------


class League:
    """One real-fixture league: a store row, the fake read API over its roster, settings, schedule and player pool, a
    write transport that applies what lands, the fake site and a browser showing it, and a private flow registry."""

    def __init__(self, store: Store, game: str) -> None:
        self.store = store
        self.game = Game(game)
        self.ids = ids_for(self.game)
        self.row = store.leagues.upsert(recorded_league(game))
        self.rosters: dict[str, Any] = load(game, "mRoster.json")
        self.settings: dict[str, Any] = load(game, "mSettings.json")
        self.schedule: dict[str, Any] = load(game, "proTeamSchedules_wl.json")
        self.pending: dict[str, Any] = load(game, "mPendingTransactions.json")
        self.pending.setdefault("pendingTransactions", [])
        self.transactions: dict[str, Any] = load(game, "mTransactions2_waiver_trade.json")
        self.cards: dict[int, dict[str, Any]] = {
            card["id"]: card for card in load(game, "kona_player_info.json")["players"]
        }
        for entry in self.entries():
            self.cards[entry["playerId"]] = entry["playerPoolEntry"]
        self.echo_period = True
        self.clock = NOW
        self.api = FakeEspnApi()
        self.api.serve("mRoster", self._serve_rosters)
        self.api.serve("mSettings", self.settings)
        self.api.serve("proTeamSchedules_wl", self.schedule)
        self.api.serve("kona_playercard", self._serve_cards)
        self.api.serve("mPendingTransactions", lambda _request: self.pending)
        self.api.serve("mTransactions2", lambda _request: self.transactions)
        self.transport = FakeTransport(on_send=self.apply)
        self.site = Site(self)
        self.browser = FakeBrowser(self.site.page)
        self.registry = FlowRegistry()
        self.registry.register(AddDrop())
        self.opener = FakeOpener(fake_runtime(self.api, self.row, transport=self.transport, browser=self.browser))

    # --- proposals and runs ---

    def propose(
        self,
        payload: AddDropPayload,
        *,
        period: int | None = None,
        approved: bool = True,
        deadline: datetime | None = DEADLINE,
    ) -> ProposalRow:
        row = propose(
            self.store,
            CONFIG,
            self.row,
            ProposalKind.ADD_DROP,
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

    def has(self, espn_id: int) -> bool:
        return any(entry["playerId"] == espn_id for entry in self.entries())

    def name_of(self, espn_id: int) -> str:
        return self.cards[espn_id]["player"]["fullName"]

    def has_room(self) -> bool:
        ir = self.ids.ir_slot
        counts = self.settings["settings"]["rosterSettings"]["lineupSlotCounts"]
        size = sum(count for slot, count in counts.items() if int(slot) != ir)
        return sum(1 for entry in self.entries() if entry["lineupSlotId"] != ir) < size

    def free_agent(self, espn_id: int, *, pro_team: int | None = None) -> dict[str, Any]:
        """Make a recorded pool player a free agent whose adds are open (the recorded NFL pool is all on waivers)."""
        card = self.cards[espn_id]
        card["status"] = "FREEAGENT"
        card["onTeamId"] = 0
        card["waiverProcessDate"] = None
        card["rosterLocked"] = False
        card["lineupLocked"] = False
        if pro_team is not None:
            card["player"]["proTeamId"] = pro_team
        return card

    def kickoff(self, pro_team: int, period: int) -> datetime:
        """When ``pro_team`` starts in ``period`` in the recorded schedule."""
        team = next(team for team in self.schedule["settings"]["proTeams"] if team["id"] == pro_team)
        game = team["proGamesByScoringPeriod"][str(period)][0]
        return datetime.fromtimestamp(game["date"] / 1000, UTC)

    def players_url(self) -> str:
        return selectors.players_page_url(self.game, self.row.espn_league_id, TEAM, self.row.season)

    def roster_fix_url(self, player_id: int, kind: RosterFixType) -> str:
        return selectors.roster_fix_url(self.game, self.row.espn_league_id, self.row.season, TEAM, player_id, kind)

    def land(self, add_id: int | None, drop_id: int | None) -> None:
        """ESPN applying an add and/or a drop to our roster."""
        if drop_id is not None:
            self.entries().remove(self.entry(drop_id))
            self.cards[drop_id] = {**self.cards[drop_id], "status": "WAIVERS", "onTeamId": 0}
        if add_id is not None:
            card = {**self.cards[add_id], "status": "ONTEAM", "onTeamId": TEAM}
            self.cards[add_id] = card
            self.entries().append({"playerId": add_id, "lineupSlotId": self.ids.bench_slot, "playerPoolEntry": card})

    def apply(self, request: WriteRequest) -> None:
        """ESPN applying a ``FREEAGENT`` or ``ROSTER`` transaction that lands."""
        items = request.body.get("items", [])
        add = next((item["playerId"] for item in items if item["type"] == "ADD"), None)
        drop = next((item["playerId"] for item in items if item["type"] == "DROP"), None)
        self.land(add, drop)

    def _serve_rosters(self, request: httpx.Request) -> dict[str, Any]:
        period = request.url.params.get("scoringPeriodId")
        if self.echo_period and period is not None:
            return {**self.rosters, "scoringPeriodId": int(period)}
        return self.rosters

    def _serve_cards(self, request: httpx.Request) -> dict[str, Any]:
        """``kona_playercard`` for the ids in the filter: the recorded pool entries, as the sync sees them."""
        header = json.loads(request.headers.get(FILTER_HEADER, "{}"))
        ids = header.get("players", {}).get("filterIds", {}).get("value", [])
        return {"players": [self.cards[player_id] for player_id in ids if player_id in self.cards]}


class Site:
    """The player list and the roster-fix page over :class:`League`, with the names the #14 capture saw: ``Add
    <Name> <Position> for <Team>`` (one click adds with roster room, else the roster-fix page opens), ``Drop Player
    <Name>`` (or ``Can't drop <Name>``), ``Continue to add <Name> and drop <Name>`` and the Confirm Transaction
    dialog's ``Confirm add <Name> and drop <Name>``. ``signed_in=False`` shows Log in Required instead."""

    def __init__(self, league: League) -> None:
        self.league = league
        self.signed_in = True
        self.listed: set[int] | None = None
        """Pool players on the list's first page (``None``: all of them)."""
        self.undroppable: set[int] = set()
        """Players the roster-fix page shows "Can't drop" for, on top of those the API marks undroppable."""
        self.fail_saves = False
        self.picked: int | None = None
        self.saves: list[tuple[str, int, int | None]] = []
        """``(type, add, drop)`` per save the site made."""
        screens: dict[str, Any] = {league.players_url(): self.render_players}
        for player_id in list(league.cards):
            for kind in RosterFixType:
                screens[league.roster_fix_url(player_id, kind)] = functools.partial(
                    self.render_roster_fix, player_id, kind
                )
        self.page = FakePage(screens=screens)

    def _label(self, player_id: int) -> str:
        player = self.league.cards[player_id]["player"]
        position = self.league.ids.position_label(player["defaultPositionId"])
        return f"{player['fullName']} {position} for {self.league.ids.pro_team(player['proTeamId'])}"

    def render_players(self) -> list[FakeElement]:
        if not self.signed_in:
            return [FakeElement(role="heading", name="Log in Required")]
        elements = [
            FakeElement(
                role="combobox",
                name="Status",
                options=("ALL", "AVAILABLE", "WAIVERS", "FREEAGENT", "ONTEAM"),
                value="AVAILABLE",
            )
        ]
        for player_id, card in self.league.cards.items():
            verb = {"FREEAGENT": "Add", "WAIVERS": "Claim"}.get(card.get("status") or "")
            if verb is None or (self.listed is not None and player_id not in self.listed):
                continue
            elements.append(
                FakeElement(
                    role="button",
                    name=f"{verb} {self._label(player_id)}",
                    text=verb.upper(),
                    on_click=functools.partial(self._list_click, player_id, verb),
                )
            )
        return elements

    def _list_click(self, player_id: int, verb: str, page: FakePage) -> None:
        kind = RosterFixType.ADD if verb == "Add" else RosterFixType.CLAIM
        if kind is RosterFixType.ADD and self.league.has_room():
            self._save(kind, player_id, None, page)
        else:
            self.picked = None
            page.goto(self.league.roster_fix_url(player_id, kind))

    def render_roster_fix(self, player_id: int, kind: RosterFixType) -> list[FakeElement]:
        if not self.signed_in:
            return [FakeElement(role="heading", name="Log in Required")]
        elements: list[FakeElement] = []
        for entry in self.league.entries():
            name = entry["playerPoolEntry"]["player"]["fullName"]
            droppable = entry["playerPoolEntry"]["player"].get("droppable", True)
            if droppable and entry["playerId"] not in self.undroppable:
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
        page.show(*self.render_roster_fix(player_id, kind))

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
        self._save(kind, player_id, self.picked, page)

    def _save(self, kind: RosterFixType, add_id: int, drop_id: int | None, page: FakePage) -> None:
        self.saves.append((kind.value, add_id, drop_id))
        self.picked = None
        if self.fail_saves:
            page.show(FakeElement(text="Oops! Looks like something went wrong. Please try again"))
            return
        if kind is RosterFixType.ADD:
            self.league.land(add_id, drop_id)
        page.show(FakeElement(role="heading", name="Transaction complete"))


@pytest.fixture
def nba() -> Iterator[League]:
    with Store.open() as store:  # paths.state_db() in the per-test config dir
        yield League(store, "fba")


@pytest.fixture
def nfl() -> Iterator[League]:
    with Store.open() as store:
        yield League(store, "ffl")


def artifact_names(result: ExecutionResult) -> set[str]:
    return {Path(relative).name for attempt in result.attempts for relative in attempt.artifacts}


# --- registration and selectors ---------------------------------------------------------------------------------------


def test_add_drop_serves_add_drop_proposals_in_both_sports() -> None:
    for sport in ("nfl", "nba"):
        assert flow_for(ProposalKind.ADD_DROP, sport) is ADD_DROP
    assert ADD_DROP.name == "add_drop" and ADD_DROP.modes == (Mode.API, Mode.UI)
    registry = FlowRegistry()
    registry.register(AddDrop())
    assert len(registry) == 2


def test_the_player_list_and_roster_fix_addresses() -> None:
    assert selectors.players_page_url("fba", 2020202, 1, 2027) == (
        "https://fantasy.espn.com/basketball/players/add?leagueId=2020202&teamId=1&seasonId=2027"
    )
    assert selectors.roster_fix_url(Game.FFL, 1010101, 2026, 1, COKER, "claim") == (
        f"https://fantasy.espn.com/football/rosterfix?leagueId=1010101&seasonId=2026&teamId=1&players={COKER}&type=claim"
    )
    assert selectors.roster_fix_url(Game.FBA, 2020202, 2027, 1, DRAYMOND, RosterFixType.ADD).endswith(
        f"&players={DRAYMOND}&type=add"
    )


def shown(page: FakePage, entry: selectors.Selector) -> bool:
    scope = page if entry.within is None else selectors.selector(entry.within).locate(page)
    return entry.locate(scope).count() > 0


def test_every_player_list_selector_resolves_as_its_presence_says(nba: League) -> None:
    """A canary over the fake player list: free agents have Add buttons, waiver players Claim buttons."""
    nba.cards[SIMONS]["status"] = "WAIVERS"
    registered = selectors.selectors_for(selectors.WebPage.PLAYERS)
    assert [entry.key for entry in registered] == ["players.status_filter", "players.add", "players.claim"]
    page = nba.site.page
    page.goto(nba.players_url())
    assert all(shown(page, entry) for entry in registered)
    assert selectors.add_button(page, "Draymond Green").count() == 1
    assert selectors.claim_button(page, "Anfernee Simons").count() == 1
    assert selectors.add_button(page, "Anfernee Simons").count() == 0  # on waivers: Claim, not Add
    nba.site.signed_in = False
    page.goto(nba.players_url())
    assert selectors.LOGIN_REQUIRED.locate(page).count() == 1 and not shown(page, selectors.STATUS_FILTER)


def test_every_roster_fix_selector_resolves_as_its_presence_says(nba: League) -> None:
    nba.entry(CADE)["playerPoolEntry"]["player"]["droppable"] = False  # so the undroppable notice shows too
    registered = selectors.selectors_for(selectors.WebPage.ROSTERFIX)
    revealed = [entry for entry in registered if entry.after is not None]
    assert [entry.key for entry in revealed] == ["rosterfix.confirm_dialog", "rosterfix.confirm"]
    assert selectors.selector("rosterfix.confirm").within == "rosterfix.confirm_dialog"
    page = nba.site.page
    page.goto(nba.roster_fix_url(DRAYMOND, RosterFixType.ADD))
    for entry in registered:
        if entry not in revealed:
            assert shown(page, entry) is (entry.presence is not selectors.Presence.NEVER), entry.key
    assert not any(shown(page, entry) for entry in revealed)
    assert selectors.undroppable_notice(page, "Cade Cunningham").count() == 1
    assert selectors.drop_player_button(page, "Cade Cunningham").count() == 0
    selectors.drop_player_button(page, "Saddiq Bey").click()
    selectors.continue_button(page, "Draymond Green", "Saddiq Bey").click()
    assert all(shown(page, entry) for entry in revealed)  # the dialog and its Confirm, once Continue was clicked
    dialog = selectors.CONFIRM_DIALOG.locate(page)
    assert selectors.confirm_transaction_button(dialog, "Draymond Green", "Saddiq Bey").count() == 1


# --- API mode ---------------------------------------------------------------------------------------------------------


def test_an_add_with_a_drop_sends_the_captured_roster_fix_body_and_verifies_it(nba: League) -> None:
    result = nba.run(nba.propose(ADD_BEY_FOR_DRAYMOND, period=1))

    assert result.ok and result.status == "verified" and result.flow == "add_drop"
    (request,) = nba.transport.sent
    assert request.url == transactions_url("fba") and dict(request.headers) == dict(WEB_CLIENT_HEADERS)
    assert request.body == captured_body("fba", "write_FREEAGENT_2.json")  # the roster-fix page's request
    assert nba.has(DRAYMOND) and not nba.has(BEY)
    (attempt,) = result.attempts
    assert attempt.mode == "api" and attempt.verification is not None
    assert attempt.verification["detail"] == "the roster shows the move"
    observed = result.preconditions.observed
    assert (observed["scoring_period_id"], observed["latest_scoring_period"]) == (1, 1)
    assert observed["roster_lock_type"] == "FIRSTGAME_SCORINGPERIOD"
    assert (observed["roster_count"], observed["roster_size"]) == (13, 13)
    assert observed["add"]["name"] == "Draymond Green" and observed["add"]["status"] == "FREEAGENT"
    assert observed["add"]["cutoff"] == FIRST_TIP.isoformat()  # everyone locks at the day's first tip
    assert observed["drop"]["slot"] == NBA_BENCH and observed["drop"]["cutoff"] == FIRST_TIP.isoformat()
    assert nba.browser.opened == []


def test_a_one_click_add_sends_the_captured_body_without_member_id(nba: League) -> None:
    nba.entry(BEY)["lineupSlotId"] = NBA_IR  # the roster has room now: 12 of 13
    result = nba.run(nba.propose(AddDropPayload(add_espn_id=DRAYMOND), period=1))

    assert result.ok
    (request,) = nba.transport.sent
    assert request.body == captured_body("fba", "write_FREEAGENT_1.json") and "memberId" not in request.body
    assert nba.has(DRAYMOND)


def test_a_bare_drop_is_a_roster_transaction(nba: League) -> None:
    result = nba.run(nba.propose(AddDropPayload(drop_espn_id=BEY), period=1))

    assert result.ok
    (request,) = nba.transport.sent
    assert request.body == {
        "isLeagueManager": False,
        "teamId": TEAM,
        "type": "ROSTER",
        "memberId": FAKE_SWID,
        "scoringPeriodId": 1,
        "executionType": "EXECUTE",
        "items": [{"playerId": BEY, "type": "DROP", "fromTeamId": TEAM}],
    }
    assert not nba.has(BEY)


def test_without_a_period_the_move_is_for_the_current_one(nba: League) -> None:
    result = nba.run(nba.propose(ADD_BEY_FOR_DRAYMOND, period=None))

    assert result.ok and nba.transport.sent[0].body["scoringPeriodId"] == 1
    assert result.preconditions.observed["scoring_period_id"] == 1


# --- preconditions ----------------------------------------------------------------------------------------------------


def test_preconditions_name_every_problem_at_once(nba: League) -> None:
    nba.cards[SIMONS]["status"] = "WAIVERS"
    nba.cards[SIMONS]["waiverProcessDate"] = ms(datetime(2026, 10, 21, 8, 0, tzinfo=UTC))
    result = nba.run(nba.propose(AddDropPayload(add_espn_id=SIMONS, drop_espn_id=999), period=2))

    assert result.preconditions.failures == (
        "scoring period 2 is not current yet: ESPN is on scoring period 1, and a free-agent move is made in the "
        "current period",
        f"Anfernee Simons ({SIMONS}) is on waivers until 2026-10-21 08:00 UTC: claim him with a waiver proposal "
        "instead of adding him",
        "player 999 is not on team 1 in scoring period 2",
        "the roster would hold 14 players and the league allows 13: add a drop to the proposal",
    )
    assert result.proposal.status == "failed" and nba.transport.sent == []
    (attempt,) = result.attempts
    assert attempt.status == "failed" and attempt.error is not None
    assert attempt.error.startswith("preconditions failed, nothing was sent: scoring period 2")


def test_a_rostered_player_an_unknown_one_and_an_undroppable_one_are_refused(nba: League) -> None:
    nba.entry(BEY)["playerPoolEntry"]["player"]["droppable"] = False
    result = nba.run(nba.propose(AddDropPayload(add_espn_id=CADE, drop_espn_id=BEY), period=1))
    assert result.preconditions.failures == (
        f"Cade Cunningham ({CADE}) is already on our team, not a free agent",
        f"Saddiq Bey ({BEY}) is on ESPN's undroppable list",
    )

    unknown = nba.run(nba.propose(AddDropPayload(add_espn_id=424242, drop_espn_id=BEY), period=1))
    assert unknown.preconditions.failures == (
        "ESPN has no player 424242",
        f"Saddiq Bey ({BEY}) is on ESPN's undroppable list",
    )
    assert nba.transport.sent == []


def test_nba_adds_and_drops_close_for_everyone_at_the_days_first_tip(nba: League) -> None:
    nba.clock = FIRST_TIP + timedelta(minutes=30)  # Draymond's Warriors do not even play on day 1
    result = nba.run(nba.propose(ADD_BEY_FOR_DRAYMOND, period=1, deadline=None))

    assert result.preconditions.failures == (
        f"adding Draymond Green ({DRAYMOND}) closed at 2026-10-20 19:00 UTC: adds and drops close for everyone at "
        "the period's first game",
        f"dropping Saddiq Bey ({BEY}) closed at 2026-10-20 19:00 UTC: adds and drops close for everyone at the "
        "period's first game",
    )
    assert result.proposal.status == "failed" and nba.transport.sent == []

    nba.clock = FIRST_TIP - timedelta(hours=1)
    before = nba.run(nba.propose(ADD_BEY_FOR_DRAYMOND, period=1, deadline=None, approved=False), dry_run=True)
    assert before.ok and before.preconditions.ok


def test_nfl_adds_and_drops_close_at_each_players_own_kickoff(nfl: League) -> None:
    nfl.free_agent(COKER, pro_team=MONDAY_NIGHT_TEAM)  # his kickoff is tonight, after NOW
    assert nfl.kickoff(MONDAY_NIGHT_TEAM, 4) > NOW
    hampton_kickoff = nfl.kickoff(nfl.entry(HAMPTON)["playerPoolEntry"]["player"]["proTeamId"], 4)
    assert hampton_kickoff < NOW

    locked = nfl.run(nfl.propose(AddDropPayload(add_espn_id=COKER, drop_espn_id=HAMPTON), period=4))
    assert locked.preconditions.failures == (
        f"dropping Omarion Hampton ({HAMPTON}) closed at {hampton_kickoff:%Y-%m-%d %H:%M} UTC: adds and drops close "
        "at each player's own game",
    )

    undroppable = nfl.run(nfl.propose(AddDropPayload(add_espn_id=COKER, drop_espn_id=BIJAN), period=4))
    assert undroppable.preconditions.failures == (f"Bijan Robinson ({BIJAN}) is on ESPN's undroppable list",)

    nfl.entry(BIJAN)["playerPoolEntry"]["player"]["droppable"] = True
    dry = nfl.run(
        nfl.propose(AddDropPayload(add_espn_id=COKER, drop_espn_id=BIJAN), period=4, approved=False), dry_run=True
    )
    assert dry.ok and dry.last is not None and dry.last.request is not None
    assert dry.last.request["url"] == transactions_url("ffl")
    assert dry.last.request["body"]["type"] == "FREEAGENT" and dry.last.request["body"]["memberId"] == FAKE_SWID
    assert dry.last.request["body"]["items"] == [
        {"playerId": COKER, "type": "ADD", "toTeamId": TEAM},
        {"playerId": BIJAN, "type": "DROP", "fromTeamId": TEAM},
    ]
    assert dry.preconditions.observed["drop"]["cutoff"] == nfl.kickoff(1, 4).isoformat()
    assert nfl.transport.sent == []


def test_espns_own_roster_lock_flag_refuses_too(nba: League) -> None:
    nba.entry(BEY)["playerPoolEntry"]["rosterLocked"] = True
    result = nba.run(nba.propose(ADD_BEY_FOR_DRAYMOND, period=1))

    assert result.preconditions.failures == (f"ESPN marks Saddiq Bey ({BEY}) roster-locked: his game has started",)
    assert nba.transport.sent == []


def test_an_unmapped_roster_lock_type_is_refused_rather_than_guessed(nba: League) -> None:
    nba.settings["settings"]["rosterSettings"]["rosterLocktimeType"] = "FIRSTGAME_WEEKLY"
    result = nba.run(nba.propose(ADD_BEY_FOR_DRAYMOND, period=1))

    assert result.preconditions.failures == (
        "the league's roster lock type FIRSTGAME_WEEKLY is not mapped, so when adds and drops close is unknown; the "
        "move is refused rather than timed by a guess",
    )
    assert result.preconditions.observed["add"]["cutoff"] is None and nba.transport.sent == []


def test_a_past_period_and_an_answer_for_another_period_are_refused(nfl: League) -> None:
    nfl.free_agent(COKER, pro_team=MONDAY_NIGHT_TEAM)
    nfl.entry(BIJAN)["playerPoolEntry"]["player"]["droppable"] = True
    past = nfl.run(nfl.propose(AddDropPayload(add_espn_id=COKER, drop_espn_id=BIJAN), period=3))
    assert past.preconditions.failures == (
        "scoring period 3 is over: ESPN is on scoring period 4, and a free-agent move is made in the current period",
    )

    nfl.echo_period = False  # ESPN answers with week 4's rosters whatever was asked
    proposal = nfl.propose(AddDropPayload(add_espn_id=COKER, drop_espn_id=BIJAN), period=5)
    with pytest.raises(PreconditionReadError, match="asked for scoring period 5's rosters, ESPN answered with"):
        nfl.run(proposal)
    after = get_proposal(nfl.store, proposal.row_id)
    assert after.status == "approved" and after.token_consumed_at is None
    assert nfl.transport.sent == []


# --- verification and the dry run -------------------------------------------------------------------------------------


def test_an_accepted_write_the_re_read_does_not_show_is_a_failure(nba: League) -> None:
    nba.transport.replies.append(Reply(status=200, applies=False))  # ESPN said yes and changed nothing
    result = nba.run(nba.propose(ADD_BEY_FOR_DRAYMOND, period=1))

    assert result.proposal.status == "failed" and result.status == "failed"
    assert result.last is not None and result.last.error == (
        f"the re-read does not show the change: Draymond Green ({DRAYMOND}) is not on team 1; "
        f"Saddiq Bey ({BEY}) is still on team 1"
    )
    assert len(nba.transport.sent) == 1 and nba.browser.opened == []  # no retry, no fallback after an acceptance
    assert result.last.verification is not None
    assert result.last.verification["observed"] == {"add_on_roster": False, "drop_on_roster": True}


def test_a_dry_run_builds_and_saves_the_request_and_sends_nothing(nba: League) -> None:
    proposal = nba.propose(ADD_BEY_FOR_DRAYMOND, period=1, approved=False)
    result = nba.run(proposal, dry_run=True)

    assert result.ok and result.status == "dry_run" and result.dry_run
    assert nba.transport.sent == []  # and a dry run's transport refuses every send anyway
    (attempt,) = result.attempts
    assert attempt.request is not None and attempt.request["body"] == captured_body("fba", "write_FREEAGENT_2.json")
    assert {"preconditions.json", "api-request.json"} <= artifact_names(result)
    assert get_proposal(nba.store, proposal.row_id).status == "proposed"
    assert nba.has(BEY) and not nba.has(DRAYMOND)
    assert nba.api.reads("mRoster") == 1  # no verification re-read


# --- the UI fallback --------------------------------------------------------------------------------------------------


def test_api_failure_falls_back_to_the_roster_fix_page(nba: League) -> None:
    nba.transport.replies.append(Reply(status=400, body={"messages": ["Bad request"], "details": []}))
    result = nba.run(nba.propose(ADD_BEY_FOR_DRAYMOND, period=1))

    assert result.ok and result.proposal.status == "verified"
    assert [(attempt.mode, attempt.status) for attempt in result.attempts] == [("api", "failed"), ("ui", "verified")]
    api, ui = result.attempts
    assert api.error is not None and api.error.startswith("ESPN rejected the request: HTTP 400")
    assert len(nba.transport.sent) == 1
    assert ui.request is not None and ui.request["confirms"] == [ROSTER_FIX_CONFIRM]
    assert nba.site.saves == [("add", DRAYMOND, BEY)]
    assert nba.has(DRAYMOND) and not nba.has(BEY)
    page = nba.site.page
    assert page.did("goto") == [nba.roster_fix_url(DRAYMOND, RosterFixType.ADD)]
    drop, cont, confirm = page.did("click")
    assert "drop player" in drop and "continue to" in cont and "confirm" in confirm
    assert {"ui-trace.zip", "ui-01-rosterfix.png"} <= artifact_names(result)


def test_ui_mode_adds_with_one_click_when_the_roster_has_room(nba: League) -> None:
    nba.entry(BEY)["lineupSlotId"] = NBA_IR
    result = nba.run(nba.propose(AddDropPayload(add_espn_id=DRAYMOND), period=1), mode=Mode.UI)

    assert result.ok and [attempt.mode for attempt in result.attempts] == ["ui"]
    assert result.last is not None and result.last.request is not None
    assert result.last.request["confirms"] == ["add Draymond Green"]
    assert nba.site.saves == [("add", DRAYMOND, None)] and nba.has(DRAYMOND)
    assert nba.site.page.did("goto") == [nba.players_url()] and nba.transport.sent == []


def test_a_ui_dry_run_stops_before_the_confirm(nba: League) -> None:
    result = nba.run(nba.propose(ADD_BEY_FOR_DRAYMOND, period=1, approved=False), dry_run=True, mode=Mode.UI)

    assert result.ok and result.status == "dry_run"
    assert result.last is not None and result.last.request is not None
    assert result.last.request["stopped_before"] == ROSTER_FIX_CONFIRM
    assert nba.site.saves == [] and len(nba.site.page.did("click")) == 2  # the drop and Continue, then nothing
    assert nba.has(BEY) and not nba.has(DRAYMOND)
    assert any(name.startswith("ui-02-before-add") for name in artifact_names(result))


@pytest.mark.parametrize(
    ("prepare", "expected"),
    [
        (lambda site: setattr(site, "signed_in", False), "the roster-fix page says Log in Required"),
        (lambda site: site.undroppable.add(BEY), "the roster-fix page says it can't drop Saddiq Bey"),
    ],
    ids=["signed-out", "undroppable-on-the-page"],
)
def test_the_ui_stops_before_any_click_when_the_page_is_not_as_expected(
    nba: League, prepare: Callable[[Site], object], expected: str
) -> None:
    prepare(nba.site)  # the page disagrees with the API, which the preconditions read
    result = nba.run(nba.propose(ADD_BEY_FOR_DRAYMOND, period=1), mode=Mode.UI)

    assert result.proposal.status == "failed"
    assert result.last is not None and result.last.error is not None
    assert expected in result.last.error
    assert nba.site.saves == [] and nba.site.page.did("click") == []


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        (lambda nba: nba.entries().remove(nba.entry(BEY)), f"Saddiq Bey ({BEY}) is no longer on team 1"),
        (
            lambda nba: nba.cards[DRAYMOND].update(status="ONTEAM", onTeamId=4),
            f"Draymond Green ({DRAYMOND}) is no longer FREEAGENT (ONTEAM on team 4)",
        ),
    ],
    ids=["drop-gone", "add-taken"],
)
def test_the_ui_stops_when_the_league_changed_since_the_preconditions(
    nba: League, change: Callable[[League], object], expected: str
) -> None:
    def page_after_the_change() -> FakePage:  # the UI attempt opens its page after the preconditions were read
        change(nba)
        return nba.site.page

    nba.opener.runtime = replace(nba.opener.runtime, browser=FakeBrowser(page_after_the_change))
    result = nba.run(nba.propose(ADD_BEY_FOR_DRAYMOND, period=1), mode=Mode.UI)

    assert result.proposal.status == "failed" and result.preconditions.ok
    assert result.last is not None and result.last.error is not None
    assert f"the league changed since the preconditions were read: {expected}" in result.last.error
    assert nba.site.page.did("goto") == [] and nba.site.saves == []


def test_the_ui_leaves_a_player_missing_from_the_lists_first_page_to_api_mode(nba: League) -> None:
    nba.entry(BEY)["lineupSlotId"] = NBA_IR
    nba.site.listed = {SIMONS, POOLE}
    result = nba.run(nba.propose(AddDropPayload(add_espn_id=DRAYMOND), period=1), mode=Mode.UI)

    assert result.proposal.status == "failed"
    assert result.last is not None and result.last.error is not None
    assert "the player list shows no Add button for Draymond Green" in result.last.error
    assert nba.site.page.did("click") == []


def test_a_bare_drop_has_no_click_through(nba: League) -> None:
    result = nba.run(nba.propose(AddDropPayload(drop_espn_id=BEY), period=1), mode=Mode.UI)

    assert result.proposal.status == "failed"
    assert result.last is not None and result.last.error is not None
    assert "a bare drop has no captured click-through" in result.last.error
    assert nba.site.page.did("goto") == [] and nba.has(BEY)


def test_a_save_the_page_reports_failed_is_an_unknown_outcome_until_the_re_read(nba: League) -> None:
    nba.site.fail_saves = True  # the Oops! text after the confirm
    result = nba.run(nba.propose(ADD_BEY_FOR_DRAYMOND, period=1), mode=Mode.UI)

    assert result.proposal.status == "failed" and result.status == "unknown"
    assert result.last is not None and result.last.error is not None
    assert "the UI walk failed after a confirm click" in result.last.error
    assert "Oops! Looks like something went wrong" in result.last.error
    assert "the re-read does not confirm it" in result.last.error
    assert nba.site.saves == [("add", DRAYMOND, BEY)] and nba.has(BEY)
