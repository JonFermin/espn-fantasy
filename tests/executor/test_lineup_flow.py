"""``set_lineup`` (ROADMAP #25): lineup moves in API mode, the UI fallback on a fake page, and the fixture dry run.

Everything runs offline over the real-league fixtures of #14 (``tests/fixtures/espn/real``): the NBA league on opening
day (day 1 of 2027, nobody locked yet) and the NFL league on the Monday of week 4 (everyone locked but Bijan Robinson,
who plays that night). ``fm.browser.fakes`` serves their ``mRoster`` and ``mSettings`` to the real read client, and
the fake write transport applies a ``LINEUP`` transaction that lands to the roster it serves. :class:`TeamPage` models
ESPN's lineup editor over the same roster: ``MOVE`` on a player's row, then ``HERE`` on the row he goes to, which
saves the move (a swap when that row holds a player). The last tests drive ``fm execute --dry-run --fixtures``.
"""

from __future__ import annotations

import functools
import json
import shutil
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
import typer
from typer.testing import CliRunner

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
    view_key,
)
from fm.browser.flows import FlowRegistry, Mode, WriteRequest, flow_for
from fm.browser.flows.lineup import SET_LINEUP, SetLineup
from fm.browser.transactions import WEB_CLIENT_HEADERS
from fm.commands import execute as execute_cmd
from fm.config import Config, Sport
from fm.espn.ids import Game
from fm.executor import ExecutionResult, ExecutorError, ExecutorOptions, PreconditionReadError, execute
from fm.proposals import LineupMove, LineupPayload, ProposalKind, approve, get_proposal, propose
from fm.store import LeagueRow, ProposalRow, Store

REAL = Path(__file__).resolve().parents[1] / "fixtures" / "espn" / "real"
NOW = datetime(2026, 10, 5, 20, 0, tzinfo=UTC)
DEADLINE = NOW + timedelta(hours=3)
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
                "policy": {"bench_inactive": "auto", "lineup": "approve"},
            }
            for game in ("ffl", "fba")
        ]
    }
)

# NBA, day 1: our starters, UTIL and bench (fba/mRoster.json, team 1)
CADE, SENGUN, MURPHY = 4432166, 4871144, 4397688
OKONGWU, WIGGINS = 4431680, 3059319  # UTIL, with Immanuel Quickley
JAQUEZ, BEY = 4432848, 4397136  # bench, with Jabari Smith Jr.
PG, SF, C, SG_SF, UTIL, BENCH, IR = 0, 2, 4, 7, 11, 12, 13
# NFL, the Monday of week 4 (ffl/mRoster.json, team 1)
BIJAN, HAMPTON = 4430807, 4685382
RB, BE = 2, 20

SWAP = LineupPayload(
    moves=(
        LineupMove(espn_id=WIGGINS, from_slot_id=UTIL, to_slot_id=BENCH),
        LineupMove(espn_id=JAQUEZ, from_slot_id=BENCH, to_slot_id=UTIL),
    )
)
SWAP_ITEMS = [
    {"playerId": WIGGINS, "type": "LINEUP", "fromLineupSlotId": UTIL, "toLineupSlotId": BENCH},
    {"playerId": JAQUEZ, "type": "LINEUP", "fromLineupSlotId": BENCH, "toLineupSlotId": UTIL},
]
SWAP_CONFIRM = "move Andrew Wiggins to Bench for Jaime Jaquez Jr."
BENCH_BIJAN = LineupPayload(moves=(LineupMove(espn_id=BIJAN, from_slot_id=RB, to_slot_id=BE),))
RECORDED_OWNER = "{00000000-0000-0000-0000-000000000001}"
"""Our team's owner in the recorded ``mTeam+mStandings`` (the scrubbed SWID of Manager 1)."""

runner = CliRunner()


def load(game: str, name: str) -> Any:
    return json.loads((REAL / game / name).read_text(encoding="utf-8"))


def transactions_url(game: str) -> str:
    return (
        f"https://lm-api-writes.fantasy.espn.com/apis/v3/games/{game}/seasons/{SEASONS[game]}/segments/0/leagues/"
        f"{LEAGUE_IDS[game]}/transactions/"
    )


def name_of(entry: dict[str, Any]) -> str:
    return entry["playerPoolEntry"]["player"]["fullName"]


def eligible(entry: dict[str, Any]) -> list[int]:
    return entry["playerPoolEntry"]["player"]["eligibleSlots"]


def locked(entry: dict[str, Any]) -> bool:
    return bool(entry["playerPoolEntry"].get("lineupLocked"))


def recorded_league(game: str) -> LeagueRow:
    """The store's row for a recorded league: the scrubbed ids of tests/fixtures/espn/real, our team 1."""
    sport = SPORTS[game]
    return LeagueRow(
        key=sport, sport=sport, espn_league_id=LEAGUE_IDS[game], season=SEASONS[game], team_id=TEAM, as_of=NOW
    )


# --- the fake league and its team page --------------------------------------------------------------------------------


class League:
    """One real-fixture league: a store row, the fake read API over its roster and settings, a write transport that
    applies what lands, the team page and a browser showing it, and a private flow registry with ``set_lineup``."""

    def __init__(self, store: Store, game: str) -> None:
        self.store = store
        self.game = Game(game)
        self.row = store.leagues.upsert(recorded_league(game))
        self.rosters: dict[str, Any] = load(game, "mRoster.json")
        self.settings: dict[str, Any] = load(game, "mSettings.json")
        self.echo_period = True
        """ESPN answers a roster read with the period it asked for; False serves the recorded period regardless."""
        self.api = FakeEspnApi()
        self.api.serve("mRoster", self._serve_rosters)
        self.api.serve("mSettings", self.settings)
        self.transport = FakeTransport(on_send=self.apply)
        self.team = TeamPage(self)
        self.browser = FakeBrowser(self.team.page)
        self.registry = FlowRegistry()
        self.registry.register(SetLineup())
        self.opener = FakeOpener(fake_runtime(self.api, self.row, transport=self.transport, browser=self.browser))

    # --- proposals and runs ---

    def propose(
        self,
        payload: LineupPayload,
        *,
        period: int | None,
        kind: ProposalKind = ProposalKind.LINEUP,
        approved: bool = True,
    ) -> ProposalRow:
        row = propose(
            self.store,
            CONFIG,
            self.row,
            kind,
            payload,
            created_by="test",
            scoring_period_id=period,
            deadline=DEADLINE,
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
            clock=lambda: NOW,
        )

    # --- the league's state ---

    def entries(self) -> list[dict[str, Any]]:
        team = next(team for team in self.rosters["teams"] if team["id"] == TEAM)
        return team["roster"]["entries"]

    def entry(self, espn_id: int) -> dict[str, Any]:
        return next(entry for entry in self.entries() if entry["playerId"] == espn_id)

    def slot(self, espn_id: int) -> int:
        return self.entry(espn_id)["lineupSlotId"]

    def set_slot(self, espn_id: int, slot: int) -> None:
        self.entry(espn_id)["lineupSlotId"] = slot

    def team_url(self, period: int) -> str:
        return selectors.team_page_url(self.game, self.row.espn_league_id, TEAM, self.row.season, period)

    def lineup_rows(self) -> list[tuple[int, dict[str, Any] | None]]:
        """The team page's rows: each slot the league uses, its players, then an Empty row per open place."""
        counts = self.settings["settings"]["rosterSettings"]["lineupSlotCounts"]
        rows: list[tuple[int, dict[str, Any] | None]] = []
        for raw_slot, count in counts.items():
            slot = int(raw_slot)
            holders = [entry for entry in self.entries() if entry["lineupSlotId"] == slot]
            rows.extend((slot, entry) for entry in holders)
            rows.extend((slot, None) for _ in range(count - len(holders)))
        return rows

    def apply(self, request: WriteRequest) -> None:
        """ESPN applying a ``LINEUP`` transaction that lands."""
        for item in request.body["items"]:
            self.set_slot(item["playerId"], item["toLineupSlotId"])

    def roster_reads(self) -> list[httpx.Request]:
        return [request for request in self.api.requests if view_key(request) == "mRoster"]

    def _serve_rosters(self, request: httpx.Request) -> dict[str, Any]:
        period = request.url.params.get("scoringPeriodId")
        if self.echo_period and period is not None:
            return {**self.rosters, "scoringPeriodId": int(period)}
        return self.rosters


class TeamPage:
    """ESPN's lineup editor over :class:`League`'s roster, with the accessible names the #14 capture saw. Each row
    holds the slot label, the player (or Empty) and ``MOVE`` (named ``Select <Player> to move``) for an unlocked
    player. After ``MOVE`` the rows the picked player may go to show ``HERE`` (named ``Confirm move of <Player> to
    <Slot>``, or ``Move`` on an empty row) and the picked row's button becomes ``Cancel Move of <Player>``; ``HERE``
    saves the move, swapping with the row's player, and redraws the page. ``signed_in=False`` shows ESPN's Log in
    Required heading instead."""

    def __init__(self, league: League) -> None:
        self.league = league
        self.signed_in = True
        self.picked: dict[str, Any] | None = None
        self.saves: list[tuple[int, int]] = []
        """``(player, slot)`` per HERE that saved a move."""
        self.page = FakePage(screens={league.team_url(period): self.render for period in range(1, 30)})

    def render(self) -> list[FakeElement]:
        if not self.signed_in:
            return [FakeElement(role="heading", name="Log in Required")]
        return [FakeElement(role="table").add(*(self._row(slot, entry) for slot, entry in self.league.lineup_rows()))]

    def _row(self, slot: int, entry: dict[str, Any] | None) -> FakeElement:
        label = selectors.slot_label(self.league.game, slot)
        row = FakeElement(role="row").add(
            FakeElement(role="cell", name=label, text=label),
            FakeElement(role="cell", text="Empty" if entry is None else name_of(entry)),
        )
        if self.picked is None:
            if entry is not None and not locked(entry):
                row.add(
                    FakeElement(
                        role="button",
                        name=f"Select {name_of(entry)} to move",  # reads MOVE; named as the #14 capture saw
                        text="MOVE",
                        on_click=functools.partial(self._pick, entry),
                    )
                )
        elif entry is not None and entry is self.picked:
            row.add(FakeElement(role="button", name=f"Cancel Move of {name_of(entry)}", text="MOVE"))
        elif self._may_take(slot, entry):
            here = f"Confirm move of {name_of(entry)} to {label}" if entry is not None else "Move"
            row.add(
                FakeElement(role="button", name=here, text="HERE", on_click=functools.partial(self._here, slot, entry))
            )
        return row

    def _may_take(self, slot: int, entry: dict[str, Any] | None) -> bool:
        picked = self.picked
        assert picked is not None
        if entry is picked or slot == picked["lineupSlotId"] or slot not in eligible(picked):
            return False
        return entry is None or (not locked(entry) and picked["lineupSlotId"] in eligible(entry))

    def _pick(self, entry: dict[str, Any], page: FakePage) -> None:
        self.picked = entry
        page.show(*self.render())

    def _here(self, slot: int, entry: dict[str, Any] | None, page: FakePage) -> None:
        picked = self.picked
        assert picked is not None
        if entry is not None:
            entry["lineupSlotId"] = picked["lineupSlotId"]
        picked["lineupSlotId"] = slot
        self.saves.append((picked["playerId"], slot))
        self.picked = None
        page.show(*self.render())


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


# --- registration -----------------------------------------------------------------------------------------------------


def test_set_lineup_serves_both_lineup_kinds_in_both_sports() -> None:
    for kind in (ProposalKind.LINEUP, ProposalKind.BENCH_INACTIVE):
        for sport in ("nfl", "nba"):
            assert flow_for(kind, sport) is SET_LINEUP
    assert SET_LINEUP.name == "set_lineup" and SET_LINEUP.modes == (Mode.API, Mode.UI)
    registry = FlowRegistry()
    registry.register(SetLineup())
    assert len(registry) == 4


# --- selectors --------------------------------------------------------------------------------------------------------


def test_team_page_address_and_slot_labels() -> None:
    assert selectors.team_page_url(Game.FFL, 1010101, 1, 2026) == (
        "https://fantasy.espn.com/football/team?leagueId=1010101&teamId=1&seasonId=2026"  # what the #14 capture loaded
    )
    assert selectors.team_page_url("fba", 2020202, 1, 2027, 3).endswith(
        "/basketball/team?leagueId=2020202&teamId=1&seasonId=2027&scoringPeriodId=3"
    )
    labels = {
        game: [selectors.slot_label(game, slot) for slot in slots]
        for game, slots in {Game.FFL: (0, 2, 16, 20, 21, 23), Game.FBA: (0, 5, 11, 12, 13)}.items()
    }
    assert labels == {
        Game.FFL: ["QB", "RB", "D/ST", "Bench", "IR", "FLEX"],
        Game.FBA: ["PG", "G", "UTIL", "Bench", "IR"],
    }


def test_every_roster_selector_resolves_as_its_presence_says(nba: League) -> None:
    """A canary over the fake team page: what the flow clicks and what the registry promises agree."""
    registered = selectors.selectors_for(selectors.WebPage.ROSTER)
    everything = selectors.registered_selectors()  # the player list and roster-fix pages (#27) register here too
    assert all(entry in everything for entry in registered) and len({s.key for s in everything}) == len(everything)
    assert selectors.selector("roster.here").after == "roster.move"
    with pytest.raises(KeyError, match="no selector 'roster.nope'"):
        selectors.selector("roster.nope")
    nba.set_slot(OKONGWU, IR)  # an open UTIL place, so every SOMETIMES selector shows up too
    page = nba.team.page
    page.goto(nba.team_url(1))

    def shown(entry: selectors.Selector) -> bool:
        scope = page if entry.within is None else selectors.selector(entry.within).locate(page)
        return entry.locate(scope).count() > 0

    revealed = [entry for entry in registered if entry.after is not None]
    for entry in registered:
        if entry not in revealed:
            assert shown(entry) is (entry.presence is not selectors.Presence.NEVER), entry.key
    assert not any(shown(entry) for entry in revealed)
    selectors.MOVE_BUTTON.locate(selectors.player_row(page, "Andrew Wiggins")).click()
    assert revealed and all(shown(entry) for entry in revealed)  # HERE shows once MOVE picked a player up
    nba.team.signed_in = False
    page.goto(nba.team_url(1))
    assert selectors.LOGIN_REQUIRED.locate(page).count() == 1 and selectors.ROSTER_TABLE.locate(page).count() == 0


# --- API mode ---------------------------------------------------------------------------------------------------------


def test_api_mode_sends_one_roster_transaction_and_verifies_it(nba: League) -> None:
    result = nba.run(nba.propose(SWAP, period=1))

    assert result.ok and result.status == "verified" and result.flow == "set_lineup"
    assert result.proposal.status == "verified"
    (request,) = nba.transport.sent
    assert request.url == transactions_url("fba") and dict(request.headers) == dict(WEB_CLIENT_HEADERS)
    assert request.body == {
        "isLeagueManager": False,
        "teamId": TEAM,
        "type": "ROSTER",
        "memberId": FAKE_SWID,
        "scoringPeriodId": 1,
        "executionType": "EXECUTE",
        "items": SWAP_ITEMS,
    }
    assert (nba.slot(WIGGINS), nba.slot(JAQUEZ)) == (BENCH, UTIL)
    (attempt,) = result.attempts
    assert attempt.mode == "api" and attempt.verification is not None
    assert attempt.verification["detail"] == "every moved player is in his new slot"
    observed = result.preconditions.observed
    assert (observed["transaction_type"], observed["scoring_period_id"], observed["latest_scoring_period"]) == (
        "ROSTER",
        1,
        1,
    )
    assert observed["moves"] == [
        {"espn_id": WIGGINS, "name": f"Andrew Wiggins ({WIGGINS})", "from": "UTIL (11)", "to": "BE (12)"},
        {"espn_id": JAQUEZ, "name": f"Jaime Jaquez Jr. ({JAQUEZ})", "from": "BE (12)", "to": "UTIL (11)"},
    ]
    assert nba.browser.opened == []  # the UI was never needed
    assert nba.api.reads("mSettings") == 1


def test_a_later_day_is_a_future_roster_transaction(nba: League) -> None:
    result = nba.run(nba.propose(SWAP, period=3, kind=ProposalKind.BENCH_INACTIVE))

    assert result.ok
    (request,) = nba.transport.sent
    assert (request.body["type"], request.body["scoringPeriodId"]) == ("FUTURE_ROSTER", 3)
    assert {request.url.params.get("scoringPeriodId") for request in nba.roster_reads()} == {"3"}


def test_nfl_monday_benches_the_unlocked_player_and_refuses_a_locked_one(nfl: League) -> None:
    dry = nfl.run(nfl.propose(BENCH_BIJAN, period=4, approved=False), dry_run=True)
    assert dry.ok and dry.status == "dry_run"
    assert dry.last is not None and dry.last.request is not None
    assert dry.last.request["url"] == transactions_url("ffl")
    assert dry.last.request["body"]["type"] == "ROSTER" and dry.last.request["body"]["scoringPeriodId"] == 4
    assert dry.last.request["body"]["items"] == [
        {"playerId": BIJAN, "type": "LINEUP", "fromLineupSlotId": RB, "toLineupSlotId": BE}
    ]

    swap = LineupPayload(
        moves=(
            LineupMove(espn_id=BIJAN, from_slot_id=RB, to_slot_id=BE),
            LineupMove(espn_id=HAMPTON, from_slot_id=BE, to_slot_id=RB),
        )
    )
    result = nfl.run(nfl.propose(swap, period=4))
    assert result.proposal.status == "failed" and nfl.transport.sent == []
    assert result.preconditions.failures == (f"Omarion Hampton ({HAMPTON}) is locked: his game has started",)


def test_preconditions_name_every_problem_at_once(nba: League) -> None:
    nba.entry(CADE)["playerPoolEntry"]["lineupLocked"] = True  # his game has started
    payload = LineupPayload(
        moves=(
            LineupMove(espn_id=999, from_slot_id=BENCH, to_slot_id=UTIL),
            LineupMove(espn_id=WIGGINS, from_slot_id=BENCH, to_slot_id=UTIL),
            LineupMove(espn_id=CADE, from_slot_id=PG, to_slot_id=BENCH),
            LineupMove(espn_id=JAQUEZ, from_slot_id=BENCH, to_slot_id=SG_SF),
            LineupMove(espn_id=SENGUN, from_slot_id=C, to_slot_id=PG),
            LineupMove(espn_id=BEY, from_slot_id=BENCH, to_slot_id=UTIL),
        )
    )
    result = nba.run(nba.propose(payload, period=1))

    assert result.preconditions.failures == (
        "player 999 is not on team 1 in scoring period 1",
        f"Andrew Wiggins ({WIGGINS}) is in UTIL (11), not BE (12)",
        f"Cade Cunningham ({CADE}) is locked: his game has started",
        "SG/SF (7) is not a slot in this league's lineup",
        f"Alperen Sengun ({SENGUN}) cannot play PG (0); he is eligible for C (4), UTIL (11), BE (12), IR (13)",
        "UTIL (11) would hold 4 players and the league has 3: move the player it holds out in the same proposal",
    )
    assert result.proposal.status == "failed" and nba.transport.sent == []
    (attempt,) = result.attempts
    assert attempt.status == "failed" and attempt.error is not None
    assert attempt.error.startswith("preconditions failed, nothing was sent: player 999")
    assert len(nba.roster_reads()) == 1  # nothing to verify


def test_a_player_moved_twice_or_a_past_week_is_refused(nba: League, nfl: League) -> None:
    twice = LineupPayload(
        moves=(
            LineupMove(espn_id=WIGGINS, from_slot_id=UTIL, to_slot_id=BENCH),
            LineupMove(espn_id=WIGGINS, from_slot_id=UTIL, to_slot_id=SF),
        )
    )
    dry = nba.run(nba.propose(twice, period=1, approved=False), dry_run=True)
    assert not dry.ok and dry.status == "dry_run"
    assert f"Andrew Wiggins ({WIGGINS}) is moved 2 times; give each player one move" in dry.preconditions.failures

    past = nfl.run(nfl.propose(BENCH_BIJAN, period=3))
    assert past.preconditions.failures == (
        "scoring period 3 is over: ESPN is on scoring period 4; a past lineup cannot change",
    )
    assert past.proposal.status == "failed" and nfl.transport.sent == []


def test_an_answer_for_another_period_is_a_refusal_that_spends_nothing(nba: League) -> None:
    nba.echo_period = False  # ESPN answers with day 1's rosters whatever was asked
    proposal = nba.propose(SWAP, period=2)

    with pytest.raises(PreconditionReadError, match="asked for scoring period 2's rosters, ESPN answered with"):
        nba.run(proposal)
    after = get_proposal(nba.store, proposal.row_id)
    assert after.status == "approved" and after.token_consumed_at is None
    assert nba.transport.sent == []


def test_a_dry_run_builds_and_saves_the_request_and_sends_nothing(nba: League) -> None:
    proposal = nba.propose(SWAP, period=1, approved=False)
    result = nba.run(proposal, dry_run=True)

    assert result.ok and result.status == "dry_run" and result.dry_run
    assert nba.transport.sent == []  # and a dry run's transport refuses every send anyway
    (attempt,) = result.attempts
    assert attempt.request is not None and attempt.request["body"]["items"] == SWAP_ITEMS
    assert attempt.request["headers"] == dict(WEB_CLIENT_HEADERS)
    assert {"preconditions.json", "api-request.json"} <= artifact_names(result)
    assert get_proposal(nba.store, proposal.row_id).status == "proposed"
    assert (nba.slot(WIGGINS), nba.slot(JAQUEZ)) == (UTIL, BENCH)
    assert (nba.api.reads("mRoster"), nba.api.reads("mSettings")) == (1, 1)  # no verification re-read


# --- the UI fallback --------------------------------------------------------------------------------------------------


def test_api_failure_falls_back_to_ui_mode_on_the_fake_page(nba: League) -> None:
    nba.transport.replies.append(Reply(status=400, body={"messages": ["Bad request"], "details": []}))
    result = nba.run(nba.propose(SWAP, period=1))

    assert result.ok and result.proposal.status == "verified"
    assert [(attempt.mode, attempt.status) for attempt in result.attempts] == [("api", "failed"), ("ui", "verified")]
    api, ui = result.attempts
    assert api.error is not None and api.error.startswith("ESPN rejected the request: HTTP 400")
    assert len(nba.transport.sent) == 1  # one API request, never retried
    assert ui.request is not None and ui.request["confirms"] == [SWAP_CONFIRM]
    assert nba.team.saves == [(WIGGINS, BENCH)]
    assert (nba.slot(WIGGINS), nba.slot(JAQUEZ)) == (BENCH, UTIL)
    page = nba.team.page
    assert page.did("goto") == [nba.team_url(1)]
    move, here = page.did("click")
    assert "row" in move and "to move" in move and "confirm move of" in here
    assert {"ui-trace.zip", "ui-01-roster.png"} <= artifact_names(result)


@pytest.mark.parametrize(
    "reply",
    [
        Reply.espn_error(409, "TRAN_LINEUP_LOCKED", "Lineup is locked"),  # a league rule: the UI enforces it too
        Reply.espn_error(401, "AUTH_MISSING_CREDENTIALS"),  # signed out: so is the page
        Reply(status=429, body={"messages": ["Too Many Requests"]}),  # the UI would be throttled the same way
    ],
    ids=["league-rule", "signed-out", "throttled"],
)
def test_no_ui_after_a_league_rule_a_sign_in_problem_or_throttling(nba: League, reply: Reply) -> None:
    nba.transport.replies.append(reply)
    result = nba.run(nba.propose(SWAP, period=1))

    assert [(attempt.mode, attempt.status) for attempt in result.attempts] == [("api", "failed")]
    assert result.proposal.status == "failed" and len(nba.transport.sent) == 1
    assert nba.browser.opened == [] and nba.team.saves == []


def test_ui_mode_fills_an_open_slot(nba: League) -> None:
    nba.set_slot(OKONGWU, IR)  # UTIL has an open place now
    fill = LineupPayload(moves=(LineupMove(espn_id=BEY, from_slot_id=BENCH, to_slot_id=UTIL),))
    result = nba.run(nba.propose(fill, period=1), mode=Mode.UI)

    assert result.ok and [attempt.mode for attempt in result.attempts] == ["ui"]
    assert result.last is not None and result.last.request is not None
    assert result.last.request["confirms"] == ["move Saddiq Bey to UTIL"]
    assert nba.team.saves == [(BEY, UTIL)] and nba.slot(BEY) == UTIL
    assert nba.transport.sent == []


def test_a_ui_dry_run_stops_before_the_first_save(nba: League) -> None:
    result = nba.run(nba.propose(SWAP, period=1, approved=False), dry_run=True, mode=Mode.UI)

    assert result.ok and result.status == "dry_run"
    assert result.last is not None and result.last.request is not None
    assert result.last.request["stopped_before"] == SWAP_CONFIRM
    assert nba.team.saves == [] and len(nba.team.page.did("click")) == 1  # MOVE, then nothing
    assert (nba.slot(WIGGINS), nba.slot(JAQUEZ)) == (UTIL, BENCH)
    assert any(name.startswith("ui-02-before-move") for name in artifact_names(result))


def test_ui_mode_needs_a_web_sign_in(nba: League) -> None:
    nba.team.signed_in = False
    result = nba.run(nba.propose(SWAP, period=1), mode=Mode.UI)

    assert result.proposal.status == "failed"
    assert result.last is not None and result.last.error is not None
    assert "the team page says Log in Required" in result.last.error
    assert nba.team.page.did("click") == [] and nba.team.saves == []


def test_moves_the_editor_cannot_click_are_left_to_api_mode(nba: League) -> None:
    rotation = LineupPayload(  # a three-way rotation through full slots: no swap and no open slot to start from
        moves=(
            LineupMove(espn_id=WIGGINS, from_slot_id=UTIL, to_slot_id=BENCH),
            LineupMove(espn_id=JAQUEZ, from_slot_id=BENCH, to_slot_id=SF),
            LineupMove(espn_id=MURPHY, from_slot_id=SF, to_slot_id=UTIL),
        )
    )
    ui_only = nba.run(nba.propose(rotation, period=1), mode=Mode.UI)
    assert ui_only.proposal.status == "failed"
    assert ui_only.last is not None and ui_only.last.error is not None
    assert "do not break down into those; API mode can make them in one transaction" in ui_only.last.error
    assert nba.team.page.did("goto") == []  # refused before the page was even loaded

    api = nba.run(nba.propose(rotation, period=1))
    assert api.ok and len(nba.transport.sent[0].body["items"]) == 3
    assert (nba.slot(WIGGINS), nba.slot(JAQUEZ), nba.slot(MURPHY)) == (BENCH, SF, UTIL)


def test_the_ui_stops_when_the_roster_changed_since_the_preconditions(nba: League) -> None:
    def half_applied(request: WriteRequest) -> None:  # ESPN refuses the request, yet Wiggins has moved
        nba.set_slot(WIGGINS, BENCH)

    nba.transport.on_send = half_applied
    nba.transport.replies.append(Reply(status=400, body={"messages": ["Bad request"]}, applies=True))
    result = nba.run(nba.propose(SWAP, period=1))

    assert [(attempt.mode, attempt.status) for attempt in result.attempts] == [("api", "failed"), ("ui", "failed")]
    error = result.attempts[1].error
    assert error is not None
    assert (
        f"the roster changed since the preconditions were read: Andrew Wiggins ({WIGGINS}) is in BE (12) now" in error
    )
    assert nba.team.page.did("click") == [] and nba.team.saves == []


def test_the_ui_will_not_guess_between_rows_with_the_same_name(nba: League) -> None:
    nba.entry(BEY)["playerPoolEntry"]["player"]["fullName"] = "Andrew Wiggins Jr."
    result = nba.run(nba.propose(SWAP, period=1), mode=Mode.UI)

    assert result.proposal.status == "failed"
    assert result.last is not None and result.last.error is not None
    assert "Andrew Wiggins's row cannot be told apart by name" in result.last.error
    assert nba.team.page.did("click") == []


# --- fm execute -------------------------------------------------------------------------------------------------------


def cli() -> typer.Typer:
    root = typer.Typer()
    root.callback()(lambda: None)  # keep it a group, as fm.cli does
    execute_cmd.register(root)
    return root


def stored_proposal(game: str, payload: LineupPayload, period: int) -> int:
    """A proposed (not approved) lineup move in the state DB ``fm execute`` opens, for the recorded league. No deadline:
    the command runs on the real clock."""
    with Store.open() as store:
        league = store.leagues.upsert(recorded_league(game))
        row = propose(store, CONFIG, league, ProposalKind.LINEUP, payload, created_by="test", scoring_period_id=period)
        return row.row_id


@pytest.mark.parametrize(("game", "payload", "period"), [("fba", SWAP, 1), ("ffl", BENCH_BIJAN, 4)])
def test_fm_execute_dry_runs_a_fixture_proposal_offline(game: str, payload: LineupPayload, period: int) -> None:
    proposal_id = stored_proposal(game, payload, period)
    result = runner.invoke(cli(), ["execute", str(proposal_id), "--dry-run", "--fixtures", str(REAL / game)])

    assert result.exit_code == 0, result.output
    output = result.output
    assert f"note: reading recorded views from {REAL / game}" in output
    assert f"#{proposal_id} lineup in {SPORTS[game]} via set_lineup:" in output
    assert "preconditions: ok" in output and "api: dry run" in output
    assert f"request: POST {transactions_url(game)}" in output
    assert '"type":"ROSTER"' in output and f'"scoringPeriodId":{period}' in output
    assert f'"memberId":"{RECORDED_OWNER}"' in output
    assert "dry run: nothing was sent; it would have been sent" in output
    with Store.open() as store:
        assert get_proposal(store, proposal_id).status == "proposed"


def test_fixtures_back_dry_runs_only() -> None:
    folder = str(REAL / "fba")
    live = runner.invoke(cli(), ["execute", "1", "--fixtures", folder])
    assert live.exit_code == 1 and "error: --fixtures works only with --dry-run" in live.output
    ui = runner.invoke(cli(), ["execute", "1", "--dry-run", "--mode", "ui", "--fixtures", folder])
    assert ui.exit_code == 1 and "cannot walk the UI" in ui.output

    with pytest.raises(ExecutorError, match="recorded views back dry runs only"):
        with execute_cmd.open_fixture_runtime(REAL / "fba", recorded_league("fba"), dry_run=False):
            pass
    views = execute_cmd.RecordedViews(REAL / "fba")
    write = httpx.Request("POST", transactions_url("fba"), json={})
    assert views.handle(write).status_code == 405
    escape = httpx.Request("GET", "https://lm-api-reads.fantasy.espn.com/apis/v3/games/fba", params={"view": "../x"})
    assert views.handle(escape).status_code == 404 and views.served == []


def test_a_view_missing_from_the_folder_is_a_refusal(tmp_path: Path) -> None:
    folder = tmp_path / "recorded"
    folder.mkdir()
    shutil.copy(REAL / "fba" / "mRoster.json", folder / "mRoster.json")  # no mSettings.json
    proposal_id = stored_proposal("fba", SWAP, 1)
    result = runner.invoke(cli(), ["execute", str(proposal_id), "--dry-run", "--fixtures", str(folder)])

    assert result.exit_code == 1
    assert "could not read ESPN" in result.output and "no recorded mSettings" in result.output
    with Store.open() as store:
        assert get_proposal(store, proposal_id).status == "proposed"


def test_the_command_explains_espn_error_codes(nba: League) -> None:
    nba.transport.replies.append(Reply.espn_error(409, "TRAN_LINEUP_LOCKED", "Lineup is locked"))
    result = nba.run(nba.propose(SWAP, period=1))
    lines = execute_cmd.result_lines(result, "nba")

    assert "  response: HTTP 409 TRAN_LINEUP_LOCKED" in lines
    assert "  ESPN TRAN_LINEUP_LOCKED: a player in the move is locked (his game has started)" in lines
    assert lines[-2] == f"proposal #{result.proposal.row_id}: failed"
