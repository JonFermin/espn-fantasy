"""Open trade offers and the answers drafted for the incoming ones (ROADMAP #44).

Reading: the real NBA league's recorded ``mTransactions2`` (three offers of ours, each closed by a ``CANCEL`` record
that still says ``isPending``) shows what "open" means (docs/espn-api.md section 1 #2). Answering: the hand-made NFL
league of ``test_trades.py`` (six teams, 14 players each, we are team 3, deep at running back with a hole at tight end,
team 2 the mirror image), cut down to what these tests need, with incoming offers built by hand. Nothing here reaches
the network or an executor: ``fm.decide.offers`` has no write path, and a test proves it.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from fm.browser.fakes import FakeEspnApi
from fm.config import Config
from fm.decide import offers as offers_module
from fm.decide import registry as decisions
from fm.decide.offers import (
    ANSWER_KINDS,
    OFFERS_CREATED_BY,
    OFFERS_KIND,
    Direction,
    OffersError,
    OfferView,
    decide_offers,
    duplicate_blockers,
    open_offers,
    view_of,
)
from fm.decide.trades import TradeContext, load_trade_context
from fm.espn.ids import FFL
from fm.espn.models import Transaction, TransactionsView
from fm.espn.settings import LeagueSettings, parse_league_settings
from fm.model.projections import ESPN, BlendWeights
from fm.model.valuation import GAMES_STAT
from fm.proposals import (
    PolicyError,
    ProposalKind,
    TradePayload,
    TradeResponsePayload,
    approve,
    auto_approve_due,
    evaluate,
    get_proposal,
    propose,
    reject,
)
from fm.store import (
    LeagueRow,
    LeagueSettingsRow,
    PlayerRow,
    ProjectionRow,
    RosterEntryRow,
    Store,
    TeamRow,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
REAL = FIXTURES / "espn" / "real"
PPR = FIXTURES / "espn" / "ffl_settings_ppr.json"
SEASON, WEEK, LEAGUE_ID, US, THEM = 2026, 4, 1234567, 3, 2
NOW = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)  # Sunday 11 a.m. ET, before the early games
EQUAL_WEIGHTS = BlendWeights.parse("[nfl.default]\nespn = 1.0\nsleeper = 1.0\n\n[nba.default]\nespn = 1.0\n")
QB, RB, WR, TE, DST, K, FLEX = (FFL.slot_id(label) for label in ("QB", "RB", "WR", "TE", "D/ST", "K", "FLEX"))
BENCH, IR = FFL.bench_slot, FFL.ir_slot
SLOTS = {"QB": {QB}, "RB": {RB, FLEX}, "WR": {WR, FLEX}, "TE": {TE, FLEX}, "D/ST": {DST}, "K": {K}}
TEAMS = (1, 2, 3, 4, 5, 6)
CONFIG = Config.model_validate(
    {
        "league": [
            {"key": "nfl", "sport": "nfl", "espn_league_id": LEAGUE_ID, "season": SEASON, "team_id": US},
        ]
    }
)

TEMPLATE: tuple[tuple[str, str, int, float], ...] = (
    ("QB1", "QB", QB, 20),
    ("QB2", "QB", BENCH, 10),
    ("RB1", "RB", RB, 15),
    ("RB2", "RB", RB, 12),
    ("RB3", "RB", FLEX, 9),
    ("RB4", "RB", BENCH, 5),
    ("WR1", "WR", WR, 14),
    ("WR2", "WR", WR, 11),
    ("WR3", "WR", BENCH, 8),
    ("WR4", "WR", BENCH, 5),
    ("TE1", "TE", TE, 9),
    ("TE2", "TE", BENCH, 4),
    ("DST", "D/ST", DST, 8),
    ("K", "K", K, 8),
)
FACTOR = {1: 1.10, 2: 1.06, 3: 1.00, 4: 0.98, 5: 0.92, 6: 0.90}
OVERRIDES: dict[tuple[int, str], float] = {
    (US, "RB1"): 18,
    (US, "RB2"): 15,
    (US, "RB3"): 12,
    (US, "TE1"): 3,
    (US, "TE2"): 2,
    (THEM, "RB1"): 9,
    (THEM, "RB2"): 7,
    (THEM, "RB3"): 5,
    (THEM, "TE1"): 14,
    (THEM, "TE2"): 9,
}


def pid(team: int, name: str) -> int:
    return 1000 + team * 20 + [row[0] for row in TEMPLATE].index(name)


def points_of(team: int, name: str) -> float:
    base = {row[0]: row[3] for row in TEMPLATE}[name]
    return OVERRIDES.get((team, name), base * FACTOR[team])


def ppr() -> LeagueSettings:
    view = json.loads(PPR.read_text(encoding="utf-8"))
    view["settings"]["size"] = len(TEAMS)
    view["settings"]["scheduleSettings"].update({"playoffTeamCount": 4})
    return parse_league_settings(view)


def seed_league(store: Store, *, extra: tuple[tuple[int, str, str, int, int, float], ...] = ()) -> LeagueRow:
    """The six-team league as ``fm sync`` would store it: every roster, the players and ESPN's week and season lines."""
    league = store.leagues.upsert(
        LeagueRow(key="nfl", sport="nfl", espn_league_id=LEAGUE_ID, season=SEASON, team_id=US, as_of=NOW)
    )
    store.settings.upsert(LeagueSettingsRow(league_id=league.row_id, settings=ppr().model_dump(mode="json"), as_of=NOW))
    store.teams.upsert_many(
        TeamRow(league_id=league.row_id, team_id=team, name=f"Team {team}", as_of=NOW) for team in TEAMS
    )
    people: list[tuple[int, str, str, int, int, float]] = [
        (team, f"T{team} {name}", position, slot, pid(team, name), points_of(team, name))
        for team in TEAMS
        for name, position, slot, _ in TEMPLATE
    ]
    people.extend(extra)
    for team in TEAMS:
        store.rosters.replace(
            league.row_id,
            WEEK,
            team,
            [
                RosterEntryRow(
                    league_id=league.row_id,
                    scoring_period_id=WEEK,
                    team_id=team,
                    espn_id=espn_id,
                    lineup_slot_id=slot,
                    as_of=NOW,
                )
                for owner, _, _, slot, espn_id, _ in people
                if owner == team
            ],
        )
    store.players.upsert_many(
        [
            PlayerRow(
                sport="nfl",
                espn_id=espn_id,
                full_name=name,
                default_position_id=FFL.position_id(position),
                position=position,
                pro_team_id=40 + team,
                eligible_slot_ids=[*SLOTS[position], BENCH, IR],
                as_of=NOW,
            )
            for team, name, position, _, espn_id, _ in people
        ]
    )
    store.projections.upsert_many(
        [
            ProjectionRow(
                sport="nfl",
                espn_id=row[4],
                source=ESPN,
                season=SEASON,
                scoring_period_id=period,
                stats=stats,
                as_of=NOW,
            )
            for row in people
            for period, stats in ((WEEK, {"REY": row[5] * 10}), (0, {"REY": row[5] * 140, GAMES_STAT: 14}))
        ]
    )
    return league


@pytest.fixture
def store() -> Iterator[Store]:
    with Store.open(":memory:") as opened:
        yield opened


def context(store: Store, *, now: datetime = NOW, **seeded: Any) -> TradeContext:
    league = seed_league(store, **seeded)
    return load_trade_context(store, league, now=now, config=CONFIG, weights=EQUAL_WEIGHTS)


def offer_record(
    offer_id: str,
    *,
    proposer: int,
    give: tuple[int, ...] = (),
    get: tuple[int, ...] = (),
    other: int = THEM,
    expires: datetime | None = None,
    proposed: datetime | None = None,
    extra_items: tuple[dict[str, Any], ...] = (),
    status: str = "PENDING",
    execution: str = "EXECUTE",
    related: str | None = None,
) -> Transaction:
    """A ``mTransactions2`` trade record: ``give`` are our players it takes, ``get`` the ones it brings."""
    items = [{"playerId": p, "type": "TRADE", "fromTeamId": US, "toTeamId": other} for p in give]
    items += [{"playerId": p, "type": "TRADE", "fromTeamId": other, "toTeamId": US} for p in get]
    items += list(extra_items)
    data: dict[str, Any] = {
        "id": offer_id,
        "type": "TRADE_PROPOSAL",
        "status": status,
        "executionType": execution,
        "teamId": proposer,
        "isPending": True,
        "proposedDate": int((proposed or NOW - timedelta(hours=1)).timestamp() * 1000),
        "expirationDate": int((expires or NOW + timedelta(hours=47)).timestamp() * 1000),
        "items": items,
    }
    if related is not None:
        data["relatedTransactionId"] = related
    return Transaction.model_validate(data)


def an_offer_we_like(offer_id: str = "in-1", **kwargs: Any) -> Transaction:
    """Team 2 asks for our second back and gives its tight end: it fills our hole and theirs."""
    return offer_record(offer_id, proposer=THEM, give=(pid(US, "RB2"),), get=(pid(THEM, "TE1"),), **kwargs)


# --- what is open -----------------------------------------------------------------------------------------------------


def test_the_real_leagues_offers_are_all_closed_by_their_cancel_records() -> None:
    view = TransactionsView.model_validate_json((REAL / "fba" / "mTransactions2_waiver_trade.json").read_text("utf-8"))
    before_expiry = datetime(2026, 10, 5, 16, 0, tzinfo=UTC)
    assert open_offers(view.transactions, 1, before_expiry) == ()  # six records said pending, none was open

    without_cancels = [t for t in view.transactions if not t.is_cancellation]
    live = open_offers(without_cancels, 1, before_expiry)
    assert [offer.direction for offer in live] == [Direction.OUTGOING] * 3  # all three are ours, to team 15
    assert {offer.other_team_id for offer in live} == {15}
    assert sorted((len(offer.give), len(offer.get)) for offer in live) == [(1, 1), (2, 2), (2, 2)]
    assert open_offers(without_cancels, 1, datetime(2026, 10, 6, tzinfo=UTC)) == ()  # offers expire after 48 hours


def test_an_offer_is_incoming_when_another_team_proposed_it() -> None:
    mine = view_of(offer_record("a", proposer=US, give=(1, 2), get=(3,)), US)
    theirs = view_of(offer_record("b", proposer=THEM, give=(1,), get=(3, 4)), US)
    assert mine is not None and theirs is not None
    assert (mine.direction, mine.give, mine.get, mine.other_team_id) == (Direction.OUTGOING, (1, 2), (3,), THEM)
    assert (theirs.direction, theirs.give, theirs.get, theirs.proposer_team_id) == (
        Direction.INCOMING,
        (1,),
        (3, 4),
        THEM,
    )
    assert isinstance(mine.payload(), TradePayload) and not isinstance(mine.payload(), TradeResponsePayload)
    payload = theirs.payload()
    assert isinstance(payload, TradeResponsePayload) and payload.espn_transaction_id == "b"
    assert (payload.give_espn_ids, payload.get_espn_ids) == ((1,), (3, 4))
    assert view_of(offer_record("c", proposer=4, give=(), get=(), other=5), US) is None  # nothing of ours in it


def test_an_offer_this_tool_cannot_value_is_kept_with_its_reasons() -> None:
    faab = {"type": "ACQUISITION_BUDGET_TRADE", "playerId": 0, "fromTeamId": THEM, "toTeamId": US}
    budget = view_of(offer_record("a", proposer=THEM, give=(1,), get=(2,), extra_items=(faab,)), US)
    assert budget is not None and not budget.supported
    assert budget.unsupported == (
        "it carries ACQUISITION_BUDGET_TRADE items, which this tool does not value or answer",
    )

    three = offer_record("b", proposer=THEM, give=(1,), get=(2,))
    extra = {"type": "TRADE", "playerId": 9, "fromTeamId": 5, "toTeamId": US}
    wide = view_of(
        Transaction.model_validate(
            {**three.model_dump(by_alias=True), "items": [*three.model_dump(by_alias=True)["items"], extra]}
        ),
        US,
    )
    assert wide is not None and any("two-team deals only" in reason for reason in wide.unsupported)
    with pytest.raises(OffersError, match="cannot be evaluated"):
        wide.spec()


def test_a_duplicate_is_an_open_offer_to_the_same_team_or_one_that_reuses_a_player() -> None:
    offers = open_offers(
        [
            offer_record("ours", proposer=US, give=(1,), get=(2,)),
            offer_record("theirs", proposer=4, give=(7,), get=(8,), other=4),
        ],
        US,
        NOW,
    )
    assert duplicate_blockers(offers, THEM, (50,), (51,)) == ["an offer of ours to team 2 is already open (ours)"]
    assert duplicate_blockers(offers, 5, (1, 7), (60,)) == [
        "players [1] are already in the open offer ours",
        "players [7] are already in the open offer theirs",
    ]
    assert duplicate_blockers(offers, 5, (50,), (51,)) == []
    assert duplicate_blockers([], THEM, (1,), (2,)) == []


# --- answering incoming offers ----------------------------------------------------------------------------------------


def test_an_offer_that_helps_us_becomes_an_approve_only_accept_proposal(store: Store) -> None:
    ctx = context(store)
    record = an_offer_we_like()
    decision = decide_offers(
        store, CONFIG, ctx.league, now=NOW, offers=open_offers([record], US, NOW), weights=EQUAL_WEIGHTS
    )

    (outcome,) = decision.answers
    assert outcome.blocked is None and outcome.evaluation is not None and outcome.evaluation.recommendation == "accept"
    (row,) = decision.proposals
    assert row.kind == ProposalKind.TRADE_ACCEPT.value and row.status == "proposed"
    assert row.policy == "approve" and row.created_by == OFFERS_CREATED_BY and row.dedupe_key == "offer:in-1"
    payload = TradeResponsePayload.model_validate(row.payload)
    assert payload.espn_transaction_id == "in-1" and payload.other_team_id == THEM
    assert payload.give_espn_ids == (pid(US, "RB2"),) and payload.get_espn_ids == (pid(THEM, "TE1"),)
    assert row.deadline == NOW + timedelta(hours=47)  # the proposal runs out with the offer
    assert row.scoring_period_id == ctx.lock_period
    assert row.engine_numbers["offer"] == {
        "id": "in-1",
        "from_team_id": THEM,
        "expires": (NOW + timedelta(hours=47)).isoformat(),
    }
    assert row.engine_numbers["recommendation"] == "accept" and row.engine_numbers["acceptance"]["p_accept"] > 0
    assert row.rationale is not None and row.rationale.startswith("Offer in-1: give T3 RB2 to Team 2 for T2 TE1.")
    assert "Recommendation: accept" in row.rationale


def test_auto_never_applies_to_the_answers_it_drafts(store: Store) -> None:
    ctx = context(store)
    offers = open_offers([an_offer_we_like()], US, NOW)
    row = decide_offers(store, CONFIG, ctx.league, now=NOW, offers=offers, weights=EQUAL_WEIGHTS).proposals[0]
    assert row.policy == "approve" and row.deadline is not None
    assert auto_approve_due(store, now=row.deadline - timedelta(minutes=5)) == []  # T-15 sweep: trades never fire
    assert get_proposal(store, row.row_id).status == "proposed"
    for kind in (ProposalKind.TRADE_ACCEPT, ProposalKind.TRADE_DECLINE):
        verdict = evaluate(
            store,
            CONFIG,
            ctx.league,
            kind,
            TradeResponsePayload.model_validate(row.payload),
            max_setting="auto",
            deadline=row.deadline,
            now=NOW,
        )
        assert verdict.setting == "approve"


def test_an_offer_that_hurts_us_is_declined_and_a_counter_is_declined_with_a_note(store: Store) -> None:
    ctx = context(store)
    hurts = offer_record("hurts", proposer=THEM, give=(pid(US, "K"),), get=(pid(THEM, "WR4"),))
    counter = offer_record("loves", proposer=THEM, give=(pid(US, "QB1"),), get=(pid(THEM, "QB2"),), other=THEM)
    decision = decide_offers(
        store, CONFIG, ctx.league, now=NOW, offers=open_offers([hurts, counter], US, NOW), weights=EQUAL_WEIGHTS
    )

    assert [row.kind for row in decision.proposals] == [ProposalKind.TRADE_DECLINE.value] * 2
    by_offer = {TradeResponsePayload.model_validate(row.payload).espn_transaction_id: row for row in decision.proposals}
    assert by_offer["loves"].rationale is not None
    assert "Recommendation: counter" in by_offer["loves"].rationale
    assert "worth a counter-offer made by hand" in by_offer["loves"].rationale
    assert all(row.policy == "approve" for row in decision.proposals)


def test_the_same_offer_is_answered_once_however_many_ticks_see_it(store: Store) -> None:
    ctx = context(store)
    offers = open_offers([an_offer_we_like()], US, NOW)
    first = decide_offers(store, CONFIG, ctx.league, now=NOW, offers=offers, weights=EQUAL_WEIGHTS)
    again = decide_offers(
        store, CONFIG, ctx.league, now=NOW + timedelta(minutes=30), offers=offers, weights=EQUAL_WEIGHTS
    )

    assert again.answers[0].existing and again.proposals == first.proposals
    assert len(store.proposals.find(league_id=ctx.league.row_id, kinds=ANSWER_KINDS)) == 1

    reject(store, first.proposals[0].row_id, decided_by="test", now=NOW)
    after_reject = decide_offers(store, CONFIG, ctx.league, now=NOW, offers=offers, weights=EQUAL_WEIGHTS)
    assert after_reject.answers[0].existing  # a recommendation we rejected is not drafted again
    assert len(store.proposals.find(league_id=ctx.league.row_id, kinds=ANSWER_KINDS)) == 1


def test_a_tick_that_finds_only_answered_offers_loads_no_trade_context(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx = context(store)
    offers = open_offers([an_offer_we_like()], US, NOW)
    first = decide_offers(store, CONFIG, ctx.league, now=NOW, offers=offers, weights=EQUAL_WEIGHTS)

    def no_context(*_: Any, **__: Any) -> TradeContext:
        raise AssertionError("every offer is answered: nothing to evaluate")

    monkeypatch.setattr(offers_module, "load_trade_context", no_context)
    again = decide_offers(store, CONFIG, ctx.league, now=NOW + timedelta(minutes=30), offers=offers)
    assert again.answers[0].existing and again.proposals == first.proposals


def test_an_accept_that_needs_a_drop_is_reported_not_proposed(store: Store) -> None:
    fillers = tuple((US, f"Filler {n}", "WR", BENCH, 7300 + n, 0.5) for n in range(2))  # we hold 16: no room
    ctx = context(store, extra=fillers)
    offer = offer_record(
        "big", proposer=THEM, give=(pid(US, "RB4"),), get=(pid(THEM, "TE1"), pid(THEM, "RB1"))
    )  # two for one into a full roster
    decision = decide_offers(
        store, CONFIG, ctx.league, now=NOW, offers=open_offers([offer], US, NOW), weights=EQUAL_WEIGHTS
    )

    (outcome,) = decision.answers
    assert outcome.proposal is None and outcome.blocked is not None
    assert outcome.blocked.startswith("accepting needs us to drop a player to make room")
    assert decision.blocked == ((outcome.offer, outcome.blocked),) and decision.proposals == ()
    assert store.proposals.find(league_id=ctx.league.row_id, kinds=ANSWER_KINDS) == []


def test_unsupported_stale_and_expired_offers_get_no_proposal(store: Store) -> None:
    ctx = context(store)
    faab = {"type": "ACQUISITION_BUDGET_TRADE", "playerId": 0, "fromTeamId": THEM, "toTeamId": US}
    unsupported = offer_record(
        "faab", proposer=THEM, give=(pid(US, "RB2"),), get=(pid(THEM, "TE1"),), extra_items=(faab,)
    )
    stale = offer_record("stale", proposer=THEM, give=(pid(THEM, "RB1"),), get=(pid(THEM, "TE1"),))  # not ours to give
    expired = an_offer_we_like("old", expires=NOW - timedelta(minutes=1))
    offers = open_offers([unsupported, stale, expired], US, NOW)
    assert [offer.offer_id for offer in offers] == ["faab", "stale"]  # the expired one never gets here

    decision = decide_offers(store, CONFIG, ctx.league, now=NOW, offers=offers, weights=EQUAL_WEIGHTS)
    reasons = {offer.offer_id: reason for offer, reason in decision.blocked}
    assert "ACQUISITION_BUDGET_TRADE items" in reasons["faab"]
    assert "not on our roster" in reasons["stale"] and reasons["stale"].endswith("run fm sync")
    assert decision.proposals == ()


def test_the_trade_deadline_blocks_an_accept_but_never_a_decline(store: Store) -> None:
    late = datetime(2027, 1, 1, tzinfo=UTC)  # after the league's deadline
    ctx = context(store, now=late)
    offers = open_offers([an_offer_we_like(expires=late + timedelta(days=1))], US, late)
    decision = decide_offers(store, CONFIG, ctx.league, now=late, offers=offers, weights=EQUAL_WEIGHTS)
    # the evaluation says the deal is not legal now, so the answer is a decline, which the deadline allows
    assert [row.kind for row in decision.proposals] == [ProposalKind.TRADE_DECLINE.value]
    assert decision.answers[0].evaluation is not None and not decision.answers[0].evaluation.legal

    accept = TradeResponsePayload(
        other_team_id=THEM,
        give_espn_ids=(pid(US, "RB2"),),
        get_espn_ids=(pid(THEM, "TE1"),),
        espn_transaction_id="in-1",
    )
    with pytest.raises(PolicyError, match="trade deadline"):
        propose(store, CONFIG, ctx.league, ProposalKind.TRADE_ACCEPT, accept, created_by="test", now=late)


def test_without_incoming_offers_nothing_is_loaded_and_ours_are_only_listed(store: Store) -> None:
    league = store.leagues.upsert(
        LeagueRow(key="nfl", sport="nfl", espn_league_id=LEAGUE_ID, season=SEASON, team_id=US, as_of=NOW)
    )  # not synced: building a trade context would fail
    ours = offer_record("ours", proposer=US, give=(1,), get=(2,))
    decision = decide_offers(store, CONFIG, league, now=NOW, offers=open_offers([ours], US, NOW))
    assert decision.answers == () and [offer.offer_id for offer in decision.outgoing] == ["ours"]

    waiting = open_offers([an_offer_we_like()], US, NOW)
    with pytest.raises(OffersError, match="no synced settings"):
        decide_offers(store, CONFIG, league, now=NOW, offers=waiting)


def test_without_storing_offers_are_only_evaluated(store: Store) -> None:
    ctx = context(store)
    decision = decide_offers(
        store,
        CONFIG,
        ctx.league,
        now=NOW,
        offers=open_offers([an_offer_we_like()], US, NOW),
        weights=EQUAL_WEIGHTS,
        store_proposals=False,
    )
    assert decision.answers[0].evaluation is not None and decision.proposals == ()
    assert store.proposals.find(league_id=ctx.league.row_id, kinds=ANSWER_KINDS) == []


# --- the tick's decision ----------------------------------------------------------------------------------------------


def test_the_decision_is_registered_for_both_sports_and_reads_through_a_client(store: Store) -> None:
    assert (
        decisions.lookup("nfl", OFFERS_KIND) is decide_offers and decisions.lookup("nba", OFFERS_KIND) is decide_offers
    )
    assert OFFERS_KIND == "offers"

    ctx = context(store)
    api = FakeEspnApi()
    record = an_offer_we_like().model_dump(by_alias=True, mode="json")
    record["proposedDate"] = int((NOW - timedelta(hours=1)).timestamp() * 1000)
    record["expirationDate"] = int((NOW + timedelta(hours=47)).timestamp() * 1000)
    cancelled = an_offer_we_like("gone").model_dump(by_alias=True, mode="json")
    cancelled.update(proposedDate=record["proposedDate"], expirationDate=record["expirationDate"])
    closing = {
        "id": "close-gone",
        "type": "TRADE_PROPOSAL",
        "status": "CANCELED",
        "executionType": "CANCEL",
        "teamId": THEM,
        "relatedTransactionId": "gone",
        "isPending": True,
    }
    api.serve("mPendingTransactions", {"pendingTransactions": []})
    api.serve("mTransactions2", {"transactions": [record, cancelled, closing]})
    client = api.client("ffl", LEAGUE_ID, SEASON)

    decision = decide_offers(store, CONFIG, ctx.league, now=NOW, client=client, weights=EQUAL_WEIGHTS)
    assert [outcome.offer.offer_id for outcome in decision.answers] == ["in-1"]
    assert [row.kind for row in decision.proposals] == [ProposalKind.TRADE_ACCEPT.value]
    approved = approve(store, decision.proposals[0].row_id, decided_by="test", now=NOW)
    assert approved.status == "approved" and approved.execution_token  # a person's approval is what arms it


# --- no write path ----------------------------------------------------------------------------------------------------


def test_the_module_imports_nothing_that_can_write_to_espn() -> None:
    tree = ast.parse(Path(offers_module.__file__ or "").read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
            imported.update(f"{node.module}.{alias.name}" for alias in node.names)
    forbidden = [name for name in imported if name.startswith(("fm.executor", "fm.browser", "playwright"))]
    assert forbidden == []
    assert not any("transaction_request" in name or "WriteRequest" in name for name in imported)


def test_importing_it_loads_no_executor_flow_or_envelope_code() -> None:
    code = (
        "import sys, fm.decide.offers; "
        "bad = sorted(m for m in sys.modules if m.startswith(('fm.executor', 'fm.browser.flows', "
        "'fm.browser.transactions', 'fm.browser.selectors'))); "
        "print(bad)"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert done.stdout.strip() == "[]"


def test_offer_views_are_plain_values() -> None:
    view = OfferView("x", Direction.INCOMING, THEM, (1,), (2,))
    assert view.describe() == "offer x from team 2: give 1; get 2" and view.dedupe_key == "offer:x"
    assert view.players == (1, 2) and view.supported
