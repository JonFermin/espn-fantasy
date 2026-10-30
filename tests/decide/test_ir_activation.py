"""``ir_activation``: a player stranded in an IR slot by a status that no longer allows it is moved off IR, or room is
made for him with the cheapest drop, and adds wait meanwhile. Built on the NFL waiver tests' hand-made league, whose
"Hurt Receiver" sits in the IR slot."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import timedelta
from pathlib import Path

import pytest
from tests.decide.test_waivers import (
    BENCH,
    DUD,
    GONE,
    HURT,
    IR,
    MYSTERY,
    NOW,
    ROSTER,
    WEEK,
    decide,
    fixture_config,
    schedule,
    seed,
)

from fm.decide.ir_activation import CREATED_BY, DECISION_KIND, propose_ir_activation, stuck_on_ir
from fm.decide.registry import registry
from fm.jobs.deadlines import ACQUISITION_DECISIONS, LINEUP_DECISIONS
from fm.proposals import AddDropPayload, LineupMove, LineupPayload, ProposalKind
from fm.proposals.policy import acquisitions_this_week, parse_payload, stored_settings
from fm.store import LeagueRow, Store


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        yield opened


def designate(store: Store, espn_id: int, status: str | None) -> None:
    player = store.players.get("nfl", espn_id)
    assert player is not None
    store.players.upsert(player.model_copy(update={"injury_status": status}))


def run(store: Store, league: LeagueRow, **policy: object):  # noqa: ANN201
    return propose_ir_activation(store, fixture_config(**policy), league, schedule=schedule(), now=NOW)


def test_it_is_registered_for_both_sports_and_runs_in_the_lock_windows() -> None:
    assert registry.lookup("nfl", DECISION_KIND) is not None and registry.lookup("nba", DECISION_KIND) is not None
    assert DECISION_KIND in LINEUP_DECISIONS and DECISION_KIND in ACQUISITION_DECISIONS


def test_a_player_still_out_or_with_no_known_status_is_left_on_ir(store: Store) -> None:
    league = seed(store)
    settings = stored_settings(store, league)
    assert settings is not None
    designate(store, HURT, "OUT")
    assert stuck_on_ir(store, league, settings) == () and run(store, league).proposals == ()
    designate(store, HURT, None)  # a gap in the data is no reason to drop anyone
    assert stuck_on_ir(store, league, settings) == () and run(store, league).proposals == ()


def test_with_room_he_is_moved_to_the_bench_on_approval_only(store: Store) -> None:
    league = seed(store, roster=ROSTER[:-1])  # a roster spot is open
    designate(store, HURT, "ACTIVE")
    result = run(store, league, lineup="auto")  # the league's lineups are auto; this one still waits
    (row,) = result.proposals
    assert (row.kind, row.policy, row.created_by, row.scoring_period_id) == ("lineup", "approve", CREATED_BY, None)
    assert row.deadline is not None  # his own kickoff: the move is void once he locks
    assert parse_payload(row) == LineupPayload(moves=(LineupMove(espn_id=HURT, from_slot_id=IR, to_slot_id=BENCH),))
    assert "Activate Hurt Receiver from IR" in (row.rationale or "")
    assert [p.row_id for p in run(store, league).proposals] == [row.row_id]  # the next run hands it back


def test_with_a_full_roster_the_cheapest_drop_makes_room_first(store: Store) -> None:
    league = seed(store)
    designate(store, HURT, "ACTIVE")
    result = run(store, league)
    (row,) = result.proposals
    assert (row.kind, row.policy, row.created_by) == (ProposalKind.ADD_DROP.value, "approve", CREATED_BY)
    payload = parse_payload(row)
    assert isinstance(payload, AddDropPayload) and payload.add_espn_id is None
    assert payload.drop_espn_id == GONE  # out for the season: known to be worth nothing
    assert payload.drop_espn_id not in (HURT, MYSTERY)  # never the IR player, never an unknown value
    assert row.engine_numbers["activate"]["espn_id"] == HURT and row.engine_numbers["value_lost"] == pytest.approx(0)
    assert row.deadline is not None  # the dropped player's roster lock
    assert "Drop Gone Back to make room for Hurt Receiver" in (row.rationale or "")
    settings = stored_settings(store, league)
    assert (
        acquisitions_this_week(store, league, settings, scoring_period_id=WEEK, now=NOW) == 0
    )  # a bare drop adds no one


def test_untouchables_are_never_the_drop(store: Store) -> None:
    league = seed(store)
    designate(store, HURT, "ACTIVE")
    (row,) = run(store, league, untouchables=[GONE]).proposals
    payload = parse_payload(row)
    assert isinstance(payload, AddDropPayload) and payload.drop_espn_id not in (GONE, HURT, MYSTERY)
    assert payload.drop_espn_id == DUD  # the next cheapest: two points a game on the bench


def test_waivers_propose_no_add_while_a_player_is_stranded_on_ir(store: Store) -> None:
    seed(store, roster=ROSTER[:-1])
    designate(store, HURT, "ACTIVE")
    decision = decide(store, fixture_config(), schedule=schedule())
    assert decision.moves and decision.proposals == ()
    assert all("Hurt Receiver (ACTIVE) in an IR slot" in why for _, why in decision.blocked)


def test_a_player_whose_drop_has_closed_is_passed_over(store: Store) -> None:
    league = seed(store)
    designate(store, HURT, "ACTIVE")
    started = schedule({(29, WEEK): NOW - timedelta(hours=1)})  # Gone Back's game kicked off: ESPN takes no drop now
    result = propose_ir_activation(store, fixture_config(), league, schedule=started, now=NOW)
    (row,) = result.proposals
    payload = parse_payload(row)
    assert isinstance(payload, AddDropPayload) and payload.drop_espn_id == DUD
    assert "the free agents who would fill in for him are worth as much" in (row.rationale or "")
