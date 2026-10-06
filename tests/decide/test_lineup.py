"""The lineup optimizer and the NFL lineup decision (ROADMAP #20, DESIGN section 9.1).

Three layers:

- The optimizer on hand-built rosters: expected points over slot eligibility (FLEX included), the hard rules (never
  start an OUT, bye or no-game player while a legal alternative exists; never move a locked player), the
  ``bench_inactive`` lineup, and the switch to a win-probability objective when the matchup is lopsided.
- Property tests (hypothesis) on random small rosters, checked against a brute-force enumeration of every legal
  lineup: the lineup is always legal, it starts as many players who play as any legal lineup can (so no zero starter
  while an alternative exists) and then the most expected points, it leaves an optimal lineup alone, and it never
  moves a locked player.
- The store-backed decision on the real NFL league (``tests/fixtures/espn/real/ffl/``): its settings, our week-4
  roster with ESPN's projections (the league's scorer reproduces ESPN's ``appliedTotal`` for every player) and the real
  week-4 pro schedule. The capture is from Monday, when ESPN had locked everyone but Bijan Robinson (Monday night);
  most tests replay the week from Thursday noon ET with ESPN's lock flags cleared, and the Monday test keeps them.
  Proposals go through ``fm.proposals.propose``.

Blend weights are the tests' own (:data:`WEIGHTS`): the committed ``data/blend_weights.toml`` belongs to ROADMAP #39.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from functools import cache
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fm.config import Config, League, Policy
from fm.decide import registry as decide_registry
from fm.decide.lineup import (
    CREATED_BY,
    DECISION_KIND,
    LOPSIDED_AT,
    LineupCandidate,
    LineupDecision,
    LineupError,
    LineupPlan,
    MatchupOutlook,
    Objective,
    active_slot_counts,
    bench_inactive_lineup,
    lineup_inputs,
    optimize_lineup,
    plan_lineup,
    propose_lineup,
    score_lineup,
    team_outlook,
    win_probability,
)
from fm.espn.ids import FFL
from fm.espn.models import ProSchedule, RostersView
from fm.espn.settings import LeagueSettings, LockType, ScoringKind, ScoringType, load_league_settings
from fm.model.availability import assess_stored
from fm.model.projections import BlendWeights
from fm.proposals import LineupMove, LineupPayload, ProposalKind, parse_payload
from fm.sports.base import StatSchema
from fm.sports.nfl import NFL
from fm.store import (
    AvailabilityRow,
    LeagueRow,
    LeagueSettingsRow,
    PlayerRow,
    ProjectionRow,
    RosterEntryRow,
    Store,
    TeamRow,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
REAL_FFL = FIXTURES / "espn" / "real" / "ffl"
SEASON, WEEK = 2026, 4
OUR_TEAM, THEIR_TEAM = 1, 2

QB, RB, WR, TE, OP, DST, K, FLEX = (FFL.slot_id(label) for label in ("QB", "RB", "WR", "TE", "OP", "D/ST", "K", "FLEX"))
BENCH, IR = FFL.bench_slot, FFL.ir_slot

SYNCED = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)  # 8 a.m. ET Thursday: when the replayed sync read the league
THURSDAY = datetime(2026, 10, 1, 16, 0, tzinfo=UTC)  # noon ET, before Thursday night
TNF = datetime(2026, 10, 2, 0, 15, tzinfo=UTC)  # PIT at CLE, 8:15 p.m. ET
SUNDAY_NOON = datetime(2026, 10, 4, 16, 0, tzinfo=UTC)  # after Thursday night and the 9:30 a.m. ET London game
SUNDAY_1PM = datetime(2026, 10, 4, 17, 0, tzinfo=UTC)  # the early Sunday window
SUNDAY_LATE = datetime(2026, 10, 4, 20, 25, tzinfo=UTC)  # 4:25 p.m. ET: LAC at SEA, KC at LV, DEN at SF
MONDAY = datetime(2026, 10, 5, 16, 0, tzinfo=UTC)  # the capture: only ATL at NO, Monday night, is still to start
MNF = datetime(2026, 10, 6, 0, 15, tzinfo=UTC)

# Our real week-4 roster (team 1), by name.
BURROW, BIJAN, CHASE_BROWN, HAMPTON = 3915511, 4430807, 4362238, 4685382
RICE, NABERS, MCLAURIN, BURDEN, METCALF, WILSON = 4428331, 4595348, 3121422, 4685278, 4047650, 4360761
WHITE, PIERCE, KITTLE, KELCE, BROWNS, SANTOS = 4697815, 4360078, 3040151, 15847, -16005, 17427

WEIGHTS = BlendWeights.parse(
    """
    [nfl.default]
    espn = 1.0
    sleeper = 1.0

    [nfl.sd]
    floor = 2.0
    default = 0.5
    """,
    where="test weights",
)


def cand(
    espn_id: int,
    slot_id: int,
    position: str,
    points: float = 10.0,
    *,
    sd: float | None = None,
    p_active: float = 1.0,
    has_game: bool = True,
    locked: bool = False,
    lock_at: datetime | None = None,
    eligible: Sequence[int] | None = None,
    designation: str | None = None,
) -> LineupCandidate:
    """A candidate eligible where the NFL plugin puts his position (``eligible`` overrides it, as ESPN's own
    ``eligibleSlots`` would), with a projection SD of half his points unless given."""
    return LineupCandidate(
        espn_id=espn_id,
        slot_id=slot_id,
        eligible=NFL.eligible_slots(position, include_reserve=False) if eligible is None else frozenset(eligible),
        points=points,
        sd=0.5 * abs(points) if sd is None else sd,
        p_active=p_active,
        has_game=has_game,
        locked=locked,
        lock_at=lock_at,
        position=position,
        name=f"P{espn_id}",
        designation=designation,
    )


def optimize(players: Sequence[LineupCandidate], counts: Mapping[int, int], **options: Any) -> LineupPlan:
    return optimize_lineup(players, counts, sport="nfl", **options)


def moved(plan: LineupPlan) -> dict[int, tuple[int, int]]:
    return {move.espn_id: (move.from_slot_id, move.to_slot_id) for move in plan.moves}


# --- expected points over slot eligibility ----------------------------------------------------------------------------


def test_starts_the_most_expected_points_and_moves_the_fewest_players() -> None:
    counts = {QB: 1, RB: 2, WR: 2, TE: 1, FLEX: 1}
    players = [
        cand(1, QB, "QB", 18),
        cand(2, RB, "RB", 15),
        cand(3, RB, "RB", 8),
        cand(4, WR, "WR", 12),
        cand(5, WR, "WR", 9),
        cand(6, TE, "TE", 7),
        cand(7, FLEX, "WR", 6),
        cand(8, BENCH, "RB", 11),
        cand(9, BENCH, "WR", 10.5),
        cand(10, BENCH, "TE", 6.5),
    ]
    plan = optimize(players, counts)
    assert plan.objective is Objective.EXPECTED_POINTS and plan.win_probability is None
    # Player 5 (9) and player 9 (10.5) share WR and FLEX either way round; the one that keeps player 5 put wins.
    assert moved(plan) == {3: (RB, BENCH), 7: (FLEX, BENCH), 8: (BENCH, RB), 9: (BENCH, FLEX)}
    assert plan.expected == pytest.approx(18 + 15 + 11 + 12 + 9 + 7 + 10.5)
    assert set(plan.starters) == {1, 2, 4, 5, 6, 8, 9}
    assert plan.sd == pytest.approx(math.sqrt(sum((0.5 * points) ** 2 for points in (18, 15, 11, 12, 9, 7, 10.5))))
    assert plan.open_slots == () and plan.idle_starters == ()


def test_benches_an_out_starter_for_the_best_bench_player() -> None:
    players = [
        cand(1, WR, "WR", 14, p_active=0.0, designation="OUT"),
        cand(2, WR, "WR", 9),
        cand(3, BENCH, "WR", 8),
        cand(4, BENCH, "WR", 6),
    ]
    plan = optimize(players, {WR: 2})
    assert moved(plan) == {1: (WR, BENCH), 3: (BENCH, WR)}
    assert plan.expected == pytest.approx(17) and plan.idle_starters == ()


def test_benches_a_starter_without_a_game_whatever_his_projection() -> None:
    plan = optimize([cand(1, RB, "RB", 12, has_game=False), cand(2, BENCH, "RB", 4)], {RB: 1})
    assert moved(plan) == {1: (RB, BENCH), 2: (BENCH, RB)}


def test_shifts_a_starter_between_slots_to_make_room_for_a_replacement() -> None:
    players = [
        cand(1, RB, "RB", 15, p_active=0.0, designation="OUT"),
        cand(2, FLEX, "RB", 12),
        cand(3, BENCH, "WR", 9),
    ]
    plan = optimize(players, {RB: 1, FLEX: 1})
    assert moved(plan) == {1: (RB, BENCH), 2: (FLEX, RB), 3: (BENCH, FLEX)}


def test_a_zero_starter_without_an_alternative_stays_put() -> None:
    players = [
        cand(1, QB, "QB", 20, p_active=0.0, designation="OUT"),
        cand(2, RB, "RB", 10),
        cand(3, BENCH, "RB", 5),  # cannot play QB
        cand(4, BENCH, "QB", 9, has_game=False),  # on bye: no better than the OUT starter
    ]
    plan = optimize(players, {QB: 1, RB: 1})
    assert plan.moves == () and plan.payload() is None
    assert plan.idle_starters == (1,)


def test_fills_an_empty_slot_even_with_a_negative_projection() -> None:
    players = [cand(1, BENCH, "D/ST", -1.5), cand(2, K, "K", 7)]
    counts = {DST: 1, K: 1}
    assert score_lineup(players, counts, sport="nfl").open_slots == (DST,)
    plan = optimize(players, counts)
    assert moved(plan) == {1: (BENCH, DST)} and plan.open_slots == ()
    assert plan.expected == pytest.approx(5.5)


def test_the_better_of_two_negative_defenses_starts() -> None:
    plan = optimize([cand(1, DST, "D/ST", -1.5), cand(2, BENCH, "D/ST", -0.5)], {DST: 1})
    assert moved(plan) == {1: (DST, BENCH), 2: (BENCH, DST)}


def test_a_locked_player_is_never_moved() -> None:
    kickoff = datetime(2026, 10, 4, 17, 0, tzinfo=UTC)
    players = [
        cand(1, QB, "QB", 5, locked=True, lock_at=kickoff),  # a poor starter whose game has begun keeps his slot
        cand(2, BENCH, "QB", 25),  # so the better quarterback cannot come in
        cand(3, WR, "WR", 3),
        cand(4, BENCH, "WR", 30, locked=True, lock_at=kickoff),  # his game started while he sat on the bench
        cand(5, BENCH, "WR", 2),
    ]
    assert optimize(players, {QB: 1, WR: 1}).moves == ()
    locked_out = [cand(1, WR, "WR", 0, p_active=0.0, locked=True), cand(2, BENCH, "WR", 9)]
    plan = optimize(locked_out, {WR: 1})
    assert plan.moves == () and plan.idle_starters == (1,)


def test_a_locked_starter_holds_one_of_his_slots_openings() -> None:
    players = [
        cand(1, RB, "RB", 3, locked=True),
        cand(2, RB, "RB", 5),
        cand(3, BENCH, "RB", 9),
        cand(4, BENCH, "RB", 7),
    ]
    assert moved(optimize(players, {RB: 2})) == {2: (RB, BENCH), 3: (BENCH, RB)}


def test_a_questionable_player_counts_at_his_chance_of_playing() -> None:
    healthy = cand(2, BENCH, "WR", 11)
    likely_out = optimize([cand(1, WR, "WR", 15, p_active=0.7, designation="QUESTIONABLE"), healthy], {WR: 1})
    assert moved(likely_out) == {1: (WR, BENCH), 2: (BENCH, WR)}
    assert optimize([cand(1, WR, "WR", 15, p_active=0.95), healthy], {WR: 1}).moves == ()


def test_equally_good_lineups_keep_the_current_one() -> None:
    players = [cand(1, RB, "RB", 10), cand(2, FLEX, "RB", 10), cand(3, BENCH, "RB", 10)]
    assert optimize(players, {RB: 1, FLEX: 1}).moves == ()


def test_eligibility_comes_from_espns_list_where_given() -> None:
    tight_end_at_wr = cand(2, BENCH, "TE", 11, eligible=(WR, TE, FLEX))  # ESPN lists some tight ends at WR too
    plan = optimize([cand(1, WR, "WR", 6), tight_end_at_wr], {WR: 1})
    assert moved(plan) == {1: (WR, BENCH), 2: (BENCH, WR)}
    no_wr = cand(2, BENCH, "TE", 11, eligible=(TE, FLEX))
    assert optimize([cand(1, WR, "WR", 6), no_wr], {WR: 1}).moves == ()


def test_ir_players_stay_on_ir_and_nobody_is_moved_there() -> None:
    players = [
        cand(1, IR, "RB", 25),
        cand(2, RB, "RB", 0, p_active=0.0, designation="OUT"),
        cand(3, BENCH, "RB", 4),
    ]
    assert moved(optimize(players, {RB: 1})) == {2: (RB, BENCH), 3: (BENCH, RB)}


def test_the_payload_lists_both_sides_of_every_swap() -> None:
    plan = optimize([cand(1, WR, "WR", 5), cand(2, BENCH, "WR", 9)], {WR: 1})
    assert plan.payload() == LineupPayload(
        moves=(
            LineupMove(espn_id=1, from_slot_id=WR, to_slot_id=BENCH),
            LineupMove(espn_id=2, from_slot_id=BENCH, to_slot_id=WR),
        )
    )


@pytest.mark.parametrize(
    ("players", "counts", "options", "match"),
    [
        ([cand(1, WR, "WR"), cand(1, BENCH, "WR")], {WR: 1}, {}, "listed twice"),
        ([cand(1, WR, "WR")], {WR: 1, BENCH: 7}, {}, "active slots only"),
        ([cand(1, WR, "WR")], {WR: -1}, {}, "negative count"),
        ([cand(1, WR, "WR")], {WR: 1}, {"objective": Objective.WIN_PROBABILITY}, "MatchupOutlook"),
        ([cand(1, WR, "WR")], {WR: 1}, {"lopsided_at": 0.5}, "lopsided_at"),
    ],
)
def test_refuses_inputs_it_cannot_plan(
    players: list[LineupCandidate], counts: dict[int, int], options: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        optimize(players, counts, **options)


@pytest.mark.parametrize(
    ("fields", "match"),
    [
        ({"p_active": 1.5}, "p_active"),
        ({"sd": -1.0}, "sd >= 0"),
        ({"points": math.nan}, "finite"),
        ({"lock_at": datetime(2026, 10, 4, 13, 0)}, "aware"),
    ],
)
def test_a_candidate_rejects_impossible_numbers(fields: dict[str, Any], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        LineupCandidate(espn_id=1, slot_id=WR, **fields)


def test_a_candidates_expected_points_and_variance() -> None:
    questionable = cand(1, WR, "WR", 10, sd=4, p_active=0.7)
    assert questionable.expected == pytest.approx(7.0)
    assert questionable.variance == pytest.approx(0.7 * 16 + 0.7 * 0.3 * 100)
    on_bye = cand(2, WR, "WR", 10, sd=4, has_game=False)
    assert (on_bye.plays, on_bye.expected, on_bye.variance) == (False, 0.0, 0.0)


# --- the win-probability objective ------------------------------------------------------------------------------------


def test_win_probability_is_the_normal_approximation() -> None:
    assert win_probability(110, 6, MatchupOutlook(100, 8)) == pytest.approx(0.841345, abs=1e-6)  # Phi(10 / 10)
    assert win_probability(90, 6, MatchupOutlook(100, 8)) == pytest.approx(0.158655, abs=1e-6)
    assert [win_probability(mean, 0, MatchupOutlook(100)) for mean in (101, 100, 99)] == [1.0, 0.5, 0.0]
    with pytest.raises(ValueError, match="sd >= 0"):
        MatchupOutlook(100, -1)


def steady_or_volatile(steady_points: float, volatile_points: float) -> list[LineupCandidate]:
    """A locked quarterback worth 100 +- 10 and one WR slot for a steady receiver (sd 1) or a volatile one (sd 12)."""
    return [
        cand(1, QB, "QB", 100, sd=10, locked=True),
        cand(2, BENCH, "WR", steady_points, sd=1),
        cand(3, BENCH, "WR", volatile_points, sd=12),
    ]


def test_a_heavy_favorite_trades_expected_points_for_a_steadier_lineup() -> None:
    players = steady_or_volatile(12, 12.5)
    outlook = MatchupOutlook(opponent_mean=80, opponent_sd=10)
    by_points = optimize(players, {QB: 1, WR: 1}, outlook=outlook, objective=Objective.EXPECTED_POINTS)
    assert by_points.slots[3] == WR and by_points.win_probability == pytest.approx(0.9601, abs=1e-4)
    plan = optimize(players, {QB: 1, WR: 1}, outlook=outlook)
    assert plan.objective is Objective.WIN_PROBABILITY and plan.variance_weight < 0
    assert (plan.slots[2], plan.slots[3]) == (WR, BENCH)
    assert plan.win_probability == pytest.approx(0.9880, abs=1e-4)
    assert plan.expected == pytest.approx(by_points.expected - 0.5)


def test_a_heavy_underdog_takes_on_variance() -> None:
    players = steady_or_volatile(12.5, 12)
    outlook = MatchupOutlook(opponent_mean=140, opponent_sd=10)
    by_points = optimize(players, {QB: 1, WR: 1}, outlook=outlook, objective=Objective.EXPECTED_POINTS)
    assert by_points.slots[2] == WR and by_points.win_probability == pytest.approx(0.0262, abs=1e-4)
    plan = optimize(players, {QB: 1, WR: 1}, outlook=outlook)
    assert plan.objective is Objective.WIN_PROBABILITY and plan.variance_weight > 0
    assert (plan.slots[2], plan.slots[3]) == (BENCH, WR)
    assert plan.win_probability == pytest.approx(0.0656, abs=1e-4)


def test_a_close_matchup_keeps_expected_points() -> None:
    plan = optimize(steady_or_volatile(12, 12.5), {QB: 1, WR: 1}, outlook=MatchupOutlook(110, 10))
    assert plan.objective is Objective.EXPECTED_POINTS and plan.slots[3] == WR
    assert 1 - LOPSIDED_AT < (plan.win_probability or 0) < LOPSIDED_AT


def test_lopsided_at_sets_where_the_switch_happens() -> None:
    players = steady_or_volatile(12, 12.5)
    outlook = MatchupOutlook(opponent_mean=80, opponent_sd=10)  # the expected-points lineup wins 96% of the time
    assert optimize(players, {QB: 1, WR: 1}, outlook=outlook, lopsided_at=0.97).objective is Objective.EXPECTED_POINTS
    assert optimize(players, {QB: 1, WR: 1}, outlook=outlook, lopsided_at=0.95).objective is Objective.WIN_PROBABILITY


def test_the_win_probability_objective_still_never_starts_a_zero() -> None:
    players = [
        cand(1, QB, "QB", 100, sd=10, locked=True),
        cand(2, BENCH, "WR", 30, sd=30, has_game=False),
        cand(3, BENCH, "WR", 30, sd=30, p_active=0.0, designation="OUT"),
        cand(4, BENCH, "WR", 1, sd=1),
    ]
    plan = optimize(players, {QB: 1, WR: 1}, outlook=MatchupOutlook(160, 10))
    assert plan.objective is Objective.WIN_PROBABILITY
    assert moved(plan) == {4: (BENCH, WR)}


# --- the bench_inactive lineup ----------------------------------------------------------------------------------------


def test_the_bench_inactive_lineup_only_replaces_zero_starters() -> None:
    players = [
        cand(1, WR, "WR", 14, p_active=0.0, designation="OUT"),
        cand(2, WR, "WR", 5),
        cand(3, BENCH, "WR", 12),
        cand(4, BENCH, "WR", 10),
    ]
    assert moved(bench_inactive_lineup(players, {WR: 2}, sport="nfl")) == {1: (WR, BENCH), 3: (BENCH, WR)}
    assert moved(optimize(players, {WR: 2})) == {1: (WR, BENCH), 2: (WR, BENCH), 3: (BENCH, WR), 4: (BENCH, WR)}


def test_the_bench_inactive_lineup_is_empty_without_a_zero_starter() -> None:
    players = [cand(1, WR, "WR", 5), cand(2, BENCH, "WR", 9)]
    assert bench_inactive_lineup(players, {WR: 1}, sport="nfl").moves == ()
    assert optimize(players, {WR: 1}).moves != ()


# --- properties against a brute-force enumeration ---------------------------------------------------------------------

ACTIVE_SLOTS = (QB, RB, WR, TE, OP, DST, K, FLEX)
POSITIONS = ("QB", "RB", "WR", "TE", "D/ST", "K")
MAX_OPENINGS = 5
PROPERTY_SETTINGS = settings(max_examples=200, deadline=None, database=None, derandomize=True)


def is_active(slot_id: int) -> bool:
    return slot_id not in (BENCH, IR)


@st.composite
def rosters(draw: st.DrawFn) -> tuple[tuple[LineupCandidate, ...], dict[int, int]]:
    """Up to seven players and five openings over up to four slot kinds, in a legal current lineup. Points are quarter
    points and ``p_active`` a designation's rate, so distinct lineups never tie by accident; some players are locked,
    on bye, out or on IR, and some carry an extra ESPN eligibility."""
    counts: dict[int, int] = {}
    for slot_id in draw(st.lists(st.sampled_from(ACTIVE_SLOTS), min_size=1, max_size=4, unique=True)):
        room = MAX_OPENINGS - sum(counts.values())
        if room > 0:
            counts[slot_id] = min(room, draw(st.integers(min_value=1, max_value=2)))
    held: Counter[int] = Counter()
    players: list[LineupCandidate] = []
    for espn_id in range(1, draw(st.integers(min_value=1, max_value=7)) + 1):
        position = draw(st.sampled_from(POSITIONS))
        eligible = NFL.eligible_slots(position, include_reserve=False) | draw(
            st.sets(st.sampled_from(ACTIVE_SLOTS), max_size=1)
        )
        room = [slot_id for slot_id in sorted(counts) if slot_id in eligible and held[slot_id] < counts[slot_id]]
        slot_id = draw(st.sampled_from([*room, BENCH, BENCH, IR]))
        held[slot_id] += 1
        players.append(
            LineupCandidate(
                espn_id=espn_id,
                slot_id=slot_id,
                eligible=frozenset(eligible),
                points=draw(st.integers(min_value=-8, max_value=120)) / 4,
                sd=draw(st.sampled_from((0.0, 2.0, 5.0, 10.0))),
                p_active=draw(st.sampled_from((0.0, 0.05, 0.7, 0.95, 1.0, 1.0))),
                has_game=draw(st.sampled_from((True, True, True, False))),
                locked=draw(st.sampled_from((False, False, False, True))),
                position=position,
            )
        )
    return tuple(players), counts


def legal_lineups(players: Sequence[LineupCandidate], counts: Mapping[int, int]) -> Iterator[dict[int, int]]:
    """Every legal lineup: locked and IR players where they are, everyone else on the bench or in a slot of the league
    he is eligible for (or already in), no slot over its count."""
    fixed = {player.espn_id: player.slot_id for player in players if player.locked or player.slot_id == IR}
    held = Counter(fixed.values())
    room = {slot_id: max(0, count - held[slot_id]) for slot_id, count in counts.items()}
    free = [player for player in players if player.espn_id not in fixed]

    def walk(index: int, chosen: dict[int, int]) -> Iterator[dict[int, int]]:
        if index == len(free):
            yield {**fixed, **chosen}
            return
        player = free[index]
        chosen[player.espn_id] = BENCH
        yield from walk(index + 1, chosen)
        for slot_id in sorted((player.eligible | {player.slot_id}) & counts.keys()):
            if room[slot_id] > 0:
                room[slot_id] -= 1
                chosen[player.espn_id] = slot_id
                yield from walk(index + 1, chosen)
                room[slot_id] += 1
        del chosen[player.espn_id]

    yield from walk(0, {})


def filled(players: Sequence[LineupCandidate], slots: Mapping[int, int]) -> int:
    """Active slots holding a player who plays."""
    return sum(1 for player in players if is_active(slots[player.espn_id]) and player.plays)


def expected(players: Sequence[LineupCandidate], slots: Mapping[int, int]) -> float:
    return math.fsum(player.expected for player in players if is_active(slots[player.espn_id]))


def current(players: Sequence[LineupCandidate]) -> dict[int, int]:
    return {player.espn_id: player.slot_id for player in players}


def best_possible(players: Sequence[LineupCandidate], counts: Mapping[int, int]) -> tuple[int, float]:
    """The most slots any legal lineup fills with players who play, and the most expected points among those."""
    scored = [(filled(players, slots), expected(players, slots)) for slots in legal_lineups(players, counts)]
    most = max(count for count, _ in scored)
    return most, max(points for count, points in scored if count == most)


def assert_legal(players: Sequence[LineupCandidate], counts: Mapping[int, int], plan: LineupPlan) -> None:
    """Checked independently of the module: every player placed, locked and IR players unmoved, nobody moved onto IR
    or into a slot he cannot fill, no slot over its count, and moves that take the current lineup to the plan."""
    assert set(plan.slots) == {player.espn_id for player in players}
    for player in players:
        slot_id = plan.slots[player.espn_id]
        if player.locked or player.slot_id == IR:
            assert slot_id == player.slot_id, f"{player} was moved"
        if slot_id != player.slot_id and slot_id != BENCH:
            assert slot_id in player.eligible and slot_id in counts, f"{player} moved into slot {slot_id}"
    held = Counter(plan.slots.values())
    assert all(held[slot_id] <= count for slot_id, count in counts.items())
    assert all(slot_id in counts for slot_id in held if is_active(slot_id))
    changes = {
        player.espn_id: (player.slot_id, plan.slots[player.espn_id])
        for player in players
        if plan.slots[player.espn_id] != player.slot_id
    }
    assert moved(plan) == changes and len(plan.moves) == len(changes)
    assert plan.payload() == (LineupPayload(moves=plan.moves) if changes else None)


def assert_no_avoidable_zero(players: Sequence[LineupCandidate], counts: Mapping[int, int], plan: LineupPlan) -> None:
    """No unlocked starter who will not play, and no empty opening, that an unlocked bench player who plays could
    take directly."""
    bench = [
        player
        for player in players
        if plan.slots[player.espn_id] == BENCH and player.plays and not player.locked and player.slot_id != IR
    ]
    for player in players:
        slot_id = plan.slots[player.espn_id]
        if is_active(slot_id) and not player.plays and not player.locked:
            assert not any(slot_id in other.eligible for other in bench), f"{player} starts though a player can"
    held = Counter(plan.slots.values())
    for slot_id, count in counts.items():
        if held[slot_id] < count:
            assert not any(slot_id in other.eligible for other in bench), f"slot {slot_id} left empty"


@PROPERTY_SETTINGS
@given(rosters())
def test_the_lineup_is_always_legal_and_the_best_there_is(case: tuple[tuple[LineupCandidate, ...], dict[int, int]]):
    players, counts = case
    plan = optimize(players, counts)
    assert_legal(players, counts, plan)
    most, points = best_possible(players, counts)
    assert filled(players, plan.slots) == most
    assert expected(players, plan.slots) == pytest.approx(points, abs=1e-9)
    assert plan.expected == pytest.approx(points, abs=1e-9)
    # A lineup that is already the best is left alone, and the bench never grows.
    before = current(players)
    if filled(players, before) == most and expected(players, before) >= points - 1e-9:
        assert plan.moves == ()
    assert sum(map(is_active, plan.slots.values())) >= sum(map(is_active, before.values()))


@PROPERTY_SETTINGS
@given(rosters())
def test_no_out_bye_or_no_game_starter_while_a_legal_alternative_exists(
    case: tuple[tuple[LineupCandidate, ...], dict[int, int]],
):
    players, counts = case
    most, _ = best_possible(players, counts)
    for plan in (optimize(players, counts), bench_inactive_lineup(players, counts, sport="nfl")):
        # No legal lineup starts more players who play, so every remaining zero starter (or empty slot) is forced.
        assert filled(players, plan.slots) == most
        assert_no_avoidable_zero(players, counts, plan)
        assert set(plan.idle_starters) == {
            player.espn_id for player in players if is_active(plan.slots[player.espn_id]) and not player.plays
        }


@PROPERTY_SETTINGS
@given(rosters(), st.integers(min_value=-40, max_value=40), st.sampled_from((0.0, 5.0, 15.0)))
def test_locked_players_are_never_moved(
    case: tuple[tuple[LineupCandidate, ...], dict[int, int]], margin: int, opponent_sd: float
):
    players, counts = case
    outlook = MatchupOutlook(opponent_mean=optimize(players, counts).expected - margin, opponent_sd=opponent_sd)
    plans = (
        optimize(players, counts),
        optimize(players, counts, outlook=outlook),
        optimize(players, counts, outlook=outlook, objective=Objective.WIN_PROBABILITY),
        bench_inactive_lineup(players, counts, sport="nfl", outlook=outlook),
    )
    for plan in plans:
        assert_legal(players, counts, plan)
        for player in players:
            if player.locked:
                assert plan.slots[player.espn_id] == player.slot_id
                assert player.espn_id not in moved(plan)


@PROPERTY_SETTINGS
@given(rosters())
def test_the_bench_inactive_lineup_keeps_every_starter_who_plays(
    case: tuple[tuple[LineupCandidate, ...], dict[int, int]],
):
    players, counts = case
    plan = bench_inactive_lineup(players, counts, sport="nfl")
    assert_legal(players, counts, plan)
    for player in players:
        if is_active(player.slot_id) and player.plays and not player.locked and player.slot_id in counts:
            assert is_active(plan.slots[player.espn_id]), f"{player} was benched"
    most, _ = best_possible(players, counts)
    assert (plan.moves == ()) == (filled(players, current(players)) == most)


@PROPERTY_SETTINGS
@given(rosters(), st.integers(min_value=-40, max_value=40), st.sampled_from((0.0, 5.0, 15.0)))
def test_the_win_probability_objective_is_legal_and_never_worse(
    case: tuple[tuple[LineupCandidate, ...], dict[int, int]], margin: int, opponent_sd: float
):
    players, counts = case
    by_points = optimize(players, counts)
    outlook = MatchupOutlook(opponent_mean=by_points.expected - margin, opponent_sd=opponent_sd)
    plan = optimize(players, counts, outlook=outlook, objective=Objective.WIN_PROBABILITY)
    assert_legal(players, counts, plan)
    assert_no_avoidable_zero(players, counts, plan)
    assert filled(players, plan.slots) == filled(players, by_points.slots)
    assert plan.win_probability is not None
    assert plan.win_probability >= win_probability(by_points.expected, by_points.sd, outlook) - 1e-12


# --- the store-backed decision on the real league ---------------------------------------------------------------------


@cache
def real_settings() -> LeagueSettings:
    return load_league_settings(REAL_FFL / "mSettings.json")


@cache
def real_rosters() -> RostersView:
    return RostersView.model_validate(json.loads((REAL_FFL / "mRoster.json").read_text(encoding="utf-8")))


@cache
def real_schedule() -> ProSchedule:
    return ProSchedule.model_validate(json.loads((REAL_FFL / "proTeamSchedules_wl.json").read_text(encoding="utf-8")))


def on_bye(schedule: ProSchedule, pro_team_id: int, period: int) -> ProSchedule:
    """The schedule with the team's game in the period gone from every team's list (ESPN files a game under both
    teams), so the team and its opponent are idle that period, as on a bye week."""
    gone = {game.id for game in schedule.games_for(pro_team_id, period)}
    teams = tuple(
        team.model_copy(
            update={
                "pro_games_by_scoring_period": {
                    week: tuple(game for game in games if game.id not in gone)
                    for week, games in team.pro_games_by_scoring_period.items()
                }
            }
        )
        for team in schedule.pro_teams
    )
    return schedule.model_copy(update={"pro_teams": teams})


def applied_totals(team_id: int) -> dict[int, float]:
    """ESPN's own week-4 points (``appliedTotal``) for each player of a team."""
    totals: dict[int, float] = {}
    for entry in real_rosters().roster(team_id).entries:
        line = entry.player.projection(SEASON, WEEK, game="ffl")
        assert line is not None and line.applied_total is not None
        totals[entry.player_id] = line.applied_total
    return totals


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        yield opened


def seed(store: Store, *, espn_locks: bool = False) -> LeagueRow:
    """Store the real league the way ``fm sync`` does: settings, teams, both rosters for week 4, the players and
    ESPN's week-4 projections. ``espn_locks`` keeps ESPN's ``lineupLocked`` flags from the Monday capture."""
    synced = real_settings()
    league = store.leagues.upsert(
        LeagueRow(
            key="nfl",
            sport="nfl",
            espn_league_id=synced.league_id,
            season=SEASON,
            team_id=OUR_TEAM,
            as_of=SYNCED,
        )
    )
    store.settings.upsert(
        LeagueSettingsRow(league_id=league.row_id, settings=synced.model_dump(mode="json"), as_of=SYNCED)
    )
    schema = StatSchema.for_game("ffl")
    for roster in real_rosters().teams:
        store.teams.upsert(
            TeamRow(league_id=league.row_id, team_id=roster.team_id, name=f"Team {roster.team_id}", as_of=SYNCED)
        )
        entries = [
            RosterEntryRow(
                league_id=league.row_id,
                scoring_period_id=WEEK,
                team_id=roster.team_id,
                espn_id=entry.player_id,
                lineup_slot_id=entry.lineup_slot_id,
                lineup_locked=espn_locks and entry.lineup_locked,
                as_of=SYNCED,
            )
            for entry in roster.entries
        ]
        store.rosters.replace(league.row_id, WEEK, roster.team_id, entries)
        for entry in roster.entries:
            player = entry.player
            assert player.default_position_id is not None and player.pro_team_id is not None
            store.players.upsert(
                PlayerRow(
                    sport="nfl",
                    espn_id=player.id,
                    full_name=player.full_name,
                    default_position_id=player.default_position_id,
                    position=FFL.position_label(player.default_position_id),
                    pro_team_id=player.pro_team_id,
                    pro_team=FFL.pro_team(player.pro_team_id),
                    eligible_slot_ids=list(player.eligible_slots),
                    injury_status=player.injury_status,
                    injured=player.injured,
                    active=player.active,
                    as_of=SYNCED,
                )
            )
            line = player.projection(SEASON, WEEK, game="ffl")
            assert line is not None
            store.projections.upsert(
                ProjectionRow(
                    sport="nfl",
                    espn_id=player.id,
                    source="espn",
                    season=SEASON,
                    scoring_period_id=WEEK,
                    stats=schema.from_espn(line.stats),
                    as_of=SYNCED,
                )
            )
    return league


def designate(store: Store, espn_id: int, status: str) -> None:
    row = store.players.get("nfl", espn_id)
    assert row is not None
    store.players.upsert(row.model_copy(update={"injury_status": status}))


def scale_projection(store: Store, espn_id: int, factor: float) -> None:
    row = store.projections.get("nfl", espn_id, "espn", SEASON, WEEK)
    assert row is not None
    store.projections.upsert(
        row.model_copy(update={"stats": {stat: value * factor for stat, value in row.stats.items()}})
    )


def plan_week(
    store: Store, league: LeagueRow, *, now: datetime = THURSDAY, schedule: ProSchedule | None = None, **options: Any
) -> LineupDecision:
    on = real_schedule() if schedule is None else schedule
    return plan_lineup(store, league, schedule=on, now=now, weights=WEIGHTS, **options)


def draft_moves(decision: LineupDecision, index: int) -> dict[int, tuple[int, int]]:
    return {move.espn_id: (move.from_slot_id, move.to_slot_id) for move in decision.drafts[index].payload.moves}


def split_wr_and_flex(moves: Mapping[int, tuple[int, int]]) -> bool:
    """Wilson and Burden come off the bench into one WR and the FLEX. Either way round is the same lineup in points
    and in moves, so the tests do not pin which."""
    return moves.keys() == {WILSON, BURDEN} and sorted(moves.values()) == sorted([(BENCH, WR), (BENCH, FLEX)])


def config_for(**policy: Any) -> Config:
    league = League(
        key="nfl",
        sport="nfl",
        espn_league_id=real_settings().league_id,
        season=SEASON,
        team_id=OUR_TEAM,
        policy=Policy(**policy),
    )
    return Config(leagues=(league,))


def test_the_real_roster_reads_with_espns_points_designations_and_kickoffs(store: Store) -> None:
    league = seed(store)
    inputs = lineup_inputs(
        store, league, real_settings(), schedule=real_schedule(), period=WEEK, now=THURSDAY, weights=WEIGHTS
    )
    assert inputs.warnings == () and (inputs.team_id, inputs.season, inputs.scoring_period_id) == (1, SEASON, WEEK)
    real_slots = {QB: 1, RB: 2, WR: 2, TE: 1, DST: 1, K: 1, FLEX: 1}  # QB, 2 RB, 2 WR, TE, D/ST, K, FLEX
    assert dict(inputs.slot_counts) == active_slot_counts(real_settings()) == real_slots
    by_id = {player.espn_id: player for player in inputs.players}
    totals = applied_totals(OUR_TEAM)
    assert by_id.keys() == totals.keys()
    for espn_id, total in totals.items():
        # The league's own scoring from ESPN's stat lines gives ESPN's appliedTotal (sent to 8 decimals).
        assert by_id[espn_id].points == pytest.approx(total, abs=1e-6)
        assert by_id[espn_id].sd == pytest.approx(max(2.0, 0.5 * by_id[espn_id].points))
    assert (by_id[RICE].designation, by_id[RICE].p_active) == ("QUESTIONABLE", 0.7)
    assert {espn_id for espn_id, player in by_id.items() if not player.plays} == {MCLAURIN, WHITE, PIERCE}
    assert by_id[BIJAN].eligible == {RB, FLEX} and by_id[KELCE].eligible == {TE, FLEX}
    assert by_id[BROWNS].eligible == {DST} and by_id[SANTOS].eligible == {K} and by_id[BURROW].eligible == {QB}
    assert not any(player.locked for player in inputs.players)
    locks = [by_id[espn_id].lock_at for espn_id in (METCALF, BROWNS, WILSON, CHASE_BROWN, HAMPTON, BIJAN)]
    assert locks == [TNF, TNF, SUNDAY_1PM, SUNDAY_1PM, SUNDAY_LATE, MNF]


def test_thursday_the_plan_starts_wilson_and_burden_over_rice_and_kelce(store: Store) -> None:
    league = seed(store)
    decision = plan_week(store, league)
    assert decision.warnings == () and decision.outlook is None
    assert decision.rescue.moves == () and decision.current.idle_starters == ()
    (draft,) = decision.drafts
    assert draft.kind is ProposalKind.LINEUP and draft.scoring_period_id == WEEK
    moves = draft_moves(decision, 0)
    assert (moves.pop(RICE), moves.pop(KELCE)) == ((WR, BENCH), (FLEX, BENCH))  # Rice is questionable: 70% of 14.9
    assert split_wr_and_flex(moves)
    assert draft.deadline == SUNDAY_1PM  # Wilson and Burden kick off first
    totals = applied_totals(OUR_TEAM)
    gain = totals[WILSON] + totals[BURDEN] - 0.7 * totals[RICE] - totals[KELCE]
    assert draft.engine_numbers["gain"] == pytest.approx(gain)
    assert draft.engine_numbers["objective"] == "expected_points"
    assert draft.dedupe_key.startswith("lineup:2026:4:")
    assert draft.rationale == (
        "Start Michael Wilson, Luther Burden III; bench Travis Kelce, Rashee Rice: 123.9 expected points (+3.6)."
    )
    stored = json.loads(json.dumps(draft.engine_numbers))  # storable as JSON as it is
    assert [(move["name"], move["from"], move["to"]) for move in stored["moves"][:2]] == [
        ("Travis Kelce", "RB/WR/TE", "BE"),  # leaving the lineup first, by ESPN id; FLEX is ESPN's RB/WR/TE slot
        ("Rashee Rice", "WR", "BE"),
    ]
    assert stored["moves"][1]["p_active"] == 0.7 and stored["moves"][1]["designation"] == "QUESTIONABLE"


def test_the_monday_capture_moves_nobody(store: Store) -> None:
    league = seed(store, espn_locks=True)
    decision = plan_week(store, league, now=MONDAY)
    assert [player.espn_id for player in decision.inputs.players if not player.locked] == [BIJAN]
    assert decision.best.moves == () and decision.drafts == ()


def test_players_whose_game_has_started_stay_where_they_are(store: Store) -> None:
    league = seed(store)
    scale_projection(store, METCALF, 3.0)  # a big Thursday night projected for Metcalf
    assert METCALF in plan_week(store, league).best.starters
    sunday = plan_week(store, league, now=SUNDAY_NOON)
    locked = {player.espn_id for player in sunday.inputs.players if player.locked}
    assert locked == {METCALF, BROWNS, MCLAURIN, WHITE, PIERCE}  # Thursday night and the London game
    assert sunday.best.slots[METCALF] == BENCH
    assert all(move.espn_id not in locked for draft in sunday.drafts for move in draft.payload.moves)


def test_an_out_starter_yields_bench_inactive_and_lineup_alternatives(store: Store) -> None:
    league = seed(store)
    designate(store, KITTLE, "OUT")
    decision = plan_week(store, league)
    assert decision.current.idle_starters == (KITTLE,)
    rescue, best = decision.drafts
    assert (rescue.kind, best.kind) == (ProposalKind.BENCH_INACTIVE, ProposalKind.LINEUP)
    # Kelce is the only other tight end: he moves over, and the best bench receiver takes his FLEX spot.
    assert draft_moves(decision, 0) == {KITTLE: (TE, BENCH), KELCE: (FLEX, TE), WILSON: (BENCH, FLEX)}
    # The full lineup also starts Burden over Rice (questionable); he and Wilson split WR and FLEX either way round.
    moves = draft_moves(decision, 1)
    assert [moves.pop(espn_id) for espn_id in (KITTLE, KELCE, RICE)] == [(TE, BENCH), (FLEX, TE), (WR, BENCH)]
    assert split_wr_and_flex(moves)
    assert rescue.deadline == best.deadline == SUNDAY_1PM
    assert rescue.engine_numbers["basis"] == best.engine_numbers["basis"]
    assert rescue.dedupe_key != best.dedupe_key and rescue.dedupe_key.startswith("bench_inactive:2026:4:")
    assert rescue.rationale.startswith("Bench George Kittle (OUT); start Michael Wilson:")
    assert decision.rescue.idle_starters == decision.best.idle_starters == ()


def test_a_fresh_availability_row_outranks_the_designation_model(store: Store) -> None:
    league = seed(store)

    def nabers(at: datetime, p_active: float) -> LineupCandidate:
        store.availability.upsert(
            AvailabilityRow(
                sport="nfl", espn_id=NABERS, season=SEASON, scoring_period_id=WEEK, p_active=p_active, as_of=at
            )
        )
        decision = plan_week(store, league)
        return next(player for player in decision.inputs.players if player.espn_id == NABERS)

    assert nabers(SYNCED + timedelta(hours=1), 0.2).p_active == 0.2  # assessed after the sync: the full model's
    assert nabers(SYNCED - timedelta(hours=1), 0.2).p_active == 1.0  # older than the designation: assessed again
    designate(store, NABERS, "OUT")
    assert nabers(SYNCED + timedelta(hours=1), 0.6).p_active == 0.0  # OUT is a zero whatever a row says


@pytest.mark.parametrize("written_by", ["assess_stored without a schedule", "a hand-built row"])
def test_a_stored_row_never_starts_a_starter_whose_team_has_no_game(store: Store, written_by: str) -> None:
    """Whether a player has a game comes from the schedule the lineup is planned on, never from his availability row:
    a row assessed without a schedule (ROADMAP #29's ``assess_stored(..., practice=...)``) says everyone with a pro
    team has a game, and the row outranks the designation model once it is newer than the sync."""
    league = seed(store)
    giants = store.players.get("nfl", NABERS)
    assert giants is not None and giants.pro_team_id is not None
    bye = on_bye(real_schedule(), giants.pro_team_id, WEEK)  # NYG at ARI is gone: Nabers and Wilson are both idle
    assert set(bye.idle_teams(WEEK)) == {giants.pro_team_id, 22}
    after_sync = SYNCED + timedelta(hours=1)
    if written_by == "assess_stored without a schedule":
        (row,) = assess_stored(store, "nfl", [NABERS], season=SEASON, scoring_period=WEEK, as_of=after_sync)
        assert row.inputs["schedule"] is False
    else:
        row = store.availability.upsert(
            AvailabilityRow(
                sport="nfl", espn_id=NABERS, season=SEASON, scoring_period_id=WEEK, p_active=1.0, as_of=after_sync
            )
        )
    assert row.has_game and row.p_active == 1.0  # the row says he plays

    decision = plan_week(store, league, schedule=bye)
    by_id = {player.espn_id: player for player in decision.inputs.players}
    nabers = by_id[NABERS]
    assert (nabers.has_game, nabers.p_active, nabers.plays, nabers.lock_at) == (False, 0.0, False, None)
    assert not by_id[WILSON].has_game  # the opponent's receiver sits idle on the bench, so he is no alternative
    assert decision.current.idle_starters == (NABERS,)
    assert decision.rescue.slots[NABERS] == BENCH and decision.best.slots[NABERS] == BENCH
    assert decision.rescue.idle_starters == decision.best.idle_starters == ()
    (rescue,) = decision.drafts  # with Wilson idle the full optimum goes no further than the rescue
    assert rescue.kind is ProposalKind.BENCH_INACTIVE
    assert draft_moves(decision, 0) == {NABERS: (WR, BENCH), BURDEN: (BENCH, WR)}
    assert rescue.rationale.startswith("Bench Malik Nabers (no game); start Luther Burden III:")
    assert rescue.deadline == SUNDAY_1PM  # Burden's kickoff: a player without a game never locks
    benched = rescue.engine_numbers["moves"][0]
    assert benched["name"] == "Malik Nabers"
    assert (benched["p_active"], benched["has_game"], benched["expected"]) == (0.0, False, 0.0)


def test_refuses_what_it_cannot_plan(store: Store) -> None:
    league = seed(store)
    with pytest.raises(LineupError, match="no roster for team 1 .* scoring period 5; run fm sync"):
        plan_week(store, league, period=5)
    unknown = real_settings().model_copy(update={"lineup_lock_type": LockType.UNKNOWN})
    with pytest.raises(LineupError, match="UNKNOWN"):
        plan_week(store, league, settings=unknown)
    categories = real_settings().model_copy(
        update={"scoring_kind": ScoringKind.CATEGORIES, "scoring_type": ScoringType.H2H_MOST_CATEGORIES}
    )
    with pytest.raises(LineupError, match="categories"):
        plan_week(store, league, settings=categories)
    with pytest.raises(ValueError, match="aware"):
        plan_week(store, league, now=THURSDAY.replace(tzinfo=None))
    unsynced = store.leagues.upsert(league.model_copy(update={"key": "other", "espn_league_id": 99}))
    with pytest.raises(LineupError, match="no synced settings; run fm sync"):
        plan_week(store, unsynced)


def test_the_opponents_roster_gives_the_outlook_and_a_lopsided_matchup_switches(store: Store) -> None:
    league = seed(store)
    decision = plan_week(store, league, opponent_team_id=THEIR_TEAM)
    theirs = applied_totals(THEIR_TEAM)  # the fixture keeps three of their players: Allen, Henry, Kyren Williams
    assert decision.outlook is not None and decision.outlook.opponent_mean == pytest.approx(sum(theirs.values()))
    assert decision.outlook.opponent_sd == pytest.approx(
        math.sqrt(sum((0.5 * points) ** 2 for points in theirs.values()))
    )
    assert decision.best.objective is Objective.WIN_PROBABILITY
    by_points = optimize(
        decision.inputs.players,
        decision.inputs.slot_counts,
        outlook=decision.outlook,
        objective=Objective.EXPECTED_POINTS,
    )
    assert by_points.win_probability is not None and decision.best.win_probability is not None
    assert decision.best.win_probability >= by_points.win_probability
    assert [draft.kind for draft in decision.drafts] == [ProposalKind.LINEUP]
    assert decision.drafts[0].engine_numbers["objective"] == "win_probability"
    assert "lopsided matchup" in decision.drafts[0].rationale
    opponent = lineup_inputs(
        store,
        league,
        real_settings(),
        schedule=real_schedule(),
        period=WEEK,
        now=THURSDAY,
        team_id=THEIR_TEAM,
        weights=WEIGHTS,
    )
    assert team_outlook(opponent, sport="nfl") == decision.outlook


def test_an_opponent_without_a_stored_roster_leaves_expected_points(store: Store) -> None:
    league = seed(store)
    decision = plan_week(store, league, opponent_team_id=7)
    assert decision.outlook is None and decision.best.objective is Objective.EXPECTED_POINTS
    assert decision.warnings == (
        "opponent: no roster for team 7 of league 'nfl' in scoring period 4; run fm sync; planning for expected points",
    )


def test_propose_lineup_stores_each_draft_under_its_policy_once(store: Store) -> None:
    league = seed(store)
    designate(store, KITTLE, "OUT")
    result = propose_lineup(store, config_for(), league, schedule=real_schedule(), now=THURSDAY, weights=WEIGHTS)
    assert result.blocked == ()
    rescue, best = result.proposals
    assert (rescue.kind, rescue.policy, rescue.status) == ("bench_inactive", "auto", "proposed")
    assert (best.kind, best.policy, best.status) == ("lineup", "approve", "proposed")
    assert {rescue.created_by, best.created_by} == {CREATED_BY}
    assert (rescue.scoring_period_id, rescue.deadline) == (WEEK, SUNDAY_1PM)
    assert parse_payload(rescue) == result.decision.drafts[0].payload
    assert parse_payload(best) == result.decision.drafts[1].payload
    assert rescue.engine_numbers["moves"][0]["name"] == "George Kittle"  # the player leaving the lineup comes first
    assert rescue.rationale == result.decision.drafts[0].rationale
    again = propose_lineup(store, config_for(), league, schedule=real_schedule(), now=THURSDAY, weights=WEIGHTS)
    assert [row.id for row in again.proposals] == [rescue.id, best.id]  # the next tick finds them, no duplicates
    assert len(store.proposals.find(league_id=league.row_id)) == 2


def test_a_draft_the_policy_refuses_is_reported_not_raised(store: Store) -> None:
    league = seed(store)
    designate(store, KITTLE, "OUT")
    result = propose_lineup(
        store, config_for(lineup="off"), league, schedule=real_schedule(), now=THURSDAY, weights=WEIGHTS
    )
    assert [row.kind for row in result.proposals] == ["bench_inactive"]
    (blocked,) = result.blocked
    assert "lineup is off for league 'nfl'" in blocked


def test_registered_as_the_nfl_lineup_decision() -> None:
    assert DECISION_KIND == "lineup"
    assert decide_registry.lookup("nfl", DECISION_KIND) is propose_lineup
    assert decide_registry.lookup("ffl", "lineup") is propose_lineup
    registration = decide_registry.get("nfl", "lineup")
    assert registration is not None and registration.target == "fm.decide.lineup.propose_lineup"
