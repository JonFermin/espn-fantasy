"""Trade evaluator and finder (ROADMAP #38): Δ rest-of-season value and Δ title odds for both sides, legality from the
league's settings, P(accept) from market values and the other manager's needs, the ranked finder, the proposals it
drafts, and ``fm trade eval|find``.

The NFL league is hand-made so every number is visible: six teams in the hand-built PPR league
(``ffl_settings_ppr.json`` cut to 6 teams and a 4-team playoff), 14 players each, every player projected a flat
number of points a game (a line of receiving yards, 0.1 a yard). We are team 3, the middle of the pack: deep at
running back, with a hole at tight end, and team 2 is the mirror image, so a running back for a tight end helps both
of us. The NBA leagues are built from
``fba_settings_points.json`` / ``fba_settings_9cat.json`` with a short season so a handful of players and days stand
for it. Nothing reaches the network: the market source is a fake that records what it was asked for.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from typer.testing import CliRunner

from fm.config import Config
from fm.decide import trades as trades_module
from fm.decide.trades import (
    ACCEPT,
    BASIS_ROS,
    BASIS_TITLE,
    COUNTER,
    DECLINE,
    DEFAULT_SEED,
    TRADES_CREATED_BY,
    AcceptanceParams,
    MarketBook,
    RankCurve,
    SearchOptions,
    TradeContext,
    TradeError,
    TradeSpec,
    acceptance_probability,
    build_market_book,
    check_legality,
    engine_p_active,
    evaluate_trade,
    find_trades,
    load_trade_context,
    package_value,
    parse_trade_text,
    propose_trades,
    quick_value,
    rank_type_for,
    rank_value,
)
from fm.espn.ids import FBA, FFL
from fm.espn.models import MatchupsView, ProSchedule
from fm.espn.settings import LeagueSettings, LockType, load_league_settings, parse_league_settings
from fm.model.projections import ESPN, BlendWeights, ProjectionSourceRegistry
from fm.model.valuation import GAMES_STAT
from fm.proposals import PolicyError, ProposalKind, TradePayload, parse_payload, pause, resume
from fm.sources.base import Fetched
from fm.sources.market import MarketValue
from fm.store import (
    AvailabilityRow,
    LeagueRow,
    LeagueSettingsRow,
    PlayerRow,
    ProjectionRow,
    ProposalRow,
    RosterEntryRow,
    Store,
    TeamRow,
)

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
ESPN_FIXTURES = FIXTURES / "espn"
PPR = ESPN_FIXTURES / "ffl_settings_ppr.json"
SEASON, WEEK, LEAGUE_ID, US = 2026, 4, 1234567, 3
NOW = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)  # Sunday 11 a.m. ET, before the early games
EQUAL_WEIGHTS = BlendWeights.parse("[nfl.default]\nespn = 1.0\nsleeper = 1.0\n\n[nba.default]\nespn = 1.0\n")
QB, RB, WR, TE, DST, K, FLEX = (FFL.slot_id(label) for label in ("QB", "RB", "WR", "TE", "D/ST", "K", "FLEX"))
BENCH, IR = FFL.bench_slot, FFL.ir_slot
SLOTS = {"QB": {QB}, "RB": {RB, FLEX}, "WR": {WR, FLEX}, "TE": {TE, FLEX}, "D/ST": {DST}, "K": {K}}
TEAMS = (1, 2, 3, 4, 5, 6)
RUNS = 3000

runner = CliRunner()

# A team's roster by slot: (name, position, lineup slot, points a game).
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
"""How strong each team's whole roster is; team 3 (us) is third of six, on the playoff bubble."""
OVERRIDES: dict[tuple[int, str], float] = {
    (US, "RB1"): 18,  # deep at running back...
    (US, "RB2"): 15,
    (US, "RB3"): 12,
    (US, "TE1"): 3,  # ...with a hole at tight end
    (US, "TE2"): 2,
    (2, "RB1"): 9,  # the mirror image
    (2, "RB2"): 7,
    (2, "RB3"): 5,
    (2, "TE1"): 14,
    (2, "TE2"): 9,
}


def pid(team: int, name: str) -> int:
    """The ESPN id of a team's player by template name."""
    return 1000 + team * 20 + [row[0] for row in TEMPLATE].index(name)


def points_of(team: int, name: str) -> float:
    base = {row[0]: row[3] for row in TEMPLATE}[name]
    return OVERRIDES.get((team, name), base * FACTOR[team])


def line(points: float) -> dict[str, float]:
    return {"REY": points * 10}  # 0.1 a receiving yard in the PPR league


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ppr(**schedule: Any) -> LeagueSettings:
    """The hand-built PPR league cut to six teams and four playoff teams (two rounds, weeks 15 and 16)."""
    view = load(PPR)
    view["settings"]["size"] = len(TEAMS)
    view["settings"]["scheduleSettings"].update({"playoffTeamCount": 4, **schedule})
    return parse_league_settings(view)


def player_row(espn_id: int, name: str, position: str, team: int, *, injury: str | None = None) -> PlayerRow:
    return PlayerRow(
        sport="nfl",
        espn_id=espn_id,
        full_name=name,
        default_position_id=FFL.position_id(position),
        position=position,
        pro_team_id=team,
        eligible_slot_ids=[*SLOTS[position], BENCH, IR],
        injury_status=injury,
        as_of=NOW,
    )


def round_robin(teams: tuple[int, ...], periods: range) -> list[tuple[int, int, int]]:
    order = list(teams)
    half = len(order) // 2
    rounds = []
    for _ in range(len(order) - 1):
        rounds.append([(order[i], order[-1 - i]) for i in range(half)])
        order = [order[0], order[-1], *order[1:-1]]
    return [(period, home, away) for n, period in enumerate(periods) for home, away in rounds[n % len(rounds)]]


def schedule_json(settings: LeagueSettings, *, current: int = WEEK) -> dict[str, Any]:
    """A regular season for the six teams as an ``mMatchup`` view, undecided from ``current`` on (earlier weeks: the
    higher id wins)."""
    entries = []
    for number, (period, home, away) in enumerate(
        round_robin(TEAMS, range(1, settings.schedule.regular_season_matchups + 1)), start=1
    ):
        decided = period < current
        entries.append(
            {
                "id": number,
                "matchupPeriodId": period,
                "winner": ("HOME" if home > away else "AWAY") if decided else "UNDECIDED",
                "home": {"teamId": home, "totalPoints": 100.0 + home if decided else 0.0},
                "away": {"teamId": away, "totalPoints": 100.0 + away if decided else 0.0},
            }
        )
    return {"schedule": entries, "status": {"currentMatchupPeriod": current}}


def schedule_view(settings: LeagueSettings, *, current: int = WEEK) -> MatchupsView:
    return MatchupsView.model_validate(schedule_json(settings, current=current))


def seed_league(
    store: Store,
    *,
    settings: LeagueSettings | None = None,
    extra: Iterable[tuple[int, str, str, int, int, float]] = (),
    locked: Iterable[int] = (),
    ir: Iterable[int] = (),
    unprojected: Iterable[int] = (),
    key: str = "nfl",
) -> LeagueRow:
    """The six-team league as ``fm sync`` would store it: every roster, the players, ESPN's week line and season line
    (``GP`` 14) for everyone, and a small wire. ``extra`` adds ``(team, name, position, slot, id, points)`` players;
    ``ir`` moves players to the IR slot, ``unprojected`` leaves players without a
    projection."""
    league = store.leagues.upsert(
        LeagueRow(key=key, sport="nfl", espn_league_id=LEAGUE_ID, season=SEASON, team_id=US, as_of=NOW)
    )
    chosen = settings if settings is not None else ppr()
    store.settings.upsert(
        LeagueSettingsRow(league_id=league.row_id, settings=chosen.model_dump(mode="json"), as_of=NOW)
    )
    store.teams.upsert_many(
        TeamRow(league_id=league.row_id, team_id=team, name=f"Team {team}", as_of=NOW) for team in TEAMS
    )
    stuck, benched = set(locked), set(ir)
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
                    lineup_slot_id=IR if espn_id in benched else slot,
                    lineup_locked=espn_id in stuck,
                    as_of=NOW,
                )
                for owner, _, _, slot, espn_id, _ in people
                if owner == team
            ],
        )
    wire = [
        (9001, "Free RB", "RB", 5.0),
        (9002, "Free WR", "WR", 5.0),
        (9003, "Free TE", "TE", 2.0),
        (9004, "Free QB", "QB", 6.0),
        (9005, "Free K", "K", 4.0),
        (9006, "Free D/ST", "D/ST", 3.0),
    ]
    store.players.upsert_many(
        [
            player_row(espn_id, name, position, 40 + team, injury="OUT" if espn_id in benched else None)
            for team, name, position, _, espn_id, _ in people
        ]
        + [player_row(espn_id, name, position, 60 + n) for n, (espn_id, name, position, _) in enumerate(wire)]
    )
    rows: list[ProjectionRow] = []
    silent = set(unprojected)
    for espn_id, points in [(row[4], row[5]) for row in people] + [(row[0], row[3]) for row in wire]:
        if espn_id in silent:
            continue
        for period, stats in ((WEEK, line(points)), (0, {**line(points * 14), GAMES_STAT: 14})):
            rows.append(
                ProjectionRow(
                    sport="nfl",
                    espn_id=espn_id,
                    source=ESPN,
                    season=SEASON,
                    scoring_period_id=period,
                    stats=stats,
                    as_of=NOW,
                )
            )
    store.projections.upsert_many(rows)
    return league


class FakeMarket:
    """A market source that answers from memory and records what it was asked for. FantasyCalc values follow the
    projections (400 a point) except where ``overrides`` say otherwise."""

    def __init__(
        self, overrides: Mapping[int, int] | None = None, *, degraded: bool = False, ranks: bool = False
    ) -> None:
        self.overrides = dict(overrides or {})
        self.degraded = degraded
        self.ranks = ranks
        self.calls: list[tuple[LeagueSettings, str]] = []

    def market_values(
        self, settings: LeagueSettings, *, rank_type: str = "STANDARD"
    ) -> Fetched[dict[int, MarketValue]]:
        self.calls.append((settings, rank_type))
        found: dict[int, MarketValue] = {}
        if not self.degraded:
            for team in TEAMS:
                for name, position, _, _ in TEMPLATE:
                    espn_id = pid(team, name)
                    value = self.overrides.get(espn_id, int(points_of(team, name) * 400))
                    found[espn_id] = MarketValue(
                        espn_id=espn_id,
                        name=f"T{team} {name}",
                        position=position,
                        trade_value=None if self.ranks else value,
                        espn_ranks={rank_type: max(1, 300 - int(value / 40))} if self.ranks else {},
                    )
        return Fetched(
            found, NOW, "market", "values", "fake", degraded=self.degraded, warnings=("fake: degraded",) * self.degraded
        )


@pytest.fixture
def store() -> Iterator[Store]:
    with Store.open(":memory:") as opened:
        yield opened


def context(
    store: Store, *, market: FakeMarket | None = None, matchups: bool = True, now: datetime = NOW, **seeded: Any
) -> TradeContext:
    league = seed_league(store, **seeded)
    settings = ppr()
    return load_trade_context(
        store,
        league,
        now=now,
        matchups=schedule_view(settings) if matchups else None,
        market=market if market is not None else FakeMarket(),
        weights=EQUAL_WEIGHTS,
    )


# --- league shape -> rank type, and what the market is asked for ------------------------------------------------------


def test_the_rank_type_is_derived_from_the_leagues_settings() -> None:
    assert rank_type_for(ppr()) == "PPR"
    view = load(PPR)
    view["settings"]["scoringSettings"]["scoringItems"] = [
        item for item in view["settings"]["scoringSettings"]["scoringItems"] if item["statId"] != 53
    ]
    assert rank_type_for(parse_league_settings(view)) == "STANDARD"  # receptions score nothing
    superflex = ppr()
    flex = superflex.lineup_slots[0].model_copy(update={"slot_id": FFL.slot_id("OP"), "label": "OP", "count": 1})
    assert rank_type_for(superflex.model_copy(update={"lineup_slots": (*superflex.lineup_slots, flex)})) == "SUPERFLEX"
    assert rank_type_for(load_league_settings(ESPN_FIXTURES / "fba_settings_points.json")) == "STANDARD"
    nine = load(ESPN_FIXTURES / "fba_settings_9cat.json")
    nine["settings"]["scoringSettings"]["scoringType"] = "ROTO"
    assert rank_type_for(parse_league_settings(nine)) == "ROTO"
    assert rank_type_for(load_league_settings(ESPN_FIXTURES / "fba_settings_9cat.json")) == "STANDARD"


def test_the_market_is_asked_with_the_leagues_settings_and_derived_rank_type(store: Store) -> None:
    market = FakeMarket()
    ctx = context(store, market=market)
    assert [(settings.league_id, rank_type) for settings, rank_type in market.calls] == [(LEAGUE_ID, "PPR")]
    assert ctx.market.rank_type == "PPR"
    assert ctx.market.as_of == NOW
    assert set(ctx.market.basis.values()) == {"fantasycalc"} | {"ros_rank"}  # the wire players have no market entry


def test_market_values_prefer_fantasycalc_then_espn_rank_then_our_own_rank() -> None:
    values = {1: 30.0, 2: 20.0, 3: 10.0, 4: 5.0}
    market = {
        1: MarketValue(espn_id=1, name="a", trade_value=7000),
        2: MarketValue(espn_id=2, name="b", espn_ranks={"PPR": 10}, total_ranking=3),
        3: MarketValue(espn_id=3, name="c", total_ranking=40),
    }
    book = build_market_book(market, values, rank_type="PPR", starters=[1, 2])
    assert book.basis == {1: "fantasycalc", 2: "espn_rank", 3: "espn_rank", 4: "ros_rank"}
    assert book.value(1) == 7000.0
    assert book.value(2) == pytest.approx(rank_value(10, 40))  # the draft rank under the league's type wins
    assert book.value(3) == pytest.approx(rank_value(40, 40))
    assert book.value(4) == pytest.approx(rank_value(4, 40))  # fourth by our value, on the feed's size
    assert book.value(99) == 0.0
    assert book.scale == pytest.approx((7000.0 + rank_value(10, 40)) / 2)
    assert build_market_book(None, values, rank_type="PPR").basis == dict.fromkeys(values, "ros_rank")


def test_espn_ranks_are_priced_on_fantasycalcs_curve_not_the_generic_one() -> None:
    # Live NFL, 2026 week 5: FantasyCalc prices skill players only; a kicker and a D/ST ranked 281 and 244 by ESPN in a
    # 1762-deep feed came out at 3334 and 3855 on rank_value, above Davante Adams (2937), so K + D/ST "bought" a QB and
    # an RB at 68% P(accept).
    priced = {
        10 + n: (rank, value)
        for n, (rank, value) in enumerate(
            [(5, 9000), (20, 6000), (45, 2937), (50, 2845), (56, 824), (90, 1500), (150, 600), (220, 250), (300, 120)]
        )
    }
    market = {
        espn_id: MarketValue(espn_id=espn_id, name=str(espn_id), trade_value=value, espn_ranks={"PPR": rank})
        for espn_id, (rank, value) in priced.items()
    }
    market[1] = MarketValue(espn_id=1, name="K", espn_ranks={"PPR": 281}, total_ranking=1762)
    market[2] = MarketValue(espn_id=2, name="D/ST", espn_ranks={"PPR": 244})
    values = {espn_id: float(100 - i) for i, espn_id in enumerate([*priced, 1, 2, 3])}
    book = build_market_book(market, values, rank_type="PPR")
    assert book.basis[1] == book.basis[2] == "espn_rank" and book.basis[3] == "ros_rank"
    assert 120 <= book.value(1) <= 250 and 120 <= book.value(2) <= 250  # between FantasyCalc's 220th and 300th
    assert book.value(2) > book.value(1)  # ranked higher, worth more
    assert book.value(1) + book.value(2) < 2845  # together less than one starting RB
    assert book.value(3) < book.value(13)  # our own 12th is priced on the same curve, below FantasyCalc's ranked ones
    fit = RankCurve.fit([(rank, float(value)) for rank, value in priced.values()], 1762)
    assert fit.value(50) >= fit.value(56) >= fit.value(90)  # the 824 outlier is pooled: values never rise with rank
    assert RankCurve.fit([(1, 9000.0)], 100).value(10) == pytest.approx(rank_value(10, 100))  # too few: generic


def test_espn_ranked_and_fallback_players_are_valued_on_one_scale() -> None:
    # 300 valued players; the ESPN feed ranks some of them up to 1000 deep (a universe longer than our own pool).
    values = {espn_id: float(1000 - espn_id) for espn_id in range(1, 301)}  # id n is our n-th best
    market = {
        1: MarketValue(espn_id=1, name="a", espn_ranks={"PPR": 100}),
        2: MarketValue(espn_id=2, name="b", espn_ranks={"PPR": 1000}),
    }
    book = build_market_book(market, values, rank_type="PPR")
    assert book.basis[1] == "espn_rank" and book.basis[100] == "ros_rank"
    assert book.value(1) == pytest.approx(rank_value(100, 1000))
    # a fallback player who is 100th by our values is worth the ESPN-ranked 100th (about 5000, not about 1800)
    assert book.value(100) == pytest.approx(book.value(1))
    assert book.value(100) == pytest.approx(rank_value(100, 1000))
    # a package of one of each: both bases agree, so it is worth what two 100th-ranked players are
    assert package_value([book.value(1), book.value(100)]) == pytest.approx(
        package_value([rank_value(100, 1000), rank_value(100, 1000)])
    )
    fallback = [book.value(espn_id) for espn_id in range(3, 301)]  # and our own order is kept among the fallback
    assert fallback == sorted(fallback, reverse=True)


def test_a_rank_is_a_value_on_fantasycalcs_scale() -> None:
    assert rank_value(1, 100) == 10_000.0
    assert rank_value(100, 100) == pytest.approx(10.0)
    assert rank_value(500, 100) == pytest.approx(10.0)  # past the pool it is the tail
    assert rank_value(2, 100) > rank_value(3, 100) > rank_value(50, 100)
    with pytest.raises(ValueError, match="at least 1"):
        rank_value(0, 100)


# --- P(accept) --------------------------------------------------------------------------------------------------------


def book_of(values: Mapping[int, float]) -> MarketBook:
    return MarketBook(dict(values), dict.fromkeys(values, "fantasycalc"), "PPR", 1000.0)


def test_a_package_is_its_best_piece_and_most_of_the_rest() -> None:
    params = AcceptanceParams(depth_weight=0.8)
    assert package_value([], params) == 0.0
    assert package_value([5000], params) == 5000
    assert package_value([2000, 5000, 1000], params) == pytest.approx(5000 + 0.8 * 3000)


def test_p_accept_rises_with_what_the_other_manager_gains_in_the_market_and_in_his_lineup() -> None:
    book = book_of({1: 6000, 2: 5000, 3: 4000})
    fair = acceptance_probability(book, [1], [2], need_gain=0.0)
    generous = acceptance_probability(book, [1], [3], need_gain=0.0)
    stingy = acceptance_probability(book, [3], [1], need_gain=0.0)
    assert stingy.p_accept < fair.p_accept < generous.p_accept
    assert generous.surplus == pytest.approx((6000 - 4000) / 10_000)
    assert (fair.receive_value, fair.send_value) == (6000.0, 5000.0)
    needy = acceptance_probability(book, [3], [1], need_gain=1.0)
    assert needy.p_accept > stingy.p_accept
    unwanted = acceptance_probability(book, [1], [2], need_gain=-1.0)
    assert unwanted.p_accept < fair.p_accept
    assert fair.basis == ("fantasycalc",)


def test_p_accept_is_bounded_and_a_forced_drop_costs_a_tenth() -> None:
    book = book_of({1: 9000, 2: 10})
    certain = acceptance_probability(book, [1], [2], need_gain=5.0)
    hopeless = acceptance_probability(book, [2], [1], need_gain=-5.0)
    params = AcceptanceParams()
    assert certain.p_accept == params.ceiling and hopeless.p_accept == params.floor
    even = book_of({1: 6000, 2: 5000})
    middling = acceptance_probability(even, [1], [2], need_gain=-1.0)
    dropped = acceptance_probability(even, [1], [2], need_gain=-1.0, drops=2)
    assert dropped.p_accept == pytest.approx(middling.p_accept * params.drop_friction**2)
    assert dropped.drops == 2 and "market" in middling.describe()


@settings(max_examples=200, deadline=None)
@given(
    mine=st.floats(0.0, 10_000.0),
    better=st.floats(0.0, 5_000.0),
    theirs=st.floats(1.0, 10_000.0),
    need=st.floats(-6.0, 6.0),
    drops=st.integers(0, 3),
)
def test_p_accept_is_a_probability_that_never_falls_as_we_offer_more_or_he_needs_it_more(
    mine: float, better: float, theirs: float, need: float, drops: int
) -> None:
    params = AcceptanceParams()
    low = acceptance_probability(book_of({1: mine, 2: theirs}), [1], [2], need_gain=need, drops=drops)
    high = acceptance_probability(book_of({1: mine + better, 2: theirs}), [1], [2], need_gain=need, drops=drops)
    needier = acceptance_probability(book_of({1: mine, 2: theirs}), [1], [2], need_gain=need + 1.0, drops=drops)
    for found in (low, high, needier):
        assert params.floor <= found.p_accept <= params.ceiling
        assert -1.0 <= found.surplus <= 1.0
    assert high.p_accept >= low.p_accept - 1e-12
    assert needier.p_accept >= low.p_accept - 1e-12


@settings(max_examples=100, deadline=None)
@given(st.integers(1, 500), st.integers(1, 500), st.integers(2, 500))
def test_a_better_rank_is_never_worth_less(first: int, second: int, size: int) -> None:
    better, worse = sorted((first, second))
    assert rank_value(better, size) >= rank_value(worse, size)
    assert 10.0 - 1e-9 <= rank_value(worse, size) <= 10_000.0 + 1e-9


def test_every_weight_of_p_accept_is_a_parameter() -> None:
    book = book_of({1: 6000, 2: 5000})
    gentle = acceptance_probability(book, [2], [1], need_gain=0.0, params=AcceptanceParams(market_weight=1.0))
    harsh = acceptance_probability(book, [2], [1], need_gain=0.0, params=AcceptanceParams(market_weight=20.0))
    assert harsh.p_accept < gentle.p_accept


# --- the context ------------------------------------------------------------------------------------------------------


def test_the_context_holds_every_roster_and_values_every_player(store: Store) -> None:
    ctx = context(store)
    assert set(ctx.rosters) == set(TEAMS) and ctx.team_id == US and ctx.other_teams == (1, 2, 4, 5, 6)
    assert all(len(roster) == len(TEMPLATE) for roster in ctx.rosters.values())
    assert ctx.period == WEEK and ctx.matchup_period == WEEK
    assert ctx.model.unit == "points" and ctx.model.simulates
    assert set(ctx.model.values) >= {espn_id for roster in ctx.rosters.values() for espn_id in roster}
    assert ctx.model.values[pid(US, "QB1")] > ctx.model.values[pid(US, "QB2")] > 0
    assert ctx.name(pid(US, "RB1")) == "T3 RB1" and ctx.team_name(2) == "Team 2"
    assert ctx.owner(pid(2, "TE1")) == 2 and ctx.owner(9001) is None
    assert ctx.scale > 0


def test_a_context_needs_a_synced_league(store: Store) -> None:
    league = store.leagues.upsert(
        LeagueRow(key="nfl", sport="nfl", espn_league_id=LEAGUE_ID, season=SEASON, team_id=US, as_of=NOW)
    )
    with pytest.raises(TradeError, match="no synced settings"):
        load_trade_context(store, league, now=NOW)
    store.settings.upsert(LeagueSettingsRow(league_id=league.row_id, settings=ppr().model_dump(mode="json"), as_of=NOW))
    with pytest.raises(TradeError, match="no roster snapshot"):
        load_trade_context(store, league, now=NOW)
    with pytest.raises(ValueError, match="aware"):
        load_trade_context(store, league, now=NOW.replace(tzinfo=None))


def test_without_a_matchup_schedule_or_a_market_the_context_says_so(store: Store) -> None:
    league = seed_league(store)
    ctx = load_trade_context(store, league, now=NOW, weights=EQUAL_WEIGHTS)
    assert "no mMatchup schedule given: deals are judged on rest-of-season value, not title odds" in ctx.warnings
    assert "no market source: P(accept) uses our own rest-of-season ranks" in ctx.warnings
    assert set(ctx.market.basis.values()) == {"ros_rank"}
    degraded = context(store, market=FakeMarket(degraded=True))
    assert "market values are unavailable: P(accept) uses our own rest-of-season ranks" in degraded.warnings
    assert "fake: degraded" in degraded.warnings


def test_the_engines_p_active_is_the_number_before_claudes_news(store: Store) -> None:
    ctx = context(store)
    questionable = pid(2, "WR1")
    row = AvailabilityRow(
        sport="nfl",
        espn_id=questionable,
        season=SEASON,
        scoring_period_id=WEEK,
        designation="QUESTIONABLE",
        p_active=0.95,
        inputs={"news": {"before": 0.4, "delta": 0.3}},
        as_of=NOW,
    )
    store.availability.upsert(row)
    players = [ctx.players[questionable], ctx.players[pid(2, "WR2")]]
    found = engine_p_active(store, players, season=SEASON, period=WEEK, now=NOW)
    assert found[questionable] == 0.4  # not the 0.95 Claude's signal made of it
    assert 0.0 <= found[pid(2, "WR2")] <= 1.0  # no stored row: a bare assess
    plain = AvailabilityRow(
        sport="nfl", espn_id=questionable, season=SEASON, scoring_period_id=WEEK, p_active=0.7, as_of=NOW
    )
    store.availability.upsert(plain)
    assert engine_p_active(store, players, season=SEASON, period=WEEK, now=NOW)[questionable] == 0.7
    bad = plain.model_copy(update={"inputs": {"news": {"before": 7}}})
    store.availability.upsert(bad)
    assert engine_p_active(store, players, season=SEASON, period=WEEK, now=NOW)[questionable] == 0.7


def test_a_claude_only_signal_does_not_change_what_a_player_is_worth(store: Store) -> None:
    before = context(store)
    store.availability.upsert(
        AvailabilityRow(
            sport="nfl",
            espn_id=pid(2, "WR1"),
            season=SEASON,
            scoring_period_id=WEEK,
            designation="QUESTIONABLE",
            p_active=0.1,
            inputs={"news": {"before": 0.9, "delta": -0.8}},
            as_of=NOW,
        )
    )
    after = load_trade_context(
        store, before.league, now=NOW, matchups=before.matchups, market=FakeMarket(), weights=EQUAL_WEIGHTS
    )
    assert after.p_active[pid(2, "WR1")] == 0.9
    assert dict(after.model.values) == dict(before.model.values)


# --- quick values -----------------------------------------------------------------------------------------------------


def test_a_greedy_lineup_fills_the_most_specific_slot_and_leaves_the_rest_at_replacement(store: Store) -> None:
    model = context(store).model
    full = quick_value(model, [pid(US, name) for name, *_ in TEMPLATE])
    exact = model.roster_value([pid(US, name) for name, *_ in TEMPLATE])
    assert full == pytest.approx(exact, rel=0.05)  # the screen is close to the exact lineup on a full roster
    without_te = quick_value(model, [pid(US, name) for name, *_ in TEMPLATE if not name.startswith("TE")])
    assert without_te < full
    assert quick_value(model, []) == pytest.approx(sum(model.replacement[slot] for slot in model.slots))
    assert quick_value(model, [123456]) == quick_value(model, [])  # unvalued players are ignored


# --- evaluating a deal ------------------------------------------------------------------------------------------------


def swap_a_running_back_for_a_tight_end() -> TradeSpec:
    return TradeSpec(2, (pid(US, "RB2"),), (pid(2, "TE1"),))


def test_a_trade_that_fills_our_hole_and_fills_theirs_helps_both_sides(store: Store) -> None:
    ctx = context(store)
    found = evaluate_trade(ctx, swap_a_running_back_for_a_tight_end(), runs=RUNS)
    assert found.legal and found.simulated and found.score_basis == BASIS_TITLE
    assert found.ours.delta_ros > 0 and found.theirs.delta_ros > 0
    assert found.ours.delta_title is not None and found.ours.delta_title > 0
    assert found.ours.delta_playoffs is not None and found.ours.delta_bye == 0.0  # a four-team bracket has no byes
    assert found.theirs.delta_title is not None
    assert found.ours.ros_after == pytest.approx(found.ours.ros_before + found.ours.delta_ros)
    assert found.acceptance.need_gain > 0 and 0.0 < found.acceptance.p_accept < 1.0
    assert found.score == pytest.approx(found.ours.delta_title * found.acceptance.p_accept)
    assert found.recommendation == ACCEPT and "title odds" in found.reason
    assert (found.runs, found.seed, found.unit) == (RUNS, DEFAULT_SEED, "points")


def test_a_symmetric_trade_changes_nothing_for_either_side(store: Store) -> None:
    twin_ours = (US, "Twin A", "WR", BENCH, 7001, 11.0)
    twin_theirs = (2, "Twin B", "WR", BENCH, 7002, 11.0)
    ctx = context(store, extra=(twin_ours, twin_theirs))
    found = evaluate_trade(ctx, TradeSpec(2, (7001,), (7002,)), runs=RUNS)
    assert found.ours.delta_ros == pytest.approx(0.0, abs=1e-9)
    assert found.theirs.delta_ros == pytest.approx(0.0, abs=1e-9)
    assert found.ours.delta_title == pytest.approx(0.0, abs=1e-12)
    assert found.theirs.delta_title == pytest.approx(0.0, abs=1e-12)
    assert found.ours.delta_playoffs == pytest.approx(0.0, abs=1e-12)
    assert found.score == pytest.approx(0.0, abs=1e-12)
    assert found.recommendation != ACCEPT  # nothing to gain


def test_a_roster_is_valued_by_its_lineup_not_by_the_sum_of_its_players(store: Store) -> None:
    ctx = context(store)
    values = ctx.model.values
    mine = ctx.playing(US)
    # our spare tight end for their fourth receiver: the receiver is projected for more, but neither would start
    give, get = pid(US, "TE2"), pid(2, "WR4")
    bench = evaluate_trade(ctx, TradeSpec(2, (give,), (get,)), simulate=False)
    assert values[get] > values[give] and bench.ours.delta_ros == pytest.approx(0.0, abs=1e-9)
    # our best back for their best receiver: the change is the two lineups' difference
    give, get = pid(US, "RB1"), pid(2, "WR1")
    found = evaluate_trade(ctx, TradeSpec(2, (give,), (get,)), simulate=False)
    assert found.ours.ros_before == pytest.approx(ctx.model.roster_value(mine))
    assert found.ours.ros_after == pytest.approx(ctx.model.roster_value((mine - {give}) | {get}))
    assert found.ours.delta_ros != 0.0


def test_an_evaluation_is_reproducible_from_its_seed(store: Store) -> None:
    ctx = context(store)
    spec = swap_a_running_back_for_a_tight_end()
    first = evaluate_trade(ctx, spec, runs=RUNS, seed=7)
    again = evaluate_trade(ctx, spec, runs=RUNS, seed=7)
    other = evaluate_trade(ctx, spec, runs=RUNS, seed=8)
    assert first == again
    assert first.ours.odds_after != other.ours.odds_after


def test_without_matchups_a_trade_is_judged_on_rest_of_season_value(store: Store) -> None:
    ctx = context(store, matchups=False)
    found = evaluate_trade(ctx, swap_a_running_back_for_a_tight_end())
    assert not found.simulated and found.score_basis == BASIS_ROS and found.runs == 0
    assert found.ours.delta_title is None and found.ours.odds_before is None
    assert found.gain == pytest.approx(found.ours.delta_ros / ctx.scale)
    assert found.score == pytest.approx(found.gain * found.acceptance.p_accept)
    assert found.recommendation == ACCEPT and "starter seasons" in found.reason


def test_a_trade_that_hurts_us_is_declined_and_one_he_loves_is_countered(store: Store) -> None:
    ctx = context(store, matchups=False)
    # our quarterback for their backup: he would love it, we lose a starter
    loved = evaluate_trade(ctx, TradeSpec(2, (pid(US, "QB1"),), (pid(2, "QB2"),)), simulate=False)
    assert loved.ours.delta_ros < 0.0 and loved.acceptance.p_accept >= 0.6
    assert loved.recommendation == COUNTER and "ask for more" in loved.reason
    # our kicker for their fourth receiver: nobody gains and he is not keen
    hurts = evaluate_trade(ctx, TradeSpec(2, (pid(US, "K"),), (pid(2, "WR4"),)), simulate=False)
    assert hurts.ours.delta_ros < 0.0 and hurts.recommendation in (DECLINE, COUNTER)
    # their quarterback for our fourth receiver: worth asking for, however unlikely
    mine = evaluate_trade(ctx, TradeSpec(2, (pid(US, "WR4"),), (pid(2, "QB1"),)), simulate=False)
    assert mine.ours.delta_ros > 0.0 and mine.acceptance.p_accept < loved.acceptance.p_accept
    assert mine.recommendation == ACCEPT


def test_an_unknown_player_or_team_is_an_error_not_a_verdict(store: Store) -> None:
    ctx = context(store)
    with pytest.raises(TradeError, match="not on our roster"):
        evaluate_trade(ctx, TradeSpec(2, (pid(2, "RB1"),), (pid(2, "TE1"),)))
    with pytest.raises(TradeError, match="not on Team 2's roster"):
        evaluate_trade(ctx, TradeSpec(2, (pid(US, "RB1"),), (pid(4, "TE1"),)))
    with pytest.raises(TradeError, match="not another team"):
        evaluate_trade(ctx, TradeSpec(US, (pid(US, "RB1"),), (pid(US, "TE1"),)))
    with pytest.raises(TradeError, match="not another team"):
        evaluate_trade(ctx, TradeSpec(9, (pid(US, "RB1"),), (pid(2, "TE1"),)))
    with pytest.raises(TradeError, match="on both sides"):
        TradeSpec(2, (1,), (1,))
    with pytest.raises(TradeError, match="at least one side"):
        TradeSpec(2, (), ())
    with pytest.raises(TradeError, match="twice"):
        TradeSpec(2, (1, 1), (2,))


def test_a_player_nothing_projects_is_flagged_and_never_asked_for(store: Store) -> None:
    ctx = context(store, extra=((2, "Mystery", "RB", BENCH, 7100, 9.0),), unprojected=[7100])
    assert 7100 not in ctx.model.values and ctx.model.knows(7100)  # known, but his value is unknown, not zero
    found = evaluate_trade(ctx, TradeSpec(2, (pid(US, "RB4"),), (7100,)), simulate=False)
    assert found.legal and "Mystery has no projection: his value is unknown and counts as 0" in found.warnings
    search = find_trades(ctx, opponents=[2], options=SearchOptions(runs=RUNS, limit=100, finalists=100))
    assert all(7100 not in result.spec.get for result in search.results)


# --- legality ---------------------------------------------------------------------------------------------------------


def test_a_clean_deal_is_legal_and_names_nothing_to_fix(store: Store) -> None:
    legality = check_legality(context(store), swap_a_running_back_for_a_tight_end())
    assert legality.legal and legality.problems == () and legality.drops_ours == () and legality.drops_theirs == ()


def test_the_trade_deadline_comes_from_the_leagues_settings(store: Store) -> None:
    ctx = context(store)
    deadline = ctx.settings.trade.deadline
    assert deadline is not None and ctx.now < deadline
    late = context(store, now=datetime(2027, 1, 1, tzinfo=UTC))
    legality = check_legality(late, swap_a_running_back_for_a_tight_end())
    assert not legality.legal and "trade deadline" in legality.problems[0]
    found = evaluate_trade(late, swap_a_running_back_for_a_tight_end(), simulate=False)
    assert found.score is None and found.recommendation == DECLINE and found.reason.startswith("not legal")


def test_a_league_without_a_deadline_never_closes(store: Store) -> None:
    view = load(PPR)
    view["settings"]["size"] = len(TEAMS)
    view["settings"]["scheduleSettings"]["playoffTeamCount"] = 4
    view["settings"]["tradeSettings"]["deadlineDate"] = None
    league = seed_league(store, settings=parse_league_settings(view))
    ctx = load_trade_context(
        store, league, now=datetime(2030, 1, 1, tzinfo=UTC), market=FakeMarket(), weights=EQUAL_WEIGHTS
    )
    assert check_legality(ctx, swap_a_running_back_for_a_tight_end()).legal


def test_a_full_roster_must_drop_to_make_room_and_the_cheapest_player_goes(store: Store) -> None:
    fillers = tuple((2, f"Filler {n}", "WR", BENCH, 7200 + n, 1.0) for n in range(2))
    ctx = context(store, extra=fillers)  # team 2 now holds 16: full
    spec = TradeSpec(2, (pid(US, "RB4"), pid(US, "WR4")), (pid(2, "TE2"),))  # 2 for 1: they take on one more
    legality = check_legality(ctx, spec)
    assert legality.legal and legality.drops_ours == ()
    assert len(legality.drops_theirs) == 1 and legality.drops_theirs[0] in ctx.rosters[2] - {pid(2, "TE2")}
    assert any("must drop" in note for note in legality.notes)
    cheapest = min(
        (espn_id for espn_id in ctx.rosters[2] if espn_id != pid(2, "TE2")),
        key=lambda espn_id: (ctx.model.values[espn_id], espn_id),
    )
    assert legality.drops_theirs == (cheapest,)


def test_a_two_for_one_into_our_full_roster_needs_a_drop_on_our_side(store: Store) -> None:
    fillers = tuple((US, f"Filler {n}", "WR", BENCH, 7300 + n, 0.5) for n in range(2))
    ctx = context(store, extra=fillers)  # we hold 16
    legality = check_legality(ctx, TradeSpec(2, (pid(US, "RB4"),), (pid(2, "TE1"), pid(2, "RB1"))))
    assert legality.legal and len(legality.drops_ours) == 1
    assert legality.drops_ours[0] not in ctx.untouchables
    found = evaluate_trade(ctx, TradeSpec(2, (pid(US, "RB4"),), (pid(2, "TE1"), pid(2, "RB1"))), simulate=False)
    assert found.legality.drops_ours == legality.drops_ours


def test_a_roster_that_cannot_drop_anyone_blocks_the_trade(store: Store) -> None:
    crowded = tuple((2, f"Filler {n}", "WR", BENCH, 7400 + n, 1.0) for n in range(2))
    league = seed_league(store, extra=crowded, locked=[row[4] for row in crowded])
    ctx = load_trade_context(store, league, now=NOW, market=FakeMarket(), weights=EQUAL_WEIGHTS)
    stuck = {row[4] for row in crowded}
    tight = replace(ctx, locked=ctx.locked | (ctx.rosters[2] - {pid(2, "TE2")}))  # nobody else may be dropped
    assert stuck <= tight.locked
    legality = check_legality(tight, TradeSpec(2, (pid(US, "RB4"), pid(US, "WR4")), (pid(2, "TE2"),)))
    assert not legality.legal and "nobody could be dropped" in legality.problems[0]


def test_position_limits_are_the_leagues(store: Store) -> None:
    view = load(PPR)
    view["settings"]["size"] = len(TEAMS)
    view["settings"]["scheduleSettings"]["playoffTeamCount"] = 4
    view["settings"]["rosterSettings"]["positionLimits"]["4"] = 2  # at most two tight ends
    settings = parse_league_settings(view)
    league = seed_league(store, settings=settings)
    ctx = load_trade_context(store, league, now=NOW, market=FakeMarket(), weights=EQUAL_WEIGHTS)
    spec = TradeSpec(2, (pid(US, "RB4"),), (pid(2, "TE1"), pid(2, "TE2")))
    legality = check_legality(ctx, spec)
    assert not legality.legal and any("would hold 4 TE and the league allows 2" in why for why in legality.problems)
    # a limit the roster already breaks is not the trade's doing
    fine = check_legality(ctx, TradeSpec(2, (pid(US, "TE1"),), (pid(2, "TE1"),)))
    assert fine.legal


def test_untouchables_are_never_given_away(store: Store) -> None:
    config = Config.model_validate(
        {
            "league": [
                {
                    "key": "nfl",
                    "sport": "nfl",
                    "espn_league_id": LEAGUE_ID,
                    "season": SEASON,
                    "team_id": US,
                    "policy": {"untouchables": ["T3 RB1"]},
                }
            ]
        }
    )
    league = seed_league(store)
    ctx = load_trade_context(store, league, now=NOW, config=config, market=FakeMarket(), weights=EQUAL_WEIGHTS)
    assert ctx.untouchables == {pid(US, "RB1"): "T3 RB1"}
    legality = check_legality(ctx, TradeSpec(2, (pid(US, "RB1"),), (pid(2, "TE1"),)))
    assert not legality.legal and "T3 RB1 is untouchable" in legality.problems[0]
    assert check_legality(ctx, TradeSpec(2, (pid(US, "RB2"),), (pid(2, "TE1"),))).legal


def test_a_player_whose_game_has_started_is_locked_under_an_individual_game_lock(store: Store) -> None:
    ctx = context(store, locked=[pid(2, "TE1")])
    assert ctx.settings.roster_lock_type is LockType.INDIVIDUAL_GAME
    legality = check_legality(ctx, swap_a_running_back_for_a_tight_end())
    assert not legality.legal and "T2 TE1 is locked: his game has started" in legality.problems


def pro_schedule(kickoffs: Mapping[int, datetime]) -> ProSchedule:
    """Week 4 for every pro team the league's players are on (``40 + team`` and ``60 + n``): each plays at its
    ``kickoffs`` time, the others on Monday night."""
    monday = datetime(2026, 10, 5, 23, 0, tzinfo=UTC)
    teams = []
    for team in [*range(41, 47), *range(60, 66)]:
        when = kickoffs.get(team, monday)
        game = {
            "id": team,
            "date": int(when.timestamp() * 1000),
            "scoringPeriodId": WEEK,
            "homeProTeamId": team,
            "awayProTeamId": 99,
        }
        teams.append({"id": team, "proGamesByScoringPeriod": {str(WEEK): [game]}})
    return ProSchedule.model_validate({"proTeams": teams})


def test_a_players_roster_lock_comes_from_his_teams_kickoff_under_an_individual_game_lock(store: Store) -> None:
    league = seed_league(store)
    kickoff = datetime(2026, 10, 4, 17, 0, tzinfo=UTC)  # team 2's players (pro team 42) kick off at 1 p.m. ET
    schedule = pro_schedule({42: kickoff})
    spec = swap_a_running_back_for_a_tight_end()
    early = load_trade_context(store, league, now=NOW, schedule=schedule, market=FakeMarket(), weights=EQUAL_WEIGHTS)
    assert early.settings.roster_lock_type is LockType.INDIVIDUAL_GAME
    assert check_legality(early, spec).legal  # 11 a.m.: nobody has started
    late = replace(early, now=kickoff.replace(hour=18))
    problems = check_legality(late, spec).problems
    assert problems == ("T2 TE1 is locked: transactions closed at 2026-10-04 17:00Z",)  # only his game started
    elsewhere = TradeSpec(4, (pid(US, "RB2"),), (pid(4, "TE1"),))  # team 4's players kick off on Monday night
    assert check_legality(late, elsewhere).legal


def two_week_schedule() -> ProSchedule:
    """Weeks 4 and 5 for every pro team the league's players are on: everyone plays Sunday 1 p.m. ET in each."""
    teams = []
    for team in [*range(41, 47), *range(60, 66)]:
        games = {}
        for week, day in (
            (WEEK, datetime(2026, 10, 4, 17, 0, tzinfo=UTC)),
            (WEEK + 1, datetime(2026, 10, 11, 17, 0, tzinfo=UTC)),
        ):
            games[str(week)] = [
                {
                    "id": team * 10 + week,
                    "date": int(day.timestamp() * 1000),
                    "scoringPeriodId": week,
                    "homeProTeamId": team,
                    "awayProTeamId": 99,
                }
            ]
        teams.append({"id": team, "proGamesByScoringPeriod": games})
    return ProSchedule.model_validate({"proTeams": teams})


def test_locks_are_judged_in_the_current_period_when_the_last_sync_is_older(store: Store) -> None:
    league = seed_league(store)  # the roster snapshot is week 4
    sunday_next = datetime(2026, 10, 11, 15, 0, tzinfo=UTC)  # week 5, 11 a.m. ET, before its games
    ctx = load_trade_context(
        store, league, now=sunday_next, schedule=two_week_schedule(), market=FakeMarket(), weights=EQUAL_WEIGHTS
    )
    assert ctx.period == WEEK and ctx.current_period == WEEK + 1 and ctx.lock_period == WEEK + 1
    assert any(f"period {WEEK + 1} is current" in warning for warning in ctx.warnings)
    # week 4's kickoffs are long past; judged there every player would be "transactions closed"
    assert check_legality(ctx, swap_a_running_back_for_a_tight_end()).legal
    assert find_trades(ctx, options=SearchOptions(runs=RUNS)).results
    fresh = load_trade_context(
        store, league, now=NOW, schedule=two_week_schedule(), market=FakeMarket(), weights=EQUAL_WEIGHTS
    )
    assert fresh.current_period == WEEK and not any("is current" in warning for warning in fresh.warnings)


def test_a_first_game_lock_closes_every_trade_at_the_periods_first_kickoff(store: Store) -> None:
    league = seed_league(store)
    kickoff = datetime(2026, 10, 4, 17, 0, tzinfo=UTC)
    schedule = pro_schedule({42: kickoff})
    spec = swap_a_running_back_for_a_tight_end()
    ctx = load_trade_context(store, league, now=NOW, schedule=schedule, market=FakeMarket(), weights=EQUAL_WEIGHTS)
    first_game = replace(
        ctx, settings=ctx.settings.model_copy(update={"roster_lock_type": LockType.FIRSTGAME_SCORINGPERIOD})
    )
    assert check_legality(first_game, spec).legal
    after = replace(first_game, now=kickoff.replace(hour=18))
    problems = check_legality(after, spec).problems
    assert len(problems) == 2 and all(
        "transactions closed at 2026-10-04 17:00Z" in why for why in problems
    )  # both players


def test_a_lock_type_the_plugins_will_not_read_is_not_guessed(store: Store) -> None:
    league = seed_league(store)
    kickoff = datetime(2026, 10, 4, 17, 0, tzinfo=UTC)
    ctx = load_trade_context(
        store,
        league,
        now=kickoff.replace(hour=18),
        schedule=pro_schedule({42: kickoff}),
        market=FakeMarket(),
        weights=EQUAL_WEIGHTS,
    )
    unknown = replace(ctx, settings=ctx.settings.model_copy(update={"roster_lock_type": LockType.UNKNOWN}))
    assert check_legality(
        unknown, swap_a_running_back_for_a_tight_end()
    ).legal  # the executor re-checks the live league
    assert any("lock type" in warning for warning in _lock_warnings(store, league, kickoff))


def _lock_warnings(store: Store, league: LeagueRow, kickoff: datetime) -> tuple[str, ...]:
    settings = ppr().model_copy(
        update={"roster_lock_type": LockType.UNKNOWN, "roster_lock_type_raw": "FIRSTGAME_WEEKLY"}
    )
    store.settings.upsert(
        LeagueSettingsRow(league_id=league.row_id, settings=settings.model_dump(mode="json"), as_of=NOW)
    )
    again = load_trade_context(
        store, league, now=NOW, schedule=pro_schedule({42: kickoff}), market=FakeMarket(), weights=EQUAL_WEIGHTS
    )
    return again.warnings


def test_a_player_in_the_ir_slot_lands_in_ir_when_there_is_room(store: Store) -> None:
    ctx = context(store, ir=[pid(2, "RB4")])
    assert ctx.ir[2] == {pid(2, "RB4")} and pid(2, "RB4") not in ctx.playing(2)
    legality = check_legality(ctx, TradeSpec(2, (pid(US, "TE2"),), (pid(2, "RB4"),)))
    assert legality.legal and any("is in an IR slot" in note for note in legality.notes)
    full = context(store, ir=[pid(2, "RB4"), pid(US, "WR4")])  # one IR slot each, ours is taken
    assert check_legality(full, TradeSpec(2, (pid(US, "TE2"),), (pid(2, "RB4"),))).legal


def heal(store: Store, team: int, name: str, position: str) -> None:
    """The player is no longer injured as far as the last sync can tell (``seed_league`` makes IR players OUT)."""
    store.players.upsert(player_row(pid(team, name), f"T{team} {name}", position, 40 + team))


def fresh_context(store: Store, league: LeagueRow) -> TradeContext:
    return load_trade_context(
        store, league, now=NOW, matchups=schedule_view(ppr()), market=FakeMarket(), weights=EQUAL_WEIGHTS
    )


def test_a_team_holding_an_ir_ineligible_player_in_an_ir_slot_cannot_trade(store: Store) -> None:
    league = seed_league(store, ir=[pid(2, "RB4")])
    heal(store, 2, "RB4", "RB")  # healthy but still parked in IR: ESPN blocks that team's moves
    ctx = fresh_context(store, league)
    spec = TradeSpec(2, (pid(US, "RB2"),), (pid(2, "TE1"),))  # a deal that does not touch him
    problems = check_legality(ctx, spec).problems
    assert (
        len(problems) == 1 and "Team 2 holds T2 RB4 in an IR slot without an IR-eligible injury status" in problems[0]
    )
    # ours is flagged the same way
    ours = seed_league(store, ir=[pid(US, "WR4")])
    heal(store, US, "WR4", "WR")
    assert any("we hold T3 WR4" in why for why in check_legality(fresh_context(store, ours), spec).problems)


def test_a_player_who_is_out_may_sit_in_ir_without_blocking_the_team(store: Store) -> None:
    ctx = context(store, ir=[pid(2, "RB4")])
    assert check_legality(ctx, TradeSpec(2, (pid(US, "RB2"),), (pid(2, "TE1"),))).legal


def test_an_incoming_player_lands_in_ir_only_if_his_injury_status_allows_it(store: Store) -> None:
    ctx = context(store, ir=[pid(2, "RB4")])  # OUT, in team 2's IR slot
    mine = trades_module._after(ctx, US, (pid(US, "TE2"),), (pid(2, "RB4"),))
    assert pid(2, "RB4") in mine.ir and pid(2, "RB4") not in mine.playing
    healed = replace(ctx, players={**ctx.players, pid(2, "RB4"): player_row(pid(2, "RB4"), "T2 RB4", "RB", 42)})
    after = trades_module._after(healed, US, (pid(US, "TE2"),), (pid(2, "RB4"),))
    assert pid(2, "RB4") not in after.ir and pid(2, "RB4") in after.everyone  # he takes a roster spot: no IR for him
    assert after.held == mine.held + 1
    assert not healed.ir_eligible(pid(2, "RB4")) and ctx.ir_eligible(pid(2, "RB4"))
    assert not ctx.ir_eligible(123456789)  # a player the store does not know


def test_the_leagues_trade_limit_stops_us_once_it_is_used(store: Store) -> None:
    capped = ppr().model_copy(update={"trade": ppr().trade.model_copy(update={"max_trades": 1})})
    league = seed_league(store, settings=capped)
    spec = swap_a_running_back_for_a_tight_end()
    before = check_legality(fresh_context(store, league), spec)
    assert before.legal and any("allows 1 trades a season" in note for note in before.notes)
    store.proposals.insert(
        ProposalRow(
            league_id=league.row_id,
            kind=ProposalKind.TRADE_ACCEPT.value,
            status="verified",
            policy="approve",
            payload={"espn_transaction_id": "t1"},
            created_by="test",
            created_at=NOW,
        )
    )
    used = fresh_context(store, league)
    assert used.trades_made == 1
    assert check_legality(used, spec).problems == ("we have made 1 trades and the league allows 1 a season",)


def test_a_league_without_a_trade_limit_says_nothing_about_one(store: Store) -> None:
    legality = check_legality(context(store), swap_a_running_back_for_a_tight_end())
    assert legality.legal and not any("trades a season" in note for note in legality.notes)


# --- finding deals ----------------------------------------------------------------------------------------------------


def test_the_finder_returns_only_legal_deals_in_ranked_order(store: Store) -> None:
    ctx = context(store)
    search = find_trades(ctx, options=SearchOptions(runs=RUNS))
    assert search.results, search.warnings
    assert search.enumerated > search.screened >= search.rescored >= len(search.results)
    scores = [found.score for found in search.results]
    assert all(score is not None and score > 0 for score in scores)
    assert scores == sorted(scores, reverse=True)  # type: ignore[type-var]
    for found in search.results:
        assert found.legal and found.legality.drops_ours == ()
        assert found.score_basis == BASIS_TITLE and found.ours.delta_title is not None and found.ours.delta_title > 0
        assert found.acceptance.p_accept >= SearchOptions().min_accept
        assert check_legality(ctx, found.spec).legal
        assert not set(found.spec.give) & set(ctx.untouchables)
    best = search.results[0]  # the best deal fills our tight end hole from a team with one to spare
    assert any(ctx.players[espn_id].position == "TE" for espn_id in best.spec.get)
    assert (best.ours.delta_title or 0.0) > 0.0


def test_the_finder_reproduces_from_its_seed(store: Store) -> None:
    ctx = context(store)
    options = SearchOptions(runs=RUNS, seed=11, limit=5)
    first = find_trades(ctx, options=options)
    again = find_trades(ctx, options=options)
    assert [(found.spec, found.score) for found in first.results] == [
        (found.spec, found.score) for found in again.results
    ]
    assert len(first.results) <= 5


def test_the_finder_without_matchups_ranks_by_rest_of_season_value(store: Store) -> None:
    ctx = context(store, matchups=False)
    search = find_trades(ctx, options=SearchOptions(limit=5))
    assert search.results and all(found.score_basis == BASIS_ROS for found in search.results)
    scores = [found.score or 0.0 for found in search.results]
    assert scores == sorted(scores, reverse=True)
    assert all(found.ours.delta_ros > 0 for found in search.results)


def test_the_finder_can_be_limited_to_opponents_and_to_deal_shapes(store: Store) -> None:
    ctx = context(store)
    only_four = find_trades(ctx, opponents=[4], options=SearchOptions(runs=RUNS))
    assert {found.spec.other_team_id for found in only_four.results} <= {4}
    assert only_four.opponents == (4,)
    ones = find_trades(ctx, opponents=[2], options=SearchOptions(runs=RUNS, shapes=((1, 1),)))
    assert ones.results and all(len(found.spec.give) == len(found.spec.get) == 1 for found in ones.results)
    with pytest.raises(TradeError, match="not another team"):
        find_trades(ctx, opponents=[US])


def test_the_finder_enumerates_one_for_one_two_for_one_and_two_for_two(store: Store) -> None:
    ctx = context(store)
    shapes: set[tuple[int, int]] = set()
    for shape in ((1, 1), (2, 1), (1, 2), (2, 2)):
        search = find_trades(
            ctx, opponents=[2], options=SearchOptions(runs=RUNS, shapes=(shape,), finalists=50, limit=50)
        )
        shapes.update((len(found.spec.give), len(found.spec.get)) for found in search.results)
    assert shapes == {(1, 1), (2, 1), (1, 2), (2, 2)}


def test_the_finder_leaves_out_what_needs_a_drop_unless_asked(store: Store) -> None:
    fillers = tuple((US, f"Filler {n}", "WR", BENCH, 7500 + n, 0.5) for n in range(2))
    ctx = context(store, extra=fillers)  # we hold 16
    strict = find_trades(ctx, opponents=[2], options=SearchOptions(runs=RUNS, shapes=((1, 2),), finalists=50, limit=50))
    assert all(found.legality.drops_ours == () for found in strict.results)
    relaxed = find_trades(
        ctx, opponents=[2], options=SearchOptions(runs=RUNS, shapes=((1, 2),), finalists=50, limit=50, allow_drops=True)
    )
    assert len(relaxed.results) >= len(strict.results)
    assert any(found.legality.drops_ours for found in relaxed.results)


def test_the_finder_never_offers_an_untouchable_or_asks_for_a_player_who_is_out(store: Store) -> None:
    config = Config.model_validate(
        {
            "league": [
                {
                    "key": "nfl",
                    "sport": "nfl",
                    "espn_league_id": LEAGUE_ID,
                    "season": SEASON,
                    "team_id": US,
                    "policy": {"untouchables": ["T3 RB1", "T3 RB2", "T3 RB3"]},
                }
            ]
        }
    )
    league = seed_league(store)
    store.availability.upsert(
        AvailabilityRow(
            sport="nfl", espn_id=pid(2, "TE1"), season=SEASON, scoring_period_id=WEEK, p_active=0.0, as_of=NOW
        )
    )
    ctx = load_trade_context(
        store, league, now=NOW, config=config, matchups=schedule_view(ppr()), market=FakeMarket(), weights=EQUAL_WEIGHTS
    )
    search = find_trades(ctx, options=SearchOptions(runs=RUNS, finalists=60, limit=60))
    gave = {espn_id for found in search.results for espn_id in found.spec.give}
    asked = {espn_id for found in search.results for espn_id in found.spec.get}
    assert not gave & {pid(US, "RB1"), pid(US, "RB2"), pid(US, "RB3")}
    assert pid(2, "TE1") not in asked


def test_a_deal_the_market_hates_is_not_worth_finding(store: Store) -> None:
    hated = {pid(2, "TE1"): 1_000_000}  # he would never part with his tight end
    ctx = context(store, market=FakeMarket(hated))
    search = find_trades(ctx, opponents=[2], options=SearchOptions(runs=RUNS, limit=50, finalists=50))
    assert all(pid(2, "TE1") not in found.spec.get or found.acceptance.p_accept < 0.5 for found in search.results)


# --- proposals --------------------------------------------------------------------------------------------------------


def config_for(**policy: Any) -> Config:
    league = {"key": "nfl", "sport": "nfl", "espn_league_id": LEAGUE_ID, "season": SEASON, "team_id": US}
    return Config.model_validate({"league": [{**league, "policy": policy}]})


def test_the_best_deals_become_approve_only_trade_proposals(store: Store) -> None:
    ctx = context(store)
    search = find_trades(ctx, options=SearchOptions(runs=RUNS))
    outcomes = propose_trades(store, config_for(), ctx, search.results, max_offers=2)
    stored = [outcome for outcome in outcomes if outcome.proposal is not None]
    assert 1 <= len(stored) <= 2
    first = stored[0].proposal
    assert first is not None and first.kind == ProposalKind.TRADE_PROPOSE.value
    assert (first.status, first.policy, first.created_by) == ("proposed", "approve", TRADES_CREATED_BY)
    payload = parse_payload(first)
    assert isinstance(payload, TradePayload) and payload == search.results[0].spec.payload()
    assert first.scoring_period_id == WEEK and first.dedupe_key == f"nfl:{search.results[0].spec.key}"
    numbers = first.engine_numbers
    assert numbers["score_basis"] == BASIS_TITLE and numbers["acceptance"]["rank_type"] == "PPR"
    assert numbers["ours"]["delta_title"] == search.results[0].ours.delta_title
    assert "Give" in (first.rationale or "") and "P(accept)" in (first.rationale or "")
    assert len({outcome.evaluation.spec.other_team_id for outcome in stored}) == len(stored)  # one offer a team


def test_trades_are_approve_only_whatever_the_config_says(store: Store) -> None:
    ctx = context(store)
    outcome = propose_trades(
        store, config_for(), ctx, [evaluate_trade(ctx, swap_a_running_back_for_a_tight_end(), runs=RUNS)]
    )[0]
    assert outcome.proposal is not None and outcome.proposal.policy == "approve"
    with pytest.raises(ValueError, match="trade"):
        config_for(trade_propose="auto")


def test_one_open_offer_per_team_and_a_repeat_is_the_same_proposal(store: Store) -> None:
    ctx = context(store)
    first = evaluate_trade(ctx, swap_a_running_back_for_a_tight_end(), runs=RUNS)
    second = evaluate_trade(ctx, TradeSpec(2, (pid(US, "RB3"),), (pid(2, "TE2"),)), runs=RUNS)
    again = propose_trades(store, config_for(), ctx, [first])
    assert again[0].proposal is not None and not again[0].existing
    repeat = propose_trades(store, config_for(), ctx, [first, second])
    assert repeat[0].existing and repeat[0].proposal == again[0].proposal
    assert repeat[1].proposal is None and "one offer a team" in (repeat[1].blocked or "")
    assert len(store.proposals.open(ctx.league.row_id)) == 1


def test_a_dry_run_asks_policy_and_stores_nothing(store: Store) -> None:
    ctx = context(store)
    found = evaluate_trade(ctx, swap_a_running_back_for_a_tight_end(), runs=RUNS)
    outcomes = propose_trades(store, config_for(), ctx, [found], dry_run=True)
    assert outcomes[0].dry and outcomes[0].blocked is None and outcomes[0].proposal is None
    assert store.proposals.open(ctx.league.row_id) == []
    late = context(store, now=datetime(2027, 1, 1, tzinfo=UTC))
    blocked = propose_trades(
        store, config_for(), late, [evaluate_trade(late, swap_a_running_back_for_a_tight_end(), simulate=False)]
    )
    assert blocked[0].proposal is None and blocked[0].blocked is not None


def test_policy_refusals_are_reported_not_raised(store: Store) -> None:
    ctx = context(store)
    found = evaluate_trade(ctx, swap_a_running_back_for_a_tight_end(), runs=RUNS)
    pause("test")
    try:
        outcome = propose_trades(store, config_for(), ctx, [found])[0]
    finally:
        resume()
    assert outcome.proposal is None and outcome.blocked is not None and "paused" in outcome.blocked.lower()
    with pytest.raises(PolicyError):
        from fm.proposals import propose

        pause("test")
        try:
            propose(store, config_for(), ctx.league, ProposalKind.TRADE_PROPOSE, found.spec.payload(), created_by="x")
        finally:
            resume()


def test_deals_that_are_illegal_need_a_drop_or_do_not_help_are_skipped(store: Store) -> None:
    fillers = tuple((US, f"Filler {n}", "WR", BENCH, 7600 + n, 0.5) for n in range(2))
    ctx = context(store, extra=fillers)
    needs_drop = evaluate_trade(ctx, TradeSpec(2, (pid(US, "RB4"),), (pid(2, "TE1"), pid(2, "RB1"))), simulate=False)
    hurts = evaluate_trade(ctx, TradeSpec(2, (pid(US, "RB1"),), (pid(2, "TE2"),)), simulate=False)
    outcomes = propose_trades(store, config_for(), ctx, [needs_drop, hurts])
    assert [outcome.proposal for outcome in outcomes] == [None, None]
    assert "cannot carry" in (outcomes[0].blocked or "")
    assert store.proposals.open(ctx.league.row_id) == []


# --- reading a deal from text -----------------------------------------------------------------------------------------


def test_a_deal_is_read_from_names_and_ids(store: Store) -> None:
    ctx = context(store)
    spec = parse_trade_text(ctx, "give T3 RB2, T3 RB3 get T2 TE1")
    assert spec == TradeSpec(2, (pid(US, "RB2"), pid(US, "RB3")), (pid(2, "TE1"),))
    assert parse_trade_text(ctx, f"GIVE {pid(US, 'RB2')} for  {pid(2, 'TE1')}") == TradeSpec(
        2, (pid(US, "RB2"),), (pid(2, "TE1"),)
    )
    assert parse_trade_text(ctx, "give t3 rb2 get t2 te1").other_team_id == 2  # case and the unique part of a name


def test_a_deal_that_cannot_be_read_says_why(store: Store) -> None:
    ctx = context(store)
    with pytest.raises(TradeError, match="write it as: give A, B get C"):
        parse_trade_text(ctx, "trade my guys")
    with pytest.raises(TradeError, match="no player named 'Nobody' on our roster"):
        parse_trade_text(ctx, "give Nobody get T2 TE1")
    with pytest.raises(TradeError, match="no player named 'Nobody' on any other roster"):
        parse_trade_text(ctx, "give T3 RB2 get Nobody")
    with pytest.raises(TradeError, match="matches"):
        parse_trade_text(ctx, "give RB get T2 TE1")  # RB1, RB2, RB3, RB4 on our roster
    with pytest.raises(TradeError, match="all be on one team"):
        parse_trade_text(ctx, "give T3 RB2 get T2 TE1, T4 TE1")
    with pytest.raises(TradeError, match="is not on our roster"):
        parse_trade_text(ctx, f"give {pid(2, 'RB1')} get {pid(4, 'TE1')}")
    with pytest.raises(TradeError, match="both sides need a player"):
        parse_trade_text(ctx, "give , get T2 TE1")


# --- no write path ----------------------------------------------------------------------------------------------------


def test_the_module_has_no_way_to_write_to_espn() -> None:
    source = Path(trades_module.__file__).read_text(encoding="utf-8")
    imported = {
        alias.name if isinstance(node, ast.Import) else node.module or ""
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import | ast.ImportFrom)
        for alias in (node.names if isinstance(node, ast.Import) else [ast.alias(node.module or "")])
    }
    assert not {name for name in imported if name.startswith(("fm.executor", "fm.browser", "playwright", "httpx"))}
    assert not any(name.endswith("transport") for name in imported)


# --- the NBA ----------------------------------------------------------------------------------------------------------

NBA_SEASON, NBA_DAYS, NBA_US = 2027, 12, 1
NBA_NOW = datetime(2026, 10, 20, 15, 0, tzinfo=UTC)
NBA_TEAMS = (1, 2, 3, 4)
NBA_BENCH = FBA.bench_slot
NBA_SLOTS = {"PG": [0, 5, 11], "SG": [1, 5, 11], "SF": [2, 6, 11], "PF": [3, 6, 11], "C": [4, 11]}
NBA_POSITIONS = ("PG", "SG", "SF", "PF", "C", "PG", "SF")
"""Each team's seven players by position; the last two are bench players."""
NBA_GAMES = 70
"""Games in the season totals the stored ESPN lines carry (``GP``)."""


def nba_stats(team: int, slot: int) -> dict[str, float]:
    """A per-game line that is easy to reason about: better teams and earlier slots score more, and the categories
    differ by position (centers block and rebound, guards pass and steal) so categories matter."""
    quality = 1.3 - 0.1 * team - 0.08 * slot - (0.3 if slot >= 5 else 0.0)
    position = NBA_POSITIONS[slot]
    big = position in ("C", "PF")
    guard = position in ("PG", "SG")
    fga = 6 + 9 * quality
    return {
        "PTS": 8 + 16 * quality,
        "REB": (4 + 6 * quality) * (1.6 if big else 0.7),
        "AST": (1 + 5 * quality) * (1.8 if guard else 0.6),
        "STL": (0.3 + 0.9 * quality) * (1.4 if guard else 0.7),
        "BLK": (0.1 + 0.8 * quality) * (2.2 if big else 0.4),
        "TO": 0.8 + 1.6 * quality,
        "3PM": 0.2 + 1.8 * quality * (1.5 if guard else 0.3),
        "FGM": fga * (0.44 + 0.05 * quality + (0.06 if big else 0.0)),
        "FGA": fga,
        "FTM": 1 + 2.5 * quality,
        "FTA": 1.4 + 3 * quality * (1.3 if big else 0.9),
    }


def nba_id(team: int, slot: int) -> int:
    return 5000 + team * 10 + slot


def nba_settings_for(name: str, *, scoring: str | None = None) -> LeagueSettings:
    """A fixture NBA league cut to four teams and a short season: three matchup weeks of three days, then a final."""
    view = load(ESPN_FIXTURES / name)
    view["settings"]["size"] = len(NBA_TEAMS)
    view["settings"]["scheduleSettings"].update(
        {
            "matchupPeriodCount": 3,
            "playoffTeamCount": 2,
            "matchupPeriods": {"1": [1, 2, 3], "2": [4, 5, 6], "3": [7, 8, 9], "4": [10, 11, 12]},
        }
    )
    view["status"].update({"currentMatchupPeriod": 1, "finalScoringPeriod": NBA_DAYS})
    view["scoringPeriodId"] = 1
    if scoring is not None:
        view["settings"]["scoringSettings"]["scoringType"] = scoring
    return parse_league_settings(view)


def nba_schedule(days: int = NBA_DAYS, *, off: int = 7) -> ProSchedule:
    """Every pro team plays every day, except team ``off``, which plays on odd days only."""
    teams = []
    for team in range(1, 8):
        games = {
            str(day): [
                {
                    "id": team * 1000 + day,
                    "date": int((datetime(2026, 10, 19, 23, 0, tzinfo=UTC) + timedelta(days=day)).timestamp() * 1000),
                    "scoringPeriodId": day,
                    "homeProTeamId": team,
                    "awayProTeamId": 99,
                }
            ]
            for day in range(1, days + 1)
            if team != off or day % 2
        }
        teams.append({"id": team, "proGamesByScoringPeriod": games})
    return ProSchedule.model_validate({"proTeams": teams})


def nba_matchups() -> MatchupsView:
    entries = []
    for number, (period, home, away) in enumerate(round_robin(NBA_TEAMS, range(1, 4)), start=1):
        entries.append(
            {
                "id": number,
                "matchupPeriodId": period,
                "winner": "UNDECIDED",
                "home": {"teamId": home},
                "away": {"teamId": away},
            }
        )
    return MatchupsView.model_validate({"schedule": entries, "status": {"currentMatchupPeriod": 1}})


def espn_only() -> ProjectionSourceRegistry:
    registry = ProjectionSourceRegistry()
    registry.register("nba", ESPN, label="stored ESPN line", stored=True)
    return registry


def seed_nba(store: Store, settings: LeagueSettings, *, twins: bool = False, out: Iterable[int] = ()) -> LeagueRow:
    """Four NBA teams of seven, a few free agents, ESPN's stored season lines. ``twins`` gives team 1 and team 2 a pair
    of identical bench players (ids 7001 and 7002); ``out`` are ESPN ids whose injury status is OUT."""
    league = store.leagues.upsert(
        LeagueRow(
            key="nba",
            sport="nba",
            espn_league_id=settings.league_id,
            season=NBA_SEASON,
            team_id=NBA_US,
            as_of=NBA_NOW,
        )
    )
    store.settings.upsert(
        LeagueSettingsRow(league_id=league.row_id, settings=settings.model_dump(mode="json"), as_of=NBA_NOW)
    )
    store.teams.upsert_many(
        TeamRow(league_id=league.row_id, team_id=team, name=f"Team {team}", as_of=NBA_NOW) for team in NBA_TEAMS
    )
    people: list[tuple[int, int, str, dict[str, float], int]] = []  # id, team, position, stats, pro team
    for team in NBA_TEAMS:
        for slot, position in enumerate(NBA_POSITIONS):
            people.append((nba_id(team, slot), team, position, nba_stats(team, slot), 1 + (team + slot) % 6))
    free = [
        (8000 + n, 0, position, nba_stats(5, n + 2), 1 + n % 6) for n, position in enumerate(("PG", "SF", "C", "SG"))
    ]
    if twins:
        shared = nba_stats(3, 5)
        people.extend([(7001, 1, "SG", shared, 3), (7002, 2, "SG", shared, 3)])
    for team in NBA_TEAMS:
        store.rosters.replace(
            league.row_id,
            1,
            team,
            [
                RosterEntryRow(
                    league_id=league.row_id,
                    scoring_period_id=1,
                    team_id=team,
                    espn_id=espn_id,
                    lineup_slot_id=NBA_BENCH,
                    as_of=NBA_NOW,
                )
                for espn_id, owner, _, _, _ in people
                if owner == team
            ],
        )
    benched = set(out)
    store.players.upsert_many(
        PlayerRow(
            sport="nba",
            espn_id=espn_id,
            full_name=f"N{espn_id}",
            default_position_id=FBA.position_id(position),
            position=position,
            pro_team_id=pro,
            pro_team=f"T{pro}",
            eligible_slot_ids=[*NBA_SLOTS[position], NBA_BENCH, FBA.ir_slot],
            injury_status="OUT" if espn_id in benched else "ACTIVE",
            as_of=NBA_NOW,
        )
        for espn_id, _, position, _, pro in [*people, *free]
    )
    store.projections.upsert_many(
        ProjectionRow(
            sport="nba",
            espn_id=espn_id,
            source=ESPN,
            kind="projected",
            season=NBA_SEASON,
            scoring_period_id=0,
            stats={stat: value * NBA_GAMES for stat, value in stats.items()} | {"GP": float(NBA_GAMES)},
            as_of=NBA_NOW,
        )
        for espn_id, _, _, stats, _ in [*people, *free]
    )
    return league


class RankMarket:
    """A market that knows only ESPN's ranks (the NBA has no FantasyCalc), recording what it was asked."""

    def __init__(self, ranks: Mapping[int, int]) -> None:
        self.ranks = dict(ranks)
        self.calls: list[tuple[LeagueSettings, str]] = []

    def market_values(
        self, settings: LeagueSettings, *, rank_type: str = "STANDARD"
    ) -> Fetched[dict[int, MarketValue]]:
        self.calls.append((settings, rank_type))
        found = {
            espn_id: MarketValue(espn_id=espn_id, name=f"N{espn_id}", espn_ranks={rank_type: rank})
            for espn_id, rank in self.ranks.items()
        }
        return Fetched(found, NBA_NOW, "market", "values", "rank")


def nba_context(
    store: Store,
    name: str,
    *,
    scoring: str | None = None,
    matchups: bool = True,
    market: RankMarket | None = None,
    **seeded: Any,
) -> TradeContext:
    settings = nba_settings_for(name, scoring=scoring)
    league = seed_nba(store, settings, **seeded)
    return load_trade_context(
        store,
        league,
        now=NBA_NOW,
        schedule=nba_schedule(),
        matchups=nba_matchups() if matchups else None,
        market=market,
        weights=BlendWeights.load(),
        sources=espn_only(),
    )


def test_an_nba_points_league_is_valued_in_matchup_weeks(store: Store) -> None:
    ctx = nba_context(store, "fba_settings_points.json")
    assert ctx.model.unit == "points" and ctx.model.simulates
    assert isinstance(ctx.model, trades_module.PointsTradeModel)
    assert tuple(ctx.model.spans or ()) == (1, 2, 3, 4)  # every matchup week from the current one on
    assert all(ctx.model.values[nba_id(team, 0)] > ctx.model.values[nba_id(team, 6)] > 0 for team in NBA_TEAMS)
    # a team of better players is worth more: team 1's first player beats team 4's
    assert ctx.model.values[nba_id(1, 0)] > ctx.model.values[nba_id(4, 0)]
    assert ctx.model.roster_value(ctx.playing(1)) > ctx.model.roster_value(ctx.playing(4))
    assert ctx.market.rank_type == "STANDARD" and set(ctx.market.basis.values()) == {"ros_rank"}


def test_an_nba_points_trade_is_judged_on_the_season_simulation(store: Store) -> None:
    ctx = nba_context(store, "fba_settings_points.json")
    spec = TradeSpec(4, (nba_id(1, 6),), (nba_id(4, 0),))  # our bench scrub for their best player
    found = evaluate_trade(ctx, spec, runs=RUNS)
    assert found.simulated and found.legal and found.ours.delta_ros > 0 > found.theirs.delta_ros
    assert found.ours.delta_title is not None and found.ours.delta_title >= 0.0
    assert found.theirs.delta_title is not None and found.theirs.delta_title <= 0.0
    assert found.acceptance.p_accept < 0.5 and found.recommendation == ACCEPT


def test_nba_twins_are_a_wash_in_points_and_in_categories() -> None:
    for name in ("fba_settings_points.json", "fba_settings_9cat.json"):
        with Store.open(":memory:") as fresh:
            ctx = nba_context(fresh, name, twins=True)
            found = evaluate_trade(ctx, TradeSpec(2, (7001,), (7002,)), runs=RUNS)
            assert found.ours.delta_ros == pytest.approx(0.0, abs=1e-9), name
            assert found.theirs.delta_ros == pytest.approx(0.0, abs=1e-9), name
            assert found.ours.delta_title == pytest.approx(0.0, abs=1e-12), name
            assert found.simulated and found.score == pytest.approx(0.0, abs=1e-12)


def test_an_nba_category_league_values_players_by_what_they_add_to_each_category(store: Store) -> None:
    ctx = nba_context(store, "fba_settings_9cat.json")
    model = ctx.model
    assert isinstance(model, trades_module.CategoryTradeModel)
    assert model.unit == "G-score" and model.simulates
    assert "no game history to estimate the G-score's tau from, so G-scores equal z-scores" in ctx.warnings
    center, guard = nba_id(1, 4), nba_id(1, 0)
    assert model.player_profile(center)["BLK"] > model.player_profile(guard)["BLK"]
    assert model.player_profile(guard)["AST"] > model.player_profile(center)["AST"]
    profile = model.profile(ctx.playing(1))
    assert set(profile) == set(ctx.settings.categories)
    assert model.roster_value(ctx.playing(1)) > 0.0


def test_a_category_deal_says_which_categories_it_moves(store: Store) -> None:
    ctx = nba_context(store, "fba_settings_9cat.json")
    spec = TradeSpec(2, (nba_id(1, 4),), (nba_id(2, 0),))  # our center for their point guard
    found = evaluate_trade(ctx, spec, runs=RUNS)
    ups = next(note for note in found.fit if note.startswith("categories up"))
    downs = next(note for note in found.fit if note.startswith("categories down"))
    assert "AST" in ups and "BLK" in downs
    assert found.unit == "G-score" and found.simulated


def test_the_nba_finder_ranks_legal_deals_and_the_market_is_asked_for_the_rank_type(store: Store) -> None:
    ranks = {nba_id(team, slot): 1 + 7 * (team - 1) + slot for team in NBA_TEAMS for slot in range(7)}
    market = RankMarket(ranks)
    ctx = nba_context(store, "fba_settings_points.json", market=market)
    assert [rank_type for _, rank_type in market.calls] == ["STANDARD"]
    assert set(ctx.market.basis.values()) == {"espn_rank", "ros_rank"}  # the free agents have no rank
    search = find_trades(ctx, options=SearchOptions(runs=RUNS, min_gain=0.0))
    assert search.results
    scores = [found.score or 0.0 for found in search.results]
    assert scores == sorted(scores, reverse=True)
    assert all(found.legal for found in search.results)
    with Store.open(":memory:") as other:
        roto = nba_context(other, "fba_settings_9cat.json", scoring="ROTO", market=RankMarket(ranks))
        assert roto.market.rank_type == "ROTO"


def test_a_roto_league_is_judged_on_value_because_it_has_no_matchups_to_simulate(store: Store) -> None:
    ctx = nba_context(store, "fba_settings_9cat.json", scoring="ROTO")
    assert ctx.model.unit == "z-score" and not ctx.model.simulates
    found = evaluate_trade(ctx, TradeSpec(2, (nba_id(1, 6),), (nba_id(2, 0),)))
    assert not found.simulated and found.score_basis == BASIS_ROS and found.unit == "z-score"
    assert found.ours.delta_ros > 0


def test_the_nba_needs_the_pro_schedule_to_count_games(store: Store) -> None:
    settings = nba_settings_for("fba_settings_points.json")
    league = seed_nba(store, settings)
    with pytest.raises(TradeError, match="pro schedule, which is missing"):
        load_trade_context(store, league, now=NBA_NOW, weights=BlendWeights.load(), sources=espn_only())


def test_an_nba_league_whose_weeks_cannot_be_resolved_is_valued_for_the_whole_season(store: Store) -> None:
    view = load(ESPN_FIXTURES / "fba_settings_points.json")
    view["settings"]["size"] = len(NBA_TEAMS)
    view["settings"]["scheduleSettings"].update({"periodTypeId": 2, "matchupPeriodCount": 3, "playoffTeamCount": 2})
    view["status"].update({"currentMatchupPeriod": 1, "finalScoringPeriod": NBA_DAYS})
    settings = parse_league_settings(view)
    league = seed_nba(store, settings)
    ctx = load_trade_context(
        store,
        league,
        now=NBA_NOW,
        schedule=nba_schedule(),
        matchups=nba_matchups(),
        weights=BlendWeights.load(),
        sources=espn_only(),
    )
    assert not ctx.model.simulates
    assert any("matchup weeks cannot be resolved" in warning for warning in ctx.warnings)
    found = evaluate_trade(ctx, TradeSpec(2, (nba_id(1, 6),), (nba_id(2, 0),)))
    assert not found.simulated and found.ours.delta_ros > 0


def test_a_player_the_engine_has_ruled_out_is_flagged_and_never_asked_for(store: Store) -> None:
    ctx = nba_context(store, "fba_settings_points.json", out=[nba_id(4, 0)])
    store.availability.upsert(
        AvailabilityRow(
            sport="nba", espn_id=nba_id(4, 0), season=NBA_SEASON, scoring_period_id=1, p_active=0.0, as_of=NBA_NOW
        )
    )
    ctx = load_trade_context(
        store,
        ctx.league,
        now=NBA_NOW,
        schedule=nba_schedule(),
        matchups=ctx.matchups,
        weights=BlendWeights.load(),
        sources=espn_only(),
    )
    assert ctx.p_active[nba_id(4, 0)] == 0.0
    found = evaluate_trade(ctx, TradeSpec(4, (nba_id(1, 6),), (nba_id(4, 0),)), simulate=False)
    assert any("is ruled out now" in warning for warning in found.warnings)
    search = find_trades(ctx, opponents=[4], options=SearchOptions(runs=RUNS, min_gain=0.0, limit=50, finalists=50))
    assert all(nba_id(4, 0) not in found.spec.get for found in search.results)


# --- roster size, the weekly cap and the command line -----------------------------------------------------------------


def test_a_roster_already_past_the_limit_is_not_the_trades_doing(store: Store) -> None:
    fillers = tuple((2, f"Filler {n}", "WR", BENCH, 7700 + n, 1.0) for n in range(3))
    ctx = context(store, extra=fillers)  # team 2 holds 17 of 16 (a league whose bench is unlimited can do that)
    even = check_legality(ctx, TradeSpec(2, (pid(US, "RB4"),), (pid(2, "TE2"),)))
    assert even.legal and even.drops_theirs == ()
    more = check_legality(ctx, TradeSpec(2, (pid(US, "RB4"), pid(US, "WR4")), (pid(2, "TE2"),)))
    assert more.legal and len(more.drops_theirs) == 1  # one more than they held


def test_the_weekly_cap_counts_offers_already_made(store: Store) -> None:
    ctx = context(store)
    config = config_for()
    for team in (4, 5):
        spec = TradeSpec(team, (pid(US, "TE2"),), (pid(team, "TE2"),))
        propose_trades(store, config, ctx, [evaluate_trade(ctx, spec, runs=RUNS)])
    open_before = len(store.proposals.open(ctx.league.row_id))
    wanted = evaluate_trade(ctx, swap_a_running_back_for_a_tight_end(), runs=RUNS)
    capped = propose_trades(store, config, ctx, [wanted], weekly_cap=open_before)
    assert capped[0].proposal is None and "weekly cap of 2 offers" in (capped[0].blocked or "")
    allowed = propose_trades(store, config, ctx, [wanted], weekly_cap=open_before + 1)
    assert allowed[0].proposal is not None
    later = replace(ctx, now=NOW.replace(day=20))  # sixteen days on, those offers no longer count toward the week
    fresh = evaluate_trade(later, TradeSpec(6, (pid(US, "TE2"),), (pid(6, "TE2"),)), runs=RUNS)
    assert propose_trades(store, config, later, [fresh], weekly_cap=1, now=later.now)[0].proposal is not None


def write_config(*, untouchables: Iterable[str] = ()) -> None:
    """The league's ``config.toml`` in the test's private config dir, where the commands read it."""
    from fm import paths

    names = ", ".join(json.dumps(name) for name in untouchables)
    paths.config_dir().mkdir(parents=True, exist_ok=True)
    (paths.config_dir() / "config.toml").write_text(
        "\n".join(
            [
                "[[league]]",
                'key = "nfl"',
                'sport = "nfl"',
                f"espn_league_id = {LEAGUE_ID}",
                f"season = {SEASON}",
                f"team_id = {US}",
                "[league.policy]",
                f"untouchables = [{names}]",
                "",
            ]
        ),
        encoding="utf-8",
    )


@pytest.fixture
def matchups_file(tmp_path: Path) -> Iterator[Path]:
    """A config dir with ``config.toml`` and a synced six-team league in the state database, and the recorded
    ``mMatchup`` view (``--matchups``) beside it."""
    from fm import paths

    write_config()
    with Store.open() as opened:
        seed_league(opened)
    matchups = tmp_path / "matchups.json"
    matchups.write_text(json.dumps(schedule_json(ppr())), encoding="utf-8")
    yield matchups
    assert paths.config_dir().exists()


class _FakeSource:
    def __init__(self, market: FakeMarket) -> None:
        self.market = market

    def __enter__(self) -> FakeMarket:
        return self.market

    def __exit__(self, *exc: object) -> None:
        return None


def fm(*args: str, expect: int = 0) -> str:
    from fm.cli import app

    result = runner.invoke(app, list(args))
    assert result.exit_code == expect, result.output
    return result.output


AS_OF = "2026-10-04T15:00Z"


def test_the_trade_commands_are_discovered_and_have_help() -> None:
    assert "trade" in fm("--help")
    assert "eval" in fm("trade", "--help") and "find" in fm("trade", "--help")
    assert "give A, B get C" in fm("trade", "eval", "--help")
    assert "--propose" in fm("trade", "find", "--help")


def test_fm_trade_eval_judges_a_deal_named_by_players(matchups_file: Path) -> None:
    out = fm(
        "trade", "eval", "give T3 RB2 get T2 TE1", "--no-market", "--matchups", str(matchups_file), "--as-of", AS_OF,
        "--runs", "2000",
    )  # fmt: skip
    assert out.startswith("nfl: trade with Team 2 (as of 2026-10-04 15:00Z)")
    assert "give: T3 RB2 (RB)" in out and "get:  T2 TE1 (TE)" in out
    assert "ACCEPT: it improves our title odds" in out and "legal: yes" in out
    assert "Team 3" in out and "Team 2" in out and "title" in out
    assert "simulated 2000 seasons (seed 38)" in out and "P(accept)" in out
    assert "warning: no market source: P(accept) uses our own rest-of-season ranks" in out


def test_fm_trade_eval_without_matchups_says_the_numbers_are_values(matchups_file: Path) -> None:
    out = fm("trade", "eval", "give T3 RB2 get T2 TE1", "--no-market", "--as-of", AS_OF)
    assert "no season simulation: judged on rest-of-season value (points)" in out
    assert "warning: no mMatchup schedule" in out or "no mMatchup schedule is captured" in out


def test_fm_trade_eval_reports_what_it_cannot_read_or_do(matchups_file: Path) -> None:
    assert "no player named 'Nobody' on our roster" in fm(
        "trade", "eval", "give Nobody get T2 TE1", "--no-market", "--as-of", AS_OF, expect=1
    )
    assert "write it as: give A, B get C" in fm("trade", "eval", "hello", "--no-market", "--as-of", AS_OF, expect=1)
    late = fm("trade", "eval", "give T3 RB2 get T2 TE1", "--no-market", "--as-of", "2027-01-01T00:00Z")
    assert "DECLINE: not legal: the league's trade deadline" in late and "legal: NO" in late
    assert "no league 'nope' in config.toml" in fm(
        "trade", "eval", "give T3 RB2 get T2 TE1", "-l", "nope", "--no-market", expect=1
    )


def test_fm_trade_eval_asks_the_market_with_the_leagues_rank_type(
    matchups_file: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fm.commands import trade as trade_command

    market = FakeMarket()
    monkeypatch.setattr(trade_command, "market_source", lambda: _FakeSource(market))
    out = fm(
        "trade", "eval", "give T3 RB2 get T2 TE1", "--matchups", str(matchups_file), "--as-of", AS_OF, "--runs", "500"
    )
    assert [rank_type for _, rank_type in market.calls] == ["PPR"]
    assert "market 6,000 to them for 5,600 from them" in out
    assert "no market source" not in out


def test_fm_trade_find_ranks_deals_and_can_be_pointed_at_one_team(matchups_file: Path) -> None:
    out = fm(
        "trade",
        "find",
        "--no-market",
        "--matchups",
        str(matchups_file),
        "--as-of",
        AS_OF,
        "--top",
        "4",
        "--runs",
        "1500",
    )
    lines = out.splitlines()
    assert lines[0].startswith("nfl: trades for Team 3 (as of 2026-10-04 15:00Z); ") and "re-scored" in lines[0]
    table = [line for line in lines if line.strip().startswith(tuple("1234"))]
    assert len(table) == 4 and "score: our change in title odds" in out
    assert "P(accept)" in out and "with" in lines[1]
    only = fm(
        "trade",
        "find",
        "--no-market",
        "--matchups",
        str(matchups_file),
        "--as-of",
        AS_OF,
        "--with",
        "2",
        "--runs",
        "1500",
    )
    rows = [line for line in only.splitlines() if line.strip()[:1].isdigit()]
    assert rows and all("Team 2" in row for row in rows)
    assert fm("trade", "find", "--no-market", "--with", "99", "--as-of", AS_OF, expect=1).startswith("error:")


def test_fm_trade_find_propose_stores_approve_only_offers_and_dry_run_stores_nothing(matchups_file: Path) -> None:
    args = ("trade", "find", "--no-market", "--matchups", str(matchups_file), "--as-of", AS_OF, "--runs", "1500")
    dry = fm(*args, "--propose", "--dry-run", "--max-offers", "2")
    assert dry.count("would be proposed") == 2 and "--dry-run stored nothing" in dry
    with Store.open() as opened:
        assert opened.proposals.find() == []
    stored = fm(*args, "--propose", "--max-offers", "2")
    assert stored.count("proposed as #") == 2 and "fm proposals approve" in stored
    with Store.open() as opened:
        rows = opened.proposals.find()
    assert [(row.kind, row.status, row.policy, row.created_by) for row in rows] == [
        ("trade_propose", "proposed", "approve", TRADES_CREATED_BY)
    ] * 2
    again = fm(*args, "--propose", "--max-offers", "2")
    assert "already open as #" in again or "already has an open offer from us" in again
    assert "applies to --propose" in fm(*args, "--dry-run", expect=1)


def test_fm_trade_find_respects_the_configured_untouchables(matchups_file: Path) -> None:
    write_config(untouchables=["T3 RB1", "T3 RB2", "T3 RB3", "T3 QB2"])
    out = fm(
        "trade",
        "find",
        "--no-market",
        "--matchups",
        str(matchups_file),
        "--as-of",
        AS_OF,
        "--runs",
        "1500",
        "--top",
        "20",
    )
    assert "T3 RB1" not in out and "T3 RB2" not in out and "T3 RB3" not in out and "T3 QB2" not in out


def test_fm_trade_says_when_the_league_is_not_synced(tmp_path: Path) -> None:
    write_config()
    assert "nfl: not synced; run fm sync" in fm("trade", "find", "--no-market", "--as-of", AS_OF)
    assert "not synced; run fm sync" in fm("trade", "eval", "give a get b", "--no-market", "--as-of", AS_OF, expect=1)
