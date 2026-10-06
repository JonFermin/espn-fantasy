"""NFL waivers and free agents (ROADMAP #21): the wire, the FAAB heuristic, (add, drop) ranking with its guardrails
(untouchables, IR, unknown values, locks, position limits), timing to the league's waiver run, and the proposals the
decision stores through ``fm.proposals.propose``.

The wire comes from real captures: ``tests/fixtures/espn/real/ffl/kona_player_info.json`` (every player on waivers until
the Wednesday 3 a.m. ET run) and, end to end, the pool pages the real sync job (#16) captures from the hand-built PPR
league (``tests/fixtures/espn/ffl_*.json``, $100 FAAB) through an ``httpx.MockTransport``. Ranking runs on a hand-made
league whose numbers are visible here: a full 16-man roster in the PPR league's lineup (QB, 2 RB, 2 WR, TE, FLEX, D/ST,
K), each player projected a flat number of points a game, with weeks 15-17 (the playoffs) weighted 1.5.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fm import paths
from fm.config import Config, League, Policy
from fm.decide import registry
from fm.decide.faab import MIN_WINNING_BIDS, bid_strength, fit_bid_model, modeled_bid
from fm.decide.waivers import (
    POOL_KIND,
    STARTED_REASON,
    WAIVERS_CREATED_BY,
    WAIVERS_KIND,
    Bidding,
    WaiverError,
    WaiverMove,
    Wire,
    WireEntry,
    WireStatus,
    decide_waivers,
    heuristic_bid,
    load_wire,
    plan_moves,
    rank_moves,
)
from fm.espn.client import FILTER_HEADER, EspnClient
from fm.espn.ids import FFL
from fm.espn.models import PlayersView, PoolEntry, ProSchedule, Transaction
from fm.espn.settings import LeagueSettings, LockType, load_league_settings
from fm.jobs.sync import sync_league
from fm.model.projections import ESPN, BlendWeights
from fm.model.valuation import GAMES_STAT, Horizon, LeagueValuation, PlayerOutlook, load_valuation
from fm.proposals import (
    AddDropPayload,
    ProposalKind,
    WaiverPayload,
    evaluate,
    faab_bid_cap,
    parse_payload,
    pause,
    resume,
)
from fm.sports.base import EASTERN
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
ESPN_FIXTURES = FIXTURES / "espn"
PPR = ESPN_FIXTURES / "ffl_settings_ppr.json"
SEASON, WEEK, LEAGUE_ID, OUR_TEAM = 2026, 4, 1234567, 1
NOW = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)  # Sunday 11 a.m. ET, before the early games
CLEARS = datetime(2026, 10, 7, 7, 0, tzinfo=UTC)  # Wednesday 3 a.m. ET: the waiver run
STALE = datetime(2026, 10, 3, 7, 0, tzinfo=UTC)  # a run that has already happened
EQUAL_WEIGHTS = BlendWeights.parse("[nfl.default]\nespn = 1.0\nsleeper = 1.0\n\n[nba.default]\nespn = 1.0\n")

QB, RB, WR, TE, DST, K, FLEX = (FFL.slot_id(label) for label in ("QB", "RB", "WR", "TE", "D/ST", "K", "FLEX"))
BENCH, IR = FFL.bench_slot, FFL.ir_slot
SLOTS = {"QB": {QB}, "RB": {RB, FLEX}, "WR": {WR, FLEX}, "TE": {TE, FLEX}, "D/ST": {DST}, "K": {K}}

GONE, DUD, SPARE, BO, FLEX_BACK, ROB, HURT, MYSTERY = 13, 12, 17, 11, 7, 3, 14, 15
ROSTER: tuple[tuple[int, str, str, int, int, float | None, float | None], ...] = (
    # espn id, name, position, pro team, lineup slot, points a game, games ESPN projects (GP)
    (1, "Quinn Passer", "QB", 2, QB, 20, 14),
    (2, "Rex Runner", "RB", 4, RB, 16, 14),
    (ROB, "Rob Rusher", "RB", 8, RB, 13, 14),
    (4, "Will Wideout", "WR", 12, WR, 15, 14),
    (5, "Wes Wideout", "WR", 21, WR, 12, 14),
    (6, "Ty Tightend", "TE", 22, TE, 9, 14),
    (FLEX_BACK, "Flex Back", "RB", 25, FLEX, 8, 14),
    (-16021, "Eagles D/ST", "D/ST", 21, DST, 7, 14),
    (9, "Kai Kicker", "K", 3, K, 8, 14),
    (10, "Ben Bench", "WR", 26, BENCH, 6, 14),
    (BO, "Bo Bench", "RB", 27, BENCH, 4, 14),
    (DUD, "Dud Receiver", "WR", 28, BENCH, 2, 14),
    (GONE, "Gone Back", "RB", 29, BENCH, 0, 0),  # out for the season: known to be worth nothing
    (HURT, "Hurt Receiver", "WR", 30, IR, 3, 14),  # in the IR slot
    (MYSTERY, "Mystery Back", "RB", 33, BENCH, None, None),  # nothing projects him
    (16, "Backup Passer", "QB", 34, BENCH, 14, 14),
    (SPARE, "Spare End", "TE", 1, BENCH, 3, 14),
)
WALLY, FRED, TIM, LOU, DAN = 101, 102, 103, 104, 105
WIRE: tuple[tuple[int, str, str, int, float, WireStatus, datetime | None, bool], ...] = (
    # espn id, name, position, pro team, points a game, status, waiver run, game started
    (WALLY, "Wally Claim", "RB", 17, 13, WireStatus.WAIVERS, CLEARS, False),
    (FRED, "Fred Agent", "WR", 18, 14, WireStatus.FREE_AGENT, None, False),
    (TIM, "Tim Tightend", "TE", 19, 5, WireStatus.FREE_AGENT, None, False),
    (LOU, "Lou Locked", "RB", 20, 10, WireStatus.FREE_AGENT, None, True),
    (DAN, "Dan Done", "WR", 23, 11, WireStatus.WAIVERS, STALE, False),
)
ROSTER_IDS = tuple(row[0] for row in ROSTER)


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ppr() -> LeagueSettings:
    return load_league_settings(PPR)


def real_nfl() -> LeagueSettings:
    return load_league_settings(ESPN_FIXTURES / "real" / "ffl" / "mSettings.json")


def wire_of(rows: Iterable[tuple[int, str, str, int, float, WireStatus, datetime | None, bool]] = WIRE) -> Wire:
    entries = {row[0]: WireEntry(row[0], row[5], row[6], row[7]) for row in rows}
    return Wire(MappingProxyType(entries), NOW)


# --- the wire ---------------------------------------------------------------------------------------------------------


def real_pool() -> tuple[PoolEntry, ...]:
    return PlayersView.model_validate(load(ESPN_FIXTURES / "real" / "ffl" / "kona_player_info.json")).players


def test_the_real_pools_claims_are_timed_to_the_wednesday_waiver_run() -> None:
    pool = real_pool()
    wire = Wire.from_pool(pool, rostered=[pool[2].id], as_of=NOW)
    assert len(wire) == 3 and pool[2].id not in wire
    for entry in wire.entries.values():
        assert entry.status is WireStatus.WAIVERS
        assert entry.clears_at == CLEARS
    local = CLEARS.astimezone(EASTERN)
    assert (local.strftime("%A"), local.hour) == ("Wednesday", 3)


def test_the_wire_knows_free_agents_from_players_on_waivers() -> None:
    hand_built = PlayersView.model_validate(load(ESPN_FIXTURES / "ffl_free_agents_week4.json")).players
    rostered = hand_built[0].model_copy(update={"id": 1, "on_team_id": 3})
    odd = hand_built[1].model_copy(update={"id": 2, "status": "ONTEAM", "on_team_id": 0})
    wire = Wire.from_pool([*hand_built, rostered, odd])
    assert {espn_id: entry.status for espn_id, entry in wire.entries.items()} == {
        4038815: WireStatus.FREE_AGENT,
        4569173: WireStatus.WAIVERS,
        4361050: WireStatus.FREE_AGENT,
        4429160: WireStatus.FREE_AGENT,
    }
    assert wire.get(4569173) == WireEntry(4569173, WireStatus.WAIVERS, datetime(2026, 10, 5, 16, 40, tzinfo=UTC))
    free_agent = wire.get(4038815)
    assert free_agent is not None and free_agent.clears_at is None
    assert wire.warnings == ("1 pool players were neither free agents nor on waivers; left off the wire",)


# --- the real sync's captures -----------------------------------------------------------------------------------------


def espn_json(name: str) -> dict[str, Any]:
    return load(ESPN_FIXTURES / name)


def card_pool() -> dict[int, dict[str, Any]]:
    """Every player the NFL fixtures carry, by id, so a card request is answered per id."""
    cards: dict[int, dict[str, Any]] = {}
    for team in espn_json("ffl_rosters_week4.json")["teams"]:
        for entry in team["roster"]["entries"]:
            cards[entry["playerId"]] = {
                **entry["playerPoolEntry"],
                "id": entry["playerId"],
                "onTeamId": team["id"],
                "status": "ONTEAM",
            }
    for entry in [
        *espn_json("ffl_free_agents_week4.json")["players"],
        *espn_json("ffl_player_cards_week4.json")["players"],
    ]:
        cards[entry["id"]] = entry
    return cards


def espn_views(request: httpx.Request) -> httpx.Response:
    """The hand-built PPR league's views, as ESPN would answer them (only reads exist here)."""
    views = "+".join(request.url.params.get_list("view"))
    query = json.loads(request.headers.get(FILTER_HEADER, "{}")).get("players", {})
    if views == "mSettings":
        body = espn_json("ffl_settings_ppr.json")
    elif views == "mTeam+mStandings":
        body = espn_json("ffl_teams.json")
    elif views == "mRoster":
        body = espn_json("ffl_rosters_week4.json")
    elif views == POOL_KIND:
        body = espn_json("ffl_free_agents_week4.json") if not query.get("offset") else {"players": []}
    elif views == "kona_playercard":
        cards = card_pool()
        body = {"players": [cards[espn_id] for espn_id in query["filterIds"]["value"] if espn_id in cards]}
    else:
        return httpx.Response(404, json={"details": [{"type": "UNEXPECTED", "message": views}]})
    return httpx.Response(200, json=body)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        yield opened


NO_BID_MODEL = "nfl: no FAAB bid model (only 0 winning bids in the history (need 5)); bids use the heuristic"
"""The warning a $100 FAAB league without captured transaction history gets: its claims bid the heuristic."""


def fixture_config(**policy: Any) -> Config:
    league = {"key": "nfl", "sport": "nfl", "espn_league_id": LEAGUE_ID, "season": SEASON, "team_id": OUR_TEAM}
    return Config.model_validate({"league": [{**league, "policy": policy}]})


def synced(store: Store, config: Config) -> LeagueRow:
    """The hand-built league synced by the real sync job, its reads captured under the test's cache dir."""
    transport = httpx.Client(transport=httpx.MockTransport(espn_views))
    client = EspnClient.for_league(config.league("nfl"), None, client=transport, min_interval_s=0.0)
    sync_league(store, config.league("nfl"), client)
    row = store.leagues.by_key("nfl")
    assert row is not None
    return row


def test_the_wire_is_read_back_from_the_pool_pages_the_sync_captured(store: Store) -> None:
    league = synced(store, fixture_config())
    wire = load_wire(store, league)
    assert wire.warnings == ()
    assert set(wire.entries) == {4038815, 4569173, 4361050, 4429160}
    assert wire.get(4569173) == WireEntry(4569173, WireStatus.WAIVERS, datetime(2026, 10, 5, 16, 40, tzinfo=UTC))
    assert wire.as_of is not None


def test_an_older_syncs_pool_page_is_not_the_wire(store: Store) -> None:
    league = synced(store, fixture_config())
    old = store.raw_snapshots.find(ESPN, POOL_KIND, league_id=league.row_id)[0]
    stale = {"players": [{"id": 77, "status": "FREEAGENT", "onTeamId": 0, "player": {"id": 77, "fullName": "Old"}}]}
    (paths.cache_dir() / "old_pool.json").write_text(json.dumps(stale), encoding="utf-8")
    earlier = old.fetched_at - timedelta(days=2)
    store.raw_snapshots.insert(old.model_copy(update={"id": None, "path": "old_pool.json", "fetched_at": earlier}))
    assert 77 not in load_wire(store, league)


def test_a_pool_page_gone_from_the_cache_is_reported(store: Store) -> None:
    league = synced(store, fixture_config())
    page = store.raw_snapshots.find(ESPN, POOL_KIND, league_id=league.row_id)[0]
    (paths.cache_dir() / page.path).unlink()
    wire = load_wire(store, league)
    assert len(wire) == 0
    assert wire.warnings == (f"nfl: pool page {page.path} is gone from the cache; run fm sync",)


def test_without_a_captured_pool_the_wire_is_empty_and_says_why(store: Store) -> None:
    league = synced(store, fixture_config())
    assert load_wire(store, league, scoring_period=5).warnings == (
        "nfl: no free-agent pool captured for scoring period 5; run fm sync",
    )
    other = store.leagues.upsert(league.model_copy(update={"id": None, "key": "other", "espn_league_id": 99}))
    assert load_wire(store, other).warnings == ("other: no roster snapshot; run fm sync",)


def test_the_synced_league_proposes_a_claim_timed_to_its_run_and_an_add_before_kickoff(store: Store) -> None:
    config = fixture_config(untouchables=["Josh Allen"])
    league = synced(store, config)
    schedule = ProSchedule.model_validate(load(FIXTURES / "sports" / "ffl_pro_schedule_2026.json"))
    decision = decide_waivers(store, config, league, now=NOW, schedule=schedule, weights=EQUAL_WEIGHTS)
    assert decision.warnings == (NO_BID_MODEL,)
    claim, add = decision.moves
    assert (claim.kind, claim.add.name, claim.drop) == (ProposalKind.WAIVER, "Jaylen Warren", None)
    assert claim.deadline == claim.clears_at == datetime(2026, 10, 5, 16, 40, tzinfo=UTC)  # waiverProcessDate
    assert claim.start == WEEK + 1  # Pittsburgh played Thursday night, before the claim clears
    cap = faab_bid_cap(config.league("nfl").policy, ppr())
    assert cap == 35 and claim.bid is not None and 0 < claim.bid <= cap
    assert (add.kind, add.add.name) == (ProposalKind.ADD_DROP, "Rico Dowdle")
    assert add.start == WEEK  # Carolina plays Sunday night, so he can still count this week
    assert add.deadline == datetime(2026, 10, 5, 0, 20, tzinfo=UTC)  # Carolina's kickoff: then he goes on waivers
    assert [row.kind for row in decision.proposals] == ["waiver", "add_drop"]
    assert parse_payload(decision.proposals[0]) == WaiverPayload(add_espn_id=4569173, bid_amount=claim.bid)
    assert all(row.status == "proposed" and row.created_by == WAIVERS_CREATED_BY for row in decision.proposals)
    rerun = decide_waivers(store, config, league, now=NOW, schedule=schedule, weights=EQUAL_WEIGHTS)
    assert rerun.moves == () and rerun.proposals == ()  # both moves are pending: nothing new is worth it
    assert len(store.proposals.open(league.row_id)) == 2


# --- FAAB -------------------------------------------------------------------------------------------------------------


@settings(max_examples=300, deadline=None)
@given(
    gain=st.floats(min_value=-50, max_value=5000, allow_nan=False),
    roster_value=st.floats(min_value=0, max_value=5000, allow_nan=False),
    budget_left=st.integers(min_value=0, max_value=1000),
    cap=st.integers(min_value=0, max_value=1000),
    minimum=st.integers(min_value=0, max_value=5),
)
def test_a_bid_never_exceeds_the_cap_or_the_budget(
    gain: float, roster_value: float, budget_left: int, cap: int, minimum: int
) -> None:
    bid = heuristic_bid(gain, roster_value=roster_value, budget_left=budget_left, cap=cap, minimum_bid=minimum)
    if min(cap, budget_left) < minimum:
        assert bid is None
    else:
        assert bid is not None and minimum <= bid <= min(cap, budget_left)


@settings(max_examples=200, deadline=None)
@given(
    gains=st.lists(st.floats(min_value=0, max_value=2000, allow_nan=False), min_size=2, max_size=6),
    roster_value=st.floats(min_value=1, max_value=5000, allow_nan=False),
    budget_left=st.integers(min_value=0, max_value=1000),
    cap=st.integers(min_value=0, max_value=1000),
)
def test_a_bid_grows_with_the_gain(gains: list[float], roster_value: float, budget_left: int, cap: int) -> None:
    bids = [heuristic_bid(gain, roster_value=roster_value, budget_left=budget_left, cap=cap) for gain in sorted(gains)]
    assert all(bid is not None for bid in bids)
    assert bids == sorted(bid or 0 for bid in bids)


def test_a_bid_is_the_gains_share_of_a_quarter_of_the_roster_value() -> None:
    assert heuristic_bid(50, roster_value=1000, budget_left=80, cap=35) == 16  # 80 * 50 / 250
    assert heuristic_bid(500, roster_value=1000, budget_left=80, cap=35) == 35  # held to the cap
    assert heuristic_bid(0, roster_value=1000, budget_left=80, cap=35, minimum_bid=1) == 1
    assert heuristic_bid(10, roster_value=1000, budget_left=0, cap=35, minimum_bid=1) is None
    with pytest.raises(ValueError, match="budget_left"):
        heuristic_bid(10, roster_value=1000, budget_left=-1, cap=35)
    with pytest.raises(ValueError, match="share"):
        heuristic_bid(10, roster_value=1000, budget_left=80, cap=35, share=0.0)
    with pytest.raises(ValueError, match="finite"):
        heuristic_bid(10, roster_value=float("inf"), budget_left=80, cap=35)


def test_the_bid_cap_is_the_policys_share_of_the_synced_budget() -> None:
    policy = Policy(max_faab_pct_per_bid=0.35)
    bidding = Bidding.for_league(ppr(), policy, spent=20)
    assert bidding == Bidding(budget_left=80, cap=35, minimum_bid=0)
    assert bidding is not None and bidding.cap == faab_bid_cap(policy, ppr())
    assert Bidding.for_league(ppr(), policy, spent=120) == Bidding(budget_left=0, cap=35, minimum_bid=0)
    assert Bidding.for_league(real_nfl(), policy, spent=0) is None  # the real league claims by priority


# --- ranking and planning ---------------------------------------------------------------------------------------------


def flat(espn_id: int, name: str, position: str, team: int, points: float | None, games: float | None) -> PlayerOutlook:
    """A player projected ``points`` a game every week of the season (``None``: nothing projects him)."""
    season = Horizon.for_league(ppr(), current=WEEK)
    weekly = dict.fromkeys(season.periods, 0.0 if points is None or not games else float(points))
    return PlayerOutlook(
        espn_id=espn_id,
        name=name,
        position=position,
        pro_team_id=team,
        slots=frozenset(SLOTS[position]),
        weekly=MappingProxyType(weekly),
        per_game=points or 0.0,
        games=games,
        basis="none" if points is None else "season",
    )


def player_row(espn_id: int, name: str, position: str, team: int) -> PlayerRow:
    return PlayerRow(
        sport="nfl",
        espn_id=espn_id,
        full_name=name,
        default_position_id=FFL.position_id(position),
        position=position,
        pro_team_id=team,
        eligible_slot_ids=[*SLOTS[position], BENCH, IR],
        as_of=NOW,
    )


LEAGUE = LeagueRow(id=1, key="nfl", sport="nfl", espn_league_id=LEAGUE_ID, season=SEASON, team_id=OUR_TEAM, as_of=NOW)


def valuation(
    roster: Iterable[tuple[int, str, str, int, int, float | None, float | None]] = ROSTER,
    wire: Iterable[tuple[int, str, str, int, float, WireStatus, datetime | None, bool]] = WIRE,
    *,
    league_settings: LeagueSettings | None = None,
    locked: Iterable[int] = (),
) -> LeagueValuation:
    chosen = league_settings if league_settings is not None else ppr()
    held, pool = tuple(roster), tuple(wire)
    outlooks = {row[0]: flat(row[0], row[1], row[2], row[3], row[5], row[6]) for row in held}
    outlooks.update((row[0], flat(row[0], row[1], row[2], row[3], row[4], 14)) for row in pool)
    players = {row[0]: player_row(row[0], row[1], row[2], row[3]) for row in (*held, *pool)}
    stuck = set(locked)
    team = tuple(
        RosterEntryRow(
            league_id=1,
            scoring_period_id=WEEK,
            team_id=OUR_TEAM,
            espn_id=row[0],
            lineup_slot_id=row[4],
            lineup_locked=row[0] in stuck,
            as_of=NOW,
        )
        for row in held
    )
    return LeagueValuation(
        league=LEAGUE,
        settings=chosen,
        horizon=Horizon.for_league(chosen, current=WEEK),
        outlooks=MappingProxyType(outlooks),
        players=MappingProxyType(players),
        team=team,
        rostered=frozenset(row[0] for row in held),
        wire=frozenset(row[0] for row in pool),
        as_of=NOW,
    )


WEEKS_FROM_4 = 11 + 3 * 1.5  # weeks 4-14, then the weeks 15-17 playoffs at 1.5
WEEKS_FROM_5 = 10 + 3 * 1.5
WEEKS_FROM_6 = 9 + 3 * 1.5


def pairs(moves: Iterable[WaiverMove]) -> list[tuple[int, int | None]]:
    return [(move.add.espn_id, move.drop.espn_id if move.drop is not None else None) for move in moves]


def test_pairs_are_ranked_by_the_gain_in_rest_of_season_value() -> None:
    moves = rank_moves(valuation(), wire_of(), now=NOW)
    best = moves[0]
    assert pairs([best]) == [(FRED, GONE)]  # equal gains drop the least valuable player first
    assert best.kind is ProposalKind.ADD_DROP and best.start == WEEK
    assert best.gain == pytest.approx((14 - 8) * WEEKS_FROM_4)  # Fred starts; Flex Back's 8 leaves the lineup
    claim = next(move for move in moves if move.add.espn_id == WALLY)
    assert claim.kind is ProposalKind.WAIVER and claim.start == WEEK + 1  # the run is after this week's games
    assert claim.gain == pytest.approx((13 - 8) * WEEKS_FROM_5)
    assert [move.gain for move in moves] == sorted((move.gain for move in moves), reverse=True)
    starter_drop = next(move for move in moves if pairs([move]) == [(FRED, ROB)])
    assert starter_drop.gain == pytest.approx((14 - 13) * WEEKS_FROM_4)  # Fred's 14 for Rob's 13 at a flex spot
    assert starter_drop.gain < best.gain


def test_with_a_full_roster_every_add_comes_with_a_drop_and_an_open_spot_needs_none() -> None:
    assert all(move.drop is not None for move in rank_moves(valuation(), wire_of(), now=NOW))
    roomy = rank_moves(valuation(ROSTER[:-1]), wire_of(), now=NOW)
    assert pairs(roomy[:1]) == [(FRED, None)]  # with a spot open, keeping everyone ranks first


@settings(max_examples=60, deadline=None)
@given(protected=st.sets(st.sampled_from(ROSTER_IDS)))
def test_untouchables_are_never_dropped(protected: set[int]) -> None:
    league = valuation()
    moves = rank_moves(league, wire_of(), now=NOW, protected=protected)
    planned = plan_moves(league, wire_of(), now=NOW, protected=protected, max_moves=3)
    for move in (*moves, *planned):
        assert move.drop is None or move.drop.espn_id not in protected


def test_ir_and_unprojected_players_are_never_dropped() -> None:
    dropped = {move.drop.espn_id for move in rank_moves(valuation(), wire_of(), now=NOW) if move.drop is not None}
    assert HURT not in dropped  # in the IR slot: dropping him frees no roster spot
    assert MYSTERY not in dropped  # nothing projects him: his value is unknown, not zero
    assert GONE in dropped  # out for the season: known to be worth nothing


def test_a_locked_player_cannot_be_dropped_for_an_add_but_can_for_next_weeks_claim() -> None:
    moves = rank_moves(valuation(locked=[GONE]), wire_of(), now=NOW)
    assert all(move.drop is None or move.drop.espn_id != GONE for move in moves if move.kind is ProposalKind.ADD_DROP)
    assert any(pairs([move]) == [(WALLY, GONE)] for move in moves)


def test_the_leagues_position_limits_hold() -> None:
    limits = {**ppr().position_limits, FFL.position_id("WR"): 5}  # we hold five receivers already
    capped = ppr().model_copy(update={"position_limits": limits})
    moves = rank_moves(valuation(league_settings=capped), wire_of(), now=NOW)
    receivers = {row[0] for row in ROSTER if row[2] == "WR"}
    fred = [move for move in moves if move.add.espn_id == FRED]
    assert fred and all(move.drop is not None and move.drop.espn_id in receivers for move in fred)


def test_moves_the_policy_turns_off_or_cannot_time_are_skipped() -> None:
    claims_only = rank_moves(valuation(), wire_of(), now=NOW, kinds=[ProposalKind.WAIVER])
    assert {move.add.espn_id for move in claims_only} == {WALLY}  # Dan's run has already passed
    adds_only = rank_moves(valuation(), wire_of(), now=NOW, kinds=[ProposalKind.ADD_DROP])
    assert {move.add.espn_id for move in adds_only} == {FRED, TIM}  # Lou's game has started
    undated = wire_of([(WALLY, "Wally Claim", "RB", 17, 13, WireStatus.WAIVERS, None, False)])
    assert rank_moves(valuation(), undated, now=NOW) == ()


def test_a_free_agent_whose_game_has_started_is_not_proposed_until_the_period_turns(store: Store) -> None:
    """The roster lock closes his add for the period (``transaction_cutoff``; the executor checks it per player), and
    the real league has him on waivers by then, so an add would hold a transaction slot for a move that cannot run:
    whether the pool says so (``lineupLocked``) or his kickoff passed after the pool was read."""
    assert all(move.add.espn_id != LOU for move in rank_moves(valuation(), wire_of(), now=NOW))
    thursday = schedule({(18, WEEK): THURSDAY})  # Fred's team played Thursday night, after the pool was read
    timed = rank_moves(valuation(), wire_of(), now=NOW, schedule=thursday)
    assert all(move.add.espn_id != FRED for move in timed)
    assert all(move.start == move.scoring_period == WEEK for move in timed if move.kind is ProposalKind.ADD_DROP)
    seed(store)
    decision = decide(store, fixture_config(), store_proposals=False)
    assert all(move.add.espn_id != LOU for move in decision.ranked)
    assert f"nfl: 1 wire candidates skipped: {STARTED_REASON}" in decision.warnings


def test_a_league_whose_roster_lock_type_is_not_mapped_gets_no_move(store: Store) -> None:
    """ESPN's weekly lock types parse as ``UNKNOWN`` and the plugins refuse to say when adds and drops close (so will
    the executor, ROADMAP #27): rather than time moves to a guessed kickoff, every candidate is turned away."""
    weekly = ppr().model_copy(update={"roster_lock_type": LockType.UNKNOWN, "roster_lock_type_raw": "FIRSTGAME_WEEKLY"})
    assert rank_moves(valuation(league_settings=weekly), wire_of(), now=NOW, schedule=schedule()) == ()
    assert rank_moves(valuation(league_settings=weekly), wire_of(), now=NOW) == ()
    seed(store, league_settings=weekly)
    decision = decide(store, fixture_config())
    assert decision.ranked == () and decision.moves == () and decision.proposals == ()
    assert decision.warnings == (
        NO_BID_MODEL,
        "nfl: 5 wire candidates skipped: the league's roster lock type FIRSTGAME_WEEKLY is not mapped, so when adds "
        "and drops close is unknown",
    )


def test_claims_bid_from_the_budget_left_within_the_cap() -> None:
    bidding = Bidding(budget_left=80, cap=35)
    moves = rank_moves(valuation(), wire_of(), now=NOW, bidding=bidding)
    for move in moves:
        if move.kind is ProposalKind.WAIVER:
            assert move.roster_value is not None and move.bidding == bidding
            assert move.bid == heuristic_bid(move.gain, roster_value=move.roster_value, budget_left=80, cap=35)
            assert move.bid is not None and move.bid <= bidding.cap
        else:
            assert move.bid is None and move.bidding is None
    unaffordable = rank_moves(valuation(), wire_of(), now=NOW, bidding=Bidding(budget_left=0, cap=35, minimum_bid=1))
    assert all(move.kind is ProposalKind.ADD_DROP for move in unaffordable)


def test_a_later_claim_bids_from_what_the_earlier_ones_leave() -> None:
    cal = (106, "Cal Claim", "WR", 24, 14, WireStatus.WAIVERS, CLEARS, False)
    wire = [WIRE[0], cal]
    bidding = Bidding(budget_left=80, cap=35)
    first, second = plan_moves(valuation(wire=wire), wire_of(wire), now=NOW, bidding=bidding, max_moves=2)
    assert (first.add.espn_id, second.add.espn_id) == (106, WALLY)
    assert first.bid is not None and 0 < first.bid <= 35 and first.bidding == bidding
    assert second.bidding == Bidding(budget_left=80 - first.bid, cap=35)
    assert second.bid is not None and second.bid <= 35


def test_the_plan_takes_moves_one_at_a_time_without_moving_a_player_twice() -> None:
    plan = plan_moves(valuation(), wire_of(), now=NOW, max_moves=3)
    assert pairs(plan) == [(FRED, GONE), (WALLY, DUD)]
    assert plan[1].gain == pytest.approx((13 - 12) * WEEKS_FROM_5)  # after Fred, Wally only beats Wes at FLEX
    assert plan_moves(valuation(), wire_of(), now=NOW, max_moves=1) == plan[:1]
    assert plan_moves(valuation(), wire_of(), now=NOW, max_moves=3, min_gain=1000) == ()
    assert plan_moves(valuation(), wire_of(), now=NOW, max_moves=0) == ()


def test_a_claim_landing_next_week_does_not_block_an_add_for_this_week() -> None:
    wire = [WIRE[0], (FRED, "Fred Agent", "WR", 18, 11, WireStatus.FREE_AGENT, None, False)]
    plan = plan_moves(valuation(wire=wire), wire_of(wire), now=NOW, max_moves=2, min_gain=1.0)
    assert pairs(plan) == [(WALLY, GONE), (FRED, DUD)]
    assert plan[1].gain == pytest.approx(11 - 8)  # this week only: from next week Wally holds FLEX


# --- timing from the pro schedule -------------------------------------------------------------------------------------


def sunday(period: int) -> datetime:
    return datetime(2026, 10, 4, 17, 0, tzinfo=UTC) + timedelta(weeks=period - WEEK)


def schedule(
    starts: Mapping[tuple[int, int], datetime] | None = None, *, byes: Iterable[tuple[int, int]] = ()
) -> ProSchedule:
    """Every pro team plays at 1 p.m. ET each Sunday of weeks 4-18, but for ``starts`` ((team, period) -> kickoff) and
    ``byes`` ((team, period) pairs without a game); team 34 plays Monday night of week 4."""
    special = {(34, WEEK): datetime(2026, 10, 6, 0, 15, tzinfo=UTC), **(starts or {})}
    off = frozenset(byes)
    teams = [
        {
            "id": team,
            "proGamesByScoringPeriod": {
                str(period): [
                    {
                        "id": team * 100 + period,
                        "date": int(special.get((team, period), sunday(period)).timestamp() * 1000),
                        "scoringPeriodId": period,
                        "homeProTeamId": team,
                        "awayProTeamId": 0,
                    }
                ]
                for period in range(WEEK, 19)
                if (team, period) not in off
            },
        }
        for team in range(1, 35)
    ]
    return ProSchedule.model_validate({"proTeams": teams})


THURSDAY = datetime(2026, 10, 2, 0, 15, tzinfo=UTC)  # week 4's Thursday night game
NEXT_THURSDAY = datetime(2026, 10, 9, 0, 15, tzinfo=UTC)
WEEK_4_ENDS = datetime(2026, 10, 6, 7, 0, tzinfo=UTC)  # 3 a.m. ET after Monday night


def test_a_claim_is_timed_to_its_waiver_run() -> None:
    moves = rank_moves(valuation(), wire_of(), now=NOW, schedule=schedule())
    claim = next(move for move in moves if move.add.espn_id == WALLY)
    assert (claim.deadline, claim.scoring_period, claim.start) == (CLEARS, WEEK + 1, WEEK + 1)
    late = wire_of(
        [(WALLY, "Wally Claim", "RB", 17, 13, WireStatus.WAIVERS, NEXT_THURSDAY + timedelta(hours=7), False)]
    )
    thursday = schedule({(17, WEEK + 1): NEXT_THURSDAY})
    played = next(iter(rank_moves(valuation(), late, now=NOW, schedule=thursday)))
    assert (played.scoring_period, played.start) == (WEEK + 1, WEEK + 2)  # his week-5 game is over when it clears


def test_an_add_must_beat_the_add_and_the_drop_to_kickoff() -> None:
    early_drop = schedule({(29, WEEK): sunday(WEEK) - timedelta(hours=1)})  # Gone Back's team kicks off at noon
    moves = rank_moves(valuation(), wire_of(), now=NOW, schedule=early_drop)
    fred = next(move for move in moves if pairs([move]) == [(FRED, GONE)])
    assert fred.deadline == sunday(WEEK) - timedelta(hours=1)
    alone = next(move for move in moves if pairs([move]) == [(FRED, DUD)])
    assert alone.deadline == sunday(WEEK)  # Fred's own kickoff


def test_an_add_with_no_kickoff_to_beat_is_held_to_the_periods_end() -> None:
    bye = schedule(byes=[(18, WEEK)])  # Fred's team sits out week 4
    roomy = rank_moves(valuation(ROSTER[:-1]), wire_of(), now=NOW, schedule=bye)
    alone = next(move for move in roomy if pairs([move]) == [(FRED, None)])
    assert alone.deadline == WEEK_4_ENDS  # 3 a.m. ET after Monday night, when ESPN turns the week
    assert (alone.scoring_period, alone.start) == (WEEK, WEEK)


def test_a_drop_whose_game_has_started_waits_for_a_claim_in_a_later_week() -> None:
    thursday_drop = schedule({(29, WEEK): THURSDAY})
    moves = rank_moves(valuation(), wire_of(), now=NOW, schedule=thursday_drop)
    assert not any(pairs([move]) == [(FRED, GONE)] for move in moves)
    assert any(pairs([move]) == [(WALLY, GONE)] for move in moves)


def test_dropping_a_starter_for_a_claim_that_counts_later_costs_his_games_in_between() -> None:
    late = wire_of(
        [(WALLY, "Wally Claim", "RB", 17, 13, WireStatus.WAIVERS, NEXT_THURSDAY + timedelta(hours=7), False)]
    )
    thursday = schedule({(17, WEEK + 1): NEXT_THURSDAY})  # Wally plays Thursday night of week 5; the run is Friday
    moves = {pairs([move])[0]: move for move in rank_moves(valuation(), late, now=NOW, schedule=thursday)}
    assert (moves[(WALLY, GONE)].scoring_period, moves[(WALLY, GONE)].start) == (WEEK + 1, WEEK + 2)
    assert moves[(WALLY, GONE)].gain == pytest.approx((13 - 8) * WEEKS_FROM_6)
    # Dropped when the claim is processed, Flex Back misses his week-5 game at FLEX, where Ben's 6 stands in.
    assert moves[(WALLY, FLEX_BACK)].gain == pytest.approx((13 - 8) * WEEKS_FROM_6 - (8 - 6))


# --- the decision from the store --------------------------------------------------------------------------------------


def line(points: float) -> dict[str, float]:
    return {"REY": points * 10}  # 0.1 a receiving yard in the PPR league


def seed(
    store: Store,
    *,
    league_settings: LeagueSettings | None = None,
    spent: int = 20,
    wire: Iterable[tuple[int, str, str, int, float, WireStatus, datetime | None, bool]] = WIRE,
) -> LeagueRow:
    """The hand-made roster and wire as ``fm sync`` would store them: ESPN's week line and season line (``GP`` 14)."""
    pool = tuple(wire)
    league = store.leagues.upsert(
        LeagueRow(key="nfl", sport="nfl", espn_league_id=LEAGUE_ID, season=SEASON, team_id=OUR_TEAM, as_of=NOW)
    )
    chosen = league_settings if league_settings is not None else ppr()
    store.settings.upsert(
        LeagueSettingsRow(league_id=league.row_id, settings=chosen.model_dump(mode="json"), as_of=NOW)
    )
    store.teams.upsert(
        TeamRow(league_id=league.row_id, team_id=OUR_TEAM, name="Us", acquisition_budget_spent=spent, as_of=NOW)
    )
    store.rosters.replace(
        league.row_id,
        WEEK,
        OUR_TEAM,
        [
            RosterEntryRow(
                league_id=league.row_id,
                scoring_period_id=WEEK,
                team_id=OUR_TEAM,
                espn_id=row[0],
                lineup_slot_id=row[4],
                as_of=NOW,
            )
            for row in ROSTER
        ],
    )
    players = [player_row(row[0], row[1], row[2], row[3]) for row in (*ROSTER, *pool)]
    store.players.upsert_many(players)
    lines: list[ProjectionRow] = []
    for espn_id, points, games in [(row[0], row[5], row[6]) for row in ROSTER] + [(row[0], row[4], 14) for row in pool]:
        if points is None or games is None:
            continue
        for period, stats in ((WEEK, line(points if games else 0)), (0, {**line(points * games), GAMES_STAT: games})):
            lines.append(
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
    store.projections.upsert_many(lines)
    return league


def decide(store: Store, config: Config, **options: Any) -> Any:
    return decide_waivers(
        store, config, "nfl", now=NOW, wire=options.pop("wire", wire_of()), weights=EQUAL_WEIGHTS, **options
    )


def test_the_store_backed_league_values_like_the_hand_made_one(store: Store) -> None:
    league = seed(store)
    stored = load_valuation(store, league, now=NOW, wire=wire_of().ids, weights=EQUAL_WEIGHTS)
    made = valuation()
    for espn_id, outlook in made.outlooks.items():
        assert stored.outlooks[espn_id].weekly == pytest.approx(dict(outlook.weekly)), espn_id


def test_the_decision_proposes_its_plan_through_policy(store: Store) -> None:
    league = seed(store)
    config = fixture_config()
    decision = decide(store, config)
    assert pairs(decision.moves) == [(FRED, GONE), (WALLY, DUD)]
    add, claim = decision.proposals
    assert parse_payload(add) == AddDropPayload(add_espn_id=FRED, drop_espn_id=GONE)
    payload = parse_payload(claim)
    assert isinstance(payload, WaiverPayload)
    assert (payload.add_espn_id, payload.drop_espn_id) == (WALLY, DUD)
    cap = faab_bid_cap(config.league("nfl").policy, ppr())
    assert cap is not None and payload.bid_amount is not None and payload.bid_amount <= cap
    assert claim.deadline == CLEARS and add.deadline is None  # no schedule: an add has no lock to beat
    assert (add.scoring_period_id, claim.scoring_period_id) == (WEEK, WEEK + 1)
    assert {row.policy for row in (add, claim)} == {"approve"}
    numbers = claim.engine_numbers
    assert numbers["gain"] == pytest.approx(decision.moves[1].gain, abs=1e-3)
    assert numbers["bid"]["cap"] == cap and numbers["bid"]["budget_left"] == 80
    assert numbers["replacement"]["RB"]["espn_id"] == WALLY  # the levels the decision ran against
    assert numbers["horizon"] == {"periods": [WEEK, 17], "playoff_periods": [15, 16, 17], "playoff_weight": 1.5}
    assert claim.rationale is not None and claim.rationale.startswith(
        "Claim Wally Claim (RB) off waivers, dropping Dud"
    )
    assert claim.dedupe_key == f"{WAIVERS_KIND}:nfl:{SEASON}:waiver:{WALLY}:{DUD}"
    assert evaluate(store, config, league, claim.kind, payload, scoring_period_id=WEEK + 1, now=NOW).allowed
    assert decision.blocked == ()


def test_a_rerun_proposes_nothing_twice(store: Store) -> None:
    league = seed(store)
    config = fixture_config()
    first = decide(store, config)
    again = decide(store, config)
    assert again.moves == () and again.proposals == ()
    assert len(store.proposals.open(league.row_id)) == len(first.proposals) == 2
    assert "nfl: 4 players in open proposals were left out of new moves" in again.warnings


def test_an_untouchable_named_in_the_policy_is_never_dropped(store: Store) -> None:
    seed(store)
    decision = decide(store, fixture_config(untouchables=["Gone Back", DUD]))
    dropped = {move.drop.espn_id for move in (*decision.ranked, *decision.moves) if move.drop is not None}
    assert dropped.isdisjoint({GONE, DUD})
    assert pairs(decision.moves) == [(FRED, SPARE), (WALLY, BO)]


def test_each_move_spends_a_slot_in_the_week_it_executes_in(store: Store) -> None:
    """An add spends this week's transaction slot and a claim one in the week its run falls in, where ``evaluate``
    counts it (a Wednesday run is next week's, ROADMAP #29); a move whose week is full is passed over for the next
    best in a week with room, so the plan and the per-proposal check agree."""
    gus = 107
    wire = (*WIRE, (gus, "Gus Agent", "WR", 24, 13, WireStatus.FREE_AGENT, None, False))
    league = seed(store, wire=wire)
    config = fixture_config(max_transactions_per_week=1)
    decision = decide(store, config, wire=wire_of(wire))
    ranked = [move.add.espn_id for move in decision.ranked]
    assert ranked.index(gus) < ranked.index(WALLY)  # Gus is the better add, but Fred spends this week's only slot
    assert pairs(decision.moves) == [(FRED, GONE), (WALLY, DUD)]
    assert [row.scoring_period_id for row in decision.proposals] == [WEEK, WEEK + 1] and decision.blocked == ()
    verdict = evaluate(
        store, config, league, ProposalKind.ADD_DROP, AddDropPayload(add_espn_id=gus), scoring_period_id=WEEK, now=NOW
    )
    assert not verdict.allowed and "1 of 1 transactions already used this week" in str(verdict.reasons)
    again = decide(store, config, wire=wire_of(wire))
    assert again.moves == () and again.proposals == () and again.blocked == ()  # both weeks are full now


def test_the_policy_shapes_what_is_proposed(store: Store) -> None:
    seed(store)
    no_claims = decide(store, fixture_config(waiver="off"), store_proposals=False)
    assert {move.kind for move in no_claims.ranked} == {ProposalKind.ADD_DROP}
    assert "nfl: 1 wire candidates skipped: waiver is off in the league's policy" in no_claims.warnings
    capped = decide(store, fixture_config(max_transactions_per_week=0))
    assert capped.moves == () and capped.proposals == () and capped.ranked  # ranked, but no slot to spend


def test_policy_refusals_are_reported_not_raised(store: Store) -> None:
    seed(store)
    decision = decide(store, fixture_config(), max_moves=1)
    assert len(decision.proposals) == 1
    pause(reason="test")
    try:
        paused = decide(store, fixture_config(), max_moves=1)
    finally:
        resume()
    assert paused.proposals == ()
    assert len(paused.blocked) == 1 and "paused" in paused.blocked[0][1]


def test_ranking_alone_stores_nothing(store: Store) -> None:
    league = seed(store)
    decision = decide(store, fixture_config(), store_proposals=False)
    assert decision.moves and decision.proposals == ()
    assert store.proposals.open(league.row_id) == []


def test_a_league_without_faab_claims_by_priority(store: Store) -> None:
    seed(store, league_settings=real_nfl())  # week 4 of the real league, which claims by waiver priority
    decision = decide(store, fixture_config(), store_proposals=False)
    claims = [move for move in decision.ranked if move.kind is ProposalKind.WAIVER]
    assert claims and all(move.bid is None and move.bidding is None for move in claims)


ZED = 108


def two_claims() -> tuple[tuple[int, str, str, int, float, WireStatus, datetime | None, bool], ...]:
    """The wire with a second claim, worth less than Wally's: only waiver claims are proposed with ``add_drop`` off."""
    return (*WIRE, (ZED, "Zed Claim", "WR", 24, 12.5, WireStatus.WAIVERS, CLEARS, False))


def won(bids: Iterable[int]) -> list[Transaction]:
    return [
        Transaction.model_validate({"id": f"w{i}", "type": "WAIVER", "status": "EXECUTED", "bidAmount": bid})
        for i, bid in enumerate(bids)
    ]


def test_a_second_run_bids_from_what_the_first_runs_open_claim_leaves(store: Store) -> None:
    """The synced spend does not include a claim still waiting for its run: two runs that each bid from the whole
    budget could together bid more than the team has."""
    league = seed(store, wire=two_claims())  # $20 spent: $80 left
    config = fixture_config(add_drop="off")
    first = decide(store, config, wire=wire_of(two_claims()), max_moves=1)
    (first_claim,) = first.moves
    assert first_claim.add.espn_id == WALLY and first_claim.bid is not None and first_claim.bid > 0
    assert first_claim.bidding is not None and (first_claim.bidding.budget_left, first_claim.bidding.pledged) == (80, 0)
    second = decide(store, config, wire=wire_of(two_claims()), max_moves=1)
    (second_claim,) = second.moves
    assert second_claim.add.espn_id == ZED  # Wally's claim is open: left alone
    assert second_claim.bidding is not None
    assert (second_claim.bidding.budget_left, second_claim.bidding.pledged) == (80 - first_claim.bid, first_claim.bid)
    assert second_claim.roster_value is not None
    assert second_claim.bid == heuristic_bid(
        second_claim.gain, roster_value=second_claim.roster_value, budget_left=80 - first_claim.bid, cap=35
    )
    numbers = second.proposals[0].engine_numbers["bid"]
    assert (numbers["budget_left"], numbers["pledged"]) == (80 - first_claim.bid, first_claim.bid)
    pledge = f"nfl: ${first_claim.bid} of the FAAB budget is pledged by open or submitted waiver claims"
    assert pledge in second.warnings
    assert len(store.proposals.open(league.row_id)) == 2
    assert first_claim.bid + (second_claim.bid or 0) <= 80  # together the open claims never exceed what is left


def test_a_claim_past_its_deadline_pledges_nothing(store: Store) -> None:
    seed(store, wire=two_claims())
    config = fixture_config(add_drop="off")
    decide(store, config, wire=wire_of(two_claims()), max_moves=1)
    later = decide_waivers(
        store,
        config,
        "nfl",
        now=CLEARS + timedelta(hours=1),  # the run has passed: the claim expires unexecuted
        wire=wire_of(two_claims()),
        weights=EQUAL_WEIGHTS,
        store_proposals=False,
    )
    assert all(move.bidding is not None and move.bidding.pledged == 0 for move in later.ranked)
    assert not any("pledged" in warning for warning in later.warnings)


def test_a_submitted_claim_pledges_its_bid_until_a_sync_after_its_run(store: Store) -> None:
    """The executor's verified claim stays pending on ESPN until the waiver run, and the synced spend excludes it."""
    league = seed(store, wire=two_claims())
    config = fixture_config(add_drop="off")
    first = decide(store, config, wire=wire_of(two_claims()), max_moves=1)
    (claim,) = first.moves
    (proposal,) = first.proposals
    assert claim.bid is not None and claim.bid > 0 and proposal.deadline == CLEARS
    store.proposals.update(proposal.model_copy(update={"status": "verified"}))
    assert store.proposals.open(league.row_id) == []
    second = decide(store, config, wire=wire_of(two_claims()), max_moves=1)
    (next_claim,) = second.moves
    assert next_claim.add.espn_id == ZED  # the submitted claim's player is left alone
    assert next_claim.bidding is not None and next_claim.bidding.pledged == claim.bid
    assert next_claim.bidding.budget_left == 80 - claim.bid
    # a sync after the run has read the spend: nothing is pledged any more
    team = store.teams.get(league.row_id, OUR_TEAM)
    assert team is not None
    store.teams.upsert(team.model_copy(update={"as_of": CLEARS + timedelta(hours=1)}))
    later = decide_waivers(
        store,
        config,
        "nfl",
        now=CLEARS + timedelta(hours=2),
        wire=wire_of(two_claims()),
        weights=EQUAL_WEIGHTS,
        store_proposals=False,
    )
    assert all(move.bidding is not None and move.bidding.pledged == 0 for move in later.ranked)


def test_a_submitted_claim_not_yet_synced_after_its_run_still_pledges(store: Store) -> None:
    league = seed(store, wire=two_claims())
    config = fixture_config(add_drop="off")
    first = decide(store, config, wire=wire_of(two_claims()), max_moves=1)
    (claim,) = first.moves
    (proposal,) = first.proposals
    store.proposals.update(proposal.model_copy(update={"status": "verified"}))
    assert store.teams.get(league.row_id, OUR_TEAM) is not None
    later = decide_waivers(
        store,
        config,
        "nfl",
        now=CLEARS + timedelta(hours=1),  # past the run, but no sync has read it yet
        wire=wire_of(two_claims()),
        weights=EQUAL_WEIGHTS,
        store_proposals=False,
    )
    assert f"nfl: ${claim.bid} of the FAAB budget is pledged by open or submitted waiver claims" in later.warnings


def test_a_league_history_of_winning_bids_prices_the_claims(store: Store) -> None:
    seed(store, wire=two_claims())
    config = fixture_config(add_drop="off")
    history = won([30] * 12)  # this league's winners pay $30 for what they want
    modeled = decide(store, config, wire=wire_of(two_claims()), history=history, store_proposals=False)
    heuristic = decide(store, config, wire=wire_of(two_claims()), history=[], store_proposals=False)
    claim = next(move for move in modeled.ranked if move.add.espn_id == WALLY)
    base = next(move for move in heuristic.ranked if move.add.espn_id == WALLY)
    model = fit_bid_model(history, budget=100)
    assert claim.bidding is not None and claim.bidding.modeled and claim.roster_value is not None
    assert claim.bid == modeled_bid(model, bid_strength(claim.gain, claim.roster_value), budget_left=80, cap=35)
    assert (
        claim.bid is not None and base.bid is not None and claim.bid > base.bid
    )  # the league pays more than the heuristic
    assert claim.bid <= 35  # the policy's cap
    numbers = claim.engine_numbers()["bid"]
    assert numbers["source"] == "model" and numbers["history"]["winning_bids"] == 12 and numbers["amount"] == claim.bid
    assert not any("bid model" in warning for warning in modeled.warnings)


def test_too_little_history_bids_the_heuristic_and_says_so(store: Store) -> None:
    seed(store, wire=two_claims())
    config = fixture_config(add_drop="off")
    thin = won([30] * (MIN_WINNING_BIDS - 1))
    decision = decide(store, config, wire=wire_of(two_claims()), history=thin, store_proposals=False)
    reason = f"only {MIN_WINNING_BIDS - 1} winning bids in the history (need {MIN_WINNING_BIDS})"
    assert f"nfl: no FAAB bid model ({reason}); bids use the heuristic" in decision.warnings
    claim = next(move for move in decision.ranked if move.add.espn_id == WALLY)
    assert claim.bidding is not None and not claim.bidding.modeled and claim.roster_value is not None
    assert claim.bid == heuristic_bid(claim.gain, roster_value=claim.roster_value, budget_left=80, cap=35)
    numbers = claim.engine_numbers()["bid"]
    assert numbers["source"] == "heuristic" and numbers["history"] == {
        "fitted": False,
        "reason": reason,
        "winning_bids": MIN_WINNING_BIDS - 1,
    }


def test_a_league_without_faab_needs_no_bid_model(store: Store) -> None:
    seed(store, league_settings=real_nfl())
    decision = decide(store, fixture_config(), history=won([30] * 12), store_proposals=False)
    assert not any("bid model" in warning for warning in decision.warnings)


def test_only_a_configured_nfl_league_is_decided(store: Store) -> None:
    league = seed(store)
    with pytest.raises(WaiverError, match="no league 'nba'"):
        decide_waivers(store, fixture_config(), "nba", now=NOW)
    nba = store.leagues.upsert(league.model_copy(update={"id": None, "key": "nba", "sport": "nba", "season": 2027}))
    with pytest.raises(WaiverError, match="NFL"):
        decide_waivers(store, fixture_config(), nba, now=NOW)
    other = Config.model_validate(
        {"league": [League(key="mine", sport="nfl", espn_league_id=5, season=SEASON, team_id=1)]}
    )
    with pytest.raises(WaiverError, match="no league 'nfl' in config.toml"):
        decide_waivers(store, other, league, now=NOW)


def test_the_decision_is_registered_for_nfl_waivers() -> None:
    assert registry.lookup("nfl", WAIVERS_KIND) is decide_waivers
    assert registry.get("nba", WAIVERS_KIND) is None
