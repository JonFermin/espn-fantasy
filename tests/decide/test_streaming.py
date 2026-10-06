"""NBA streaming (ROADMAP #31, DESIGN section 9.3): the add/drop sequence that fills the matchup week's open slot-days.

The league is the real NBA one (``tests/fixtures/espn/real/fba/mSettings.json``: H2H points, PG SG SF PF C G F, three
UTIL, three bench, ``rosterLocktimeType: FIRSTGAME_SCORINGPERIOD``, 3 adds per weekly matchup arriving as 3/7 per day,
the weekly matchups resolved by ESPN's calendar) with a hand-built roster of 13 and a pool of free agents, and the 9-cat
stand-in for the category ranking. The pro schedule is a test double (the surface of ``fm.sports.base.ScheduleLike``):
one 7:30 p.m. ET game per team on the days it plays; Tue Oct 20, 2026 is day 1 and the plan is made on day 3's morning.

Day 3 has our SG, C and the third UTIL without a game (the bench guards cover the first and the third; nobody is a
center). Day 4 is a thin night: only teams 1-7 and the free agents' play, so all three UTIL slots are open. Days 5 and
6 everyone plays.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType

import pytest

from fm.config import Config, League, Policy
from fm.decide import registry as decide_registry
from fm.decide.lineup_daily import DayPlayer, daily_inputs, plan_week
from fm.decide.streaming import (
    STREAMING_CREATED_BY,
    STREAMING_KIND,
    StreamingError,
    acquisition_budget,
    core_players,
    plan_streaming,
    plan_streams,
    propose_streaming,
    rank_streamers,
)
from fm.decide.waivers import Wire, WireEntry, WireStatus
from fm.espn.ids import FBA
from fm.espn.settings import LeagueSettings, LockType, load_league_settings
from fm.proposals import AddDropPayload, ProposalKind, parse_payload, propose
from fm.sports.nba import NBA
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
SEASON = 2027
OUR_TEAM, THEIR_TEAM = 1, 2
PG, SG, SF, PF, C, G, F, UTIL = (FBA.slot_id(label) for label in ("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"))
BENCH = FBA.bench_slot
SYNCED = datetime(2026, 10, 22, 11, 0, tzinfo=UTC)  # 7 a.m. ET Thursday, day 3
MORNING = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)  # 8 a.m. ET, before the day's tips
FIRST_TIP = datetime(2026, 10, 22, 23, 30, tzinfo=UTC)  # 7:30 p.m. ET on day 3
DAY = 3


def tip(day: int, hours: float = 0.0) -> datetime:
    return datetime(2026, 10, 20, 23, 30, tzinfo=UTC) + timedelta(days=day - 1, hours=hours)


@dataclass(frozen=True)
class FakeGame:
    id: int | None
    date: datetime
    home_pro_team_id: int
    away_pro_team_id: int
    start_time_tbd: bool = False
    valid_for_locking: bool = True


class FakeSchedule:
    """``plays`` maps a day to the pro teams with a game; ``late`` maps ``(team, day)`` to hours after the usual tip."""

    def __init__(
        self,
        plays: Mapping[int, Iterable[int]],
        *,
        late: Mapping[tuple[int, int], float] | None = None,
        teams: Iterable[int] = range(1, 31),
    ) -> None:
        self.teams = tuple(teams)
        self._games = {
            day: [
                FakeGame(team * 1000 + day, tip(day, (late or {}).get((team, day), 0.0)), team, 99)
                for team in sorted(set(playing))
            ]
            for day, playing in plays.items()
        }

    @property
    def scoring_periods(self) -> tuple[int, ...]:
        return tuple(sorted(self._games))

    def games(self, scoring_period: int) -> tuple[FakeGame, ...]:
        return tuple(sorted(self._games.get(scoring_period, ()), key=lambda game: game.date))

    def games_for(self, pro_team_id: int, scoring_period: int) -> tuple[FakeGame, ...]:
        return tuple(game for game in self.games(scoring_period) if game.home_pro_team_id == pro_team_id)

    def idle_teams(self, scoring_period: int) -> tuple[int, ...]:
        playing = {game.home_pro_team_id for game in self.games(scoring_period)}
        return tuple(team for team in self.teams if team not in playing)


ALL = set(range(1, 31))
WEEK = {
    1: ALL,
    2: ALL,
    3: ALL - {2, 5, 10},
    4: set(range(1, 8)) | set(range(20, 31)),
    5: ALL,
    6: ALL,
}


def schedule(**options: object) -> FakeSchedule:
    return FakeSchedule(WEEK, **options)  # type: ignore[arg-type]


# (espn_id, name, position, pro team, slot, points per game)
ROSTER = [
    (101, "Pat Guard", "PG", 1, PG, 30.0),
    (102, "Sam Wing", "SG", 2, SG, 28.0),
    (103, "Sid Small", "SF", 3, SF, 26.0),
    (104, "Pete Power", "PF", 4, PF, 24.0),
    (105, "Cal Center", "C", 5, C, 22.0),
    (106, "Gus Shooter", "SG", 6, G, 20.0),
    (107, "Fred Forward", "SF", 7, F, 18.0),
    (108, "Uri Util", "PG", 8, UTIL, 16.0),
    (109, "Una Util", "PF", 9, UTIL, 14.0),
    (110, "Ulf Util", "C", 10, UTIL, 12.0),
    (111, "Ben Bench", "SG", 11, BENCH, 8.0),
    (112, "Bo Bench", "PF", 12, BENCH, 7.0),
    (113, "Bea Bench", "PG", 13, BENCH, 6.0),
]
FREE_AGENTS = [  # (espn_id, name, position, pro team, points per game)
    (301, "Free One", "PG", 20, 15.0),
    (302, "Free Two", "SF", 21, 12.0),
    (303, "Free Three", "C", 22, 10.0),
    (304, "Free Four", "PF", 23, 9.0),
    (305, "Free Five", "SG", 24, 0.5),
]
DROPPABLE = {111, 112, 113}


def real_points() -> LeagueSettings:
    return load_league_settings(FIXTURES / "espn" / "real" / "fba" / "mSettings.json")


def nine_cat() -> LeagueSettings:
    return load_league_settings(FIXTURES / "espn" / "fba_settings_9cat.json")


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        yield opened


def config_for(settings: LeagueSettings, **policy: object) -> Config:
    return Config(
        leagues=(
            League(
                key="nba",
                sport="nba",
                espn_league_id=settings.league_id,
                season=SEASON,
                team_id=OUR_TEAM,
                policy=Policy(**policy),  # type: ignore[arg-type]
            ),
        )
    )


def player_row(espn_id: int, name: str, position: str, team: int) -> PlayerRow:
    return PlayerRow(
        sport="nba",
        espn_id=espn_id,
        full_name=name,
        default_position_id=FBA.position_id(position),
        position=position,
        pro_team_id=team,
        eligible_slot_ids=sorted(NBA.eligible_slots(position, include_reserve=False)),
        as_of=SYNCED,
    )


def blend_row(espn_id: int, stats: Mapping[str, float], day: int = DAY) -> ProjectionRow:
    return ProjectionRow(
        sport="nba",
        espn_id=espn_id,
        source="blend",
        kind="projected",
        season=SEASON,
        scoring_period_id=day,
        stats=dict(stats),
        as_of=SYNCED,
    )


def seed(
    store: Store,
    settings: LeagueSettings | None = None,
    *,
    roster: list[tuple[int, str, str, int, int, float]] | None = None,
    opponent: bool = False,
    lines: Mapping[int, Mapping[str, float]] | None = None,
) -> LeagueRow:
    """A synced league: settings, our roster for day 3, every player's blended per-game line (points leagues: ``PTS``
    only, so a player is worth his ``points per game``; ``lines`` overrides) and the free agents' ``players`` rows."""
    league_settings = settings or real_points()
    league = store.leagues.upsert(
        LeagueRow(
            key="nba",
            sport="nba",
            espn_league_id=league_settings.league_id,
            season=SEASON,
            team_id=OUR_TEAM,
            as_of=SYNCED,
        )
    )
    store.settings.upsert(
        LeagueSettingsRow(league_id=league.row_id, settings=league_settings.model_dump(mode="json"), as_of=SYNCED)
    )
    for team_id in (OUR_TEAM, THEIR_TEAM):
        store.teams.upsert(TeamRow(league_id=league.row_id, team_id=team_id, name=f"Team {team_id}", as_of=SYNCED))
    mine = ROSTER if roster is None else roster
    entries = []
    for espn_id, name, position, team, slot, ppg in mine:
        store.players.upsert(player_row(espn_id, name, position, team))
        entries.append(
            RosterEntryRow(
                league_id=league.row_id,
                scoring_period_id=DAY,
                team_id=OUR_TEAM,
                espn_id=espn_id,
                lineup_slot_id=slot,
                as_of=SYNCED,
            )
        )
        store.projections.upsert(blend_row(espn_id, (lines or {}).get(espn_id) or {"PTS": ppg, "GP": 1.0}))
    store.rosters.replace(league.row_id, DAY, OUR_TEAM, entries)
    for espn_id, name, position, team, ppg in FREE_AGENTS:
        store.players.upsert(player_row(espn_id, name, position, team))
        store.projections.upsert(blend_row(espn_id, (lines or {}).get(espn_id) or {"PTS": ppg, "GP": 1.0}))
    if opponent:
        theirs = []
        for espn_id, position, team, slot in ((201, "PG", 21, PG), (202, "C", 22, C), (203, "SF", 23, SF)):
            store.players.upsert(player_row(espn_id, f"Opp {espn_id}", position, team))
            store.projections.upsert(blend_row(espn_id, {"PTS": 18.0, "GP": 1.0}))
            theirs.append(
                RosterEntryRow(
                    league_id=league.row_id,
                    scoring_period_id=DAY,
                    team_id=THEIR_TEAM,
                    espn_id=espn_id,
                    lineup_slot_id=slot,
                    as_of=SYNCED,
                )
            )
        store.rosters.replace(league.row_id, DAY, THEIR_TEAM, theirs)
    return league


def pool(*ids: int) -> Wire:
    """The wire: these ESPN ids as free agents (everyone by default)."""
    wanted = ids or tuple(espn_id for espn_id, *_ in FREE_AGENTS)
    return Wire(MappingProxyType({i: WireEntry(i, WireStatus.FREE_AGENT) for i in wanted}), MORNING)


def decide(store: Store, league: LeagueRow, **options: object):  # noqa: ANN201
    options.setdefault("wire", pool())
    options.setdefault("now", MORNING)
    options.setdefault("schedule", schedule())
    return plan_streaming(store, league, period=DAY, **options)  # type: ignore[arg-type]


# --- ranking and the sequence -----------------------------------------------------------------------------------------


def test_open_slot_days_decide_who_ranks(store: Store) -> None:
    league = seed(store)
    decision = decide(store, league)
    ranked = {score.espn_id: score for score in decision.ranked}
    assert decision.inputs.days == (3, 4, 5, 6)
    # The baseline's open slots: day 3 the C slot; day 4 the three UTIL (the two UTIL guards and bench are off).
    assert decision.baseline.open_slots_by_day()[DAY] == (C,)
    assert len(decision.baseline.day(4).open_slots) >= 3
    # A free-agent center fills day 3's open C: his day-3 game adds his whole game (10); a pure guard cannot.
    assert ranked[303].open_slot_games >= 2  # day 3's C and a day-4 UTIL
    assert ranked[301].open_slot_games >= 1  # day 4's UTILs
    assert ranked[305].value == pytest.approx(0.5 * 3, abs=3.0)  # a half-point scrub is worth little
    assert decision.ranked[0].value >= decision.ranked[-1].value
    assert [score.value for score in decision.ranked] == sorted((s.value for s in decision.ranked), reverse=True)
    assert ranked[301].games == 4 and ranked[304].games == 4


def test_the_plan_fills_open_slot_days_and_drops_only_players_who_are_not_core(store: Store) -> None:
    league = seed(store)
    decision = decide(store, league)
    plan = decision.plan
    assert plan is not None and plan.gain > 0
    assert 1 <= len(plan.moves) <= decision.budget.left
    for move in plan.moves:
        assert move.drop in DROPPABLE  # the roster is full: every add drops a bench-quality player
        assert move.add in {301, 302, 303, 304}  # never the half-point scrub
    assert sum(move.gain for move in plan.moves) == pytest.approx(plan.gain)
    assert len({move.add for move in plan.moves}) == len(plan.moves)
    assert {move.drop for move in plan.moves}.isdisjoint({101, 102, 103, 104, 105, 106, 107, 108, 109, 110})
    assert decision.protected >= {101, 110}  # the core (the best ten by per-game value)
    assert plan.baseline == pytest.approx(decision.baseline.total)


def test_the_acquisition_limit_is_the_leagues_matchup_rate_and_the_policy_cap(store: Store) -> None:
    league = seed(store)
    settings = real_points()
    assert settings.acquisition.matchup_limit_rate == pytest.approx(3 / 7)
    budget = acquisition_budget(store, league, settings, Policy(max_transactions_per_week=7), period=DAY, now=MORNING)
    assert (budget.espn_limit, budget.limit, budget.used, budget.left) == (2, 2, 0, 2)  # 3/7 x 6 days, floored
    week_two = acquisition_budget(store, league, settings, Policy(max_transactions_per_week=7), period=8, now=MORNING)
    assert week_two.espn_limit == 3  # 3/7 x 7 days
    capped = acquisition_budget(store, league, settings, Policy(max_transactions_per_week=1), period=DAY, now=MORNING)
    assert (capped.limit, capped.left) == (1, 1)
    assert (
        acquisition_budget(
            store, league, settings, Policy(max_transactions_per_week=7), period=DAY, now=MORNING, used=2
        ).left
        == 0
    )
    one = decide(store, league, policy=Policy(max_transactions_per_week=1))
    assert one.plan is not None and len(one.plan.moves) == 1
    none = decide(store, league, acquisitions_used=2)
    assert none.plan is None and none.drafts == () and none.budget.left == 0


def test_moves_already_proposed_use_up_the_matchups_slots(store: Store) -> None:
    league = seed(store)
    config = config_for(real_points(), max_transactions_per_week=1)
    propose(
        store,
        config,
        league,
        ProposalKind.ADD_DROP,
        AddDropPayload(add_espn_id=399, drop_espn_id=113),
        created_by="test",
        scoring_period_id=5,  # day 5 is matchup 1 (days 1-6): the calendar places it, not the weekly ids
        now=MORNING,
    )
    decision = decide(store, league, policy=config.league("nba").policy)
    assert decision.budget.used == 1 and decision.budget.left == 0 and decision.plan is None
    assert 113 in decision.protected  # a player in an open proposal is left alone


def test_a_slot_with_no_games_left_under_a_games_played_limit_is_not_filled(store: Store) -> None:
    league = seed(store)
    capped = real_points().model_copy(update={"slot_stat_limits": {C: {42: 5}, UTIL: {42: 246}}})
    only_center = pool(303)
    free = decide(store, league, wire=only_center)
    assert free.plan is not None and [move.add for move in free.plan.moves] == [303]
    limited = decide(store, league, wire=only_center, settings=capped, games_used={C: 5})
    assert limited.inputs.slot_limits[C] == 0
    assert limited.baseline.open_slots_by_day()[DAY] == (C,)  # still open: the cap is what keeps it empty
    # The center's day-3 game was worth 10 in the C slot; capped, it only replaces a weak UTIL (3 points more).
    assert limited.plan is None or limited.plan.gain <= free.plan.gain - 5.0


def test_untouchables_and_protected_players_are_never_dropped(store: Store) -> None:
    league = seed(store)
    shielded = Policy(untouchables=(111, 112, 113))
    decision = decide(store, league, policy=shielded)
    assert decision.plan is None and decision.drafts == ()  # a full roster and nobody droppable
    assert decision.protected >= DROPPABLE
    some = decide(store, league, protected=[111, 112])
    assert some.plan is not None and some.plan.moves[0].drop == 113
    added = {move.add for move in some.plan.moves}
    assert {move.drop for move in some.plan.moves} <= {113} | added  # a streamer may be dropped for the next one
    named = decide(store, league, policy=Policy(untouchables=("Ben Bench", "Bo Bench", "Bea Bench")))
    assert named.plan is None
    wide = decide(store, league, core_size=6)  # a smaller core frees more players to drop
    assert 107 not in wide.protected and 101 in wide.protected


def test_a_roster_spot_is_filled_without_a_drop(store: Store) -> None:
    league = seed(store, roster=ROSTER[:-1])  # twelve players: one spot open
    decision = decide(store, league, policy=Policy(max_transactions_per_week=1))
    assert decision.plan is not None
    assert [move.drop for move in decision.plan.moves] == [None]  # nobody is dropped for the add


def test_a_player_in_the_pool_who_is_on_waivers_or_rostered_is_not_a_streamer(store: Store) -> None:
    league = seed(store)
    wire = Wire(
        MappingProxyType(
            {
                301: WireEntry(301, WireStatus.WAIVERS, MORNING + timedelta(hours=20)),  # a claim, not an immediate add
                302: WireEntry(302, WireStatus.FREE_AGENT),
                101: WireEntry(101, WireStatus.FREE_AGENT),  # ours: never a candidate
            }
        ),
        MORNING,
    )
    decision = decide(store, league, wire=wire)
    assert {score.espn_id for score in decision.ranked} == {302}


# --- timing: adds land before the first tip ---------------------------------------------------------------------------


def test_adds_are_scheduled_before_the_days_first_tip(store: Store) -> None:
    league = seed(store)
    config = config_for(real_points(), max_transactions_per_week=3)
    result = propose_streaming(store, config, league, schedule=schedule(), now=MORNING, period=DAY, wire=pool())
    assert result.blocked == ()
    assert result.proposals, "a move on the target day was expected"
    for row in result.proposals:
        assert (
            row.kind == ProposalKind.ADD_DROP.value
            and row.created_by == STREAMING_CREATED_BY
            and row.scoring_period_id == DAY
        )
        assert row.deadline == FIRST_TIP  # FIRSTGAME_SCORINGPERIOD closes adds and drops at the day's first tip
        assert row.deadline is not None and row.deadline > MORNING
        payload = parse_payload(row)
        assert isinstance(payload, AddDropPayload) and payload.drop_espn_id in DROPPABLE
        assert row.engine_numbers["add"]["espn_id"] == payload.add_espn_id
        assert row.engine_numbers["acquisitions"]["left"] >= 1
        assert row.rationale is not None and "Add " in row.rationale
    config = config_for(real_points(), max_transactions_per_week=1)
    store.proposals.update(result.proposals[0].model_copy(update={"status": "rejected"}))  # give its slot back
    first = propose_streaming(store, config, league, schedule=schedule(), now=MORNING, period=DAY, wire=pool())
    assert len(first.proposals) == 1
    again = propose_streaming(store, config, league, schedule=schedule(), now=MORNING, period=DAY, wire=pool())
    # The week's one transaction is held by the first proposal, and its players are left alone: nothing new.
    assert again.proposals == () and again.decision.budget.left == 0 and again.decision.plan is None


def test_once_the_first_tip_has_passed_nothing_is_added_that_day(store: Store) -> None:
    league = seed(store)
    after = FIRST_TIP + timedelta(minutes=1)
    decision = decide(store, league, now=after)
    assert decision.drafts == ()  # day 3's adds and drops closed at the first tip
    if decision.plan is not None:
        assert all(move.day > DAY for move in decision.plan.moves)  # whatever is planned is for a later day


def test_an_individual_game_lock_closes_each_move_at_the_earlier_of_the_two_teams(store: Store) -> None:
    league = seed(store)
    per_game = real_points().model_copy(update={"roster_lock_type": LockType.INDIVIDUAL_GAME})
    # The free agents' teams tip at 7:30; our bench guards' teams (11-13) tip an hour earlier on day 3.
    early = schedule(late={(11, DAY): -1.0, (12, DAY): -1.0, (13, DAY): -1.0})
    decision = decide(store, league, settings=per_game, schedule=early)
    assert decision.drafts
    for draft in decision.drafts:
        assert draft.deadline == tip(DAY, -1.0)  # the dropped bench player's game starts first
        assert draft.scoring_period_id == DAY


def test_an_unknown_roster_lock_type_is_refused(store: Store) -> None:
    league = seed(store)
    weekly = real_points().model_copy(
        update={"roster_lock_type": LockType.UNKNOWN, "roster_lock_type_raw": "FIRSTGAME_WEEKLY"}
    )
    with pytest.raises(StreamingError, match="UNKNOWN"):
        decide(store, league, settings=weekly)


def test_refuses_what_it_cannot_plan(store: Store) -> None:
    league = seed(store)
    other = store.leagues.upsert(league.model_copy(update={"key": "nfl", "sport": "nfl"}))
    with pytest.raises(StreamingError, match="NBA"):
        decide(store, other)
    bare = store.leagues.upsert(
        LeagueRow(key="x", sport="nba", espn_league_id=1, season=SEASON, team_id=OUR_TEAM, as_of=SYNCED)
    )
    with pytest.raises(StreamingError, match="no synced settings"):
        decide(store, bare)
    with pytest.raises(StreamingError, match="no roster"):
        plan_streaming(store, league, schedule=schedule(), now=MORNING, period=5, wire=pool())
    with pytest.raises(StreamingError, match="no ESPN calendar"):
        decide(store, league, settings=real_points().model_copy(update={"season": 2099}))
    with pytest.raises(StreamingError, match="not in config|no league|nba2"):
        propose_streaming(
            store,
            Config(leagues=(League(key="nba2", sport="nba", espn_league_id=1, season=SEASON, team_id=1),)),
            league,
            schedule=schedule(),
            now=MORNING,
            period=DAY,
            wire=pool(),
        )


def test_a_policy_refusal_is_reported_not_raised(store: Store) -> None:
    league = seed(store)
    config = config_for(real_points(), add_drop="off")
    result = propose_streaming(store, config, league, schedule=schedule(), now=MORNING, period=DAY, wire=pool())
    assert result.proposals == () and len(result.blocked) == len(result.decision.drafts) >= 1
    assert "is off for league" in result.blocked[0]


def test_an_empty_wire_plans_nothing_and_says_so(store: Store) -> None:
    league = seed(store)
    decision = decide(store, league, wire=Wire(MappingProxyType({}), None, ("nba: no free-agent pool captured",)))
    assert decision.plan is None and decision.ranked == () and decision.drafts == ()
    assert any("no free-agent pool" in warning for warning in decision.warnings)


# --- category leagues -------------------------------------------------------------------------------------------------


def category_lines() -> dict[int, dict[str, float]]:
    """Our roster scores in every volume category a three-man opponent trails in, so those are decided; threes are a
    toss-up (nobody has any but two free agents). Free agent 301 piles up points (decided); 302 only threes."""
    lines = {espn_id: {"PTS": ppg, "REB": ppg / 4, "AST": ppg / 5, "GP": 1.0} for espn_id, *_, ppg in ROSTER}
    lines[301] = {"PTS": 24.0, "REB": 4.0, "AST": 3.0, "GP": 1.0}
    lines[302] = {"PTS": 6.0, "REB": 1.0, "AST": 1.0, "3PM": 3.0, "GP": 1.0}
    lines[303] = {"PTS": 4.0, "REB": 5.0, "AST": 0.5, "GP": 1.0}
    lines[304] = {"PTS": 3.0, "REB": 1.0, "AST": 0.5, "GP": 1.0}
    lines[305] = {"PTS": 1.0, "REB": 0.5, "AST": 0.5, "GP": 1.0}
    return lines


def test_a_category_league_ranks_streamers_by_swing_not_by_volume(store: Store) -> None:
    league = seed(store, nine_cat(), opponent=True, lines=category_lines())
    flat = decide(store, league, settings=nine_cat(), lines=category_lines())
    versus = decide(store, league, settings=nine_cat(), lines=category_lines(), opponent_team_id=THEIR_TEAM)
    assert flat.inputs.values.outlook is None and versus.inputs.values.outlook is not None
    outlook = versus.inputs.values.outlook
    assert outlook.weights["3PM"] > 100 * outlook.weights["PTS"]  # points are decided, threes a toss-up
    by_flat = {score.espn_id: score for score in flat.ranked}
    by_swing = {score.espn_id: score for score in versus.ranked}
    # Without the opponent a 24-point scorer is the best streamer (equal G-scores); against him the three-point shooter.
    assert flat.ranked[0].espn_id == 301
    assert versus.ranked[0].espn_id == 302
    assert by_swing[302].swing["3PM"] > 0 and by_swing[302].open_slot_games >= 1
    assert by_swing[301].value < 1e-6 < by_swing[302].value  # a decided category adds nothing
    assert by_swing[301].swing["PTS"] == pytest.approx(0.0, abs=1e-6)
    assert by_flat[301].value > by_flat[302].value
    assert versus.plan is not None and versus.plan.moves[0].add == 302


def test_core_players_are_the_best_by_per_game_value() -> None:
    assert core_players({1: 5.0, 2: 9.0, 3: 7.0, 4: 7.0}, 2) == {2, 3}  # ties by ESPN id
    assert core_players({1: 5.0}, 0) == frozenset()


def test_rank_streamers_reports_the_open_slot_games_each_candidate_fills(store: Store) -> None:
    league = seed(store)
    inputs = daily_inputs(store, league, real_points(), schedule=schedule(), period=DAY, now=MORNING)
    rows = {row.espn_id: row for row in store.players.many("nba", [i for i, *_ in ROSTER] + [301, 303])}
    candidates = {espn_id: inputs.model.day_player(rows[espn_id], BENCH) for espn_id in (301, 303)}
    plan = plan_week(inputs.players, inputs.slot_counts, inputs.days)
    ranked = rank_streamers(
        inputs, rows, {k: v for k, v in candidates.items() if v is not None}, schedule(), real_points(), baseline=plan
    )
    scores = {score.espn_id: score for score in ranked}
    assert (scores[303].open_slot_games, scores[301].open_slot_games) == (2, 1)  # the center fills C on day 3
    assert scores[301].value > scores[303].value  # but the guard is worth more points


def test_the_sequence_is_the_best_there_is_over_single_moves_and_pairs(store: Store) -> None:
    """With one acquisition the dynamic program finds the best (day, add, drop) of all of them, tried one at a time;
    with two it does better, since the second move pays for itself."""
    league = seed(store)
    inputs = daily_inputs(store, league, real_points(), schedule=schedule(), period=DAY, now=MORNING)
    ids = [i for i, *_ in ROSTER] + [i for i, *_ in FREE_AGENTS]
    rows = {row.espn_id: row for row in store.players.many("nba", ids)}
    pool_players = {
        espn_id: player
        for espn_id, *_ in FREE_AGENTS
        if (player := inputs.model.day_player(rows[espn_id], BENCH)) is not None
    }

    def best(budget: int, candidates: dict[int, DayPlayer], droppable: Iterable[int]) -> float:
        plan = plan_streams(
            inputs,
            rows,
            candidates,
            real_points(),
            schedule(),
            budget=budget,
            droppable=droppable,
            min_gain=0.0,
            beam=500,
            finalists=50,
        )
        return 0.0 if plan is None else plan.gain

    singles = {
        (add, drop): best(1, {add: pool_players[add]}, [drop]) for add in pool_players for drop in sorted(DROPPABLE)
    }
    assert best(1, pool_players, DROPPABLE) == pytest.approx(max(singles.values()))
    two = best(2, pool_players, DROPPABLE)
    assert two > max(singles.values())  # the thin night (day 4) leaves room for a second streamer
    assert best(0, pool_players, DROPPABLE) == 0.0


def test_the_decision_is_registered_for_nba_beside_the_waivers() -> None:
    assert decide_registry.lookup("nba", STREAMING_KIND) is propose_streaming
    assert STREAMING_KIND == "streaming"
    # fm.jobs.tick calls ``fn(store, config, league_row, schedule=..., now=..., opponent_team_id=...)`` with what
    # the signature accepts.
    parameters = inspect.signature(propose_streaming).parameters
    assert list(parameters)[:3] == ["store", "config", "league"]
    assert {"schedule", "now", "opponent_team_id"} <= set(parameters)
