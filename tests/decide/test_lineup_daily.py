"""NBA daily lineups across the matchup week (ROADMAP #31, DESIGN sections 9.1 and 9.3).

Two layers:

- The planner (:func:`plan_week`, :func:`prefer_pivots`) on hand-built rosters: open slot-days are filled from the
  bench, a starter who will not play is benched while a player who plays can take his slot, locked players never move,
  a games-played limit is spent on the best days, equal lineups stay as they are and a late-swap pivot decides between
  arrangements.
- The store-backed decision on a points league (the real league's ``mSettings``: PG SG SF PF C G F, three UTIL, three
  bench; ESPN's weekly matchup periods resolved by the season calendar) and the 9-cat stand-in: availability read from
  the official injury report instead of ESPN's raw status, the matchup week from ESPN's calendar, proposals through
  ``fm.proposals.propose``, and the category swing weights from the opponent's roster.

The pro schedule is a test double with the surface of ``fm.sports.base.ScheduleLike``: one 7:30 p.m. ET game per team
on the days it plays (Tue Oct 20, 2026 is day 1).
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from fm.config import Config, League, Policy
from fm.decide import lineup as _nfl_lineup  # noqa: F401  (registers the NFL decision)
from fm.decide import registry as decide_registry
from fm.decide.lineup_daily import (
    DAILY_LINEUP_CREATED_BY,
    DAILY_LINEUP_KIND,
    DailyLineupError,
    DayPlayer,
    PlayerDay,
    daily_inputs,
    hold_week,
    matchup_window,
    plan_daily_lineup,
    plan_week,
    prefer_pivots,
    propose_daily_lineup,
    slot_limits_left,
    swing_outlook,
    swing_weights,
)
from fm.espn.ids import FBA
from fm.espn.settings import LeagueSettings, LockType, load_league_settings
from fm.model.availability import assess
from fm.proposals import LineupPayload, ProposalKind, parse_payload
from fm.sources.nba_injuries import OfficialInjuryEntry, OfficialInjuryReport
from fm.sports.nba import NBA
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
SEASON = 2027
OUR_TEAM, THEIR_TEAM = 1, 2
PG, SG, SF, PF, C, G, F, UTIL = (FBA.slot_id(label) for label in ("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"))
BENCH, IR = FBA.bench_slot, FBA.ir_slot
COUNTS = {PG: 1, SG: 1, SF: 1, PF: 1, C: 1, G: 1, F: 1, UTIL: 3}
SYNCED = datetime(2026, 10, 22, 11, 0, tzinfo=UTC)  # 7 a.m. ET Thursday, day 3
MORNING = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)  # 8 a.m. ET, before the day's tips
FIRST_TIP = datetime(2026, 10, 22, 23, 30, tzinfo=UTC)  # 7:30 p.m. ET on day 3
DAY = 3


def tip(day: int, hours: float = 0.0) -> datetime:
    """The 7:30 p.m. ET tip of a day (day 1 is Tue Oct 20, 2026), ``hours`` later."""
    return datetime(2026, 10, 20, 23, 30, tzinfo=UTC) + timedelta(days=day - 1, hours=hours)


# --- the pro schedule double ------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FakeGame:
    id: int | None
    date: datetime
    home_pro_team_id: int
    away_pro_team_id: int
    start_time_tbd: bool = False
    valid_for_locking: bool = True


class FakeSchedule:
    """``plays`` maps a day to the pro teams with a game (each against the pseudo-team 99); ``late`` maps ``(team,
    day)`` to hours after the usual tip."""

    def __init__(
        self,
        plays: Mapping[int, Iterable[int]],
        *,
        late: Mapping[tuple[int, int], float] | None = None,
        teams: Iterable[int] = range(1, 31),
    ) -> None:
        self.teams = tuple(teams)
        self._games: dict[int, list[FakeGame]] = {}
        for day, playing in plays.items():
            self._games[day] = [
                FakeGame(team * 1000 + day, tip(day, (late or {}).get((team, day), 0.0)), team, 99)
                for team in sorted(set(playing))
            ]

    @property
    def scoring_periods(self) -> tuple[int, ...]:
        return tuple(sorted(self._games))

    def games(self, scoring_period: int) -> tuple[FakeGame, ...]:
        return tuple(sorted(self._games.get(scoring_period, ()), key=lambda game: game.date))

    def games_for(self, pro_team_id: int, scoring_period: int) -> tuple[FakeGame, ...]:
        return tuple(game for game in self.games(scoring_period) if pro_team_id in (game.home_pro_team_id,))

    def idle_teams(self, scoring_period: int) -> tuple[int, ...]:
        playing = {game.home_pro_team_id for game in self.games(scoring_period)}
        return tuple(team for team in self.teams if team not in playing)


def everyone(days: Iterable[int], *, except_: Mapping[int, Iterable[int]] | None = None) -> dict[int, list[int]]:
    """Every pro team (1-30) plays each of ``days`` but the teams ``except_`` names for a day."""
    return {day: [t for t in range(1, 31) if t not in set((except_ or {}).get(day, ()))] for day in days}


# --- planner layer ----------------------------------------------------------------------------------------------------


def played(days: Mapping[int, float] | Iterable[int], per_game: float, **fields: object) -> dict[int, PlayerDay]:
    """A player's days: ``{day: p_active}`` or the days he plays at ``p_active`` 1; every other day he has no game."""
    chances = days if isinstance(days, Mapping) else dict.fromkeys(days, 1.0)
    return {
        day: PlayerDay(has_game=True, p_active=p, value=p * per_game, lock_at=tip(day), **fields)  # type: ignore[arg-type]
        for day, p in chances.items()
    }


def dp(
    espn_id: int,
    slot: int,
    position: str,
    per_game: float,
    days: Mapping[int, float] | Iterable[int],
    *,
    locked: Iterable[int] = (),
    eligible: Iterable[int] | None = None,
) -> DayPlayer:
    table = played(days, per_game)
    for day in locked:
        table[day] = PlayerDay(
            has_game=table[day].has_game,
            p_active=table[day].p_active,
            value=table[day].value,
            locked=True,
            lock_at=table[day].lock_at,
        )
    return DayPlayer(
        espn_id=espn_id,
        slot_id=slot,
        eligible=NBA.eligible_slots(position, include_reserve=False) if eligible is None else frozenset(eligible),
        per_game=per_game,
        days=table,
        position=position,
        name=f"P{espn_id}",
    )


def slots_on(plan_day_slots: Mapping[int, int], *ids: int) -> tuple[int, ...]:
    return tuple(plan_day_slots[espn_id] for espn_id in ids)


def test_a_bench_player_with_a_game_fills_a_starters_empty_day() -> None:
    players = [
        dp(1, PG, "PG", 30, [1, 2]),
        dp(2, SG, "SG", 20, [1]),  # no game on day 2
        dp(3, BENCH, "SG", 10, [1, 2]),
    ]
    plan = plan_week(players, {PG: 1, SG: 1}, [1, 2])
    assert plan is not None
    assert plan.day(1).moves == () and plan.day(1).slots[3] == BENCH  # everyone plays on day 1: nothing to fix
    assert plan.day(2).slots == {1: PG, 2: BENCH, 3: SG}
    assert {(m.espn_id, m.from_slot_id, m.to_slot_id) for m in plan.day(2).moves} == {(2, SG, BENCH), (3, BENCH, SG)}
    assert plan.total == pytest.approx(30 + 20 + 30 + 10)
    assert plan.open_slot_days == 0


def test_open_slot_days_are_the_slots_nobody_who_plays_can_fill() -> None:
    players = [
        dp(1, PG, "PG", 30, [1, 2, 3]),
        dp(2, C, "C", 20, [1, 3]),  # no game on day 2: nobody else is a center
        dp(3, BENCH, "PG", 6, [2]),  # a guard cannot fill C
    ]
    plan = plan_week(players, {PG: 1, C: 1, UTIL: 1}, [1, 2, 3])
    assert plan is not None
    assert plan.open_slots_by_day() == {1: (UTIL,), 2: (C,), 3: (UTIL,)}  # day 2: the guard takes UTIL, C stays open
    assert plan.open_slot_days == 3
    assert plan.day(2).slots[3] == UTIL and plan.day(2).idle_starters == (
        2,
    )  # the center sits in C with nobody to fill
    assert (
        hold_week(players, {PG: 1, C: 1, UTIL: 1}, [1, 2, 3]).open_slot_days == 4
    )  # the idle center's C slot counts as open on day 2 as well


def test_a_zero_starter_is_benched_for_a_player_who_plays_and_stays_when_nobody_can_replace_him() -> None:
    players = [
        dp(1, PG, "PG", 25, {}),  # ruled out all week (p_active 0 is a day with no value)
        dp(2, SG, "SG", 12, [1]),
        dp(3, BENCH, "PG", 9, [1]),
        dp(4, BENCH, "C", 8, [1]),  # a center cannot take the PG slot
    ]
    plan = plan_week(players, {PG: 1, SG: 1, C: 1}, [1])
    assert plan is not None
    day = plan.day(1)
    assert slots_on(day.slots, 1, 2, 3, 4) == (BENCH, SG, PG, C)
    out = [dp(1, PG, "PG", 25, {1: 0.0}), dp(2, BENCH, "C", 8, [1])]
    stuck = plan_week(out, {PG: 1}, [1])
    assert stuck is not None
    assert stuck.day(1).moves == () and stuck.day(1).idle_starters == (1,)  # nobody can play PG: he stays


def test_equally_good_lineups_keep_the_current_one() -> None:
    players = [dp(1, PG, "PG", 10, [1, 2]), dp(2, UTIL, "PG", 10, [1, 2]), dp(3, BENCH, "PG", 10, [1, 2])]
    plan = plan_week(players, {PG: 1, UTIL: 1}, [1, 2])
    assert plan is not None and all(not day.moves for day in plan.days)


def test_a_locked_player_is_never_moved_and_his_slot_stays_his() -> None:
    players = [
        dp(1, PG, "PG", 5, [1], locked=[1]),  # a poor starter whose game has begun keeps the slot
        dp(2, BENCH, "PG", 25, [1]),
        dp(3, BENCH, "SG", 30, [1], locked=[1]),  # his game started while he sat on the bench
        dp(4, SG, "SG", 3, [1]),
    ]
    plan = plan_week(players, {PG: 1, SG: 1, UTIL: 1}, [1])
    assert plan is not None
    day = plan.day(1)
    assert day.slots[1] == PG and day.slots[3] == BENCH  # locked: not moved
    assert day.slots[2] == UTIL and day.slots[4] == SG  # the free guard goes to UTIL, the free SG keeps his slot
    assert {move.espn_id for move in day.moves} == {2}


def test_a_player_on_ir_stays_there() -> None:
    players = [dp(1, IR, "PG", 40, [1]), dp(2, PG, "PG", 0, {1: 0.0}), dp(3, BENCH, "PG", 5, [1])]
    plan = plan_week(players, {PG: 1}, [1])
    assert plan is not None
    assert slots_on(plan.day(1).slots, 1, 2, 3) == (IR, BENCH, PG)


def test_a_games_played_limit_is_spent_on_the_best_days() -> None:
    star = dp(1, PG, "PG", 30, [1, 2, 3], eligible=[PG])
    plan = plan_week([star, dp(2, BENCH, "PG", 5, [1, 2, 3], eligible=[PG])], {PG: 1}, [1, 2, 3], slot_limits={PG: 2})
    assert plan is not None
    assert plan.starts(1) == 2 and plan.starts(2) == 0  # two starts for the star: the limit is the slot's
    assert sum(day.games.get(PG, 0) for day in plan.days) == 2 and plan.total == pytest.approx(60)
    scarce = [
        dp(1, PG, "PG", 5, [1, 2], eligible=[PG, UTIL]),
        dp(2, BENCH, "PG", 30, [2], eligible=[PG, UTIL]),
    ]
    best = plan_week(scarce, {PG: 1, UTIL: 1}, [1, 2], slot_limits={PG: 1, UTIL: 1})
    assert best is not None
    assert best.day(2).slots[2] in (PG, UTIL)
    spent = sum(day.games.get(PG, 0) for day in best.days)
    assert spent <= 1 and sum(day.games.get(UTIL, 0) for day in best.days) <= 1
    assert slot_limits_left(_settings("fba_settings_9cat.json"), {PG: 80})[0][PG] == 2


def test_a_limit_of_zero_holds_nobody() -> None:
    plan = plan_week([dp(1, BENCH, "PG", 30, [1])], {PG: 1}, [1], slot_limits={PG: 0})
    assert plan is not None and plan.day(1).slots == {1: BENCH} and plan.day(1).open_slots == (PG,)


def test_keeping_starters_who_play_only_benches_zeros() -> None:
    players = [
        dp(1, PG, "PG", 10, [1]),
        dp(2, BENCH, "PG", 30, [1]),  # better, but the bench_inactive lineup keeps the starter
        dp(3, SG, "SG", 8, {}),
        dp(4, BENCH, "SG", 6, [1]),
    ]
    rescue = plan_week(players, {PG: 1, SG: 1}, [1], keep_starters=True)
    best = plan_week(players, {PG: 1, SG: 1}, [1])
    assert rescue is not None and best is not None
    assert slots_on(rescue.day(1).slots, 1, 2, 3, 4) == (PG, BENCH, BENCH, SG)
    assert slots_on(best.day(1).slots, 1, 2, 3, 4) == (BENCH, PG, BENCH, SG)


def test_planner_refuses_what_it_cannot_plan() -> None:
    with pytest.raises(ValueError, match="listed twice"):
        plan_week([dp(1, PG, "PG", 1, [1]), dp(1, BENCH, "PG", 1, [1])], {PG: 1}, [1])
    with pytest.raises(ValueError, match="active slots only"):
        plan_week([dp(1, PG, "PG", 1, [1])], {PG: 1, BENCH: 3}, [1])
    with pytest.raises(ValueError, match="negative count"):
        plan_week([dp(1, PG, "PG", 1, [1])], {PG: -1}, [1])
    with pytest.raises(ValueError, match="p_active"):
        PlayerDay(has_game=True, p_active=1.5)


def availability(espn_id: int, p_active: float, game_time: datetime) -> AvailabilityRow:
    return AvailabilityRow(
        sport="nba",
        espn_id=espn_id,
        season=SEASON,
        scoring_period_id=1,
        p_active=p_active,
        game_time=game_time,
        as_of=SYNCED,
    )


def test_a_questionable_starter_is_slotted_where_a_later_pivot_can_cover_him() -> None:
    """The FLEX/UTIL trick: A may sit (questionable, early game) and B, a UTIL-only bench player with a later game, can
    cover him only in UTIL. Slotting A in UTIL and the healthy C (eligible for both) in PG is worth B's expected points
    times A's chance of sitting; the starters, and so the day's base value, are the same."""
    early, late = tip(1, -1), tip(1, 2)

    def player(
        espn_id: int, slot: int, position: str, per_game: float, p: float, at: datetime, **kw: object
    ) -> DayPlayer:
        row = availability(espn_id, p, at)
        day = PlayerDay(has_game=True, p_active=p, value=p * per_game, lock_at=at, availability=row)
        return DayPlayer(
            espn_id=espn_id,
            slot_id=slot,
            eligible=kw.get("eligible", NBA.eligible_slots(position, include_reserve=False)),  # type: ignore[arg-type]
            per_game=per_game,
            days={1: day},
            position=position,
            name=f"P{espn_id}",
        )

    a = player(1, PG, "PG", 30, 0.5, early)
    c = player(3, UTIL, "PG", 20, 1.0, early)
    b = player(2, BENCH, "SG", 8, 1.0, late, eligible=frozenset({UTIL}))
    counts = {PG: 1, UTIL: 1}
    base = plan_week([a, b, c], counts, [1])
    assert base is not None and base.day(1).moves == ()
    swapped = prefer_pivots([a, b, c], counts, base.day(1))
    assert swapped.slots == {1: UTIL, 2: BENCH, 3: PG}
    assert swapped.value == pytest.approx(base.day(1).value)
    assert {(m.espn_id, m.to_slot_id) for m in swapped.moves} == {(1, UTIL), (3, PG)}
    # Too small a gain is not worth the moves.
    assert prefer_pivots([a, b, c], counts, base.day(1), min_gain=100.0) is base.day(1)
    # A bench player who locks before the starter's status settles cannot pivot: nothing to arrange for.
    early_bench = player(2, BENCH, "SG", 8, 1.0, tip(1, -3), eligible=frozenset({UTIL}))
    assert prefer_pivots([a, early_bench, c], counts, base.day(1)) is base.day(1)


# --- category outlook -------------------------------------------------------------------------------------------------


def test_a_close_category_swings_the_most_and_a_decided_one_almost_not_at_all() -> None:
    outlook = swing_outlook(
        {"PTS": 10.0, "REB": 20.0, "STL": 1.0},
        {"PTS": 10.0, "REB": 0.0, "STL": 20.0},
        games_ours=50,
        games_theirs=50,
        game_sd=1.0,
    )
    assert outlook.win_probability["PTS"] == pytest.approx(0.5)
    assert outlook.win_probability["REB"] > 0.9 > 0.1 > outlook.win_probability["STL"]
    assert (
        outlook.weights["PTS"] > 5 * outlook.weights["REB"] > 0 and outlook.weights["PTS"] > 5 * outlook.weights["STL"]
    )
    assert outlook.weights["PTS"] == pytest.approx(1 / math.sqrt(2 * math.pi) / 10.0)  # phi(0) / sd, sd = sqrt(100)
    assert outlook.expected_wins == pytest.approx(sum(outlook.win_probability.values()))
    banked = swing_outlook({"PTS": 0.0}, {"PTS": 0.0}, games_ours=1, games_theirs=1, margin_so_far={"PTS": 30.0})
    assert banked.win_probability["PTS"] > 0.99
    assert swing_weights(None, ["PTS"]) is None
    assert swing_weights(outlook, ["PTS", "AST"]) == {"PTS": outlook.weights["PTS"], "AST": 0.0}


# --- the store-backed decision ----------------------------------------------------------------------------------------

# (espn_id, name, position, pro team, slot, points per game): our real-shaped roster of 13 under the points league.
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
ESPN_IDS = {name: espn_id for espn_id, name, *_ in ROSTER}
OPPONENT = [  # a thin opposing roster for the category tests: (id, name, position, pro team, slot, PTS, REB, STL)
    (201, "Olly One", "PG", 21, PG, 20.0, 3.0, 2.0),
    (202, "Otto Two", "C", 22, C, 12.0, 11.0, 0.5),
    (203, "Omar Three", "SF", 23, SF, 15.0, 6.0, 1.0),
]


def _settings(name: str) -> LeagueSettings:
    return load_league_settings(FIXTURES / "espn" / name)


def real_points() -> LeagueSettings:
    return load_league_settings(FIXTURES / "espn" / "real" / "fba" / "mSettings.json")


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


def player_row(espn_id: int, name: str, position: str, team: int, **fields: object) -> PlayerRow:
    return PlayerRow.model_validate(
        {
            "sport": "nba",
            "espn_id": espn_id,
            "full_name": name,
            "default_position_id": FBA.position_id(position),
            "position": position,
            "pro_team_id": team,
            "eligible_slot_ids": sorted(NBA.eligible_slots(position, include_reserve=False)),
            "as_of": SYNCED,
            **fields,
        }
    )


def seed(
    store: Store, settings: LeagueSettings | None = None, *, opponent: bool = False, day: int = DAY, **updates: object
) -> LeagueRow:
    """Store a synced NBA league: settings, our roster for ``day`` (and the opponent's, for categories), the players
    and each player's blended per-game line for the day."""
    league_settings = (settings or real_points()).model_copy(update=updates) if updates else settings or real_points()
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
    entries = []
    for espn_id, name, position, team, slot, ppg in ROSTER:
        store.players.upsert(player_row(espn_id, name, position, team))
        entries.append(
            RosterEntryRow(
                league_id=league.row_id,
                scoring_period_id=day,
                team_id=OUR_TEAM,
                espn_id=espn_id,
                lineup_slot_id=slot,
                as_of=SYNCED,
            )
        )
        store.projections.upsert(blend_row(espn_id, day, line_for(ppg, league_settings)))
    store.rosters.replace(league.row_id, day, OUR_TEAM, entries)
    if opponent:
        theirs = []
        for espn_id, name, position, team, slot, pts, reb, stl in OPPONENT:
            store.players.upsert(player_row(espn_id, name, position, team))
            theirs.append(
                RosterEntryRow(
                    league_id=league.row_id,
                    scoring_period_id=day,
                    team_id=THEIR_TEAM,
                    espn_id=espn_id,
                    lineup_slot_id=slot,
                    as_of=SYNCED,
                )
            )
            store.projections.upsert(blend_row(espn_id, day, {"PTS": pts, "REB": reb, "STL": stl, "GP": 1.0}))
        store.rosters.replace(league.row_id, day, THEIR_TEAM, theirs)
    return league


def line_for(ppg: float, settings: LeagueSettings) -> dict[str, float]:
    """A per-game line worth ``ppg`` points under the points league, with a plausible category shape under 9-cat."""
    if settings.is_points:
        return {"PTS": ppg, "GP": 1.0}
    return {"PTS": ppg, "REB": ppg / 4, "AST": ppg / 5, "STL": ppg / 20, "BLK": ppg / 30, "GP": 1.0}


def blend_row(espn_id: int, day: int, stats: Mapping[str, float]) -> ProjectionRow:
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


# Day 3 (Thursday): our center (team 5), the third UTIL (team 10) and the SG (team 2) are off; both of the first two
# bench players (teams 11 and 12) play. Days 4-6 everyone plays.
WEEK = everyone([1, 2, 3, 4, 5, 6], except_={3: [2, 5, 10]})


def schedule(**options: object) -> FakeSchedule:
    return FakeSchedule(WEEK, **options)  # type: ignore[arg-type]


def test_the_matchup_week_comes_from_the_calendar_not_the_weekly_ids() -> None:
    settings = real_points()
    assert settings.schedule.matchup_period_for(3) is None  # the settings alone cannot place a day
    assert matchup_window(settings, 3) == (1, (3, 4, 5, 6))
    assert matchup_window(settings, 7) == (2, (7, 8, 9, 10, 11, 12, 13))
    assert matchup_window(settings, 7, horizon=2) == (2, (7, 8))
    assert matchup_window(settings, 1) == (1, (1, 2, 3, 4, 5, 6))
    with pytest.raises(DailyLineupError, match="no ESPN calendar"):
        matchup_window(settings.model_copy(update={"season": 2099}), 3)


def test_the_plan_fills_the_days_open_slots_from_the_bench(store: Store) -> None:
    league = seed(store)
    decision = plan_daily_lineup(store, league, schedule=schedule(), now=MORNING, period=DAY)
    assert decision.inputs.days == (3, 4, 5, 6) and decision.inputs.matchup_period == 1
    today = decision.best.day(DAY)
    # SG (team 2), C (team 5) and the UTIL on team 10 have no game: the bench's guards and forward take what they can.
    assert {move.espn_id for move in today.moves} == {102, 110, 111, 112}  # the third bench guard is not needed
    assert today.slots[111] in (SG, G, UTIL) and today.slots[112] in (UTIL, PF, F) and today.slots[113] == BENCH
    assert today.open_slots == (C,)  # nobody else is a center
    assert today.idle_starters == (105,)  # the center sits in C: nobody can fill it
    assert decision.best.open_slots_by_day() == {DAY: (C,)}
    assert decision.current.day(DAY).value == pytest.approx(30 + 26 + 24 + 20 + 18 + 16 + 14)
    assert decision.best.total > decision.current.total
    for later in (4, 5, 6):  # every team plays: the lineup as it stands is the best there is
        assert decision.best.day(later).moves == ()


def test_the_decision_proposes_the_target_days_drafts_once(store: Store) -> None:
    league = seed(store)
    config = config_for(real_points(), max_transactions_per_week=3)
    first = propose_daily_lineup(store, config, league, schedule=schedule(), now=MORNING, period=DAY)
    kinds = [row.kind for row in first.proposals]
    assert kinds == [ProposalKind.BENCH_INACTIVE.value, ProposalKind.LINEUP.value][: len(kinds)] and kinds
    assert first.blocked == ()
    for row in first.proposals:
        assert row.created_by == DAILY_LINEUP_CREATED_BY and row.scoring_period_id == DAY
        assert row.deadline == FIRST_TIP - timedelta(0)  # the earliest lock among the moved: the 7:30 p.m. ET tip
        payload = parse_payload(row)
        assert isinstance(payload, LineupPayload)
        sides = {(move.espn_id, move.from_slot_id, move.to_slot_id) for move in payload.moves}
        assert (102, SG, BENCH) in sides  # the SG without a game goes to the bench
    again = propose_daily_lineup(store, config, league, schedule=schedule(), now=MORNING, period=DAY)
    assert [row.row_id for row in again.proposals] == [row.row_id for row in first.proposals]  # deduplicated
    ahead = plan_daily_lineup(store, league, schedule=schedule(), now=MORNING, period=DAY, days_ahead=2)
    assert {draft.scoring_period_id for draft in ahead.drafts} == {DAY}  # later days need no moves
    assert plan_daily_lineup(store, league, schedule=schedule(), now=MORNING, period=DAY, horizon=1).inputs.days == (3,)


def test_a_policy_refusal_is_reported_not_raised(store: Store) -> None:
    league = seed(store)
    config = config_for(real_points(), bench_inactive="off", lineup="off")
    result = propose_daily_lineup(store, config, league, schedule=schedule(), now=MORNING, period=DAY)
    assert result.proposals == () and len(result.blocked) == len(result.decision.drafts) >= 1
    assert all("is off for league" in message for message in result.blocked)


def test_a_locked_player_stays_and_a_tipped_off_game_cannot_be_changed(store: Store) -> None:
    league = seed(store)
    after_tip = FIRST_TIP + timedelta(minutes=5)  # every team's game tipped off at once
    decision = plan_daily_lineup(store, league, schedule=schedule(), now=after_tip, period=DAY)
    assert decision.best.day(DAY).moves == () and decision.drafts == ()
    hurt = store.players.get("nba", 101)
    assert hurt is not None
    store.players.upsert(hurt.model_copy(update={"injury_status": "OUT"}))
    free = plan_daily_lineup(store, league, schedule=schedule(), now=MORNING, period=DAY)
    assert free.best.day(DAY).slots[101] == BENCH  # an OUT starter is benched...
    flagged = store.rosters.team(league.row_id, DAY, OUR_TEAM)
    store.rosters.replace(
        league.row_id,
        DAY,
        OUR_TEAM,
        [entry.model_copy(update={"lineup_locked": entry.espn_id == 101}) for entry in flagged],
    )
    held = plan_daily_lineup(store, league, schedule=schedule(), now=MORNING, period=DAY)
    assert held.best.day(DAY).slots[101] == PG  # ...unless ESPN says his slot is locked


def report(*entries: tuple[str, str, str]) -> OfficialInjuryReport:
    """An official report for day 3 listing ``(player as printed, team name, status)``."""
    return OfficialInjuryReport(
        entries=tuple(
            OfficialInjuryEntry(
                game_day=date(2026, 10, 22),
                game_time="07:30 (ET)",
                matchup="ATL@BOS",
                team=team,
                player=player,
                status=status,
                reason="Injury Management",
            )
            for player, team, status in entries
        )
    )


def test_the_official_report_overrides_espns_raw_out_designation(store: Store) -> None:
    league = seed(store)
    row = store.players.get("nba", 101)
    assert row is not None
    store.players.upsert(row.model_copy(update={"injury_status": "OUT"}))  # ESPN still shows our PG out
    without = plan_daily_lineup(store, league, schedule=schedule(), now=MORNING, period=DAY)
    assert without.best.day(DAY).slots[101] == BENCH  # ESPN's designation: zero, so benched
    cleared = report(("Guard, Pat", "Atlanta Hawks", "Available"))
    with_report = plan_daily_lineup(store, league, schedule=schedule(), now=MORNING, period=DAY, official=cleared)
    assert with_report.best.day(DAY).slots[101] == PG  # the report clears him: he starts
    assert with_report.inputs.players[0].day(DAY).p_active == 1.0
    out = report(("Guard, Pat", "Atlanta Hawks", "Out"))
    benched = plan_daily_lineup(store, league, schedule=schedule(), now=MORNING, period=DAY, official=out)
    assert benched.best.day(DAY).slots[101] == BENCH
    # The row the player's designation sets: a stored row wins when it is no older than his synced status.
    stored = assess(row, season=SEASON, scoring_period=DAY, as_of=MORNING, schedule=schedule())
    store.availability.upsert(stored.model_copy(update={"p_active": 0.0}))
    assert (
        plan_daily_lineup(store, league, schedule=schedule(), now=MORNING, period=DAY).best.day(DAY).slots[101] == BENCH
    )


def test_a_questionable_designation_counts_at_his_chance_of_playing(store: Store) -> None:
    league = seed(store)
    row = store.players.get("nba", 111)
    assert row is not None
    store.players.upsert(row.model_copy(update={"injury_status": "QUESTIONABLE"}))
    inputs = daily_inputs(store, league, real_points(), schedule=schedule(), period=DAY, now=MORNING)
    ben = next(player for player in inputs.players if player.espn_id == 111)
    assert ben.day(DAY).p_active == pytest.approx(0.5) and ben.day(DAY).value == pytest.approx(0.5 * 8.0)


def test_the_lock_rules_the_planner_obeys_come_from_the_league(store: Store) -> None:
    league = seed(store)
    first_game = real_points().model_copy(update={"lineup_lock_type": LockType.FIRSTGAME_SCORINGPERIOD})
    inputs = daily_inputs(store, league, first_game, schedule=schedule(), period=DAY, now=MORNING)
    assert {player.day(DAY).lock_at for player in inputs.players} == {FIRST_TIP}  # everyone locks at the first tip
    with pytest.raises(DailyLineupError, match="UNKNOWN"):
        plan_daily_lineup(
            store,
            league,
            schedule=schedule(),
            now=MORNING,
            period=DAY,
            settings=real_points().model_copy(update={"lineup_lock_type": LockType.UNKNOWN}),
        )


def test_the_plan_refuses_what_it_cannot_plan(store: Store) -> None:
    league = seed(store)
    with pytest.raises(DailyLineupError, match="no roster"):
        plan_daily_lineup(store, league, schedule=schedule(), now=MORNING, period=4)  # nothing synced for day 4
    with pytest.raises(DailyLineupError, match="no ESPN calendar"):
        plan_daily_lineup(
            store,
            league,
            schedule=schedule(),
            now=MORNING,
            period=DAY,
            settings=real_points().model_copy(update={"season": 2099}),
        )
    with pytest.raises(ValueError, match="aware"):
        plan_daily_lineup(store, league, schedule=schedule(), now=MORNING.replace(tzinfo=None), period=DAY)
    other = store.leagues.upsert(league.model_copy(update={"key": "nfl", "sport": "nfl"}))
    with pytest.raises(DailyLineupError, match="NBA"):
        daily_inputs(store, other, real_points(), schedule=schedule(), period=DAY, now=MORNING)
    bare = LeagueRow(key="x", sport="nba", espn_league_id=1, season=SEASON, team_id=OUR_TEAM, as_of=SYNCED)
    with pytest.raises(DailyLineupError, match="no synced settings"):
        plan_daily_lineup(store, store.leagues.upsert(bare), schedule=schedule(), now=MORNING, period=DAY)


def test_the_games_played_limit_of_the_league_is_read_from_its_settings(store: Store) -> None:
    nine = _settings("fba_settings_9cat.json")
    limits, warnings = slot_limits_left(nine)
    assert limits == {PG: 82, SG: 82, SF: 82, PF: 82, C: 82, G: 82, F: 82, UTIL: 246}  # the season's caps, none spent
    assert warnings and "games used" in warnings[0]
    left, quiet = slot_limits_left(nine, {PG: 80, UTIL: 300})
    assert left[PG] == 2 and left[UTIL] == 0 and quiet == ()
    assert slot_limits_left(real_points()) == ({}, ())


def test_a_category_league_weights_each_game_by_its_swing_against_the_opponent(store: Store) -> None:
    nine = _settings("fba_settings_9cat.json")
    league = seed(store, nine, opponent=True)
    flat = plan_daily_lineup(store, league, schedule=schedule(), now=MORNING, period=DAY)
    assert flat.inputs.values.outlook is None and any("flat" in note for note in flat.warnings)
    versus = plan_daily_lineup(store, league, schedule=schedule(), now=MORNING, period=DAY, opponent_team_id=THEIR_TEAM)
    outlook = versus.inputs.values.outlook
    assert outlook is not None and set(outlook.weights) == set(nine.categories)
    assert versus.inputs.values.unit == "category score"
    assert all(weight >= 0 for weight in outlook.weights.values())
    assert 0.0 < outlook.expected_wins < len(nine.categories)
    # Our deep roster outscores a three-man opponent in volume categories, which are then close to decided.
    assert outlook.win_probability["PTS"] > 0.99 and outlook.weights["PTS"] < 1e-6  # decided: worth nothing more
    assert outlook.win_probability["3PM"] == pytest.approx(0.5) and outlook.weights["3PM"] > 0.05  # a toss-up swings
    assert versus.best.total > 0
    gone = plan_daily_lineup(store, league, schedule=schedule(), now=MORNING, period=DAY, opponent_team_id=99)
    assert gone.inputs.values.outlook is None and any("no roster for team 99" in note for note in gone.warnings)


def test_the_decision_is_registered_for_nba_beside_the_nfl_lineup() -> None:
    assert decide_registry.lookup("nba", DAILY_LINEUP_KIND) is propose_daily_lineup
    assert DAILY_LINEUP_KIND == "lineup_daily"
    # fm.jobs.tick calls ``fn(store, config, league_row, schedule=..., now=..., opponent_team_id=...)`` with what
    # the signature accepts.
    parameters = inspect.signature(propose_daily_lineup).parameters
    assert list(parameters)[:3] == ["store", "config", "league"]
    assert {"schedule", "now", "opponent_team_id"} <= set(parameters)
    assert "lineup" in [entry.kind for entry in decide_registry.registered("nfl")]
