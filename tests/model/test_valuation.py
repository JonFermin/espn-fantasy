"""Points-league valuation (ROADMAP #21): which ESPN season line is rest-of-season, the horizon and its playoff
weights, weekly expected points, the lineup a week fields, replacement level, value over replacement, and the change a
move makes to a roster's rest-of-season value.

The rest-of-season question is settled on the real week-4 2026 captures (``tests/fixtures/espn/real/ffl``: the
free-agent pool, the player cards, the pro schedule's bye weeks and ESPN's calendar). League settings are the hand-built
PPR league (``ffl_settings_ppr.json``: $100 FAAB, weeks 15-17 playoffs) and the real NFL league (weeks 14-17
playoffs). Lineup and roster arithmetic runs on hand-made outlooks, so every expected number is visible in the test.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from fm.espn.ids import FFL
from fm.espn.models import STAT_SOURCE_PROJECTED, PlayersView, ProSchedule
from fm.espn.settings import LeagueSettings, load_league_settings
from fm.model.availability import assess
from fm.model.projections import BLEND, BlendWeights
from fm.model.scoring import Scorer
from fm.model.valuation import (
    BASIS_NONE,
    BASIS_SEASON,
    BASIS_SEASON_WITHOUT_GAMES,
    BASIS_THIS_WEEK,
    DEFAULT_PLAYOFF_WEIGHT,
    GAMES_STAT,
    Horizon,
    PlayerOutlook,
    RosterValuer,
    ValuationError,
    eligible_active_slots,
    last_scoring_period,
    league_settings,
    load_valuation,
    project_outlook,
    slot_instances,
    start_value,
)
from fm.sports.base import StatSchema
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
NOW = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)
SEASON, WEEK = 2026, 4

QB, RB, WR, TE, DST, K, FLEX = (FFL.slot_id(label) for label in ("QB", "RB", "WR", "TE", "D/ST", "K", "FLEX"))
BENCH, IR = FFL.bench_slot, FFL.ir_slot
LINEUP = (QB, RB, RB, WR, WR, TE, FLEX)
"""A lineup without the kicker and defense, so the arithmetic below stays short."""
EQUAL_WEIGHTS = BlendWeights.parse("[nfl.default]\nespn = 1.0\nsleeper = 1.0\n\n[nba.default]\nespn = 1.0\n")


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ppr() -> LeagueSettings:
    return load_league_settings(PPR)


def real_nfl() -> LeagueSettings:
    return load_league_settings(REAL / "ffl" / "mSettings.json")


def horizon(periods: Iterable[int], *, playoff: Iterable[int] = (), weight: float = 2.0) -> Horizon:
    """A horizon over ``periods`` (the first is current) with ``playoff`` periods weighted ``weight``."""
    span = tuple(periods)
    playoffs = frozenset(playoff)
    weights = {period: weight if period in playoffs else 1.0 for period in span}
    return Horizon(span[0], span, MappingProxyType(weights), playoffs, True, weight)


def outlook(
    espn_id: int, slots: Iterable[int], weekly: Mapping[int, float], *, basis: str = BASIS_SEASON
) -> PlayerOutlook:
    return PlayerOutlook(
        espn_id=espn_id,
        name=f"Player {espn_id}",
        position=None,
        pro_team_id=1,
        slots=frozenset(slots),
        weekly=MappingProxyType(dict(weekly)),
        basis=basis,
    )


def flat(espn_id: int, slots: Iterable[int], points: float, periods: Iterable[int] = (4, 5, 6)) -> PlayerOutlook:
    return outlook(espn_id, slots, dict.fromkeys(periods, points))


def kickoff(period: int) -> datetime:
    """A Sunday 1 p.m. ET kickoff in ``period`` (week 1 is Sep 13, 2026)."""
    return datetime(2026, 9, 13, 17, 0, tzinfo=UTC) + timedelta(weeks=period - 1)


def pro_schedule(byes: Mapping[int, int], periods: Iterable[int] = range(1, 19)) -> ProSchedule:
    """A pro schedule where each team in ``byes`` (team -> bye week) plays every period but its bye."""
    teams = []
    for team, bye in byes.items():
        games = {
            str(period): [
                {
                    "id": team * 1000 + period,
                    "date": int(kickoff(period).timestamp() * 1000),
                    "scoringPeriodId": period,
                    "homeProTeamId": team,
                    "awayProTeamId": 99,
                }
            ]
            for period in periods
            if period != bye
        }
        teams.append({"id": team, "proGamesByScoringPeriod": games})
    return ProSchedule.model_validate({"proTeams": teams})


# --- which season line is rest-of-season ------------------------------------------------------------------------------


def last_regular_season_week() -> int:
    """The NFL regular season's last scoring period, from ESPN's own calendar (its season-long period type)."""
    calendar = load(REAL / "ffl" / "calendar.json")
    season_long = next(kind for kind in calendar["periodTypes"] if kind["seasonLong"])
    return min(period["scoringPeriodEnd"] for period in season_long["periods"])


def season_lines() -> dict[str, tuple[float, int]]:
    """Name -> (``GP`` of ESPN's ``102026`` line, games his team has left after week 4), from the real captures."""
    byes = {
        team["id"]: team["byeWeek"] for team in load(REAL / "ffl" / "proTeamSchedules_wl.json")["settings"]["proTeams"]
    }
    current = load(REAL / "ffl" / "mSettings.json")["scoringPeriodId"]
    last = last_regular_season_week()
    gp = FFL.stat_id(GAMES_STAT)
    found: dict[str, tuple[float, int]] = {}
    for name in ("kona_player_info.json", "kona_playercard.json"):
        for entry in PlayersView.model_validate(load(REAL / "ffl" / name)).players:
            player = entry.player
            line = player.projection(SEASON, 0)
            assert line is not None and line.id == "102026"
            assert player.pro_team_id is not None
            left = sum(1 for week in range(current + 1, last + 1) if week != byes[player.pro_team_id])
            found[player.full_name] = (line.stats[gp], left)
    return found


def test_the_split_0_season_line_is_espns_rest_of_season_projection() -> None:
    """``102026``'s games are each healthy player's games left (weeks 5-18 less his bye) and fewer only for the
    injured; the line ``Player.projection(season, 0)`` reads, and the sync stores as period 0, is rest-of-season."""
    assert last_regular_season_week() == 18
    lines = season_lines()
    assert set(lines) == {
        "DeVonta Smith",
        "A.J. Brown",
        "Colston Loveland",
        "Jalen Coker",
        "Chase Brown",
        "Rashee Rice",
    }
    assert all(games <= left for games, left in lines.values())
    assert {name for name, (games, left) in lines.items() if games == left} == {
        "Colston Loveland",
        "Jalen Coker",  # OUT in week 4, whose game was already played: all 13 games left
        "Chase Brown",
        "Rashee Rice",
    }
    assert lines["DeVonta Smith"] == (12.0, 13)  # OUT
    assert lines["A.J. Brown"] == (10.0, 13)  # on IR


def test_the_split_2_line_labelled_rest_of_season_projects_more_games_than_are_left() -> None:
    smith = PlayersView.model_validate(load(REAL / "ffl" / "kona_player_info.json")).players[0].player
    assert smith.full_name == "DeVonta Smith"
    second = next(
        line
        for line in smith.stats
        if line.stat_source_id == STAT_SOURCE_PROJECTED and line.scoring_period_id == 0 and line.stat_split_type_id == 2
    )
    assert second.id == "122026"
    games, left = season_lines()["DeVonta Smith"]
    assert second.stats[FFL.stat_id(GAMES_STAT)] == 16 > left > games


def test_the_stored_season_line_carries_its_games() -> None:
    """The sync stores ESPN's lines keyed by abbreviation; ``GP`` is what the valuation divides by."""
    card = PlayersView.model_validate(load(REAL / "ffl" / "kona_playercard.json")).players[0].player
    line = card.projection(SEASON, 0)
    assert line is not None
    stored = StatSchema.for_game("ffl").from_espn(line.stats)
    assert stored[GAMES_STAT] == 13.0


# --- the horizon ------------------------------------------------------------------------------------------------------


def test_the_horizon_runs_to_the_last_fantasy_week_with_the_playoffs_weighted_up() -> None:
    season = Horizon.for_league(real_nfl())
    assert season.current == WEEK
    assert season.periods == tuple(range(4, 18))
    assert season.future == tuple(range(5, 18))
    assert season.playoff_periods == frozenset({14, 15, 16, 17})
    assert season.playoff_weight == DEFAULT_PLAYOFF_WEIGHT > 1
    assert [season.weight(period) for period in season.periods] == [1.0] * 10 + [DEFAULT_PLAYOFF_WEIGHT] * 4
    assert season.weight(3) == season.weight(18) == 0.0  # outside the horizon
    assert season.since(15) == (15, 16, 17)


def test_the_last_week_is_the_last_one_a_matchup_lists() -> None:
    settings = ppr()  # finalScoringPeriod 18, matchups end at period 17, playoffs 15-17
    assert settings.final_scoring_period == 18
    assert last_scoring_period(settings) == 17
    doubled = Horizon.for_league(settings, current=10, playoff_weight=2.0)
    assert doubled.periods == tuple(range(10, 18))
    assert {period: doubled.weight(period) for period in (14, 15, 17)} == {14: 1.0, 15: 2.0, 17: 2.0}


def test_past_the_last_week_the_horizon_is_empty() -> None:
    assert Horizon.for_league(ppr(), current=18).periods == ()


def test_a_league_whose_matchups_are_weeks_of_days_weights_nothing_up() -> None:
    settings = load_league_settings(REAL / "fba" / "mSettings.json")
    season = Horizon.for_league(settings, current=1)
    assert not season.playoffs_known
    assert season.periods[-1] == settings.final_scoring_period
    assert set(season.weights.values()) == {1.0}


def test_the_horizon_refuses_what_it_cannot_place() -> None:
    with pytest.raises(ValueError, match="playoff_weight"):
        Horizon.for_league(ppr(), playoff_weight=-1.0)
    with pytest.raises(ValueError, match="playoff_weight"):
        Horizon.for_league(ppr(), playoff_weight=math.nan)
    with pytest.raises(ValuationError, match="current scoring period"):
        Horizon.for_league(ppr().model_copy(update={"current_scoring_period": None}))


# --- one player's weeks -----------------------------------------------------------------------------------------------


def wr_row(**changes: Any) -> PlayerRow:
    row = PlayerRow(
        sport="nfl",
        espn_id=7,
        full_name="Seven Receiver",
        default_position_id=FFL.position_id("WR"),
        position="WR",
        pro_team_id=2,
        eligible_slot_ids=[WR, FLEX, BENCH, IR],
        as_of=NOW,
    )
    return row.model_copy(update=changes)


SEASON_LINE = {"REC": 60.0, "REY": 650.0, GAMES_STAT: 10.0}  # 60 + 65 = 125 PPR points over 10 games
WEEK_LINE = {"REC": 5.0, "REY": 60.0}  # 11 PPR points


def test_later_weeks_are_the_season_line_per_game_times_games_and_health() -> None:
    scorer = Scorer(ppr())
    season = Horizon.for_league(real_nfl())
    result = project_outlook(
        wr_row(),
        horizon=season,
        scorer=scorer,
        slots=frozenset({WR, FLEX}),
        current_line=WEEK_LINE,
        season_line=SEASON_LINE,
        p_active=0.5,
        schedule=pro_schedule({2: 9}),
    )
    left = 13  # weeks 5-18 less the week-9 bye: the season line spans past the fantasy season's week 17
    assert scorer.points(SEASON_LINE) == pytest.approx(125.0)
    assert result.basis == BASIS_SEASON
    assert result.per_game == pytest.approx(12.5)
    assert result.games == 10.0
    assert result.health == pytest.approx(10 / left)
    assert result.expected(4) == pytest.approx(0.5 * 11.0)  # this week: p_active times this week's line
    assert result.expected(9) == 0.0  # bye
    assert result.expected(5) == result.expected(17) == pytest.approx(12.5 * 10 / left)
    expected = math.fsum(season.weight(period) * result.expected(period) for period in season.periods)
    assert result.value(season) == pytest.approx(expected)
    assert result.value(season, start=14) == pytest.approx(4 * DEFAULT_PLAYOFF_WEIGHT * 12.5 * 10 / left)


def test_health_never_exceeds_one_and_without_a_schedule_every_week_has_a_game() -> None:
    season = Horizon.for_league(real_nfl())
    common: dict[str, Any] = {"horizon": season, "scorer": Scorer(ppr()), "slots": frozenset({WR})}
    healthy = project_outlook(wr_row(), season_line={**SEASON_LINE, GAMES_STAT: 20.0}, **common)
    assert healthy.health == 1.0
    assert healthy.expected(9) == pytest.approx(125.0 / 20)
    unscheduled = project_outlook(wr_row(), season_line=SEASON_LINE, **common)
    assert unscheduled.health == pytest.approx(10 / len(season.future))  # 13 weeks 5-17 assumed to have a game


def test_a_season_line_without_games_is_spread_over_the_games_left() -> None:
    result = project_outlook(
        wr_row(),
        horizon=Horizon.for_league(real_nfl()),
        scorer=Scorer(ppr()),
        slots=frozenset({WR}),
        season_line={"REC": 60.0, "REY": 650.0},
        schedule=pro_schedule({2: 9}),
    )
    assert result.basis == BASIS_SEASON_WITHOUT_GAMES
    assert result.games is None
    assert result.per_game == pytest.approx(125.0 / 13)
    assert result.health == 1.0


def test_without_a_season_line_this_weeks_projection_stands_in_and_without_either_nothing_is_known() -> None:
    common: dict[str, Any] = {"horizon": Horizon.for_league(real_nfl()), "scorer": Scorer(ppr()), "slots": frozenset()}
    stand_in = project_outlook(wr_row(), current_line=WEEK_LINE, **common)
    assert stand_in.basis == BASIS_THIS_WEEK
    assert stand_in.expected(10) == pytest.approx(11.0)
    assert stand_in.has_projection
    for empty in (None, {}):
        unknown = project_outlook(wr_row(), current_line=empty, **common)
        assert unknown.basis == BASIS_NONE
        assert not unknown.has_projection
        assert unknown.value(common["horizon"]) == 0.0


def test_a_player_out_for_the_season_is_known_to_be_worth_nothing() -> None:
    gone = project_outlook(
        wr_row(),
        horizon=Horizon.for_league(real_nfl()),
        scorer=Scorer(ppr()),
        slots=frozenset({WR}),
        current_line={},
        season_line={GAMES_STAT: 0.0},
    )
    assert gone.has_projection and gone.basis == BASIS_SEASON
    assert gone.value(Horizon.for_league(real_nfl())) == 0.0


@pytest.mark.parametrize("changes", [{"pro_team_id": 0}, {"pro_team_id": None}, {"active": False}])
def test_a_player_without_a_pro_team_or_active_roster_spot_scores_nothing(changes: dict[str, Any]) -> None:
    result = project_outlook(
        wr_row(**changes),
        horizon=Horizon.for_league(real_nfl()),
        scorer=Scorer(ppr()),
        slots=frozenset({WR}),
        current_line=WEEK_LINE,
        season_line=SEASON_LINE,
    )
    assert set(result.weekly.values()) == {0.0}


def test_p_active_must_be_a_probability() -> None:
    with pytest.raises(ValueError, match="p_active"):
        project_outlook(
            wr_row(), horizon=Horizon.for_league(real_nfl()), scorer=Scorer(ppr()), slots=frozenset(), p_active=1.2
        )


# --- lineup slots -----------------------------------------------------------------------------------------------------


def test_slot_instances_are_the_leagues_active_slots_one_per_slot() -> None:
    assert slot_instances(ppr()) == (QB, RB, RB, WR, WR, TE, DST, K, FLEX)


def test_eligible_slots_prefer_espns_list_and_fall_back_to_the_position_rule() -> None:
    settings = ppr()
    listed = wr_row(eligible_slot_ids=[RB, FFL.slot_id("RB/WR"), FLEX, FFL.slot_id("OP"), BENCH, IR])
    assert eligible_active_slots(listed, settings) == {RB, FLEX}  # RB/WR and OP are not in this league
    assert eligible_active_slots(wr_row(eligible_slot_ids=[], position="TE"), settings) == {TE, FLEX}
    assert eligible_active_slots(wr_row(eligible_slot_ids=[], position=None), settings) == frozenset()
    defense = wr_row(espn_id=-16021, eligible_slot_ids=[], position=None)  # a D/ST id with no position stored
    assert eligible_active_slots(defense, settings) == {DST}


# --- one week's lineup ------------------------------------------------------------------------------------------------


ROSTER = (
    flat(1, {QB}, 20),
    flat(2, {RB, FLEX}, 15),
    flat(3, {RB, FLEX}, 12),
    flat(4, {RB, FLEX}, 9),
    flat(5, {WR, FLEX}, 14),
    flat(6, {WR, FLEX}, 13),
    flat(7, {WR, FLEX}, 10),
    flat(8, {TE, FLEX}, 8),
)


def test_the_best_lineup_fills_every_slot_by_eligibility() -> None:
    week = start_value(ROSTER, LINEUP, 4)
    assert week.total == week.points == 20 + 15 + 12 + 14 + 13 + 8 + 10  # FLEX: the 10-point receiver over 9
    assert week.filled == 0.0 and week.fills == ()
    assert set(week.starters) == {(QB, 1), (RB, 2), (RB, 3), (WR, 5), (WR, 6), (TE, 8), (FLEX, 7)}


def test_a_slot_the_roster_cannot_fill_takes_the_best_wire_player_who_fits() -> None:
    bye = (*ROSTER[:-1], outlook(8, {TE, FLEX}, {4: 0.0}))  # the tight end is on bye
    wire = (flat(101, {TE, FLEX}, 7), flat(102, {TE, FLEX}, 6), flat(103, {RB, FLEX}, 30))
    week = start_value(bye, LINEUP, 4, fill=wire)
    assert week.fills == ((TE, 101),)
    assert week.filled == 7.0
    assert week.points == 20 + 15 + 12 + 14 + 13 + 10  # the 30-point wire back displaces nobody on the roster


def test_holes_are_filled_by_distinct_wire_players() -> None:
    no_backs = tuple(player for player in ROSTER if RB not in player.slots and player.espn_id != 7)
    wire = (flat(101, {RB, FLEX}, 9), flat(102, {RB, FLEX}, 8), flat(103, {WR, FLEX}, 7))
    week = start_value(no_backs, LINEUP, 4, fill=wire)
    assert sorted(week.fills) == [(RB, 101), (RB, 102), (FLEX, 103)]
    assert week.filled == 9 + 8 + 7


def test_a_player_projected_at_or_below_zero_never_starts() -> None:
    negative = (outlook(1, {QB}, {4: -1.5}), *ROSTER[1:])
    week = start_value(negative, LINEUP, 4, fill=(flat(101, {QB}, 6),))
    assert (QB, 1) not in week.starters
    assert week.fills == ((QB, 101),)


# --- replacement level, value over replacement, and moves -------------------------------------------------------------

PERIODS = (4, 5, 6)


def valuer(
    roster: Iterable[PlayerOutlook] = ROSTER, wire: Iterable[PlayerOutlook] = (), season: Horizon | None = None
) -> RosterValuer:
    rostered, available = tuple(roster), tuple(wire)
    return RosterValuer(
        {player.espn_id: player for player in (*rostered, *available)},
        roster=[player.espn_id for player in rostered],
        wire=[player.espn_id for player in available],
        slots=LINEUP,
        horizon=season if season is not None else horizon(PERIODS),
        labels={QB: "QB", RB: "RB", WR: "WR", TE: "TE", FLEX: "RB/WR/TE"},
    )


WIRE = (
    flat(101, {QB}, 14),
    flat(102, {QB}, 12),
    flat(103, {RB, FLEX}, 11),
    flat(104, {RB, FLEX}, 6),
    flat(105, {WR, FLEX}, 9),
    flat(106, {TE, FLEX}, 7),
)


def test_replacement_level_is_the_best_player_on_the_wire_for_each_slot() -> None:
    levels = valuer(wire=WIRE).replacement
    assert {slot: level.espn_id for slot, level in levels.items()} == {QB: 101, RB: 103, WR: 105, TE: 106, FLEX: 103}
    assert levels[QB].value == 3 * 14
    assert levels[FLEX].label == "RB/WR/TE"


def test_rostered_players_never_set_the_replacement_level() -> None:
    levels = valuer(wire=(flat(105, {WR, FLEX}, 9),)).replacement
    assert levels[RB].espn_id is None and levels[RB].value == 0.0  # the roster's backs are not on the wire
    assert levels[FLEX].espn_id == 105


def test_value_over_replacement_is_measured_at_the_players_own_position() -> None:
    values = valuer(wire=WIRE)
    assert values.vor(2) == pytest.approx(3 * (15 - 11))  # against the RB replacement, not FLEX's
    assert values.vor(5) == pytest.approx(3 * (14 - 9))
    assert values.vor(103) == 0.0  # the replacement himself
    assert values.vor(104) == pytest.approx(3 * (6 - 11))


def test_a_move_gains_the_change_in_the_rosters_start_value() -> None:
    values = valuer(wire=WIRE)
    assert values.value() == pytest.approx(3 * 92)
    assert values.gain(103) == pytest.approx(3 * (11 - 10))  # takes FLEX from the 10-point receiver
    assert values.gain(103, 4) == pytest.approx(3 * (11 - 10))  # the bench back never started
    assert values.gain(drop=2) == pytest.approx(3 * (9 - 15))  # the 9-point back comes off the bench
    assert values.gain() == 0.0


def test_a_claimed_player_is_worth_his_edge_over_the_next_best_on_the_wire() -> None:
    bye = (outlook(1, {QB}, {4: 20, 5: 0, 6: 20}), *ROSTER[1:])  # the quarterback's bye is week 5
    values = valuer(bye, WIRE)
    assert values.gain(101) == pytest.approx(14 - 12)  # in week 5 the 12-point quarterback would have filled in
    assert values.gain(102) == pytest.approx(12 - 14)  # holding the worse one keeps the better one off the field


def test_a_move_counts_only_from_the_week_it_takes_effect() -> None:
    values = valuer(wire=WIRE)
    assert values.gain(103, start=5) == pytest.approx(2 * (11 - 10))
    assert values.gain(103, start=7) == 0.0


def test_a_drop_made_now_loses_his_games_before_the_add_counts() -> None:
    values = valuer(wire=WIRE)
    together = values.gain(103, 7, start=5)  # the flex receiver leaves when the back arrives
    early = values.gain(103, 7, start=5, drop_start=4)  # he leaves now; the back counts from next week
    assert together == pytest.approx(2 * (11 - 10))
    assert early == pytest.approx(together - (10 - 9))  # this week the 9-point back takes his FLEX spot
    planned = values.after(103, 7, start=5, drop_start=4)
    assert planned.roster_at(4) == {player.espn_id for player in ROSTER} - {7}
    assert planned.roster_at(5) == planned.roster == ({player.espn_id for player in ROSTER} - {7}) | {103}


def test_a_hole_is_filled_from_the_wire_only_after_this_week() -> None:
    bye = (outlook(1, {QB}, {4: 0, 5: 0, 6: 20}), *ROSTER[1:])  # the quarterback has no game this week or next
    values = valuer(bye, WIRE)
    assert values.lineup(4).fills == ()  # the wire's quarterbacks may have played already
    assert values.lineup(5).fills == ((QB, 101),)
    assert values.value() == pytest.approx((92 - 20) + (92 - 20 + 14) + 92)
    # This week the hole is his alone; next week, rostered, he starts where the 14-point 101 would have streamed in.
    assert values.gain(102) == pytest.approx(12 - (14 - 12))


def test_playoff_weeks_count_more() -> None:
    late = outlook(103, {RB, FLEX}, {4: 0.0, 5: 0.0, 6: 16.0})  # only plays in the playoff week
    flat_season, playoff_season = horizon(PERIODS, playoff=(6,), weight=1.0), horizon(PERIODS, playoff=(6,), weight=2.0)
    assert valuer(wire=(late,), season=flat_season).gain(103) == pytest.approx(16 - 10)
    assert valuer(wire=(late,), season=playoff_season).gain(103) == pytest.approx(2 * (16 - 10))


def test_a_planned_move_changes_the_roster_from_its_start() -> None:
    values = valuer(wire=WIRE).after(103, 4, start=5)
    assert values.roster_at(4) == {player.espn_id for player in ROSTER}
    assert values.roster_at(5) == values.roster == ({player.espn_id for player in ROSTER} - {4}) | {103}
    assert 103 not in values.wire
    assert values.value() == pytest.approx(3 * 92 + 2 * (11 - 10))
    assert values.lineup(5).starters == tuple(
        sorted({(QB, 1), (RB, 2), (RB, 3), (WR, 5), (WR, 6), (TE, 8), (FLEX, 103)})
    )


def test_moves_must_come_from_the_wire_and_the_roster() -> None:
    values = valuer(wire=WIRE)
    with pytest.raises(ValueError, match="not on the wire"):
        values.gain(add=2)
    with pytest.raises(ValueError, match="not on the roster"):
        values.gain(drop=103)
    with pytest.raises(ValueError, match="no outlook"):
        RosterValuer({}, roster=[1], wire=[], slots=LINEUP, horizon=horizon(PERIODS))


# --- a league from the store ------------------------------------------------------------------------------------------

LEAGUE_ID, OUR_TEAM, OTHER_TEAM = 1234567, 1, 2
QB_A, WR_B, RB_C, WR_D, RB_E, RB_F, RB_G = 11, 12, 13, 14, 15, 16, 17


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        yield opened


def player(espn_id: int, position: str, team: int, **changes: Any) -> PlayerRow:
    slots = {"QB": [QB], "RB": [RB, FLEX], "WR": [WR, FLEX]}[position]
    row = PlayerRow(
        sport="nfl",
        espn_id=espn_id,
        full_name=f"Player {espn_id}",
        default_position_id=FFL.position_id(position),
        position=position,
        pro_team_id=team,
        eligible_slot_ids=[*slots, BENCH, IR],
        as_of=NOW,
    )
    return row.model_copy(update=changes)


def projected(espn_id: int, source: str, period: int, stats: dict[str, float]) -> ProjectionRow:
    return ProjectionRow(
        sport="nfl",
        espn_id=espn_id,
        source=source,
        season=SEASON,
        scoring_period_id=period,
        stats=stats,
        as_of=NOW,
    )


def seed(store: Store, settings: LeagueSettings | None = None) -> LeagueRow:
    """Our team: a quarterback ESPN and Sleeper both project, a questionable receiver, a back nothing projects, and a
    receiver with no ``players`` row. Team 2 holds back E; back F is on the wire; back G only Sleeper projects."""
    league = store.leagues.upsert(
        LeagueRow(key="nfl", sport="nfl", espn_league_id=LEAGUE_ID, season=SEASON, team_id=OUR_TEAM, as_of=NOW)
    )
    chosen = settings if settings is not None else ppr()
    store.settings.upsert(
        LeagueSettingsRow(league_id=league.row_id, settings=chosen.model_dump(mode="json"), as_of=NOW)
    )
    entries = [(OUR_TEAM, QB_A, QB), (OUR_TEAM, WR_B, WR), (OUR_TEAM, RB_C, BENCH), (OUR_TEAM, WR_D, BENCH)]
    entries.append((OTHER_TEAM, RB_E, RB))
    store.teams.upsert_many(
        TeamRow(league_id=league.row_id, team_id=team, name=f"Team {team}", as_of=NOW)
        for team in (OUR_TEAM, OTHER_TEAM)
    )
    for team in (OUR_TEAM, OTHER_TEAM):
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
                for owner, espn_id, slot in entries
                if owner == team
            ],
        )
    store.players.upsert_many(
        [
            player(QB_A, "QB", 2),
            player(WR_B, "WR", 4, injury_status="QUESTIONABLE"),
            player(RB_C, "RB", 8),
            player(RB_E, "RB", 1),
            player(RB_F, "RB", 3),
            player(RB_G, "RB", 9),
        ]
    )
    store.projections.upsert_many(
        [
            projected(QB_A, "espn", WEEK, {"PY": 250.0}),  # 10 points
            projected(QB_A, "sleeper", WEEK, {"PY": 300.0}),  # 12 points
            projected(QB_A, "espn", 0, {"PY": 3250.0, GAMES_STAT: 13.0}),  # 130 points over 13 games
            projected(WR_B, "espn", WEEK, {"REC": 5.0, "REY": 60.0}),  # 11 points
            projected(WR_B, "espn", 0, {"REC": 60.0, "REY": 650.0, GAMES_STAT: 10.0}),
            projected(RB_E, "espn", WEEK, {"RY": 100.0}),
            projected(RB_F, "espn", WEEK, {"RY": 80.0}),  # 8 points
            projected(RB_F, "espn", 0, {"RY": 1040.0, GAMES_STAT: 13.0}),  # 8 a game
            projected(RB_G, "sleeper", WEEK, {"RY": 90.0}),
            projected(RB_F, BLEND, WEEK, {"RY": 500.0}),  # an old blended row: recomputed, never read
        ]
    )
    return league


def test_a_synced_league_is_valued_from_its_stored_lines(store: Store) -> None:
    league = seed(store)
    valuation = load_valuation(store, league, now=NOW, weights=EQUAL_WEIGHTS)
    assert valuation.scoring_period == WEEK
    assert valuation.rostered == {QB_A, WR_B, RB_C, WR_D, RB_E}
    assert valuation.wire == {RB_F}  # E is rostered elsewhere; G has no ESPN line, so the sync never pooled him
    assert valuation.roster == {QB_A, WR_B, RB_C}  # D has no players row
    assert any(f"[{WR_D}]" in warning for warning in valuation.warnings)
    quarterback = valuation.outlooks[QB_A]
    assert quarterback.expected(WEEK) == pytest.approx((10.0 + 12.0) / 2)  # ESPN and Sleeper blended
    assert quarterback.per_game == pytest.approx(10.0)
    assert quarterback.basis == BASIS_SEASON
    receiver = valuation.outlooks[WR_B]
    designated = assess(
        store.players.get("nfl", WR_B) or player(WR_B, "WR", 4), season=SEASON, scoring_period=WEEK, as_of=NOW
    )
    assert 0 < designated.p_active < 1
    assert receiver.p_active == designated.p_active
    assert receiver.expected(WEEK) == pytest.approx(designated.p_active * 11.0)
    assert not valuation.outlooks[RB_C].has_projection
    assert valuation.outlooks[RB_F].expected(WEEK) == pytest.approx(8.0)  # the stored blended row is ignored
    assert valuation.valuer().replacement[RB].espn_id == RB_F


def test_an_explicit_wire_replaces_the_stored_pool(store: Store) -> None:
    league = seed(store)
    valuation = load_valuation(store, league, now=NOW, weights=EQUAL_WEIGHTS, wire=[RB_G, RB_E])
    assert valuation.wire == {RB_G}  # E is rostered
    assert valuation.outlooks[RB_G].basis == BASIS_THIS_WEEK


def test_every_rostered_player_is_valued_on_request(store: Store) -> None:
    """League-wide rankings, the season simulator and trade evaluation (ROADMAP #35, #32, #38) need the other teams'
    players too; our roster, the wire and the replacement levels do not change."""
    league = seed(store)
    ours = load_valuation(store, league, now=NOW, weights=EQUAL_WEIGHTS)
    assert RB_E not in ours.outlooks  # team 2's back: not ours, not on the wire
    whole = load_valuation(store, league, now=NOW, weights=EQUAL_WEIGHTS, include_rostered=True)
    assert set(whole.outlooks) == {QB_A, WR_B, RB_C, RB_E, RB_F}  # D still has no players row
    assert whole.outlooks[RB_E].basis == BASIS_THIS_WEEK
    assert (whole.roster, whole.wire, whole.rostered) == (ours.roster, ours.wire, ours.rostered)
    assert whole.valuer().replacement[RB].espn_id == RB_F  # a rostered player never sets the level


def test_a_schedule_must_cover_the_current_week(store: Store) -> None:
    league = seed(store)
    with pytest.raises(ValuationError, match="covers"):
        load_valuation(store, league, now=NOW, weights=EQUAL_WEIGHTS, schedule=pro_schedule({2: 9}, range(6, 9)))


def test_only_a_synced_points_league_with_our_roster_is_valued(store: Store) -> None:
    league = seed(store)
    with pytest.raises(ValuationError, match="category league"):
        load_valuation(
            store, league, now=NOW, settings=load_league_settings(FIXTURES / "espn" / "fba_settings_9cat.json")
        )
    with pytest.raises(ValueError, match="aware"):
        load_valuation(store, league, now=NOW.replace(tzinfo=None), weights=EQUAL_WEIGHTS)
    stranger = store.leagues.upsert(league.model_copy(update={"key": "other", "espn_league_id": 99, "id": None}))
    with pytest.raises(ValuationError, match="no synced settings"):
        league_settings(store, stranger)
    store.settings.upsert(
        LeagueSettingsRow(league_id=stranger.row_id, settings=ppr().model_dump(mode="json"), as_of=NOW)
    )
    with pytest.raises(ValuationError, match="no roster snapshot"):
        load_valuation(store, stranger, now=NOW, weights=EQUAL_WEIGHTS)
    wrong_team = league.model_copy(update={"team_id": 7})
    with pytest.raises(ValuationError, match="team 7 has no roster"):
        load_valuation(store, wrong_team, now=NOW, weights=EQUAL_WEIGHTS)
