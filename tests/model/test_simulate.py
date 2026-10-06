"""Season simulator (ROADMAP #32): the playoff format from the league's settings, standings from the ``mMatchup``
read, Monte Carlo odds that reproduce from a seed and sum to the playoff spots, the byes and one title, the
tiebreakers and the bracket, and NBA category matchups.

Settings are the hand-built PPR league (``ffl_settings_ppr.json``: 10 teams, 14 regular-season weeks, 6 playoff teams
over weeks 15-17, seeded by total points), the real NFL league (4 playoff teams, two-week rounds) and NBA league
(6 playoff teams), and the 9-cat stand-in with a playoff bracket added. Schedules are hand-built round robins so every
expected number is visible; with ``sd = 0`` every matchup is decided by the means and the odds are 0 or 1.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import pytest

from fm.decide.lineup import MatchupOutlook, win_probability
from fm.espn.ids import FFL
from fm.espn.models import Matchup, MatchupsView
from fm.espn.settings import LeagueSettings, load_league_settings, parse_league_settings
from fm.model.categories import league_categories
from fm.model.simulate import (
    DEFAULT_CV,
    SEEDING_H2H,
    SEEDING_POINTS_FOR,
    CategoryOutlook,
    MatchupOdds,
    PlayoffFormat,
    SeasonOdds,
    SimulationError,
    Standing,
    TeamOdds,
    TeamOutlook,
    bracket_order,
    category_outlook,
    category_totals,
    points_outlook,
    simulate_matchup,
    simulate_season,
    standings_from,
)
from fm.model.valuation import PlayerOutlook

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
REAL = FIXTURES / "espn" / "real"
PPR = FIXTURES / "espn" / "ffl_settings_ppr.json"
NINE_CAT = FIXTURES / "espn" / "fba_settings_9cat.json"
NINE = ("PTS", "BLK", "STL", "AST", "REB", "TO", "3PM", "FG%", "FT%")
TEAMS = tuple(range(1, 11))
RUNS = 4000
TOLERANCE = 1e-9
"""Every run has exactly the playoff spots, the byes and one champion, so the sums are exact up to float error."""


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ppr(**schedule: Any) -> LeagueSettings:
    """The PPR league, its ``scheduleSettings`` overridden by ``schedule``."""
    view = load(PPR)
    view["settings"]["scheduleSettings"].update(schedule)
    return parse_league_settings(view)


def nine_cat(scoring_type: str = "H2H_MOST_CATEGORIES", *, current: int = 16, **schedule: Any) -> LeagueSettings:
    """The 9-cat league cut to a 4-team bracket after 16 matchup weeks (the fixture's own season is 19 weeks and
    three rounds), so a hand-built schedule stays short, in matchup period ``current`` (the fixture says 1)."""
    view = load(NINE_CAT)
    view["settings"]["scoringSettings"]["scoringType"] = scoring_type
    view["settings"]["scheduleSettings"].update({"matchupPeriodCount": 16, "playoffTeamCount": 4, **schedule})
    view["status"]["currentMatchupPeriod"] = current
    return parse_league_settings(view)


def matchup(
    number: int,
    period: int,
    home: int,
    away: int | None,
    *,
    home_points: float = 0.0,
    away_points: float = 0.0,
    decided: bool = False,
    tier: str | None = None,
    cumulative: tuple[tuple[int, int, int], tuple[int, int, int]] | None = None,
) -> Matchup:
    """An ``mMatchup`` schedule entry; ``decided`` sets ESPN's ``winner`` from the points (a tie is ``TIE``)."""
    winner = "UNDECIDED"
    if decided:
        winner = "HOME" if home_points > away_points else "AWAY" if away_points > home_points else "TIE"
    raw: dict[str, Any] = {"id": number, "matchupPeriodId": period, "winner": winner}
    if tier is not None:
        raw["playoffTierType"] = tier
    raw["home"] = {"teamId": home, "totalPoints": home_points}
    if away is not None:
        raw["away"] = {"teamId": away, "totalPoints": away_points}
    if cumulative is not None:
        for side, (wins, losses, ties) in zip(("home", "away"), cumulative, strict=True):
            raw[side]["cumulativeScore"] = {"wins": wins, "losses": losses, "ties": ties}
    return Matchup.model_validate(raw)


def round_robin(teams: Iterable[int], periods: Iterable[int]) -> list[tuple[int, int, int]]:
    """``(period, home, away)`` pairings by the circle method, one game a team per period (a bye with an odd count),
    repeating the cycle over ``periods``."""
    order = list(teams)
    if len(order) % 2:
        order.append(0)
    half = len(order) // 2
    rounds: list[list[tuple[int, int]]] = []
    for _ in range(len(order) - 1):
        rounds.append([(order[i], order[-1 - i]) for i in range(half) if order[i] and order[-1 - i]])
        order = [order[0], order[-1], *order[1:-1]]
    return [(period, home, away) for n, period in enumerate(periods) for home, away in rounds[n % len(rounds)]]


def schedule(
    settings: LeagueSettings,
    *,
    current: int,
    teams: Iterable[int] = TEAMS,
    strength: Mapping[int, float] | None = None,
) -> list[Matchup]:
    """A regular season for ``teams`` over the settings' regular-season periods: the periods before ``current`` are
    decided (the stronger team wins, scoring its ``strength``; by id without one), the rest undecided."""
    scores = strength or {team: float(100 + team) for team in teams}
    entries: list[Matchup] = []
    pairings = round_robin(teams, range(1, settings.schedule.regular_season_matchups + 1))
    for n, (period, home, away) in enumerate(pairings):
        decided = period < current
        entries.append(
            matchup(
                n + 1,
                period,
                home,
                away,
                home_points=scores[home] if decided else 0.0,
                away_points=scores[away] if decided else 0.0,
                decided=decided,
            )
        )
    return entries


def outlooks(strength: Mapping[int, float | Mapping[int, float]], sd: float = 0.0) -> dict[int, TeamOutlook]:
    return {team: TeamOutlook(team, mean, sd) for team, mean in strength.items()}


def table(odds: SeasonOdds) -> dict[int, tuple[Any, ...]]:
    return {
        team_id: (
            team.win_week, team.playoffs, team.bye, team.title, team.seeds, team.expected_wins, dict(team.categories)
        )
        for team_id, team in odds.teams.items()
    }  # fmt: skip


def check_sums(odds: SeasonOdds) -> None:
    """The probabilities add up to the playoff spots, the byes, one title and one seed a team."""
    teams = odds.teams.values()
    assert math.fsum(team.playoffs for team in teams) == pytest.approx(odds.format.teams, abs=TOLERANCE)
    assert math.fsum(team.bye for team in teams) == pytest.approx(odds.format.byes, abs=TOLERANCE)
    assert math.fsum(team.title for team in teams) == pytest.approx(1.0, abs=TOLERANCE)
    for team in teams:
        assert math.fsum(team.seeds) == pytest.approx(1.0, abs=TOLERANCE)
        assert 0.0 <= team.bye <= team.playoffs <= 1.0
        assert 0.0 <= team.title <= team.playoffs
    for seed in range(len(odds.teams)):
        assert math.fsum(team.seeds[seed] for team in teams) == pytest.approx(1.0, abs=TOLERANCE)


# --- the format ------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("settings", "teams", "rounds", "byes", "periods"),
    [
        (ppr(), 6, 3, 2, (15, 16, 17)),
        (load_league_settings(REAL / "ffl" / "mSettings.json"), 4, 2, 0, (14, 15)),
        (load_league_settings(REAL / "fba" / "mSettings.json"), 6, 3, 2, (19, 20, 21)),
        (nine_cat(), 4, 2, 0, (17, 18)),
    ],
)
def test_playoff_format_comes_from_the_settings(
    settings: LeagueSettings, teams: int, rounds: int, byes: int, periods: tuple[int, ...]
) -> None:
    fmt = PlayoffFormat.from_settings(settings)
    assert (fmt.teams, fmt.rounds, fmt.byes, fmt.matchup_periods) == (teams, rounds, byes, periods)
    assert fmt.bracket_size == 2**rounds
    assert fmt.regular_season_matchups == settings.schedule.regular_season_matchups
    assert fmt.seeding_rule == settings.schedule.playoff_seeding_rule
    assert not fmt.reseed and PlayoffFormat.from_settings(settings, reseed=True).reseed


def test_playoff_format_refuses_a_bracket_the_settings_cannot_hold() -> None:
    with pytest.raises(SimulationError, match="no playoff teams"):
        PlayoffFormat.from_settings(ppr(playoffTeamCount=0))
    with pytest.raises(SimulationError, match="6 playoff teams need 3 rounds but the settings list 1 playoff matchup"):
        PlayoffFormat.from_settings(ppr(matchupPeriodCount=16))
    with pytest.raises(SimulationError, match="8 playoff teams need 3 rounds"):
        PlayoffFormat.from_settings(ppr(matchupPeriodCount=15, playoffTeamCount=8))
    assert PlayoffFormat.from_settings(ppr(matchupPeriodCount=13)).matchup_periods == (14, 15, 16)  # one spare


def test_bracket_order_pairs_the_best_seed_with_the_worst() -> None:
    assert bracket_order(1) == (1,)
    assert bracket_order(2) == (1, 2)
    assert bracket_order(4) == (1, 4, 2, 3)
    assert bracket_order(8) == (1, 8, 4, 5, 2, 7, 3, 6)
    assert bracket_order(16)[:4] == (1, 16, 8, 9)
    with pytest.raises(ValueError, match="power of two"):
        bracket_order(6)


# --- standings -------------------------------------------------------------------------------------------------------


def test_standings_come_from_the_decided_regular_season_matchups() -> None:
    view = MatchupsView.model_validate(load(FIXTURES / "espn" / "ffl_matchups.json"))
    standings = standings_from(view, regular_season_matchups=14)
    assert set(standings) == {team for entry in view.schedule for team in entry.team_ids}
    assert standings[2] == Standing(2, wins=1, losses=0, ties=0, points_for=140.3, points_against=118.9)
    assert standings[1] == Standing(1, wins=0, losses=1, ties=0, points_for=118.9, points_against=140.3)
    decided = [entry for entry in view.schedule if entry.is_decided and entry.matchup_period_id <= 14]
    assert len(decided) == 2  # the decided period-15 playoff matchup does not count
    assert sum(team.wins + team.losses + team.ties for team in standings.values()) == 4
    assert standings_from(view, regular_season_matchups=14, team_ids=(99,))[99] == Standing(99)


def test_standings_count_a_tie_and_each_category_when_asked() -> None:
    tie = matchup(1, 1, 1, 2, home_points=100.0, away_points=100.0, decided=True)
    tied = standings_from([tie], regular_season_matchups=14)
    assert tied[1] == Standing(1, ties=1, points_for=100.0, points_against=100.0)
    categories = matchup(2, 2, 1, 2, home_points=6, away_points=3, decided=True, cumulative=((6, 2, 1), (2, 6, 1)))
    each = standings_from([categories], regular_season_matchups=14, each_category=True)
    assert each[1] == Standing(1, wins=6, losses=2, ties=1, points_for=6, points_against=2)
    assert each[2] == Standing(2, wins=2, losses=6, ties=1, points_for=2, points_against=6)
    one_game = standings_from([categories], regular_season_matchups=14)
    assert (one_game[1].wins, one_game[2].losses) == (1, 1)


# --- a points season -------------------------------------------------------------------------------------------------


STRENGTH = {team: 100.0 + 5.0 * team for team in TEAMS}
"""Team 10 is the best by 5 points a step; with ``sd = 0`` the better team always wins."""


def test_a_deterministic_season_seeds_by_strength_and_crowns_the_strongest() -> None:
    settings = ppr()
    odds = simulate_season(settings, schedule(settings, current=4), outlooks(STRENGTH), seed=1, runs=50)
    check_sums(odds)
    assert odds.matchup_period == 4 and odds.runs == 50 and odds.seed == 1 and odds.warnings == ()
    by_strength = sorted(TEAMS, key=lambda team: -STRENGTH[team])
    for seed, team in enumerate(by_strength):
        assert odds.team(team).seeds[seed] == 1.0
    assert {team: odds.team(team).playoffs for team in TEAMS} == {team: float(team >= 5) for team in TEAMS}
    assert {team: odds.team(team).bye for team in TEAMS} == {team: float(team >= 9) for team in TEAMS}
    assert odds.team(10).title == 1.0 and odds.team(10).win_week == 1.0
    assert odds.team(1).title == 0.0 and odds.team(1).win_week == 0.0
    assert odds.team(10).expected_wins == 14.0  # a game every week, all won
    assert all(team.categories == {} for team in odds.teams.values())
    with pytest.raises(KeyError, match="team 11 was not simulated"):
        odds.team(11)


def test_a_team_on_a_bye_this_week_has_no_week_to_win() -> None:
    settings = ppr()
    nine = TEAMS[:9]
    strength = {team: STRENGTH[team] for team in nine}
    entries = schedule(settings, current=4, teams=nine, strength=strength)
    on_bye = [team for team in nine if not any(e.matchup_period_id == 4 and team in e.team_ids for e in entries)]
    assert len(on_bye) == 1
    odds = simulate_season(settings, entries, outlooks(strength), seed=3, runs=20)
    assert odds.team(on_bye[0]).win_week is None
    assert all(odds.team(team).win_week is not None for team in nine if team != on_bye[0])
    check_sums(odds)


def test_seeded_runs_reproduce_and_the_sums_hold_under_noise() -> None:
    settings = ppr()
    entries = schedule(settings, current=4)
    noisy = outlooks(STRENGTH, sd=25.0)
    first = simulate_season(settings, entries, noisy, seed=42, runs=RUNS)
    again = simulate_season(settings, entries, noisy, seed=42, runs=RUNS)
    other = simulate_season(settings, entries, noisy, seed=43, runs=RUNS)
    assert table(first) == table(again)
    assert table(first) != table(other)
    check_sums(first)
    check_sums(other)
    ranked = sorted(TEAMS, key=lambda team: first.team(team).title)
    assert ranked[-1] == 10 and ranked[0] in (1, 2)
    assert first.team(10).playoffs > first.team(5).playoffs > first.team(1).playoffs
    assert 0.0 < first.team(5).playoffs < 1.0


def test_this_week_agrees_with_the_lineup_normal_approximation() -> None:
    settings = ppr()
    entries = schedule(settings, current=4)
    noisy = outlooks(STRENGTH, sd=25.0)
    odds = simulate_season(settings, entries, noisy, seed=7, runs=20_000)
    this_week = [entry for entry in entries if entry.matchup_period_id == 4]
    assert len(this_week) == 5
    for entry in this_week:
        home, away = entry.team_ids
        expected = win_probability(STRENGTH[home], 25.0, MatchupOutlook(STRENGTH[away], 25.0))
        assert odds.team(home).win_week == pytest.approx(expected, abs=0.015)
        assert odds.team(away).win_week == pytest.approx(1.0 - expected, abs=0.015)


def test_simulate_matchup_scores_one_pairing() -> None:
    settings = ppr()
    odds = simulate_matchup(settings, TeamOutlook(1, 120.0, 20.0), TeamOutlook(2, 100.0, 20.0), 4, seed=5, runs=20_000)
    assert odds == MatchupOdds(1, 2, odds.home, odds.away, odds.tie, odds.categories)
    assert odds.home == pytest.approx(win_probability(120.0, 20.0, MatchupOutlook(100.0, 20.0)), abs=0.015)
    assert odds.home + odds.away + odds.tie == pytest.approx(1.0)
    assert odds.categories == {}
    dead_heat = simulate_matchup(settings, TeamOutlook(1, 100.0), TeamOutlook(2, 100.0), 4, seed=5, runs=10)
    assert (dead_heat.home, dead_heat.away, dead_heat.tie) == (0.0, 0.0, 1.0)


def test_given_standings_replace_the_schedule_record() -> None:
    settings = ppr()
    entries = schedule(settings, current=14)  # one week left
    standings = {team: Standing(team, wins=13.0) if team == 1 else Standing(team, losses=13.0) for team in TEAMS}
    odds = simulate_season(settings, entries, outlooks(STRENGTH), seed=1, runs=10, standings=standings)
    assert odds.team(1).seeds[0] == 1.0 and odds.team(1).bye == 1.0
    assert odds.team(1).expected_wins == 13.0  # the weakest team loses its last matchup
    assert odds.team(1).title == 0.0 and odds.team(10).title == 1.0
    check_sums(odds)


@pytest.mark.parametrize(("rule", "better"), [(SEEDING_POINTS_FOR, 2), (SEEDING_H2H, 1), ("TOTAL_POINTS_AGAINST", 1)])
def test_the_seeding_rule_breaks_a_tie(rule: str, better: int) -> None:
    """Teams 1 and 2 finish 2-1; 1 beat 2 head to head and allowed more points (305 to 280), 2 scored more (355
    to 300). A two-team bracket, so the top seed is the whole story."""
    settings = ppr(playoffSeedingRule=rule, playoffTeamCount=2)
    entries = [
        matchup(1, 1, 1, 2, home_points=100.0, away_points=95.0, decided=True),
        matchup(2, 2, 1, 4, home_points=100.0, away_points=90.0, decided=True),
        matchup(3, 2, 2, 3, home_points=130.0, away_points=90.0, decided=True),
        matchup(4, 3, 1, 3, home_points=100.0, away_points=120.0, decided=True),
        matchup(5, 3, 2, 4, home_points=130.0, away_points=90.0, decided=True),
    ]
    standings = standings_from(entries, regular_season_matchups=14)
    assert (standings[1].wins, standings[2].wins, standings[1].points_for, standings[2].points_for) == (2, 2, 300, 355)
    assert (standings[1].points_against, standings[2].points_against) == (305, 280)
    strength = {1: 100.0, 2: 100.0, 3: 50.0, 4: 60.0}
    odds = simulate_season(settings, entries, outlooks(strength), seed=1, runs=10, current_matchup_period=4)
    assert odds.team(better).seeds[0] == 1.0
    assert odds.team(3 - better).seeds[1] == 1.0
    assert odds.team(better).title + odds.team(3 - better).title == 1.0  # a dead heat in the final goes to the seed
    assert odds.team(better).title == 1.0


def test_division_winners_seed_first() -> None:
    settings = ppr()
    entries = schedule(settings, current=4)
    divisions = {team: 0 if team <= 5 else 1 for team in TEAMS}
    odds = simulate_season(settings, entries, outlooks(STRENGTH), seed=1, runs=10, divisions=divisions)
    assert odds.team(10).seeds[0] == 1.0  # the best team wins its division
    assert odds.team(5).seeds[1] == 1.0  # the weak division's winner takes the second seed and a bye
    assert odds.team(5).bye == 1.0 and odds.team(9).bye == 0.0 and odds.team(9).seeds[2] == 1.0
    check_sums(odds)
    with pytest.raises(SimulationError, match="team\\(s\\) 10 have none"):
        simulate_season(settings, entries, outlooks(STRENGTH), seed=1, runs=10, divisions={t: 0 for t in TEAMS[:9]})


def test_reseeding_changes_who_meets_whom() -> None:
    """A 6-team bracket after the regular season: seeds 1-6 are teams 1-6. Round one is 3 v 6 and 4 v 5; without
    reseeding the winners meet seeds 2 and 1 in bracket order, with it the best remaining seed meets the worst."""
    settings = ppr()
    entries = schedule(settings, current=15)
    standings = {team: Standing(team, wins=float(14 - team), losses=float(team)) for team in TEAMS}
    weeks = {
        15: {1: 100.0, 2: 100.0, 3: 50.0, 4: 100.0, 5: 50.0, 6: 150.0},  # 6 beats 3, 4 beats 5
        16: {1: 120.0, 2: 110.0, 4: 100.0, 6: 130.0},  # fixed: 1 v 4 -> 1, 2 v 6 -> 6; reseeded: 1 v 6 -> 6, 2 v 4 -> 2
        17: {1: 100.0, 2: 150.0, 6: 120.0},  # fixed final 1 v 6 -> 6; reseeded final 6 v 2 -> 2
    }
    strength: dict[int, float | Mapping[int, float]] = {
        team: {period: weeks.get(period, {}).get(team, 0.0) for period in range(1, 18)} for team in TEAMS
    }
    fixed = simulate_season(settings, entries, outlooks(strength), seed=1, runs=10, standings=standings)
    reseeded = simulate_season(settings, entries, outlooks(strength), seed=1, runs=10, standings=standings, reseed=True)
    assert fixed.team(6).title == 1.0 and reseeded.team(2).title == 1.0
    assert all(odds.team(1).bye == odds.team(2).bye == 1.0 for odds in (fixed, reseeded))
    assert all(odds.team(7).playoffs == 0.0 for odds in (fixed, reseeded))
    check_sums(fixed)
    check_sums(reseeded)


def test_decided_playoff_rounds_keep_their_winners() -> None:
    settings = ppr()
    entries = schedule(settings, current=16)
    standings = {team: Standing(team, wins=float(14 - team), losses=float(team)) for team in TEAMS}
    entries += [  # round one upsets: 6 beat 3, 5 beat 4, though 3 and 4 are stronger
        matchup(900, 15, 3, 6, home_points=90.0, away_points=100.0, decided=True, tier="WINNERS_BRACKET"),
        matchup(901, 15, 4, 5, home_points=90.0, away_points=100.0, decided=True, tier="WINNERS_BRACKET"),
        matchup(902, 15, 7, 8, home_points=90.0, away_points=100.0, decided=True, tier="LOSERS_CONSOLATION_LADDER"),
    ]
    strength = {1: 100.0, 2: 90.0, 3: 200.0, 4: 200.0, 5: 80.0, 6: 85.0, 7: 300.0, 8: 300.0, 9: 1.0, 10: 1.0}
    odds = simulate_season(settings, entries, outlooks(strength), seed=1, runs=10, standings=standings)
    assert odds.team(3).title == odds.team(4).title == odds.team(7).title == 0.0
    assert odds.team(1).title == 1.0  # 1 beats 5, 2 beats 6, 1 beats 2
    assert odds.team(3).playoffs == 1.0 and odds.team(1).bye == 1.0
    check_sums(odds)


def test_the_simulator_refuses_what_it_cannot_simulate() -> None:
    settings = ppr()
    entries = schedule(settings, current=4)
    with pytest.raises(SimulationError, match="no outlook for team\\(s\\) 9, 10"):
        simulate_season(settings, entries, outlooks({team: STRENGTH[team] for team in TEAMS[:8]}), seed=1, runs=5)
    with pytest.raises(SimulationError, match="team 3's outlook mean has no value for matchup period 5"):
        partial = outlooks(STRENGTH) | {3: TeamOutlook(3, {4: 100.0}, 0.0)}
        simulate_season(settings, entries, partial, seed=1, runs=5)
    with pytest.raises(SimulationError, match="sd in matchup period 4 should be >= 0"):
        simulate_season(settings, entries, outlooks(STRENGTH) | {3: TeamOutlook(3, 100.0, -1.0)}, seed=1, runs=5)
    with pytest.raises(SimulationError, match="scores points: team\\(s\\) 3 need a TeamOutlook"):
        wrong = outlooks(STRENGTH) | {3: CategoryOutlook(3, {"PTS": 1.0})}
        simulate_season(settings, entries, wrong, seed=1, runs=5)
    with pytest.raises(SimulationError, match="no head-to-head matchups"):
        simulate_season(nine_cat("ROTO"), entries, outlooks(STRENGTH), seed=1, runs=5)
    with pytest.raises(SimulationError, match="current matchup period is unknown"):
        view = load(PPR)
        view["status"] = {}
        simulate_season(parse_league_settings(view), entries, outlooks(STRENGTH), seed=1, runs=5)
    with pytest.raises(ValueError, match="runs should be a positive int"):
        simulate_season(settings, entries, outlooks(STRENGTH), seed=1, runs=0)
    with pytest.raises(ValueError, match="seed should be an int"):
        simulate_season(settings, entries, outlooks(STRENGTH), seed="x", runs=5)  # type: ignore[arg-type]
    with pytest.raises(SimulationError, match="finite number"):
        simulate_season(settings, entries, outlooks(STRENGTH) | {3: TeamOutlook(3, math.nan)}, seed=1, runs=5)


def test_an_undecided_matchup_before_the_current_period_is_skipped_with_a_warning() -> None:
    settings = ppr()
    entries = schedule(settings, current=4)
    stale = next(entry for entry in entries if entry.matchup_period_id == 2)
    entries[entries.index(stale)] = matchup(stale.id, 2, *stale.team_ids)
    odds = simulate_season(settings, entries, outlooks(STRENGTH), seed=1, runs=5)
    assert odds.warnings == (f"matchup {stale.id} in period 2 is undecided but before the current period 4; skipped",)
    check_sums(odds)


def test_the_current_period_comes_from_the_view_before_the_settings() -> None:
    settings = ppr()  # the settings say matchup period 4
    view = MatchupsView.model_validate({"schedule": [], "status": {"currentMatchupPeriod": 6}})
    entries = schedule(settings, current=6)
    view = view.model_copy(update={"schedule": tuple(entries)})
    odds = simulate_season(settings, view, outlooks(STRENGTH), seed=1, runs=5)
    assert odds.matchup_period == 6
    assert simulate_season(settings, entries, outlooks(STRENGTH), seed=1, runs=5).matchup_period == 4


# --- outlooks from player weeks --------------------------------------------------------------------------------------


QB, RB, WR = (FFL.slot_id(label) for label in ("QB", "RB", "WR"))


def player(espn_id: int, slot: int, weekly: Mapping[int, float]) -> PlayerOutlook:
    return PlayerOutlook(espn_id, f"Player {espn_id}", None, 1, frozenset({slot}), dict(weekly), basis="season")


def test_points_outlook_sums_the_best_lineup_over_the_matchup_periods() -> None:
    settings = load_league_settings(REAL / "ffl" / "mSettings.json")  # matchup 14 spans weeks 14 and 15
    players = [
        player(1, QB, {13: 20.0, 14: 22.0, 15: 18.0}),
        player(2, QB, {13: 25.0, 14: 10.0, 15: 30.0}),  # the better quarterback in weeks 13 and 15
        player(3, RB, {13: 12.0, 14: 12.0, 15: 12.0}),
    ]
    outlook = points_outlook(7, players, settings, (13, 14), slots=(QB, RB))
    assert outlook.team_id == 7
    assert outlook.score(13) == (37.0, pytest.approx(DEFAULT_CV * 37.0))
    assert outlook.score(14) == (pytest.approx(34.0 + 42.0), pytest.approx(math.hypot(0.2 * 34.0, 0.2 * 42.0)))
    overridden = points_outlook(7, players, settings, (13, 14), slots=(QB, RB), sd={13: 9.0, 99: 1.0}, cv=0.1)
    assert overridden.score(13) == (37.0, 9.0)
    assert overridden.score(14)[1] == pytest.approx(math.hypot(3.4, 4.2))
    with pytest.raises(SimulationError, match="has no value for matchup period 15"):
        outlook.score(15)
    with pytest.raises(ValueError, match="cv should be"):
        points_outlook(7, players, settings, (13,), cv=-1.0)


def test_points_outlook_needs_matchups_listed_in_scoring_periods() -> None:
    nba = load_league_settings(REAL / "fba" / "mSettings.json")  # matchup periods are weeks, not days
    with pytest.raises(SimulationError, match="not listed in scoring periods"):
        points_outlook(1, [player(1, QB, {1: 10.0})], nba, (1,))
    with pytest.raises(SimulationError, match="matchup period 99 lists no scoring periods"):
        points_outlook(1, [player(1, QB, {1: 10.0})], ppr(), (99,))


# --- a category season -----------------------------------------------------------------------------------------------


def line(pts: float, to: float = 10.0, fgm: float = 40.0, fga: float = 90.0) -> dict[str, float]:
    return {
        "PTS": pts, "BLK": 20.0, "STL": 30.0, "AST": 100.0, "OREB": 40.0, "DREB": 120.0, "TO": to,
        "3PM": 45.0, "FGM": fgm, "FGA": fga, "FTM": 60.0, "FTA": 80.0,
    }  # fmt: skip


def category_means(team: int) -> dict[str, float]:
    """A team's category totals in every period: the higher its id, the more points it scores but the more it turns
    the ball over; everything else is equal across teams."""
    return {
        "PTS": 400.0 + 10.0 * team, "BLK": 20.0, "STL": 30.0, "AST": 100.0, "REB": 160.0, "TO": 50.0 + 5.0 * team,
        "3PM": 45.0, "FG%": 0.45, "FT%": 0.75,
    }  # fmt: skip


def category_league(teams: Iterable[int], cv: float = 0.0) -> dict[int, CategoryOutlook]:
    return {
        team: CategoryOutlook(team, means, {stat: cv * value for stat, value in means.items()})
        for team in teams
        for means in (category_means(team),)
    }


def test_a_category_week_is_won_on_the_categories_with_turnovers_reversed() -> None:
    settings = nine_cat()
    six = tuple(range(1, 7))
    entries = schedule(settings, current=16, teams=six)
    odds = simulate_season(settings, entries, category_league(six), seed=1, runs=10)
    this_week = [entry for entry in entries if entry.matchup_period_id == 16]
    assert len(this_week) == 3
    for entry in this_week:
        home, away = entry.team_ids
        strong, weak = max(home, away), min(home, away)
        assert odds.team(strong).categories["PTS"] == 1.0 and odds.team(strong).categories["TO"] == 0.0
        assert odds.team(weak).categories["PTS"] == 0.0 and odds.team(weak).categories["TO"] == 1.0
        assert odds.team(strong).categories["REB"] == 0.5  # a dead heat counts half
        assert odds.team(strong).win_week == 0.5 and odds.team(weak).win_week == 0.5  # one category each, 7 tied
        assert set(odds.team(strong).categories) == set(NINE)
    check_sums(odds)
    assert odds.format.teams == 4 and odds.format.byes == 0


def test_category_odds_reproduce_and_sum_under_noise() -> None:
    settings = nine_cat(current=10)
    six = tuple(range(1, 7))
    entries = schedule(settings, current=10, teams=six)
    noisy = category_league(six, cv=0.02)
    first = simulate_season(settings, entries, noisy, seed=11, runs=RUNS)
    again = simulate_season(settings, entries, noisy, seed=11, runs=RUNS)
    assert table(first) == table(again)
    assert table(first) != table(simulate_season(settings, entries, noisy, seed=12, runs=RUNS))
    check_sums(first)
    team = first.team(6)
    assert 0.5 < team.categories["PTS"] <= 1.0 and 0.0 <= team.categories["TO"] < 0.5
    assert 0.4 < team.categories["FG%"] < 0.6  # the same shooting either side: a coin flip under noise
    assert team.win_week is not None and 0.0 < team.win_week < 1.0
    home, away = next(entry.team_ids for entry in entries if entry.matchup_period_id == 10 and 6 in entry.team_ids)
    pairing = simulate_matchup(settings, noisy[home], noisy[away], 10, seed=11, runs=RUNS)
    assert set(pairing.categories) == set(NINE)
    assert pairing.home + pairing.away + pairing.tie == pytest.approx(1.0)


def test_each_category_scoring_counts_every_category_in_the_record() -> None:
    settings = nine_cat("H2H_CATEGORY")
    six = tuple(range(1, 7))
    entries = schedule(settings, current=16, teams=six)
    empty = {team: Standing(team) for team in six}
    odds = simulate_season(settings, entries, category_league(six), seed=1, runs=10, standings=empty)
    # Every team plays its last week: one category won, one lost, seven tied, on top of an empty record.
    assert all(team.expected_wins == 1.0 for team in odds.teams.values())
    check_sums(odds)
    decided = matchup(500, 15, 1, 2, home_points=5, away_points=4, decided=True, cumulative=((5, 3, 1), (3, 5, 1)))
    last_week = [entry for entry in entries if entry.matchup_period_id == 16]
    with_record = simulate_season(settings, [decided, *last_week], category_league(six), seed=1, runs=10)
    assert with_record.team(1).expected_wins == 6.0 and with_record.team(2).expected_wins == 4.0


def test_category_outlooks_refuse_a_missing_category_or_period() -> None:
    settings = nine_cat(playoffTeamCount=2)
    entries = schedule(settings, current=16, teams=(1, 2))
    short = {1: CategoryOutlook(1, {"PTS": 1.0}), 2: category_league((2,))[2]}
    with pytest.raises(SimulationError, match="team 1's outlook in BLK: no mean; the outlook covers PTS"):
        simulate_season(settings, entries, short, seed=1, runs=5)
    wrong = {1: TeamOutlook(1, 100.0), 2: category_league((2,))[2]}
    with pytest.raises(SimulationError, match="competes on categories: team\\(s\\) 1 need a CategoryOutlook"):
        simulate_season(settings, entries, wrong, seed=1, runs=5)
    per_period = CategoryOutlook(1, {**category_means(1), "PTS": {16: 400.0}})
    with pytest.raises(SimulationError, match="in PTS mean has no value for matchup period 17"):
        simulate_season(settings, entries, {1: per_period, 2: category_league((2,))[2]}, seed=1, runs=5)
    with pytest.raises(SimulationError, match="2 team\\(s\\) to simulate but the settings give 4 playoff spots"):
        simulate_season(nine_cat(), entries, category_league((1, 2)), seed=1, runs=5)


def test_category_totals_sum_counting_stats_and_ratio_the_rates() -> None:
    categories = league_categories(nine_cat())
    values, volume = category_totals([line(30.0, fgm=10.0, fga=20.0), line(20.0, fgm=5.0, fga=30.0)], categories)
    assert values["PTS"] == 50.0 and values["REB"] == 320.0 and values["TO"] == 20.0
    assert values["FG%"] == pytest.approx(15.0 / 50.0) and volume["FG%"] == 50.0
    assert values["FT%"] == pytest.approx(120.0 / 160.0) and volume["PTS"] == 2.0
    assert category_totals([], categories)[0]["FG%"] == 0.0


def test_category_outlook_builds_team_normals_from_the_lines() -> None:
    settings = nine_cat()
    outlook = category_outlook(4, {16: [line(30.0), line(20.0)], 17: [line(10.0)]}, settings, cv=0.1)
    assert outlook.team_id == 4
    assert outlook.score("PTS", 16) == (50.0, pytest.approx(5.0))
    assert outlook.score("PTS", 17) == (10.0, pytest.approx(1.0))
    assert outlook.score("TO", 16) == (20.0, pytest.approx(2.0))
    rate, spread = outlook.score("FG%", 16)
    assert rate == pytest.approx(80.0 / 180.0) and spread == pytest.approx(math.sqrt(rate * (1 - rate) / 180.0))
    assert outlook.score("FG%", 17)[1] == pytest.approx(math.sqrt((40 / 90) * (50 / 90) / 90.0))
    forced = category_outlook(4, {16: [line(30.0)]}, settings, sd={"PTS": {16: 7.0}})
    assert forced.score("PTS", 16) == (30.0, 7.0)
    with pytest.raises(ValueError, match="'XYZ' is not a category here"):
        category_outlook(4, {16: [line(30.0)]}, settings, sd={"XYZ": {16: 1.0}})
    with pytest.raises(ValueError, match="scores points"):
        category_outlook(4, {}, load_league_settings(REAL / "fba" / "mSettings.json"))
    with pytest.raises(SimulationError, match="has no value for matchup period 18"):
        outlook.score("PTS", 18)


def test_a_category_outlook_feeds_the_simulator() -> None:
    """Team 1 scores more, turns it over less and shoots far better from the field (``cv = 0``: the counting totals
    are certain, the percentages binomial); the two shoot the same from the line, a coin flip."""
    settings = nine_cat(playoffTeamCount=2)
    better = [line(40.0, to=5.0, fgm=60.0)] * 5
    strong = category_outlook(1, {16: better, 17: better}, settings, cv=0.0)
    weak = category_outlook(2, {16: [line(20.0)] * 5, 17: [line(20.0)] * 5}, settings, cv=0.0)
    entries = [matchup(1, 16, 1, 2)]
    odds = simulate_season(settings, entries, {1: strong, 2: weak}, seed=1, runs=2000)
    won = odds.team(1).categories
    assert won["PTS"] == won["TO"] == won["FG%"] == 1.0 and won["BLK"] == 0.5
    assert 0.4 < won["FT%"] < 0.6
    assert odds.team(1).win_week == 1.0 and odds.team(1).title == 1.0
    assert isinstance(odds.team(1), TeamOdds)
    check_sums(odds)
