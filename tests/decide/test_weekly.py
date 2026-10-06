"""The NBA weekly category planner (ROADMAP #37, DESIGN sections 9.3 and 9.5).

Two layers:

- :func:`plan_categories` on hand-built outlooks against a model fit on the 9-cat stand-in league
  (``tests/fixtures/espn/fba_settings_9cat.json``): the punt threshold, the cap on punts, the weights, the stat gaps,
  the simulator's probabilities and the matchup's win probability.
- :func:`plan_weekly` on a hand-built store: both rosters share their players' shapes except two categories we win
  and lose by a wide margin, so the plan has a safe category, a conceded one and toss-ups. A points league is refused.

The pro schedule is a test double with the surface of ``fm.sports.base.ScheduleLike``: one 7:30 p.m. ET game per team
on the days it plays (Tue Oct 20, 2026 is day 1).
"""

from __future__ import annotations

import inspect
import math
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from statistics import NormalDist

import pytest

from fm.decide import registry as decide_registry
from fm.decide.lineup_daily import DEFAULT_GAME_SD, swing_outlook
from fm.decide.weekly import (
    PUNT_BELOW,
    SAFE_ABOVE,
    CategoryStatus,
    WeeklyPlan,
    WeeklyPlanError,
    check_category_league,
    default_max_punts,
    game_sd_from_tau,
    matchup_win_probability,
    plan_categories,
    plan_weekly,
)
from fm.espn.ids import FBA
from fm.espn.settings import LeagueSettings, ScoringType, load_league_settings
from fm.model.categories import CategoryModel, fit_categories, within_player_sd
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
NORMAL = NormalDist()


def nine() -> LeagueSettings:
    return load_league_settings(FIXTURES / "espn" / "fba_settings_9cat.json")


def real_points() -> LeagueSettings:
    return load_league_settings(FIXTURES / "espn" / "real" / "fba" / "mSettings.json")


# --- the model --------------------------------------------------------------------------------------------------------


def stat_line(ppg: float, espn_id: int, *, ours: bool = False) -> dict[str, float]:
    """A per-game line for all nine categories. The shape depends on ``ppg`` and the id's last digit only, so two sides
    with the same ids modulo 100 have the same lines, except that ours scores 1.5 times the points and a tenth of the
    blocks of theirs (theirs block three times a normal rate)."""
    attempts = ppg * 0.8
    made = attempts * (0.40 + 0.02 * (espn_id % 5))
    free_throws = ppg / 5
    return {
        "PTS": ppg * (1.5 if ours else 1.0),
        "REB": ppg / 4,
        "AST": ppg / 5,
        "STL": ppg / 20,
        "BLK": ppg / 30 * (0.1 if ours else 3.0),
        "3PM": ppg / 10,
        "TO": ppg / 12,
        "FGM": made,
        "FGA": attempts,
        "FTM": free_throws * (0.70 + 0.05 * (espn_id % 4)),
        "FTA": free_throws,
        "GP": 1.0,
    }


@pytest.fixture(scope="module")
def model() -> CategoryModel:
    lines = {i: stat_line(6.0 + i, i) for i in range(1, 31)}
    return fit_categories(lines, nine())


def outlook_for(margins: Mapping[str, float], *, games: float = 8.0, sd: float = 1.0):  # noqa: ANN201
    """A swing outlook whose margins are ``margins`` (score units) and whose standard deviation is ``sd * sqrt(2 *
    games)``: ours scores the margin and theirs nothing."""
    return swing_outlook(dict(margins), dict.fromkeys(margins, 0.0), games_ours=games, games_theirs=games, game_sd=sd)


GAMES = 8.0
SD = math.sqrt(2 * GAMES)  # the default game_sd of 1 over 16 started games


def margin_at(probability: float) -> float:
    """The margin (score units) that gives ``probability`` over the outlook's standard deviation."""
    return NORMAL.inv_cdf(probability) * SD


def test_a_punt_is_recommended_below_the_threshold_and_not_above_it(model: CategoryModel) -> None:
    margins = dict.fromkeys(model.categories, 0.0)
    below, above = model.categories[0], model.categories[1]
    margins[below] = margin_at(PUNT_BELOW - 0.02)
    margins[above] = margin_at(PUNT_BELOW + 0.02)
    plan = plan_categories(model, outlook_for(margins), games=GAMES)
    assert plan.category(below).status is CategoryStatus.PUNT
    assert plan.category(above).status is CategoryStatus.CONTEST
    assert plan.punts == (below,)
    assert plan.weights[below] == 0.0 and plan.weights[above] > 0.0
    # the threshold is a parameter: raise it and the second category is conceded too
    wider = plan_categories(model, outlook_for(margins), games=GAMES, punt_below=PUNT_BELOW + 0.05)
    assert set(wider.punts) == {below, above}
    # a category exactly at the threshold is not below it
    exact = plan_categories(model, outlook_for({**margins, below: margin_at(PUNT_BELOW + 1e-9)}), games=GAMES)
    assert below not in exact.punts


def test_the_weights_favor_contested_categories(model: CategoryModel) -> None:
    names = model.categories
    toss_up, safe, lost, slight = names[0], names[1], names[2], names[3]
    margins = dict.fromkeys(names, 0.0)
    margins[safe] = margin_at(0.97)
    margins[lost] = margin_at(0.03)
    margins[slight] = margin_at(0.6)
    plan = plan_categories(model, outlook_for(margins), games=GAMES)
    assert plan.category(safe).status is CategoryStatus.SAFE
    assert plan.category(lost).status is CategoryStatus.PUNT
    assert plan.category(toss_up).status is CategoryStatus.CONTEST
    assert plan.weights[toss_up] > plan.weights[slight] > plan.weights[safe] > plan.weights[lost] == 0.0
    assert plan.relative_weights[toss_up] > 1.0 > plan.relative_weights[safe]
    live = [name for name in names if plan.category(name).status is not CategoryStatus.PUNT]
    assert sum(plan.relative_weights[name] for name in live) / len(live) == pytest.approx(1.0)
    assert plan.relative_weights[lost] == 0.0
    # streamers aim at the contested categories, the largest swing first; never a punt or a safe category
    assert plan.streamer_targets[0] == toss_up
    assert lost not in plan.streamer_targets and safe not in plan.streamer_targets
    # the weights are swings in the lineup planner's unit: dP/dscore = phi(margin / sd) / sd
    expected = NORMAL.pdf(0.0) / SD
    assert plan.weights[toss_up] == pytest.approx(expected)
    assert plan.category(toss_up).swing == pytest.approx(expected)


def test_everything_decided_leaves_flat_relative_weights(model: CategoryModel) -> None:
    margins = dict.fromkeys(model.categories, 10_000.0)
    plan = plan_categories(model, outlook_for(margins), games=GAMES)
    assert set(plan.safe) == set(model.categories) and plan.streamer_targets == ()
    assert set(plan.relative_weights.values()) == {1.0}
    assert all(0 <= weight < 1e-9 for weight in plan.weights.values())


def test_a_majority_must_stay_in_play_so_the_lowest_probabilities_are_conceded_first(model: CategoryModel) -> None:
    names = model.categories
    assert len(names) == 9 and default_max_punts(nine()) == 4
    margins = {name: margin_at(0.001 * (i + 1)) for i, name in enumerate(names)}  # every category is lost
    plan = plan_categories(model, outlook_for(margins), games=GAMES)
    assert plan.max_punts == 4 and plan.punts == names[:4]
    assert [entry.status for entry in plan.categories[4:]] == [CategoryStatus.CONTEST] * 5
    assert any("at most 4 may be conceded" in warning for warning in plan.warnings)
    assert plan_categories(model, outlook_for(margins), games=GAMES, max_punts=0).punts == ()
    assert len(plan_categories(model, outlook_for(margins), games=GAMES, max_punts=9).punts) == 9


def test_a_league_that_scores_every_category_keeps_one_contested() -> None:
    each = nine().model_copy(update={"scoring_type": ScoringType.H2H_CATEGORY})
    assert default_max_punts(each) == 8
    assert default_max_punts(nine(), 8) == 3  # eight categories: five are needed
    assert default_max_punts(nine(), 1) == 0


def test_the_stat_gap_is_in_the_stats_own_units(model: CategoryModel) -> None:
    margins = dict.fromkeys(model.categories, 0.0)
    margins.update({"AST": -3.0, "TO": -2.0, "FG%": 4.0, "PTS": 4.0})
    plan = plan_categories(model, outlook_for(margins), games=GAMES)
    ast = plan.category("AST")
    spread = model.stat("AST").spread(model.metric, model.kappa)
    assert ast.gap == pytest.approx(3.0 * spread) and ast.gap > 0 and not ast.reverse  # that many more assists
    assert ast.gap_per_game == pytest.approx(ast.gap / GAMES)
    assert plan.category("TO").reverse and plan.category("TO").gap == pytest.approx(
        2.0 * model.stat("TO").spread(model.metric, model.kappa)
    )  # that many turnovers fewer
    lead = plan.category("PTS")
    assert lead.gap == pytest.approx(-4.0 * model.stat("PTS").spread(model.metric, model.kappa)) and lead.gap < 0
    rate = plan.category("FG%")  # a percentage: the team's rate over its games, not a count
    assert rate.gap == pytest.approx(-4.0 * model.stat("FG%").spread(model.metric, model.kappa) / GAMES)
    assert abs(rate.gap) < 0.1
    assert plan.category("REB").gap == 0.0


def test_the_simulators_probabilities_replace_the_normal_approximation(model: CategoryModel) -> None:
    names = model.categories
    margins = dict.fromkeys(names, 0.0)
    plan = plan_categories(model, outlook_for(margins), games=GAMES, simulated={names[0]: 0.3, names[1]: 0.05})
    first, second = plan.category(names[0]), plan.category(names[1])
    assert first.source == "simulation" and first.win_probability == 0.3 and first.status is CategoryStatus.CONTEST
    assert first.margin == pytest.approx(NORMAL.inv_cdf(0.3) * SD) and first.gap > 0  # behind, so a gap to close
    assert first.swing == pytest.approx(NORMAL.pdf(NORMAL.inv_cdf(0.3)) / SD)
    assert second.status is CategoryStatus.PUNT and second.weight == 0.0
    assert plan.category(names[2]).source == "normal" and plan.category(names[2]).win_probability == 0.5
    with pytest.raises(ValueError, match="should be in"):
        plan_categories(model, outlook_for(margins), games=GAMES, simulated={names[0]: 1.2})
    with pytest.raises(ValueError, match="no such categories"):
        plan_categories(model, outlook_for(margins), games=GAMES, simulated={"XYZ": 0.5})


def test_the_matchups_win_probability_is_the_chance_of_a_majority(model: CategoryModel) -> None:
    assert matchup_win_probability({"a": 0.5, "b": 0.5}) == pytest.approx(0.5)  # 1/4 both + half of 1/2 even
    assert matchup_win_probability({"a": 1.0, "b": 1.0, "c": 0.0}) == 1.0
    assert matchup_win_probability(dict.fromkeys("abcde", 0.5)) == pytest.approx(0.5)
    assert matchup_win_probability({"a": 0.9, "b": 0.9, "c": 0.1}) == pytest.approx(0.9 * 0.9 + 2 * 0.9 * 0.1 * 0.1)
    plan = plan_categories(model, outlook_for(dict.fromkeys(model.categories, 0.0)), games=GAMES)
    assert plan.matchup_win_probability == pytest.approx(0.5) and plan.expected_wins == pytest.approx(4.5)
    each = plan_categories(model, outlook_for(dict.fromkeys(model.categories, 0.0)), games=GAMES, most_categories=False)
    assert each.matchup_win_probability is None


def test_a_category_without_spread_is_left_out_with_a_warning() -> None:
    lines = {i: {"PTS": 10.0 + i, "GP": 1.0} for i in range(1, 12)}  # no line carries anything but points
    flat = fit_categories(lines, nine())
    plan = plan_categories(flat, outlook_for(dict.fromkeys(flat.categories, 0.0)), games=GAMES)
    assert plan.names == ("PTS",)
    assert any("BLK" in warning and "no spread" in warning for warning in plan.warnings)
    assert set(plan.weights) == {"PTS"}


def test_game_sd_comes_from_the_models_tau(model: CategoryModel) -> None:
    assert game_sd_from_tau(model) == {}  # no tau: swing_outlook's default stands
    games = {i: [stat_line(10.0 + (g % 5) * (1 + i / 10), i) for g in range(12)] for i in range(1, 6)}
    tau = within_player_sd(games, model, games_per_matchup=4.0)
    fitted = model.with_tau(tau)
    spreads = game_sd_from_tau(fitted, games_per_matchup=4.0)
    pts = fitted.stat("PTS")
    assert spreads["PTS"] == pytest.approx(pts.tau * 2.0 / pts.spread(fitted.metric, fitted.kappa))
    assert spreads["PTS"] > 0 and "BLK" in spreads
    with pytest.raises(ValueError, match="positive"):
        game_sd_from_tau(fitted, games_per_matchup=0.0)
    # a noisier category has a wider outlook, so the same margin is a smaller chance of a flip
    wide = swing_outlook({"PTS": -4.0}, {"PTS": 0.0}, games_ours=8, games_theirs=8, game_sd=3.0)
    narrow = swing_outlook({"PTS": -4.0}, {"PTS": 0.0}, games_ours=8, games_theirs=8, game_sd=1.0)
    assert wide.win_probability["PTS"] > narrow.win_probability["PTS"]
    assert DEFAULT_GAME_SD == 1.0


def test_thresholds_and_leagues_are_checked(model: CategoryModel) -> None:
    outlook = outlook_for(dict.fromkeys(model.categories, 0.0))
    for bad in ({"punt_below": 0.0}, {"punt_below": 0.5}, {"safe_above": 0.5}, {"safe_above": 1.0}, {"max_punts": -1}):
        with pytest.raises(ValueError, match="should be"):
            plan_categories(model, outlook, games=GAMES, **bad)  # type: ignore[arg-type]
    assert 0.0 < PUNT_BELOW < 0.5 < SAFE_ABOVE < 1.0
    check_category_league(nine())
    with pytest.raises(WeeklyPlanError, match="points, not categories"):
        check_category_league(real_points())
    with pytest.raises(WeeklyPlanError, match="no weekly category matchups"):
        check_category_league(nine().model_copy(update={"scoring_type": ScoringType.ROTO}))
    nfl = load_league_settings(FIXTURES / "espn" / "ffl_settings_ppr.json")
    with pytest.raises(WeeklyPlanError, match="not an NBA"):
        check_category_league(nfl)


# --- the store-backed plan --------------------------------------------------------------------------------------------

PG, SG, SF, PF, C, G, F, UTIL = (FBA.slot_id(label) for label in ("PG", "SG", "SF", "PF", "C", "G", "F", "UTIL"))
BENCH = FBA.bench_slot
SYNCED = datetime(2026, 10, 22, 11, 0, tzinfo=UTC)  # 7 a.m. ET Thursday, day 3
MORNING = datetime(2026, 10, 22, 12, 0, tzinfo=UTC)
DAY = 3

# (espn_id, name, position, pro team, slot, points per game); the opponent's ids are ours plus 100, its teams plus 13
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


def tip(day: int) -> datetime:
    return datetime(2026, 10, 20, 23, 30, tzinfo=UTC) + timedelta(days=day - 1)


@dataclass(frozen=True)
class FakeGame:
    id: int | None
    date: datetime
    home_pro_team_id: int
    away_pro_team_id: int
    start_time_tbd: bool = False
    valid_for_locking: bool = True


class FakeSchedule:
    def __init__(self, plays: Mapping[int, Iterable[int]], teams: Iterable[int] = range(1, 31)) -> None:
        self.teams = tuple(teams)
        self._games = {
            day: [FakeGame(team * 1000 + day, tip(day), team, 99) for team in sorted(set(playing))]
            for day, playing in plays.items()
        }

    @property
    def scoring_periods(self) -> tuple[int, ...]:
        return tuple(sorted(self._games))

    def games(self, scoring_period: int) -> tuple[FakeGame, ...]:
        return tuple(sorted(self._games.get(scoring_period, ()), key=lambda game: game.date))

    def games_for(self, pro_team_id: int, scoring_period: int) -> tuple[FakeGame, ...]:
        return tuple(game for game in self.games(scoring_period) if pro_team_id == game.home_pro_team_id)

    def idle_teams(self, scoring_period: int) -> tuple[int, ...]:
        playing = {game.home_pro_team_id for game in self.games(scoring_period)}
        return tuple(team for team in self.teams if team not in playing)


def schedule() -> FakeSchedule:
    """Days 1-6, every team plays every day."""
    return FakeSchedule({day: range(1, 31) for day in range(1, 7)})


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        yield opened


def player_row(espn_id: int, name: str, position: str, team: int) -> PlayerRow:
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
        }
    )


def seed(store: Store, settings: LeagueSettings, *, opponent: bool = True) -> LeagueRow:
    league = store.leagues.upsert(
        LeagueRow(
            key="nba",
            sport="nba",
            espn_league_id=settings.league_id,
            season=SEASON,
            team_id=OUR_TEAM,
            as_of=SYNCED,
        )
    )
    store.settings.upsert(
        LeagueSettingsRow(league_id=league.row_id, settings=settings.model_dump(mode="json"), as_of=SYNCED)
    )
    for team_id in (OUR_TEAM, THEIR_TEAM):
        store.teams.upsert(TeamRow(league_id=league.row_id, team_id=team_id, name=f"Team {team_id}", as_of=SYNCED))
    for team_id, offset, mine in ((OUR_TEAM, 0, True), (THEIR_TEAM, 100, False)):
        if team_id == THEIR_TEAM and not opponent:
            continue
        entries = []
        for espn_id, name, position, team, slot, ppg in ROSTER:
            key = espn_id + offset
            store.players.upsert(player_row(key, name, position, team + (13 if offset else 0)))
            entries.append(
                RosterEntryRow(
                    league_id=league.row_id,
                    scoring_period_id=DAY,
                    team_id=team_id,
                    espn_id=key,
                    lineup_slot_id=slot,
                    as_of=SYNCED,
                )
            )
            store.projections.upsert(
                ProjectionRow(
                    sport="nba",
                    espn_id=key,
                    source="blend",
                    kind="projected",
                    season=SEASON,
                    scoring_period_id=DAY,
                    stats=stat_line(ppg, espn_id, ours=mine),
                    as_of=SYNCED,
                )
            )
        store.rosters.replace(league.row_id, DAY, team_id, entries)
    return league


def test_the_plan_concedes_the_lost_category_and_leaves_the_won_one(store: Store) -> None:
    settings = nine()
    league = seed(store, settings)
    decision = plan_weekly(store, league, schedule=schedule(), now=MORNING, period=DAY, opponent_team_id=THEIR_TEAM)
    plan = decision.plan
    assert isinstance(plan, WeeklyPlan)
    assert plan.period == DAY and plan.matchup_period == 1 and plan.opponent_team_id == THEIR_TEAM
    assert plan.names == settings.categories  # the league's categories, in the league's order
    assert plan.category("BLK").status is CategoryStatus.PUNT and plan.category("BLK").win_probability < 0.01
    assert plan.category("PTS").status is CategoryStatus.SAFE and plan.category("PTS").win_probability > 0.99
    assert plan.punts == ("BLK",)
    toss_ups = [name for name in plan.names if name not in {"BLK", "PTS"}]
    for name in toss_ups:
        entry = plan.category(name)
        assert entry.status is CategoryStatus.CONTEST and 0.3 < entry.win_probability < 0.7, name
        assert entry.weight > 100 * plan.weights["PTS"] and entry.weight > plan.weights["BLK"]
    assert plan.weights["BLK"] == 0.0 and set(plan.weights) == set(settings.categories)
    assert plan.games > 30 and plan.max_punts == 4
    # a won, a lost and seven coin flips: a majority takes four of the seven
    assert plan.matchup_win_probability == pytest.approx(0.5, abs=0.02)
    assert plan.category("TO").reverse and not plan.category("PTS").reverse
    assert plan.category("BLK").gap > 0 and plan.category("PTS").gap < 0  # to close, and a cushion
    assert set(plan.streamer_targets) == set(toss_ups)
    # information only: nothing was proposed or written
    assert store.proposals.find(league_id=league.row_id) == []


def test_the_plan_takes_the_simulators_probabilities_and_the_models_tau(store: Store) -> None:
    league = seed(store, nine())
    decision = plan_weekly(
        store,
        league,
        schedule=schedule(),
        now=MORNING,
        period=DAY,
        opponent_team_id=THEIR_TEAM,
        simulated={"REB": 0.08, "AST": 0.4},
        punt_below=0.1,
        game_sd=2.0,
    )
    plan = decision.plan
    assert plan is not None
    assert plan.category("REB").status is CategoryStatus.PUNT and plan.category("REB").source == "simulation"
    assert plan.category("AST").win_probability == 0.4 and plan.category("AST").status is CategoryStatus.CONTEST
    assert set(plan.punts) == {"REB", "BLK"} and plan.punt_below == 0.1
    wide = plan.category("STL").sd
    base = plan_weekly(store, league, schedule=schedule(), now=MORNING, period=DAY, opponent_team_id=THEIR_TEAM)
    assert base.plan is not None and wide == pytest.approx(2.0 * base.plan.category("STL").sd)


def test_without_an_opponent_there_is_nothing_to_plan_against(store: Store) -> None:
    league = seed(store, nine(), opponent=False)
    flat = plan_weekly(store, league, schedule=schedule(), now=MORNING, period=DAY)
    assert flat.plan is None and any("no opponent" in warning for warning in flat.warnings)
    gone = plan_weekly(store, league, schedule=schedule(), now=MORNING, period=DAY, opponent_team_id=THEIR_TEAM)
    assert gone.plan is None and any("no stored roster for opponent team 2" in warning for warning in gone.warnings)
    with pytest.raises(WeeklyPlanError, match="cannot be itself"):
        plan_weekly(store, league, schedule=schedule(), now=MORNING, period=DAY, opponent_team_id=OUR_TEAM)


def test_a_points_league_is_refused(store: Store) -> None:
    league = seed(store, real_points())
    with pytest.raises(WeeklyPlanError, match="points, not categories"):
        plan_weekly(store, league, schedule=schedule(), now=MORNING, period=DAY, opponent_team_id=THEIR_TEAM)
    unsynced = store.leagues.upsert(
        LeagueRow(key="other", sport="nba", espn_league_id=7, season=SEASON, team_id=OUR_TEAM, as_of=SYNCED)
    )
    with pytest.raises(WeeklyPlanError, match="no synced settings"):
        plan_weekly(store, unsynced, schedule=schedule(), now=MORNING, period=DAY)


def test_the_planner_gives_information_and_registers_no_decision() -> None:
    import fm.decide.weekly  # noqa: F401, PLC0415

    assert "weekly" not in decide_registry.registry.kinds("nba")  # the strategist (#45) calls plan_weekly itself
    parameters = inspect.signature(plan_weekly).parameters
    assert list(parameters)[:2] == ["store", "league"] and {"schedule", "now", "opponent_team_id"} <= set(parameters)
