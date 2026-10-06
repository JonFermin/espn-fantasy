"""NBA category valuation (ROADMAP #24): volume-weighted z-scores, G-scores, punt weights and tau.

Settings: the hand-built 9-cat head-to-head stand-in (``tests/fixtures/espn/fba_settings_9cat.json``: PTS, BLK, STL,
AST, REB, TO reversed, 3PM, FG%, FT%), since neither real league competes on categories; the real points league
(``tests/fixtures/espn/real/fba/mSettings.json``) is what the model refuses. Hand-built lines where the arithmetic is
the point; where real data is, ESPN's real 2027 season projections of the 16 rostered players in
``tests/fixtures/espn/real/fba/mRoster.json`` (as per-game lines) and Draymond Green's real 2026 season line in
``.../kona_player_info.json``.
"""

from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fm.espn.ids import FBA
from fm.espn.models import PlayersView, RostersView
from fm.espn.settings import LeagueSettings, load_league_settings, parse_league_settings
from fm.model.categories import (
    MIN_TAU_GAMES,
    NBA_STATS_COLUMNS,
    RATE_CATEGORIES,
    CategoryMetric,
    CategoryModel,
    StatCategory,
    actual_game_lines,
    fit_categories,
    game_log_lines,
    kappa_for,
    league_categories,
    metric_for,
    punt_weights,
    within_player_sd,
)
from fm.model.scoring import Ratio, ScoringError
from fm.model.value_nba import per_game_line
from fm.sources.nba_stats import parse_result_sets, pick_result_set
from fm.sports.base import StatSchema
from fm.store import ProjectionRow

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
NINE_CAT = FIXTURES / "espn" / "fba_settings_9cat.json"
REAL = FIXTURES / "espn" / "real" / "fba"
GAME_LOGS = FIXTURES / "sources" / "nba_stats" / "leaguegamelog_2025-26.json"
NINE = ("PTS", "BLK", "STL", "AST", "REB", "TO", "3PM", "FG%", "FT%")
JOKIC, JAYLEN_BROWN, DRAYMOND = 3112335, 3917376, 6589
SEASON = 2027


def nine_cat_view() -> dict[str, Any]:
    return json.loads(NINE_CAT.read_text(encoding="utf-8"))


@cache
def nine_cat_settings() -> LeagueSettings:
    return load_league_settings(NINE_CAT)


@pytest.fixture(scope="module")
def nine_cat() -> LeagueSettings:
    return nine_cat_settings()


def with_scoring(
    view: dict[str, Any], *, scoring_type: str | None = None, reverse_to: bool | None = None
) -> LeagueSettings:
    """The 9-cat stand-in with another scoring type, or with turnovers' ``isReverseItem`` flipped."""
    scoring = view["settings"]["scoringSettings"]
    if scoring_type is not None:
        scoring["scoringType"] = scoring_type
    if reverse_to is not None:
        for item in scoring["scoringItems"]:
            if item["statId"] == FBA.stat_id("TO"):
                item["isReverseItem"] = reverse_to
    return parse_league_settings(view)


@cache
def real_per_game() -> dict[int, dict[str, float]]:
    """The real rostered players' 2027 ESPN projections as per-game lines."""
    view = RostersView.model_validate(json.loads((REAL / "mRoster.json").read_text(encoding="utf-8")))
    schema = StatSchema.for_game("fba")
    lines: dict[int, dict[str, float]] = {}
    for team in view.teams:
        for entry in team.entries:
            projection = entry.player.projection(SEASON, 0)
            assert projection is not None
            line = per_game_line(schema.from_espn(projection.stats))
            assert line is not None
            lines[entry.player_id] = line
    return lines


def line(**stats: float) -> dict[str, float]:
    return dict(stats)


# --- the league's categories ---


def test_categories_their_order_and_direction_come_from_the_settings(nine_cat: LeagueSettings) -> None:
    categories = league_categories(nine_cat)
    assert tuple(category.stat for category in categories) == NINE
    by_stat = {category.stat: category for category in categories}
    assert by_stat["TO"].reverse and by_stat["TO"].sign == -1.0
    assert not any(category.reverse for category in categories if category.stat != "TO")
    assert by_stat["FG%"].rate == Ratio("FGM", "FGA") and by_stat["FT%"].rate == Ratio("FTM", "FTA")
    assert all(by_stat[stat].rate is None for stat in ("PTS", "BLK", "STL", "AST", "REB", "TO", "3PM"))
    # turnovers count against because the league says so, not because the code knows TO
    flipped = league_categories(with_scoring(nine_cat_view(), reverse_to=False))
    assert not any(category.reverse for category in flipped)


def test_only_an_nba_category_league_has_categories(nine_cat: LeagueSettings) -> None:
    with pytest.raises(ValueError, match="scores points, not categories"):
        league_categories(load_league_settings(REAL / "mSettings.json"))
    with pytest.raises(ValueError, match="not an NBA"):
        league_categories(load_league_settings(FIXTURES / "espn" / "ffl_settings_ppr.json"))
    with pytest.raises(ValueError, match="scores points"):
        fit_categories(real_per_game(), load_league_settings(REAL / "mSettings.json"))


def test_rate_categories_are_the_ratios_espn_reports() -> None:
    # Draymond Green's real 2026 season line carries every fba stat; each rate is its two counting stats' ratio
    view = PlayersView.model_validate(json.loads((REAL / "kona_player_info.json").read_text(encoding="utf-8")))
    actual = view.entry(DRAYMOND).player.actual(2026, 0)
    assert actual is not None
    season = StatSchema.for_game("fba").from_espn(actual.stats)
    assert set(RATE_CATEGORIES) <= set(season)
    for stat, rule in RATE_CATEGORIES.items():
        category = StatCategory(stat, rate=rule)
        assert category.made(season) / category.volume(season) == pytest.approx(season[stat], rel=1e-6), stat
    assert set(RATE_CATEGORIES) == {
        *("FG%", "FT%", "3PT%", "AFG%", "APG", "BPG", "MPG", "PPG", "RPG", "SPG", "TOPG", "3PG"),
        *("PPM", "A/TO", "STR", "FTR"),
    }


# --- values and z-scores ---


def shooters() -> dict[int, dict[str, float]]:
    """Four shooters and a player without a shot: 50 % on 20, 70 % on 2, 40 % on 10, 37.5 % on 8, nothing."""
    return {
        1: line(FGM=10.0, FGA=20.0, PTS=24.0),
        2: line(FGM=1.4, FGA=2.0, PTS=3.0),
        3: line(FGM=4.0, FGA=10.0, PTS=10.0),
        4: line(FGM=3.0, FGA=8.0, PTS=8.0),
        5: line(PTS=1.0),
    }


def test_percentage_categories_are_volume_weighted(nine_cat: LeagueSettings) -> None:
    lines = shooters()
    model = fit_categories(lines, nine_cat, metric=CategoryMetric.Z)
    fg = model.stat("FG%")
    made = np.array([lines[espn_id].get("FGM", 0.0) for espn_id in sorted(lines)])
    attempts = np.array([lines[espn_id].get("FGA", 0.0) for espn_id in sorted(lines)])
    pool_pct = made.sum() / attempts.sum()  # 18.4 / 40 = 46 %, not the mean of the percentages
    assert fg.pool_rate == pytest.approx(0.46) and fg.mean_volume == pytest.approx(8.0)
    assert pool_pct != pytest.approx(np.mean([0.5, 0.7, 0.4, 0.375]))
    for espn_id, stats in lines.items():
        attempts_p = stats.get("FGA", 0.0)
        expected = (attempts_p / 8.0) * ((stats["FGM"] / attempts_p if attempts_p else 0.0) - pool_pct)
        assert model.values(stats)["FG%"] == pytest.approx(expected), espn_id
    z = {espn_id: model.z_scores(stats)["FG%"] for espn_id, stats in lines.items()}
    assert z[1] > z[2] > 0 > z[3] > z[4]  # the 50 % shooter on 20 shots beats the 70 % shooter on two
    assert z[5] == 0.0  # no attempts: no effect on the team's percentage
    assert fg.mean == 0.0
    assert z[1] == pytest.approx(
        model.values(lines[1])["FG%"] / float(np.std([model.values(s)["FG%"] for s in lines.values()]))
    )
    # an unweighted z-score of the percentages would have put the 70 % shooter first
    pct = {espn_id: stats["FGM"] / stats["FGA"] for espn_id, stats in lines.items() if "FGA" in stats}
    assert max(pct, key=pct.__getitem__) == 2


def test_free_throws_are_weighted_by_attempts_too(nine_cat: LeagueSettings) -> None:
    lines = {
        1: line(FTM=9.0, FTA=10.0),
        2: line(FTM=1.0, FTA=1.0),
        3: line(FTM=3.0, FTA=6.0),
        4: line(FTM=0.0, FTA=0.0),
    }
    model = fit_categories(lines, nine_cat)
    ft = model.stat("FT%")
    assert ft.pool_rate == pytest.approx(13.0 / 17.0) and ft.mean_volume == pytest.approx(17.0 / 4)
    perfect, sure = model.scores(lines[2])["FT%"], model.scores(lines[1])["FT%"]
    assert sure > perfect > 0  # 90 % on ten attempts beats 100 % on one


def test_turnovers_count_against(nine_cat: LeagueSettings) -> None:
    lines = {espn_id: line(TO=turnovers, PTS=10.0 + espn_id) for espn_id, turnovers in enumerate((1.0, 2.0, 3.0, 6.0))}
    model = fit_categories(lines, nine_cat, metric=CategoryMetric.Z)
    to = model.stat("TO")
    scores = {espn_id: model.scores(stats)["TO"] for espn_id, stats in lines.items()}
    assert scores[0] > scores[1] > scores[2] > scores[3]
    assert scores[3] == pytest.approx(-(6.0 - to.mean) / to.sd)
    assert all(model.contributions(stats)["TO"] < 0 for stats in lines.values())  # any turnover hurts a team


def test_contributions_measure_a_line_from_an_empty_slot(nine_cat: LeagueSettings) -> None:
    lines = real_per_game()
    model = fit_categories(lines, nine_cat, metric=CategoryMetric.Z)
    jokic = lines[JOKIC]
    contributions, scores = model.contributions(jokic), model.scores(jokic)
    pts = model.stat("PTS")
    assert contributions["PTS"] == pytest.approx(jokic["PTS"] / pts.sd)
    assert scores["PTS"] == pytest.approx((jokic["PTS"] - pts.mean) / pts.sd)
    assert contributions["FG%"] == pytest.approx(scores["FG%"])  # the pool's mean FG% value is 0 by construction
    assert model.contributions({}) == dict.fromkeys(NINE, 0.0)
    assert model.contribution(jokic) == pytest.approx(math.fsum(contributions.values()))
    assert model.contribution({}) == 0.0 and model.total({}) < 0  # an empty slot is below the average player


# --- G-scores ---


def test_kappa_is_2n_over_2n_minus_1(nine_cat: LeagueSettings) -> None:
    assert kappa_for(1) == 2.0
    assert kappa_for(13) == pytest.approx(26 / 25)
    with pytest.raises(ValueError, match="at least 1"):
        kappa_for(0)
    model = fit_categories(real_per_game(), nine_cat)
    assert model.kappa == pytest.approx(20 / 19)  # ten active slots a side by default
    assert fit_categories(real_per_game(), nine_cat, players_per_team=13).kappa == pytest.approx(26 / 25)


def test_the_metric_follows_the_scoring_type(nine_cat: LeagueSettings) -> None:
    assert metric_for(nine_cat) is CategoryMetric.G  # H2H_MOST_CATEGORIES
    assert metric_for(with_scoring(nine_cat_view(), scoring_type="H2H_CATEGORY")) is CategoryMetric.G
    roto = with_scoring(nine_cat_view(), scoring_type="ROTO")
    assert metric_for(roto) is CategoryMetric.Z
    assert fit_categories(real_per_game(), roto).metric is CategoryMetric.Z
    assert fit_categories(real_per_game(), nine_cat).metric is CategoryMetric.G


def test_the_g_score_of_real_players_is_the_z_score_when_tau_is_0(nine_cat: LeagueSettings) -> None:
    lines = real_per_game()
    for model in (fit_categories(lines, nine_cat), fit_categories(lines, nine_cat, tau=dict.fromkeys(NINE, 0.0))):
        for stats in lines.values():
            assert model.g_scores(stats) == model.z_scores(stats)


def test_g_scores_discount_categories_that_swing_from_week_to_week(nine_cat: LeagueSettings) -> None:
    lines = real_per_game()
    plain = fit_categories(lines, nine_cat)
    noisy = plain.with_tau({"STL": 0.8, "FG%": 0.5})
    kappa = plain.kappa
    for stats in lines.values():
        z, g = plain.z_scores(stats), noisy.g_scores(stats)
        stl, fg = plain.stat("STL"), plain.stat("FG%")
        assert g["STL"] == pytest.approx(z["STL"] * stl.sd / math.sqrt(stl.sd**2 + kappa * 0.8**2))
        assert g["FG%"] == pytest.approx(z["FG%"] * fg.sd / math.sqrt(fg.sd**2 + kappa * 0.5**2))
        assert {stat: g[stat] for stat in NINE if stat not in ("STL", "FG%")} == {
            stat: z[stat] for stat in NINE if stat not in ("STL", "FG%")
        }
        assert abs(g["STL"]) <= abs(z["STL"])
    assert noisy.z_scores(lines[JOKIC]) == plain.z_scores(lines[JOKIC])  # tau leaves the z-score alone
    with pytest.raises(ValueError, match="tau for OREB"):
        plain.with_tau({"OREB": 1.0})
    with pytest.raises(ValueError, match="tau of STL"):
        plain.with_tau({"STL": -1.0})


_shot = st.floats(min_value=0.0, max_value=30.0, allow_nan=False)
_pct = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
_player = st.fixed_dictionaries(
    {
        "PTS": _shot,
        "BLK": st.floats(min_value=0.0, max_value=5.0),
        "STL": st.floats(min_value=0.0, max_value=4.0),
        "AST": st.floats(min_value=0.0, max_value=12.0),
        "REB": st.floats(min_value=0.0, max_value=15.0),
        "TO": st.floats(min_value=0.0, max_value=6.0),
        "3PM": st.floats(min_value=0.0, max_value=5.0),
        "FGA": _shot,
        "FG_PCT": _pct,
        "FTA": st.floats(min_value=0.0, max_value=12.0),
        "FT_PCT": _pct,
    }
)


def _as_line(draw: dict[str, float]) -> dict[str, float]:
    stats = {key: value for key, value in draw.items() if not key.endswith("_PCT")}
    stats["FGM"] = draw["FGA"] * draw["FG_PCT"]
    stats["FTM"] = draw["FTA"] * draw["FT_PCT"]
    return stats


@settings(deadline=None)
@given(
    players=st.lists(_player, min_size=2, max_size=14),
    tau=st.dictionaries(st.sampled_from(NINE), st.floats(min_value=0.01, max_value=5.0)),
    pool_size=st.integers(min_value=2, max_value=14),
)
def test_g_score_reduces_to_z_score_when_tau_is_0(
    players: list[dict[str, float]], tau: dict[str, float], pool_size: int
) -> None:
    lines = {espn_id: _as_line(draw) for espn_id, draw in enumerate(players)}
    model = fit_categories(lines, nine_cat_settings(), pool_size=pool_size)
    for stats in lines.values():
        assert model.g_scores(stats) == model.z_scores(stats)  # exactly: sqrt(sigma^2 + kappa * 0^2) is sigma
    noisy = model.with_tau(tau)
    spread = [stat for stat in NINE if model.stat(stat).sd > 0]  # without spread a z-score is 0 and a G-score is not
    for stats in lines.values():
        z, g = model.z_scores(stats), noisy.g_scores(stats)
        assert all(abs(g[stat]) <= abs(z[stat]) for stat in spread)
        assert all(g[stat] == z[stat] for stat in NINE if stat not in tau)


# --- the pool and punts ---


def test_the_pool_is_the_players_a_league_could_roster(nine_cat: LeagueSettings) -> None:
    lines = {espn_id: line(PTS=float(espn_id), REB=float(espn_id) / 2, FGM=4.0, FGA=9.0) for espn_id in range(1, 41)}
    model = fit_categories(lines, nine_cat, pool_size=10)
    assert model.pool == tuple(range(31, 41))  # the ten best, after re-picking
    assert model.stat("PTS").mean == pytest.approx(np.mean(range(31, 41)))
    assert model.stat("PTS").sd == pytest.approx(np.std(range(31, 41)))
    assert model.total(lines[40]) > 0 > model.total(lines[30])  # replacement level sits below the pool
    everyone = fit_categories(lines, nine_cat)  # the default, 10 teams x 13 rostered, is more than 40 players
    assert everyone.pool == tuple(range(1, 41)) and everyone.stat("PTS").mean == pytest.approx(20.5)
    one_round = fit_categories(lines, nine_cat, pool_size=10, iterations=0)
    assert one_round.pool == tuple(range(1, 41))


def test_punted_categories_drop_out_of_the_total(nine_cat: LeagueSettings) -> None:
    lines = real_per_game()
    model = fit_categories(lines, nine_cat)
    weights = punt_weights(model.categories, ["FT%", "TO"])
    assert weights == {**dict.fromkeys(NINE, 1.0), "FT%": 0.0, "TO": 0.0}
    for stats in lines.values():
        scores = model.scores(stats)
        assert model.total(stats, weights) == pytest.approx(
            math.fsum(v for k, v in scores.items() if k not in ("FT%", "TO"))
        )
        assert model.total(stats) == pytest.approx(math.fsum(scores.values()))
    ranked = model.rank(lines, weights)
    assert [entry.total for entry in ranked] == sorted((entry.total for entry in ranked), reverse=True)
    assert ranked[0].espn_id == JOKIC and ranked[0].scores == model.scores(lines[JOKIC])
    assert model.total(lines[JOKIC], {"PTS": 2.0}) == pytest.approx(
        model.total(lines[JOKIC]) + model.scores(lines[JOKIC])["PTS"]
    )
    with pytest.raises(ValueError, match="cannot punt OREB"):
        punt_weights(model.categories, ["OREB"])
    with pytest.raises(ValueError, match="weights for OREB"):
        model.total(lines[JOKIC], {"OREB": 1.0})
    with pytest.raises(ValueError, match="weight of FT%"):
        model.total(lines[JOKIC], {"FT%": -1.0})


def test_real_rostered_players(nine_cat: LeagueSettings) -> None:
    lines = real_per_game()
    model = fit_categories(lines, nine_cat)
    assert len(model.pool) == 16 and model.warnings == ()
    fg = model.stat("FG%")
    assert fg.pool_rate == pytest.approx(sum(s["FGM"] for s in lines.values()) / sum(s["FGA"] for s in lines.values()))
    turnovers = max(lines, key=lambda espn_id: lines[espn_id]["TO"])
    assert min(lines, key=lambda espn_id: model.scores(lines[espn_id])["TO"]) == turnovers
    assert max(lines, key=lambda espn_id: model.scores(lines[espn_id])["FG%"]) == JOKIC  # volume and accuracy
    assert model.rank(lines)[0].espn_id == JOKIC


def test_a_category_nothing_separates_scores_0_and_is_named(nine_cat: LeagueSettings) -> None:
    lines = {1: line(PTS=20.0, FGM=8.0, FGA=16.0), 2: line(PTS=10.0, FGM=4.0, FGA=10.0), 3: line(PTS=5.0)}
    model = fit_categories(lines, nine_cat)
    assert model.scores(lines[1])["BLK"] == 0.0 and model.scores(lines[1])["FT%"] == 0.0
    assert any(warning.startswith("BLK: no spread") for warning in model.warnings)
    assert any(warning.startswith("FT%: no spread") and "FTM or FTA" in warning for warning in model.warnings)


def test_bad_lines_and_parameters_are_refused(nine_cat: LeagueSettings) -> None:
    with pytest.raises(ValueError, match="at least two players"):
        fit_categories({1: line(PTS=1.0)}, nine_cat)
    with pytest.raises(ScoringError, match="ESPN 2: .*'PTS'"):
        fit_categories({1: line(PTS=1.0), 2: line(PTS=math.nan)}, nine_cat)
    with pytest.raises(ValueError, match="tau for DD"):
        fit_categories(real_per_game(), nine_cat, tau={"DD": 1.0})
    with pytest.raises(ValueError, match="pool_size"):
        fit_categories(real_per_game(), nine_cat, pool_size=1)
    with pytest.raises(KeyError, match="'OREB' is not a category"):
        fit_categories(real_per_game(), nine_cat).stat("OREB")


# --- tau from single games ---


def logs(rows: list[tuple[int, float, float, float, float, float]]) -> pl.DataFrame:
    """A stats.nba.com game-log frame: (player, minutes, points, steals, field goals made, attempted)."""
    return pl.DataFrame(
        {
            "PLAYER_ID": [row[0] for row in rows],
            "MIN": [float(row[1]) for row in rows],
            "PTS": [float(row[2]) for row in rows],
            "STL": [float(row[3]) for row in rows],
            "FGM": [float(row[4]) for row in rows],
            "FGA": [float(row[5]) for row in rows],
            "TOV": [1.0 for _ in rows],
        }
    )


def test_game_logs_become_single_game_lines() -> None:
    frame = pick_result_set(parse_result_sets(GAME_LOGS.read_bytes()), "LeagueGameLog", ("PLAYER_ID",))
    games = game_log_lines(frame)
    assert games[203999][0] == {
        **{"GP": 1.0, "MIN": 34.0, "PTS": 30.0, "FGM": 11.0, "FGA": 19.0, "3PM": 1.0, "3PA": 4.0, "FTM": 7.0},
        **{"FTA": 8.0, "OREB": 3.0, "DREB": 10.0, "REB": 13.0, "AST": 12.0, "STL": 1.0, "BLK": 1.0, "TO": 3.0},
        "PF": 2.0,
    }
    assert set(NBA_STATS_COLUMNS) <= set(frame.columns)
    sparse = game_log_lines(
        pl.DataFrame({"PLAYER_ID": [1, 1, 2], "MIN": ["31:30", "0:00", None], "PTS": [12, 0, 3], "TOV": [2, 0, 1]})
    )
    assert sparse == {1: [{"GP": 1.0, "MIN": 31.5, "PTS": 12.0, "TO": 2.0}], 2: [{"GP": 1.0, "PTS": 3.0, "TO": 1.0}]}
    with pytest.raises(ValueError, match="PLAYER_ID"):
        game_log_lines(pl.DataFrame({"PTS": [1]}))


def test_tau_is_the_mean_game_to_game_variance_over_a_matchups_games(nine_cat: LeagueSettings) -> None:
    model = fit_categories(
        {1: line(PTS=20.0, STL=1.0, FGM=8.0, FGA=16.0), 2: line(PTS=10.0, STL=2.0, FGM=3.0, FGA=10.0)}, nine_cat
    )
    fg = model.stat("FG%")
    rows = [
        *((7, 30.0, points, steals, made, 10.0) for points, steals, made in ((10, 0, 4), (20, 2, 6), (30, 1, 5))),
        *((8, 25.0, points, steals, made, 8.0) for points, steals, made in ((5, 1, 2), (15, 1, 3), (10, 4, 4))),
        (9, 20.0, 40.0, 3.0, 9.0, 12.0),  # one game only: below min_games
    ]
    games = game_log_lines(logs(rows))
    tau = within_player_sd(games, model, games_per_matchup=3.5, min_games=3)
    assert set(tau) == {"PTS", "STL", "TO", "FG%"}  # the logs carry nothing for BLK, AST, REB, 3PM or FT%
    assert tau["PTS"] == pytest.approx(
        math.sqrt((np.var([10, 20, 30], ddof=1) + np.var([5, 15, 10], ddof=1)) / 2 / 3.5)
    )
    assert tau["STL"] == pytest.approx(math.sqrt((np.var([0, 2, 1], ddof=1) + np.var([1, 1, 4], ddof=1)) / 2 / 3.5))
    assert tau["TO"] == 0.0

    def fg_value(made: float, attempts: float) -> float:
        return (made - fg.pool_rate * attempts) / fg.mean_volume

    variances = [
        np.var([fg_value(m, 10.0) for m in (4, 6, 5)], ddof=1),
        np.var([fg_value(m, 8.0) for m in (2, 3, 4)], ddof=1),
    ]
    assert tau["FG%"] == pytest.approx(math.sqrt(sum(variances) / 2 / 3.5))
    only_7 = within_player_sd(games, model, games_per_matchup=1.0, min_games=3, players=[7])
    assert only_7["PTS"] == pytest.approx(10.0)  # the sample SD of 10, 20, 30
    tuned = model.with_tau(tau)
    assert tuned.stat("PTS").tau == tau["PTS"] and tuned.stat("BLK").tau == 0.0
    with pytest.raises(ValueError, match=f"no player has {MIN_TAU_GAMES} games"):
        within_player_sd(games, model, games_per_matchup=3.5)
    with pytest.raises(ValueError, match="games_per_matchup"):
        within_player_sd(games, model, games_per_matchup=0.0, min_games=3)
    with pytest.raises(ValueError, match="min_games"):
        within_player_sd(games, model, games_per_matchup=3.5, min_games=1)


def test_espns_daily_actuals_feed_tau_as_well() -> None:
    def row(espn_id: int, period: int, *, kind: str = "actual", sport: str = "nba", **stats: float) -> ProjectionRow:
        return ProjectionRow.model_validate(
            {
                "sport": sport,
                "espn_id": espn_id,
                "source": "espn",
                "kind": kind,
                "season": SEASON,
                "scoring_period_id": period,
                "stats": stats,
                "as_of": datetime(2026, 10, 23, 12, tzinfo=UTC),
            }
        )

    rows = [
        row(JOKIC, 3, MIN=35.0, PTS=31.0),
        row(JOKIC, 1, MIN=34.0, PTS=25.0, GP=1.0),
        row(JOKIC, 2, MIN=0.0, PTS=0.0),  # did not play
        row(JOKIC, 0, MIN=69.0, PTS=56.0),  # the season line
        row(JOKIC, 4, kind="projected", MIN=36.0, PTS=29.0),
        row(3918298, 4, sport="nfl", PY=250.0),
        row(JAYLEN_BROWN, 2, PTS=22.0),
    ]
    assert actual_game_lines(rows) == {
        JOKIC: [{"GP": 1.0, "MIN": 34.0, "PTS": 25.0}, {"GP": 1.0, "MIN": 35.0, "PTS": 31.0}],
        JAYLEN_BROWN: [{"GP": 1.0, "PTS": 22.0}],
    }


def test_values_derive_rebounds_from_their_parts(nine_cat: LeagueSettings) -> None:
    model: CategoryModel = fit_categories(real_per_game(), nine_cat)
    assert model.categories == NINE
    assert model.values(real_per_game()[JOKIC])["REB"] == pytest.approx(real_per_game()[JOKIC]["REB"])
    # REB derived from OREB and DREB when a line has only those (DARKO's lines carry both and REB)
    assert model.values({"OREB": 2.0, "DREB": 6.0})["REB"] == 8.0
