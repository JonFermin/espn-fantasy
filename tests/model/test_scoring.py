"""League scoring (ROADMAP #15): PPR, half-PPR and custom scoring from league settings, derived stats, position
overrides, NBA points, and category pass-through.

``tests/fixtures/model/espn_ffl_pool_week4.json`` holds ESPN's own week-4 lines (actual and projected) for eight
players, each with ESPN's ``appliedTotal`` under its default PPR league, so the scorer is checked against ESPN's
arithmetic and not only against hand sums. Half-PPR and the custom formats are the PPR fixture league with edited
scoring items: league settings are data, so every format is the same ``mSettings`` payload with different items.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
from collections.abc import Iterable, Mapping
from functools import cache
from pathlib import Path
from typing import Any, cast

import pytest

from fm.espn.ids import FFL, Game
from fm.espn.models import PlayerStats, PlayersView
from fm.espn.settings import LeagueSettings, load_league_settings, parse_league_settings
from fm.model.scoring import (
    DERIVATIONS,
    FBA_DERIVATIONS,
    FFL_DERIVATIONS,
    Bracket,
    Every,
    Score,
    Scorer,
    ScoringError,
    Total,
    breakdown,
    categories,
    derive_stats,
    points,
    rule_inputs,
    score,
)
from fm.sports.base import StatSchema

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
FFL_PPR = FIXTURES / "espn" / "ffl_settings_ppr.json"
FBA_POINTS = FIXTURES / "espn" / "fba_settings_points.json"
FBA_9CAT = FIXTURES / "espn" / "fba_settings_9cat.json"
POOL = FIXTURES / "model" / "espn_ffl_pool_week4.json"
SEASON, WEEK = 2026, 4
SCHEMA = StatSchema.for_game(Game.FFL)

ALLEN, MAHOMES, HENRY, GIBBS = 3918298, 3139477, 3043078, 4429795
MCBRIDE, CHASE, BUTKER, EAGLES = 4361307, 4362628, 3055899, -16021

# One NBA box score from tests/fixtures/espn/fba_scoreboard_9cat_day1.json (the first away starter on day 1).
NBA_LINE: dict[str, float] = {
    "PTS": 33.0,
    "BLK": 1.0,
    "STL": 2.0,
    "AST": 7.0,
    "REB": 5.0,
    "TO": 2.0,
    "FGM": 12.0,
    "FGA": 22.0,
    "FTM": 8.0,
    "FTA": 9.0,
    "3PM": 1.0,
}


def ppr_view() -> dict[str, Any]:
    return json.loads(FFL_PPR.read_text(encoding="utf-8"))


def variant(
    *,
    points: Mapping[str, float] | None = None,
    overrides: Mapping[str, Mapping[str, float]] | None = None,
    drop: Iterable[str] = (),
) -> LeagueSettings:
    """The PPR fixture league with scoring items changed (``points``), per-position points set (``overrides``, by
    position label) or items removed (``drop``), all by ESPN stat abbreviation, parsed by the real parser."""
    view = copy.deepcopy(ppr_view())
    items: list[dict[str, Any]] = view["settings"]["scoringSettings"]["scoringItems"]
    dropped = {FFL.stat_id(stat) for stat in drop}
    items[:] = [item for item in items if item["statId"] not in dropped]

    def item_for(stat: str) -> dict[str, Any]:
        stat_id = FFL.stat_id(stat)
        found = next((item for item in items if item["statId"] == stat_id), None)
        if found is None:
            found = {"statId": stat_id, "points": 0.0, "pointsOverrides": {}, "isReverseItem": False}
            items.append(found)
        return found

    for stat, value in (points or {}).items():
        item_for(stat)["points"] = value
    for stat, per_position in (overrides or {}).items():
        by_id = {str(FFL.position_id(label)): value for label, value in per_position.items()}
        item_for(stat)["pointsOverrides"] = by_id
    return parse_league_settings(view)


@cache
def pool() -> PlayersView:
    return PlayersView.model_validate(json.loads(POOL.read_text(encoding="utf-8")))


def entry(espn_id: int, *, projected: bool = False) -> PlayerStats:
    found = pool().entry(espn_id).player.stat_entry(season=SEASON, scoring_period=WEEK, projected=projected)
    assert found is not None
    return found


def line(espn_id: int, *, projected: bool = False) -> dict[str, float]:
    return SCHEMA.from_espn(entry(espn_id, projected=projected).stats)


def position(espn_id: int) -> int:
    position_id = pool().entry(espn_id).player.default_position_id
    assert position_id is not None
    return position_id


@pytest.fixture(scope="module")
def ppr() -> LeagueSettings:
    return load_league_settings(FFL_PPR)


@pytest.fixture(scope="module")
def half_ppr() -> LeagueSettings:
    return variant(points={"REC": 0.5})


@pytest.fixture(scope="module")
def custom() -> LeagueSettings:
    """Six-point passing TDs, a point per 25 passing yards instead of per-yard decimals, half-PPR with a full-point
    TE premium on top, half a point per rushing and receiving first down, and 300-yard and 100-yard game bonuses."""
    return variant(
        points={"PTD": 6.0, "PY25": 1.0, "REC": 0.5, "RFD": 0.5, "REFD": 0.5, "P300": 3.0, "RY100": 3.0},
        overrides={"REC": {"TE": 1.5}},
        drop=("PY",),
    )


# --- PPR against ESPN's own arithmetic ---


def all_lines() -> list[tuple[int, bool]]:
    return [(player.id, projected) for player in pool().players for projected in (False, True)]


@pytest.mark.parametrize(("espn_id", "projected"), all_lines())
def test_ppr_reproduces_espn_applied_total_for_every_real_line(espn_id: int, projected: bool) -> None:
    # ESPN's default PPR league (where the pool came from) scores what the PPR fixture does, except missed PATs.
    espn_default = variant(drop=("PATM",))
    stats = entry(espn_id, projected=projected)
    assert stats.applied_total is not None
    got = Scorer(espn_default).points(SCHEMA.from_espn(stats.stats), position=position(espn_id))
    assert got == pytest.approx(stats.applied_total, abs=1e-6)


def test_the_ppr_fixture_charges_missed_pats_that_espns_default_does_not(ppr: LeagueSettings) -> None:
    butker = line(BUTKER, projected=True)
    applied = entry(BUTKER, projected=True).applied_total
    assert applied is not None
    assert Scorer(ppr).points(butker, position="K") == pytest.approx(applied - butker["PATM"])


def test_ppr_scores_a_real_line_item_by_item(ppr: LeagueSettings) -> None:
    gibbs = line(GIBBS)
    assert breakdown(gibbs, ppr, position="RB") == pytest.approx({"RY": 4.6, "RTD": 6.0, "REY": 3.1, "REC": 4.0})
    assert points(gibbs, ppr, position="RB") == pytest.approx(17.7)
    allen = line(ALLEN)
    assert breakdown(allen, ppr, position="QB") == pytest.approx(
        {"PY": 10.12, "PTD": 4.0, "INTT": -2.0, "RY": 0.4, "RTD": 6.0}
    )


def test_d_st_points_use_the_position_override(ppr: LeagueSettings) -> None:
    eagles = line(EAGLES)  # 2 INT, 1 sack, 24 points and 415 yards allowed
    scorer = Scorer(ppr)
    assert scorer.points(eagles, position="D/ST") == pytest.approx(2.0)  # INT 2.0 each for a D/ST, sack 1, YA449 -3
    assert scorer.points(eagles, position=16) == pytest.approx(2.0)
    assert scorer.points(eagles) == pytest.approx(4.0)  # without a position every interception earns the base 3.0
    assert scorer.breakdown(eagles, position="D/ST") == pytest.approx({"INT": 4.0, "SK": 1.0, "YA449": -3.0})


# --- half-PPR ---


@pytest.mark.parametrize(("espn_id", "projected"), all_lines())
def test_half_ppr_is_ppr_less_half_a_point_per_reception(
    ppr: LeagueSettings, half_ppr: LeagueSettings, espn_id: int, projected: bool
) -> None:
    stats = line(espn_id, projected=projected)
    where = position(espn_id)
    difference = points(stats, ppr, position=where) - points(stats, half_ppr, position=where)
    assert difference == pytest.approx(0.5 * stats.get("REC", 0.0))


def test_half_ppr_by_hand(half_ppr: LeagueSettings) -> None:
    assert points(line(GIBBS), half_ppr, position="RB") == pytest.approx(15.7)  # 4.6 + 6 + 3.1 + 4 x 0.5
    assert points(line(MCBRIDE), half_ppr, position="TE") == pytest.approx(6.6)  # 3.1 + 7 x 0.5
    assert points(line(ALLEN), half_ppr, position="QB") == pytest.approx(18.52)  # no receptions, no change


# --- custom scoring ---


def test_custom_scoring_by_hand(custom: LeagueSettings) -> None:
    scorer = Scorer(custom)
    # 253 yards = 10 x PY25 (ESPN's line carries it), TD 6, INT -2, 4 rushing yards, rushing TD, 2 rushing first downs
    assert scorer.breakdown(line(ALLEN), position="QB") == pytest.approx(
        {"PTD": 6.0, "INTT": -2.0, "RY": 0.4, "RTD": 6.0, "PY25": 10.0, "RFD": 1.0}
    )
    assert scorer.points(line(GIBBS), position="RB") == pytest.approx(17.2)  # 4.6 + 6 + 3.1 + 2 + 1.5 first downs
    assert scorer.points(line(HENRY), position="RB") == pytest.approx(18.4)  # 7.3 + 6 + 0.6 + 1 + 3.5 first downs
    assert scorer.points(line(CHASE), position="WR") == pytest.approx(4.7)  # 2.7 + 1.5 + 0.5 for a receiving 1st down


def test_custom_te_premium_applies_to_tight_ends_only(custom: LeagueSettings) -> None:
    mcbride = line(MCBRIDE)  # 7 catches, 31 yards
    scorer = Scorer(custom)
    assert scorer.points(mcbride, position="TE") == pytest.approx(3.1 + 7 * 1.5)
    assert scorer.points(mcbride, position="WR") == pytest.approx(3.1 + 7 * 0.5)


def test_custom_scoring_derives_what_a_non_espn_line_lacks(custom: LeagueSettings) -> None:
    scorer = Scorer(custom)
    sleeper_like = {"PY": 225.81, "PTD": 1.22, "RY": 104.0}  # no PY25, P300 or RY100: other sources never carry them
    assert scorer.breakdown(sleeper_like, position="QB") == pytest.approx(
        {"PTD": 7.32, "RY": 10.4, "PY25": 9.0, "RY100": 3.0}
    )
    big_day = {"PY": 312.0, "PTD": 2.0}
    assert scorer.breakdown(big_day, position="QB") == pytest.approx({"PTD": 12.0, "PY25": 12.0, "P300": 3.0})
    assert Scorer(custom, derive=False).points(sleeper_like, position="QB") == pytest.approx(7.32 + 10.4)


def test_espn_carried_bonuses_win_over_derivation(custom: LeagueSettings) -> None:
    # ESPN's projection carries P300 as a probability; it is used as given, never recomputed from projected yards.
    allen = line(ALLEN, projected=True)
    assert Scorer(custom).breakdown(allen, position="QB")["P300"] == pytest.approx(3.0 * allen["P300"])


# --- derived stats ---


def test_derive_stats_fills_only_missing_derivable_stats() -> None:
    derived = derive_stats({"PY": 237.0, "PY25": 7.0, "PFUML": 0.2, "RFUML": 0.1, "INTT": 0.7}, "nfl")
    assert derived["PY25"] == 7.0  # kept, not recomputed (9)
    assert derived["PY10"] == 23.0
    assert derived["P300"] == 0.0
    assert derived["FUML"] == pytest.approx(0.3)
    assert derived["TT"] == pytest.approx(1.0)  # INTT + the derived FUML
    assert "RY10" not in derived  # no rushing yards, nothing derived from them


def test_every_n_never_goes_negative_and_brackets_have_open_ends() -> None:
    assert derive_stats({"RY": -4.0}, "nfl", ("RY10",)) == {"RY": -4.0, "RY10": 0.0}
    assert derive_stats({"YA": 80.0}, "nfl", ("YA100", "YA550")) == {"YA": 80.0, "YA100": 1.0, "YA550": 0.0}
    assert derive_stats({"PTSA": 0.0}, "ffl", ("PA0", "PA1")) == {"PTSA": 0.0, "PA0": 1.0, "PA1": 0.0}
    assert derive_stats({"PTSA": 46.0}, Game.FFL, ("PA35", "PA46"))["PA46"] == 1.0


BRACKETS: dict[str, tuple[str, ...]] = {
    "PTSA": ("PA0", "PA1", "PA7", "PA14", "PA18", "PA22", "PA28", "PA35", "PA46"),
    "YA": ("YA100", "YA199", "YA299", "YA349", "YA399", "YA449", "YA499", "YA549", "YA550"),
    "PY": ("P300", "P400"),
    "RY": ("RY100", "RY200"),
    "REY": ("REY100", "REY200"),
}
"""Each bracket family by the stat it reads. ESPN's names give the ranges: PA1 is 1-6 points allowed, PA7 7-13 and so
on to PA46, 46 or more; YA100 is under 100 yards allowed, YA199 100-199 and so on to YA550, 550 or more; P300 is a
300-399 yard passing game and P400 400 or more; RY100 and REY100 are 100-199 yard games, RY200 and REY200 200 or
more."""

BRACKET_EDGES: dict[str, tuple[tuple[float, str | None], ...]] = {
    "PTSA": (
        (0, "PA0"),
        (1, "PA1"),
        (6, "PA1"),
        (7, "PA7"),
        (13, "PA7"),
        (14, "PA14"),
        (17, "PA14"),
        (18, "PA18"),
        (21, "PA18"),
        (22, "PA22"),
        (27, "PA22"),
        (28, "PA28"),
        (34, "PA28"),
        (35, "PA35"),
        (45, "PA35"),
        (46, "PA46"),
    ),
    "YA": (
        (0, "YA100"),
        (99, "YA100"),
        (100, "YA199"),
        (199, "YA199"),
        (200, "YA299"),
        (299, "YA299"),
        (300, "YA349"),
        (349, "YA349"),
        (350, "YA399"),
        (399, "YA399"),
        (400, "YA449"),
        (449, "YA449"),
        (450, "YA499"),
        (499, "YA499"),
        (500, "YA549"),
        (549, "YA549"),
        (550, "YA550"),
    ),
    "PY": ((299, None), (300, "P300"), (399, "P300"), (400, "P400")),
    "RY": ((99, None), (100, "RY100"), (199, "RY100"), (200, "RY200")),
    "REY": ((99, None), (100, "REY100"), (199, "REY100"), (200, "REY200")),
}
"""Amounts on each side of every bracket edge, with the one bracket each falls in (``None``: below them all)."""


@pytest.mark.parametrize(
    ("base", "amount", "bracket"),
    [(base, amount, bracket) for base, edges in BRACKET_EDGES.items() for amount, bracket in edges],
)
def test_bracket_edges_fall_in_exactly_one_bracket(base: str, amount: float, bracket: str | None) -> None:
    family = BRACKETS[base]
    derived = derive_stats({base: float(amount)}, "nfl", family)
    assert {derived[stat] for stat in family} <= {0.0, 1.0}
    assert [stat for stat in family if derived[stat] == 1.0] == ([bracket] if bracket else [])


def test_the_bracket_edges_cover_every_bracket() -> None:
    brackets = {stat: rule.base for stat, rule in FFL_DERIVATIONS.items() if isinstance(rule, Bracket)}
    assert brackets == {stat: base for base, family in BRACKETS.items() for stat in family}


def test_d_st_aliases_keep_espns_bracket_probabilities() -> None:
    projection = line(EAGLES, projected=True)  # ESPN projects PTSA and the PA brackets, but no DPTSA or DPA brackets
    assert "DPA14" not in projection
    derived = derive_stats(projection, "nfl", ("DPTSA", "DPA14", "HALFSK"))
    assert derived["DPTSA"] == projection["PTSA"]
    assert derived["DPA14"] == projection["PA14"]  # a probability (0.15), not a 0/1 indicator on PTSA
    assert derived["HALFSK"] == projection["HALFSK"]  # already there: twice the sacks, as ESPN counts them
    dpa_league = variant(points={"DPA14": 1.0}, drop=("PA14",))
    assert points(projection, dpa_league, position="D/ST") == pytest.approx(
        points(projection, variant(), position="D/ST")
    )


def test_derivation_tables_name_real_espn_stats() -> None:
    for game, rules in DERIVATIONS.items():
        schema = StatSchema.for_game(game)
        for stat, rule in rules.items():
            assert stat in schema, f"{game.value}: {stat}"
            names: list[str] = []
            for field in dataclasses.fields(rule):
                value = getattr(rule, field.name)
                values = value if isinstance(value, tuple) else (value,)
                names.extend(name for name in values if isinstance(name, str))
            assert rule_inputs(rule) == tuple(names), f"{game.value}: {stat}"
            for name in names:
                assert name in schema, f"{game.value}: {stat} reads {name}"
    assert FFL_DERIVATIONS["PY25"] == Every("PY", 25)
    assert FFL_DERIVATIONS["RY100"] == Bracket("RY", 100, 200)
    assert FBA_DERIVATIONS["REB"] == Total(("OREB", "DREB"))


# --- NBA: points and categories ---


def test_nba_points_league_scores_espns_default_items() -> None:
    settings = load_league_settings(FBA_POINTS)
    # 33 + 4 + 8 + 14 + 5 - 4 + 24 - 22 + 8 - 9 + 1
    assert points(NBA_LINE, settings) == pytest.approx(62.0)
    assert categories(NBA_LINE, settings) == {}
    split = {**NBA_LINE, "OREB": 1.0, "DREB": 4.0}
    del split["REB"]
    assert points(split, settings) == pytest.approx(62.0)  # REB from offensive + defensive rebounds


def test_category_league_passes_the_stats_through() -> None:
    settings = load_league_settings(FBA_9CAT)
    scorer = Scorer(settings)
    assert scorer.is_categories
    got = scorer.categories(NBA_LINE)
    assert list(got) == list(settings.categories)
    assert got == pytest.approx(
        {
            "PTS": 33.0,
            "BLK": 1.0,
            "STL": 2.0,
            "AST": 7.0,
            "REB": 5.0,
            "TO": 2.0,  # passed through as is; "lower wins" is the settings' is_reverse, for the consumer
            "3PM": 1.0,
            "FG%": 12 / 22,
            "FT%": 8 / 9,
        }
    )
    assert scorer.points(NBA_LINE) == 0.0
    assert scorer.categories({"PTS": 10.0, "FG%": 0.4})["FG%"] == 0.4  # a carried percentage is used as given
    assert scorer.categories({"PTS": 10.0})["FG%"] == 0.0  # no attempts, no percentage


def test_score_bundles_points_breakdown_and_categories(ppr: LeagueSettings) -> None:
    result = score(line(GIBBS), ppr, position="RB")
    assert isinstance(result, Score)
    assert result.points == pytest.approx(math.fsum(result.breakdown.values()))
    assert result.categories == {}
    nine_cat = score(NBA_LINE, load_league_settings(FBA_9CAT))
    assert nine_cat.points == 0.0
    assert nine_cat.breakdown == {}
    assert nine_cat.categories["PTS"] == 33.0


def test_scorer_reports_what_it_scores_and_can_score_many(ppr: LeagueSettings) -> None:
    scorer = Scorer(ppr)
    assert scorer.is_points and not scorer.is_categories
    assert scorer.stats[:2] == ("PY", "PTD")
    assert "YA449" in scorer.derivable and "PY" not in scorer.derivable
    got = scorer.many({GIBBS: line(GIBBS), EAGLES: line(EAGLES)}, {EAGLES: "D/ST"})
    assert got == pytest.approx({GIBBS: 17.7, EAGLES: 2.0})


# --- errors ---


@pytest.mark.parametrize("bad", ["12", None, True, float("nan"), float("inf")])
def test_unusable_values_raise(ppr: LeagueSettings, bad: object) -> None:
    with pytest.raises(ScoringError, match="REY"):
        Scorer(ppr).points(cast(dict[str, float], {"REY": bad}))


def test_unknown_position_label_raises(ppr: LeagueSettings) -> None:
    with pytest.raises(ScoringError, match="FLEX"):
        Scorer(ppr).points(line(GIBBS), position="FLEX")
    assert Scorer(ppr).points(line(GIBBS), position=99) == pytest.approx(17.7)  # an id without overrides: base points
