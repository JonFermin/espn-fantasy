"""Projection sources, the ESPN + Sleeper blend, blend weights and uncertainty (ROADMAP #15).

Real week-4 2026 data, offline: ESPN's projections from ``tests/fixtures/model/espn_ffl_pool_week4.json`` (stored the
way the sync job stores them), Sleeper's from ``tests/fixtures/sources/sleeper/projections_2026_4.json`` (served by a
mock transport to the real adapter) and the crosswalk built from ``tests/fixtures/sources/nflverse/db_playerids.csv``.
Five players are in both: Josh Allen, Patrick Mahomes, Derrick Henry, Jahmyr Gibbs and the Eagles D/ST.

Arithmetic runs against :data:`EQUAL_WEIGHTS`, the tests' own table: the committed ``data/blend_weights.toml`` belongs
to ROADMAP #39, which fits its values, so it is checked for structure only. The process-wide source registry is checked
for membership only, since the in-house baselines (ROADMAP #42, #43) add to it when imported.
"""

from __future__ import annotations

import json
import math
import tomllib
from collections.abc import Iterator
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from typing import Any, cast

import httpx
import numpy as np
import polars as pl
import pytest

from fm import paths
from fm.espn.ids import FFL, Game
from fm.espn.models import PlayersView
from fm.espn.settings import LeagueSettings, load_league_settings, parse_league_settings
from fm.model.ids import SLEEPER, Crosswalk, build_crosswalk
from fm.model.projections import (
    BLEND,
    DARKO,
    DEFAULT_WEIGHTS,
    ESPN,
    LINE_DERIVED,
    RECOMPUTED_RATIOS,
    SLEEPER_DEFENSE_STATS,
    SLEEPER_PROJECTION_EXCLUDED,
    SLEEPER_STATS,
    TEAM_DEFENSE_POSITION,
    BlendWeights,
    BlendWeightsError,
    DuplicateProjectionSourceError,
    PeriodBlend,
    Projection,
    ProjectionSourceRegistry,
    SdModel,
    SleeperLoader,
    UnknownProjectionSourceError,
    blend,
    blend_line,
    blend_period,
    fit_position_sd,
    load_stored,
    position_for,
    project,
    projection_source,
    projection_sources,
    register_source,
    sleeper_line,
    sleeper_rows,
    source_registry,
)
from fm.model.scoring import Scorer, ScoringError
from fm.sources.base import Fetched, FetchOptions, RateLimiter
from fm.sources.sleeper import SleeperSource, SleeperStatLine, parse_stat_lines
from fm.sports.base import StatSchema
from fm.store import PlayerRow, ProjectionRow, Store, utc_now

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
POOL = FIXTURES / "model" / "espn_ffl_pool_week4.json"
SLEEPER_WEEK4 = FIXTURES / "sources" / "sleeper" / "projections_2026_4.json"
PLAYER_IDS = FIXTURES / "sources" / "nflverse" / "db_playerids.csv"
FFL_PPR = FIXTURES / "espn" / "ffl_settings_ppr.json"
SEASON, WEEK = 2026, 4
ESPN_AS_OF = datetime(2026, 10, 4, 15, 0, tzinfo=UTC)
SLEEPER_AS_OF = datetime(2026, 10, 4, 16, 0, tzinfo=UTC)

ALLEN, MAHOMES, HENRY, GIBBS, EAGLES = 3918298, 3139477, 3043078, 4429795, -16021
CHASE, MCBRIDE, BUTKER = 4362628, 4361307, 3055899
JENNINGS, JUDKINS, SHRADER, WILSON, TEXANS = 3886598, 4685702, 4571557, 4887558, -16034
BOTH = (ALLEN, MAHOMES, HENRY, GIBBS, EAGLES)
ESPN_ONLY = (CHASE, MCBRIDE, BUTKER)
SLEEPER_ONLY = (JENNINGS, JUDKINS, SHRADER, WILSON, TEXANS)

EQUAL_WEIGHTS = """
[nfl.default]
espn = 1.0
sleeper = 1.0

[nfl.sd]
floor = 2.0
default = 0.55
QB = 0.40
WR = 0.60
K = 0.50
"D/ST" = 0.75

[nba.default]
espn = 1.0
darko = 1.0

[nba.sd]
floor = 3.0
default = 0.35
"""
"""Equal weights, as DESIGN starts the blend, and an uncertainty model the arithmetic below is written against."""


# --- fixtures ---


@cache
def pool() -> PlayersView:
    return PlayersView.model_validate(json.loads(POOL.read_text(encoding="utf-8")))


def espn_rows(*, projected: bool = True) -> list[ProjectionRow]:
    """ESPN's week-4 lines as the sync job stores them: source ``espn``, keyed by stat abbreviation."""
    schema = StatSchema.for_game(Game.FFL)
    rows: list[ProjectionRow] = []
    for entry in pool().players:
        line = entry.player.stat_entry(season=SEASON, scoring_period=WEEK, projected=projected)
        assert line is not None
        rows.append(
            ProjectionRow(
                sport="nfl",
                espn_id=entry.id,
                source=ESPN,
                kind="projected" if projected else "actual",
                season=SEASON,
                scoring_period_id=WEEK,
                stats=schema.from_espn(line.stats),
                as_of=ESPN_AS_OF,
            )
        )
    return rows


def players() -> list[PlayerRow]:
    """The ``players`` rows the sync job writes for the pool (positions decoded through the id maps)."""
    rows: list[PlayerRow] = []
    for entry in pool().players:
        player = entry.player
        assert player.default_position_id is not None
        rows.append(
            PlayerRow(
                sport="nfl",
                espn_id=player.id,
                full_name=player.full_name,
                default_position_id=player.default_position_id,
                position=FFL.position_label(player.default_position_id),
                pro_team_id=player.pro_team_id,
                as_of=ESPN_AS_OF,
            )
        )
    return rows


def positions() -> dict[int, str | None]:
    return {player.espn_id: player.position for player in players()}


@cache
def crosswalk() -> Crosswalk:
    return build_crosswalk(pl.read_csv(PLAYER_IDS, null_values=["NA", "NULL", ""]), as_of=ESPN_AS_OF)


def sleeper_lines(*, drop_placeholders: bool = True) -> list[SleeperStatLine]:
    return parse_stat_lines(SLEEPER_WEEK4.read_bytes(), drop_placeholders=drop_placeholders)


def sleeper_week4() -> list[ProjectionRow]:
    return list(sleeper_rows(sleeper_lines(), crosswalk(), as_of=SLEEPER_AS_OF).rows)


def by_id(rows: list[ProjectionRow] | tuple[ProjectionRow, ...]) -> dict[int, ProjectionRow]:
    return {row.espn_id: row for row in rows}


@pytest.fixture(scope="module")
def ppr() -> LeagueSettings:
    return load_league_settings(FFL_PPR)


def v1_registry() -> ProjectionSourceRegistry:
    """The v1 sources on a registry of the test's own, so nothing another module registers changes the result."""
    registry = ProjectionSourceRegistry()
    registry.register("nfl", ESPN, stored=True)
    registry.register("nfl", SLEEPER, loader=SleeperLoader())
    registry.register("nba", ESPN, stored=True)
    registry.register("nba", DARKO)
    return registry


@pytest.fixture(scope="module")
def equal() -> BlendWeights:
    return BlendWeights.parse(EQUAL_WEIGHTS, where="EQUAL_WEIGHTS", sources=v1_registry())


def every_n_league() -> LeagueSettings:
    """The PPR fixture league scoring a point per 25 passing, 10 rushing and 10 receiving yards instead of per yard."""
    view = json.loads(FFL_PPR.read_text(encoding="utf-8"))
    items: list[dict[str, Any]] = view["settings"]["scoringSettings"]["scoringItems"]
    per_yard = {FFL.stat_id(stat) for stat in ("PY", "RY", "REY")}
    items[:] = [item for item in items if item["statId"] not in per_yard]
    items.extend(
        {"statId": FFL.stat_id(stat), "points": 1.0, "pointsOverrides": {}, "isReverseItem": False}
        for stat in ("PY25", "RY10", "REY10")
    )
    return parse_league_settings(view)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    with Store.open(tmp_path / "state.db") as opened:
        yield opened


class SleeperServer:
    """The real Sleeper adapter over a mock transport serving the week-4 capture (or failing with ``status``)."""

    def __init__(self, cache_root: Path, *, status: int = 200) -> None:
        self.status = status
        self.requests: list[httpx.Request] = []
        self.source = SleeperSource(
            client=httpx.Client(transport=httpx.MockTransport(self.handle)),
            cache_root=cache_root,
            limiter=RateLimiter(0),
            clock=lambda: SLEEPER_AS_OF,
            sleep=lambda _: None,
        )

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        assert (request.url.host, request.url.path) == ("api.sleeper.com", f"/projections/nfl/{SEASON}/{WEEK}")
        if self.status != 200:
            return httpx.Response(self.status)
        return httpx.Response(200, content=SLEEPER_WEEK4.read_bytes(), headers={"content-type": "application/json"})


def nfl_registry(loader: SleeperLoader) -> ProjectionSourceRegistry:
    registry = ProjectionSourceRegistry()
    registry.register("nfl", ESPN, stored=True)
    registry.register("nfl", SLEEPER, loader=loader)
    return registry


# --- the registry ---


def test_v1_sources_are_registered_with_where_their_rows_come_from() -> None:
    # Membership, not the whole list: the baselines (ROADMAP #42, #43) register at import, and #24 re-registers darko
    # with the loader it builds.
    assert {ESPN, SLEEPER} <= set(source_registry.names("nfl"))
    assert {ESPN, DARKO} <= set(source_registry.names("fba"))
    espn = projection_source("nfl", ESPN)
    assert espn.stored and espn.loader is None and espn.loadable
    sleeper = projection_source(Game.FFL, SLEEPER)
    assert isinstance(sleeper.loader, SleeperLoader) and not sleeper.stored
    assert projection_source("nba", ESPN).stored and projection_source("nba", DARKO).name == DARKO
    assert {("nba", ESPN), ("nba", DARKO)} <= {source.key for source in projection_sources("nba")}
    assert len(projection_sources()) == len(projection_sources("nfl")) + len(projection_sources("nba"))


def test_registry_rejects_duplicates_and_malformed_registrations() -> None:
    registry = ProjectionSourceRegistry()
    registry.register("nfl", "baseline", label="Opportunity baseline", stored=True)
    with pytest.raises(DuplicateProjectionSourceError, match="nfl:baseline"):
        registry.register("ffl", "baseline")
    for bad in ("Baseline", "2nd", "base-line", ""):
        with pytest.raises(ValueError, match="lower-case identifier"):
            registry.register("nfl", bad)
    with pytest.raises(ValueError, match="blended output"):
        registry.register("nfl", BLEND)
    with pytest.raises(ValueError, match="not both"):
        registry.register("nba", "minutes", loader=SleeperLoader(), stored=True)
    with pytest.raises(TypeError, match="callable"):
        registry.register("nba", "minutes", loader=cast(Any, "not a loader"))
    with pytest.raises(ValueError, match="mlb"):
        registry.register("mlb", "minutes")


def test_registry_lookup_unregister_and_membership() -> None:
    registry = ProjectionSourceRegistry()
    registry.register("nba", ESPN, stored=True)
    registry.register("nba", DARKO)
    with pytest.raises(UnknownProjectionSourceError, match="registered: espn, darko"):
        registry.lookup("nba", "hashtag")
    assert ("fba", DARKO) in registry and ("nba", "hashtag") not in registry and "darko" not in registry
    assert ("mlb", DARKO) not in registry
    removed = registry.unregister("nba", DARKO)  # how ROADMAP #24 attaches a loader to darko
    assert removed.name == DARKO
    assert registry.register("nba", DARKO, loader=SleeperLoader()).loadable
    assert len(registry) == 2 and [source.name for source in registry] == [ESPN, DARKO]
    with pytest.raises(UnknownProjectionSourceError):
        registry.unregister("nfl", ESPN)
    registry.clear()
    assert len(registry) == 0 and registry.registered() == ()


def test_register_source_adds_to_the_process_wide_registry() -> None:
    source = register_source("nfl", "test_baseline", label="test", stored=True)
    try:
        assert projection_source("nfl", "test_baseline") is source
        weights = BlendWeights.parse("[nfl.default]\nespn = 1.0\ntest_baseline = 0.5")
        assert weights.weights("nfl")["test_baseline"] == 0.5
    finally:
        source_registry.unregister("nfl", "test_baseline")
    assert ("nfl", "test_baseline") not in source_registry


# --- Sleeper lines ---


def sleeper_line_of(player_id: str) -> SleeperStatLine:
    return next(line for line in sleeper_lines() if line.player_id == player_id)


def test_sleeper_line_maps_a_quarterback_and_drops_projected_first_downs() -> None:
    allen = sleeper_line_of("4984")
    projected = sleeper_line(allen.stats, position=allen.position, projected=True)
    assert projected == pytest.approx(
        {
            "PA": 30.22,
            "PC": 19.07,
            "INC": 11.15,
            "PY": 225.81,
            "PTD": 1.22,
            "2PC": 0.09,
            "INTT": 0.46,
            "SKD": 2.65,
            "RA": 8.34,
            "RY": 40.52,
            "RTD": 0.99,
            "2PR": 0.07,
            "FUM": 0.45,
            "FUML": 0.2,
            "GP": 1.0,
        }
    )
    # 22.6 passing first downs on 19.1 completions cannot be: projected first downs are left out, actuals kept
    assert allen.stats["pass_fd"] > allen.stats["pass_cmp"]
    assert sleeper_line(allen.stats, position="QB") == pytest.approx({**projected, "PFD": 22.58, "RFD": 4.05})
    assert SLEEPER_PROJECTION_EXCLUDED <= set(SLEEPER_STATS)


def test_sleeper_line_sums_kicking_distance_splits() -> None:
    shrader = sleeper_line_of("12185")
    assert sleeper_line(shrader.stats, position="K", projected=True) == pytest.approx(
        {
            "FG": 1.85,
            "FGA": 2.23,
            "FGY": 78.31,
            "FG40": 0.57,
            "FGM40": 0.13,
            "FG50P": 0.38,
            "FG50": 0.38,
            "FGM50P": 0.19,
            "FGM50": 0.19,
            "PAT": 2.55,
            "PATA": 2.62,
            "PATM": 0.06,
            "GP": 1.0,
            "FG0": 0.89,  # 20-29 plus 30-39 yard makes
            "FGM0": 0.06,
        }
    )


def test_sleeper_line_reads_only_defensive_keys_for_a_defense() -> None:
    eagles = sleeper_line_of("PHI")
    assert eagles.position == "DEF"
    assert sleeper_line(eagles.stats, position="DEF", projected=True) == pytest.approx(
        {
            "SK": 2.09,
            "INT": 0.68,
            "FR": 0.45,
            "FF": 0.64,
            "SF": 0.05,
            "BLKK": 0.05,
            "DEFRETTD": 0.18,  # interception plus fumble return touchdowns, as ESPN counts DEFRETTD
            "INTTD": 0.09,
            "FRTD": 0.09,
            "KR": 68.22,
            "PR": 81.86,  # def_pr_yd; the line's own pr_yd (13.64) is not ESPN's PR for a defense
            "PTSA": 23.0,
            "YA": 365.17,
            "STF": 4.0,
            "GP": 1.0,
        }
    )
    assert set(SLEEPER_DEFENSE_STATS).isdisjoint({"pr_yd", "pass_yd", "rec"})
    assert sleeper_line({"pass_int_td": 1.0, "pass_yd": 200.0}, position="QB") == {"PY": 200.0}


def test_sleeper_line_skips_unusable_values() -> None:
    line = sleeper_line({"pass_yd": math.nan, "rec": math.inf, "rec_yd": 55.0, "rush_yd": True})
    assert line == {"REY": 55.0}


def test_sleeper_rows_join_lines_to_espn_ids_through_the_crosswalk() -> None:
    converted = sleeper_rows(sleeper_lines(drop_placeholders=False), crosswalk(), as_of=SLEEPER_AS_OF)
    assert converted.unmapped == () and converted.warnings == ()  # the two ADP-only placeholders are skipped silently
    rows = by_id(converted.rows)
    assert set(rows) == {*BOTH, *SLEEPER_ONLY}
    assert all(
        (row.sport, row.source, row.kind, row.season, row.scoring_period_id, row.as_of)
        == ("nfl", SLEEPER, "projected", SEASON, WEEK, SLEEPER_AS_OF)
        for row in rows.values()
    )
    assert rows[ALLEN].stats["PY"] == 225.81
    assert rows[TEXANS].stats["PTSA"] > 0  # Sleeper's HOU defense is ESPN's -16034
    assert "PFD" not in rows[ALLEN].stats


def test_sleeper_rows_report_what_they_cannot_convert() -> None:
    def stat_line(player_id: str, **fields: object) -> SleeperStatLine:
        data: dict[str, object] = {"player_id": player_id, "season": SEASON, "week": WEEK, "category": "proj"}
        data.update(fields)
        data.setdefault("stats", {"rec": 2.0, "rec_fd": 1.0})
        return SleeperStatLine.model_validate(data)

    lines = [
        stat_line("4984", category="stat"),  # an actual: kind "actual", first downs kept
        stat_line("99999"),  # no ESPN id
        stat_line("9221", season_type="post"),  # a postseason week is not an ESPN scoring period
        stat_line("3198", category="rank"),
        stat_line("4046", stats={"adp_dd_ppr": 40.0}),  # placeholder
    ]
    converted = sleeper_rows(lines, crosswalk(), as_of=SLEEPER_AS_OF)
    assert [(row.espn_id, row.kind, dict(row.stats)) for row in converted.rows] == [
        (ALLEN, "actual", {"REC": 2.0, "REFD": 1.0})
    ]
    assert converted.unmapped == ("99999",)
    assert converted.warnings == (
        "sleeper: 1 of 2 lines belong to players with no ESPN id (99999)",
        "sleeper: skipped 1 lines outside the regular season (post)",
        "sleeper: skipped lines of unknown category rank",
    )


# --- blend weights ---


def test_the_committed_weights_file_resolves_through_paths_and_covers_both_sports() -> None:
    # Structure only: ROADMAP #39 fits the weights and the uncertainty and rewrites the file, so no value is pinned here
    committed = BlendWeights.load()
    assert DEFAULT_WEIGHTS == paths.data_file("blend_weights.toml")
    assert committed.path == DEFAULT_WEIGHTS
    raw = tomllib.loads(DEFAULT_WEIGHTS.read_text(encoding="utf-8"))
    for sport in ("nfl", "nba"):
        assert sport in committed.sports
        assert set(committed.sources(sport)) <= set(source_registry.names(sport))
        assert committed.weights(sport), f"{sport}: nothing is weighted by default"
        assert all(weight >= 0 for table in committed.tables[sport].values() for weight in table.values())
        assert isinstance(raw[sport].get("sd"), dict), f"no [{sport}.sd] table"


def test_position_tables_override_the_default_and_zero_drops_a_source() -> None:
    parsed = BlendWeights.parse(
        """
        [nfl.default]
        espn = 1.0
        sleeper = 1.0
        [nfl.QB]
        sleeper = 0
        [nfl.K]
        espn = 3
        ["nfl"."D/ST"]
        espn = 0.5
        """
    )
    assert parsed.weights("nfl", "QB") == {ESPN: 1.0}
    assert parsed.weights("nfl", "K") == {ESPN: 3.0, SLEEPER: 1.0}
    assert parsed.weights("nfl", "D/ST") == {ESPN: 0.5, SLEEPER: 1.0}
    assert parsed.weights("nfl", "WR") == parsed.weights("nfl", None) == {ESPN: 1.0, SLEEPER: 1.0}
    assert parsed.positions("nfl") == ("QB", "K", "D/ST")
    assert parsed.sources("nfl") == (ESPN, SLEEPER)
    assert parsed.sd_model("nfl") == SdModel()  # no [nfl.sd]: the default model
    with pytest.raises(BlendWeightsError, match="no blend weights for nba"):
        parsed.weights("nba")


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("[mlb.default]\nespn = 1", r"\[mlb\] is not a sport"),
        ("[nfl.default]\nespm = 1", "'espm' is not a registered nfl projection source"),
        ("[nfl.default]\ndarko = 1", "'darko' is not a registered nfl projection source"),
        ("[nfl.default]\nespn = -1", "finite number >= 0"),
        ("[nfl.default]\nespn = 'one'", "expected a number"),
        ("[nfl.default]\nespn = true", "expected a number"),
        ("[nfl.default]\nespn = inf", "finite number >= 0"),
        ("[nfl.default]\nespn = 1\n[nfl.FLEX]\nespn = 2", "'FLEX' is not a nfl position"),
        ("[nfl.QB]\nespn = 1", r"has no \[nfl.default\] table"),
        ("[nfl.default]\nespn = 0\nsleeper = 0", "weights every source 0"),
        ("[nfl.default]\nespn = 1\n[nfl.sd]\nWR = -0.2", "finite number >= 0"),
        ("[nfl.default]\nespn = 1\n[nfl.sd]\nPG = 0.3", "'PG' is not a nfl position"),
        ("[nfl]\ndefault = 3", "expected a table of source = weight entries"),
        ("nfl = 1", r"\[nfl\] should be a table"),
        ("[nfl.default\nespn = 1", "custom.toml"),
    ],
)
def test_bad_weight_files_are_rejected(text: str, message: str) -> None:
    with pytest.raises(BlendWeightsError, match=message):
        BlendWeights.parse(text, where="custom.toml")


def test_a_missing_weights_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(BlendWeightsError, match="not found"):
        BlendWeights.load(tmp_path / "blend_weights.toml")


def test_uniform_weights_cover_every_registered_source() -> None:
    uniform = BlendWeights.uniform()
    for sport in ("nfl", "nba"):
        assert uniform.weights(sport) == dict.fromkeys(source_registry.names(sport), 1.0)
    assert {ESPN, SLEEPER} <= set(uniform.weights("nfl"))
    isolated = BlendWeights.uniform(sources=v1_registry())
    assert isolated.weights("nfl") == {ESPN: 1.0, SLEEPER: 1.0}
    assert isolated.weights("nba") == {ESPN: 1.0, DARKO: 1.0}
    only_nfl = ProjectionSourceRegistry()
    only_nfl.register("nfl", ESPN, stored=True)
    assert BlendWeights.uniform(sources=only_nfl).sports == ("nfl",)


# --- the blend ---


def test_blend_line_is_a_weighted_mean_over_the_sources_that_carry_each_stat() -> None:
    lines = {ESPN: {"PTD": 1.6, "P300": 0.2}, SLEEPER: {"PTD": 1.2}, "other": {"PTD": 0.4}}
    assert blend_line(lines, {ESPN: 1.0, SLEEPER: 1.0}, sport="nfl") == pytest.approx({"PTD": 1.4, "P300": 0.2})
    weighted = blend_line(lines, {ESPN: 3.0, SLEEPER: 1.0, "other": 0.0}, sport="nfl")
    assert weighted == pytest.approx({"PTD": 1.5, "P300": 0.2})
    assert blend_line(lines, {SLEEPER: 2.0}, sport="nfl") == {"PTD": 1.2}  # a single weighted line, as it is
    assert blend_line(lines, {}, sport="nfl") == {}


def test_blend_line_gives_each_line_the_derived_stats_its_own_stats_give() -> None:
    espn = {"PY": 243.94, "PY25": 9.0, "RY": 19.98, "RY10": 1.0, "INTT": 0.655, "FUML": 0.157, "TT": 0.812}
    sleeper = {"PY": 246.22, "RY": 22.54, "INTT": 0.6, "FUML": 0.21}
    blended = blend_line({ESPN: {**espn, "P300": 0.2}, SLEEPER: sleeper}, {ESPN: 1.0, SLEEPER: 1.0}, sport="nfl")
    assert blended["PY25"] == pytest.approx(9.0)  # Sleeper's 246 yards are 9 x 25 as well
    assert blended["RY10"] == pytest.approx((1 + 2) / 2)  # its 22.5 rushing yards are 2 x 10: ESPN's 1 no longer stands
    assert blended["RY20"] == pytest.approx((0 + 1) / 2)  # ESPN omits a zero; its own 19.98 yards say 0 x 20
    assert blended["TT"] == pytest.approx((0.812 + 0.6 + 0.21) / 2)  # Sleeper's turnovers from its INTT and FUML
    assert blended["P300"] == 0.2  # a bracket is ESPN's probability, never derived per source
    defenses = {ESPN: {"PTSA": 19.4, "PA14": 0.15}, SLEEPER: {"PTSA": 23.0}}
    defense = blend_line(defenses, {ESPN: 1.0, SLEEPER: 1.0}, sport="nfl")
    assert defense["PA14"] == 0.15 and "DPA14" not in defense  # DPA14 reads PA14, so it is left to the scorer
    assert defense["DPTSA"] == pytest.approx(defense["PTSA"])  # an alias of a plain stat blends like it
    per_line = set(LINE_DERIVED[Game.FFL])
    assert {"PY25", "RY10", "REY10", "PC5", "TT", "FUM", "FUML", "FGM", "HALFSK", "DPTSA"} <= per_line
    assert per_line.isdisjoint({"P300", "RY100", "PA14", "DPA14", "YA449"})
    assert set(LINE_DERIVED[Game.FBA]) == {"REB", "FGMI", "FTMI", "3PMI"}
    assert RECOMPUTED_RATIOS[Game.FBA] == ("FG%", "FT%", "3PT%", "AFG%")


def test_a_kickers_missed_field_goals_blend_from_both_sources(ppr: LeagueSettings) -> None:
    # ESPN's line carries the total FGM (scored -1 in the PPR fixture); Sleeper's only the misses by distance
    espn = by_id(espn_rows())[BUTKER].stats
    sleeper = sleeper_line(sleeper_line_of("12185").stats, position="K", projected=True)
    assert "FGM" in espn and "FGM" not in sleeper
    blended = blend_line({ESPN: espn, SLEEPER: sleeper}, {ESPN: 1.0, SLEEPER: 1.0}, sport="nfl")
    missed = sleeper["FGM0"] + sleeper["FGM40"] + sleeper["FGM50P"]
    assert blended["FGM"] == pytest.approx((espn["FGM"] + missed) / 2)
    assert Scorer(ppr).breakdown(blended, position="K")["FGM"] == pytest.approx(-(espn["FGM"] + missed) / 2)


def test_an_empty_line_is_a_projection_of_zero_not_a_missing_one(ppr: LeagueSettings, equal: BlendWeights) -> None:
    line = {"PTD": 1.5, "RTD": 0.6, "GP": 1.0}
    both = {ESPN: 1.0, SLEEPER: 1.0}
    assert blend_line({ESPN: {}, SLEEPER: line}, both, sport="nfl") == pytest.approx(
        {"PTD": 0.75, "RTD": 0.3, "GP": 0.5}
    )
    no_game = blend_line({ESPN: {"GP": 0.0}, SLEEPER: line}, {ESPN: 3.0, SLEEPER: 1.0}, sport="nfl")
    assert no_game == pytest.approx({"GP": 0.25, "PTD": 0.375, "RTD": 0.15})  # all zeros is a zero projection too
    assert blend_line({ESPN: {}, SLEEPER: {}}, both, sport="nfl") == {}

    # ESPN's line for a player on bye is empty: with Sleeper projecting him anyway the blend is half of Sleeper's
    bye = by_id(espn_rows())[GIBBS].model_copy(update={"stats": {}})
    sleeper = by_id(sleeper_week4())[GIBBS]
    result = blend([bye, sleeper], weights=equal, positions=positions())
    (row,) = result.rows
    assert result.sources_for(GIBBS, SEASON, WEEK) == (ESPN, SLEEPER)
    assert row.stats["RY"] == pytest.approx(sleeper.stats["RY"] / 2)
    scorer = Scorer(ppr)
    assert scorer.points(row.stats, position="RB") == pytest.approx(scorer.points(sleeper.stats, position="RB") / 2)


def test_blended_percentages_follow_the_blended_makes_and_attempts() -> None:
    espn = {"FGM": 8.0, "FGA": 16.0, "FG%": 0.5, "REB": 10.0}
    darko = {"FGM": 3.0, "FGA": 4.0, "OREB": 2.0, "DREB": 6.0}
    blended = blend_line({ESPN: espn, DARKO: darko}, {ESPN: 1.0, DARKO: 1.0}, sport="nba")
    assert blended["FG%"] == pytest.approx(5.5 / 10.0)  # not ESPN's 0.5 alone, nor the mean of 0.5 and 0.75
    assert blended["REB"] == pytest.approx((10.0 + 8.0) / 2)  # DARKO's rebounds are its offensive plus defensive
    assert blended["FGMI"] == pytest.approx(blended["FGA"] - blended["FGM"])
    assert list(blended)[:4] == ["FGM", "FGA", "FG%", "REB"]


def test_blend_names_the_player_and_source_of_an_unusable_stat(equal: BlendWeights) -> None:
    allen = by_id(espn_rows())[ALLEN]
    broken = by_id(sleeper_week4())[ALLEN].model_copy(update={"stats": {"PY": math.nan}})
    with pytest.raises(ScoringError, match=rf"ESPN {ALLEN}, {SEASON} period {WEEK}: sleeper line: .*'PY'"):
        blend([allen, broken], weights=equal)


def test_blend_of_real_week4_espn_and_sleeper_projections(equal: BlendWeights) -> None:
    espn, sleeper = by_id(espn_rows()), by_id(sleeper_week4())
    result = blend([*espn.values(), *sleeper.values()], weights=equal, positions=positions())
    assert result.warnings == ()
    rows = by_id(result.rows)
    assert set(rows) == {*BOTH, *ESPN_ONLY, *SLEEPER_ONLY}
    assert all(
        (row.source, row.kind, row.season, row.scoring_period_id) == (BLEND, "projected", SEASON, WEEK)
        for row in rows.values()
    )

    allen = rows[ALLEN].stats
    assert allen["PY"] == pytest.approx((237.0758958 + 225.81) / 2)
    assert allen["PY"] == pytest.approx((espn[ALLEN].stats["PY"] + sleeper[ALLEN].stats["PY"]) / 2)
    assert allen["P300"] == espn[ALLEN].stats["P300"]  # only ESPN projects bonuses: kept at ESPN's value
    assert allen["PFD"] == espn[ALLEN].stats["PFD"]  # Sleeper's projected first downs are not in its line
    sleeper_turnovers = sleeper[ALLEN].stats["INTT"] + sleeper[ALLEN].stats["FUML"]
    assert allen["TT"] == pytest.approx((espn[ALLEN].stats["TT"] + sleeper_turnovers) / 2)
    assert rows[EAGLES].stats["PTSA"] == pytest.approx((espn[EAGLES].stats["PTSA"] + 23.0) / 2)
    assert rows[EAGLES].stats["PA14"] == espn[EAGLES].stats["PA14"]
    assert rows[CHASE].stats == espn[CHASE].stats  # one source: its line as it is
    assert rows[JENNINGS].stats == sleeper[JENNINGS].stats

    assert result.sources_for(ALLEN, SEASON, WEEK) == (ESPN, SLEEPER)
    assert result.sources_for(CHASE, SEASON, WEEK) == (ESPN,)
    assert result.sources_for(TEXANS, SEASON, WEEK) == (SLEEPER,)
    assert result.sources_for(999999, SEASON, WEEK) == ()
    assert rows[ALLEN].as_of == ESPN_AS_OF  # the older of the two inputs
    assert rows[JENNINGS].as_of == SLEEPER_AS_OF


def test_an_every_n_league_scores_the_blend_with_both_sources_yards(equal: BlendWeights) -> None:
    # A point per 25 passing, 10 rushing and 10 receiving yards. ESPN's lines carry those counts and Sleeper's do not,
    # so each source's count comes from its own yards (floor(yards / N), ESPN's form) and the blend averages the counts.
    espn, sleeper = by_id(espn_rows()), by_id(sleeper_week4())
    rows = by_id(blend([*espn.values(), *sleeper.values()], weights=equal, positions=positions()).rows)
    scorer = Scorer(every_n_league())
    for espn_id, position in ((MAHOMES, "QB"), (GIBBS, "RB")):
        got = scorer.breakdown(rows[espn_id].stats, position=position)
        for stat, base, size in (("PY25", "PY", 25), ("RY10", "RY", 10), ("REY10", "REY", 10)):
            counts = [math.floor(line.stats.get(base, 0.0) / size) for line in (espn[espn_id], sleeper[espn_id])]
            assert got.get(stat, 0.0) == pytest.approx(sum(counts) / 2), (espn_id, stat)
        # every scored stat is the mean of the two lines' points, but FTD, which only ESPN projects
        by_espn = scorer.breakdown(espn[espn_id].stats, position=position)
        by_sleeper = scorer.breakdown(sleeper[espn_id].stats, position=position)
        for stat in {*got, *by_espn, *by_sleeper}:
            expected = by_espn[stat] if stat == "FTD" else (by_espn.get(stat, 0.0) + by_sleeper.get(stat, 0.0)) / 2
            assert got.get(stat, 0.0) == pytest.approx(expected), (espn_id, stat)
    # with ESPN's counts beside the averaged yards these were 17.82, and 25.26 for Gibbs (ESPN's line alone: 25.30)
    assert scorer.points(rows[MAHOMES].stats, position="QB") == pytest.approx(18.323, abs=5e-4)
    assert scorer.points(rows[GIBBS].stats, position="RB") == pytest.approx(24.760, abs=5e-4)
    gibbs = rows[GIBBS].stats
    assert gibbs["RY10"] == 9.5  # ESPN's 101.4 rushing yards are 10 x 10 and Sleeper's 94.0 are 9: the mean count,
    assert math.floor(gibbs["RY"] / 10) == 9  # as for every other stat, not the count of the mean (97.7) yards

    more = sleeper[GIBBS].model_copy(update={"stats": {**sleeper[GIBBS].stats, "RY": sleeper[GIBBS].stats["RY"] + 30}})
    moved = by_id(blend([espn[GIBBS], more], weights=equal, positions=positions()).rows)[GIBBS]
    gained = scorer.points(moved.stats, position="RB") - scorer.points(gibbs, position="RB")
    assert gained == pytest.approx(1.5)  # Sleeper's 30 more yards are 3 more tens, half of them in the blend


def test_blend_weights_by_source_and_position() -> None:
    tuned = BlendWeights.parse("[nfl.default]\nespn = 3\nsleeper = 1\n[nfl.QB]\nsleeper = 0")
    espn, sleeper = by_id(espn_rows()), by_id(sleeper_week4())
    rows = by_id(blend([*espn.values(), *sleeper.values()], weights=tuned, positions=positions()).rows)
    assert rows[ALLEN].stats == espn[ALLEN].stats  # sleeper weighted 0 at QB
    assert rows[GIBBS].stats["RY"] == pytest.approx((3 * espn[GIBBS].stats["RY"] + sleeper[GIBBS].stats["RY"]) / 4)
    assert rows[JENNINGS].stats == sleeper[JENNINGS].stats  # a lone source is the blend at any weight


def test_a_defense_missing_from_positions_is_still_weighted_as_a_d_st() -> None:
    no_dst_sleeper = BlendWeights.parse('[nfl.default]\nespn = 1\nsleeper = 1\n[nfl."D/ST"]\nsleeper = 0')
    rows = by_id(blend([*espn_rows(), *sleeper_week4()], weights=no_dst_sleeper).rows)  # no positions at all
    assert rows[EAGLES].stats == by_id(espn_rows())[EAGLES].stats
    assert TEXANS not in rows  # its only source is weighted 0 for a D/ST
    assert position_for("nfl", EAGLES) == TEAM_DEFENSE_POSITION
    assert position_for("nfl", EAGLES, {EAGLES: "D/ST"}) == "D/ST"
    assert position_for("nfl", ALLEN) is None
    assert position_for("nba", EAGLES) is None


def test_blend_ignores_actuals_and_old_blends_and_reports_unweighted_sources(equal: BlendWeights) -> None:
    allen = by_id(espn_rows())[ALLEN]
    old_blend = allen.model_copy(update={"source": BLEND, "stats": {"PY": 1.0}})
    actual = by_id(espn_rows(projected=False))[ALLEN]
    stranger = allen.model_copy(update={"source": "fantasypros", "stats": {"PY": 999.0}})
    zeroed = BlendWeights.parse("[nfl.default]\nespn = 1\nsleeper = 1\n[nfl.QB]\nespn = 0\nsleeper = 0")
    result = blend([allen, old_blend, actual, stranger], weights=equal)
    assert [row.stats["PY"] for row in result.rows] == [allen.stats["PY"]]
    assert result.warnings == (
        "1 rows were actuals, not projections; they were not blended",
        "nfl: projection source 'fantasypros' has no entry in the blend weights; its rows were not blended",
    )
    nothing = blend([allen], weights=zeroed, positions={ALLEN: "QB"})
    assert nothing.rows == ()
    assert nothing.warnings == (
        "nfl: 1 players had rows only from sources weighted 0 at their position; no blended row",
    )
    assert blend([], weights=equal).rows == ()


def test_blend_takes_one_sport_at_a_time(equal: BlendWeights) -> None:
    nfl = by_id(espn_rows())[ALLEN]
    nba = nfl.model_copy(update={"sport": "nba", "espn_id": 3112335})
    with pytest.raises(ValueError, match="one sport at a time"):
        blend([nfl, nba], weights=equal)


# --- blending a stored period ---


def seed(store: Store) -> None:
    """What ``fm sync`` leaves behind: players, ESPN's lines (projected and actual) and the crosswalk."""
    store.players.upsert_many(players())
    store.projections.upsert_many([*espn_rows(), *espn_rows(projected=False)])
    crosswalk().save(store)


def test_blend_period_blends_stored_espn_with_sleeper_through_the_adapter(
    store: Store, tmp_path: Path, equal: BlendWeights
) -> None:
    seed(store)
    server = SleeperServer(tmp_path / "cache")
    registry = nfl_registry(SleeperLoader(server.source))
    result = blend_period(store, "nfl", SEASON, WEEK, weights=equal, sources=registry)
    assert isinstance(result, PeriodBlend)
    assert result.warnings == ()
    assert len(server.requests) == 1
    assert {row.espn_id for row in result.rows} == {*BOTH, *ESPN_ONLY, *SLEEPER_ONLY}
    assert result.saved == len(SLEEPER_ONLY) + len(BOTH) + len(result.rows)  # Sleeper's rows, then the blend

    espn_load, sleeper_load = result.load(ESPN), result.load(SLEEPER)
    assert espn_load is not None and sleeper_load is not None
    assert (espn_load.cached, espn_load.degraded, espn_load.as_of, len(espn_load.data)) == (True, False, ESPN_AS_OF, 8)
    assert (sleeper_load.cached, sleeper_load.stale, sleeper_load.degraded) == (False, False, False)
    assert sleeper_load.as_of == SLEEPER_AS_OF and len(sleeper_load.data) == 10

    stored_sleeper = store.projections.for_period("nfl", SEASON, WEEK, source=SLEEPER)
    assert by_id(stored_sleeper) == by_id(sleeper_week4())  # replayable inputs
    stored_blend = by_id(store.projections.for_period("nfl", SEASON, WEEK, source=BLEND))
    assert stored_blend == by_id(result.rows)
    assert stored_blend[ALLEN].stats["PY"] == pytest.approx((237.0758958 + 225.81) / 2)
    assert store.projections.get("nfl", ALLEN, ESPN, SEASON, WEEK, kind="actual") is not None  # untouched

    again = blend_period(store, "nfl", SEASON, WEEK, weights=equal, sources=registry)
    assert len(server.requests) == 1  # the adapter served its cache, as it does after a sync
    cached = again.load(SLEEPER)
    assert cached is not None and cached.cached
    assert by_id(again.rows) == by_id(result.rows)


def test_blend_period_runs_on_espn_alone_when_sleeper_breaks(store: Store, tmp_path: Path, equal: BlendWeights) -> None:
    seed(store)
    server = SleeperServer(tmp_path / "cache", status=500)
    registry = nfl_registry(SleeperLoader(server.source))
    result = blend_period(store, "nfl", SEASON, WEEK, weights=equal, sources=registry)
    sleeper_load = result.load(SLEEPER)
    assert sleeper_load is not None and sleeper_load.degraded and sleeper_load.data == ()
    assert any("HTTP 500" in warning for warning in result.warnings)
    rows = by_id(result.rows)
    assert set(rows) == {*BOTH, *ESPN_ONLY}
    assert rows[ALLEN].stats == by_id(espn_rows())[ALLEN].stats
    assert result.saved == len(result.rows)
    assert result.retired == ()  # nothing was blended for the period before


def test_blend_period_empties_blended_rows_no_source_projects_any_more(
    store: Store, tmp_path: Path, equal: BlendWeights
) -> None:
    seed(store)
    working = nfl_registry(SleeperLoader(SleeperServer(tmp_path / "up").source))
    first = blend_period(store, "nfl", SEASON, WEEK, weights=equal, sources=working)
    assert first.retired == () and {row.espn_id for row in first.rows} == {*BOTH, *ESPN_ONLY, *SLEEPER_ONLY}

    broken = nfl_registry(SleeperLoader(SleeperServer(tmp_path / "down", status=500).source))
    before = utc_now()
    second = blend_period(store, "nfl", SEASON, WEEK, weights=equal, sources=broken)
    assert set(second.retired) == set(SLEEPER_ONLY)  # Sleeper was their only source, and it failed this time
    assert second.warnings[-1] == (
        "nfl: 5 players with a blended row from an earlier run have no weighted source now; their blended rows were "
        "emptied"
    )
    assert second.saved == len(second.rows) + len(SLEEPER_ONLY)
    stored = by_id(store.projections.for_period("nfl", SEASON, WEEK, source=BLEND))
    assert set(stored) == {*BOTH, *ESPN_ONLY, *SLEEPER_ONLY}  # the repository cannot delete: emptied instead
    for espn_id in SLEEPER_ONLY:
        assert stored[espn_id].stats == {} and stored[espn_id].as_of >= before  # a projection of zero, as of now
    assert stored[ALLEN].stats == by_id(espn_rows())[ALLEN].stats  # ESPN's line alone this time

    third = blend_period(store, "nfl", SEASON, WEEK, weights=equal, sources=broken)
    assert third.retired == () and third.saved == len(third.rows)  # already empty: left alone, not reported again
    assert not any("emptied" in warning for warning in third.warnings)


def test_sleeper_loader_uses_the_crosswalk_saved_in_the_store(store: Store, tmp_path: Path) -> None:
    server = SleeperServer(tmp_path / "cache")
    options: FetchOptions = {}
    before_sync = SleeperLoader(server.source)(store, SEASON, WEEK, options)
    assert before_sync.data == ()
    assert before_sync.warnings == (
        "sleeper: 10 of 10 lines belong to players with no ESPN id (PHI, 7049, HOU, 4984, 9221 and 5 more)",
    )
    crosswalk().save(store)
    after_sync = SleeperLoader(server.source)(store, SEASON, WEEK, {"force": True})
    assert isinstance(after_sync, Fetched) and len(after_sync.data) == 10
    assert len(server.requests) == 2  # force skipped the cache


def test_blend_period_warns_about_sources_it_cannot_load(store: Store) -> None:
    nba = ProjectionSourceRegistry()
    nba.register("nba", ESPN, stored=True)
    nba.register("nba", DARKO)
    weights = BlendWeights.parse("[nba.default]\nespn = 1\ndarko = 1", sources=nba)
    result = blend_period(store, "nba", 2027, 1, weights=weights, sources=nba)
    assert result.rows == () and result.saved == 0
    assert result.warnings == (
        "espn: no nba projections stored for 2027 period 1; run fm sync",
        "nba: projection source 'darko' is weighted but has no loader; its rows were not blended",
    )
    empty = result.load(ESPN)
    assert empty is not None and empty.degraded and empty.cached


def test_blend_period_drops_rows_a_loader_returns_for_another_period(store: Store, equal: BlendWeights) -> None:
    def loader(
        store: Store, season: int, scoring_period: int, options: FetchOptions
    ) -> Fetched[tuple[ProjectionRow, ...]]:
        rows = tuple(row.model_copy(update={"source": SLEEPER}) for row in espn_rows())
        stray = rows[0].model_copy(update={"scoring_period_id": scoring_period + 1})
        return Fetched((*rows, stray), SLEEPER_AS_OF, SLEEPER, "projections", "test")

    registry = ProjectionSourceRegistry()
    registry.register("nfl", SLEEPER, loader=loader)
    result = blend_period(store, "nfl", SEASON, WEEK, weights=equal, sources=registry, save=False)
    assert len(result.rows) == 8
    assert result.warnings == ("sleeper: dropped 1 rows for another sport or period",)
    assert result.saved == 0
    assert store.projections.for_period("nfl", SEASON, WEEK, source=BLEND) == []


def test_load_stored_reads_one_source_with_its_oldest_as_of(store: Store) -> None:
    seed(store)
    loaded = load_stored(store, "ffl", ESPN, SEASON, WEEK)
    assert len(loaded.data) == 8 and all(row.kind == "projected" for row in loaded.data)
    assert (loaded.source, loaded.dataset, loaded.key, loaded.as_of) == (ESPN, "projections", "nfl_2026_4", ESPN_AS_OF)


# --- points and uncertainty ---


def test_projection_sd_by_position_with_a_floor(equal: BlendWeights) -> None:
    assert equal.projection_sd(20.0, sport="nfl", position="QB") == pytest.approx(8.0)
    assert equal.projection_sd(20.0, sport="nfl", position="WR") == pytest.approx(12.0)
    assert equal.projection_sd(20.0, sport="nfl", position="D/ST") == pytest.approx(15.0)
    assert equal.projection_sd(20.0, sport="nfl") == pytest.approx(11.0)  # the default coefficient
    assert equal.projection_sd(1.0, sport="nfl", position="QB") == 2.0  # the floor
    assert equal.projection_sd(-4.0, sport="nfl", position="K") == 2.0
    assert equal.projection_sd(40.0, sport="nba", position="C") == pytest.approx(14.0)
    assert equal.projection_sd(5.0, sport=Game.FBA) == 3.0


def test_project_scores_blended_rows_for_a_league_with_an_sd_each(ppr: LeagueSettings, equal: BlendWeights) -> None:
    result = blend([*espn_rows(), *sleeper_week4()], weights=equal)
    projected = {item.espn_id: item for item in project(result.rows, ppr, weights=equal, positions=positions())}
    chase = projected[CHASE]
    assert isinstance(chase, Projection)
    applied = pool().entry(CHASE).player.stat_entry(season=SEASON, scoring_period=WEEK, projected=True)
    assert applied is not None and chase.points == pytest.approx(applied.applied_total)  # ESPN-only: ESPN's points
    assert (chase.position, chase.source, chase.kind) == ("WR", BLEND, "projected")
    assert chase.sd == pytest.approx(0.6 * chase.points)
    assert projected[ALLEN].sd == pytest.approx(0.4 * projected[ALLEN].points)
    texans = projected[TEXANS]  # Sleeper-only defense, not in positions: scored and weighted as a D/ST
    assert texans.position == TEAM_DEFENSE_POSITION
    assert texans.sd == pytest.approx(max(2.0, 0.75 * texans.points))
    jennings = projected[JENNINGS]  # Sleeper-only, not in positions: the default coefficient
    assert jennings.position is None and jennings.sd == pytest.approx(0.55 * jennings.points)
    with pytest.raises(ValueError, match="is nba"):
        project([result.rows[0].model_copy(update={"sport": "nba"})], ppr, weights=equal)


def test_fit_position_sd_recovers_the_residual_spread(ppr: LeagueSettings) -> None:
    residual_yards = np.array([-60.0, -35.0, -20.0, -10.0, -5.0, 0.0, 5.0, 10.0, 25.0, 30.0] * 3)
    projected: list[ProjectionRow] = []
    actual: list[ProjectionRow] = []
    where: dict[int, str | None] = {}
    for espn_id, extra in enumerate(residual_yards, start=1):
        projected.append(stat_row(espn_id, "projected", {"REY": 80.0, "REC": 5.0}))  # 13 PPR points
        actual.append(stat_row(espn_id, "actual", {"REY": 80.0 + float(extra), "REC": 5.0}))
        where[espn_id] = "WR"
    for espn_id in range(100, 105):  # five quarterbacks: too few to estimate
        projected.append(stat_row(espn_id, "projected", {"PY": 250.0}))
        actual.append(stat_row(espn_id, "actual", {"PY": 300.0}))
        where[espn_id] = "QB"
    projected.append(stat_row(200, "projected", {}))  # a bye: no projection, no residual
    actual.append(stat_row(200, "actual", {"REY": 90.0}))
    where[200] = "WR"
    estimates = fit_position_sd(projected, actual, ppr, positions=where)
    assert set(estimates) == {"WR"}
    wr = estimates["WR"]
    assert wr.samples == 30
    assert wr.sd == pytest.approx(float(np.std(0.1 * residual_yards, ddof=1)))
    assert wr.mean_projected == pytest.approx(13.0)
    assert wr.mean_actual == pytest.approx(13.0 + 0.1 * float(residual_yards.mean()))
    assert wr.cv == pytest.approx(wr.sd / 13.0)
    assert set(fit_position_sd(projected, actual, ppr, positions=where, min_samples=5)) == {"WR", "QB"}
    model = SdModel(by_position={"QB": 0.4}).updated(estimates)
    assert model.cv("WR") == pytest.approx(wr.cv) and model.cv("QB") == 0.4 and model.cv("TE") == model.default


def stat_row(espn_id: int, kind: str, stats: dict[str, float]) -> ProjectionRow:
    return ProjectionRow(
        sport="nfl",
        espn_id=espn_id,
        source=ESPN,
        kind="projected" if kind == "projected" else "actual",
        season=SEASON,
        scoring_period_id=WEEK,
        stats=stats,
        as_of=ESPN_AS_OF,
    )
