"""Blend weight tuning (ROADMAP #39): the fitter, its held-out guard, and the weights file it writes.

Two data sets. The shipped four-week replay fixture is the honest toy: too little to move any weight, so the tuning
must fall back to the base weights and say it is not meaningful. A synthetic league (built here, written to the
fixture format and read back by :func:`fm.eval.backtest.load_fixture`) has one source clearly better at running backs
and the other at receivers, so the fit has something real to find: its held-out MAE must beat equal weights'.
Everything is offline and writes only under ``tmp_path``; the committed ``data/blend_weights.toml`` is read, never
written.
"""

from __future__ import annotations

import json
import random
import shutil
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from fm.cli import app
from fm.eval.backtest import BacktestData, load_fixture
from fm.eval.tune import (
    HeldOutError,
    SportTuning,
    TuneConfig,
    TuneError,
    candidate_grid,
    fit_sd,
    header_of,
    render_blend_weights,
    tune_sport,
    write_weights,
)
from fm.model.projections import DEFAULT_WEIGHTS, BlendWeights, BlendWeightsError, SdEstimate

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "backtest"
NFL_FIXTURE = FIXTURES / "nfl"
runner = CliRunner()

WEEKS = 10
RBS = range(101, 109)
WRS = range(201, 209)
BENCH, RB_SLOT, WR_SLOT = 20, 2, 4
SLOTS = {"RB": [2, 3, 23, 20, 21], "WR": [3, 4, 5, 23, 20, 21]}
# half-width of each source's error (uniform) in points per position: espn is sharp on RB, sleeper on WR
NOISE = {("espn", "RB"): 0.4, ("sleeper", "RB"): 5.0, ("espn", "WR"): 5.0, ("sleeper", "WR"): 0.4}


def write_synthetic(directory: Path, *, weeks: int = WEEKS, seed: int = 7) -> Path:
    """A replay fixture in the shipped format: 8 RBs and 8 WRs, every stat a reception (one point in the fixture
    league's PPR scoring), two sources whose errors are uniform noise of the half-widths in :data:`NOISE`."""
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(NFL_FIXTURE / "settings.json", directory / "settings.json")
    rng = random.Random(seed)
    players = [
        {"espn_id": espn_id, "name": f"{position} {espn_id}", "position": position, "eligible_slots": SLOTS[position]}
        for position, ids in (("RB", RBS), ("WR", WRS))
        for espn_id in ids
    ]
    base = {espn_id: rng.uniform(8, 14) for espn_id in (*RBS, *WRS)}
    rows: list[dict[str, Any]] = []
    for period in range(1, weeks + 1):
        actuals = {espn_id: round(base[espn_id] + rng.uniform(-3, 3), 2) for espn_id in base}
        projections: dict[str, dict[str, dict[str, float]]] = {"espn": {}, "sleeper": {}}
        for espn_id, actual in actuals.items():
            position = "RB" if espn_id in RBS else "WR"
            for source in projections:
                miss = rng.uniform(-NOISE[(source, position)], NOISE[(source, position)])
                projections[source][str(espn_id)] = {"REC": round(max(actual + miss, 0.5), 2)}
        rows.append(
            {
                "period": period,
                "lineup": {str(RBS[0]): RB_SLOT, str(WRS[0]): WR_SLOT, str(RBS[1]): BENCH},
                "projections": projections,
                "actuals": {str(espn_id): {"REC": actual} for espn_id, actual in actuals.items()},
            }
        )
    document = {
        "format": 1,
        "sport": "nfl",
        "season": 2026,
        "settings": "settings.json",
        "as_of": "2026-09-01T12:00:00Z",
        "players": players,
        "weeks": rows,
    }
    (directory / "backtest.json").write_text(json.dumps(document), encoding="utf-8")
    return directory


@pytest.fixture(scope="module")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> BacktestData:
    return load_fixture(write_synthetic(tmp_path_factory.mktemp("synthetic")))


@pytest.fixture(scope="module")
def base() -> BlendWeights:
    return BlendWeights.load()


@pytest.fixture(scope="module")
def tuned(synthetic: BacktestData, base: BlendWeights) -> SportTuning:
    return tune_sport(synthetic, base)


@pytest.fixture(scope="module")
def toy(base: BlendWeights) -> SportTuning:
    return tune_sport(load_fixture(NFL_FIXTURE), base)


# --- the candidate grid -----------------------------------------------------------------------------------------------


def test_grid_starts_with_equal_and_covers_the_simplex() -> None:
    grid = candidate_grid(3, 4, 1000)
    assert grid[0] == pytest.approx((1 / 3, 1 / 3, 1 / 3))
    assert len(grid) == 15 + 1  # C(6, 2) compositions of 4 into 3 parts, none of them the 1/3 split, plus equal
    assert all(sum(shares) == pytest.approx(1.0) for shares in grid)
    assert len(set(grid)) == len(grid)
    assert (1.0, 0.0, 0.0) in grid


def test_grid_does_not_repeat_the_equal_split_and_coarsens_to_fit() -> None:
    two = candidate_grid(2, 10, 1000)
    assert two.count((0.5, 0.5)) == 1
    assert len(two) == 11
    coarse = candidate_grid(5, 10, 100)
    assert len(coarse) <= 101  # steps lowered until the grid fits (the equal split rides along)
    assert candidate_grid(1, 10, 100) == ((1.0,),)


# --- the fit on a league with something to find -----------------------------------------------------------------------


def test_the_better_source_gets_more_weight_at_each_position(tuned: SportTuning) -> None:
    rb, wr = tuned.positions["RB"], tuned.positions["WR"]
    assert rb.weights is not None and wr.weights is not None
    assert rb.weights["espn"] > rb.weights["sleeper"]
    assert wr.weights["sleeper"] > wr.weights["espn"]
    assert rb.best_mae < rb.equal_mae
    assert wr.best_mae < wr.equal_mae
    assert tuned.fitted_positions == ("RB", "WR")


def test_weights_are_scaled_to_the_base_mass_and_shrunk_toward_equal(tuned: SportTuning) -> None:
    rb = tuned.positions["RB"]
    assert rb.weights is not None
    assert sum(rb.weights.values()) == pytest.approx(2.0, abs=0.01)  # two sources at 1.0 in the file
    # n / (n + shrinkage) of the way to the best grid point: 80 player-weeks, so 80% (never all the way to 2 / 0)
    assert 1.0 < rb.weights["espn"] < 2.0
    assert 0.0 < rb.weights["sleeper"] < 1.0


def test_held_out_mae_is_no_worse_than_equal_weights(tuned: SportTuning) -> None:
    overall = tuned.held_out_overall
    assert overall is not None
    assert overall.tuned_mae <= overall.equal_mae
    assert overall.improvement > 0.2  # a real gain, not a tie
    for position in ("RB", "WR"):
        held = tuned.held_out[position]
        assert isinstance(held, HeldOutError)
        assert held.samples == WEEKS * len(RBS)
        assert held.tuned_mae <= held.equal_mae
    assert tuned.meaningful
    assert tuned.notes == ()


def test_the_tuned_blend_beats_equal_weights_on_the_backtest(synthetic: BacktestData, tuned: SportTuning) -> None:
    from fm.eval.backtest import run_backtest, with_blend

    equal = BlendWeights.uniform()
    data = with_blend(synthetic, equal, name="equal")
    data = with_blend(data, tuned.weights, name="tuned")
    report = run_backtest(data, sources=["equal", "tuned"], restrict_to_common=True)
    for position in ("RB", "WR"):
        assert report.source("tuned").by_position[position].mae < report.source("equal").by_position[position].mae
    assert report.source("tuned").mae < report.source("equal").mae


def test_uncertainty_is_refit_from_the_residuals(synthetic: BacktestData, tuned: SportTuning) -> None:
    assert set(tuned.sd) == {"RB", "WR"}
    model = tuned.weights.sd_model("nfl")
    for position, estimate in tuned.sd.items():
        assert isinstance(estimate, SdEstimate)
        assert estimate.samples == WEEKS * len(RBS)
        assert model.cv(position) == pytest.approx(estimate.cv, abs=1e-4)
    assert model.cv("QB") == 0.40  # a position without a fit keeps the file's value
    assert model.floor == 2.0
    direct = fit_sd(synthetic, tuned.weights, sources=tuned.sources)
    assert direct["RB"].cv == pytest.approx(tuned.sd["RB"].cv)


def test_fit_is_deterministic(synthetic: BacktestData, base: BlendWeights, tuned: SportTuning) -> None:
    again = tune_sport(synthetic, base)
    assert again.weights.tables == tuned.weights.tables
    assert again.held_out_overall == tuned.held_out_overall


# --- falling back, and the shipped toy fixture ------------------------------------------------------------------------


def test_a_position_with_too_few_samples_keeps_the_base_weights(synthetic: BacktestData, base: BlendWeights) -> None:
    result = tune_sport(synthetic, base, config=TuneConfig(min_samples=WEEKS * len(RBS) + 1))
    assert result.fitted_positions == ()
    assert all(fit.weights is None and fit.reason for fit in result.positions.values())
    assert result.weights.tables == base.tables
    assert not result.meaningful
    assert result.held_out_overall is not None
    assert result.held_out_overall.tuned_mae == pytest.approx(result.held_out_overall.equal_mae)


def test_on_the_shipped_fixture_nothing_moves_and_held_out_equals_equal(toy: SportTuning, base: BlendWeights) -> None:
    assert toy.periods == (1, 2, 3, 4)
    assert toy.sources == ("espn", "sleeper")
    assert toy.fitted_positions == ()  # 4 to 19 player-weeks per position, under the 30 a position needs
    assert toy.weights.tables == base.tables
    assert toy.sd == {}
    overall = toy.held_out_overall
    assert overall is not None
    assert overall.tuned_mae <= overall.equal_mae
    for held in toy.held_out.values():
        assert held.tuned_mae <= held.equal_mae
    assert not toy.meaningful
    assert any("fewer than 6" in note for note in toy.notes)


def test_a_toy_fit_forced_through_a_low_bar_is_reported_honestly(base: BlendWeights) -> None:
    """With the bars lowered the fixture's few player-weeks do move weights, and the held-out score is what says that
    is noise: the tuning is not meaningful whenever it is worse than equal weights."""
    loose = TuneConfig(min_samples=4, min_weeks=2)
    result = tune_sport(load_fixture(NFL_FIXTURE), base, config=loose)
    assert result.fitted_positions  # it did move
    overall = result.held_out_overall
    assert overall is not None
    assert result.meaningful == (overall.tuned_mae <= overall.equal_mae)


def test_the_blend_sources_must_exist_and_be_at_least_two(synthetic: BacktestData, base: BlendWeights) -> None:
    with pytest.raises(TuneError, match="at least two"):
        tune_sport(synthetic, base, sources=["espn"])
    with pytest.raises(TuneError, match="no such projection source: darko"):
        tune_sport(synthetic, base, sources=["espn", "darko"])
    with pytest.raises(TuneError, match="at least two"):
        tune_sport(synthetic.without_source("sleeper"), base)


def test_a_single_week_has_no_held_out_score(synthetic: BacktestData, base: BlendWeights) -> None:
    result = tune_sport(synthetic.restrict_to([1]), base)
    assert result.held_out == {}
    assert result.held_out_overall is None
    assert not result.meaningful


def test_config_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="steps"):
        TuneConfig(steps=0)
    with pytest.raises(ValueError, match="min_samples"):
        TuneConfig(min_samples=0)


# --- the weights file -------------------------------------------------------------------------------------------------


def copy_weights(tmp_path: Path) -> Path:
    target = tmp_path / "blend_weights.toml"
    shutil.copy(DEFAULT_WEIGHTS, target)
    return target


def test_the_committed_file_survives_a_render_round_trip(tmp_path: Path) -> None:
    original = BlendWeights.load()
    text = DEFAULT_WEIGHTS.read_text(encoding="utf-8")
    rendered = render_blend_weights(original, header=header_of(text))
    again = BlendWeights.parse(rendered)
    assert again.tables == original.tables
    assert again.sd_models == original.sd_models
    assert rendered.startswith(header_of(text))
    assert "ROADMAP #39 owns this file" in rendered
    assert "[nfl.sd]\nfloor = 2.0\ndefault = 0.55\nQB = 0.4\n" in rendered
    assert '"D/ST" = 0.75' in rendered
    # a second pass changes nothing: the renderer is stable
    assert render_blend_weights(again, header=header_of(rendered)) == rendered


def test_tuned_weights_round_trip_through_the_file(tmp_path: Path, tuned: SportTuning) -> None:
    path = copy_weights(tmp_path)
    outcome = write_weights(path, [tuned])
    assert outcome.applied == ("nfl",)
    assert outcome.changed
    assert outcome.skipped == {}
    loaded = BlendWeights.load(path)
    assert loaded.tables == tuned.weights.tables
    assert loaded.sd_models == tuned.weights.sd_models
    assert outcome.weights.tables == loaded.tables
    assert loaded.weights("nfl", "RB") == tuned.positions["RB"].weights
    assert loaded.weights("nfl", "WR") == tuned.positions["WR"].weights
    assert loaded.weights("nfl", "QB") == {"espn": 1.0, "sleeper": 1.0}  # untouched positions use the default table
    assert loaded.weights("nba") == {"espn": 1.0, "darko": 1.0}  # the other sport carries over
    assert loaded.sd_model("nfl").cv("RB") == tuned.weights.sd_model("nfl").cv("RB")
    assert loaded.sd_model("nfl").cv("TE") == 0.65
    text = path.read_text(encoding="utf-8")
    assert text.startswith(header_of(DEFAULT_WEIGHTS.read_text(encoding="utf-8")))
    assert "[nfl.RB]" in text
    assert "[nfl.WR]" in text


def test_retuning_a_written_file_is_idempotent(tmp_path: Path, synthetic: BacktestData, tuned: SportTuning) -> None:
    path = copy_weights(tmp_path)
    write_weights(path, [tuned])
    first = path.read_text(encoding="utf-8")
    again = tune_sport(synthetic, BlendWeights.load(path))
    write_weights(path, [again])
    loaded = BlendWeights.load(path)
    assert loaded.weights("nfl", "RB").keys() == {"espn", "sleeper"}
    assert loaded.weights("nfl", "RB")["espn"] > loaded.weights("nfl", "RB")["sleeper"]
    assert len(path.read_text(encoding="utf-8").splitlines()) == len(first.splitlines())  # same tables, no growth


def test_a_tuning_that_is_not_meaningful_leaves_the_file_untouched(tmp_path: Path, toy: SportTuning) -> None:
    path = copy_weights(tmp_path)
    before = path.read_bytes()
    outcome = write_weights(path, [toy])
    assert not outcome.changed
    assert "nfl" in outcome.skipped
    assert "fewer than 6" in outcome.skipped["nfl"]
    assert path.read_bytes() == before


def test_force_writes_a_tuning_that_is_not_meaningful(tmp_path: Path, toy: SportTuning) -> None:
    path = copy_weights(tmp_path)
    outcome = write_weights(path, [toy], force=True)
    assert outcome.applied == ("nfl",)
    assert BlendWeights.load(path).tables == BlendWeights.load().tables  # nothing fitted, so the same tables


def test_the_committed_weights_still_load_and_were_not_rewritten_by_these_tests() -> None:
    committed = BlendWeights.load()
    assert committed.weights("nfl") == {"espn": 1.0, "sleeper": 1.0}
    assert committed.positions("nfl") == ()  # still equal weights: the toy fixture does not justify moving them
    with pytest.raises(BlendWeightsError):
        BlendWeights.load(Path("does-not-exist.toml"))


# --- the command ------------------------------------------------------------------------------------------------------


def test_the_command_reports_the_toy_fit_and_writes_nothing(tmp_path: Path) -> None:
    path = copy_weights(tmp_path)
    before = path.read_bytes()
    result = runner.invoke(
        app, ["tune", "--sport", "nfl", "--fixtures", str(FIXTURES), "--weights", str(path), "--write"]
    )
    assert result.exit_code == 0, result.output
    assert "Blend tuning: nfl 2026" in result.output
    assert "Held-out MAE" in result.output
    assert "not meaningful" in result.output
    assert "left" in result.output
    assert path.read_bytes() == before


def test_the_command_writes_a_meaningful_tuning_to_the_named_file(tmp_path: Path) -> None:
    fixtures = tmp_path / "fixtures"
    write_synthetic(fixtures / "nfl")
    path = copy_weights(tmp_path)
    result = runner.invoke(
        app, ["tune", "--sport", "nfl", "--fixtures", str(fixtures), "--weights", str(path), "--write"]
    )
    assert result.exit_code == 0, result.output
    assert "Verdict: meaningful" in result.output
    assert f"wrote {path}" in result.output
    assert BlendWeights.load(path).weights("nfl", "RB")["espn"] > 1.0


def test_the_command_fails_cleanly_without_a_fixture(tmp_path: Path) -> None:
    result = runner.invoke(app, ["tune", "--sport", "nba", "--fixtures", str(tmp_path)])
    assert result.exit_code == 1
    assert "no nba backtest fixture" in result.output
