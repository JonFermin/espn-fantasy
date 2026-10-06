"""``fm tune``: fit the projection blend weights and uncertainty on replayed weeks (DESIGN section 8.1, ROADMAP #39).

Loads the same replay data as ``fm backtest`` (``--fixtures DIR`` or the weeks the state database holds), fits
per-(source, position) weights and per-position coefficients of variation with :mod:`fm.eval.tune`, and prints the fit
with its leave-one-week-out MAE against equal weights. Nothing is written unless ``--write`` is given, and even then
only when the evidence is meaningful (enough weeks, a fitted position, held-out MAE no worse than equal weights);
``--force`` writes a tuning that is not. The weights file is ``data/blend_weights.toml`` unless ``--weights`` names
another.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, NoReturn

import typer

from fm.commands import backtest as backtest_command
from fm.config import ConfigError
from fm.eval.backtest import BacktestError
from fm.eval.tune import SportTuning, TuneConfig, TuneError, tune_sport, write_weights
from fm.model.projections import DEFAULT_WEIGHTS, BlendWeights, BlendWeightsError
from fm.model.scoring import ScoringError
from fm.model.valuation import ValuationError
from fm.render import Align, columns, warning_lines

SportOption = Annotated[str, typer.Option("--sport", help="nfl or nba.")]
FixturesOption = Annotated[
    Path | None,
    typer.Option("--fixtures", help="Replay fixtures: reads DIR/<sport>/backtest.json instead of the state database."),
]
LeagueOption = Annotated[
    str | None,
    typer.Option("--league", "-l", help="League key from config.toml (default: the one league of the sport)."),
]
WeightsOption = Annotated[
    Path | None, typer.Option("--weights", help="Blend weights file to start from and write (default: the committed).")
]
WriteOption = Annotated[bool, typer.Option("--write", help="Rewrite the weights file when the tuning is meaningful.")]
ForceOption = Annotated[bool, typer.Option("--force", help="With --write, write a tuning that is not meaningful.")]


def tune(
    sport: SportOption = "nfl",
    fixtures: FixturesOption = None,
    league: LeagueOption = None,
    weights: WeightsOption = None,
    write: WriteOption = False,
    force: ForceOption = False,
) -> None:
    """Fit blend weights and uncertainty on replayed weeks; report held-out MAE against equal weights."""
    chosen = backtest_command._sport(sport)  # pyright: ignore[reportPrivateUsage]
    path = weights if weights is not None else DEFAULT_WEIGHTS
    try:
        data = backtest_command._load(chosen, fixtures, league)  # pyright: ignore[reportPrivateUsage]
        tuning = tune_sport(data, BlendWeights.load(path), config=TuneConfig())
        outcome = write_weights(path, [tuning], force=force) if write else None
    except (BacktestError, ConfigError, ValuationError, BlendWeightsError, ScoringError, TuneError, OSError) as exc:
        _fail(str(exc))
    for line in (*render(tuning), *warning_lines(data.warnings, indent="")):
        typer.echo(line)
    if outcome is not None:
        if outcome.changed:
            typer.echo(f"wrote {outcome.path}")
        else:
            typer.echo(f"left {outcome.path} as it is: {outcome.skipped.get(tuning.sport, 'nothing to apply')}")


def _gain(value: float) -> str:
    return f"{value:+.3f}"


def render(tuning: SportTuning) -> list[str]:
    """The fit as printable lines: per position the equal-weight and best MAE and the weights, the held-out score,
    the refit uncertainty and the verdict."""
    periods = tuning.periods
    span = f"{periods[0]}-{periods[-1]}" if periods else "none"
    lines = [
        f"Blend tuning: {tuning.sport} {tuning.season}, periods {span} ({len(periods)} weeks), sources "
        f"{', '.join(tuning.sources)} (player-weeks every source projects)",
        "",
        "Weights by position (relative; 1.0 is an equal share), training MAE in league points",
    ]
    rows = [["position", "samples", "equal", "best", "weights", "note"]]
    for position, fit in tuning.positions.items():
        shown = (
            ", ".join(f"{source} {weight:.3f}" for source, weight in fit.weights.items())
            if fit.weights is not None
            else "(base weights)"
        )
        rows.append(
            [position, str(fit.samples), f"{fit.equal_mae:.3f}", f"{fit.best_mae:.3f}", shown, fit.reason or ""]
        )
    align: list[Align] = ["<", ">", ">", ">", "<", "<"]
    lines.extend(columns(rows, align=align))
    lines += ["", "Held-out MAE (leave one week out: the fit is redone without each week and scores it)"]
    held = [["position", "samples", "tuned", "current", "equal", "gain vs current"]]
    for error in (*tuning.held_out.values(), *([tuning.held_out_overall] if tuning.held_out_overall else [])):
        held.append(
            [
                error.position,
                str(error.samples),
                f"{error.tuned_mae:.3f}",
                f"{error.current_mae:.3f}",
                f"{error.equal_mae:.3f}",
                _gain(error.improvement),
            ]
        )
    lines.extend(columns(held, align=["<", ">", ">", ">", ">", ">"]))
    lines += ["", "Uncertainty (coefficient of variation of the blend's residuals, refit by position)"]
    if tuning.sd:
        lines.extend(
            columns(
                [["position", "samples", "sd", "cv"]]
                + [[p, str(e.samples), f"{e.sd:.2f}", f"{e.cv:.4f}"] for p, e in tuning.sd.items()],
                align=["<", ">", ">", ">"],
            )
        )
    else:
        lines.append("  no position has enough player-weeks; the file's values stand")
    verdict = "meaningful: --write applies it" if tuning.meaningful else f"not meaningful: {'; '.join(tuning.notes)}"
    return [*lines, "", f"Verdict: {verdict}"]


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)


def register(root: typer.Typer) -> None:
    root.command("tune")(tune)
