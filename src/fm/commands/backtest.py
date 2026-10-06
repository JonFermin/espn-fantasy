"""``fm backtest``: replay past weeks and print how well each projection source and your own lineups did (DESIGN
section 15, ROADMAP #33).

Three metrics, per projection source and for the lineups you actually set (the efficiency baseline to beat):

- projection MAE by position, in league points per player-week;
- lineup efficiency: actual points over the hindsight-optimal lineup's from the same roster;
- start/sit regret: the points left on the bench against that optimum.

The data is a replay fixture (``--fixtures DIR``, which reads ``DIR/<sport>/backtest.json``; no network, no state
database) or, without it, the weeks the state database holds for one configured league. The blend of the sources under
``data/blend_weights.toml`` (or ``--weights``) is evaluated as one more source. The library is
:mod:`fm.eval.backtest`; this module only loads, calls it and renders.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, NoReturn

import typer

from fm.config import ConfigError, Sport, load_config
from fm.eval.backtest import (
    FIXTURE_FILE,
    BacktestData,
    BacktestError,
    BacktestReport,
    LineupReport,
    load_fixture,
    load_store_data,
    run_backtest,
    with_blend,
)
from fm.model.projections import BlendWeights, BlendWeightsError
from fm.model.scoring import ScoringError
from fm.model.valuation import ValuationError, league_settings
from fm.render import Align, columns, points, warning_lines
from fm.store import SPORTS, Store

SportOption = Annotated[str, typer.Option("--sport", help="nfl or nba.")]
FixturesOption = Annotated[
    Path | None,
    typer.Option("--fixtures", help="Replay fixtures: reads DIR/<sport>/backtest.json instead of the state database."),
]
LeagueOption = Annotated[
    str | None,
    typer.Option("--league", "-l", help="League key from config.toml (default: the one league of the sport)."),
]
SourceOption = Annotated[
    list[str] | None,
    typer.Option("--source", "-s", help="Only these sources (repeatable; 'blend' is the blend). Default: all."),
]
WeightsOption = Annotated[
    Path | None, typer.Option("--weights", help="Blend weights file (default: data/blend_weights.toml).")
]
CommonOption = Annotated[
    bool, typer.Option("--common", help="Score every source's MAE on the player-weeks all of them project.")
]


def backtest(
    sport: SportOption = "nfl",
    fixtures: FixturesOption = None,
    league: LeagueOption = None,
    source: SourceOption = None,
    weights: WeightsOption = None,
    common: CommonOption = False,
) -> None:
    """Replay past weeks: projection MAE by position, lineup efficiency and start/sit regret per source.

    Includes the lineup efficiency of the lineups you actually set, the baseline any automation has to beat.
    """
    chosen = _sport(sport)
    try:
        data = _load(chosen, fixtures, league)
        data, notes = _with_blend(data, weights)
        report = run_backtest(data, sources=source, restrict_to_common=common)
    except (BacktestError, ConfigError, ValuationError, BlendWeightsError, ScoringError) as exc:
        _fail(str(exc))
    for line in (*render(report), *(f"note: {note}" for note in notes), *warning_lines(report.warnings, indent="")):
        typer.echo(line)


def _sport(value: str) -> Sport:
    for sport in SPORTS:
        if sport == value:
            return sport
    _fail(f"--sport must be one of {', '.join(SPORTS)}, got {value!r}")


def _load(sport: Sport, fixtures: Path | None, league_key: str | None) -> BacktestData:
    if fixtures is not None:
        directory = fixtures / sport
        if not (directory / FIXTURE_FILE).is_file():
            raise BacktestError(f"no {sport} backtest fixture: {directory / FIXTURE_FILE} does not exist")
        return load_fixture(directory)
    config = load_config()
    leagues = [candidate for candidate in config.leagues if candidate.sport == sport]
    if league_key is not None:
        leagues = [candidate for candidate in leagues if candidate.key == league_key]
    if len(leagues) != 1:
        wanted = f"league {league_key!r}" if league_key else f"a {sport} league"
        raise BacktestError(f"expected exactly one configured {sport} league for {wanted}, found {len(leagues)}")
    with Store.open() as store:
        row = store.leagues.by_key(leagues[0].key)
        if row is None:
            raise BacktestError(f"league {leagues[0].key!r} is not in the store; run fm sync")
        return load_store_data(store, row, league_settings(store, row))


def _with_blend(data: BacktestData, weights_path: Path | None) -> tuple[BacktestData, list[str]]:
    """``data`` plus the blend as one more source when there are at least two sources to blend."""
    if len(data.projections) < 2:
        return data, ["fewer than two projection sources; no blend evaluated"]
    weights = BlendWeights.load(weights_path)
    named = set(weights.sources(data.sport)) if data.sport in weights.sports else set()
    blendable = [name for name in data.projections if name in named]
    skipped = [name for name in data.projections if name not in named]
    notes = [f"not in the blend weights, left out of the blend: {', '.join(skipped)}"] if skipped else []
    if len(blendable) < 2:
        return data, [*notes, "fewer than two weighted sources; no blend evaluated"]
    return with_blend(data, weights, sources=blendable), notes


# --- rendering --------------------------------------------------------------------------------------------------------


def _percent(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.1%}"


def _lineup_row(label: str, report: LineupReport) -> list[str]:
    worst = report.worst_week
    return [
        label,
        points(report.points),
        points(report.optimal),
        _percent(report.efficiency),
        _percent(report.mean_weekly_efficiency),
        points(report.regret),
        points(report.regret_per_week),
        str(report.swaps),
        "-" if worst is None else f"wk {worst.period} ({points(worst.regret)})",
    ]


def render(report: BacktestReport) -> list[str]:
    """The three metrics as printable lines."""
    weeks = f"{report.periods[0]}-{report.periods[-1]}" if report.periods else "none"
    lines = [
        f"Backtest: {report.sport} {report.season}, periods {weeks} ({len(report.periods)} weeks), sources: "
        f"{', '.join(report.sources) or 'none'}",
        "",
        "Projection MAE by position (league points per player-week; samples in parentheses)"
        + (", scored on the player-weeks every source projects" if report.restricted_to_common else ""),
    ]
    names = list(report.sources)
    mae_rows: list[list[str]] = []
    for position in (*report.positions, None):
        cells = [position or "ALL"]
        for name in names:
            source = report.sources[name]
            error = source.overall if position is None else source.by_position.get(position)
            cells.append("-" if error is None else f"{error.mae:.2f} ({error.samples})")
        mae_rows.append(cells)
    mae_align: list[Align] = ["<", *(">" for _ in names)]
    lines.extend(columns([["position", *names], *mae_rows], align=mae_align))
    lines += [
        "",
        "Lineup efficiency (actual points / hindsight-optimal) and start/sit regret (points left on the bench)",
    ]
    header = ["lineup", "points", "optimal", "efficiency", "weekly mean", "regret", "per week", "swaps", "worst week"]
    rows = [_lineup_row("actual (yours)", report.baseline)]
    rows.extend(_lineup_row(f"{name} projections", source.lineups) for name, source in report.sources.items())
    lineup_align: list[Align] = ["<", ">", ">", ">", ">", ">", ">", ">", "<"]
    lines.extend(columns([header, *rows], align=lineup_align))
    lines += [
        "",
        f"Your lineup-efficiency baseline: {_percent(report.baseline.efficiency)} over {len(report.periods)} weeks "
        f"({points(report.baseline.regret)} points left on the bench)",
    ]
    return lines


def _fail(message: str) -> NoReturn:
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(1)


def register(root: typer.Typer) -> None:
    root.command("backtest")(backtest)
