"""Blend weight tuning: fit per-(source, position) weights and per-position uncertainty on replayed weeks (DESIGN
sections 8.1 and 15, ROADMAP #39).

``data/blend_weights.toml`` starts with equal weights. This module fits them from the backtest library
(:mod:`fm.eval.backtest`) and rewrites the file, with held-out weeks as the guard against overfitting (DESIGN section
16, "Overfitting blend weights").

**The fit** (:func:`tune_sport`). Weights only matter where two or more sources project the same player-week, so the fit
scores the player-weeks *every* tuned source projects (a non-zero line) and nothing else. That sample is the same for
every candidate, so a candidate cannot look better by dropping the cases a source gets wrong. Candidates are the equal
split plus a grid over the simplex of the sources (steps of ``1 / steps``, coarsened for many sources to stay under
:attr:`TuneConfig.max_candidates`). Each candidate is blended with :func:`fm.eval.backtest.with_blend` and scored with
:func:`fm.eval.backtest.run_backtest`, week by week, so the projection MAE by position is in the league's own points.
Then, per position:

1. A position with fewer than :attr:`TuneConfig.min_samples` player-weeks, or where the best candidate beats the equal
   split by less than :attr:`TuneConfig.min_improvement` (relative MAE), keeps what the base weights say for it (the
   ``default`` table and any override already in the file): too little evidence to move.
2. Otherwise the best candidate is **shrunk toward equal weights** by ``n / (n + shrinkage)`` for ``n`` player-weeks
   (ridge-style: a thin sample moves the weights a little, a thick one a lot). MAE is convex in the weights, so the
   shrunk weights cannot be worse on the training sample than the mix of the best and the equal split.

**Held out.** The fit is judged by leave-one-week-out: for each week, the whole fit (grid choice, shrinkage, fallback)
is redone without it and the resulting weights score that week alone (:attr:`SportTuning.held_out`). The comparison
is what is in the weights file now, the production weights (:attr:`HeldOutError.current_mae`): the same weeks blended
under ``base``, its position overrides included, since a position the tuning does not fit keeps those tables and so
a pure equal split would not be what the tuned run is replacing. The pure equal split's MAE rides along as
:attr:`HeldOutError.equal_mae`. A tuning is :attr:`SportTuning.meaningful`, and :func:`write_weights` applies it, only
with at least :attr:`TuneConfig.min_weeks` weeks, one fitted position and a held-out MAE no worse than the current
weights' (it beats what is in the file on held-out weeks); the four-week hand-built fixture is not enough, so the
committed file stays as it is until real history is replayed.

**Scale.** Weights in the file are relative, and a source the tuning leaves out keeps its own weights, so fitted weights
are scaled to the sum of the tuned sources' default weights (two sources at 1.0 become, say, 1.4 and 0.6: 1.0 is an
equal share) and rounded to three decimals; what is evaluated is what is written.

**Uncertainty.** :func:`fit_sd` refits the coefficients of variation of ``[<sport>.sd]`` with
:func:`fm.model.projections.fit_position_sd` on the blend under the tuned weights over every replayed week (a position
needs :attr:`TuneConfig.min_sd_samples` player-weeks; the rest keep their value).

:func:`write_weights` rewrites the file (:func:`render_blend_weights` keeps the header comment, re-renders every table;
``BlendWeights.load`` parses the result). The library does the fitting; ``fm tune`` loads the data and renders it.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Final

from fm.config import Sport
from fm.espn.ids import ids_for
from fm.eval.backtest import (
    UNKNOWN_POSITION,
    BacktestData,
    blend_source,
    run_backtest,
    with_blend,
)
from fm.model.projections import (
    ESPN,
    MIN_SD_SAMPLES,
    BlendWeights,
    SdEstimate,
    SdModel,
    fit_position_sd,
)
from fm.store import ProjectionRow, utc_now

CANDIDATE: Final = "candidate"
"""Source name a candidate blend is evaluated under."""
WEIGHT_DECIMALS: Final = 3
CV_DECIMALS: Final = 4
TIE: Final = 1e-9
"""MAE differences below this are ties; ties go to the candidate nearest the equal split."""

type Cell = tuple[float, int]
"""(sum of absolute errors, player-weeks) of one position in one week."""
type WeekCells = dict[int, dict[str, Cell]]
"""period -> position -> :data:`Cell`."""


class TuneError(ValueError):
    """The data cannot be tuned as given: fewer than two sources, a source the data lacks, no common player-weeks."""


@dataclass(frozen=True, slots=True)
class TuneConfig:
    """Knobs of the fit; the defaults suit a season of weekly NFL history."""

    steps: int = 10
    """Grid resolution: weights are multiples of ``1 / steps``."""
    max_candidates: int = 1000
    """Upper bound on grid size; ``steps`` is lowered until the grid fits."""
    min_samples: int = 30
    """Player-weeks a position needs before its weights move off the base."""
    shrinkage: float = 20.0
    """Pseudo-samples of equal weights: the fit is ``n / (n + shrinkage)`` of the way from equal to the best."""
    min_improvement: float = 0.01
    """Relative MAE gain over equal weights a position needs to move off the base."""
    min_weeks: int = 6
    """Weeks a tuning needs to be :attr:`SportTuning.meaningful` (and so written by :func:`write_weights`)."""
    min_sd_samples: int = MIN_SD_SAMPLES
    """Player-weeks a position needs for its coefficient of variation to be refit."""

    def __post_init__(self) -> None:
        if self.steps < 1 or self.max_candidates < 2:
            raise ValueError("steps must be >= 1 and max_candidates >= 2")
        if self.min_samples < 1 or self.shrinkage < 0 or self.min_improvement < 0:
            raise ValueError("min_samples must be >= 1, shrinkage and min_improvement >= 0")


@dataclass(frozen=True, slots=True)
class PositionFit:
    """One position's fit on every replayed week. ``weights`` are the tuned source weights (``None``: the position keeps
    the base weights, and ``reason`` says why); ``samples`` is the player-weeks scored, ``equal_mae`` the MAE of the
    equal split, ``best_mae`` that of the best grid candidate before shrinkage."""

    position: str
    samples: int
    equal_mae: float
    best_mae: float
    weights: Mapping[str, float] | None
    reason: str | None = None

    @property
    def fitted(self) -> bool:
        return self.weights is not None


@dataclass(frozen=True, slots=True)
class HeldOutError:
    """Leave-one-week-out MAE at one position (``ALL`` for every position) of the tuned weights against the current
    weights (the base file, what production blends with) and against a pure equal split, over the same ``samples``
    player-weeks."""

    position: str
    samples: int
    tuned_mae: float
    current_mae: float
    equal_mae: float

    @property
    def improvement(self) -> float:
        """Points of MAE the tuning saves per player-week over the current weights (positive: better than what is in
        the file)."""
        return self.current_mae - self.tuned_mae


@dataclass(frozen=True)
class SportTuning:
    """What :func:`tune_sport` found for one sport: ``weights`` is the base file with this sport's tables and
    uncertainty replaced by the fit; ``meaningful`` says whether the evidence supports writing it (``notes`` say why
    not).
    """

    sport: Sport
    season: int
    sources: tuple[str, ...]
    periods: tuple[int, ...]
    positions: Mapping[str, PositionFit]
    held_out: Mapping[str, HeldOutError]
    held_out_overall: HeldOutError | None
    sd: Mapping[str, SdEstimate]
    weights: BlendWeights
    meaningful: bool
    notes: tuple[str, ...] = ()

    @property
    def fitted_positions(self) -> tuple[str, ...]:
        return tuple(position for position, fit in self.positions.items() if fit.fitted)


# --- the candidate grid -----------------------------------------------------------------------------------------------


def _compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first, *rest)


def candidate_grid(sources: int, steps: int, max_candidates: int) -> tuple[tuple[float, ...], ...]:
    """Candidate weight shares (each sums to 1) over ``sources`` sources: the equal split first, then every multiple of
    ``1 / steps`` on the simplex, with ``steps`` lowered until at most ``max_candidates`` remain."""
    if sources < 1:
        raise ValueError("a grid needs at least one source")
    while steps > 1 and math.comb(steps + sources - 1, sources - 1) > max_candidates:
        steps -= 1
    equal = tuple(1.0 / sources for _ in range(sources))
    grid = [equal]
    for parts in _compositions(steps, sources):
        shares = tuple(part / steps for part in parts)
        if any(abs(a - b) > TIE for a, b in zip(shares, equal, strict=True)):
            grid.append(shares)
    return tuple(grid)


# --- scoring candidates -----------------------------------------------------------------------------------------------


def _plays(row: ProjectionRow) -> bool:
    return any(value != 0 for value in row.stats.values())


def _common_data(data: BacktestData, sources: Sequence[str]) -> BacktestData:
    """``data`` with each source's rows cut to the player-weeks every source projects to play (a non-zero line)."""
    periods = set(data.periods)
    keyed = {
        source: {
            (row.scoring_period_id, row.espn_id): row
            for row in data.projections[source]
            if row.scoring_period_id in periods
        }
        for source in sources
    }
    shared = set.intersection(*({key for key, row in rows.items() if _plays(row)} for rows in keyed.values()))
    if not shared:
        raise TuneError(f"no player-week is projected by every one of {', '.join(sources)}; nothing to fit")
    common = data
    for source in sources:
        common = common.with_source(source, [keyed[source][key] for key in sorted(shared)])
    return common


def _weights_for(
    base: BlendWeights, sport: Sport, default: Mapping[str, float], overrides: Mapping[str, Mapping[str, float]] | None
) -> BlendWeights:
    """``base`` with the sport's default table replaced by ``default`` and ``overrides`` (position -> source -> weight)
    added; the uncertainty model is the base's."""
    table: dict[str, Mapping[str, float]] = {"default": MappingProxyType(dict(default))}
    for position, weights in (overrides or {}).items():
        table[position] = MappingProxyType(dict(weights))
    return BlendWeights(
        MappingProxyType({sport: MappingProxyType(table)}), MappingProxyType({sport: base.sd_model(sport)})
    )


def _evaluate(weekly: Mapping[int, BacktestData], weights: BlendWeights, sources: Sequence[str]) -> WeekCells:
    """Projection error by period and position of the blend of ``sources`` under ``weights``."""
    cells: WeekCells = {}
    for period, data in weekly.items():
        report = run_backtest(with_blend(data, weights, sources=sources, name=CANDIDATE), sources=[CANDIDATE])
        by_position = report.source(CANDIDATE).by_position
        cells[period] = {
            position: (error.mae * error.samples, error.samples)
            for position, error in by_position.items()
            if position != UNKNOWN_POSITION
        }
    return cells


def _total(cells: WeekCells, periods: Iterable[int], position: str) -> Cell:
    taken = [cells[period][position] for period in periods if position in cells.get(period, {})]
    return math.fsum(total for total, _ in taken), sum(count for _, count in taken)


def _mae(cell: Cell) -> float:
    return cell[0] / cell[1] if cell[1] else 0.0


# --- choosing a position's weights ------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Choice:
    samples: int
    equal_mae: float
    best_mae: float
    weights: dict[str, float] | None
    reason: str | None = None


def _scaled(shares: Sequence[float], sources: Sequence[str], mass: float) -> dict[str, float]:
    return {source: round(share * mass, WEIGHT_DECIMALS) for source, share in zip(sources, shares, strict=True)}


def _choose(
    grid_cells: Sequence[WeekCells],
    candidates: Sequence[Sequence[float]],
    sources: Sequence[str],
    periods: Sequence[int],
    position: str,
    mass: float,
    config: TuneConfig,
) -> _Choice:
    """The tuned weights for ``position`` from the grid's errors over ``periods`` (see the module docstring)."""
    totals = [_total(cells, periods, position) for cells in grid_cells]
    samples = totals[0][1]
    equal_mae = _mae(totals[0])
    if samples < config.min_samples:
        return _Choice(samples, equal_mae, equal_mae, None, f"{samples} player-weeks, fewer than {config.min_samples}")
    equal = candidates[0]

    def key(index: int) -> tuple[float, float]:
        distance = math.fsum((a - b) ** 2 for a, b in zip(candidates[index], equal, strict=True))
        return (round(_mae(totals[index]) / TIE) * TIE, distance)

    best = min(range(len(candidates)), key=key)
    best_mae = _mae(totals[best])
    if equal_mae - best_mae <= max(config.min_improvement * equal_mae, TIE):
        return _Choice(samples, equal_mae, best_mae, None, "no source is clearly better than an equal split")
    alpha = samples / (samples + config.shrinkage)
    shares = [alpha * mine + (1 - alpha) * even for mine, even in zip(candidates[best], equal, strict=True)]
    return _Choice(samples, equal_mae, best_mae, _scaled(shares, sources, mass))


def _positions(cells: WeekCells) -> tuple[str, ...]:
    return tuple(dict.fromkeys(position for week in cells.values() for position in week))


def _position_order(sport: Sport, labels: Iterable[str]) -> list[str]:
    ids = ids_for(sport)

    def key(label: str) -> tuple[int, str]:
        try:
            return (ids.position_id(label), label)
        except KeyError:
            return (10_000, label)

    return sorted(set(labels), key=key)


# --- the uncertainty --------------------------------------------------------------------------------------------------


def fit_sd(
    data: BacktestData,
    weights: BlendWeights,
    *,
    sources: Iterable[str] | None = None,
    min_samples: int = MIN_SD_SAMPLES,
) -> dict[str, SdEstimate]:
    """Residual statistics by position of the blend of ``sources`` (default: every source) under ``weights`` against
    ``data``'s actual lines, over every replayed week (:func:`fm.model.projections.fit_position_sd`). Positions with
    fewer than ``min_samples`` player-weeks, or without a finite positive coefficient of variation, are left out."""
    projected = blend_source(data, weights, sources=sources)
    stamp = utc_now()
    actual = [
        ProjectionRow(
            sport=data.sport,
            espn_id=espn_id,
            source=ESPN,
            kind="actual",
            season=data.season,
            scoring_period_id=week.period,
            stats=dict(line),
            as_of=stamp,
        )
        for week in data.weeks
        for espn_id, line in week.actuals.items()
    ]
    estimates = fit_position_sd(projected, actual, data.settings, positions=data.positions, min_samples=min_samples)
    return {position: e for position, e in estimates.items() if math.isfinite(e.cv) and e.cv > 0}


# --- tuning a sport ---------------------------------------------------------------------------------------------------


def _merged(
    base: BlendWeights, sport: Sport, overrides: Mapping[str, Mapping[str, float]], sd: SdModel
) -> BlendWeights:
    """``base`` with ``overrides`` merged into the sport's position tables and ``sd`` as its uncertainty model."""
    table = {key: MappingProxyType(dict(value)) for key, value in base.tables[sport].items()}
    for position, weights in overrides.items():
        table[position] = MappingProxyType({**table.get(position, {}), **weights})
    return BlendWeights(
        MappingProxyType({**base.tables, sport: MappingProxyType(table)}),
        MappingProxyType({**base.sd_models, sport: sd}),
        path=base.path,
    )


def tune_sport(
    data: BacktestData,
    base: BlendWeights,
    *,
    sources: Sequence[str] | None = None,
    config: TuneConfig | None = None,
) -> SportTuning:
    """Fit the blend weights and uncertainty of ``data``'s sport (see the module docstring for the method).

    ``base`` is the current weights file: it names the sources (default: those of ``data`` it names), supplies the
    weights a position keeps when it falls back, and the sport's other tables and uncertainty floor. Raises
    :class:`TuneError` for fewer than two sources, a source ``data`` lacks, or no player-week every source projects,
    and :class:`fm.model.projections.BlendWeightsError` for a sport ``base`` has no table for.
    """
    settings = config or TuneConfig()
    sport = data.sport
    default = base.weights(sport)
    named = base.sources(sport)
    names = tuple(sources) if sources is not None else tuple(name for name in data.projections if name in named)
    missing = [name for name in names if name not in data.projections]
    if missing:
        raise TuneError(f"no such projection source: {', '.join(missing)} (have {', '.join(data.projections)})")
    if len(set(names)) < 2:
        raise TuneError(f"tuning needs at least two projection sources named in the weights, got {list(names)}")
    if not data.weeks:
        raise TuneError("no replayed weeks to tune on")
    mass = math.fsum(default.get(name, 0.0) for name in names)
    mass = mass if mass > 0 else float(len(names))

    common = _common_data(data, names)
    periods = common.periods
    weekly = {period: common.restrict_to([period]) for period in periods}
    candidates = candidate_grid(len(names), settings.steps, settings.max_candidates)
    grid_cells = [
        _evaluate(weekly, _weights_for(base, sport, dict(zip(names, shares, strict=True)), None), names)
        for shares in candidates
    ]
    positions = _position_order(sport, _positions(grid_cells[0]))

    def choices(train: Sequence[int]) -> dict[str, _Choice]:
        return {
            position: _choose(grid_cells, candidates, names, train, position, mass, settings) for position in positions
        }

    final = choices(periods)
    fits = {
        position: PositionFit(position, c.samples, c.equal_mae, c.best_mae, c.weights, c.reason)
        for position, c in final.items()
    }
    overrides = {position: c.weights for position, c in final.items() if c.weights is not None}

    held_out: dict[str, HeldOutError] = {}
    overall: HeldOutError | None = None
    if len(periods) >= 2:
        current = _evaluate(weekly, base, names)  # what is in the file now, position overrides and all
        held_out, overall = _leave_one_week_out(
            base, sport, weekly, current, grid_cells[0], names, choices, positions, periods
        )

    estimates = fit_sd(
        data,
        _merged(base, sport, overrides, base.sd_model(sport)),
        sources=names,
        min_samples=settings.min_sd_samples,
    )
    sd = base.sd_model(sport).updated(
        {position: replace(e, cv=round(e.cv, CV_DECIMALS)) for position, e in estimates.items()}
    )
    tuned = _merged(base, sport, overrides, sd)

    notes: list[str] = []
    if len(periods) < settings.min_weeks:
        notes.append(f"{len(periods)} weeks replayed, fewer than {settings.min_weeks}")
    if not overrides:
        notes.append("no position has enough evidence to move off the base weights")
    if overall is None:
        notes.append("no held-out score (needs at least two weeks)")
    elif overall.tuned_mae > overall.current_mae + TIE:
        notes.append(
            f"held-out MAE {overall.tuned_mae:.3f} is worse than the current weights' {overall.current_mae:.3f}"
        )
    return SportTuning(
        sport=sport,
        season=data.season,
        sources=names,
        periods=periods,
        positions=MappingProxyType(fits),
        held_out=MappingProxyType(held_out),
        held_out_overall=overall,
        sd=MappingProxyType(estimates),
        weights=tuned,
        meaningful=not notes,
        notes=tuple(notes),
    )


def _leave_one_week_out(
    base: BlendWeights,
    sport: Sport,
    weekly: Mapping[int, BacktestData],
    current: WeekCells,
    equal: WeekCells,
    names: Sequence[str],
    refit: Callable[[Sequence[int]], dict[str, _Choice]],
    positions: Sequence[str],
    periods: Sequence[int],
) -> tuple[dict[str, HeldOutError], HeldOutError | None]:
    """Redo the fit without each week and score that week with the resulting weights, against the ``current`` weights
    (the base file's, scored on the same weeks) and ``equal`` weights."""
    tuned: WeekCells = {}
    for held in periods:
        fold = refit([period for period in periods if period != held])
        overrides = {position: c.weights for position, c in fold.items() if c.weights is not None}
        scored = _evaluate({held: weekly[held]}, _merged(base, sport, overrides, base.sd_model(sport)), names)
        tuned.update(scored)
    result: dict[str, HeldOutError] = {}
    for position in positions:
        mine, now, even = (_total(cells, periods, position) for cells in (tuned, current, equal))
        if even[1]:
            result[position] = HeldOutError(position, even[1], _mae(mine), _mae(now), _mae(even))
    count = sum(e.samples for e in result.values())
    if not count:
        return result, None

    def mean(pick: Callable[[HeldOutError], float]) -> float:
        return math.fsum(pick(e) * e.samples for e in result.values()) / count

    return result, HeldOutError(
        "ALL", count, mean(lambda e: e.tuned_mae), mean(lambda e: e.current_mae), mean(lambda e: e.equal_mae)
    )


# --- the weights file -------------------------------------------------------------------------------------------------


def _key(label: str) -> str:
    return label if label.replace("_", "").replace("-", "").isalnum() else f'"{label}"'


def _number(value: float) -> str:
    return repr(float(value))


def header_of(text: str) -> str:
    """The comment block that opens a weights file: everything before the first table, trailing blank lines dropped,
    ending in one blank line (empty when the file starts with a table)."""
    lines: list[str] = []
    for line in text.splitlines():
        if line.lstrip().startswith("["):
            break
        lines.append(line.rstrip())
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n\n" if lines else ""


def render_blend_weights(weights: BlendWeights, *, header: str = "") -> str:
    """TOML text for ``weights`` that ``BlendWeights.parse`` reads back to the same tables and uncertainty. ``header``
    (see :func:`header_of`) goes first, verbatim; each sport gets its ``default`` table, its position overrides in ESPN
    position order, then its ``sd`` table."""
    blocks: list[str] = []
    for sport in weights.sports:
        tables = weights.tables[sport]
        for label in ("default", *_position_order(sport, weights.positions(sport))):
            lines = [f"[{sport}.{_key(label)}]"]
            lines.extend(f"{_key(source)} = {_number(weight)}" for source, weight in tables[label].items())
            blocks.append("\n".join(lines))
        model = weights.sd_model(sport)
        sd_lines = [f"[{sport}.sd]", f"floor = {_number(model.floor)}", f"default = {_number(model.default)}"]
        for position in _position_order(sport, model.by_position):
            sd_lines.append(f"{_key(position)} = {_number(model.by_position[position])}")
        blocks.append("\n".join(sd_lines))
    return header + "\n\n".join(blocks) + "\n"


@dataclass(frozen=True)
class TuningWrite:
    """What :func:`write_weights` did: the weights now in the file, the sports it applied and the ones it left alone
    with the reason."""

    path: Path
    weights: BlendWeights
    applied: tuple[Sport, ...]
    skipped: Mapping[Sport, str]

    @property
    def changed(self) -> bool:
        return bool(self.applied)


def write_weights(path: Path, tunings: Iterable[SportTuning], *, force: bool = False) -> TuningWrite:
    """Rewrite the weights file at ``path`` with the tuned tables of every :attr:`SportTuning.meaningful` tuning
    (every tuning with ``force``). A sport whose tuning is not meaningful keeps its tables, and when no tuning applies
    the file is not touched at all. The file is read, so the sports and tables the tunings do not cover carry over, and
    its header comment is kept. Returns what was applied and skipped."""
    text = path.read_text(encoding="utf-8")
    current = BlendWeights.load(path)
    tables = dict(current.tables)
    sd_models = dict(current.sd_models)
    applied: list[Sport] = []
    skipped: dict[Sport, str] = {}
    for tuning in tunings:
        sport = tuning.sport
        if not (tuning.meaningful or force):
            skipped[sport] = "; ".join(tuning.notes) or "not meaningful"
            continue
        tables[sport] = tuning.weights.tables[sport]
        sd_models[sport] = tuning.weights.sd_model(sport)
        applied.append(sport)
    result = BlendWeights(MappingProxyType(tables), MappingProxyType(sd_models), path=path)
    if applied:
        path.write_text(render_blend_weights(result, header=header_of(text)), encoding="utf-8", newline="\n")
    return TuningWrite(path, result, tuple(applied), MappingProxyType(skipped))
