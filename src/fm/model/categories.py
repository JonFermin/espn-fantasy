"""NBA category valuation: volume-weighted z-scores, G-scores for head-to-head, and punt weights (DESIGN section 8.3).

A category league (``LeagueSettings.is_categories``) competes on stats rather than points, so a player is worth how far
he moves each category against the players a team could roster instead. Which stats compete, in what order, and which
count against you are league settings (``scoringItems`` and ESPN's ``isReverseItem``), never code. The lines are the
blend's (:mod:`fm.model.value_nba`), per game, or over a span of days for a schedule-aware value
(:func:`fm.model.value_nba.period_lines`): the model works in the unit of the lines it is fit on.

**Values.** A counting category's value is the stat itself (``REB`` derived from ``OREB`` and ``DREB`` where a line has
only those). A rate category is a ratio of team totals (``FG% = ΣFGM / ΣFGA``), so a player moves it in proportion to
his volume: his value is ``(attempts / ā) × (pct - pool_pct)`` (DESIGN 8.3), computed as ``(made - pool_pct ×
attempts) / ā`` so that a player without attempts is simply 0. ``pool_pct`` is the pool's ``Σmade / Σattempts`` (not
the mean of its percentages) and ``ā`` its mean attempts. A 50 % shooter on 20 shots moves FG% more than a 70 % shooter
on two.
Every rate stat ESPN offers is such a ratio (:data:`RATE_CATEGORIES`, checked against ESPN's own season lines): the
shooting percentages (``AFG%`` counts a three as one and a half makes, as :data:`fm.model.scoring.FBA_DERIVATIONS`
does), the per-game stats (``APG`` is ``AST`` over ``GP``), ``PPM``, ``A/TO``, ``STR`` and ``FTR``.

**Z-scores** are ``(value - μ) / σ`` with the mean and population standard deviation of the values over the pool,
negated for a reversed category (``TO`` in ESPN's default leagues), and 0 in a category the pool shows no spread in.
**G-scores** (Rosenof, "Improving algorithms for fantasy basketball", arXiv:2307.02188) divide by ``√(σ² + κτ²)``
instead: ``τ`` is a player's own spread in the category from one matchup to the next and ``κ = 2N / (2N - 1)`` for
``N`` players a side, from the variance of a head-to-head difference between two teams of ``N``
(:func:`kappa_for`). Noisy categories (steals, blocks, the percentages) shrink, because a head-to-head matchup turns on
a week's games rather than season means. With ``τ = 0`` a G-score is the z-score exactly. Head-to-head leagues default
to G-scores and roto leagues, whose season totals average the weekly noise away, to z-scores (:func:`metric_for`).

**Pool.** ``μ``, ``σ``, the pool percentages and ``ā`` come from the players a league could roster: the ``team_count ×
roster_size`` best by total score, found by scoring everyone, keeping the best and refitting until the pool stops
changing. An average rostered player then scores about 0 in every category. :meth:`CategoryModel.contributions`
measures a line from an empty slot instead of from that average player: what it adds to a team's totals in the same
units, the per-game value a daily lineup or a streamer needs (an empty slot adds no points and leaves FG% alone).

**τ** comes from single-game lines: stats.nba.com game logs (:func:`game_log_lines`) or the daily actuals ``fm sync``
stores (:func:`actual_game_lines`). :func:`within_player_sd` averages the players' game-to-game variances of each
category value and scales them to a matchup of ``g`` games: per-game means over ``g`` games vary by ``τ² = var / g``.

**Punts** (:func:`punt_weights`) weight a category 0 in a player's total; any non-negative weights work, which leaves
room for the roster-aware re-weighting of ROADMAP #37.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from types import MappingProxyType
from typing import Final

import numpy as np
import polars as pl

from fm.espn.ids import FBA, Game
from fm.espn.settings import LeagueSettings, ScoringType
from fm.model.scoring import FBA_DERIVATIONS, Ratio, ScoringError, derive_stats, rule_inputs
from fm.store import ProjectionRow

GAMES_PLAYED: Final = "GP"
"""The games-played stat: the volume of a per-game category, and 1 in a single-game line."""
MINUTES: Final = "MIN"
POOL_ITERATIONS: Final = 5
"""Rounds of re-picking the pool after the first fit on everyone; :func:`fit_categories` stops sooner once the pool
stops changing."""
MIN_TAU_GAMES: Final = 10
"""Games a player needs before his game-to-game variance counts towards ``τ``."""
NBA_PLAYER_ID: Final = "PLAYER_ID"


class CategoryMetric(StrEnum):
    """How category values become scores: z-scores (roto) or G-scores (head-to-head)."""

    Z = "z"
    G = "g"


RATE_CATEGORIES: Mapping[str, Ratio] = MappingProxyType(
    {
        **{stat: rule for stat, rule in FBA_DERIVATIONS.items() if isinstance(rule, Ratio)},
        **{
            stat: Ratio(base, GAMES_PLAYED)
            for stat, base in (
                ("APG", "AST"),
                ("BPG", "BLK"),
                ("MPG", MINUTES),
                ("PPG", "PTS"),
                ("RPG", "REB"),
                ("SPG", "STL"),
                ("TOPG", "TO"),
                ("3PG", "3PM"),
            )
        },
        "PPM": Ratio("PTS", MINUTES),
        "A/TO": Ratio("AST", "TO"),
        "STR": Ratio("STL", "TO"),
        "FTR": Ratio("FTA", "FGA"),
    }
)
"""Every ``fba`` stat that is a ratio of two counting stats, by abbreviation: its team value is the ratio of the team's
totals, so it is volume-weighted. ESPN's season lines carry each of them with exactly these values (``A/TO`` is ``AST
/ TO``, ``FTR`` is ``FTA / FGA``, ``MPG`` is ``MIN / GP``)."""

for _stat, _rule in RATE_CATEGORIES.items():  # a typo fails at import rather than in a valuation
    for _abbreviation in (_stat, *rule_inputs(_rule)):
        FBA.stat_id(_abbreviation)

NBA_STATS_COLUMNS: Mapping[str, str] = MappingProxyType(
    {
        "MIN": MINUTES,
        "PTS": "PTS",
        "FGM": "FGM",
        "FGA": "FGA",
        "FG3M": "3PM",
        "FG3A": "3PA",
        "FTM": "FTM",
        "FTA": "FTA",
        "OREB": "OREB",
        "DREB": "DREB",
        "REB": "REB",
        "AST": "AST",
        "STL": "STL",
        "BLK": "BLK",
        "TOV": "TO",
        "PF": "PF",
    }
)
"""stats.nba.com box-score columns (``fm.sources.nba_stats`` game logs) to ESPN abbreviations."""

for _abbreviation in NBA_STATS_COLUMNS.values():
    FBA.stat_id(_abbreviation)


# --- categories -------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class StatCategory:
    """One competed stat: its abbreviation, whether lower wins (ESPN's ``isReverseItem``: ``TO``), and for a rate the
    counting stats it divides (:data:`RATE_CATEGORIES`)."""

    stat: str
    reverse: bool = False
    rate: Ratio | None = None

    @property
    def sign(self) -> float:
        return -1.0 if self.reverse else 1.0

    def made(self, line: Mapping[str, float]) -> float:
        """A rate's numerator in ``line``, bonus included (``AFG%`` counts half a make per ``3PM``); for a counting
        category the stat itself."""
        if self.rate is None:
            return line.get(self.stat, 0.0)
        bonus = line.get(self.rate.bonus, 0.0) if self.rate.bonus is not None else 0.0
        return line.get(self.rate.numerator, 0.0) + self.rate.bonus_weight * bonus

    def volume(self, line: Mapping[str, float]) -> float:
        """A rate's denominator in ``line`` (attempts, games, minutes or turnovers); 1 for a counting category."""
        return line.get(self.rate.denominator, 0.0) if self.rate is not None else 1.0

    def inputs(self) -> tuple[str, ...]:
        """The stats a line needs for this category's value."""
        return (self.stat,) if self.rate is None else rule_inputs(self.rate)


def league_categories(settings: LeagueSettings) -> tuple[StatCategory, ...]:
    """The league's categories in scoring-item order. Raises ``ValueError`` for a league that is not an NBA category
    league."""
    if settings.game is not Game.FBA:
        raise ValueError(f"league {settings.league_id} is {settings.game.value}, not an NBA (fba) league")
    if not settings.is_categories:
        raise ValueError(f"league {settings.league_id} scores points, not categories; value it with fm.model.value_nba")
    return tuple(
        StatCategory(item.stat, item.is_reverse, RATE_CATEGORIES.get(item.stat)) for item in settings.scoring_items
    )


def metric_for(settings: LeagueSettings) -> CategoryMetric:
    """z-scores for a roto league, G-scores for every head-to-head format."""
    return CategoryMetric.Z if settings.scoring_type is ScoringType.ROTO else CategoryMetric.G


def kappa_for(players_per_team: int) -> float:
    """``κ = 2N / (2N - 1)`` for ``N`` players a side (arXiv:2307.02188): a player's own matchup-to-matchup variance
    counts once more than his ``2N - 1`` teammates' and opponents' in the variance of a head-to-head difference."""
    if players_per_team < 1:
        raise ValueError(f"players_per_team should be at least 1, got {players_per_team}")
    return 2 * players_per_team / (2 * players_per_team - 1)


def punt_weights(categories: Iterable[str], punt: Iterable[str] = ()) -> dict[str, float]:
    """Weight 1 for every category and 0 for the punted ones. Raises ``ValueError`` naming a punt that is not one of
    ``categories``."""
    names = tuple(categories)
    punted = set(punt)
    unknown = sorted(punted - set(names))
    if unknown:
        raise ValueError(f"cannot punt {', '.join(unknown)}: the categories are {', '.join(names)}")
    return {name: 0.0 if name in punted else 1.0 for name in names}


def _prepare(line: Mapping[str, float]) -> dict[str, float]:
    """``line`` with its derivable ``fba`` stats filled (``REB`` from ``OREB`` + ``DREB``); a non-numeric or non-finite
    value raises :class:`fm.model.scoring.ScoringError`."""
    return derive_stats(line, Game.FBA)


def _finite_non_negative(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"{what} should be a finite number >= 0, got {value!r}")
    return float(value)


# --- the model --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class CategoryStats:
    """One category over the pool: the mean ``μ`` (0 for a rate, by construction) and population standard deviation
    ``σ`` of its values, its matchup-to-matchup spread ``τ``, and for a rate the pool's percentage (``Σmade /
    Σvolume``) and mean volume ``ā``."""

    category: StatCategory
    mean: float
    sd: float
    tau: float = 0.0
    pool_rate: float = 0.0
    mean_volume: float = 0.0

    @property
    def stat(self) -> str:
        return self.category.stat

    def value(self, line: Mapping[str, float]) -> float:
        """The category value of a prepared line: the stat, or a rate's ``(made - pool_rate × volume) / ā``, which is
        ``(volume / ā) × (rate - pool_rate)``."""
        if self.category.rate is None:
            return line.get(self.stat, 0.0)
        if self.mean_volume <= 0:
            return 0.0
        return (self.category.made(line) - self.pool_rate * self.category.volume(line)) / self.mean_volume

    def spread(self, metric: CategoryMetric, kappa: float) -> float:
        """What a score divides by: ``σ`` for a z-score, ``√(σ² + κτ²)`` for a G-score."""
        if metric is CategoryMetric.Z:
            return self.sd
        return math.hypot(self.sd, math.sqrt(kappa) * self.tau)

    def _scaled(self, value: float, spread: float, *, from_empty: bool) -> float:
        """``sign × (value - baseline) / spread``, the baseline being the pool mean, or an empty slot's value (0) with
        ``from_empty``; 0 without spread."""
        if spread <= 0:
            return 0.0
        return self.category.sign * (value - (0.0 if from_empty else self.mean)) / spread


@dataclass(frozen=True, slots=True)
class CategoryScores:
    """A player's score per category and their weighted total."""

    espn_id: int
    scores: Mapping[str, float]
    total: float


@dataclass(frozen=True)
class CategoryModel:
    """Category statistics fitted on a pool (:func:`fit_categories`), scoring any line in the units it was fit on.

    ``metric`` is the default for every method; ``kappa`` is the G-score's ``κ``; ``pool`` lists the ESPN ids the
    statistics come from; ``warnings`` name the categories that cannot tell players apart.
    """

    stats: tuple[CategoryStats, ...]
    kappa: float
    metric: CategoryMetric
    pool: tuple[int, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def categories(self) -> tuple[str, ...]:
        return tuple(entry.stat for entry in self.stats)

    def stat(self, category: str) -> CategoryStats:
        """The statistics of one category. Raises ``KeyError`` naming the league's categories."""
        for entry in self.stats:
            if entry.stat == category:
                return entry
        raise KeyError(f"{category!r} is not a category here; the categories are {', '.join(self.categories)}")

    def values(self, line: Mapping[str, float]) -> dict[str, float]:
        """The line's value in each category (volume-weighted for rates), before scaling."""
        prepared = _prepare(line)
        return {entry.stat: entry.value(prepared) for entry in self.stats}

    def scores(self, line: Mapping[str, float], *, metric: CategoryMetric | None = None) -> dict[str, float]:
        """Scores per category against the pool's average player (``metric`` defaults to the model's)."""
        return self._scores(_prepare(line), metric or self.metric, from_empty=False)

    def z_scores(self, line: Mapping[str, float]) -> dict[str, float]:
        return self.scores(line, metric=CategoryMetric.Z)

    def g_scores(self, line: Mapping[str, float]) -> dict[str, float]:
        return self.scores(line, metric=CategoryMetric.G)

    def contributions(self, line: Mapping[str, float], *, metric: CategoryMetric | None = None) -> dict[str, float]:
        """What the line adds to a team's categories against an empty slot, in score units: each value measured from
        an empty slot's, which is 0 (no points; a percentage left alone), instead of from the pool's average player.
        Turnovers still count against."""
        return self._scores(_prepare(line), metric or self.metric, from_empty=True)

    def total(
        self,
        line: Mapping[str, float],
        weights: Mapping[str, float] | None = None,
        *,
        metric: CategoryMetric | None = None,
    ) -> float:
        """The weighted sum of :meth:`scores`; a category missing from ``weights`` weighs 1 (see
        :func:`punt_weights`)."""
        return self._weighted(self.scores(line, metric=metric), weights)

    def contribution(
        self,
        line: Mapping[str, float],
        weights: Mapping[str, float] | None = None,
        *,
        metric: CategoryMetric | None = None,
    ) -> float:
        """The weighted sum of :meth:`contributions`: a game's value from an empty slot, for a daily lineup."""
        return self._weighted(self.contributions(line, metric=metric), weights)

    def rank(
        self,
        lines: Mapping[int, Mapping[str, float]],
        weights: Mapping[str, float] | None = None,
        *,
        metric: CategoryMetric | None = None,
    ) -> list[CategoryScores]:
        """Every player's scores and weighted total, best total first (ties by ESPN id)."""
        chosen = metric or self.metric
        checked = self._checked_weights(weights)
        ranked: list[CategoryScores] = []
        for espn_id, line in lines.items():
            scores = self._scores(_prepare(line), chosen, from_empty=False)
            total = math.fsum(checked.get(stat, 1.0) * score for stat, score in scores.items())
            ranked.append(CategoryScores(espn_id, MappingProxyType(scores), total))
        return sorted(ranked, key=lambda entry: (-entry.total, entry.espn_id))

    def with_tau(self, tau: Mapping[str, float]) -> CategoryModel:
        """This model with ``τ`` replaced for the categories in ``tau`` (others keep theirs). Raises ``ValueError``
        for a category the league does not have or a value that is not a finite number >= 0."""
        checked = _checked_tau(tau, self.categories)
        stats = tuple(replace(entry, tau=checked.get(entry.stat, entry.tau)) for entry in self.stats)
        return replace(self, stats=stats)

    def _scores(self, prepared: Mapping[str, float], metric: CategoryMetric, *, from_empty: bool) -> dict[str, float]:
        return {
            entry.stat: entry._scaled(entry.value(prepared), entry.spread(metric, self.kappa), from_empty=from_empty)
            for entry in self.stats
        }

    def _weighted(self, scores: Mapping[str, float], weights: Mapping[str, float] | None) -> float:
        checked = self._checked_weights(weights)
        return math.fsum(checked.get(stat, 1.0) * score for stat, score in scores.items())

    def _checked_weights(self, weights: Mapping[str, float] | None) -> dict[str, float]:
        if not weights:
            return {}
        unknown = sorted(set(weights) - set(self.categories))
        if unknown:
            raise ValueError(f"weights for {', '.join(unknown)}: the categories are {', '.join(self.categories)}")
        return {stat: _finite_non_negative(value, f"weight of {stat}") for stat, value in weights.items()}


def _checked_tau(tau: Mapping[str, float], categories: Sequence[str]) -> dict[str, float]:
    unknown = sorted(set(tau) - set(categories))
    if unknown:
        raise ValueError(f"tau for {', '.join(unknown)}: the categories are {', '.join(categories)}")
    return {stat: _finite_non_negative(value, f"tau of {stat}") for stat, value in tau.items()}


def _plain_total(model: CategoryModel, prepared: Mapping[str, float]) -> float:
    """A prepared line's unweighted total in the model's own metric: what the pool is picked by."""
    return math.fsum(model._scores(prepared, model.metric, from_empty=False).values())


def _fit(
    categories: Sequence[StatCategory], pool: Sequence[Mapping[str, float]], tau: Mapping[str, float]
) -> tuple[CategoryStats, ...]:
    fitted: list[CategoryStats] = []
    for category in categories:
        made = np.array([category.made(line) for line in pool], dtype=float)
        if category.rate is None:
            values = made
            mean = float(values.mean())
            rate = mean_volume = 0.0
        else:
            volume = np.array([category.volume(line) for line in pool], dtype=float)
            mean_volume = float(volume.mean())
            rate = float(made.sum() / volume.sum()) if mean_volume > 0 else 0.0
            values = (made - rate * volume) / mean_volume if mean_volume > 0 else np.zeros(len(pool))
            mean = 0.0  # Σ(made - rate × volume) is 0 over the pool the rate comes from
        fitted.append(
            CategoryStats(
                category,
                mean=mean,
                sd=float(np.sqrt(np.mean((values - mean) ** 2))),
                tau=tau.get(category.stat, 0.0),
                pool_rate=rate,
                mean_volume=mean_volume,
            )
        )
    return tuple(fitted)


def fit_categories(
    lines: Mapping[int, Mapping[str, float]],
    settings: LeagueSettings,
    *,
    metric: CategoryMetric | None = None,
    pool_size: int | None = None,
    players_per_team: int | None = None,
    tau: Mapping[str, float] | None = None,
    iterations: int = POOL_ITERATIONS,
) -> CategoryModel:
    """Fit the league's categories on ``lines`` (ESPN id -> stat line, per game or over a span; one unit throughout).

    The first fit is on everyone; then the pool is the ``pool_size`` best players by unweighted total (default
    ``team_count × roster_size``; everyone when there are no more), re-picked and refit until it stops changing or
    ``iterations`` rounds have run. ``players_per_team`` sets ``κ`` (default: the league's active lineup slots, the
    players whose games count each day); ``tau`` gives ``τ`` per category in the lines' unit (0 for the rest; see
    :func:`within_player_sd`); ``metric`` defaults to :func:`metric_for`. Raises ``ValueError`` for a league that is not
    an NBA category league, fewer than two lines, or a bad ``tau``, and :class:`fm.model.scoring.ScoringError` naming
    the player for a non-numeric or non-finite stat.
    """
    categories = league_categories(settings)
    chosen = metric or metric_for(settings)
    kappa = kappa_for(players_per_team if players_per_team is not None else settings.active_slot_count)
    size = pool_size if pool_size is not None else settings.team_count * settings.roster_size
    if size < 2:
        raise ValueError(f"pool_size should be at least 2, got {size}")
    taus = _checked_tau(tau or {}, [category.stat for category in categories])
    prepared: dict[int, dict[str, float]] = {}
    for espn_id, line in lines.items():
        try:
            prepared[espn_id] = _prepare(line)
        except ScoringError as exc:
            raise ScoringError(f"ESPN {espn_id}: {exc}") from exc
    if len(prepared) < 2:
        raise ValueError(f"category scores need at least two players to compare, got {len(prepared)}")

    def fitted(pool: list[int]) -> CategoryModel:
        stats = _fit(categories, [prepared[espn_id] for espn_id in pool], taus)
        return CategoryModel(stats, kappa, chosen, tuple(pool))

    pool = sorted(prepared)
    model = fitted(pool)
    for _ in range(iterations if size < len(prepared) else 0):
        totals = {espn_id: _plain_total(model, line) for espn_id, line in prepared.items()}
        best = sorted(sorted(prepared, key=lambda espn_id: (-totals[espn_id], espn_id))[:size])
        if best == pool:
            break
        pool = best
        model = fitted(pool)
    warnings = [
        f"{entry.stat}: no spread across the {len(pool)}-player pool (no line carries "
        f"{' or '.join(entry.category.inputs())}?); every player scores 0 in it"
        for entry in model.stats
        if entry.sd <= 0
    ]
    return replace(model, warnings=tuple(warnings))


# --- tau from single games --------------------------------------------------------------------------------------------


def _number(value: object) -> float | None:
    """A box-score value as a float: numbers, numeric strings and ``MM:SS`` minutes; ``None`` for anything else."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, str):
        text = value.strip()
        minutes, colon, seconds = text.partition(":")
        try:
            return float(minutes) + float(seconds) / 60 if colon else float(text)
        except ValueError:
            return None
    return None


def _played(line: Mapping[str, float]) -> bool:
    """True for a game the player took part in, judged on the line as recorded: minutes when it has them, else games
    played, else any non-zero stat (an empty line is a game he missed)."""
    if MINUTES in line:
        return line[MINUTES] > 0
    if GAMES_PLAYED in line:
        return line[GAMES_PLAYED] > 0
    return any(value != 0 for value in line.values())


def game_log_lines(frame: pl.DataFrame) -> dict[int, list[dict[str, float]]]:
    """stats.nba.com player game logs (``NbaStatsSource.game_logs``) as single-game stat lines keyed by ESPN
    abbreviation (:data:`NBA_STATS_COLUMNS`) with ``GP`` 1, grouped by nba.com person id in log order. Games without
    minutes are left out. Raises ``ValueError`` for a frame without ``PLAYER_ID``."""
    if NBA_PLAYER_ID not in frame.columns:
        raise ValueError(f"game logs need a {NBA_PLAYER_ID} column; got {', '.join(frame.columns)}")
    columns = [column for column in NBA_STATS_COLUMNS if column in frame.columns]
    games: dict[int, list[dict[str, float]]] = {}
    for record in frame.select(NBA_PLAYER_ID, *columns).iter_rows(named=True):
        person = _number(record[NBA_PLAYER_ID])
        if person is None:
            continue
        box = {NBA_STATS_COLUMNS[column]: value for column in columns if (value := _number(record[column])) is not None}
        if _played(box):
            games.setdefault(int(person), []).append({GAMES_PLAYED: 1.0, **box})
    return games


def actual_game_lines(rows: Iterable[ProjectionRow]) -> dict[int, list[dict[str, float]]]:
    """ESPN's single-day actual lines as ``fm sync`` stores them (``kind`` actual, a scoring period above 0: ``fba``'s
    "Game" split 5) grouped by ESPN id in period order, ``GP`` 1 where a line lacks it. Season lines, projections, other
    sports and games without minutes are left out."""
    games: dict[int, list[tuple[int, dict[str, float]]]] = {}
    for row in rows:
        if row.sport != "nba" or row.kind != "actual" or row.scoring_period_id <= 0:
            continue
        if _played(row.stats):
            games.setdefault(row.espn_id, []).append((row.scoring_period_id, {GAMES_PLAYED: 1.0, **row.stats}))
    return {espn_id: [line for _, line in sorted(days, key=lambda day: day[0])] for espn_id, days in games.items()}


def within_player_sd(
    games: Mapping[int, Sequence[Mapping[str, float]]],
    model: CategoryModel,
    *,
    games_per_matchup: float,
    min_games: int = MIN_TAU_GAMES,
    players: Iterable[int] | None = None,
) -> dict[str, float]:
    """``τ`` per category for :meth:`CategoryModel.with_tau`, for a model fit on per-game lines.

    Each player with at least ``min_games`` games (among ``players`` when given) contributes his sample variance of
    the category's per-game value (rates volume-weighted with the model's pool percentage and ``ā``); ``τ`` is the
    square root of their mean over ``games_per_matchup`` games, the spread of a matchup's per-game average. (A model fit
    on matchup totals takes ``τ × games_per_matchup``.) Categories whose stats no game line carries are left out.
    Raises ``ValueError`` when no player has ``min_games`` games or for a non-positive ``games_per_matchup``.
    """
    if not math.isfinite(games_per_matchup) or games_per_matchup <= 0:
        raise ValueError(f"games_per_matchup should be a positive number, got {games_per_matchup!r}")
    if min_games < 2:
        raise ValueError(f"min_games should be at least 2 (a variance needs two games), got {min_games}")
    wanted = None if players is None else set(players)
    prepared = {
        key: [_prepare(line) for line in lines]
        for key, lines in games.items()
        if (wanted is None or key in wanted) and len(lines) >= min_games
    }
    if not prepared:
        raise ValueError(f"no player has {min_games} games to estimate tau from")
    carried = {stat for lines in prepared.values() for line in lines for stat in line}
    tau: dict[str, float] = {}
    for entry in model.stats:
        if not set(entry.category.inputs()) <= carried:
            continue
        variances = [float(np.var([entry.value(line) for line in lines], ddof=1)) for lines in prepared.values()]
        tau[entry.stat] = math.sqrt(math.fsum(variances) / len(variances) / games_per_matchup)
    return tau
