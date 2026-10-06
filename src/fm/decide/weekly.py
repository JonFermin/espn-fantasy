"""NBA weekly category planner: which categories to contest, concede or leave alone, and what streamers should chase
(ROADMAP #37, DESIGN sections 9.3 and 9.5).

A category league decides each matchup week on stats, so a Monday plan asks, per category, how likely we are to win it
against this week's opponent and what that implies for the week's moves. Which stats compete, in what order and which
count against us (``TO``) are the league's settings (``scoringItems``, ``isReverseItem``); nothing here names a
category. A points league has no categories to plan: :func:`plan_weekly` and :func:`plan_categories` refuse it
(:class:`WeeklyPlanError`) rather than guess.

**Win probabilities.** The planner reuses :func:`fm.decide.lineup_daily.swing_outlook`: each side's expected category
score over the rest of the matchup in the category model's score units (``CategoryModel.contributions`` of the
players' per-game lines, our best lineup against the opponent's stored roster played at its best), ``P(win) =
Phi(margin / sd)`` with ``sd = game_sd * sqrt(games_ours + games_theirs)``. ``game_sd`` is a single game's spread in
score units; :func:`game_sd_from_tau` derives it per category from the model's ``tau`` (a player's own
game-to-game spread, :func:`fm.model.categories.within_player_sd`), which noisy categories (steals, blocks, the
percentages) need, and falls back to :data:`fm.decide.lineup_daily.DEFAULT_GAME_SD` where a category has none. Where the
caller already has the season simulator's per-category probabilities (``fm.model.simulate.simulate_matchup(...)``
``.categories``, or ``SeasonOdds.team(id).categories``), ``simulated`` replaces the normal approximation's probability
with them; the margin and the weights are then re-derived from it so the plan stays consistent.

**Weights (H-score style).** The H-score of Rosenof (arXiv:2409.09884) values a player by what he adds to the expected
number of categories won given the roster he joins and the opponent he plays, not by a league-wide z-score. The first
order version is each category's ``swing``: ``dP(win category)/d(score) = phi(margin / sd) / sd``. A category nearly
locked either way has a swing near 0 and a toss-up the largest, so the weight follows the roster and the opponent.
:attr:`WeeklyPlan.weights` are those swings with the conceded categories zeroed, in the same unit as the lineup
planner's weights (expected category wins per score unit), so a streamer valued with
``CategoryModel.contribution(line, plan.weights)`` is comparable with a lineup value.
:attr:`WeeklyPlan.relative_weights` rescale them to a mean of 1 over the categories still in play (flat weights when
everything is decided). This module does not feed streaming; ``fm.decide.streaming`` can take ``plan.weights`` as it
takes ``swing_weights`` now.

**Targets, punts and safe categories.** A category with ``P(win) < punt_below`` is a *punt*: conceded, weight 0,
no streamer is spent on it. One with ``P(win) >= safe_above`` is *safe*: it stays in the weights at its own (small)
swing. The rest are *contested*, and the ones to chase. The thresholds (:data:`PUNT_BELOW`, :data:`SAFE_ABOVE`) are
model parameters, as ``game_sd`` is, not league settings; the cap on the number of punts is the league's: a most
categories matchup needs a majority of the categories, so at most ``(n - 1) // 2`` of the ``n`` categories are conceded
(the lowest P first; the rest stay contested), and a league that counts every category as its own win only keeps one
contested (:func:`default_max_punts`). A category left out of the plan (no expected score or spread for it) counts as
conceded, so it comes off that cap.

**Stat gaps.** ``gap`` is how much better than expected we must be to even a category (``P = 0.5``), in the stat's own
units: a count for a counting category, the rate itself for a percentage or per-game stat (the team's rate over its
remaining started games, taking a game's attempts as the pool's average). A negative gap is a cushion. ``reverse``
categories are won by lowering the stat, so their gap is a number to cut.

Only information comes out of here: no proposals, nothing is written, and the module registers no decision. The tick
hands the opponent to the lineup and streaming decisions only (``fm.jobs.tick.OPPONENT_DECISIONS``) and runs every
registered kind in a period-open window, so a ``("nba", "weekly")`` registration is left to the weekly strategist
(ROADMAP #45), which calls :func:`plan_weekly` itself.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import datetime
from enum import StrEnum
from statistics import NormalDist
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

from fm.decide.lineup_daily import (
    DEFAULT_GAME_SD,
    DailyInputs,
    DailyLineupError,
    SwingOutlook,
    daily_inputs,
    plan_week,
    week_category_scores,
)
from fm.espn.ids import Game
from fm.espn.settings import LeagueSettings, ScoringType
from fm.model.categories import CategoryModel, CategoryStats
from fm.model.projections import BlendWeights
from fm.proposals.policy import stored_settings
from fm.sports.base import ScheduleLike
from fm.sports.nba import NBA
from fm.store import LeagueRow, Store

if TYPE_CHECKING:
    from fm.model.availability import OfficialReport
    from fm.sources.nba_injuries import OfficialInjuryReport

PUNT_BELOW: Final = 0.15
"""A category whose P(win) is below this is conceded (about one standard deviation behind): a streamer that moves it by
a typical amount is unlikely to flip it, and the same streamer moves a contested category more."""
SAFE_ABOVE: Final = 0.85
"""A category whose P(win) is at least this is safe: more of the stat adds little."""
MATCHUP_SCORING: Final = frozenset({ScoringType.H2H_CATEGORY, ScoringType.H2H_MOST_CATEGORIES})
"""The scoring types with weekly category matchups."""
_FLOOR: Final = 1e-9
"""Probabilities are kept this far from 0 and 1 before they are turned back into a margin."""
_NORMAL: Final = NormalDist()


class WeeklyPlanError(ValueError):
    """The league cannot be planned by category: not an NBA category league with head-to-head matchups, or the store
    lacks what the plan reads."""


class CategoryStatus(StrEnum):
    """What the plan does with a category."""

    CONTEST = "contest"
    PUNT = "punt"
    SAFE = "safe"


@dataclass(frozen=True, slots=True)
class CategoryPlan:
    """One category of the week.

    ``margin`` and ``sd`` are in the category model's score units (ours minus theirs, signed so that a positive margin
    wins even for a reversed category), ``win_probability`` is ``P(win)`` (a tie counts half), ``swing`` is
    ``dP/d(score)`` and ``weight`` what :attr:`WeeklyPlan.weights` carries (``swing``, 0 for a punt),
    ``relative_weight`` its mean-1 rescaling. ``gap`` is the improvement in the stat's own units that evens the
    category (negative when we lead), ``gap_per_game`` that spread over our remaining started games, and ``reverse``
    says the stat is lowered to improve. ``source`` is ``normal`` (the swing outlook) or ``simulation``.
    """

    category: str
    status: CategoryStatus
    win_probability: float
    margin: float
    sd: float
    swing: float
    weight: float
    relative_weight: float
    gap: float
    gap_per_game: float
    reverse: bool = False
    source: str = "normal"

    @property
    def leading(self) -> bool:
        """True when we are the favourite: ``P(win) >= 0.5``."""
        return self.win_probability >= 0.5


@dataclass(frozen=True)
class WeeklyPlan:
    """The week's category plan, categories in the league's scoring-item order.

    ``expected_wins`` sums the win probabilities; ``matchup_win_probability`` is P(win the matchup) for a most
    categories league (the categories taken as independent; a tie counts half), ``None`` for a league that counts
    every category as its own win. ``weights`` and ``relative_weights`` are what a streamer valuation takes;
    ``punt_below``, ``safe_above`` and ``max_punts`` are the rules the statuses came from. ``period``,
    ``matchup_period`` and ``opponent_team_id`` are filled by :func:`plan_weekly`.
    """

    categories: tuple[CategoryPlan, ...]
    games: float
    expected_wins: float
    weights: Mapping[str, float]
    relative_weights: Mapping[str, float]
    punt_below: float
    safe_above: float
    max_punts: int
    matchup_win_probability: float | None = None
    period: int | None = None
    matchup_period: int | None = None
    opponent_team_id: int | None = None
    warnings: tuple[str, ...] = ()

    def category(self, name: str) -> CategoryPlan:
        """Raises ``KeyError`` naming the league's categories."""
        for entry in self.categories:
            if entry.category == name:
                return entry
        raise KeyError(f"{name!r} is not a category here; the categories are {', '.join(self.names)}")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(entry.category for entry in self.categories)

    @property
    def punts(self) -> tuple[str, ...]:
        """The conceded categories, in league order."""
        return tuple(entry.category for entry in self.categories if entry.status is CategoryStatus.PUNT)

    @property
    def safe(self) -> tuple[str, ...]:
        return tuple(entry.category for entry in self.categories if entry.status is CategoryStatus.SAFE)

    @property
    def contested(self) -> tuple[CategoryPlan, ...]:
        """The categories worth chasing, the largest swing first (ties in league order)."""
        chosen = [entry for entry in self.categories if entry.status is CategoryStatus.CONTEST]
        return tuple(sorted(chosen, key=lambda entry: -entry.swing))

    @property
    def streamer_targets(self) -> tuple[str, ...]:
        """The categories streamers should aim at, most valuable first (:attr:`contested`)."""
        return tuple(entry.category for entry in self.contested)

    def streamer_gaps(self) -> dict[str, float]:
        """The stat gap per contested category: how much the week's adds need to move each to even it (negative: a
        cushion)."""
        return {entry.category: entry.gap for entry in self.contested}


@dataclass(frozen=True, slots=True)
class WeeklyDecision:
    """What :func:`plan_weekly` found: the :class:`WeeklyPlan` (``None`` when there is nothing to plan against, such
    as no opponent roster), the inputs it was read from and the warnings."""

    league_id: int
    plan: WeeklyPlan | None
    inputs: DailyInputs
    warnings: tuple[str, ...] = ()


# --- the planner ------------------------------------------------------------------------------------------------------


def default_max_punts(settings: LeagueSettings, count: int | None = None) -> int:
    """How many categories the league lets us concede: a most-categories matchup needs a majority, so ``(n - 1) // 2``
    of ``n``; a league that scores every category as its own win keeps one contested, ``n - 1``. ``count`` overrides
    ``n`` (the settings' category count)."""
    n = count if count is not None else len(settings.scoring_items)
    if settings.scoring_type is ScoringType.H2H_CATEGORY:
        return max(0, n - 1)
    return max(0, (n - 1) // 2)


def game_sd_from_tau(model: CategoryModel, *, games_per_matchup: float = 1.0) -> dict[str, float]:
    """A single game's spread in each category in score units, from the model's ``tau``: ``tau * sqrt(games) / spread``,
    the per-game standard deviation of the category value over the model's score divisor. ``games_per_matchup`` is
    the one :func:`fm.model.categories.within_player_sd` was given (1 for a per-game ``tau``). A category without a
    ``tau`` (or without spread) is left out, so :func:`fm.decide.lineup_daily.swing_outlook` uses its default. Raises
    ``ValueError`` for a non-positive ``games_per_matchup``."""
    if not math.isfinite(games_per_matchup) or games_per_matchup <= 0:
        raise ValueError(f"games_per_matchup should be a positive number, got {games_per_matchup!r}")
    spreads: dict[str, float] = {}
    for entry in model.stats:
        spread = entry.spread(model.metric, model.kappa)
        if entry.tau > 0 and spread > 0:
            spreads[entry.stat] = entry.tau * math.sqrt(games_per_matchup) / spread
    return spreads


def _check_rules(punt_below: float, safe_above: float, max_punts: int | None) -> None:
    if not 0.0 < punt_below < 0.5:
        raise ValueError(f"punt_below should be between 0 and 0.5, got {punt_below!r}")
    if not 0.5 < safe_above < 1.0:
        raise ValueError(f"safe_above should be between 0.5 and 1, got {safe_above!r}")
    if max_punts is not None and max_punts < 0:
        raise ValueError(f"max_punts should be at least 0, got {max_punts!r}")


def _margin_for(probability: float, sd: float) -> float:
    return _NORMAL.inv_cdf(min(1.0 - _FLOOR, max(_FLOOR, probability))) * sd


def _gap(entry: CategoryStats, margin: float, spread: float, games: float) -> float:
    """What the stat must improve by to even the category: the margin deficit in model value units (``score ×
    spread``), which for a rate is a share of the team's attempts, so the team's rate moves by that over its games."""
    value = -margin * spread
    return value if entry.category.rate is None else value / max(games, 1.0)


def matchup_win_probability(probabilities: Mapping[str, float]) -> float:
    """P(win a most-categories matchup) from independent category probabilities: the Poisson-binomial chance of winning
    more categories than we lose, an even split counting half."""
    chances = list(probabilities.values())
    counts = [1.0]
    for chance in chances:
        counts = [
            (counts[wins] * (1.0 - chance) if wins < len(counts) else 0.0)
            + (counts[wins - 1] * chance if wins > 0 else 0.0)
            for wins in range(len(counts) + 1)
        ]
    total = len(chances)
    win = math.fsum(p for wins, p in enumerate(counts) if 2 * wins > total)
    even = math.fsum(p for wins, p in enumerate(counts) if 2 * wins == total)
    return win + 0.5 * even


def plan_categories(
    model: CategoryModel,
    outlook: SwingOutlook,
    *,
    games: float,
    simulated: Mapping[str, float] | None = None,
    punt_below: float = PUNT_BELOW,
    safe_above: float = SAFE_ABOVE,
    max_punts: int | None = None,
    most_categories: bool = True,
) -> WeeklyPlan:
    """The category plan for the week from a :class:`~fm.decide.lineup_daily.SwingOutlook` and the fitted ``model``
    that scored it (the module docs give the rules).

    ``games`` is our started games over the rest of the matchup. ``simulated`` maps categories to the simulator's
    P(win); a category it lacks keeps the outlook's. ``max_punts`` defaults to ``(n - 1) // 2``
    (:func:`default_max_punts`). ``most_categories`` says the matchup is decided on a majority, which fills
    ``matchup_win_probability``. Raises
    ``ValueError`` for thresholds out of range, a ``simulated`` probability outside [0, 1] or a category the model
    lacks.
    """
    _check_rules(punt_below, safe_above, max_punts)
    names = model.categories
    unknown = sorted((set(outlook.margins) | set(simulated or ())) - set(names))
    if unknown:
        raise ValueError(f"no such categories: {', '.join(unknown)}; the categories are {', '.join(names)}")
    for name, chance in (simulated or {}).items():
        if isinstance(chance, bool) or not math.isfinite(chance) or not 0.0 <= chance <= 1.0:
            raise ValueError(f"simulated P(win {name}) should be in [0, 1], got {chance!r}")
    cap = max_punts if max_punts is not None else max(0, (len(names) - 1) // 2)

    rows: dict[str, tuple[float, float, float, float, str]] = {}  # P, margin, sd, swing, source
    warnings: list[str] = []
    for name in names:
        if name not in outlook.margins:
            warnings.append(f"{name}: the outlook has no expected score, so it is left out of the plan")
            continue
        if model.stat(name).sd <= 0:
            warnings.append(f"{name}: the model sees no spread in it across the pool, so it is left out of the plan")
            continue
        if name not in outlook.sds:
            warnings.append(
                f"{name}: the outlook has an expected score but no spread for it, so it is left out of the plan"
            )
            continue
        sd = max(outlook.sds[name], _FLOOR)
        if simulated is not None and name in simulated:
            chance = simulated[name]
            margin = _margin_for(chance, sd)
            swing = _NORMAL.pdf(_NORMAL.inv_cdf(min(1.0 - _FLOOR, max(_FLOOR, chance)))) / sd
            rows[name] = (chance, margin, sd, swing, "simulation")
        else:
            rows[name] = (outlook.win_probability[name], outlook.margins[name], sd, outlook.weights[name], "normal")
    if not rows:
        raise ValueError("the outlook covers none of the league's categories")

    # A category left out of the plan is not contested either, so it is conceded already: it uses up the punt cap, or
    # the plan would punt ``cap`` more and concede a majority.
    left_out = len(names) - len(rows)
    allowed = max(0, cap - left_out)
    losers = sorted((row[0], name) for name, row in rows.items() if row[0] < punt_below)
    punts = {name for _, name in losers[:allowed]}
    if len(losers) > allowed:
        warnings.append(
            f"{len(losers)} categories are below {punt_below:.0%} but at most {allowed} may be conceded"
            + (f" ({cap} in all, {left_out} already left out of the plan)" if left_out else "")
            + f"; conceding {', '.join(sorted(punts)) or 'none'}, still contesting "
            f"{', '.join(name for _, name in losers[allowed:])}"
        )
    statuses = {
        name: CategoryStatus.PUNT
        if name in punts
        else CategoryStatus.SAFE
        if row[0] >= safe_above
        else CategoryStatus.CONTEST
        for name, row in rows.items()
    }
    weights = {name: 0.0 if statuses[name] is CategoryStatus.PUNT else row[3] for name, row in rows.items()}
    live = [name for name in rows if statuses[name] is not CategoryStatus.PUNT]
    mean = math.fsum(weights[name] for name in live) / len(live) if live else 0.0
    relative = {name: (weights[name] / mean if mean > _FLOOR else 1.0) if name in live else 0.0 for name in rows}

    plans: list[CategoryPlan] = []
    for name in names:
        if name not in rows:
            continue
        chance, margin, sd, swing, source = rows[name]
        entry = model.stat(name)
        gap = _gap(entry, margin, entry.spread(model.metric, model.kappa), games)
        plans.append(
            CategoryPlan(
                category=name,
                status=statuses[name],
                win_probability=chance,
                margin=margin,
                sd=sd,
                swing=swing,
                weight=weights[name],
                relative_weight=relative[name],
                gap=gap,
                gap_per_game=gap / max(games, 1.0),
                reverse=entry.category.reverse,
                source=source,
            )
        )
    chances = {entry.category: entry.win_probability for entry in plans}
    return WeeklyPlan(
        categories=tuple(plans),
        games=games,
        expected_wins=math.fsum(chances.values()),
        weights=MappingProxyType(weights),
        relative_weights=MappingProxyType(relative),
        punt_below=punt_below,
        safe_above=safe_above,
        max_punts=cap,
        matchup_win_probability=matchup_win_probability(chances) if most_categories else None,
        warnings=tuple(warnings),
    )


# --- the store --------------------------------------------------------------------------------------------------------


def check_category_league(settings: LeagueSettings) -> None:
    """Raises :class:`WeeklyPlanError` unless ``settings`` are an NBA league with head-to-head category matchups."""
    if settings.game is not Game.FBA:
        raise WeeklyPlanError(f"league {settings.league_id} is {settings.game.value}, not an NBA (fba) league")
    if not settings.is_categories:
        raise WeeklyPlanError(
            f"league {settings.league_id} scores points, not categories: there are no categories to plan"
        )
    if settings.scoring_type not in MATCHUP_SCORING:
        raise WeeklyPlanError(
            f"league {settings.league_id}: scoring type {settings.scoring_type_raw or settings.scoring_type.value!r} "
            "has no weekly category matchups to plan"
        )


def plan_weekly(
    store: Store,
    league: LeagueRow,
    *,
    schedule: ScheduleLike,
    now: datetime,
    settings: LeagueSettings | None = None,
    period: int | None = None,
    opponent_team_id: int | None = None,
    lines: Mapping[int, Mapping[str, float]] | None = None,
    weights: BlendWeights | None = None,
    model: CategoryModel | None = None,
    official: OfficialReport | OfficialInjuryReport | None = None,
    game_sd: Mapping[str, float] | float | None = None,
    tau_games: float = 1.0,
    margin_so_far: Mapping[str, float] | None = None,
    simulated: Mapping[str, float] | None = None,
    punt_below: float = PUNT_BELOW,
    safe_above: float = SAFE_ABOVE,
    max_punts: int | None = None,
) -> WeeklyDecision:
    """Plan the rest of the current matchup week by category from the store, writing nothing.

    Reads what :func:`fm.decide.lineup_daily.daily_inputs` reads (the roster of ``period``, the opponent's stored
    roster, the day's blended per-game lines, availability); ``settings`` default to the synced ones and ``period`` to
    the day current at ``now`` on the pro schedule. ``game_sd`` defaults to :func:`game_sd_from_tau` of ``model`` (whose
    ``tau`` was fit for ``tau_games`` games per matchup; see :func:`fm.model.categories.within_player_sd`) and else
    :data:`~fm.decide.lineup_daily.DEFAULT_GAME_SD`; ``margin_so_far`` is what the matchup has banked (ours minus
    theirs, score units); ``simulated`` and the thresholds are :func:`plan_categories`'s. Without an opponent, or one
    whose roster is not stored, there is nothing to plan against: the decision's ``plan`` is ``None`` and the warnings
    say why. Raises :class:`WeeklyPlanError` for a league that is not an NBA head-to-head category league, for a league
    that is not synced or whose season is over, and for the lineup planner's errors as ``daily_inputs`` raises them."""
    synced = settings if settings is not None else stored_settings(store, league)
    if synced is None:
        raise WeeklyPlanError(f"league {league.key!r} has no synced settings; run fm sync")
    check_category_league(synced)
    _check_rules(punt_below, safe_above, max_punts)
    if opponent_team_id is not None and opponent_team_id == league.team_id:
        raise WeeklyPlanError(f"the opponent of team {league.team_id} cannot be itself")
    target = period if period is not None else NBA.scoring_period_at(now, schedule)
    if target is None:
        raise WeeklyPlanError(f"the pro schedule has no scoring period left at {now.isoformat()}: the season is over")
    sds: Mapping[str, float] | float = game_sd if game_sd is not None else {}
    if game_sd is None and model is not None:
        sds = game_sd_from_tau(model, games_per_matchup=tau_games)
    try:
        inputs = daily_inputs(
            store,
            league,
            synced,
            schedule=schedule,
            period=target,
            now=now,
            lines=lines,
            weights=weights,
            opponent_team_id=opponent_team_id,
            model=model,
            official=official,
            game_sd=sds if sds else DEFAULT_GAME_SD,
            margin_so_far=margin_so_far,
        )
    except DailyLineupError as exc:
        raise WeeklyPlanError(str(exc)) from exc
    notes = list(inputs.warnings)
    outlook = inputs.values.outlook
    fitted = inputs.values.model
    if outlook is None or fitted is None:
        reason = "no opponent" if opponent_team_id is None else f"no stored roster for opponent team {opponent_team_id}"
        notes.append(f"{reason}: no weekly category plan")
        return WeeklyDecision(league.row_id, None, inputs, tuple(dict.fromkeys(notes)))
    best = plan_week(inputs.players, inputs.slot_counts, inputs.days, slot_limits=inputs.slot_limits)
    if best is None:
        notes.append("no legal lineup could be planned: no weekly category plan")
        return WeeklyDecision(league.row_id, None, inputs, tuple(dict.fromkeys(notes)))
    _, games = week_category_scores(best, inputs.players, inputs.values.vectors)
    plan = plan_categories(
        fitted,
        outlook,
        games=float(games),
        simulated=simulated,
        punt_below=punt_below,
        safe_above=safe_above,
        max_punts=max_punts if max_punts is not None else default_max_punts(synced, len(fitted.categories)),
        most_categories=synced.scoring_type is ScoringType.H2H_MOST_CATEGORIES,
    )
    plan = replace(
        plan,
        period=target,
        matchup_period=inputs.matchup_period,
        opponent_team_id=opponent_team_id,
        warnings=tuple(dict.fromkeys([*plan.warnings, *notes])),
    )
    return WeeklyDecision(league.row_id, plan, inputs, plan.warnings)
