"""Trade evaluator and finder (DESIGN section 9.4, ROADMAP #38): what a deal does to both rosters, to our title odds and
to the other manager's willingness to take it.

This module evaluates and ranks. It never executes anything and has no ESPN write path (CLAUDE.md: trades are
approval-only; workers propose, the executor acts): :func:`propose_trades` drafts :class:`fm.proposals.TradePayload`
proposals through :func:`fm.proposals.propose`, whose trade kinds are hard-coded approve-only.

**The league's numbers.** :func:`load_trade_context` reads what ``fm sync`` stored (every roster, the players, ESPN's
projection lines) and builds a :class:`TradeModel` for the league's sport and scoring kind. Rosters are valued by their
best lineup, never by a sum of players.

- NFL points league: :func:`fm.model.valuation.load_valuation` with ``include_rostered`` values every player on any
  roster; a roster's value is :meth:`fm.model.valuation.RosterValuer.value` (the best lineup each week, playoff weeks
  weighted up, later-week holes filled from the wire) and a team's matchup strength is the sum of those weekly lineups
  over the matchup period's scoring periods.
- NBA points league: the weeks are the league's matchup periods (the calendar resolves the weeks the settings list as
  schedule periods, :func:`fm.espn.calendar.league_matchup_days`); a player's week is his per-game league points times
  the games his pro team plays in the rest of the week, and a week's lineup is one assignment over slot eligibility, not
  a lineup per day (a trade is judged on a week's totals, so this is the approximation to know about).
- NBA category league: a player is worth his category contributions against an empty slot
  (:meth:`fm.model.categories.CategoryModel.contribution`, G-scores for head-to-head and z-scores for roto) over the
  games left; a roster's value is the best assignment of those, a team's matchup strength is
  :func:`fm.model.simulate.category_outlook` over its lineup's lines for each matchup week, and
  :attr:`TradeEvaluation.fit` names the categories a deal moves.

**Availability.** Trades are valued on the engine's own ``p_active``, never on a Claude-only signal (CLAUDE.md): the NFL
valuation recomputes it with a bare :func:`fm.model.availability.assess`, and :func:`engine_p_active` reads the stored
availability rows through ``inputs["news"]["before"]`` (the number before the news step) for the NBA and the output. A
player the engine says is out (``p_active`` at or below :data:`OUT_BELOW`) is never offered as the piece we ask for
by :func:`find_trades`, and :func:`evaluate_trade` warns about him.

**Title odds.** :func:`fm.model.simulate.simulate_season` runs over the league's ``mMatchup`` schedule with every
team's outlook, then again with the two teams' rosters swapped, from the same seed and run count, so the draws are
common and the difference is the trade (a symmetric trade gives exactly 0). Without a schedule, or for a league the
simulator cannot run (roto, an unresolvable calendar), the trade is judged on its change in rest-of-season value and
says so.

**Legality** (:func:`check_legality`) is read from the league's settings: the trade deadline, roster size (active plus
bench slots, with IR separate) with the players each side would have to drop, per-position limits, the roster lock
(a player whose game has started under ``rosterLocktimeType``), untouchables, and IR placement. A trade that needs
a drop on our side is legal, but it cannot be proposed (a :class:`TradePayload` carries no drop), so
:func:`find_trades` leaves it out unless asked.

**Finding.** For every opponent :func:`find_trades` enumerates our give sets and their get sets of 1-for-1, 2-for-1,
1-for-2 and 2-for-2 (:attr:`SearchOptions.shapes`) from the players each side could part with, screens each deal with a
greedy lineup value (:func:`quick_value`: the best player in the most specific open slot, empty slots at replacement
level), keeps the ones that help us, are legal and could be accepted, re-scores the best ``finalists`` with the exact
lineup values and a season simulation, and ranks by our change in title odds times P(accept).

**P(accept)** (:func:`acceptance_probability`) models the other manager's view, not ours: market values from
:meth:`fm.sources.market.MarketSource.market_values` (FantasyCalc trade values for the NFL, ESPN ranks and ownership
for both sports, the league's rank type from :func:`rank_type_for`, our own rest-of-season ranks when a player has
neither) say whether the package he receives is worth at least the one he sends, and his roster's need (the change in
*his* rest-of-season lineup value, which sees his positional holes, his byes and, in a category league, his weak
categories) says whether he wants it. They combine in a logistic whose weights are :class:`AcceptanceParams`. It is a
prior to calibrate as offers resolve (DESIGN 9.4), not a fitted model: every number behind it is kept in the engine
numbers of a proposal for that.
"""

from __future__ import annotations

import itertools
import math
import re
from collections import Counter
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Final, Protocol

from fm.config import Config
from fm.espn.calendar import league_matchup_days, matchup_period_of
from fm.espn.ids import Game
from fm.espn.models import MatchupsView
from fm.espn.settings import LeagueSettings, LockType, ScoringType
from fm.model.availability import assess
from fm.model.categories import CategoryModel, fit_categories
from fm.model.projections import ESPN, BlendWeights, ProjectionSourceRegistry, source_registry
from fm.model.scoring import Scorer
from fm.model.simulate import (
    DEFAULT_CV,
    MATCHUP_SCORING_TYPES,
    CategoryOutlook,
    SeasonOdds,
    SimulationError,
    TeamOdds,
    TeamOutlook,
    category_outlook,
    simulate_season,
)
from fm.model.valuation import (
    BASIS_SEASON,
    DEFAULT_PLAYOFF_WEIGHT,
    Horizon,
    PlayerOutlook,
    RosterValuer,
    ValuationError,
    eligible_active_slots,
    last_scoring_period,
    league_settings,
    load_valuation,
    slot_instances,
    start_value,
)
from fm.model.value_nba import SEASON_PERIOD, blend_day, period_lines, team_games
from fm.proposals import PolicyError, ProposalKind, TradePayload, evaluate, find_untouchables, propose
from fm.sources.base import Fetched
from fm.sources.market import LeagueShape, MarketValue
from fm.sports.base import ScheduleLike, plugin_for
from fm.store import LeagueRow, PlayerRow, ProposalRow, Sport, Store

TRADES_KIND: Final = "trades"
"""The decision kind of this module (``fm.decide.registry``); the tick does not run it, trades are asked for."""
TRADES_CREATED_BY: Final = "decide.trades"
"""``created_by`` of the proposals :func:`propose_trades` stores."""
DEFAULT_SEED: Final = 38
"""The simulation seed every evaluation uses unless told otherwise, so a trade's numbers reproduce."""
FINDER_RUNS: Final = 4000
"""Simulation runs per finalist in :func:`find_trades`; :func:`evaluate_trade` defaults to 10 000, the simulator's."""
OUT_BELOW: Final = 0.05
"""A player whose engine ``p_active`` is at or below this is treated as out."""
MARKET_TOP: Final = 10_000.0
MARKET_TAIL: Final = 10.0
"""The scale of a market value: FantasyCalc's top redraft player is about 10 000 and its tail single digits, so a rank
turns into a value on the same scale (:func:`rank_value`)."""
ROS_RANK: Final = "ros_rank"
FANTASYCALC: Final = "fantasycalc"
ESPN_RANK: Final = "espn_rank"
"""Where a player's market value came from (:attr:`MarketBook.basis`)."""
BASIS_TITLE: Final = "title"
BASIS_ROS: Final = "ros"
"""What :attr:`TradeEvaluation.score` multiplies P(accept) by: our change in title odds, or (without a simulation) our
change in rest-of-season value in starter seasons."""
ACCEPT: Final = "accept"
DECLINE: Final = "decline"
COUNTER: Final = "counter"
EDGE_TITLE: Final = 0.002
"""The change in title odds below which a deal is a wash."""
EDGE_ROS: Final = 0.02
"""The same in starter seasons when the simulation is unavailable."""
COUNTER_ACCEPT: Final = 0.6
"""P(accept) from which a deal that does not help us is worth countering: the other manager likes it enough to give
something back."""


class TradeError(ValueError):
    """A trade cannot be evaluated: the league is not synced, a player is not on the roster it must be on, or the
    text names nobody. The message says what to fix."""


# --- the deal and what comes out of evaluating it ---------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TradeSpec:
    """A deal with one other team: ``give`` are our players (ESPN ids), ``get`` are theirs."""

    other_team_id: int
    give: tuple[int, ...]
    get: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.give and not self.get:
            raise TradeError("a trade needs players on at least one side")
        both = set(self.give) & set(self.get)
        if both:
            raise TradeError(f"players {sorted(both)} appear on both sides of the trade")
        if len(set(self.give)) != len(self.give) or len(set(self.get)) != len(self.get):
            raise TradeError("a player appears twice on one side of the trade")

    def payload(self) -> TradePayload:
        """The proposal payload: what a ``trade_propose`` carries (``give_espn_ids`` ours, ``get_espn_ids`` theirs)."""
        return TradePayload(
            other_team_id=self.other_team_id, give_espn_ids=tuple(self.give), get_espn_ids=tuple(self.get)
        )

    @property
    def key(self) -> str:
        """A stable text identity, the proposals' dedupe key: ``<team>:<give ids>:<get ids>``, ids sorted."""
        give = ",".join(map(str, sorted(self.give)))
        get = ",".join(map(str, sorted(self.get)))
        return f"trade:{self.other_team_id}:{give}:{get}"


@dataclass(frozen=True, slots=True)
class TradeLegality:
    """Whether the league lets the deal happen. ``problems`` are why not (empty: legal); ``notes`` are things worth
    knowing that do not stop it. ``drops_ours`` / ``drops_theirs`` are the players each side would have to drop to make
    room, the cheapest first: legal, but a :class:`TradePayload` cannot carry them."""

    problems: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    drops_ours: tuple[int, ...] = ()
    drops_theirs: tuple[int, ...] = ()

    @property
    def legal(self) -> bool:
        return not self.problems


@dataclass(frozen=True, slots=True)
class SideImpact:
    """What the deal does to one team: its rest-of-season lineup value (in the model's unit) and, when the season could
    be simulated, its odds, before and after."""

    team_id: int
    name: str
    gives: tuple[int, ...]
    gets: tuple[int, ...]
    ros_before: float
    ros_after: float
    odds_before: TeamOdds | None = None
    odds_after: TeamOdds | None = None

    @property
    def delta_ros(self) -> float:
        return self.ros_after - self.ros_before

    @property
    def delta_title(self) -> float | None:
        return (
            None
            if self.odds_before is None or self.odds_after is None
            else (self.odds_after.title - self.odds_before.title)
        )

    @property
    def delta_playoffs(self) -> float | None:
        return (
            None
            if self.odds_before is None or self.odds_after is None
            else (self.odds_after.playoffs - self.odds_before.playoffs)
        )

    @property
    def delta_bye(self) -> float | None:
        return (
            None
            if self.odds_before is None or self.odds_after is None
            else (self.odds_after.bye - self.odds_before.bye)
        )


@dataclass(frozen=True, slots=True)
class Acceptance:
    """The other manager's likely answer: P(accept) and what it came from. ``receive_value`` is the market value of the
    package he would receive (what we give) and ``send_value`` of the one he sends; ``surplus`` is their relative gap,
    ``(receive - send) / (receive + send)``; ``need_gain`` is his change in rest-of-season lineup value in starter
    seasons; ``drops`` the players he would have to drop; ``basis`` the sources of the market values used."""

    p_accept: float
    receive_value: float
    send_value: float
    surplus: float
    need_gain: float
    drops: int = 0
    basis: tuple[str, ...] = ()

    def describe(self) -> str:
        return (
            f"P(accept) {self.p_accept:.0%}: market {self.receive_value:,.0f} to them for {self.send_value:,.0f} "
            f"from them ({self.surplus:+.0%}), their lineup {self.need_gain:+.2f} starter seasons"
        )


@dataclass(frozen=True, slots=True)
class TradeEvaluation:
    """Everything about one deal. ``ours`` and ``theirs`` are the two sides' impacts; ``score`` is our change in title
    odds (:data:`BASIS_TITLE`) or in rest-of-season value (:data:`BASIS_ROS`, without a simulation) times P(accept), or
    ``None`` for an illegal deal; ``recommendation`` is :data:`ACCEPT`, :data:`DECLINE` or :data:`COUNTER`; ``fit``
    are notes on positions and categories; ``unit`` names the value (points, G-score, z-score)."""

    spec: TradeSpec
    ours: SideImpact
    theirs: SideImpact
    legality: TradeLegality
    acceptance: Acceptance
    recommendation: str
    reason: str
    score: float | None
    score_basis: str
    unit: str
    fit: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    runs: int = 0
    seed: int = DEFAULT_SEED

    @property
    def legal(self) -> bool:
        return self.legality.legal

    @property
    def simulated(self) -> bool:
        return self.ours.odds_before is not None

    @property
    def gain(self) -> float:
        """What we gain, in the units of :attr:`score_basis`: title odds, or starter seasons of rest-of-season value."""
        title = self.ours.delta_title
        return title if title is not None else self.ours.delta_ros / max(self.scale, 1e-9)

    scale: float = 1.0
    """One starter's rest-of-season value, the unit :attr:`gain` is in when there is no simulation."""


# --- market values and P(accept) --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AcceptanceParams:
    """The weights of the P(accept) logistic. Defaults are a prior, not a fit: calibrate them as offers resolve.

    ``logit = market_weight * (surplus - margin) + need_weight * need_gain`` where ``surplus`` is the other manager's
    relative market surplus and ``need_gain`` his lineup gain in starter seasons (clamped to ±``need_cap``). A package's
    market value is its best piece plus ``depth_weight`` of every other piece (the best one is the one that starts for
    him), and each forced drop multiplies P by ``drop_friction``."""

    market_weight: float = 8.0
    margin: float = 0.05
    need_weight: float = 2.0
    need_cap: float = 3.0
    depth_weight: float = 0.5
    drop_friction: float = 0.9
    floor: float = 0.01
    ceiling: float = 0.99


DEFAULT_ACCEPTANCE: Final = AcceptanceParams()


def rank_type_for(settings: LeagueSettings) -> str:
    """ESPN's rank type for the league, derived from its settings: NFL ``SUPERFLEX`` when more than one starting slot
    takes a QB, else ``PPR`` when receptions score and ``STANDARD`` when they do not (FantasyCalc's own shape,
    :meth:`fm.sources.market.LeagueShape.from_settings`); NBA ``ROTO`` for a roto league, else ``STANDARD``."""
    if settings.game is Game.FFL:
        shape = LeagueShape.from_settings(settings)
        if shape.num_qbs >= 2:
            return "SUPERFLEX"
        return "PPR" if shape.ppr > 0 else "STANDARD"
    return "ROTO" if settings.scoring_type is ScoringType.ROTO else "STANDARD"


def rank_value(rank: int, size: int) -> float:
    """A rank's market value on FantasyCalc's scale: :data:`MARKET_TOP` for 1st, falling geometrically to
    :data:`MARKET_TAIL` at ``size``."""
    if rank < 1 or size < 1:
        raise ValueError(f"rank and size must be at least 1, got {rank} and {size}")
    decay = math.log(MARKET_TOP / MARKET_TAIL) / max(size - 1, 1)
    return MARKET_TOP * math.exp(-decay * (min(rank, size) - 1))


class MarketLike(Protocol):
    """What :func:`load_trade_context` needs of a market source: :class:`fm.sources.market.MarketSource` has it."""

    def market_values(self, settings: LeagueSettings, *, rank_type: str = ...) -> Fetched[dict[int, MarketValue]]: ...


@dataclass(frozen=True, slots=True)
class MarketBook:
    """What league-mates believe each valued player is worth, on one scale, and where each number came from."""

    values: Mapping[int, float]
    basis: Mapping[int, str]
    rank_type: str
    scale: float = 1.0
    """The mean market value of a starter-level player: the top ``teams x active slots`` rostered."""
    warnings: tuple[str, ...] = ()
    as_of: datetime | None = None

    def value(self, espn_id: int) -> float:
        return self.values.get(espn_id, 0.0)


def build_market_book(
    market: Mapping[int, MarketValue] | None,
    values: Mapping[int, float],
    *,
    rank_type: str,
    starters: Collection[int] = (),
    warnings: Iterable[str] = (),
    as_of: datetime | None = None,
) -> MarketBook:
    """Market values for every player in ``values`` (our model's rest-of-season value by ESPN id). A player's market
    value is FantasyCalc's trade value when it has one, else his ESPN rank under ``rank_type`` (draft rank, then
    the season's total ranking), else his rank by our own value: all on :func:`rank_value`'s scale, so a package can mix
    them. ``starters`` are the ESPN ids that set :attr:`MarketBook.scale`."""
    found = market or {}
    ordered = sorted(values, key=lambda espn_id: (-values[espn_id], espn_id))
    ours = {espn_id: position for position, espn_id in enumerate(ordered, start=1)}
    ranks = [
        rank for entry in found.values() for rank in (entry.espn_ranks.get(rank_type), entry.total_ranking) if rank
    ]
    size = max([*ranks, len(ordered), 2])
    book: dict[int, float] = {}
    basis: dict[int, str] = {}
    for espn_id in values:
        entry = found.get(espn_id)
        if entry is not None and entry.trade_value is not None and entry.trade_value > 0:
            book[espn_id], basis[espn_id] = float(entry.trade_value), FANTASYCALC
            continue
        rank = (entry.espn_ranks.get(rank_type) or entry.espn_rank or entry.total_ranking) if entry else None
        if rank:
            book[espn_id], basis[espn_id] = rank_value(rank, size), ESPN_RANK
        else:
            book[espn_id], basis[espn_id] = rank_value(ours[espn_id], max(len(ordered), 2)), ROS_RANK
    top = sorted((book[espn_id] for espn_id in starters if espn_id in book), reverse=True)
    scale = math.fsum(top) / len(top) if top else 1.0
    return MarketBook(
        MappingProxyType(book), MappingProxyType(basis), rank_type, max(scale, 1.0), tuple(warnings), as_of
    )


def package_value(values: Iterable[float], params: AcceptanceParams = DEFAULT_ACCEPTANCE) -> float:
    """What a package of players is worth to the manager receiving it: the best piece in full plus ``depth_weight`` of
    every other (only one of them takes the slot the best one does)."""
    ordered = sorted(values, reverse=True)
    if not ordered:
        return 0.0
    return ordered[0] + params.depth_weight * math.fsum(ordered[1:])


def acceptance_probability(
    book: MarketBook,
    give: Iterable[int],
    get: Iterable[int],
    *,
    need_gain: float,
    drops: int = 0,
    params: AcceptanceParams = DEFAULT_ACCEPTANCE,
) -> Acceptance:
    """P(accept) of the other manager for a deal in which he receives ``give`` (ours) and sends ``get`` (his): a
    logistic of his relative market surplus and his lineup need (see :class:`AcceptanceParams`)."""
    given, taken = tuple(give), tuple(get)
    receive = package_value((book.value(espn_id) for espn_id in given), params)
    send = package_value((book.value(espn_id) for espn_id in taken), params)
    total = receive + send
    surplus = (receive - send) / total if total > 0 else 0.0
    need = max(-params.need_cap, min(params.need_cap, need_gain))
    logit = params.market_weight * (surplus - params.margin) + params.need_weight * need
    probability = 1.0 / (1.0 + math.exp(-logit)) * params.drop_friction**drops
    bases = tuple(sorted({book.basis.get(espn_id, ROS_RANK) for espn_id in (*given, *taken)}))
    return Acceptance(
        p_accept=max(params.floor, min(params.ceiling, probability)),
        receive_value=receive,
        send_value=send,
        surplus=surplus,
        need_gain=need_gain,
        drops=drops,
        basis=bases,
    )


# --- the valuation models ---------------------------------------------------------------------------------------------


class TradeModel:
    """A league's rest-of-season values behind one interface, whatever the sport and scoring kind.

    ``values`` are every projected player's value in ``unit`` (a player nothing projects is not in it: his value is
    unknown, not zero), ``eligible`` the active slots each of them may fill, ``replacement`` the wire's best at each
    slot. :meth:`roster_value` is a roster's exact lineup value, :meth:`outlook` its matchup strength for the simulator
    (``None`` when it cannot be built), :meth:`profile` its per-category totals (category leagues).
    """

    unit: str = "points"
    simulates: bool = True
    """Whether :meth:`outlook` can feed the season simulator for this league."""

    def __init__(
        self,
        *,
        slots: Sequence[int],
        outlooks: Mapping[int, PlayerOutlook],
        values: Mapping[int, float],
        replacement: Mapping[int, float],
        warnings: Iterable[str] = (),
    ) -> None:
        self.slots = tuple(slots)
        self.outlooks: Mapping[int, PlayerOutlook] = MappingProxyType(dict(outlooks))
        self.values: Mapping[int, float] = MappingProxyType(dict(values))
        self.eligible: Mapping[int, frozenset[int]] = MappingProxyType(
            {espn_id: outlook.slots for espn_id, outlook in outlooks.items()}
        )
        self.replacement: Mapping[int, float] = MappingProxyType(dict(replacement))
        self.flexibility: Mapping[int, int] = MappingProxyType(
            {slot: sum(slot in slots_ for slots_ in self.eligible.values()) for slot in set(self.slots)}
        )
        self.warnings = tuple(warnings)

    def knows(self, espn_id: int) -> bool:
        """Whether the model has an outlook for the player (his value may still be unknown)."""
        return espn_id in self.outlooks

    def roster_value(self, roster: Iterable[int]) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def outlook(self, team_id: int, roster: Iterable[int]) -> TeamOutlook | CategoryOutlook | None:
        return None

    def profile(self, roster: Iterable[int]) -> Mapping[str, float]:
        """The roster's value in each category (empty in a points league)."""
        return {}

    def player_profile(self, espn_id: int) -> Mapping[str, float]:
        return {}


class PointsTradeModel(TradeModel):
    """NFL and NBA points leagues: :class:`fm.model.valuation.RosterValuer` over the outlooks' periods.

    ``spans`` maps each matchup period still to play to the model periods it spans (an NFL week, an NBA matchup week),
    for :meth:`outlook`; ``None`` when the schedule cannot say, which turns the simulation off."""

    unit = "points"

    def __init__(
        self,
        *,
        settings: LeagueSettings,
        outlooks: Mapping[int, PlayerOutlook],
        horizon: Horizon,
        wire: Iterable[int],
        spans: Mapping[int, tuple[int, ...]] | None,
        cv: float = DEFAULT_CV,
        warnings: Iterable[str] = (),
    ) -> None:
        slots = slot_instances(settings)
        labels = {slot.slot_id: slot.label for slot in settings.active_slots}
        self.valuer = RosterValuer(outlooks, roster=(), wire=wire, slots=slots, horizon=horizon, labels=labels)
        values = {espn_id: self.valuer.ros(espn_id) for espn_id, outlook in outlooks.items() if outlook.has_projection}
        replacement = {slot: level.value for slot, level in self.valuer.replacement.items()}
        super().__init__(slots=slots, outlooks=outlooks, values=values, replacement=replacement, warnings=warnings)
        self.spans = None if spans is None else MappingProxyType(dict(spans))
        self.simulates = spans is not None
        self.cv = cv

    def roster_value(self, roster: Iterable[int]) -> float:
        return self.valuer.value(frozenset(espn_id for espn_id in roster if self.knows(espn_id)))

    def outlook(self, team_id: int, roster: Iterable[int]) -> TeamOutlook | None:
        if self.spans is None:
            return None
        members = frozenset(espn_id for espn_id in roster if self.knows(espn_id))
        pool = self.valuer.fill_pool(())
        means: dict[int, float] = {}
        sds: dict[int, float] = {}
        for matchup, span in self.spans.items():
            totals = [self.valuer.week(members, period, pool) for period in span]
            means[matchup] = math.fsum(totals)
            sds[matchup] = math.sqrt(math.fsum((self.cv * total) ** 2 for total in totals))
        return TeamOutlook(team_id, MappingProxyType(means), MappingProxyType(sds))


class CategoryTradeModel(TradeModel):
    """NBA category leagues: a player's value is what his lines over the games left add to a team's categories
    against an empty slot; a roster's value is the best assignment of those to its slots."""

    def __init__(
        self,
        *,
        settings: LeagueSettings,
        model: CategoryModel,
        outlooks: Mapping[int, PlayerOutlook],
        contributions: Mapping[int, Mapping[str, float]],
        wire: Iterable[int],
        weekly_lines: Mapping[int, Mapping[int, Mapping[str, float]]] | None,
        cv: float = DEFAULT_CV,
        unit: str,
        warnings: Iterable[str] = (),
    ) -> None:
        slots = slot_instances(settings)
        values = {espn_id: outlook.expected(0) for espn_id, outlook in outlooks.items() if outlook.has_projection}
        wired = sorted(
            (espn_id for espn_id in wire if espn_id in values), key=lambda espn_id: (-values[espn_id], espn_id)
        )
        chosen: dict[int, None] = {}
        replacement: dict[int, float] = {}
        for slot, count in Counter(slots).items():
            fits = [espn_id for espn_id in wired if slot in outlooks[espn_id].slots]
            chosen.update(dict.fromkeys(fits[: count + 1]))
            replacement[slot] = max(values[fits[0]], 0.0) if fits else 0.0
        super().__init__(slots=slots, outlooks=outlooks, values=values, replacement=replacement, warnings=warnings)
        self.settings = settings
        self.model = model
        self.contributions = MappingProxyType({espn_id: dict(scores) for espn_id, scores in contributions.items()})
        self.fill = tuple(sorted(chosen))
        self.weekly_lines = None if weekly_lines is None else MappingProxyType(dict(weekly_lines))
        self.simulates = weekly_lines is not None and settings.scoring_type in MATCHUP_SCORING_TYPES
        self.cv = cv
        self.unit = unit

    def _lineup(self, roster: Iterable[int]) -> tuple[float, tuple[int, ...]]:
        members = [self.outlooks[espn_id] for espn_id in sorted(set(roster)) if self.knows(espn_id)]
        fill = [self.outlooks[espn_id] for espn_id in self.fill]
        lineup = start_value(members, self.slots, 0, fill=fill)
        return lineup.total, tuple(espn_id for _, espn_id in lineup.starters)

    def roster_value(self, roster: Iterable[int]) -> float:
        return self._lineup(roster)[0]

    def profile(self, roster: Iterable[int]) -> Mapping[str, float]:
        totals: dict[str, float] = {category: 0.0 for category in self.model.categories}
        for espn_id in self._lineup(roster)[1]:
            for category, score in self.contributions.get(espn_id, {}).items():
                totals[category] += score
        return totals

    def player_profile(self, espn_id: int) -> Mapping[str, float]:
        return self.contributions.get(espn_id, {})

    def outlook(self, team_id: int, roster: Iterable[int]) -> CategoryOutlook | None:
        if self.weekly_lines is None or not self.simulates:
            return None
        starters = self._lineup(roster)[1]
        lines = {
            matchup: [week[espn_id] for espn_id in starters if espn_id in week]
            for matchup, week in self.weekly_lines.items()
        }
        return category_outlook(team_id, lines, self.settings, cv=self.cv)


def quick_value(model: TradeModel, roster: Iterable[int]) -> float:
    """A greedy lineup value for screening: the roster's players, best first, each into the most specific open slot he
    may fill (the slot the fewest players could fill), then every slot still open at the wire's replacement level. It
    is the exact value's cheap stand-in (:meth:`TradeModel.roster_value`): same units, no weeks, no assignment."""
    open_slots = Counter(model.slots)
    values = model.values
    total = 0.0
    ranked = sorted(
        (espn_id for espn_id in set(roster) if values.get(espn_id, 0.0) > 0),
        key=lambda espn_id: (-values[espn_id], espn_id),
    )
    for espn_id in ranked:
        fits = [slot for slot in model.eligible.get(espn_id, frozenset()) if open_slots[slot] > 0]
        if not fits:
            continue
        slot = min(fits, key=lambda candidate: (model.flexibility.get(candidate, 0), candidate))
        open_slots[slot] -= 1
        total += values[espn_id]
    return total + math.fsum(max(model.replacement.get(slot, 0.0), 0.0) * count for slot, count in open_slots.items())


# --- the league -------------------------------------------------------------------------------------------------------


@dataclass
class _Cache:
    """What the context computes once and reuses: outlooks by (team, roster), odds by (runs, seed)."""

    outlooks: dict[tuple[int, frozenset[int]], TeamOutlook | CategoryOutlook | None] = field(default_factory=dict)
    odds: dict[tuple[int, int, frozenset[tuple[int, frozenset[int]]]], SeasonOdds | None] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class TradeContext:
    """Everything one league's trades are judged on, read once: the model, every team's roster, the players, the
    market, the schedule of matchups and the guardrails. Build it with :func:`load_trade_context`.

    ``rosters`` hold every player on each team (IR included); ``ir`` those in the IR slot; ``playing`` the ones that can
    start (valued by the model, off IR). ``p_active`` is the engine's chance each player plays now, without Claude's
    signals. ``untouchables`` are ours that policy never lets go."""

    league: LeagueRow
    settings: LeagueSettings
    now: datetime
    period: int
    team_id: int
    team_names: Mapping[int, str]
    rosters: Mapping[int, frozenset[int]]
    ir: Mapping[int, frozenset[int]]
    locked: frozenset[int]
    players: Mapping[int, PlayerRow]
    model: TradeModel
    market: MarketBook
    matchups: MatchupsView | None = None
    matchup_period: int | None = None
    schedule: ScheduleLike | None = None
    p_active: Mapping[int, float] = field(default_factory=dict)
    untouchables: Mapping[int, str] = field(default_factory=dict)
    acceptance: AcceptanceParams = DEFAULT_ACCEPTANCE
    warnings: tuple[str, ...] = ()
    cache: _Cache = field(default_factory=_Cache, repr=False, compare=False)

    @property
    def other_teams(self) -> tuple[int, ...]:
        return tuple(sorted(team for team in self.rosters if team != self.team_id))

    def name(self, espn_id: int) -> str:
        player = self.players.get(espn_id)
        return player.full_name if player is not None else f"player {espn_id}"

    def team_name(self, team_id: int) -> str:
        return self.team_names.get(team_id, f"team {team_id}")

    def playing(self, team_id: int) -> frozenset[int]:
        """The players of a team who can start: valued by the model and not in the IR slot."""
        return frozenset(
            espn_id
            for espn_id in self.rosters.get(team_id, frozenset())
            if espn_id not in self.ir.get(team_id, frozenset()) and self.model.knows(espn_id)
        )

    def owner(self, espn_id: int) -> int | None:
        return next((team for team, roster in self.rosters.items() if espn_id in roster), None)

    @property
    def scale(self) -> float:
        """One starter's rest-of-season value: the mean value of the best ``teams x active slots`` rostered players."""
        rostered = {espn_id for roster in self.rosters.values() for espn_id in roster}
        values = sorted(
            (self.model.values[espn_id] for espn_id in rostered if espn_id in self.model.values), reverse=True
        )
        top = values[: max(1, len(self.rosters) * self.settings.active_slot_count)]
        scale = math.fsum(top) / len(top) if top else 0.0
        return scale if scale > 0 else 1.0


def engine_p_active(
    store: Store,
    players: Iterable[PlayerRow],
    *,
    season: int,
    period: int,
    now: datetime,
    schedule: ScheduleLike | None = None,
) -> dict[int, float]:
    """The engine's chance each player is active in ``period``, without Claude's news signals (CLAUDE.md): the stored
    availability row's ``inputs["news"]["before"]`` (the number before the news step), else its ``p_active`` when the
    row carries no news step, else a bare :func:`fm.model.availability.assess`. A player the engine cannot assess is
    left out."""
    rows = tuple(players)
    found: dict[int, float] = {}
    sports: set[Sport] = {player.sport for player in rows}
    stored = {
        (row.sport, row.espn_id): row
        for sport in sports
        for row in store.availability.for_period(sport, season, period)
    }
    for player in rows:
        row = stored.get((player.sport, player.espn_id))
        if row is not None:
            found[player.espn_id] = _before_news(row.inputs, row.p_active)
            continue
        try:
            found[player.espn_id] = assess(
                player, season=season, scoring_period=period, as_of=now, schedule=schedule
            ).p_active
        except ValueError:
            continue
    return found


def _before_news(inputs: Mapping[str, Any], stored: float) -> float:
    """``inputs["news"]["before"]`` when the row kept it as a probability, else ``stored``."""
    news = inputs.get("news")
    before = news.get("before") if isinstance(news, dict) else None
    if isinstance(before, int | float) and not isinstance(before, bool) and 0.0 <= before <= 1.0:
        return float(before)
    return stored


def _spans_for(
    settings: LeagueSettings, current_matchup: int | None, *, last: int | None = None
) -> dict[int, tuple[int, ...]] | None:
    """The scoring periods of each matchup period from the current one on, or ``None`` when they cannot be told."""
    days = league_matchup_days(settings)
    if days is None or current_matchup is None:
        return None
    return {matchup: span for matchup, span in sorted(days.items()) if matchup >= current_matchup and span}


def load_trade_context(
    store: Store,
    league: LeagueRow,
    *,
    now: datetime,
    config: Config | None = None,
    schedule: ScheduleLike | None = None,
    matchups: MatchupsView | None = None,
    market: MarketLike | None = None,
    weights: BlendWeights | None = None,
    sources: ProjectionSourceRegistry | None = None,
    playoff_weight: float = DEFAULT_PLAYOFF_WEIGHT,
    cv: float = DEFAULT_CV,
    acceptance: AcceptanceParams = DEFAULT_ACCEPTANCE,
) -> TradeContext:
    """Read a league's trade context from what ``fm sync`` stored (``now`` must be aware).

    ``schedule`` is the sport's pro schedule (byes, lock cutoffs; the NBA needs it to count games), ``matchups`` the
    league's ``mMatchup`` read (without it a deal is judged on rest-of-season value alone), ``market`` a market source
    (:class:`fm.sources.market.MarketSource`; without one P(accept) uses our own ranks), ``config`` supplies the
    untouchables. ``weights`` and ``sources`` are the projection blend's, as in :func:`fm.decide.rankings.rank_league`.
    Raises :class:`TradeError` for a league that is not synced or cannot be valued.
    """
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError(f"now must be an aware datetime, got a naive {now.isoformat()}")
    try:
        settings = league_settings(store, league)
    except ValuationError as exc:
        raise TradeError(str(exc)) from exc
    period = store.rosters.latest_period(league.row_id)
    if period is None:
        raise TradeError(f"league {league.key!r} has no roster snapshot; run fm sync")
    entries = store.rosters.league(league.row_id, period)
    rosters: dict[int, set[int]] = {}
    ir: dict[int, set[int]] = {}
    for entry in entries:
        rosters.setdefault(entry.team_id, set()).add(entry.espn_id)
        if entry.lineup_slot_id == settings.ids.ir_slot:
            ir.setdefault(entry.team_id, set()).add(entry.espn_id)
    if league.team_id not in rosters:
        raise TradeError(f"league {league.key!r}: team {league.team_id} has no roster in scoring period {period}")
    names = {team.team_id: team.name for team in store.teams.for_league(league.row_id)}
    current_matchup = _current_matchup(settings, matchups, period)
    warnings: list[str] = []
    try:
        if league.sport == "nfl":
            model, players = _nfl_model(
                store, league, settings, period, now, schedule, weights, playoff_weight, cv, current_matchup, warnings
            )
        else:
            model, players = _nba_model(
                store, league, settings, period, now, schedule, weights, sources, playoff_weight, cv, rosters, warnings
            )
    except ValuationError as exc:
        raise TradeError(str(exc)) from exc
    warnings.extend(model.warnings)
    rostered = {espn_id for roster in rosters.values() for espn_id in roster}
    book = _market_book(market, settings, model, players, rostered, rosters, warnings)
    held = tuple(sorted(rosters[league.team_id]))
    untouchables: dict[int, str] = {}
    if config is not None:
        try:
            policy = config.league(league.key).policy
        except KeyError:
            policy = None
        if policy is not None:
            untouchables = find_untouchables(store, league.sport, policy, held)
    unmodeled = sorted(rostered - set(model.outlooks))
    if unmodeled:
        warnings.append(
            f"{len(unmodeled)} rostered players have no players row and are left out of every lineup: {unmodeled[:5]}"
        )
    schedule_note = _lock_note(settings, schedule)
    if schedule_note:
        warnings.append(schedule_note)
    if matchups is None:
        warnings.append("no mMatchup schedule given: deals are judged on rest-of-season value, not title odds")
    p_active = engine_p_active(
        store,
        [players[espn_id] for espn_id in sorted(rostered) if espn_id in players],
        season=league.season,
        period=period,
        now=now,
        schedule=schedule,
    )
    return TradeContext(
        league=league,
        settings=settings,
        now=now,
        period=period,
        team_id=league.team_id,
        team_names=MappingProxyType(names),
        rosters=MappingProxyType({team: frozenset(roster) for team, roster in sorted(rosters.items())}),
        ir=MappingProxyType({team: frozenset(roster) for team, roster in ir.items()}),
        locked=frozenset(entry.espn_id for entry in entries if entry.lineup_locked),
        players=MappingProxyType(dict(players)),
        model=model,
        market=book,
        matchups=matchups,
        matchup_period=current_matchup,
        schedule=schedule,
        p_active=MappingProxyType(p_active),
        untouchables=MappingProxyType(untouchables),
        acceptance=acceptance,
        warnings=tuple(warnings),
    )


def _current_matchup(settings: LeagueSettings, matchups: MatchupsView | None, period: int) -> int | None:
    if matchups is not None and matchups.status is not None and matchups.status.current_matchup_period is not None:
        return matchups.status.current_matchup_period
    if settings.current_matchup_period is not None:
        return settings.current_matchup_period
    return settings.schedule.matchup_period_for(period) or matchup_period_of(settings, period)


def _lock_note(settings: LeagueSettings, schedule: ScheduleLike | None) -> str | None:
    if settings.roster_lock_type is LockType.UNKNOWN:
        return (
            f"roster lock type {settings.roster_lock_type_raw!r} is unknown: players whose games have started are not "
            "checked for locks (the executor re-checks the live league)"
        )
    if schedule is None:
        return "no pro schedule: lock cutoffs are not checked, only the lineup locks of the last sync"
    return None


def _market_book(
    market: MarketLike | None,
    settings: LeagueSettings,
    model: TradeModel,
    players: Mapping[int, PlayerRow],
    rostered: set[int],
    rosters: Mapping[int, set[int]],
    warnings: list[str],
) -> MarketBook:
    rank_type = rank_type_for(settings)
    found: Mapping[int, MarketValue] | None = None
    notes: list[str] = []
    as_of: datetime | None = None
    if market is None:
        notes.append("no market source: P(accept) uses our own rest-of-season ranks")
    else:
        fetched = market.market_values(settings, rank_type=rank_type)
        found, as_of = fetched.data, fetched.as_of
        notes.extend(fetched.warnings)
        if fetched.degraded or not fetched.data:
            notes.append("market values are unavailable: P(accept) uses our own rest-of-season ranks")
        elif fetched.stale:
            notes.append("market values are stale (the refresh failed)")
    warnings.extend(notes)
    valued = {espn_id: value for espn_id, value in model.values.items() if espn_id in players}
    scale_ids = sorted(
        (espn_id for espn_id in rostered if espn_id in valued), key=lambda espn_id: (-valued[espn_id], espn_id)
    )[: max(1, len(rosters) * settings.active_slot_count)]
    return build_market_book(found, valued, rank_type=rank_type, starters=scale_ids, warnings=notes, as_of=as_of)


# --- the NFL ----------------------------------------------------------------------------------------------------------


def _nfl_model(
    store: Store,
    league: LeagueRow,
    settings: LeagueSettings,
    period: int,
    now: datetime,
    schedule: ScheduleLike | None,
    weights: BlendWeights | None,
    playoff_weight: float,
    cv: float,
    current_matchup: int | None,
    warnings: list[str],
) -> tuple[TradeModel, dict[int, PlayerRow]]:
    if settings.game is not Game.FFL:
        raise TradeError(f"league {league.key!r} is {settings.game.value}, not an NFL (ffl) league")
    valuation = load_valuation(
        store,
        league,
        now=now,
        settings=settings,
        schedule=schedule,
        weights=weights,
        playoff_weight=playoff_weight,
        include_rostered=True,
    )
    warnings.extend(valuation.warnings)
    spans = _spans_for(settings, current_matchup)
    if spans is None:
        warnings.append("the league's matchup weeks cannot be told from its settings: no season simulation")
    model = PointsTradeModel(
        settings=settings,
        outlooks=valuation.outlooks,
        horizon=valuation.horizon,
        wire=valuation.wire,
        spans=spans,
        cv=cv,
    )
    return model, dict(valuation.players)


# --- the NBA ----------------------------------------------------------------------------------------------------------


def _nba_model(
    store: Store,
    league: LeagueRow,
    settings: LeagueSettings,
    period: int,
    now: datetime,
    schedule: ScheduleLike | None,
    weights: BlendWeights | None,
    sources: ProjectionSourceRegistry | None,
    playoff_weight: float,
    cv: float,
    rosters: Mapping[int, set[int]],
    warnings: list[str],
) -> tuple[TradeModel, dict[int, PlayerRow]]:
    if settings.game is not Game.FBA:
        raise TradeError(f"league {league.key!r} is {settings.game.value}, not an NBA (fba) league")
    if schedule is None:
        raise TradeError(f"league {league.key!r}: the NBA's games are counted from the pro schedule, which is missing")
    last = last_scoring_period(settings)
    if last is None:
        raise TradeError(f"league {league.key!r}: the season's last scoring period is unknown")
    rostered = {espn_id for roster in rosters.values() for espn_id in roster}
    espn_rows = store.projections.for_period(league.sport, league.season, SEASON_PERIOD, source=ESPN)
    wire = frozenset(row.espn_id for row in espn_rows) - rostered
    players = {player.espn_id: player for player in store.players.many(league.sport, rostered | wire)}
    blended = blend_day(
        store,
        league.season,
        period,
        weights=weights if weights is not None else BlendWeights.load(),
        sources=sources if sources is not None else source_registry,
        save=False,
    )
    warnings.extend(blended.warnings)
    lines = {row.espn_id: dict(row.stats) for row in blended.rows if row.espn_id in players}
    if not lines:
        raise TradeError(f"league {league.key!r}: no NBA projections are stored; run fm sync")
    left = tuple(range(period, last + 1))
    listed = set(schedule.scoring_periods)
    if sum(day in listed for day in left) < len(left):
        warnings.append(
            f"the pro schedule lists games for {sum(day in listed for day in left)} of the {len(left)} days left; "
            "values count the scheduled games only"
        )
    current = _current_matchup(settings, None, period)
    days = league_matchup_days(settings)
    weeks: dict[int, tuple[int, ...]] | None = None
    if days is not None and current is not None:
        weeks = {
            matchup: tuple(day for day in span if day >= period)
            for matchup, span in sorted(days.items())
            if matchup >= current
        }
        weeks = {matchup: span for matchup, span in weeks.items() if span} or None
    if weeks is None:
        warnings.append(
            "the league's matchup weeks cannot be resolved (no season calendar): the whole season is one period and "
            "there is no season simulation"
        )
    wire_ids = frozenset(espn_id for espn_id in wire if espn_id in lines)
    if settings.is_points:
        return _nba_points(settings, players, lines, wire_ids, schedule, left, weeks, playoff_weight, cv), players
    return _nba_categories(settings, players, lines, wire_ids, schedule, left, weeks, cv, warnings), players


def _nba_points(
    settings: LeagueSettings,
    players: Mapping[int, PlayerRow],
    lines: Mapping[int, Mapping[str, float]],
    wire: frozenset[int],
    schedule: ScheduleLike,
    left: tuple[int, ...],
    weeks: Mapping[int, tuple[int, ...]] | None,
    playoff_weight: float,
    cv: float,
) -> TradeModel:
    scorer = Scorer(settings)
    spans = dict(weeks) if weeks is not None else {0: left}
    outlooks: dict[int, PlayerOutlook] = {}
    for espn_id, line in lines.items():
        player = players[espn_id]
        per_game = scorer.points(line, position=player.position)
        weekly = {
            matchup: (per_game * team_games(schedule, player.pro_team_id, span) if player.active else 0.0)
            for matchup, span in spans.items()
        }
        outlooks[espn_id] = PlayerOutlook(
            espn_id=espn_id,
            name=player.full_name,
            position=player.position,
            pro_team_id=player.pro_team_id,
            slots=eligible_active_slots(player, settings),
            weekly=MappingProxyType(weekly),
            per_game=per_game,
            games=float(team_games(schedule, player.pro_team_id, left)),
            basis=BASIS_SEASON,
        )
    playoffs = (
        frozenset(matchup for matchup in spans if settings.schedule.is_playoff(matchup)) if weeks else frozenset()
    )
    horizon = Horizon(
        current=min(spans),
        periods=tuple(sorted(spans)),
        weights=MappingProxyType({matchup: playoff_weight if matchup in playoffs else 1.0 for matchup in spans}),
        playoff_periods=playoffs,
        playoffs_known=weeks is not None,
        playoff_weight=playoff_weight,
    )
    return PointsTradeModel(
        settings=settings,
        outlooks=outlooks,
        horizon=horizon,
        wire=wire,
        spans=({matchup: (matchup,) for matchup in spans} if weeks is not None else None),
        cv=cv,
    )


def _nba_categories(
    settings: LeagueSettings,
    players: Mapping[int, PlayerRow],
    lines: Mapping[int, Mapping[str, float]],
    wire: frozenset[int],
    schedule: ScheduleLike,
    left: tuple[int, ...],
    weeks: Mapping[int, tuple[int, ...]] | None,
    cv: float,
    warnings: list[str],
) -> TradeModel:
    scaled = period_lines(lines, players.values(), schedule, left)
    if len(scaled) < 2:
        raise TradeError(f"league {settings.league_id}: category scores need at least two projected players")
    model = fit_categories(scaled, settings)
    if model.metric.value == "g" and not any(entry.tau > 0 for entry in model.stats):
        warnings.append("no game history to estimate the G-score's tau from, so G-scores equal z-scores")
    contributions = {espn_id: model.contributions(line) for espn_id, line in scaled.items()}
    outlooks: dict[int, PlayerOutlook] = {}
    for espn_id, scores in contributions.items():
        player = players[espn_id]
        total = math.fsum(scores.values()) if player.active else 0.0
        outlooks[espn_id] = PlayerOutlook(
            espn_id=espn_id,
            name=player.full_name,
            position=player.position,
            pro_team_id=player.pro_team_id,
            slots=eligible_active_slots(player, settings),
            weekly=MappingProxyType({0: total}),
            games=scaled[espn_id].get("GP"),
            basis=BASIS_SEASON,
        )
    weekly_lines: dict[int, dict[int, dict[str, float]]] | None = None
    if weeks is not None:
        weekly_lines = {
            matchup: period_lines(lines, players.values(), schedule, span) for matchup, span in weeks.items()
        }
    trade_model = CategoryTradeModel(
        settings=settings,
        model=model,
        outlooks=outlooks,
        contributions={espn_id: scores for espn_id, scores in contributions.items() if players[espn_id].active},
        wire=wire,
        weekly_lines=weekly_lines,
        cv=cv,
        unit="G-score" if model.metric.value == "g" else "z-score",
        warnings=model.warnings,
    )
    if weekly_lines is not None and settings.scoring_type not in MATCHUP_SCORING_TYPES:
        warnings.append(f"{settings.scoring_type.value} has no head-to-head matchups to simulate")
    return trade_model


# --- legality ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _After:
    """A team's roster after a deal: everyone, who sits in IR, and who can start."""

    everyone: frozenset[int]
    ir: frozenset[int]
    playing: frozenset[int]

    @property
    def held(self) -> int:
        """Players outside the IR slot: what the roster size limits."""
        return len(self.everyone) - len(self.ir)


def _after(
    ctx: TradeContext, team_id: int, out: Collection[int], incoming: Sequence[int], dropped: Collection[int] = ()
) -> _After:
    """``team_id``'s roster once ``out`` and ``dropped`` leave and ``incoming`` arrive. A player who was in someone's IR
    slot goes to an IR slot here when the team has one free, else he takes a roster spot."""
    gone = set(out) | set(dropped)
    everyone = (ctx.rosters.get(team_id, frozenset()) - gone) | set(incoming)
    ir_kept = {espn_id for espn_id in ctx.ir.get(team_id, frozenset()) if espn_id not in gone}
    room = ctx.settings.ir_count - len(ir_kept)
    in_ir_elsewhere = {espn_id for held in ctx.ir.values() for espn_id in held}
    for espn_id in sorted(incoming):
        if room > 0 and espn_id in in_ir_elsewhere:
            ir_kept.add(espn_id)
            room -= 1
    ir = frozenset(ir_kept)
    playing = frozenset(espn_id for espn_id in everyone if espn_id not in ir and ctx.model.knows(espn_id))
    return _After(frozenset(everyone), ir, playing)


def _position_problems(ctx: TradeContext, who: str, before: frozenset[int], after: _After) -> list[str]:
    problems: list[str] = []
    counts_before = _position_counts(ctx, before)
    counts_after = _position_counts(ctx, after.everyone - after.ir)
    for position, count in sorted(counts_after.items()):
        limit = ctx.settings.position_limit(position)
        if limit is not None and count > limit and count > counts_before.get(position, 0):
            label = ctx.settings.ids.position_label(position)
            problems.append(f"{who} would hold {count} {label} and the league allows {limit}")
    return problems


def _position_counts(ctx: TradeContext, roster: Collection[int]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for espn_id in roster:
        player = ctx.players.get(espn_id)
        if player is not None and player.default_position_id is not None:
            counts[player.default_position_id] += 1
    return counts


def _locked_players(ctx: TradeContext, ids: Iterable[int]) -> dict[int, str]:
    """Players in a deal whose roster lock has closed: the league's ``rosterLocktimeType`` applied to the pro schedule
    and, for individual-game locks, the lineup locks the last sync read. ``{}`` where the lock cannot be told."""
    settings = ctx.settings
    locked: dict[int, str] = {}
    plugin = plugin_for(settings.game)
    for espn_id in ids:
        player = ctx.players.get(espn_id)
        if settings.roster_lock_type is LockType.INDIVIDUAL_GAME and espn_id in ctx.locked:
            locked[espn_id] = "his game has started"
            continue
        if (
            ctx.schedule is None
            or player is None
            or player.pro_team_id is None
            or settings.roster_lock_type is LockType.UNKNOWN
        ):
            continue
        try:
            cutoff = plugin.transaction_cutoff(
                player.pro_team_id, ctx.period, ctx.schedule, lock_type=settings.roster_lock_type
            )
        except ValueError:
            continue
        if cutoff is not None and cutoff <= ctx.now:
            locked[espn_id] = f"transactions closed at {cutoff:%Y-%m-%d %H:%MZ}"
    return locked


def _droppable(ctx: TradeContext, team_id: int, after: _After, keep: Collection[int], count: int) -> tuple[int, ...]:
    """The ``count`` cheapest players ``team_id`` could drop to make room: off IR, not in the deal, not untouchable
    (ours), not locked; players the model cannot value first (their value is unknown), then by value."""
    barred = set(keep) | ctx.locked
    if team_id == ctx.team_id:
        barred |= set(ctx.untouchables)
    options = sorted(
        (espn_id for espn_id in after.everyone - after.ir if espn_id not in barred),
        key=lambda espn_id: (ctx.model.values.get(espn_id, -1.0), espn_id),
    )
    return tuple(options[:count])


def check_legality(ctx: TradeContext, spec: TradeSpec) -> TradeLegality:
    """Whether the league lets ``spec`` happen, from its settings (see the module docs). Never raises for a rule that
    fails: the reasons are in :attr:`TradeLegality.problems`. Raises :class:`TradeError` for a team or player that is
    not where the deal says."""
    problems: list[str] = []
    notes: list[str] = []
    other = spec.other_team_id
    if other == ctx.team_id or other not in ctx.rosters:
        raise TradeError(f"team {other} is not another team in league {ctx.league.key!r}")
    ours, theirs = ctx.rosters[ctx.team_id], ctx.rosters[other]
    missing_ours = sorted(set(spec.give) - ours)
    missing_theirs = sorted(set(spec.get) - theirs)
    if missing_ours:
        raise TradeError(f"{_names(ctx, missing_ours)} are not on our roster")
    if missing_theirs:
        raise TradeError(f"{_names(ctx, missing_theirs)} are not on {ctx.team_name(other)}'s roster")
    deadline = ctx.settings.trade.deadline
    if not ctx.settings.trade.is_open(ctx.now) and deadline is not None:
        problems.append(f"the league's trade deadline {deadline:%Y-%m-%d %H:%M} UTC has passed")
    for espn_id in spec.give:
        if espn_id in ctx.untouchables:
            problems.append(f"{ctx.name(espn_id)} is untouchable (policy)")
    for espn_id, why in _locked_players(ctx, (*spec.give, *spec.get)).items():
        problems.append(f"{ctx.name(espn_id)} is locked: {why}")
    after_ours = _after(ctx, ctx.team_id, spec.give, spec.get)
    after_theirs = _after(ctx, other, spec.get, spec.give)
    drops_ours = drops_theirs = ()
    limit = ctx.settings.roster_size
    for who, team, after, out_ids in (
        ("we", ctx.team_id, after_ours, spec.give),
        ("they", other, after_theirs, spec.get),
    ):
        over = after.held - limit
        if over <= 0:
            continue
        keep = {*spec.give, *spec.get}
        chosen = _droppable(ctx, team, after, keep, over)
        name = "we" if who == "we" else ctx.team_name(other)
        if len(chosen) < over:
            problems.append(
                f"{name} would hold {after.held} players outside IR and the league allows {limit}, "
                "and nobody could be dropped to make room"
            )
            continue
        notes.append(f"{name} must drop {_names(ctx, chosen)} to make room ({after.held} of {limit} roster spots)")
        if who == "we":
            drops_ours = chosen
        else:
            drops_theirs = chosen
        del out_ids
    problems.extend(_position_problems(ctx, "we", ours - ctx.ir.get(ctx.team_id, frozenset()), after_ours))
    problems.extend(
        _position_problems(ctx, ctx.team_name(other), theirs - ctx.ir.get(other, frozenset()), after_theirs)
    )
    for espn_id in (*spec.give, *spec.get):
        if espn_id in {held for team in ctx.ir.values() for held in team}:
            notes.append(f"{ctx.name(espn_id)} is in an IR slot")
    return TradeLegality(tuple(problems), tuple(notes), drops_ours, drops_theirs)


def _names(ctx: TradeContext, ids: Iterable[int]) -> str:
    return ", ".join(ctx.name(espn_id) for espn_id in ids)


# --- evaluating one deal ----------------------------------------------------------------------------------------------


def _title_odds(
    ctx: TradeContext, overrides: Mapping[int, frozenset[int]], *, runs: int, seed: int
) -> SeasonOdds | None:
    """The season's odds with ``overrides`` (team id -> roster that plays) in place of the synced rosters; ``None`` when
    the league cannot be simulated (the reason is kept in the context's notes)."""
    if ctx.matchups is None or not ctx.model.simulates:
        return None
    key = (runs, seed, frozenset(overrides.items()))
    if key in ctx.cache.odds:
        return ctx.cache.odds[key]
    outlooks: dict[int, TeamOutlook | CategoryOutlook] = {}
    for team in ctx.rosters:
        roster = overrides.get(team, ctx.playing(team))
        slot = (team, roster)
        if slot not in ctx.cache.outlooks:
            ctx.cache.outlooks[slot] = ctx.model.outlook(team, roster)
        built = ctx.cache.outlooks[slot]
        if built is None:
            return _no_odds(ctx, key, f"no outlook could be built for {ctx.team_name(team)}")
        outlooks[team] = built
    try:
        odds = simulate_season(
            ctx.settings, ctx.matchups, outlooks, seed=seed, runs=runs, current_matchup_period=ctx.matchup_period
        )
    except SimulationError as exc:
        return _no_odds(ctx, key, f"the season cannot be simulated: {exc}")
    ctx.cache.odds[key] = odds
    return odds


def _no_odds(ctx: TradeContext, key: tuple[int, int, frozenset[tuple[int, frozenset[int]]]], note: str) -> None:
    if note not in ctx.cache.notes:
        ctx.cache.notes.append(note)
    ctx.cache.odds[key] = None


def _labels_held(ctx: TradeContext, roster: Collection[int]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for espn_id in roster:
        player = ctx.players.get(espn_id)
        if player is not None and player.position:
            counts[player.position] += 1
    return counts


def _need_notes(ctx: TradeContext, spec: TradeSpec, ours: _After, theirs: _After) -> list[str]:
    """Fit notes: where a side's depth at a position changes against the league's single-position slots (a position
    left short, or with no cover for its one starter), and, in a category league, which categories the deal moves for
    us."""
    notes: list[str] = []
    plugin = plugin_for(ctx.settings.game)
    required: Counter[str] = Counter()
    for slot in slot_instances(ctx.settings):
        positions = plugin.positions_for_slot(slot)
        if len(positions) == 1:
            required[next(iter(positions))] += 1
    sides = (
        ("we", ctx.playing(ctx.team_id), ours),
        (ctx.team_name(spec.other_team_id), ctx.playing(spec.other_team_id), theirs),
    )
    for who, before_roster, after in sides:
        before, now = _labels_held(ctx, before_roster), _labels_held(ctx, after.playing)
        for position, need in sorted(required.items()):
            if now[position] == before[position]:
                continue
            if now[position] < need:
                notes.append(f"{who} would have {now[position]} {position} for {need} starting slot(s)")
            elif now[position] == need and before[position] > need:
                notes.append(f"{who} would have no {position} cover")
            elif now[position] > before[position] + 1:
                notes.append(f"{who} would stack {now[position]} {position}")
    base = ctx.model.profile(ctx.playing(ctx.team_id))
    if base:
        change = {category: score - base[category] for category, score in ctx.model.profile(ours.playing).items()}
        ups = [
            f"{category} {delta:+.2f}"
            for category, delta in sorted(change.items(), key=lambda item: -item[1])
            if delta > 0.05
        ]
        downs = [
            f"{category} {delta:+.2f}"
            for category, delta in sorted(change.items(), key=lambda item: item[1])
            if delta < -0.05
        ]
        if ups:
            notes.append("categories up: " + ", ".join(ups))
        if downs:
            notes.append("categories down: " + ", ".join(downs))
    return notes


def evaluate_trade(
    ctx: TradeContext,
    spec: TradeSpec,
    *,
    runs: int = 10_000,
    seed: int = DEFAULT_SEED,
    simulate: bool = True,
) -> TradeEvaluation:
    """Judge one deal: both sides' change in rest-of-season lineup value, their change in title odds (a season
    simulation of ``runs`` seasons from ``seed`` with the two rosters swapped, against the same simulation without the
    deal; ``simulate=False`` skips it), legality, P(accept), and a recommendation.

    A deal that is not legal is still valued, with ``score`` ``None`` and a decline. The recommendation is
    :data:`ACCEPT` when we gain more than the edge (:data:`EDGE_TITLE` of title odds, :data:`EDGE_ROS` of a starter
    season without a simulation), :data:`COUNTER` when we do not but P(accept) is at least :data:`COUNTER_ACCEPT` (he
    likes it enough to give something back), and :data:`DECLINE` otherwise. Raises :class:`TradeError` for a player
    who is not where the deal says.
    """
    legality = check_legality(ctx, spec)
    other = spec.other_team_id
    ours_after = _after(ctx, ctx.team_id, spec.give, spec.get, legality.drops_ours)
    theirs_after = _after(ctx, other, spec.get, spec.give, legality.drops_theirs)
    ours_before, theirs_before = ctx.playing(ctx.team_id), ctx.playing(other)
    model = ctx.model
    ros = {
        "ours": (model.roster_value(ours_before), model.roster_value(ours_after.playing)),
        "theirs": (model.roster_value(theirs_before), model.roster_value(theirs_after.playing)),
    }
    base = after = None
    if simulate:
        base = _title_odds(ctx, {}, runs=runs, seed=seed)
        after = (
            _title_odds(ctx, {ctx.team_id: ours_after.playing, other: theirs_after.playing}, runs=runs, seed=seed)
            if base is not None
            else None
        )
        if after is None:
            base = None
    mine = SideImpact(
        ctx.team_id,
        ctx.team_name(ctx.team_id),
        spec.give,
        spec.get,
        *ros["ours"],
        None if base is None else base.team(ctx.team_id),
        None if after is None else after.team(ctx.team_id),
    )
    theirs = SideImpact(
        other,
        ctx.team_name(other),
        spec.get,
        spec.give,
        *ros["theirs"],
        None if base is None else base.team(other),
        None if after is None else after.team(other),
    )
    scale = ctx.scale
    chance = acceptance_probability(
        ctx.market,
        spec.give,
        spec.get,
        need_gain=theirs.delta_ros / scale,
        drops=len(legality.drops_theirs),
        params=ctx.acceptance,
    )
    warnings = _evaluation_warnings(ctx, spec, base is not None or not simulate)
    title = mine.delta_title
    basis = BASIS_TITLE if title is not None else BASIS_ROS
    gain = title if title is not None else mine.delta_ros / scale
    score = gain * chance.p_accept if legality.legal else None
    recommendation, reason = _recommend(legality, gain, basis, chance)
    return TradeEvaluation(
        spec=spec,
        ours=mine,
        theirs=theirs,
        legality=legality,
        acceptance=chance,
        recommendation=recommendation,
        reason=reason,
        score=score,
        score_basis=basis,
        unit=model.unit,
        fit=tuple(_need_notes(ctx, spec, ours_after, theirs_after)),
        warnings=warnings,
        runs=runs if title is not None else 0,
        seed=seed,
        scale=scale,
    )


def _evaluation_warnings(ctx: TradeContext, spec: TradeSpec, simulated_or_skipped: bool) -> tuple[str, ...]:
    warnings: list[str] = []
    for espn_id in (*spec.give, *spec.get):
        name = ctx.name(espn_id)
        if not ctx.model.knows(espn_id) or espn_id not in ctx.model.values:
            warnings.append(f"{name} has no projection: his value is unknown and counts as 0")
        elif ctx.p_active.get(espn_id, 1.0) <= OUT_BELOW:
            warnings.append(
                f"{name} is ruled out now (engine p_active {ctx.p_active[espn_id]:.0%}); his games count in full"
            )
    if not simulated_or_skipped:
        warnings.extend(ctx.cache.notes)
    return tuple(dict.fromkeys(warnings))


def _recommend(legality: TradeLegality, gain: float, basis: str, chance: Acceptance) -> tuple[str, str]:
    if not legality.legal:
        return DECLINE, "not legal: " + "; ".join(legality.problems)
    edge = EDGE_TITLE if basis == BASIS_TITLE else EDGE_ROS
    unit = "title odds" if basis == BASIS_TITLE else "starter seasons"
    shown = f"{gain:+.1%}" if basis == BASIS_TITLE else f"{gain:+.2f}"
    if gain > edge:
        return ACCEPT, f"it improves our {unit} ({shown})"
    if chance.p_accept >= COUNTER_ACCEPT:
        return COUNTER, f"it does not help our {unit} ({shown}) but P(accept) is {chance.p_accept:.0%}: ask for more"
    return DECLINE, f"it does not help our {unit} ({shown})"


# --- finding deals ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SearchOptions:
    """How :func:`find_trades` searches. ``shapes`` are (players we give, players we get); ``max_candidates`` per side
    bounds what is combined (a side's best by value and its biggest gaps between our value and the market's); ``window``
    is the market-value ratio of what we give to what we get a deal may have; ``min_gain`` is the smallest screening
    gain
    for us in starter seasons; ``min_accept`` the lowest P(accept) worth proposing; ``finalists`` are re-scored exactly
    and ``limit`` returned; ``runs`` / ``seed`` are the simulation's; ``allow_drops`` keeps deals that need a drop on
    our
    side; ``include_ir`` keeps IR-slot players in play; ``distinct`` keeps only the best way to pay for each set of
    players we ask for (the throw-in a deal needs is rarely a separate opportunity)."""

    shapes: tuple[tuple[int, int], ...] = ((1, 1), (2, 1), (1, 2), (2, 2))
    max_candidates: int = 10
    window: tuple[float, float] = (0.5, 2.0)
    min_gain: float = 0.02
    min_accept: float = 0.05
    finalists: int = 20
    limit: int = 10
    runs: int = FINDER_RUNS
    seed: int = DEFAULT_SEED
    allow_drops: bool = False
    include_ir: bool = False
    distinct: bool = True


DEFAULT_OPTIONS: Final = SearchOptions()


@dataclass(frozen=True, slots=True)
class TradeSearch:
    """What :func:`find_trades` found: the ranked deals (best first), how many it enumerated, how many passed the
    screen, and notes."""

    results: tuple[TradeEvaluation, ...]
    enumerated: int
    screened: int
    rescored: int
    opponents: tuple[int, ...]
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _Screened:
    spec: TradeSpec
    gain: float
    p_accept: float

    @property
    def key(self) -> tuple[float, str]:
        return (-self.gain * self.p_accept, self.spec.key)


def _candidates(ctx: TradeContext, team_id: int, options: SearchOptions, *, giving: bool) -> list[int]:
    """The players of ``team_id`` worth combining into a deal: from the top by our value and the top by the gap between
    our value and the market's (the sweeteners we give, the bargains we ask for)."""
    model, book = ctx.model, ctx.market
    barred = set(ctx.locked)
    if giving:
        barred |= set(ctx.untouchables)
    pool = [
        espn_id
        for espn_id in ctx.rosters[team_id]
        if model.knows(espn_id)
        and espn_id in model.values
        and espn_id not in barred
        and (options.include_ir or espn_id not in ctx.ir.get(team_id, frozenset()))
        and (giving or ctx.p_active.get(espn_id, 1.0) > OUT_BELOW)
    ]
    scale, market_scale = ctx.scale, book.scale
    sign = 1.0 if giving else -1.0

    def edge(espn_id: int) -> float:
        return sign * (book.value(espn_id) / market_scale - model.values[espn_id] / scale)

    half = max(1, options.max_candidates // 2)
    by_value = sorted(pool, key=lambda espn_id: (-model.values[espn_id], espn_id))[:half]
    by_edge = sorted(pool, key=lambda espn_id: (-edge(espn_id), espn_id))
    chosen = dict.fromkeys(by_value)
    for espn_id in by_edge:
        if len(chosen) >= options.max_candidates:
            break
        chosen.setdefault(espn_id)
    return list(chosen)


def _screen_team(
    ctx: TradeContext, other: int, options: SearchOptions, cache: dict[tuple[int, frozenset[int]], float]
) -> tuple[int, list[_Screened]]:
    """Enumerate and screen the deals with one opponent: (how many were enumerated, the ones that survive)."""
    model, book, scale = ctx.model, ctx.market, ctx.scale
    mine, theirs = _candidates(ctx, ctx.team_id, options, giving=True), _candidates(ctx, other, options, giving=False)
    ours_now, theirs_now = ctx.playing(ctx.team_id), ctx.playing(other)

    def value(team: int, roster: frozenset[int]) -> float:
        if (team, roster) not in cache:
            cache[(team, roster)] = quick_value(model, roster)
        return cache[(team, roster)]

    before_us, before_them = value(ctx.team_id, ours_now), value(other, theirs_now)
    enumerated = 0
    kept: list[_Screened] = []
    low, high = options.window
    for n_give, n_get in options.shapes:
        for give in itertools.combinations(mine, n_give):
            sent = package_value((book.value(espn_id) for espn_id in give), ctx.acceptance)
            for get in itertools.combinations(theirs, n_get):
                enumerated += 1
                received = package_value((book.value(espn_id) for espn_id in get), ctx.acceptance)
                if received <= 0 or not low <= sent / received <= high:
                    continue
                gain = (value(ctx.team_id, (ours_now - set(give)) | set(get)) - before_us) / scale
                if gain < options.min_gain:
                    continue
                need = (value(other, (theirs_now - set(get)) | set(give)) - before_them) / scale
                chance = acceptance_probability(book, give, get, need_gain=need, params=ctx.acceptance).p_accept
                if chance < options.min_accept:
                    continue
                kept.append(_Screened(TradeSpec(other, give, get), gain, chance))
    return enumerated, kept


def find_trades(
    ctx: TradeContext, *, opponents: Iterable[int] | None = None, options: SearchOptions = DEFAULT_OPTIONS
) -> TradeSearch:
    """Search the league for deals that help us (see the module docs): enumerate, screen with greedy lineups, re-score
    the best :attr:`SearchOptions.finalists` exactly with a simulation, and return the legal ones that raise our title
    odds (or, without a simulation, our rest-of-season value) best first by that gain times P(accept).

    ``opponents`` limits the search to those team ids (default: every other team). Only legal deals come back, and,
    unless :attr:`SearchOptions.allow_drops`, none that needs a drop on our side."""
    chosen = tuple(sorted(set(opponents))) if opponents is not None else ctx.other_teams
    unknown = [team for team in chosen if team == ctx.team_id or team not in ctx.rosters]
    if unknown:
        raise TradeError(f"not another team in league {ctx.league.key!r}: {', '.join(map(str, unknown))}")
    cache: dict[tuple[int, frozenset[int]], float] = {}
    enumerated = 0
    screened: list[_Screened] = []
    for other in chosen:
        count, kept = _screen_team(ctx, other, options, cache)
        enumerated += count
        screened.extend(kept)
    screened.sort(key=lambda item: item.key)
    legal: list[_Screened] = []
    asked: set[tuple[int, frozenset[int]]] = set()
    for item in screened:
        target = (item.spec.other_team_id, frozenset(item.spec.get))
        if options.distinct and target in asked:
            continue
        legality = check_legality(ctx, item.spec)
        if legality.legal and (options.allow_drops or not legality.drops_ours):
            legal.append(item)
            asked.add(target)
        if len(legal) >= options.finalists:
            break
    results: list[TradeEvaluation] = []
    for item in legal:
        evaluation = evaluate_trade(ctx, item.spec, runs=options.runs, seed=options.seed)
        if evaluation.score is None or evaluation.gain <= (0.0 if evaluation.simulated else 0.0):
            continue
        if evaluation.acceptance.p_accept < options.min_accept:
            continue
        results.append(evaluation)
    results.sort(key=lambda found: (-(found.score or 0.0), -found.ours.delta_ros, found.spec.key))
    notes = list(dict.fromkeys(ctx.cache.notes))
    return TradeSearch(
        results=tuple(results[: options.limit]),
        enumerated=enumerated,
        screened=len(screened),
        rescored=len(legal),
        opponents=chosen,
        warnings=tuple(notes),
    )


# --- proposing --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProposedTrade:
    """One deal's fate in :func:`propose_trades`: the stored (or already open) ``proposal``, or the ``blocked`` reason
    (policy, or a team that already has an open offer from us), and with ``dry`` what policy would say."""

    evaluation: TradeEvaluation
    proposal: ProposalRow | None = None
    blocked: str | None = None
    existing: bool = False
    dry: bool = False


def engine_numbers(ctx: TradeContext, evaluation: TradeEvaluation) -> dict[str, Any]:
    """The numbers behind a deal, for a proposal's ``engine_numbers`` (and calibrating P(accept) once it resolves)."""
    ours, theirs, chance = evaluation.ours, evaluation.theirs, evaluation.acceptance
    return {
        "unit": evaluation.unit,
        "score": evaluation.score,
        "score_basis": evaluation.score_basis,
        "ours": {
            "delta_ros": ours.delta_ros,
            "ros_before": ours.ros_before,
            "delta_title": ours.delta_title,
            "delta_playoffs": ours.delta_playoffs,
            "delta_bye": ours.delta_bye,
        },
        "theirs": {
            "team_id": theirs.team_id,
            "delta_ros": theirs.delta_ros,
            "delta_title": theirs.delta_title,
        },
        "acceptance": {
            "p_accept": chance.p_accept,
            "market_receive": chance.receive_value,
            "market_send": chance.send_value,
            "surplus": chance.surplus,
            "need_gain": chance.need_gain,
            "basis": list(chance.basis),
            "rank_type": ctx.market.rank_type,
        },
        "runs": evaluation.runs,
        "seed": evaluation.seed,
        "starter_season": evaluation.scale,
    }


def rationale(ctx: TradeContext, evaluation: TradeEvaluation) -> str:
    """One paragraph for a proposal: the deal, what it does to us and to them, and how likely he is to take it."""
    spec = evaluation.spec
    ours = evaluation.ours
    parts = [
        f"Give {_names(ctx, spec.give)} to {ctx.team_name(spec.other_team_id)} for {_names(ctx, spec.get)}.",
        f"Our rest-of-season lineup value {ours.delta_ros:+.1f} {evaluation.unit}",
    ]
    if ours.delta_title is not None:
        parts[-1] += f", title odds {ours.delta_title:+.1%}"
    parts[-1] += f"; theirs {evaluation.theirs.delta_ros:+.1f}."
    parts.append(evaluation.acceptance.describe() + ".")
    return " ".join(parts)


def propose_trades(
    store: Store,
    config: Config,
    ctx: TradeContext,
    evaluations: Iterable[TradeEvaluation],
    *,
    max_offers: int = 3,
    dry_run: bool = False,
    now: datetime | None = None,
) -> list[ProposedTrade]:
    """Draft ``trade_propose`` proposals for the best of ``evaluations`` through :func:`fm.proposals.propose`.

    Trades are approval-only whatever the config says, and nothing here executes one. Etiquette (DESIGN 9.4): at most
    one open offer per team (a team with an open ``trade_propose`` proposal is skipped, as is a deal that
    duplicates one) and at most ``max_offers`` new offers per call. A deal that is not legal, needs a drop on our side
    (a payload cannot carry one) or has no positive score is skipped. A policy refusal is reported in ``blocked``, never
    raised; with ``dry_run`` policy is evaluated and nothing is stored.
    """
    at = now if now is not None else ctx.now
    open_teams: set[int] = set()
    open_keys: dict[str, ProposalRow] = {}
    for row in store.proposals.open(ctx.league.row_id):
        if row.kind != ProposalKind.TRADE_PROPOSE.value:
            continue
        other = row.payload.get("other_team_id")
        if isinstance(other, int):
            open_teams.add(other)
        if row.dedupe_key:
            open_keys[row.dedupe_key] = row
    outcomes: list[ProposedTrade] = []
    fresh = 0
    for evaluation in evaluations:
        spec = evaluation.spec
        if evaluation.score is None or evaluation.score <= 0 or evaluation.legality.drops_ours:
            reason = (
                "needs a drop on our side, which a trade payload cannot carry"
                if evaluation.legality.drops_ours
                else ("not legal or no gain")
            )
            outcomes.append(ProposedTrade(evaluation, blocked=reason, dry=dry_run))
            continue
        key = f"{ctx.league.key}:{spec.key}"
        if key in open_keys:
            outcomes.append(ProposedTrade(evaluation, open_keys[key], existing=True, dry=dry_run))
            continue
        if spec.other_team_id in open_teams:
            outcomes.append(
                ProposedTrade(
                    evaluation,
                    blocked=f"{ctx.team_name(spec.other_team_id)} already has an open offer from us",
                    dry=dry_run,
                )
            )
            continue
        if fresh >= max_offers:
            outcomes.append(ProposedTrade(evaluation, blocked=f"already proposed {max_offers} offers", dry=dry_run))
            continue
        payload = spec.payload()
        if dry_run:
            verdict = evaluate(
                store, config, ctx.league, ProposalKind.TRADE_PROPOSE, payload, scoring_period_id=ctx.period, now=at
            )
            blocked = None if verdict.allowed else "; ".join(verdict.reasons)
            outcomes.append(ProposedTrade(evaluation, blocked=blocked, dry=True))
            if verdict.allowed:
                fresh += 1
                open_teams.add(spec.other_team_id)
            continue
        try:
            row = propose(
                store,
                config,
                ctx.league,
                ProposalKind.TRADE_PROPOSE,
                payload,
                created_by=TRADES_CREATED_BY,
                scoring_period_id=ctx.period,
                engine_numbers=engine_numbers(ctx, evaluation),
                rationale=rationale(ctx, evaluation),
                dedupe_key=key,
                now=at,
            )
        except PolicyError as exc:
            outcomes.append(ProposedTrade(evaluation, blocked=str(exc)))
            continue
        outcomes.append(ProposedTrade(evaluation, row))
        fresh += 1
        open_teams.add(spec.other_team_id)
    return outcomes


# --- reading a deal from text -----------------------------------------------------------------------------------------

_DEAL = re.compile(r"^\s*give\s+(?P<give>.+?)\s+(?:get|for)\s+(?P<get>.+?)\s*$", re.IGNORECASE | re.DOTALL)


def _normal(name: str) -> str:
    return re.sub(r"[\s.'\-]", "", name).casefold()


def _resolve(ctx: TradeContext, token: str, pool: Iterable[int], where: str) -> int:
    """One player from a name or ESPN id among ``pool``: an id, an exact name, else the one name containing the text."""
    text = token.strip()
    members = sorted(pool)
    if text.lstrip("-").isdigit():
        espn_id = int(text)
        if espn_id in members:
            return espn_id
        raise TradeError(f"ESPN id {espn_id} is not on {where}")
    wanted = _normal(text)
    if not wanted:
        raise TradeError("an empty player name in the deal")
    exact = [espn_id for espn_id in members if _normal(ctx.name(espn_id)) == wanted]
    found = exact or [espn_id for espn_id in members if wanted in _normal(ctx.name(espn_id))]
    if not found:
        raise TradeError(f"no player named {text!r} on {where}")
    if len(found) > 1:
        raise TradeError(
            f"{text!r} matches {', '.join(f'{ctx.name(espn_id)} ({espn_id})' for espn_id in found)} on {where}; "
            "use the ESPN id or more of the name"
        )
    return found[0]


def parse_trade_text(ctx: TradeContext, text: str) -> TradeSpec:
    """A deal from ``"give A, B get C"``: our players by name or ESPN id, then theirs, all from one other team. Raises
    :class:`TradeError` saying what is wrong (no match, several matches, players on two teams)."""
    match = _DEAL.match(text)
    if match is None:
        raise TradeError(f"could not read {text!r}; write it as: give A, B get C")
    give_tokens = [token for token in match["give"].split(",") if token.strip()]
    get_tokens = [token for token in match["get"].split(",") if token.strip()]
    if not give_tokens or not get_tokens:
        raise TradeError(f"could not read {text!r}; both sides need a player: give A, B get C")
    ours = ctx.rosters[ctx.team_id]
    give = tuple(_resolve(ctx, token, ours, "our roster") for token in give_tokens)
    everyone_else = {espn_id for team, roster in ctx.rosters.items() if team != ctx.team_id for espn_id in roster}
    get = tuple(_resolve(ctx, token, everyone_else, "any other roster") for token in get_tokens)
    owners = {ctx.owner(espn_id) for espn_id in get}
    if len(owners) != 1:
        raise TradeError(
            "the players we get must all be on one team: "
            + ", ".join(f"{ctx.name(espn_id)} ({ctx.team_name(ctx.owner(espn_id) or 0)})" for espn_id in get)
        )
    (owner,) = owners
    assert owner is not None
    return TradeSpec(owner, give, get)
