"""Season simulator (DESIGN section 8.4, ROADMAP #32): a Monte Carlo over the league's remaining schedule giving
every team P(win this matchup), P(playoffs), P(first-round bye), P(title) and its seed distribution, and in an NBA
category league the per-category win probabilities of the current matchup.

**Inputs are league data.** The schedule is ESPN's ``mMatchup`` read (:class:`fm.espn.models.MatchupsView`, or its
:class:`fm.espn.models.Matchup` entries): decided matchups set the standings and the head-to-head record, undecided
ones from the current matchup period on are simulated. The playoff format (:class:`PlayoffFormat`) comes from
:class:`fm.espn.settings.LeagueSettings`: ``playoffTeamCount`` teams in a single-elimination bracket of
``2^ceil(log2(teams))`` slots, so the top ``2^rounds - teams`` seeds skip the first round (a 6-team bracket gives two
byes, a 4-team bracket none), one round per playoff matchup period, seeded by ``playoffSeedingRule`` (win percentage
first, then total points scored, total points against or the record against the teams tied with, then points, then a
coin flip). ESPN does not reseed by default; ``reseed`` pairs the best remaining seed with the worst each round.
Division winners seed first when the caller passes the teams' divisions (``TeamRow.division_id``; the settings parser
does not carry them yet). Nothing here is hardcoded to a sport, a scoring system or a number of teams.

**Team strength** is what a decision module computes from its own model and hands over, per matchup period:

- a points league gives each team a :class:`TeamOutlook`, its matchup score as a normal (mean and standard
  deviation), which :func:`points_outlook` builds from :class:`fm.model.valuation.PlayerOutlook` weeks (the best
  lineup each scoring period, :func:`fm.model.valuation.start_value`, summed over the matchup period's scoring periods)
  with the lineup's variance approximated as ``(cv × mean)²`` per scoring period, the same normal approximation
  :mod:`fm.decide.lineup` makes for one week (which can pass its own ``sd`` for the current period);
- a category league gives each team a :class:`CategoryOutlook`, each competed category's team value as a normal, which
  :func:`category_outlook` builds from the players' lines over the period (:func:`fm.model.value_nba.period_lines`):
  counting categories sum, rates are ratios of the team's totals (``FG% = ΣFGM / ΣFGA``, :class:`StatCategory`) with
  a binomial spread, and a reversed category (``TO``) is won by the lower total. A matchup is won on the categories
  (``H2H_MOST_CATEGORIES``), or every category is its own win in the record (``H2H_CATEGORY``), as the league's
  ``scoringType`` says.

An outlook for the current matchup period describes the whole matchup, games already played included, as the
valuation's weekly expected points do; the simulator adds nothing from live scores.

**Runs** are vectorised with numpy over ``runs`` simulations from one :class:`numpy.random.Generator` seeded with
``seed``: the same inputs and seed give the same numbers. Exactly ``playoff teams`` teams make the playoffs in every
run, exactly ``byes`` of them skip the first round, and exactly one wins the title, so the probabilities sum to those
counts.

Pure model code: no store, no network.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Final

import numpy as np

from fm.espn.ids import Game
from fm.espn.models import Matchup, MatchupsView
from fm.espn.settings import LeagueSettings, ScoringType
from fm.model.categories import StatCategory, league_categories
from fm.model.scoring import derive_stats
from fm.model.valuation import PlayerOutlook, slot_instances, start_value

DEFAULT_RUNS: Final = 10_000
DEFAULT_CV: Final = 0.2
"""The default coefficient of variation of a team's score in one scoring period: an NFL lineup's week scores about
20 % of its projection either way, and the same figure is used for a category total."""
SEEDING_POINTS_FOR: Final = "TOTAL_POINTS_SCORED"
SEEDING_POINTS_AGAINST: Final = "TOTAL_POINTS_AGAINST"
SEEDING_H2H: Final = "H2H_RECORD"
"""ESPN ``playoffSeedingRule`` values the simulator breaks ties by after win percentage; any other value (or none)
falls back to total points scored."""
MATCHUP_SCORING_TYPES: Final = frozenset(
    {ScoringType.H2H_POINTS, ScoringType.H2H_MOST_CATEGORIES, ScoringType.H2H_CATEGORY}
)


class SimulationError(ValueError):
    """The league's settings, schedule or outlooks cannot be simulated; the message says what is missing."""


# --- the playoff format ----------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class PlayoffFormat:
    """The bracket the league's settings describe: ``teams`` seeds in ``rounds`` single-elimination rounds, the top
    ``byes`` skipping the first; ``matchup_periods`` are the rounds' matchup periods in order."""

    teams: int
    rounds: int
    byes: int
    matchup_periods: tuple[int, ...]
    regular_season_matchups: int
    seeding_rule: str | None = None
    reseed: bool = False

    @classmethod
    def from_settings(cls, settings: LeagueSettings, *, reseed: bool = False) -> PlayoffFormat:
        """The format from ``scheduleSettings``. Raises :class:`SimulationError` for a league without playoff teams or
        with fewer playoff matchup periods than its bracket has rounds; extra periods beyond the rounds are unused."""
        teams = settings.schedule.playoff_team_count
        if teams < 1:
            raise SimulationError(f"league {settings.league_id}: the settings list no playoff teams")
        rounds = math.ceil(math.log2(teams)) if teams > 1 else 0
        periods = settings.playoff_matchup_periods
        if len(periods) < rounds:
            raise SimulationError(
                f"league {settings.league_id}: {teams} playoff teams need {rounds} rounds but the settings list "
                f"{len(periods)} playoff matchup period(s)"
            )
        return cls(
            teams=teams,
            rounds=rounds,
            byes=2**rounds - teams,
            matchup_periods=periods[:rounds],
            regular_season_matchups=settings.schedule.regular_season_matchups,
            seeding_rule=settings.schedule.playoff_seeding_rule,
            reseed=reseed,
        )

    @property
    def bracket_size(self) -> int:
        return 2**self.rounds

    def is_playoff(self, matchup_period: int) -> bool:
        return matchup_period > self.regular_season_matchups


def bracket_order(size: int) -> tuple[int, ...]:
    """Seeds (1-based) by bracket slot for a bracket of ``size`` (a power of two): adjacent slots meet in round one and
    the best seed meets the worst, as in ``(1, 8, 4, 5, 2, 7, 3, 6)``; seeds above the playoff team count are byes."""
    if size < 1 or size & (size - 1):
        raise ValueError(f"a bracket's size is a power of two, got {size}")
    order = [1]
    while len(order) < size:
        width = 2 * len(order)
        order = [seed for item in order for seed in (item, width + 1 - item)]
    return tuple(order)


# --- standings --------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Standing:
    """A team's regular-season record so far (what ``TeamRow`` stores): in an ``H2H_CATEGORY`` league the wins are
    category wins, and ``points_for`` counts categories won."""

    team_id: int
    wins: float = 0.0
    losses: float = 0.0
    ties: float = 0.0
    points_for: float = 0.0
    points_against: float = 0.0


def _matchup_entries(matchups: MatchupsView | Iterable[Matchup]) -> tuple[Matchup, ...]:
    return tuple(matchups.schedule if isinstance(matchups, MatchupsView) else matchups)


def _home_result(matchup: Matchup) -> int:
    """+1 when the home side won, -1 when the away side did, 0 for a tie (by ESPN's ``winner``, else the points)."""
    if matchup.winner == "HOME":
        return 1
    if matchup.winner == "AWAY":
        return -1
    if matchup.winner == "TIE" or matchup.home is None or matchup.away is None:
        return 0
    margin = matchup.home.total_points - matchup.away.total_points
    return 1 if margin > 0 else -1 if margin < 0 else 0


def _category_counts(matchup: Matchup) -> tuple[tuple[float, float, float], tuple[float, float, float]] | None:
    """Each side's category (wins, losses, ties) from ``cumulativeScore``; ``None`` when a side lacks one."""
    if matchup.home is None or matchup.away is None:
        return None
    home, away = matchup.home.cumulative_score, matchup.away.cumulative_score
    if home is None or away is None:
        return None
    return (
        (float(home.wins), float(home.losses), float(home.ties)),
        (float(away.wins), float(away.losses), float(away.ties)),
    )


def standings_from(
    matchups: MatchupsView | Iterable[Matchup],
    *,
    regular_season_matchups: int,
    each_category: bool = False,
    team_ids: Iterable[int] = (),
) -> dict[int, Standing]:
    """Regular-season standings from the decided matchups of an ``mMatchup`` read, every team in ``team_ids`` or a
    matchup present. With ``each_category`` (``H2H_CATEGORY`` leagues) a matchup adds its ``cumulativeScore``
    category wins, losses and ties; otherwise it is one game, decided by ESPN's ``winner``."""
    records: dict[int, list[float]] = {team_id: [0.0] * 5 for team_id in team_ids}
    for matchup in _matchup_entries(matchups):
        for team_id in matchup.team_ids:
            records.setdefault(team_id, [0.0] * 5)
        if matchup.matchup_period_id > regular_season_matchups or not matchup.is_decided or matchup.is_bye:
            continue
        assert matchup.home is not None and matchup.away is not None
        home, away = records[matchup.home.team_id], records[matchup.away.team_id]
        counts = _category_counts(matchup) if each_category else None
        if counts is not None:
            for record, (wins, losses, ties) in zip((home, away), counts, strict=True):
                record[0] += wins
                record[1] += losses
                record[2] += ties
            home[3] += counts[0][0]
            away[3] += counts[1][0]
            home[4] += counts[1][0]
            away[4] += counts[0][0]
            continue
        result = _home_result(matchup)
        home[0 if result > 0 else 1 if result < 0 else 2] += 1
        away[0 if result < 0 else 1 if result > 0 else 2] += 1
        home[3] += matchup.home.total_points
        away[3] += matchup.away.total_points
        home[4] += matchup.away.total_points
        away[4] += matchup.home.total_points
    return {
        team_id: Standing(team_id, wins, losses, ties, points_for, points_against)
        for team_id, (wins, losses, ties, points_for, points_against) in sorted(records.items())
    }


# --- team outlooks ----------------------------------------------------------------------------------------------------


def _finite(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise SimulationError(f"{what} should be a finite number, got {value!r}")
    return float(value)


def _by_period(value: float | Mapping[int, float], period: int, what: str) -> float:
    if isinstance(value, Mapping):
        if period not in value:
            raise SimulationError(f"{what} has no value for matchup period {period}")
        return _finite(value[period], f"{what} in matchup period {period}")
    return _finite(value, what)


@dataclass(frozen=True)
class TeamOutlook:
    """A points-league team's matchup score as a normal: ``mean`` and ``sd`` either one number for every matchup
    period or a value per matchup period (a period a mapping lacks cannot be simulated)."""

    team_id: int
    mean: float | Mapping[int, float]
    sd: float | Mapping[int, float] = 0.0

    def score(self, period: int) -> tuple[float, float]:
        """``(mean, sd)`` in ``period``. Raises :class:`SimulationError` for a period the outlook lacks, a value
        that is not a finite number, or a negative ``sd``."""
        what = f"team {self.team_id}'s outlook"
        mean = _by_period(self.mean, period, f"{what} mean")
        sd = _by_period(self.sd, period, f"{what} sd")
        if sd < 0:
            raise SimulationError(f"{what} sd in matchup period {period} should be >= 0, got {sd!r}")
        return mean, sd


@dataclass(frozen=True)
class CategoryOutlook:
    """A category-league team's value in every competed category as a normal per matchup period: ``mean`` and
    ``sd`` map each category to one number or to a value per matchup period."""

    team_id: int
    mean: Mapping[str, float | Mapping[int, float]]
    sd: Mapping[str, float | Mapping[int, float]] = field(default_factory=dict)

    def score(self, category: str, period: int) -> tuple[float, float]:
        """``(mean, sd)`` of ``category`` in ``period`` (``sd`` 0 when not given). Raises :class:`SimulationError`
        for a category the outlook lacks, a period it lacks, or a negative ``sd``."""
        what = f"team {self.team_id}'s outlook in {category}"
        if category not in self.mean:
            raise SimulationError(f"{what}: no mean; the outlook covers {', '.join(self.mean) or 'nothing'}")
        mean = _by_period(self.mean[category], period, f"{what} mean")
        sd = _by_period(self.sd[category], period, f"{what} sd") if category in self.sd else 0.0
        if sd < 0:
            raise SimulationError(f"{what} sd in matchup period {period} should be >= 0, got {sd!r}")
        return mean, sd


def _matchup_scoring_periods(settings: LeagueSettings, periods: Iterable[int]) -> dict[int, tuple[int, ...]]:
    if not settings.schedule.lists_scoring_periods:
        raise SimulationError(
            f"league {settings.league_id}: its matchup periods are not listed in scoring periods, so an outlook cannot "
            "be summed from weekly values; build the TeamOutlook per matchup period instead"
        )
    spans: dict[int, tuple[int, ...]] = {}
    for period in periods:
        span = settings.schedule.scoring_periods(period) or ()
        if not span:
            raise SimulationError(f"league {settings.league_id}: matchup period {period} lists no scoring periods")
        spans[period] = span
    return spans


def _cv(cv: float) -> float:
    if not math.isfinite(cv) or cv < 0:
        raise ValueError(f"cv should be a finite number >= 0, got {cv!r}")
    return cv


def points_outlook(
    team_id: int,
    players: Iterable[PlayerOutlook],
    settings: LeagueSettings,
    periods: Iterable[int],
    *,
    cv: float = DEFAULT_CV,
    sd: Mapping[int, float] | None = None,
    slots: Sequence[int] | None = None,
    fill: Iterable[PlayerOutlook] = (),
) -> TeamOutlook:
    """A points-league team's :class:`TeamOutlook` over the matchup ``periods`` from its players' weeks: in each
    scoring period of a matchup period the best lineup's expected points (:func:`fm.model.valuation.start_value` in
    ``slots``, the league's active slot instances by default, holes filled from ``fill``), with variance ``(cv ×
    points)²`` per scoring period; ``sd`` overrides the matchup period's standard deviation (this week's from the
    lineup optimizer, say). Raises :class:`SimulationError` when the league's matchups are not listed in scoring
    periods and ``ValueError`` for a negative ``cv``."""
    scale = _cv(cv)
    spans = _matchup_scoring_periods(settings, periods)
    roster = tuple(players)
    pool = tuple(fill)
    columns = tuple(slots) if slots is not None else slot_instances(settings)
    means: dict[int, float] = {}
    sds: dict[int, float] = {}
    for period, span in spans.items():
        totals = [start_value(roster, columns, scoring_period, fill=pool).total for scoring_period in span]
        means[period] = math.fsum(totals)
        sds[period] = math.sqrt(math.fsum((scale * total) ** 2 for total in totals))
    for period, value in (sd or {}).items():
        if period in sds:
            sds[period] = _finite(value, f"team {team_id}'s sd in matchup period {period}")
    return TeamOutlook(team_id, MappingProxyType(means), MappingProxyType(sds))


def category_totals(
    lines: Iterable[Mapping[str, float]], categories: Iterable[StatCategory]
) -> tuple[dict[str, float], dict[str, float]]:
    """A team's value in each category from its players' lines over a span (:func:`fm.model.value_nba.period_lines`),
    and the volume behind it: a counting category sums (``REB`` derived from ``OREB`` + ``DREB``), a rate is the
    ratio of the team's totals (``FG% = ΣFGM / ΣFGA``; 0 without attempts) with the attempts as volume."""
    stats = tuple(categories)
    made = dict.fromkeys((stat.stat for stat in stats), 0.0)
    volume = dict.fromkeys(made, 0.0)
    for line in lines:
        prepared = derive_stats(line, Game.FBA)
        for stat in stats:
            made[stat.stat] += stat.made(prepared)
            volume[stat.stat] += stat.volume(prepared)
    values = {
        stat.stat: (made[stat.stat] / volume[stat.stat] if volume[stat.stat] > 0 else 0.0)
        if stat.rate is not None
        else made[stat.stat]
        for stat in stats
    }
    return values, volume


def category_outlook(
    team_id: int,
    lines: Mapping[int, Iterable[Mapping[str, float]]],
    settings: LeagueSettings,
    *,
    cv: float = DEFAULT_CV,
    sd: Mapping[str, Mapping[int, float]] | None = None,
) -> CategoryOutlook:
    """A category-league team's :class:`CategoryOutlook` from its players' lines per matchup period (``lines`` maps
    each matchup period to the lines over its games): the team totals of :func:`category_totals`, a counting category's
    spread ``cv × total`` and a percentage's the binomial ``sqrt(p (1 - p) / attempts)`` (a rate that is not a share
    in [0, 1], ``A/TO`` say, spreads by ``cv`` too); ``sd[category][period]`` overrides. Raises ``ValueError`` for a
    league that is not an NBA category league."""
    scale = _cv(cv)
    categories = league_categories(settings)
    means: dict[str, dict[int, float]] = {stat.stat: {} for stat in categories}
    sds: dict[str, dict[int, float]] = {stat.stat: {} for stat in categories}
    for period, period_lines in lines.items():
        values, volumes = category_totals(tuple(period_lines), categories)
        for stat in categories:
            value = values[stat.stat]
            means[stat.stat][period] = value
            if stat.rate is not None and 0.0 <= value <= 1.0:
                attempts = volumes[stat.stat]
                sds[stat.stat][period] = math.sqrt(value * (1.0 - value) / attempts) if attempts > 0 else 0.0
            else:
                sds[stat.stat][period] = scale * abs(value)
    for category, overrides in (sd or {}).items():
        if category not in sds:
            raise ValueError(f"{category!r} is not a category here; the categories are {', '.join(means)}")
        for period, value in overrides.items():
            sds[category][period] = _finite(value, f"team {team_id}'s sd in {category}, matchup period {period}")
    return CategoryOutlook(
        team_id,
        MappingProxyType({stat: MappingProxyType(values) for stat, values in means.items()}),
        MappingProxyType({stat: MappingProxyType(values) for stat, values in sds.items()}),
    )


# --- outcomes ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TeamOdds:
    """One team's chances: ``win_week`` in the current matchup period (``None`` without a matchup in it; a tie counts
    half), ``playoffs``, ``bye`` (a first-round bye), ``title``, ``seeds`` (``seeds[k]`` is P(seed ``k + 1``)),
    ``expected_wins`` over the regular season, and in a category league ``categories``: P(win each category) in the
    current matchup, empty without one."""

    team_id: int
    win_week: float | None
    playoffs: float
    bye: float
    title: float
    seeds: tuple[float, ...]
    expected_wins: float
    categories: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class SeasonOdds:
    """What :func:`simulate_season` found: every team's :class:`TeamOdds` by ESPN team id, the format simulated, the
    matchup period the simulation started from, the ``runs`` and ``seed`` that reproduce it, and warnings about
    schedule entries it skipped."""

    matchup_period: int
    format: PlayoffFormat
    runs: int
    seed: int
    teams: Mapping[int, TeamOdds]
    warnings: tuple[str, ...] = ()

    def team(self, team_id: int) -> TeamOdds:
        """Raises ``KeyError`` naming the teams simulated."""
        if team_id not in self.teams:
            raise KeyError(f"team {team_id} was not simulated; the teams are {', '.join(map(str, self.teams))}")
        return self.teams[team_id]


@dataclass(frozen=True, slots=True)
class MatchupOdds:
    """One matchup: P(home wins), P(away wins), P(tie), and in a category league the home side's P(win) per category
    (a tie counts half)."""

    home_team_id: int
    away_team_id: int
    home: float
    away: float
    tie: float
    categories: Mapping[str, float] = field(default_factory=dict)


# --- the engines ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class _Outcome:
    """A simulated matchup from the home side: ``result`` +1/0/-1 per run, the record each side adds (one game, or
    the category counts), the points each side scored (category wins in a category league), and per-category results
    (``runs × categories``, +1/0/-1 from the home side) when the league competes on categories."""

    result: np.ndarray
    home_record: tuple[np.ndarray, np.ndarray, np.ndarray]
    away_record: tuple[np.ndarray, np.ndarray, np.ndarray]
    home_points: np.ndarray
    away_points: np.ndarray
    categories: np.ndarray | None = None


def _sign(values: np.ndarray) -> np.ndarray:
    return np.sign(values).astype(np.int8)


def _one_game(result: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (result > 0).astype(float), (result < 0).astype(float), (result == 0).astype(float)


class _PointsEngine:
    def __init__(self, outlooks: Mapping[int, TeamOutlook], team_ids: Sequence[int]) -> None:
        self._outlooks = outlooks
        self._team_ids = tuple(team_ids)
        self._tables: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    @property
    def category_names(self) -> tuple[str, ...]:
        return ()

    def table(self, period: int) -> tuple[np.ndarray, np.ndarray]:
        cached = self._tables.get(period)
        if cached is None:
            scores = [self._outlooks[team_id].score(period) for team_id in self._team_ids]
            cached = (np.array([mean for mean, _ in scores]), np.array([sd for _, sd in scores]))
            self._tables[period] = cached
        return cached

    def play(self, period: int, home: np.ndarray, away: np.ndarray, rng: np.random.Generator) -> _Outcome:
        mean, sd = self.table(period)
        home_points = mean[home] + sd[home] * rng.standard_normal(home.shape)
        away_points = mean[away] + sd[away] * rng.standard_normal(away.shape)
        result = _sign(home_points - away_points)
        return _Outcome(result, _one_game(result), _one_game(-result), home_points, away_points)


class _CategoryEngine:
    def __init__(
        self,
        outlooks: Mapping[int, CategoryOutlook],
        team_ids: Sequence[int],
        categories: Sequence[StatCategory],
        *,
        each_category: bool,
    ) -> None:
        self._outlooks = outlooks
        self._team_ids = tuple(team_ids)
        self._categories = tuple(categories)
        self._signs = np.array([stat.sign for stat in self._categories])
        self._each = each_category
        self._tables: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    @property
    def category_names(self) -> tuple[str, ...]:
        return tuple(stat.stat for stat in self._categories)

    def table(self, period: int) -> tuple[np.ndarray, np.ndarray]:
        """``(mean, sd)`` arrays of shape ``teams × categories``."""
        cached = self._tables.get(period)
        if cached is None:
            scores = [
                [self._outlooks[team_id].score(stat.stat, period) for stat in self._categories]
                for team_id in self._team_ids
            ]
            cached = (
                np.array([[mean for mean, _ in row] for row in scores]),
                np.array([[sd for _, sd in row] for row in scores]),
            )
            self._tables[period] = cached
        return cached

    def play(self, period: int, home: np.ndarray, away: np.ndarray, rng: np.random.Generator) -> _Outcome:
        mean, sd = self.table(period)
        shape = (*home.shape, len(self._categories))
        home_values = mean[home] + sd[home] * rng.standard_normal(shape)
        away_values = mean[away] + sd[away] * rng.standard_normal(shape)
        categories = _sign((home_values - away_values) * self._signs)
        won = (categories > 0).sum(axis=-1).astype(float)
        lost = (categories < 0).sum(axis=-1).astype(float)
        tied = (categories == 0).sum(axis=-1).astype(float)
        result = _sign(won - lost)
        if self._each:
            return _Outcome(result, (won, lost, tied), (lost, won, tied), won, lost, categories)
        return _Outcome(result, _one_game(result), _one_game(-result), won, lost, categories)


_Engine = _PointsEngine | _CategoryEngine


def _check_scoring(settings: LeagueSettings) -> None:
    if settings.scoring_type not in MATCHUP_SCORING_TYPES:
        raise SimulationError(
            f"league {settings.league_id}: scoring type {settings.scoring_type_raw or settings.scoring_type.value!r} "
            "has no head-to-head matchups to simulate; the simulator covers H2H_POINTS, H2H_MOST_CATEGORIES and "
            "H2H_CATEGORY leagues"
        )


def _engine(
    settings: LeagueSettings, outlooks: Mapping[int, TeamOutlook | CategoryOutlook], team_ids: Sequence[int]
) -> _Engine:
    _check_scoring(settings)
    missing = [team_id for team_id in team_ids if team_id not in outlooks]
    if missing:
        raise SimulationError(f"no outlook for team(s) {', '.join(map(str, missing))}")
    if settings.is_categories:
        wrong = [team_id for team_id in team_ids if not isinstance(outlooks[team_id], CategoryOutlook)]
        if wrong:
            raise SimulationError(
                f"league {settings.league_id} competes on categories: team(s) {', '.join(map(str, wrong))} need a "
                "CategoryOutlook, not a TeamOutlook"
            )
        chosen = {team_id: outlook for team_id, outlook in outlooks.items() if isinstance(outlook, CategoryOutlook)}
        return _CategoryEngine(
            chosen,
            team_ids,
            league_categories(settings),
            each_category=settings.scoring_type is ScoringType.H2H_CATEGORY,
        )
    wrong = [team_id for team_id in team_ids if not isinstance(outlooks[team_id], TeamOutlook)]
    if wrong:
        raise SimulationError(
            f"league {settings.league_id} scores points: team(s) {', '.join(map(str, wrong))} need a TeamOutlook, "
            "not a CategoryOutlook"
        )
    return _PointsEngine(
        {team_id: outlook for team_id, outlook in outlooks.items() if isinstance(outlook, TeamOutlook)}, team_ids
    )


def _rng(seed: int, runs: int) -> np.random.Generator:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed should be an int, got {seed!r}")
    if isinstance(runs, bool) or not isinstance(runs, int) or runs < 1:
        raise ValueError(f"runs should be a positive int, got {runs!r}")
    return np.random.default_rng(seed)


def _win_share(result: np.ndarray) -> float:
    """P(win) with a tie counting half."""
    return float(np.mean(result > 0) + 0.5 * np.mean(result == 0))


def simulate_matchup(
    settings: LeagueSettings,
    home: TeamOutlook | CategoryOutlook,
    away: TeamOutlook | CategoryOutlook,
    period: int,
    *,
    seed: int,
    runs: int = DEFAULT_RUNS,
) -> MatchupOdds:
    """One matchup in ``period`` between two outlooks under the league's scoring, ``runs`` times from ``seed``."""
    rng = _rng(seed, runs)
    engine = _engine(settings, {home.team_id: home, away.team_id: away}, (home.team_id, away.team_id))
    outcome = engine.play(period, np.zeros(runs, dtype=np.intp), np.ones(runs, dtype=np.intp), rng)
    categories = {}
    if outcome.categories is not None:
        categories = {
            name: _win_share(outcome.categories[:, column]) for column, name in enumerate(engine.category_names)
        }
    return MatchupOdds(
        home.team_id,
        away.team_id,
        float(np.mean(outcome.result > 0)),
        float(np.mean(outcome.result < 0)),
        float(np.mean(outcome.result == 0)),
        MappingProxyType(categories),
    )


# --- the season -------------------------------------------------------------------------------------------------------


@dataclass
class _Season:
    """The arrays one simulation runs on: ``runs × teams`` records, and the head-to-head wins and games when the
    seeding rule needs them."""

    wins: np.ndarray
    losses: np.ndarray
    ties: np.ndarray
    points_for: np.ndarray
    points_against: np.ndarray
    h2h_wins: np.ndarray | None
    h2h_games: np.ndarray | None

    @classmethod
    def start(cls, runs: int, standings: Sequence[Standing], *, head_to_head: bool) -> _Season:
        count = len(standings)

        def column(values: Iterable[float]) -> np.ndarray:
            return np.tile(np.array(list(values), dtype=float), (runs, 1))

        return cls(
            wins=column(standing.wins for standing in standings),
            losses=column(standing.losses for standing in standings),
            ties=column(standing.ties for standing in standings),
            points_for=column(standing.points_for for standing in standings),
            points_against=column(standing.points_against for standing in standings),
            h2h_wins=np.zeros((runs, count, count)) if head_to_head else None,
            h2h_games=np.zeros((runs, count, count)) if head_to_head else None,
        )

    def record(self, home: int, away: int, outcome: _Outcome) -> None:
        for team, record, points, against in (
            (home, outcome.home_record, outcome.home_points, outcome.away_points),
            (away, outcome.away_record, outcome.away_points, outcome.home_points),
        ):
            self.wins[:, team] += record[0]
            self.losses[:, team] += record[1]
            self.ties[:, team] += record[2]
            self.points_for[:, team] += points
            self.points_against[:, team] += against
        self.head_to_head(home, away, outcome.home_record[0], outcome.away_record[0], outcome.home_record[2])

    def head_to_head(
        self, home: int, away: int, home_wins: np.ndarray, away_wins: np.ndarray, ties: np.ndarray
    ) -> None:
        if self.h2h_wins is None or self.h2h_games is None:
            return
        self.h2h_wins[:, home, away] += home_wins + 0.5 * ties
        self.h2h_wins[:, away, home] += away_wins + 0.5 * ties
        games = home_wins + away_wins + ties
        self.h2h_games[:, home, away] += games
        self.h2h_games[:, away, home] += games

    @property
    def win_pct(self) -> np.ndarray:
        games = self.wins + self.losses + self.ties
        return np.divide(self.wins + 0.5 * self.ties, games, out=np.zeros_like(games), where=games > 0)

    def tiebreak(self, rule: str | None) -> np.ndarray:
        """The seeding rule's second key: points for, points against, or the record against the teams tied with."""
        if rule == SEEDING_POINTS_AGAINST:
            return self.points_against
        if rule == SEEDING_H2H and self.h2h_wins is not None and self.h2h_games is not None:
            pct = self.win_pct
            tied = pct[:, :, None] == pct[:, None, :]
            tied &= ~np.eye(pct.shape[1], dtype=bool)[None, :, :]
            wins = (self.h2h_wins * tied).sum(axis=2)
            games = (self.h2h_games * tied).sum(axis=2)
            return np.divide(wins, games, out=np.zeros_like(wins), where=games > 0)
        return self.points_for


def _seed_order(
    season: _Season, rule: str | None, divisions: np.ndarray | None, rng: np.random.Generator
) -> np.ndarray:
    """Each run's teams from the first seed down (``runs × teams`` team indices): division winners first when
    ``divisions`` groups the teams, then win percentage, the seeding rule's tiebreak, points for, and a coin flip."""
    runs, count = season.wins.shape
    keys = (rng.random((runs, count)), -season.points_for, -season.tiebreak(rule), -season.win_pct)
    order = np.lexsort(keys, axis=-1)
    if divisions is None or len(np.unique(divisions)) < 2:
        return order
    position = np.argsort(order, axis=-1)
    flags = np.zeros((runs, count))
    rows = np.arange(runs)
    for division in np.unique(divisions):
        members = divisions == division
        masked = np.where(members[None, :], position, count)
        flags[rows, masked.argmin(axis=1)] = 1.0
    return np.lexsort((*keys, -flags), axis=-1)


def _decided_playoffs(
    matchups: Sequence[Matchup], index: Mapping[int, int], fmt: PlayoffFormat
) -> dict[int, list[tuple[int, int, int]]]:
    """Decided playoff matchups by period as ``(home, away, winner)`` team indices (ties go to the home side, which
    the bracket does not produce; ESPN decides them itself)."""
    decided: dict[int, list[tuple[int, int, int]]] = {}
    for matchup in matchups:
        if not fmt.is_playoff(matchup.matchup_period_id) or not matchup.is_decided or matchup.is_bye:
            continue
        assert matchup.home is not None and matchup.away is not None
        home, away = index.get(matchup.home.team_id), index.get(matchup.away.team_id)
        if home is None or away is None:
            continue
        winner = away if _home_result(matchup) < 0 else home
        decided.setdefault(matchup.matchup_period_id, []).append((home, away, winner))
    return decided


def _pair_round(
    engine: _Engine,
    period: int,
    slots: np.ndarray,
    position: np.ndarray,
    decided: Sequence[tuple[int, int, int]],
    rng: np.random.Generator,
) -> np.ndarray:
    """Play the pairs of adjacent ``slots`` (``runs × 2m`` team indices, -1 for an empty slot) in ``period``; returns
    the winners (``runs × m``). A tie goes to the better seed (``position`` is each team's seed per run); a matchup ESPN
    already decided keeps its winner."""
    runs = slots.shape[0]
    rows = np.arange(runs)
    home, away = slots[:, 0::2], slots[:, 1::2]
    winners = np.where(away < 0, home, np.where(home < 0, away, -1))
    both = (home >= 0) & (away >= 0)
    if both.any():
        safe_home, safe_away = np.where(both, home, 0), np.where(both, away, 0)
        outcome = engine.play(period, safe_home, safe_away, rng)
        better = position[rows[:, None], safe_home] <= position[rows[:, None], safe_away]
        played = np.where(outcome.result > 0, home, np.where(outcome.result < 0, away, np.where(better, home, away)))
        winners = np.where(both, played, winners)
    for first, second, winner in decided:
        known = ((home == first) & (away == second)) | ((home == second) & (away == first))
        winners = np.where(known, winner, winners)
    return winners


def _reseed(winners: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Re-pair the winners best seed against worst: ``(best, worst, 2nd, 2nd worst, ...)`` per run."""
    runs, count = winners.shape
    rows = np.arange(runs)[:, None]
    seeds = np.where(winners >= 0, position[rows, np.where(winners >= 0, winners, 0)], count + 1)
    ranked = np.take_along_axis(winners, np.argsort(seeds, axis=1), axis=1)
    slots = np.empty_like(ranked)
    slots[:, 0::2] = ranked[:, : count // 2]
    slots[:, 1::2] = ranked[:, ::-1][:, : count // 2]
    return slots


def _playoffs(
    engine: _Engine,
    fmt: PlayoffFormat,
    order: np.ndarray,
    decided: Mapping[int, Sequence[tuple[int, int, int]]],
    rng: np.random.Generator,
) -> np.ndarray:
    """The champion per run from the seed ``order`` (``runs × teams``)."""
    position = np.argsort(order, axis=-1)
    empty = np.full(order.shape[0], -1)
    slots = np.stack(
        [order[:, seed - 1] if seed <= fmt.teams else empty for seed in bracket_order(fmt.bracket_size)], axis=1
    )
    for round_index, period in enumerate(fmt.matchup_periods):
        winners = _pair_round(engine, period, slots, position, decided.get(period, ()), rng)
        slots = _reseed(winners, position) if fmt.reseed and round_index + 1 < fmt.rounds else winners
    return slots[:, 0]


def _current_period(settings: LeagueSettings, matchups: MatchupsView | Iterable[Matchup], given: int | None) -> int:
    if given is not None:
        return given
    if isinstance(matchups, MatchupsView) and matchups.status is not None:
        if matchups.status.current_matchup_period is not None:
            return matchups.status.current_matchup_period
    if settings.current_matchup_period is not None:
        return settings.current_matchup_period
    raise SimulationError(
        f"league {settings.league_id}: the current matchup period is unknown; pass current_matchup_period"
    )


def simulate_season(
    settings: LeagueSettings,
    matchups: MatchupsView | Iterable[Matchup],
    outlooks: Mapping[int, TeamOutlook | CategoryOutlook],
    *,
    seed: int,
    runs: int = DEFAULT_RUNS,
    current_matchup_period: int | None = None,
    standings: Mapping[int, Standing] | None = None,
    divisions: Mapping[int, int] | None = None,
    reseed: bool = False,
) -> SeasonOdds:
    """Monte Carlo the rest of the season ``runs`` times from ``seed`` (see the module docs).

    ``matchups`` is the league's ``mMatchup`` read: its decided regular-season matchups are the record so far (or
    ``standings`` is, by team id, when given; head-to-head records always come from the matchups), its undecided ones
    from ``current_matchup_period`` (default: the view's, then the settings' current matchup period) on are simulated
    with the teams' ``outlooks`` (a :class:`TeamOutlook` per team in a points league, a :class:`CategoryOutlook` in a
    category league), and decided playoff matchups keep their winners. ``divisions`` (team id -> division id) seeds
    division winners first. Raises :class:`SimulationError` for a league without head-to-head matchups, a bracket the
    settings cannot describe, a team without an outlook, or an outlook without a period that is played.
    """
    rng = _rng(seed, runs)
    fmt = PlayoffFormat.from_settings(settings, reseed=reseed)
    entries = _matchup_entries(matchups)
    current = _current_period(settings, matchups, current_matchup_period)
    each_category = settings.scoring_type is ScoringType.H2H_CATEGORY
    team_ids = sorted(
        {team_id for matchup in entries for team_id in matchup.team_ids} | set(outlooks) | set(standings or ())
    )
    if not team_ids:
        raise SimulationError(f"league {settings.league_id}: no teams to simulate")
    if len(team_ids) < fmt.teams:
        raise SimulationError(
            f"league {settings.league_id}: {len(team_ids)} team(s) to simulate but the settings give {fmt.teams} "
            "playoff spots"
        )
    index = {team_id: position for position, team_id in enumerate(team_ids)}
    engine = _engine(settings, outlooks, team_ids)

    derived = standings_from(
        entries, regular_season_matchups=fmt.regular_season_matchups, each_category=each_category, team_ids=team_ids
    )
    baseline = [
        (standings or {}).get(team_id) or (derived[team_id] if standings is None else Standing(team_id))
        for team_id in team_ids
    ]
    season = _Season.start(runs, baseline, head_to_head=fmt.seeding_rule == SEEDING_H2H)
    for matchup in entries:  # the head-to-head record so far
        if fmt.is_playoff(matchup.matchup_period_id) or not matchup.is_decided or matchup.is_bye:
            continue
        assert matchup.home is not None and matchup.away is not None
        home, away = index[matchup.home.team_id], index[matchup.away.team_id]
        counts = _category_counts(matchup) if each_category else None
        if counts is not None:
            (home_wins, _, ties), (away_wins, _, _) = counts
            season.head_to_head(home, away, np.full(runs, home_wins), np.full(runs, away_wins), np.full(runs, ties))
        else:
            result = np.full(runs, _home_result(matchup), dtype=np.int8)
            season.head_to_head(home, away, *_one_game(result))

    warnings: list[str] = []
    week: dict[int, float] = {}
    categories: dict[int, dict[str, float]] = {}
    pending = sorted(
        (matchup for matchup in entries if not matchup.is_decided and not fmt.is_playoff(matchup.matchup_period_id)),
        key=lambda matchup: (matchup.matchup_period_id, matchup.id),
    )
    for matchup in pending:
        if matchup.is_bye:
            continue
        if matchup.matchup_period_id < current:
            warnings.append(
                f"matchup {matchup.id} in period {matchup.matchup_period_id} is undecided but before the current "
                f"period {current}; skipped"
            )
            continue
        assert matchup.home is not None and matchup.away is not None
        home, away = index[matchup.home.team_id], index[matchup.away.team_id]
        outcome = engine.play(matchup.matchup_period_id, np.full(runs, home), np.full(runs, away), rng)
        season.record(home, away, outcome)
        if matchup.matchup_period_id == current:
            week[matchup.home.team_id] = _win_share(outcome.result)
            week[matchup.away.team_id] = _win_share(-outcome.result)
            if outcome.categories is not None:
                names = engine.category_names
                categories[matchup.home.team_id] = {
                    name: _win_share(outcome.categories[:, column]) for column, name in enumerate(names)
                }
                categories[matchup.away.team_id] = {
                    name: _win_share(-outcome.categories[:, column]) for column, name in enumerate(names)
                }

    groups = None
    if divisions:
        unknown = sorted(set(team_ids) - set(divisions))
        if unknown:
            raise SimulationError(f"divisions given but team(s) {', '.join(map(str, unknown))} have none")
        groups = np.array([divisions[team_id] for team_id in team_ids])
    order = _seed_order(season, fmt.seeding_rule, groups, rng)
    position = np.argsort(order, axis=-1)
    champion = _playoffs(engine, fmt, order, _decided_playoffs(entries, index, fmt), rng)

    count = len(team_ids)
    teams = {
        team_id: TeamOdds(
            team_id=team_id,
            win_week=week.get(team_id),
            playoffs=float(np.mean(position[:, column] < fmt.teams)),
            bye=float(np.mean(position[:, column] < fmt.byes)),
            title=float(np.mean(champion == column)),
            seeds=tuple(float(np.mean(position[:, column] == seed)) for seed in range(count)),
            expected_wins=float(np.mean(season.wins[:, column])),
            categories=MappingProxyType(categories.get(team_id, {})),
        )
        for team_id, column in index.items()
    }
    return SeasonOdds(current, fmt, runs, seed, MappingProxyType(teams), tuple(warnings))
