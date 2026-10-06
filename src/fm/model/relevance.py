"""News relevance: who matters to a league right now, which news is about them, and storing news (DESIGN 9.2, 10).

The ``news_triage`` worker reads new items about relevant players only (DESIGN section 10): our roster, this week's
opponent, the top free agents and our trade targets. :func:`relevance_for` builds that set for one league from the
store, each player tagged with every :class:`RelevanceReason` that applies, from the latest roster snapshot ``fm sync``
wrote (or the scoring period given):

- **roster**: our team's players (``LeagueRow.team_id``);
- **opponent**: the players of ``opponent_team_id``. The store holds no matchups, so the caller names the opponent,
  usually with :func:`opponent_from` over an ``mMatchup`` read (``EspnClient.matchups()``);
- **free agent**: the ``free_agents`` best players on no roster in the league, among those ESPN projects (the wire
  ``fm sync`` reads, most-owned first, plus earlier syncs' players still unrostered). Best means ESPN's season
  projection under the league's own settings (a period line stands in for a player without one): fantasy points in a
  points league; in a category league the sum of per-category z-scores across those players, percentages
  volume-weighted (makes minus the pool's rate times attempts) and reverse categories (TO) negated, a ranking
  stand-in until the category valuation (ROADMAP #24);
- **trade target**: the players we would receive in the league's open trade proposals (proposed, approved or
  executing: offers drafted for us and incoming offers under evaluation), plus any ``trade_targets`` given.

Storing: :func:`ingest_news` writes a :class:`fm.sources.news.NewsPoll`'s items to ``news_items``. An ESPN item keeps
the athletes ESPN tagged; a RotoWire item names its player only in the title, so the name is matched against the
players the store knows (:class:`PlayerIndex`, names compared as :func:`fm.model.ids_nba.normalize_name` writes them;
two players sharing a name both get the item, and the triage tells them apart from the text). Items already stored
under the same source and id are skipped, and so is the same story (:func:`fm.sources.news.fingerprint`) under another
id or source published within :data:`fm.sources.news.DUPLICATE_WINDOW` of a stored copy: the rule
:func:`fm.sources.news.dedupe` applies within one poll, applied across polls.

Filtering: :func:`relevant_news` keeps the stored items about a relevant player, each with the players it is about and
why each matters. A RotoWire item stored before its player was known has no ESPN id, so its title is matched against
the relevant players' names then. Irrelevant items are not marked triaged here; the triage marks what it has read.
"""

from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from types import MappingProxyType
from typing import Final

from pydantic import ValidationError

from fm.config import Sport
from fm.espn.models import MatchupsView
from fm.espn.settings import LeagueSettings
from fm.model.ids_nba import normalize_name
from fm.model.projections import ESPN
from fm.model.scoring import DERIVATIONS, Ratio, Scorer, ScoringError
from fm.proposals.payloads import TradePayload
from fm.proposals.policy import TRADE_KINDS, ProposalError, parse_payload
from fm.sources.news import DUPLICATE_WINDOW, ROTOWIRE, NewsItem, fingerprint, news_subject
from fm.store import OPEN_PROPOSAL_STATUSES, LeagueRow, NewsItemRow, PlayerRow, Store, utc_now

logger = logging.getLogger(__name__)

DEFAULT_FREE_AGENTS: Final = 25
"""Free agents whose news counts as relevant, best first."""


class RelevanceReason(StrEnum):
    """Why a player's news matters to a league."""

    ROSTER = "roster"
    OPPONENT = "opponent"
    FREE_AGENT = "free_agent"
    TRADE_TARGET = "trade_target"


# --- who is relevant ----------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Relevance:
    """The players whose news matters to one league, each with every reason that applies. ``players`` holds the
    stored rows of those players (a trade target the store has not seen has none); ``warnings`` say what could not be
    read (no roster snapshot yet, an opponent without a roster, settings not synced)."""

    league_id: int
    key: str
    sport: Sport
    team_id: int
    scoring_period_id: int | None
    """The roster snapshot read; ``None`` before the league's first sync."""
    opponent_team_id: int | None
    reasons: Mapping[int, frozenset[RelevanceReason]]
    players: Mapping[int, PlayerRow]
    warnings: tuple[str, ...] = ()

    def __contains__(self, espn_id: object) -> bool:
        return espn_id in self.reasons

    def why(self, espn_id: int) -> frozenset[RelevanceReason]:
        """The reasons ``espn_id`` is relevant; empty when he is not."""
        return self.reasons.get(espn_id, frozenset())

    def ids(self, reason: RelevanceReason | None = None) -> tuple[int, ...]:
        """The relevant ESPN ids (with ``reason``, when given), ascending."""
        return tuple(sorted(espn_id for espn_id, why in self.reasons.items() if reason is None or reason in why))

    def name(self, espn_id: int) -> str:
        player = self.players.get(espn_id)
        return player.full_name if player is not None else f"ESPN {espn_id}"

    def describe(self) -> str:
        """One line: ``nfl (scoring period 4): 20 relevant players: 11 roster, 3 opponent (team 2), 4 free agents,
        2 trade targets``."""
        counts = {reason: len(self.ids(reason)) for reason in RelevanceReason}
        opponent = f" (team {self.opponent_team_id})" if self.opponent_team_id is not None else " (none given)"
        period = f"scoring period {self.scoring_period_id}" if self.scoring_period_id is not None else "no roster yet"
        return (
            f"{self.key} ({period}): {len(self.reasons)} relevant players: {counts[RelevanceReason.ROSTER]} roster, "
            f"{counts[RelevanceReason.OPPONENT]} opponent{opponent}, {counts[RelevanceReason.FREE_AGENT]} free agents, "
            f"{counts[RelevanceReason.TRADE_TARGET]} trade targets"
        )


def opponent_from(matchups: MatchupsView, team_id: int, matchup_period: int | None = None) -> int | None:
    """``team_id``'s opponent in ``matchup_period`` (ESPN's current matchup period when omitted) from an ``mMatchup``
    read; ``None`` on a bye, when the period is unknown, or when the team has no matchup in it."""
    period = matchup_period
    if period is None and matchups.status is not None:
        period = matchups.status.current_matchup_period
    if period is None:
        return None
    for matchup in matchups.for_team(team_id, period):
        side = matchup.opponent(team_id)
        if side is not None:
            return side.team_id
    return None


def relevance_for(
    store: Store,
    league_id: int,
    *,
    opponent_team_id: int | None = None,
    free_agents: int = DEFAULT_FREE_AGENTS,
    trade_targets: Iterable[int] = (),
    scoring_period_id: int | None = None,
) -> Relevance:
    """The relevant players of the stored league ``league_id`` (see the module docstring).

    Reads the latest roster snapshot unless ``scoring_period_id`` is given. Before the league's first sync there is no
    roster to read: only trade targets are relevant and ``warnings`` says so. A league id the store does not know
    raises ``LookupError``; a negative ``free_agents`` or our own team as the opponent raises ``ValueError``.
    """
    league = store.leagues.get(league_id)
    if league is None:
        raise LookupError(f"no league with id {league_id} in the store")
    if free_agents < 0:
        raise ValueError(f"free_agents must be >= 0, got {free_agents!r}")
    if opponent_team_id is not None and opponent_team_id == league.team_id:
        raise ValueError(f"team {opponent_team_id} is our own team in league {league.key}, not an opponent")
    reasons: dict[int, set[RelevanceReason]] = {}
    warnings: list[str] = []

    def tag(espn_ids: Iterable[int], reason: RelevanceReason) -> None:
        for espn_id in espn_ids:
            reasons.setdefault(espn_id, set()).add(reason)

    period = scoring_period_id if scoring_period_id is not None else store.rosters.latest_period(league_id)
    if period is None:
        warnings.append(f"{league.key}: no roster snapshot yet (run fm sync); only trade targets are relevant")
    else:
        ours = [entry.espn_id for entry in store.rosters.team(league_id, period, league.team_id)]
        if not ours:
            warnings.append(f"{league.key}: team {league.team_id} has no roster in scoring period {period}")
        tag(ours, RelevanceReason.ROSTER)
        if opponent_team_id is not None:
            theirs = [entry.espn_id for entry in store.rosters.team(league_id, period, opponent_team_id)]
            if not theirs:
                warnings.append(f"{league.key}: team {opponent_team_id} has no roster in scoring period {period}")
            tag(theirs, RelevanceReason.OPPONENT)
        tag(_top_free_agents(store, league, period, free_agents, warnings), RelevanceReason.FREE_AGENT)
    tag(trade_targets, RelevanceReason.TRADE_TARGET)
    tag(_proposed_trade_targets(store, league_id, warnings), RelevanceReason.TRADE_TARGET)

    players = {player.espn_id: player for player in store.players.many(league.sport, reasons)}
    return Relevance(
        league_id=league_id,
        key=league.key,
        sport=league.sport,
        team_id=league.team_id,
        scoring_period_id=period,
        opponent_team_id=opponent_team_id,
        reasons=MappingProxyType({espn_id: frozenset(why) for espn_id, why in sorted(reasons.items())}),
        players=MappingProxyType(players),
        warnings=tuple(warnings),
    )


def _stored_settings(store: Store, league: LeagueRow, warnings: list[str]) -> LeagueSettings | None:
    row = store.settings.get(league.row_id)
    if row is None:
        warnings.append(f"{league.key}: league settings are not synced; free agents are not ranked")
        return None
    try:
        return LeagueSettings.model_validate(row.settings)
    except ValidationError as exc:
        warnings.append(f"{league.key}: stored league settings do not load ({exc.error_count()} errors); run fm sync")
        return None


def _top_free_agents(store: Store, league: LeagueRow, period: int, limit: int, warnings: list[str]) -> list[int]:
    """The ``limit`` best players ESPN projects for the season or ``period`` who are on no roster in the league."""
    if limit == 0:
        return []
    sport, season = league.sport, league.season
    season_lines = {row.espn_id: row.stats for row in store.projections.for_period(sport, season, 0, source=ESPN)}
    period_lines = (
        {row.espn_id: row.stats for row in store.projections.for_period(sport, season, period, source=ESPN)}
        if period != 0
        else {}
    )
    candidates = (season_lines.keys() | period_lines.keys()) - store.rosters.rostered_ids(league.row_id, period)
    if not candidates:
        return []
    settings = _stored_settings(store, league, warnings)
    if settings is None:
        return sorted(candidates)[:limit]
    positions = {player.espn_id: player.default_position_id for player in store.players.many(sport, candidates)}
    by_season = _values(settings, {i: season_lines[i] for i in candidates if i in season_lines}, positions)
    by_period = _values(
        settings, {i: period_lines[i] for i in candidates if i not in season_lines and i in period_lines}, positions
    )

    def rank(espn_id: int) -> tuple[int, float, int]:
        if espn_id in by_season:
            return (0, -by_season[espn_id], espn_id)
        if espn_id in by_period:
            return (1, -by_period[espn_id], espn_id)
        return (2, 0.0, espn_id)

    return sorted(candidates, key=rank)[:limit]


def _values(
    settings: LeagueSettings, lines: Mapping[int, Mapping[str, float]], positions: Mapping[int, int | None]
) -> dict[int, float]:
    """What each line is worth in the league, for ranking: points, or the category z-score sum. A line the scorer
    refuses (a non-finite stat) is left out."""
    scorer = Scorer(settings)
    prepared: dict[int, dict[str, float]] = {}
    for espn_id, line in lines.items():
        try:
            prepared[espn_id] = scorer.prepare(line)
        except ScoringError as exc:
            logger.warning("free-agent ranking: skipped ESPN %d (%s)", espn_id, exc)
    if scorer.is_points:
        return {espn_id: scorer.points(line, position=positions.get(espn_id)) for espn_id, line in prepared.items()}
    return _category_scores(settings, prepared)


def _category_scores(settings: LeagueSettings, lines: Mapping[int, Mapping[str, float]]) -> dict[int, float]:
    """The sum over the league's categories of each player's z-score among ``lines``. A percentage category scores
    its impact, ``makes - pool_rate * attempts``, so a low-volume shooter is not mistaken for an elite one; a reverse
    category counts negatively; a category with no spread adds nothing."""
    totals = dict.fromkeys(lines, 0.0)
    if len(lines) < 2:
        return totals
    rules = DERIVATIONS.get(settings.game, {})
    for item in settings.scoring_items:
        rule = rules.get(item.stat)
        if isinstance(rule, Ratio):
            made = {i: _made(line, rule) for i, line in lines.items()}
            tried = {i: line.get(rule.denominator, 0.0) for i, line in lines.items()}
            attempts = math.fsum(tried.values())
            if attempts <= 0:
                continue
            rate = math.fsum(made.values()) / attempts
            raw = {i: made[i] - rate * tried[i] for i in lines}
        else:
            raw = {i: line.get(item.stat, 0.0) for i, line in lines.items()}
        spread = statistics.pstdev(raw.values())
        if spread == 0:
            continue
        mean = statistics.fmean(raw.values())
        sign = -1.0 if item.is_reverse else 1.0
        for espn_id, value in raw.items():
            totals[espn_id] += sign * (value - mean) / spread
    return totals


def _made(line: Mapping[str, float], rule: Ratio) -> float:
    bonus = rule.bonus_weight * line.get(rule.bonus, 0.0) if rule.bonus is not None else 0.0
    return line.get(rule.numerator, 0.0) + bonus


def _proposed_trade_targets(store: Store, league_id: int, warnings: list[str]) -> list[int]:
    """The players we would receive in the league's open trade proposals, in proposal order, once each."""
    rows = store.proposals.find(
        league_id=league_id, statuses=OPEN_PROPOSAL_STATUSES, kinds=[kind.value for kind in TRADE_KINDS]
    )
    found: dict[int, None] = {}
    for row in rows:
        try:
            payload = parse_payload(row)
        except (ValueError, ProposalError) as exc:
            warnings.append(f"proposal {row.id} ({row.kind}): payload does not load ({exc}); its players are skipped")
            continue
        if isinstance(payload, TradePayload):
            found.update(dict.fromkeys(payload.get_espn_ids))
    return list(found)


# --- names --------------------------------------------------------------------------------------------------------


class PlayerIndex:
    """ESPN ids by normalized full name for one sport: how an item that names its player (RotoWire) finds him."""

    def __init__(self, sport: Sport, players: Iterable[PlayerRow]) -> None:
        self.sport: Sport = sport
        names: dict[str, set[int]] = {}
        for player in players:
            if player.sport != sport:
                raise ValueError(f"player {player.espn_id} is {player.sport}, not {sport}")
            name = normalize_name(player.full_name)
            if name:
                names.setdefault(name, set()).add(player.espn_id)
        self._names = {name: tuple(sorted(ids)) for name, ids in names.items()}

    @classmethod
    def from_store(cls, store: Store, sport: Sport) -> PlayerIndex:
        """Every player a sync wrote for the sport's leagues: on a roster in a league's latest snapshot, or carrying an
        ESPN line for the season or that period (the wire). The store has no list of all players, so this is the
        set it can name; it holds every rostered and free-agent player :func:`relevance_for` reads by default."""
        ids: set[int] = set()
        for league in store.leagues.all():
            if league.sport != sport:
                continue
            period = store.rosters.latest_period(league.row_id)
            periods = {0}
            if period is not None:
                ids |= store.rosters.rostered_ids(league.row_id, period)
                periods.add(period)
            for scoring_period in sorted(periods):
                ids.update(
                    row.espn_id
                    for row in store.projections.for_period(sport, league.season, scoring_period, source=ESPN)
                )
        return cls(sport, store.players.many(sport, ids))

    def resolve(self, name: str) -> tuple[int, ...]:
        """The ESPN ids of the players named ``name`` (every one, when several share it); empty when none is."""
        return self._names.get(normalize_name(name), ())

    def players_in(self, item: NewsItem) -> tuple[int, ...]:
        """The ESPN ids an item is about: the ones its source tagged, else the players its names resolve to."""
        if item.espn_ids:
            return item.espn_ids
        found: dict[int, None] = {}
        for name in item.player_names:
            found.update(dict.fromkeys(self.resolve(name)))
        return tuple(found)


# --- storing ------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class NewsIngest:
    """What :func:`ingest_news` did with a batch of items."""

    stored: tuple[NewsItemRow, ...]
    """The new rows, oldest first, with their ids."""
    known: int
    """Items already stored under the same source and id (every poll repeats most of the last one), or given twice."""
    repeats: int
    """Items repeating a story stored before or earlier in the batch (same text) under another id or source."""

    @property
    def unmatched(self) -> tuple[NewsItemRow, ...]:
        """New rows about no player the store knows (team news, or a player no sync has written)."""
        return tuple(row for row in self.stored if not row.espn_ids)

    def describe(self) -> str:
        """One line: ``news: 3 stored (1 about no known player), 9 already stored, 2 repeats``."""
        return (
            f"news: {len(self.stored)} stored ({len(self.unmatched)} about no known player), "
            f"{self.known} already stored, {self.repeats} repeats"
        )


def ingest_news(
    store: Store,
    items: Iterable[NewsItem],
    *,
    index: PlayerIndex | None = None,
    now: datetime | None = None,
) -> NewsIngest:
    """Store the new items among ``items`` (a :class:`fm.sources.news.NewsPoll`'s, say) in ``news_items``, in one
    transaction, oldest first; see the module docstring for duplicates and player matching.

    ``index`` names the players of its sport (built from the store per sport otherwise); ``now`` stamps
    ``fetched_at`` for an item its feed did not stamp.
    """
    first: dict[tuple[str, str], NewsItem] = {}
    known = repeats = 0
    for item in items:
        if item.key in first:
            known += 1
        else:
            first[item.key] = item
    batch = sorted(first.values(), key=lambda entry: entry.published_at)
    if not batch:
        return NewsIngest((), known, repeats)
    stamp = now if now is not None else utc_now()
    indexes: dict[Sport, PlayerIndex] = {index.sport: index} if index is not None else {}
    stored: list[NewsItemRow] = []
    with store.db.transaction():
        stories = _stored_stories(store, batch)
        for item in batch:
            told = stories.setdefault((item.sport, item.fingerprint), [])
            if any(key != item.key and abs(item.published_at - at) <= DUPLICATE_WINDOW for at, key in told):
                repeats += 1
                continue
            if item.sport not in indexes:
                indexes[item.sport] = PlayerIndex.from_store(store, item.sport)
            row = store.news.ingest(_row(item, indexes[item.sport], stamp))
            if row is None:
                known += 1
                continue
            told.append((item.published_at, item.key))
            stored.append(row)
    return NewsIngest(tuple(stored), known, repeats)


def _stored_stories(
    store: Store, batch: Iterable[NewsItem]
) -> dict[tuple[Sport, str], list[tuple[datetime, tuple[str, str]]]]:
    """(sport, fingerprint) -> when and under which (source, external_id) each stored item with that story was
    published, for the items published from :data:`DUPLICATE_WINDOW` before the batch's oldest item of the sport."""
    oldest: dict[Sport, datetime] = {}
    for item in batch:
        if item.sport not in oldest or item.published_at < oldest[item.sport]:
            oldest[item.sport] = item.published_at
    stories: dict[tuple[Sport, str], list[tuple[datetime, tuple[str, str]]]] = {}
    for sport, published in oldest.items():
        for row in store.news.published_since(published - DUPLICATE_WINDOW, sport=sport):
            told = stories.setdefault((sport, fingerprint(row.title, row.body)), [])
            told.append((row.published_at, (row.source, row.external_id)))
    return stories


def _row(item: NewsItem, index: PlayerIndex, stamp: datetime) -> NewsItemRow:
    return NewsItemRow(
        source=item.source,
        external_id=item.external_id,
        sport=item.sport,
        title=item.title,
        body=item.body,
        url=item.url,
        espn_ids=list(index.players_in(item)),
        published_at=item.published_at,
        fetched_at=item.fetched_at if item.fetched_at is not None else stamp,
    )


# --- filtering ----------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RelevantNews:
    """A stored item about at least one relevant player: who, and why each matters."""

    item: NewsItemRow
    players: Mapping[int, frozenset[RelevanceReason]]

    @property
    def espn_ids(self) -> tuple[int, ...]:
        return tuple(self.players)

    @property
    def reasons(self) -> frozenset[RelevanceReason]:
        return frozenset(reason for why in self.players.values() for reason in why)


def relevant_news(items: Iterable[NewsItemRow], relevance: Relevance) -> list[RelevantNews]:
    """The items about a relevant player, in the order given (``store.news.untriaged()``, say); items of another
    sport are skipped. A stored item's players are its ``espn_ids``; a RotoWire item stored with none is matched by
    the name in its title against the relevant players. ``untriaged`` returns the oldest items first, a page at a
    time, so a caller marks every item it has read triaged, relevant or not, or old irrelevant ones fill the page."""
    names: PlayerIndex | None = None
    found: list[RelevantNews] = []
    for item in items:
        if item.sport != relevance.sport:
            continue
        espn_ids: tuple[int, ...] = tuple(item.espn_ids)
        if not espn_ids and item.source == ROTOWIRE and (subject := news_subject(item.title)) is not None:
            if names is None:
                names = PlayerIndex(relevance.sport, relevance.players.values())
            espn_ids = names.resolve(subject)
        hits = {espn_id: relevance.reasons[espn_id] for espn_id in espn_ids if espn_id in relevance.reasons}
        if hits:
            found.append(RelevantNews(item, MappingProxyType(hits)))
    return found
