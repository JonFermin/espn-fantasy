"""Market value: what league-mates think a player is worth, for trade-acceptance modeling only (DESIGN 7, 8.3, 9.4).

Two public, no-auth APIs feed :meth:`MarketSource.market_values`:

- FantasyCalc (``https://api.fantasycalc.com/values/current``): redraft trade values computed from real trades, keyed
  by ``espnId`` (NFL only; about 10,000 for the top player, single digits at the tail). The query is the league's
  shape, read from its parsed settings by :meth:`LeagueShape.from_settings` (team count, points per reception,
  starting slots a QB can fill) with no default shape to fall back on, since league settings are data. FantasyCalc
  only publishes 8/10/12/14 teams, 0/0.5/1 PPR and 1/2 QBs, so :func:`nearest_shape` snaps the league onto that grid.
- ESPN's public player pool: ``lm-api-reads.fantasy.espn.com/apis/v3/games/{game}/seasons/{season}/segments/0/
  leaguedefaults/{id}?view=kona_player_info`` with an ``X-Fantasy-Filter`` header, no cookies, both games. Each player
  carries ``ownership`` (percent owned and started, ESPN's reported change, ADP, auction value),
  ``draftRanksByRankType`` (preseason rank per rank type: ``STANDARD``/``PPR``/``ELIMINATION``/``SUPERFLEX`` for ffl,
  ``STANDARD``/``ROTO`` for fba) and ``ratings`` (in-season positional and total ranking). ``leaguedefaults/{id}`` only
  selects ESPN's scoring preset for the stat lines in the response, which this adapter trims to one and never reads;
  ownership and ranks are the same under every preset (verified 2026-10-04: fba answers identically for ids 1 and 3).

Neither API is documented, so both datasets degrade like Sleeper's undocumented endpoints: a failed refresh serves the
last good copy as ``stale``, and with nothing cached the result is empty and ``degraded`` with the reason in
``warnings``. A payload of the wrong shape is kept as ``.rejected`` and never becomes the good copy.

Market values describe what *other managers* believe, not what a player will score. They are an input to P(accept) in
the trade modules and nothing else: no projection, lineup, waiver or drop decision reads them.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, ClassVar, Final, Unpack

from pydantic import AliasPath, BaseModel, ConfigDict, Field, ValidationError, field_validator

from fm.espn.ids import Game, IdMaps, InjuryStatus, ids_for
from fm.espn.settings import LeagueSettings
from fm.sources.base import Fetched, FetchOptions, HttpSource, SourceError, SourceSchemaError, parse_json
from fm.sports.nfl import NFL

logger = logging.getLogger(__name__)

ESPN_FILTER_HEADER: Final = "X-Fantasy-Filter"
FANTASYCALC_TEAMS: Final[tuple[int, ...]] = (8, 10, 12, 14)
FANTASYCALC_PPR: Final[tuple[float, ...]] = (0.0, 0.5, 1.0)
FANTASYCALC_QBS: Final[tuple[int, ...]] = (1, 2)


# --- FantasyCalc league shape ---------------------------------------------------------------------------------------


def _ppr_text(ppr: float) -> str:
    """``1``, ``0.5`` or ``0``: the form FantasyCalc's ``ppr`` parameter takes."""
    return str(int(ppr)) if ppr.is_integer() else f"{ppr:g}"


def _snap(value: float, grid: Sequence[float]) -> float:
    """The grid point nearest to ``value``; a tie goes to the larger point."""
    return min(grid, key=lambda point: (abs(point - value), -point))


@dataclass(frozen=True, slots=True)
class LeagueShape:
    """A point on FantasyCalc's grid of published redraft valuations. There is no default shape: a league's comes from
    its parsed settings (:meth:`from_settings`)."""

    num_teams: int
    ppr: float
    num_qbs: int

    @classmethod
    def from_settings(cls, settings: LeagueSettings) -> LeagueShape:
        """An NFL league's shape on the grid: its team count, its points per reception (the ``REC`` scoring item; 0 when
        receptions do not score) and the starting slots a QB can fill (``QB`` plus the superflex ``OP``, by the NFL
        plugin's eligibility), so a 1-QB league asks for 1 QB and a superflex or 2-QB league for 2. FantasyCalc values
        NFL players only, so another sport's league raises ``ValueError``."""
        if settings.game is not Game.FFL:
            raise ValueError(
                f"FantasyCalc values NFL leagues only; league {settings.league_id} is {settings.game.value}"
            )
        qb_slots = sum(settings.slot_count(slot_id) for slot_id in NFL.eligible_slots("QB", include_reserve=False))
        return nearest_shape(settings.team_count, settings.points_for("REC"), qb_slots)

    @property
    def key(self) -> str:
        """Cache key, e.g. ``redraft_12t_1qb_ppr1``."""
        return f"redraft_{self.num_teams}t_{self.num_qbs}qb_ppr{_ppr_text(self.ppr)}"

    def params(self) -> dict[str, str]:
        """Query parameters for ``values/current``."""
        return {
            "isDynasty": "false",
            "numQbs": str(self.num_qbs),
            "numTeams": str(self.num_teams),
            "ppr": _ppr_text(self.ppr),
        }


def nearest_shape(num_teams: int, ppr: float, num_qbs: int) -> LeagueShape:
    """Snap a league onto FantasyCalc's grid; a tie goes to the larger value (9 teams -> 10, 0.25 PPR -> 0.5)."""
    return LeagueShape(
        num_teams=int(_snap(num_teams, FANTASYCALC_TEAMS)),
        ppr=_snap(float(ppr), FANTASYCALC_PPR),
        num_qbs=int(_snap(num_qbs, FANTASYCALC_QBS)),
    )


# --- models ---------------------------------------------------------------------------------------------------------


class MarketModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)


class FantasyCalcValue(MarketModel):
    """One FantasyCalc redraft entry. ``value`` is the trade value; ``trend_30d`` is its change over 30 days."""

    fantasycalc_id: int = Field(validation_alias=AliasPath("player", "id"))
    name: str = Field(validation_alias=AliasPath("player", "name"))
    position: str = Field(validation_alias=AliasPath("player", "position"))
    team: str | None = Field(default=None, validation_alias=AliasPath("player", "maybeTeam"))
    espn_id: int | None = Field(default=None, validation_alias=AliasPath("player", "espnId"))
    sleeper_id: str | None = Field(default=None, validation_alias=AliasPath("player", "sleeperId"))
    mfl_id: str | None = Field(default=None, validation_alias=AliasPath("player", "mflId"))
    value: int
    overall_rank: int = Field(validation_alias="overallRank")
    position_rank: int = Field(validation_alias="positionRank")
    trend_30d: int | None = Field(default=None, validation_alias="trend30Day")
    tier: int | None = Field(default=None, validation_alias="maybeTier")
    adp: float | None = Field(default=None, validation_alias="maybeAdp")
    trade_frequency: float | None = Field(default=None, validation_alias="maybeTradeFrequency")
    """Share of recorded trades this player appears in."""
    roster_percent: float | None = Field(default=None, validation_alias="maybeRosterPercent")
    """Fraction (0-1) of FantasyCalc's sampled leagues rostering the player."""

    @field_validator("espn_id", mode="before")
    @classmethod
    def _numeric_id(cls, value: object) -> int | None:
        """FantasyCalc sends ``espnId`` as a string; blank or non-numeric means unknown rather than invalid."""
        if value is None or isinstance(value, bool):
            return None
        if isinstance(value, int):
            return value
        text = str(value).strip()
        return int(text) if text.isdigit() else None

    @field_validator("sleeper_id", "mfl_id", "team", mode="before")
    @classmethod
    def _text(cls, value: object) -> str | None:
        if value is None or isinstance(value, bool):
            return None
        text = str(value).strip()
        return text or None


def _epoch_ms(value: object) -> datetime | None:
    """ESPN stamps ``ownership.date`` in integer milliseconds since the epoch."""
    if value is None or isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return datetime.fromtimestamp(value / 1000, UTC)


class EspnOwnership(MarketModel):
    """ESPN's ``ownership`` block: percent owned and started with ESPN's reported change, ADP and auction value."""

    percent_owned: float | None = Field(default=None, alias="percentOwned")
    percent_started: float | None = Field(default=None, alias="percentStarted")
    percent_change: float | None = Field(default=None, alias="percentChange")
    """Change in percent owned over ESPN's trailing window (the ``+/-`` column on the Players page)."""
    average_draft_position: float | None = Field(default=None, alias="averageDraftPosition")
    adp_percent_change: float | None = Field(default=None, alias="averageDraftPositionPercentChange")
    auction_value_average: float | None = Field(default=None, alias="auctionValueAverage")
    auction_value_change: float | None = Field(default=None, alias="auctionValueAverageChange")
    league_count: int | None = Field(default=None, alias="leagueCount")
    updated_at: datetime | None = Field(default=None, alias="date")

    @field_validator("updated_at", mode="before")
    @classmethod
    def _ms(cls, value: object) -> datetime | None:
        return _epoch_ms(value)


class EspnRating(MarketModel):
    """``ratings["0"]``: ESPN's season-to-date rank among all players and among the player's position."""

    positional_ranking: int | None = Field(default=None, alias="positionalRanking")
    total_ranking: int | None = Field(default=None, alias="totalRanking")
    total_rating: float | None = Field(default=None, alias="totalRating")


class EspnPlayerMarket(MarketModel):
    """One player from ESPN's public pool: identity, preseason ranks, in-season rating and ownership. No stats."""

    espn_id: int
    name: str
    position_id: int
    position: str
    """ESPN position label from :mod:`fm.espn.ids` (``RB``, ``D/ST``, ``PG``, ...)."""
    pro_team_id: int
    pro_team: str
    """ESPN pro-team abbreviation; ``FA`` when unsigned."""
    injury_status: InjuryStatus = InjuryStatus.UNKNOWN
    injured: bool = False
    active: bool = True
    ownership: EspnOwnership = Field(default_factory=EspnOwnership)
    draft_ranks: dict[str, int] = Field(default_factory=dict)
    """Preseason rank per ESPN rank type (``PPR``, ``STANDARD``, ``SUPERFLEX``, ``ELIMINATION``, ``ROTO``)."""
    positional_ranking: int | None = None
    """Season-to-date rank among players at the same position."""
    total_ranking: int | None = None
    total_rating: float | None = None


class MarketValue(MarketModel):
    """What the market says about one player, keyed by ESPN id, as P(accept) sees it. ``trade_value`` is NFL-only."""

    espn_id: int
    name: str
    position: str | None = None
    trade_value: int | None = None
    """FantasyCalc redraft trade value; ``None`` for NBA players and for anyone FantasyCalc does not value."""
    value_rank: int | None = None
    value_trend_30d: int | None = None
    espn_rank: int | None = None
    """Preseason ESPN rank under the requested rank type."""
    espn_ranks: dict[str, int] = Field(default_factory=dict)
    positional_ranking: int | None = None
    total_ranking: int | None = None
    percent_owned: float | None = None
    percent_started: float | None = None
    percent_change: float | None = None


# --- parsers --------------------------------------------------------------------------------------------------------


def index_by_espn_id(values: Iterable[FantasyCalcValue]) -> dict[int, FantasyCalcValue]:
    """FantasyCalc values keyed by ESPN id; entries without one are dropped and the better-ranked duplicate wins."""
    indexed: dict[int, FantasyCalcValue] = {}
    for value in sorted(values, key=lambda item: item.overall_rank):
        if value.espn_id is not None:
            indexed.setdefault(value.espn_id, value)
    return indexed


def parse_fantasycalc(payload: bytes) -> list[FantasyCalcValue]:
    """Values sorted by overall rank. FantasyCalc always publishes values, so a non-list, an empty list, or a list with
    no valid entry is a schema error (a broken endpoint, not a quiet day) and never becomes the cached good copy.
    Individual bad entries are skipped and counted."""
    raw = parse_json(payload)
    if not isinstance(raw, list) or not raw:
        shape = "an empty list" if isinstance(raw, list) else type(raw).__name__
        raise SourceSchemaError(f"fantasycalc: expected a non-empty JSON list, got {shape}")
    values: list[FantasyCalcValue] = []
    skipped = 0
    for item in raw:
        if not isinstance(item, dict):
            skipped += 1
            continue
        try:
            values.append(FantasyCalcValue.model_validate(item))
        except ValidationError:
            skipped += 1
    if not values:
        raise SourceSchemaError(f"fantasycalc: none of {len(raw)} entries validated")
    if skipped:
        logger.warning("fantasycalc: skipped %d of %d entries that did not validate", skipped, len(raw))
    return sorted(values, key=lambda value: value.overall_rank)


def _draft_ranks(raw: object) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    ranks: dict[str, int] = {}
    for rank_type, info in raw.items():
        rank = info.get("rank") if isinstance(info, dict) else None
        if isinstance(rank, int | float) and not isinstance(rank, bool) and rank > 0:
            ranks[str(rank_type)] = int(rank)
    return ranks


def _season_rating(raw: object) -> EspnRating | None:
    """``ratings`` is keyed by scoring period as a string; ``"0"`` is the season to date."""
    season = raw.get("0") if isinstance(raw, dict) else None
    return EspnRating.model_validate(season) if isinstance(season, dict) else None


def _player_name(player: Mapping[str, Any]) -> str:
    parts = (player.get("firstName"), player.get("lastName"))
    name = player.get("fullName") or " ".join(str(part) for part in parts if part)
    if not name:
        raise ValueError("player has no name")
    return str(name)


def _espn_player(entry: object, maps: IdMaps) -> EspnPlayerMarket:
    if not isinstance(entry, dict):
        raise ValueError(f"expected an object, got {type(entry).__name__}")
    player = entry.get("player")
    if not isinstance(player, dict):
        raise ValueError("entry has no player object")
    position_id = int(player["defaultPositionId"])
    pro_team_id = int(player.get("proTeamId") or 0)
    rating = _season_rating(entry.get("ratings"))
    return EspnPlayerMarket(
        espn_id=int(player["id"]),
        name=_player_name(player),
        position_id=position_id,
        position=maps.position_label(position_id),
        pro_team_id=pro_team_id,
        pro_team=maps.pro_team(pro_team_id),
        injury_status=maps.injury_status(player.get("injuryStatus")),
        injured=bool(player.get("injured", False)),
        active=bool(player.get("active", True)),
        ownership=EspnOwnership.model_validate(player.get("ownership") or {}),
        draft_ranks=_draft_ranks(player.get("draftRanksByRankType")),
        positional_ranking=rating.positional_ranking if rating else None,
        total_ranking=rating.total_ranking if rating else None,
        total_rating=rating.total_rating if rating else None,
    )


def parse_espn_players(payload: bytes, game: Game | str) -> list[EspnPlayerMarket]:
    """Players in the order ESPN returned them (most owned first). The pool is never empty, so a payload without a
    non-empty ``players`` list, or one in which no entry validates, is a schema error; individual bad entries are
    skipped and counted."""
    maps = ids_for(game)
    raw = parse_json(payload)
    if not isinstance(raw, dict) or not isinstance(raw.get("players"), list):
        raise SourceSchemaError(f"espn players: expected a JSON object with a 'players' list, got {type(raw).__name__}")
    entries: list[object] = raw["players"]
    if not entries:
        raise SourceSchemaError("espn players: the player pool is empty")
    players: list[EspnPlayerMarket] = []
    skipped = 0
    for entry in entries:
        try:
            players.append(_espn_player(entry, maps))
        except (ValidationError, ValueError, TypeError, KeyError):
            skipped += 1
    if not players:
        raise SourceSchemaError(f"espn players: none of {len(entries)} entries validated")
    if skipped:
        logger.warning("espn players: skipped %d of %d entries that did not validate", skipped, len(entries))
    return players


def espn_filter(limit: int) -> str:
    """``X-Fantasy-Filter`` for the pool: the top ``limit`` players by percent owned, stats trimmed to one scoring
    period (ESPN rejects a value of 0, and without the filter every player carries ~60 stat lines)."""
    return json.dumps(
        {
            "players": {
                "limit": limit,
                "sortPercOwned": {"sortAsc": False, "sortPriority": 1},
                "filterStatsForTopScoringPeriodIds": {"value": 1, "additionalValue": []},
            }
        },
        separators=(",", ":"),
    )


def _merge(player: EspnPlayerMarket, value: FantasyCalcValue | None, rank_type: str) -> MarketValue:
    return MarketValue(
        espn_id=player.espn_id,
        name=player.name,
        position=player.position,
        trade_value=value.value if value else None,
        value_rank=value.overall_rank if value else None,
        value_trend_30d=value.trend_30d if value else None,
        espn_rank=player.draft_ranks.get(rank_type),
        espn_ranks=dict(player.draft_ranks),
        positional_ranking=player.positional_ranking,
        total_ranking=player.total_ranking,
        percent_owned=player.ownership.percent_owned,
        percent_started=player.ownership.percent_started,
        percent_change=player.ownership.percent_change,
    )


# --- source ---------------------------------------------------------------------------------------------------------


class MarketSource(HttpSource):
    """FantasyCalc values and ESPN's public pool. Both datasets degrade instead of raising (see module docstring)."""

    name: ClassVar[str] = "market"
    min_interval: ClassVar[float] = 0.5
    FANTASYCALC_URL: ClassVar[str] = "https://api.fantasycalc.com/values/current"
    ESPN_READS: ClassVar[str] = "https://lm-api-reads.fantasy.espn.com/apis/v3/games"
    ESPN_LEAGUE_DEFAULTS_ID: ClassVar[int] = 3
    """ESPN's PPR preset for ffl. Ownership and ranks are the same under every preset (fba: ids 1 and 3 match)."""
    ttl: ClassVar[Mapping[str, timedelta]] = {
        "fantasycalc": timedelta(hours=12),  # FantasyCalc recomputes about once a day
        "espn_players": timedelta(hours=6),  # ownership moves daily, preseason ranks not at all
    }

    def fantasycalc(
        self,
        settings: LeagueSettings,
        **options: Unpack[FetchOptions],
    ) -> Fetched[list[FantasyCalcValue]]:
        """Redraft trade values for an NFL league, best first, at the league's shape on FantasyCalc's grid
        (:meth:`LeagueShape.from_settings`). Degrades; another sport's league raises ``ValueError``."""
        shape = LeagueShape.from_settings(settings)
        params = shape.params()
        return self._fetch_or_degrade(
            "fantasycalc",
            shape.key,
            download=lambda: self.get_bytes(self.FANTASYCALC_URL, params=params),
            parse=parse_fantasycalc,
            empty=[],
            meta={"url": self.FANTASYCALC_URL, **params},
            options=options,
        )

    def espn_players(
        self,
        game: Game | str,
        season: int,
        *,
        limit: int = 300,
        **options: Unpack[FetchOptions],
    ) -> Fetched[list[EspnPlayerMarket]]:
        """The ``limit`` most-owned players in ESPN's public pool with ranks and ownership, most owned first. Degrades.

        ``season`` is ESPN's season id: the NFL year (``2026``) or the year an NBA season ends in (``2027``).
        """
        if limit <= 0:
            raise ValueError("limit must be positive")
        game = Game.coerce(game)
        path = f"{game.value}/seasons/{season}/segments/0/leaguedefaults/{self.ESPN_LEAGUE_DEFAULTS_ID}"
        url = f"{self.ESPN_READS}/{path}"
        params = {"view": "kona_player_info"}
        headers = {ESPN_FILTER_HEADER: espn_filter(limit), "Accept": "application/json"}
        return self._fetch_or_degrade(
            "espn_players",
            f"{game.value}_{season}_top{limit}",
            download=lambda: self.get_bytes(url, params=params, headers=headers),
            parse=lambda payload: parse_espn_players(payload, game),
            empty=[],
            meta={"url": url, **params, "limit": limit},
            options=options,
        )

    def market_values(
        self,
        settings: LeagueSettings,
        *,
        rank_type: str = "STANDARD",
        limit: int = 300,
        **options: Unpack[FetchOptions],
    ) -> Fetched[dict[int, MarketValue]]:
        """ESPN rank and ownership for every pooled player, joined with FantasyCalc values for ffl, keyed by ESPN id.

        The game, the season and FantasyCalc's league shape all come from the league's parsed ``settings``. Pass the
        league's ESPN rank type (ffl: ``PPR``/``STANDARD``/``SUPERFLEX``/``ELIMINATION``; fba: ``STANDARD``/``ROTO``)
        for ``espn_rank``; every type stays available in ``espn_ranks``. The result's ``as_of`` is the older of the two
        inputs, ``cached`` only when both came from cache, and ``stale``, ``degraded`` and ``warnings`` combine both.
        FantasyCalc is not consulted for fba, so NBA values carry no ``trade_value``.
        """
        game = settings.game
        espn = self.espn_players(game, settings.season, limit=limit, **options)
        parts: list[Fetched[Any]] = [espn]
        key = espn.key
        by_id: dict[int, FantasyCalcValue] = {}
        if game is Game.FFL:
            calc = self.fantasycalc(settings, **options)
            parts.append(calc)
            by_id = index_by_espn_id(calc.data)
            key = f"{key}_{calc.key}"
        merged: dict[int, MarketValue] = {}
        for player in espn.data:
            merged[player.espn_id] = _merge(player, by_id.pop(player.espn_id, None), rank_type)
        for espn_id, value in by_id.items():  # valued by FantasyCalc but outside ESPN's top ``limit``
            merged[espn_id] = MarketValue(
                espn_id=espn_id,
                name=value.name,
                position=value.position,
                trade_value=value.value,
                value_rank=value.overall_rank,
                value_trend_30d=value.trend_30d,
            )
        return Fetched(
            merged,
            min(part.as_of for part in parts),
            self.name,
            "values",
            key,
            cached=all(part.cached for part in parts),
            stale=any(part.stale for part in parts),
            degraded=any(part.degraded for part in parts),
            warnings=tuple(warning for part in parts for warning in part.warnings),
        )

    def _fetch_or_degrade[T](
        self,
        dataset: str,
        key: str,
        *,
        download: Callable[[], bytes],
        parse: Callable[[bytes], T],
        empty: T,
        meta: Mapping[str, Any],
        options: FetchOptions,
    ) -> Fetched[T]:
        """``fetch`` that turns a source failure into an empty ``degraded`` result instead of raising."""
        try:
            return self.fetch(dataset, key, download=download, parse=parse, meta=meta, **options)
        except SourceError as exc:
            logger.warning("%s/%s[%s]: unavailable (%s); continuing without it", self.name, dataset, key, exc)
            return Fetched(empty, self.clock(), self.name, dataset, key, degraded=True, warnings=(str(exc),))
