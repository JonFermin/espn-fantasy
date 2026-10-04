"""Sleeper NFL data: player DB, trending adds/drops, and the web app's projections and same-day stats.

Documented (``https://api.sleeper.app/v1``; no auth; stay under 1,000 calls/min; player DB at most once a day):
``players/nfl``, ``players/nfl/trending/{add,drop}`` and ``state/nfl``. These raise :class:`SourceUnavailable` when they
fail with nothing cached.

Undocumented (``https://api.sleeper.com``, the endpoints the Sleeper web app itself calls):
``projections/nfl/{season}/{week}`` (RotoWire-based stat-line projections) and ``stats/nfl/{season}/{week}`` (Sportradar
stat lines updated during games, including ``off_snp``/``tm_off_snp`` snap counts). The documented
``api.sleeper.app/v1/projections/nfl/regular/...`` endpoint broke in Sept 2026 (it still answers 200, with ADP-only junk
keyed by player id), so these may break the same way. :meth:`SleeperSource.projections`,
:meth:`SleeperSource.week_stats` and :meth:`SleeperSource.snap_counts` therefore never raise for source failures: they
return the last good copy as ``stale`` when one is cached, otherwise an empty ``degraded`` result with the reason in
``warnings``, and the projection blend runs on ESPN alone.

Sleeper's ``espn_id`` coverage is poor (none for rookies), so the crosswalk (``fm.model.ids``) uses ``ff_playerids`` and
treats these ids as a hint only. Stat lines keep Sleeper's keys (``pass_yd``, ``rush_att``, ``rec``, ``rec_yd``,
``fum_lost``, ``off_snp``, ...); Sleeper's own ``pts_ppr``/``pts_half_ppr``/``pts_std`` ride along but league points are
always computed from the league's scoring items.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from datetime import UTC, date, datetime, timedelta
from typing import Any, ClassVar, Literal, Unpack

import httpx
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from fm.sources.base import Fetched, FetchOptions, HttpSource, SourceError, SourceSchemaError, parse_json

logger = logging.getLogger(__name__)

type TrendKind = Literal["add", "drop"]

ADP_PREFIXES = ("adp_", "pos_adp_")
"""Stat keys that describe draft position, not a projection; an entry with nothing else is a placeholder."""


def epoch_ms(value: object) -> datetime | None:
    """Sleeper timestamps are integer milliseconds since the epoch; tolerate seconds and ISO strings too."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, bool):
        raise ValueError("boolean is not a timestamp")
    if isinstance(value, str):
        if value.lstrip("-").isdigit():
            value = int(value)
        else:
            parsed = datetime.fromisoformat(value)
            return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    if isinstance(value, int | float):
        seconds = value / 1000 if abs(value) >= 1e11 else value
        return datetime.fromtimestamp(seconds, UTC)
    raise ValueError(f"not a timestamp: {value!r}")


def _blank_to_none(value: object) -> object:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    return value


class SleeperModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, populate_by_name=True)


class SleeperState(SleeperModel):
    """``state/nfl``: where the NFL season stands right now."""

    season: int
    week: int
    season_type: str
    display_week: int | None = None
    league_season: int | None = None
    previous_season: int | None = None
    season_start_date: date | None = None
    season_has_scores: bool | None = None


class SleeperPlayer(SleeperModel):
    """One ``players/nfl`` entry. Team defenses are keyed by team abbreviation (``"PHI"``) with position ``DEF``."""

    player_id: str
    first_name: str | None = None
    last_name: str | None = None
    full_name: str | None = None
    position: str | None = None
    fantasy_positions: list[str] = Field(default_factory=list)
    team: str | None = None
    status: str | None = None
    """Roster status: Active, Inactive, Injured Reserve, Physically Unable to Perform, Practice Squad, ..."""
    active: bool = False
    injury_status: str | None = None
    """Questionable, Doubtful, Out, IR, PUP, Sus, COV, DNR, NA; ``None`` when healthy."""
    injury_body_part: str | None = None
    injury_notes: str | None = None
    injury_start_date: date | None = None
    practice_participation: str | None = None
    practice_description: str | None = None
    news_updated: datetime | None = None
    espn_id: int | None = None
    gsis_id: str | None = None
    rotowire_id: int | None = None
    yahoo_id: int | None = None
    sportradar_id: str | None = None
    depth_chart_position: str | None = None
    depth_chart_order: int | None = None
    number: int | None = None
    age: int | None = None
    years_exp: int | None = None
    search_rank: int | None = None

    @field_validator("fantasy_positions", mode="before")
    @classmethod
    def _none_is_empty(cls, value: object) -> object:
        return value if value is not None else []

    @field_validator("active", mode="before")
    @classmethod
    def _none_is_false(cls, value: object) -> object:
        return value if value is not None else False

    @field_validator(
        "first_name",
        "last_name",
        "full_name",
        "position",
        "team",
        "status",
        "injury_status",
        "injury_body_part",
        "injury_notes",
        "practice_participation",
        "practice_description",
        "gsis_id",
        "sportradar_id",
        "depth_chart_position",
        mode="before",
    )
    @classmethod
    def _clean(cls, value: object) -> object:
        # gsis_id arrives as " 00-0035057" for hundreds of players; empty strings mean null throughout.
        return _blank_to_none(value)

    @field_validator("injury_start_date", mode="before")
    @classmethod
    def _blank_date(cls, value: object) -> object:
        return _blank_to_none(value)

    @field_validator("news_updated", mode="before")
    @classmethod
    def _ms(cls, value: object) -> datetime | None:
        return epoch_ms(value)

    @property
    def name(self) -> str:
        return self.full_name or " ".join(part for part in (self.first_name, self.last_name) if part)


class TrendingPlayer(SleeperModel):
    """``count`` adds or drops over the lookback window."""

    player_id: str
    count: int


class SleeperStatLine(SleeperModel):
    """One player-week from ``projections/nfl`` (``category="proj"``) or ``stats/nfl`` (``category="stat"``)."""

    player_id: str
    season: int
    week: int
    season_type: str = "regular"
    category: str
    company: str | None = None
    team: str | None = None
    opponent: str | None = None
    game_id: str | None = None
    game_date: date | None = Field(default=None, alias="date")
    position: str | None = None
    stats: dict[str, float] = Field(default_factory=dict)
    updated_at: datetime | None = None

    @model_validator(mode="before")
    @classmethod
    def _flatten(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        out: dict[str, Any] = dict(data)
        player = out.get("player")
        if out.get("position") is None and isinstance(player, dict):
            out["position"] = player.get("position")
        if out.get("updated_at") is None:
            out["updated_at"] = out.get("last_modified")
        stats = out.get("stats")
        if isinstance(stats, dict):
            out["stats"] = {
                key: float(value)
                for key, value in stats.items()
                if isinstance(value, int | float) and not isinstance(value, bool)
            }
        return out

    @field_validator("updated_at", mode="before")
    @classmethod
    def _ms(cls, value: object) -> datetime | None:
        return epoch_ms(value)

    @property
    def is_placeholder(self) -> bool:
        """True when the entry carries only ADP fields, i.e. Sleeper has no projection for this player."""
        return all(key.startswith(ADP_PREFIXES) for key in self.stats)


class SleeperSnapCount(SleeperModel):
    """Offensive snaps for one player-game, derived from a ``stats/nfl`` line during or after the game."""

    player_id: str
    season: int
    week: int
    team: str | None = None
    opponent: str | None = None
    game_date: date | None = None
    position: str | None = None
    offensive_snaps: float
    team_offensive_snaps: float | None = None
    snap_share: float | None = None

    @classmethod
    def from_line(cls, line: SleeperStatLine) -> SleeperSnapCount | None:
        snaps = line.stats.get("off_snp")
        if snaps is None:
            return None
        team_snaps = line.stats.get("tm_off_snp")
        share = snaps / team_snaps if team_snaps else None
        return cls(
            player_id=line.player_id,
            season=line.season,
            week=line.week,
            team=line.team,
            opponent=line.opponent,
            game_date=line.game_date,
            position=line.position,
            offensive_snaps=snaps,
            team_offensive_snaps=team_snaps,
            snap_share=share,
        )


def _as_list_of_dicts(payload: object, what: str) -> list[dict[str, Any]]:
    if not isinstance(payload, list):
        raise SourceSchemaError(f"{what}: expected a JSON list, got {type(payload).__name__}")
    items: list[dict[str, Any]] = []
    for item in payload:
        if not isinstance(item, dict):
            raise SourceSchemaError(f"{what}: expected objects in the list, got {type(item).__name__}")
        items.append(item)
    return items


def parse_state(payload: bytes) -> SleeperState:
    raw = parse_json(payload)
    if not isinstance(raw, dict):
        raise SourceSchemaError(f"sleeper state: expected a JSON object, got {type(raw).__name__}")
    try:
        return SleeperState.model_validate(raw)
    except ValidationError as exc:
        raise SourceSchemaError(f"sleeper state: {exc}") from exc


def parse_players(payload: bytes) -> dict[str, SleeperPlayer]:
    """Player id -> player. Entries that fail validation are skipped and counted; an empty result is a schema error."""
    raw = parse_json(payload)
    if not isinstance(raw, dict) or not raw:
        raise SourceSchemaError("sleeper players: expected a non-empty JSON object keyed by player id")
    players: dict[str, SleeperPlayer] = {}
    skipped = 0
    for player_id, entry in raw.items():
        if not isinstance(entry, dict):
            skipped += 1
            continue
        try:
            players[player_id] = SleeperPlayer.model_validate({**entry, "player_id": player_id})
        except ValidationError:
            skipped += 1
    if not players:
        raise SourceSchemaError(f"sleeper players: none of {len(raw)} entries validated")
    if skipped:
        logger.warning("sleeper players: skipped %d of %d entries that did not validate", skipped, len(raw))
    return players


def parse_trending(payload: bytes) -> list[TrendingPlayer]:
    items = _as_list_of_dicts(parse_json(payload), "sleeper trending")
    try:
        trending = [TrendingPlayer.model_validate(item) for item in items]
    except ValidationError as exc:
        raise SourceSchemaError(f"sleeper trending: {exc}") from exc
    return sorted(trending, key=lambda entry: entry.count, reverse=True)


def parse_stat_lines(payload: bytes, *, drop_placeholders: bool) -> list[SleeperStatLine]:
    """Stat lines from a projections or stats payload. A non-list payload or one with no valid entry is a schema error
    (that is how the legacy endpoint broke); individual bad entries are skipped and counted."""
    items = _as_list_of_dicts(parse_json(payload), "sleeper stat lines")
    lines: list[SleeperStatLine] = []
    skipped = 0
    for item in items:
        try:
            line = SleeperStatLine.model_validate(item)
        except ValidationError:
            skipped += 1
            continue
        if drop_placeholders and line.is_placeholder:
            continue
        lines.append(line)
    if items and not lines and skipped == len(items):
        raise SourceSchemaError(f"sleeper stat lines: none of {len(items)} entries validated")
    if skipped:
        logger.warning("sleeper stat lines: skipped %d of %d entries that did not validate", skipped, len(items))
    return lines


class SleeperSource(HttpSource):
    """Sleeper adapter. Documented endpoints raise on failure; the undocumented ones degrade (see module docstring)."""

    name: ClassVar[str] = "sleeper"
    min_interval: ClassVar[float] = 0.2
    API_V1: ClassVar[str] = "https://api.sleeper.app/v1"
    DATA_API: ClassVar[str] = "https://api.sleeper.com"
    POSITIONS: ClassVar[tuple[str, ...]] = ("QB", "RB", "WR", "TE", "K", "DEF")
    ttl: ClassVar[Mapping[str, timedelta]] = {
        "players": timedelta(hours=24),  # Sleeper asks for at most one pull per day
        "trending": timedelta(minutes=15),
        "state": timedelta(hours=1),
        "projections": timedelta(minutes=30),
        "stats": timedelta(minutes=10),  # same-day snaps move during games
    }

    def state(self, **options: Unpack[FetchOptions]) -> Fetched[SleeperState]:
        url = f"{self.API_V1}/state/nfl"
        return self.fetch(
            "state", "nfl", download=lambda: self.get_bytes(url), parse=parse_state, meta={"url": url}, **options
        )

    def players(self, **options: Unpack[FetchOptions]) -> Fetched[dict[str, SleeperPlayer]]:
        """The full NFL player DB (about 15 MB), keyed by Sleeper player id."""
        url = f"{self.API_V1}/players/nfl"
        return self.fetch(
            "players", "nfl", download=lambda: self.get_bytes(url), parse=parse_players, meta={"url": url}, **options
        )

    def trending(
        self,
        kind: TrendKind,
        *,
        lookback_hours: int = 24,
        limit: int = 25,
        **options: Unpack[FetchOptions],
    ) -> Fetched[list[TrendingPlayer]]:
        """Most added or dropped players across Sleeper leagues, highest count first."""
        url = f"{self.API_V1}/players/nfl/trending/{kind}"
        params = {"lookback_hours": lookback_hours, "limit": limit}
        return self.fetch(
            "trending",
            f"{kind}_{lookback_hours}h_{limit}",
            download=lambda: self.get_bytes(url, params=params),
            parse=parse_trending,
            meta={"url": url, **params},
            **options,
        )

    def projections(
        self,
        season: int,
        week: int,
        *,
        season_type: str = "regular",
        **options: Unpack[FetchOptions],
    ) -> Fetched[list[SleeperStatLine]]:
        """RotoWire stat-line projections for one week, placeholders (ADP-only entries) dropped. Degrades."""
        return self._stat_lines(
            "projections",
            f"{self.DATA_API}/projections/nfl/{season}/{week}",
            season=season,
            week=week,
            season_type=season_type,
            drop_placeholders=True,
            options=options,
        )

    def week_stats(
        self,
        season: int,
        week: int,
        *,
        season_type: str = "regular",
        **options: Unpack[FetchOptions],
    ) -> Fetched[list[SleeperStatLine]]:
        """Actual stat lines for one week, updated during games (includes snap counts). Degrades."""
        return self._stat_lines(
            "stats",
            f"{self.DATA_API}/stats/nfl/{season}/{week}",
            season=season,
            week=week,
            season_type=season_type,
            drop_placeholders=False,
            options=options,
        )

    def snap_counts(
        self,
        season: int,
        week: int,
        *,
        season_type: str = "regular",
        **options: Unpack[FetchOptions],
    ) -> Fetched[list[SleeperSnapCount]]:
        """Same-day offensive snap counts: the ``week_stats`` lines that carry ``off_snp``. Degrades."""
        lines = self.week_stats(season, week, season_type=season_type, **options)
        snaps = [snap for line in lines.data if (snap := SleeperSnapCount.from_line(line)) is not None]
        return Fetched(
            snaps,
            lines.as_of,
            lines.source,
            lines.dataset,
            lines.key,
            cached=lines.cached,
            stale=lines.stale,
            degraded=lines.degraded,
            warnings=lines.warnings,
            raw_path=lines.raw_path,
        )

    def _stat_lines(
        self,
        dataset: str,
        url: str,
        *,
        season: int,
        week: int,
        season_type: str,
        drop_placeholders: bool,
        options: FetchOptions,
    ) -> Fetched[list[SleeperStatLine]]:
        key = f"{season_type}_{season}_w{week}"
        params = httpx.QueryParams(
            [("season_type", season_type), *(("position[]", pos) for pos in self.POSITIONS), ("order_by", "ppr")]
        )
        try:
            return self.fetch(
                dataset,
                key,
                download=lambda: self.get_bytes(url, params=params),
                parse=lambda payload: parse_stat_lines(payload, drop_placeholders=drop_placeholders),
                meta={"url": url, "season": season, "week": week, "season_type": season_type},
                **options,
            )
        except SourceError as exc:
            logger.warning("%s/%s[%s]: unavailable (%s); continuing without it", self.name, dataset, key, exc)
            return Fetched([], self.clock(), self.name, dataset, key, degraded=True, warnings=(str(exc),))
