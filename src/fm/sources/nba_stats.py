"""stats.nba.com through ``nba_api``: league-wide player game logs, Base/Advanced/Usage splits, team on/off splits.

stats.nba.com sits behind Akamai checks: cloud IPs hang, a bare or inconsistent header set gets a block page or a
hang, and throttling is undocumented. So every call carries the full nba.com header set (:data:`FULL_HEADERS`), calls
are paced ~0.6 s apart (``min_interval``), each has a hard timeout, and the TTLs are nightly-grade so a day's decisions
need one pull per dataset. Home IP only (DESIGN section 13); this is one reason the runtime is the local PC.

``nba_api`` supplies the endpoint parameter schemas (``LeagueGameLog``, ``LeagueDashPlayerStats`` and
``TeamPlayerOnOffDetails`` built with ``get_request=False``) and the HTTP layer (``NBAStatsHTTP`` over a ``requests``
session). This module owns caching, ``as_of`` stamps and parsing: the raw ``resultSets`` JSON is captured under
``cache_dir()/sources/nba_stats/`` and parsed into one Polars frame per result set, with columns named exactly as
stats.nba.com names them (``PLAYER_ID``, ``TEAM_ABBREVIATION``, ``MIN``, ``USG_PCT``, ...). A block page (HTML, or
``{"Message":"An error has occurred."}``) does not parse, so it is kept as ``.rejected.json`` and never replaces a
good copy; a timeout serves the last good copy as ``stale``.

Seasons are nba.com labels (``"2026-27"``); :func:`nba_season` converts ESPN's end-year ``seasonId`` (``2027``).

Offline tests inject a :class:`NbaStatsTransport` that serves recorded payloads; the default :class:`NbaApiTransport`
is covered by faking the ``requests`` session underneath ``nba_api``.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from datetime import date, datetime, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar, Literal, Protocol, Unpack

import polars as pl
from nba_api.stats.endpoints.leaguedashplayerstats import LeagueDashPlayerStats
from nba_api.stats.endpoints.leaguegamelog import LeagueGameLog
from nba_api.stats.endpoints.teamplayeronoffdetails import TeamPlayerOnOffDetails
from nba_api.stats.library.http import NBAStatsHTTP

from fm.sources.base import Fetched, FetchOptions, RateLimiter, Source, SourceSchemaError, parse_json, utcnow

logger = logging.getLogger(__name__)

type SeasonType = Literal["Regular Season", "Pre Season", "Playoffs", "PlayIn", "IST"]
type MeasureType = Literal["Base", "Advanced", "Usage"]
type PerMode = Literal["PerGame", "Totals", "Per36", "Per100Possessions"]
type PlayerOrTeam = Literal["P", "T"]

STATS_DATE_FORMAT = "%m/%d/%Y"
"""stats.nba.com ``DateFrom``/``DateTo`` format."""

FULL_HEADERS: Mapping[str, str] = MappingProxyType(
    {
        "Host": "stats.nba.com",
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/145.0.0.0 Safari/537.36"
        ),
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate, br",
        "Origin": "https://www.nba.com",
        "Referer": "https://www.nba.com/",
        "Connection": "keep-alive",
        "Pragma": "no-cache",
        "Cache-Control": "no-cache",
        "Sec-Ch-Ua": '"Google Chrome";v="145", "Chromium";v="145", "Not:A-Brand";v="99"',
        "Sec-Ch-Ua-Mobile": "?0",
        "Sec-Ch-Ua-Platform": '"Windows"',
        "Sec-Fetch-Dest": "empty",
        "Sec-Fetch-Mode": "cors",
        "Sec-Fetch-Site": "same-site",
        "x-nba-stats-origin": "stats",
        "x-nba-stats-token": "true",
    }
)
"""What nba.com's own front end sends to stats.nba.com. Anything less risks a block page or a hang."""

ON_OFF_RESULT_SETS = ("PlayersOnCourtTeamPlayerOnOffDetails", "PlayersOffCourtTeamPlayerOnOffDetails")


def nba_season(end_year: int) -> str:
    """nba.com season label from the season's end year: ``2027`` (ESPN's NBA ``seasonId``) becomes ``"2026-27"``."""
    if end_year < 1947:
        raise ValueError(f"not an NBA season end year: {end_year}")
    return f"{end_year - 1}-{end_year % 100:02d}"


def stats_date(value: date | None) -> str:
    """``MM/DD/YYYY`` for stats.nba.com date filters; ``None`` is the empty filter."""
    return value.strftime(STATS_DATE_FORMAT) if value is not None else ""


class NbaStatsTransport(Protocol):
    """One stats.nba.com request: the endpoint name and its parameters in, the raw body out."""

    def get(self, endpoint: str, parameters: Mapping[str, object]) -> bytes: ...


class NbaApiTransport:
    """The default transport: ``nba_api``'s ``NBAStatsHTTP`` (a shared ``requests`` session) with the full headers."""

    def __init__(self, *, headers: Mapping[str, str] = FULL_HEADERS, timeout: float = 30.0) -> None:
        self.headers = headers
        self.timeout = timeout

    def get(self, endpoint: str, parameters: Mapping[str, object]) -> bytes:
        response = NBAStatsHTTP().send_api_request(
            endpoint=endpoint,
            parameters=dict(parameters),
            headers=dict(self.headers),  # nba_api mutates the mapping it is given
            timeout=self.timeout,
        )
        return str(response.get_response()).encode("utf-8")


def _frame(headers: list[Any], rows: list[Any], label: str) -> pl.DataFrame:
    names = [str(header) for header in headers]
    for row in rows:
        if not isinstance(row, list) or len(row) != len(names):
            raise SourceSchemaError(f"{label}: a row does not match its {len(names)} headers")
    if not rows:
        return pl.DataFrame(schema=dict.fromkeys(names, pl.Null))
    return pl.DataFrame(rows, schema=names, orient="row", infer_schema_length=None)


def parse_result_sets(payload: bytes) -> dict[str, pl.DataFrame]:
    """Every ``resultSets`` entry as a frame keyed by its name. Column dtypes are inferred over all rows."""
    raw = parse_json(payload)
    if not isinstance(raw, dict):
        raise SourceSchemaError(f"nba_stats: expected a JSON object, got {type(raw).__name__}")
    sets = raw.get("resultSets", raw.get("resultSet"))
    if isinstance(sets, dict):
        sets = [sets]
    if not isinstance(sets, list) or not sets:
        raise SourceSchemaError("nba_stats: payload has no resultSets")
    frames: dict[str, pl.DataFrame] = {}
    for item in sets:
        if not isinstance(item, dict):
            raise SourceSchemaError("nba_stats: resultSets entries must be objects")
        name, headers, rows = item.get("name"), item.get("headers"), item.get("rowSet")
        if not isinstance(name, str) or not isinstance(headers, list) or not isinstance(rows, list):
            raise SourceSchemaError("nba_stats: a result set lacks name, headers or rowSet")
        frames[name] = _frame(headers, rows, f"nba_stats/{name}")
    return frames


def pick_result_set(frames: Mapping[str, pl.DataFrame], name: str, required: tuple[str, ...]) -> pl.DataFrame:
    """The named frame, checked for the columns the consumers join on."""
    frame = frames.get(name)
    if frame is None:
        raise SourceSchemaError(f"nba_stats: result set {name!r} missing; payload has {sorted(frames)}")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise SourceSchemaError(f"nba_stats/{name}: missing columns {missing}")
    return frame


def on_off_frame(frames: Mapping[str, pl.DataFrame]) -> pl.DataFrame:
    """The on-court and off-court result sets stacked, ``COURT_STATUS`` telling them apart, ``On`` before ``Off``."""
    parts = [
        pick_result_set(frames, name, ("VS_PLAYER_ID", "VS_PLAYER_NAME", "COURT_STATUS")) for name in ON_OFF_RESULT_SETS
    ]
    return pl.concat(parts, how="vertical_relaxed").sort(["VS_PLAYER_ID", "COURT_STATUS"], descending=[False, True])


class NbaStatsSource(Source):
    """stats.nba.com datasets, each a ``Fetched[pl.DataFrame]`` with stats.nba.com's own column names."""

    name: ClassVar[str] = "nba_stats"
    min_interval: ClassVar[float] = 0.6
    """Seconds between calls; stats.nba.com throttling is undocumented and this pace has held up for years."""
    timeout: ClassVar[float] = 30.0
    ttl: ClassVar[Mapping[str, timedelta]] = {
        "game_logs": timedelta(hours=6),  # box scores land overnight; one pull per morning
        "player_splits": timedelta(hours=6),
        "on_off": timedelta(hours=24),
    }

    def __init__(
        self,
        *,
        transport: NbaStatsTransport | None = None,
        cache_root: Path | None = None,
        limiter: RateLimiter | None = None,
        clock: Callable[[], datetime] = utcnow,
    ) -> None:
        super().__init__(cache_root=cache_root, limiter=limiter, clock=clock)
        self.transport: NbaStatsTransport = (
            transport if transport is not None else NbaApiTransport(timeout=self.timeout)
        )

    def game_logs(
        self,
        season: str,
        *,
        season_type: SeasonType = "Regular Season",
        player_or_team: PlayerOrTeam = "P",
        date_from: date | None = None,
        date_to: date | None = None,
        **options: Unpack[FetchOptions],
    ) -> Fetched[pl.DataFrame]:
        """``leaguegamelog``: one row per player-game (``"P"``) or team-game (``"T"``) with the box score line."""
        endpoint = LeagueGameLog(
            player_or_team_abbreviation=player_or_team,
            season=season,
            season_type_all_star=season_type,
            date_from_nullable=stats_date(date_from),
            date_to_nullable=stats_date(date_to),
            get_request=False,
        )
        key = _key(season, season_type, player_or_team, date_from, date_to)
        return self._dataset(
            "game_logs", key, endpoint, "LeagueGameLog", ("TEAM_ID", "GAME_ID", "GAME_DATE", "MIN", "PTS"), options
        )

    def player_splits(
        self,
        season: str,
        measure_type: MeasureType = "Base",
        *,
        per_mode: PerMode = "PerGame",
        season_type: SeasonType = "Regular Season",
        last_n_games: int = 0,
        date_from: date | None = None,
        date_to: date | None = None,
        **options: Unpack[FetchOptions],
    ) -> Fetched[pl.DataFrame]:
        """``leaguedashplayerstats``: one row per player. ``Base`` is the box score, ``Advanced`` adds ratings, pace,
        ``USG_PCT`` and ``POSS``, ``Usage`` is the share of team stats while on the floor."""
        endpoint = LeagueDashPlayerStats(
            measure_type_detailed_defense=measure_type,
            per_mode_detailed=per_mode,
            season=season,
            season_type_all_star=season_type,
            last_n_games=str(last_n_games),  # nba_api types every parameter as the string it sends
            date_from_nullable=stats_date(date_from),
            date_to_nullable=stats_date(date_to),
            get_request=False,
        )
        key = _key(season, season_type, measure_type, date_from, date_to, per_mode, f"last{last_n_games}")
        return self._dataset(
            "player_splits",
            key,
            endpoint,
            "LeagueDashPlayerStats",
            ("PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "GP", "MIN"),
            options,
        )

    def on_off(
        self,
        team_id: int,
        season: str,
        *,
        measure_type: MeasureType = "Base",
        per_mode: PerMode = "PerGame",
        season_type: SeasonType = "Regular Season",
        **options: Unpack[FetchOptions],
    ) -> Fetched[pl.DataFrame]:
        """``teamplayeronoffdetails``: the team's stats with each player on and off the court (``COURT_STATUS``)."""
        endpoint = TeamPlayerOnOffDetails(
            team_id=team_id,
            measure_type_detailed_defense=measure_type,
            per_mode_detailed=per_mode,
            season=season,
            season_type_all_star=season_type,
            get_request=False,
        )
        key = _key(str(team_id), season, season_type, measure_type, per_mode)
        return self._fetch_endpoint(
            "on_off", key, endpoint, parse=lambda payload: on_off_frame(parse_result_sets(payload)), options=options
        )

    def _dataset(
        self,
        dataset: str,
        key: str,
        endpoint: Any,
        result_set: str,
        required: tuple[str, ...],
        options: FetchOptions,
    ) -> Fetched[pl.DataFrame]:
        return self._fetch_endpoint(
            dataset,
            key,
            endpoint,
            parse=lambda payload: pick_result_set(parse_result_sets(payload), result_set, required),
            options=options,
        )

    def _fetch_endpoint(
        self,
        dataset: str,
        key: str,
        endpoint: Any,
        *,
        parse: Callable[[bytes], pl.DataFrame],
        options: FetchOptions,
    ) -> Fetched[pl.DataFrame]:
        name = str(endpoint.endpoint)
        parameters: dict[str, object] = dict(endpoint.parameters)
        sent = {param: value for param, value in parameters.items() if value not in (None, "")}
        return self.fetch(
            dataset,
            key,
            download=lambda: self.transport.get(name, parameters),
            parse=parse,
            meta={"endpoint": name, **sent},
            **options,
        )


def _key(*parts: object) -> str:
    return "_".join(
        part.isoformat() if isinstance(part, date) else str(part) for part in parts if part not in (None, "")
    )
