"""ESPN fantasy read client (DESIGN section 6.1): the league views over ``lm-api-reads.fantasy.espn.com``.

One :class:`EspnClient` serves one league for one season. Every method is a GET of
``/apis/v3/games/{ffl|fba}/seasons/{season}/segments/0/leagues/{league_id}?view=...`` (pro schedules use the season
URL without a league) with the ``espn_s2`` / ``SWID`` cookies from :class:`fm.espn.auth.EspnSession`, and returns an
:class:`EspnRead`: the typed model from :mod:`fm.espn.models` (or :class:`fm.espn.settings.LeagueSettings`), the
fetch time, and the :class:`RawCapture` of the response body. Every response body is saved under
``cache_dir()/espn/<game>/<season>/<league>/<view>/<timestamp>_<key>.json`` with a ``.meta.json`` beside it (URL,
query, filter, status, hash; never cookies), so a capture can become a fixture and a sync can index it as a
``raw_snapshots`` row.

Requests carry explicit timeouts, are paced ``min_interval_s`` apart, and retry transport errors, 429 and 5xx with
exponential backoff (``Retry-After`` wins when sent) up to ``max_attempts``. Reads are idempotent, so retrying is safe;
nothing here writes to ESPN. A 401 or an ``AUTH_MISSING_CREDENTIALS`` body means the session is gone and raises
:class:`EspnAuthError` at once (an :class:`fm.espn.auth.AuthError`, so ``run fm login`` handling is shared); other 4xx
raise :class:`EspnHttpError` without a retry; a body that does not parse raises :class:`EspnSchemaError` naming the
saved file.

Filtered views (``kona_player_info``, ``kona_playercard``, ``mTransactions2``, ``mMatchupScore``) take their filter in
the ``X-Fantasy-Filter`` header as compact JSON; :func:`player_filter`, :func:`transaction_filter` and
:func:`schedule_filter` build the shapes the ESPN web app and ``cwendt94/espn-api`` send. Pending offers have two
candidate views (``mPendingTransactions``, and ``mTransactions2`` filtered to ``PENDING``); both are exposed and
:meth:`EspnClient.pending_offers` merges them until the real-league capture (ROADMAP #14) settles which one ESPN serves.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any, TypedDict, Unpack

import httpx
from pydantic import BaseModel, ValidationError

from fm import paths
from fm.config import League
from fm.espn.auth import AuthError, EspnSession, is_auth_failure
from fm.espn.ids import Game
from fm.espn.models import (
    STAT_SOURCE_PROJECTED,
    STAT_SPLIT_SCORING_PERIOD,
    STAT_SPLIT_SEASON,
    TRANSACTION_PENDING,
    MatchupsView,
    PlayerStats,
    PlayersView,
    ProSchedule,
    RostersView,
    TeamsView,
    Transaction,
    TransactionsView,
)
from fm.espn.settings import LeagueSettings, SettingsParseError, parse_league_settings

logger = logging.getLogger(__name__)

type Primitive = str | int | float | bool | None
type QueryPairs = list[tuple[str, Primitive]]

READS_HOST = "lm-api-reads.fantasy.espn.com"
API_ROOT = f"https://{READS_HOST}/apis/v3/games"
FILTER_HEADER = "X-Fantasy-Filter"
CAPTURE_SUBDIR = "espn"
"""Raw responses live under ``cache_dir()/espn/``."""

DEFAULT_TIMEOUT = httpx.Timeout(30.0, connect=10.0)
DEFAULT_MAX_ATTEMPTS = 4
DEFAULT_BACKOFF_S = 1.0
"""Seconds before the first retry; doubles per attempt unless ESPN sends ``Retry-After``."""
DEFAULT_MIN_INTERVAL_S = 0.25
"""Polite pacing between requests. ESPN's limits are undocumented; the real-league capture measures them."""
RETRY_STATUSES: frozenset[int] = frozenset({429, 500, 502, 503, 504})

PLAYER_CARD_BATCH = 40
"""Most player ids per ``kona_playercard`` request (the batch size ``espn-api`` settled on)."""
DEFAULT_POOL_LIMIT = 50
FREE_AGENT_STATUSES: tuple[str, ...] = ("FREEAGENT", "WAIVERS")
DEFAULT_TRANSACTION_TYPES: tuple[str, ...] = (
    "FREEAGENT",
    "WAIVER",
    "WAIVER_ERROR",
    "TRADE_PROPOSAL",
    "TRADE_ACCEPT",
    "TRADE_DECLINE",
    "TRADE_VETO",
    "TRADE_UPHOLD",
)
"""Acquisitions (with bids) and trade steps; lineup moves (``ROSTER``) and the draft are left out."""
PENDING_OFFER_TYPES: tuple[str, ...] = ("WAIVER", "TRADE_PROPOSAL")


class View(StrEnum):
    """The ``view=`` values this client reads. :meth:`EspnClient.get_view` also accepts any other string."""

    SETTINGS = "mSettings"
    TEAM = "mTeam"
    STANDINGS = "mStandings"
    ROSTER = "mRoster"
    MATCHUP = "mMatchup"
    MATCHUP_SCORE = "mMatchupScore"
    SCOREBOARD = "mScoreboard"
    PLAYER_INFO = "kona_player_info"
    PLAYER_CARD = "kona_playercard"
    TRANSACTIONS = "mTransactions2"
    PENDING_TRANSACTIONS = "mPendingTransactions"
    PRO_SCHEDULES = "proTeamSchedules_wl"


# --- errors -----------------------------------------------------------------------------------------------------------


class EspnClientError(RuntimeError):
    """The ESPN API could not be read."""


class EspnHttpError(EspnClientError):
    """ESPN answered with an error status (or never answered, after the retries ran out)."""

    def __init__(self, status_code: int | None, url: str, detail: str) -> None:
        self.status_code = status_code
        self.url = url
        self.detail = detail
        status = f"HTTP {status_code}" if status_code is not None else "no response"
        super().__init__(f"{status} for {url}: {detail}")


class EspnAuthError(EspnClientError, AuthError):
    """ESPN rejected the session cookies: expired, revoked, or not allowed to see this league. Run ``fm login``."""


class EspnSchemaError(EspnClientError, ValueError):
    """A response did not have the expected shape. The raw body is kept in the cache for inspection."""


# --- filters ----------------------------------------------------------------------------------------------------------


def stat_entry_id(season: int, scoring_period: int = 0, *, projected: bool = False) -> str:
    """ESPN's composite stat-entry id: ``{source}{split}{season}[{period}]`` (``"1120264"`` = 2026 week 4 projection).

    These go in ``filterStatsForTopScoringPeriodIds.additionalValue`` to ask for specific stat lines.
    """
    source = STAT_SOURCE_PROJECTED if projected else 0
    split = STAT_SPLIT_SEASON if scoring_period == 0 else STAT_SPLIT_SCORING_PERIOD
    return f"{source}{split}{season}{scoring_period if scoring_period else ''}"


def player_filter(
    *,
    statuses: Sequence[str] = (),
    slot_ids: Sequence[int] = (),
    ids: Sequence[int] = (),
    limit: int | None = None,
    offset: int | None = None,
    sort_percent_owned: bool = False,
    top_scoring_periods: int | None = None,
    stat_ids: Sequence[str] = (),
) -> dict[str, Any]:
    """The ``players`` filter for ``kona_player_info`` / ``kona_playercard``, in the shape the ESPN web app sends.

    ``statuses`` are pool statuses (``FREEAGENT``, ``WAIVERS``, ``ONTEAM``); ``slot_ids`` restrict to players eligible
    for those lineup slots; ``ids`` picks specific players; ``stat_ids`` (see :func:`stat_entry_id`) requests specific
    stat lines on top of the ``top_scoring_periods`` most recent ones.
    """
    players: dict[str, Any] = {}
    if ids:
        players["filterIds"] = {"value": [int(player_id) for player_id in ids]}
    if statuses:
        players["filterStatus"] = {"value": list(statuses)}
    if slot_ids:
        players["filterSlotIds"] = {"value": [int(slot_id) for slot_id in slot_ids]}
    if top_scoring_periods is not None or stat_ids:
        players["filterStatsForTopScoringPeriodIds"] = {
            "value": top_scoring_periods if top_scoring_periods is not None else 1,
            "additionalValue": list(stat_ids),
        }
    if limit is not None:
        players["limit"] = int(limit)
    if offset is not None:
        players["offset"] = int(offset)
    if sort_percent_owned:
        players["sortPercOwned"] = {"sortPriority": 1, "sortAsc": False}
        players["sortDraftRanks"] = {"sortPriority": 100, "sortAsc": True, "value": "STANDARD"}
    return {"players": players}


def transaction_filter(types: Sequence[str]) -> dict[str, Any]:
    """The ``transactions`` filter for ``mTransactions2``: which transaction types to return."""
    return {"transactions": {"filterType": {"value": list(types)}}}


def schedule_filter(matchup_periods: Sequence[int]) -> dict[str, Any]:
    """The ``schedule`` filter for ``mMatchupScore`` / ``mScoreboard``: which matchup periods to return."""
    return {"schedule": {"filterMatchupPeriodIds": {"value": [int(period) for period in matchup_periods]}}}


def filter_header(filter: Mapping[str, Any]) -> dict[str, str]:
    """The ``X-Fantasy-Filter`` header carrying a filter as compact JSON."""
    return {FILTER_HEADER: json.dumps(filter, separators=(",", ":"))}


# --- results ----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RawCapture:
    """A saved response body. ``params`` holds the query and filter, never cookies or tokens."""

    kind: str
    path: Path
    root: Path
    url: str
    params: dict[str, Any]
    status_code: int
    sha256: str
    size_bytes: int
    fetched_at: datetime
    league_id: int | None = None
    scoring_period_id: int | None = None

    @property
    def relative_path(self) -> str:
        """``path`` relative to the cache root, as the ``raw_snapshots`` table stores it."""
        return self.path.relative_to(self.root).as_posix()

    @property
    def meta_path(self) -> Path:
        return _meta_path(self.path)


@dataclass(frozen=True, slots=True)
class EspnRead[T]:
    """A parsed ESPN response plus when it was fetched and where its raw body is."""

    data: T
    as_of: datetime
    capture: RawCapture | None = None
    """``None`` only when the client was created with ``capture=False``."""


# --- helpers ----------------------------------------------------------------------------------------------------------

_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")
_STAMP = "%Y%m%dT%H%M%S%fZ"


def _safe(value: str) -> str:
    return _UNSAFE.sub("_", value).strip("._") or "default"


def _meta_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.meta.json")


def _atomic_write(path: Path, data: bytes) -> None:
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


def _user_agent() -> str:
    try:
        installed = distribution_version("espn-fantasy")
    except PackageNotFoundError:
        installed = "0+unknown"
    return f"espn-fantasy/{installed}"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _retry_after(response: httpx.Response) -> float | None:
    header = response.headers.get("Retry-After")
    if header is None:
        return None
    try:
        return max(0.0, float(header))
    except ValueError:
        return None


def _error_detail(response: httpx.Response) -> str:
    """ESPN's error payload (``details[].type``/``message`` or ``messages[]``) as one line; never the whole body."""
    try:
        data = response.json()
    except ValueError:
        text = response.text.strip()
        return text[:200] if text else (response.reason_phrase or "no body")
    parts: list[str] = []
    if isinstance(data, Mapping):
        details = data.get("details")
        if isinstance(details, list):
            for detail in details:
                if isinstance(detail, Mapping):
                    kind = detail.get("type")
                    message = detail.get("message") or detail.get("shortMessage")
                    parts.append(": ".join(str(part) for part in (kind, message) if part))
        if not parts and isinstance(data.get("messages"), list):
            parts.extend(str(message) for message in data["messages"])
    joined = "; ".join(part for part in parts if part)
    return joined or response.reason_phrase or f"status {response.status_code}"


def _describe_validation(exc: ValidationError, limit: int = 5) -> str:
    errors = exc.errors(include_url=False)
    shown = []
    for error in errors[:limit]:
        loc = ".".join(str(part) for part in error["loc"]) or "<root>"
        shown.append(f"{loc}: {error['msg']}")
    more = f" (+{len(errors) - limit} more)" if len(errors) > limit else ""
    return f"{len(errors)} problem{'s' if len(errors) != 1 else ''}: " + "; ".join(shown) + more


class ClientOptions(TypedDict, total=False):
    """Keyword options shared by :class:`EspnClient` and :meth:`EspnClient.for_league`."""

    client: httpx.Client | None
    cache_root: Path | None
    capture: bool
    timeout: httpx.Timeout | float
    max_attempts: int
    backoff_s: float
    min_interval_s: float
    sleep: Callable[[float], object]
    clock: Callable[[], datetime]
    monotonic: Callable[[], float]


# --- client -----------------------------------------------------------------------------------------------------------


class EspnClient:
    """Reads one league's views for one season. Use as a context manager, or call :meth:`close`.

    ``session`` carries the cookies; ``None`` works only for public leagues. ``client`` may be an injected
    ``httpx.Client`` (tests, or a shared one), which is then left open on :meth:`close`. ``cache_root`` defaults to
    ``fm.paths.cache_dir()``; ``capture=False`` skips saving bodies. ``sleep``/``clock``/``monotonic`` exist so tests
    can run the backoff and pacing without waiting.
    """

    def __init__(
        self,
        game: Game | str,
        league_id: int,
        season: int,
        session: EspnSession | None = None,
        **options: Unpack[ClientOptions],
    ) -> None:
        if league_id <= 0:
            raise ValueError(f"league_id must be positive, got {league_id!r}")
        if season <= 0:
            raise ValueError(f"season must be positive, got {season!r}")
        self.game = Game.coerce(game)
        self.league_id = league_id
        self.season = season
        self._cookie_header = session.cookie_header() if session is not None else None
        self._client = options.get("client")
        self._owns_client = self._client is None
        root = options.get("cache_root")
        self.cache_root = root if root is not None else paths.cache_dir()
        self.capture_enabled = options.get("capture", True)
        self.timeout = options.get("timeout", DEFAULT_TIMEOUT)
        self.max_attempts = max(1, options.get("max_attempts", DEFAULT_MAX_ATTEMPTS))
        self.backoff_s = options.get("backoff_s", DEFAULT_BACKOFF_S)
        self.min_interval_s = options.get("min_interval_s", DEFAULT_MIN_INTERVAL_S)
        self._sleep = options.get("sleep", time.sleep)
        self._clock = options.get("clock", _utcnow)
        self._monotonic = options.get("monotonic", time.monotonic)
        self._last_request: float | None = None
        self.requests = 0
        """HTTP requests sent (attempts included), for pacing checks and tests."""

    @classmethod
    def for_league(
        cls, league: League, session: EspnSession | None = None, **options: Unpack[ClientOptions]
    ) -> EspnClient:
        """A client for a configured league (``fm.config.League``)."""
        return cls(league.game, league.espn_league_id, league.season, session, **options)

    def __repr__(self) -> str:
        return f"EspnClient({self.game.value} league {self.league_id}, season {self.season})"

    @property
    def league_url(self) -> str:
        return f"{API_ROOT}/{self.game.value}/seasons/{self.season}/segments/0/leagues/{self.league_id}"

    @property
    def season_url(self) -> str:
        """Game-level data for the season (pro schedules), not scoped to the league."""
        return f"{API_ROOT}/{self.game.value}/seasons/{self.season}"

    @property
    def client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout,
                headers={"User-Agent": _user_agent(), "Accept": "application/json"},
                follow_redirects=False,
            )
        return self._client

    def close(self) -> None:
        if self._client is not None and self._owns_client:
            self._client.close()
            self._client = None

    def __enter__(self) -> EspnClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # --- typed views ------------------------------------------------------------------------------------------------

    def settings(self) -> EspnRead[LeagueSettings]:
        """``mSettings`` parsed by :func:`fm.espn.settings.parse_league_settings`."""
        read = self.get_view(View.SETTINGS)
        try:
            parsed = parse_league_settings(read.data, game=self.game)
        except SettingsParseError as exc:
            raise EspnSchemaError(f"{View.SETTINGS}: {exc}{self._saved_at(read)}") from exc
        return EspnRead(parsed, read.as_of, read.capture)

    def teams(self) -> EspnRead[TeamsView]:
        """``mTeam`` + ``mStandings``: teams with records, seeds, waiver ranks, FAAB spent, and the members."""
        return self._typed(self.get_view(View.TEAM, View.STANDINGS), TeamsView)

    def rosters(self, scoring_period: int | None = None) -> EspnRead[RostersView]:
        """``mRoster`` for a scoring period (ESPN's current one when omitted)."""
        read = self.get_view(View.ROSTER, scoring_period=scoring_period, key=_period_key(scoring_period))
        return self._typed(read, RostersView)

    def matchups(self, matchup_period: int | None = None) -> EspnRead[MatchupsView]:
        """``mMatchup``: the season schedule with scores, narrowed to one matchup period when given."""
        read = self._typed(self.get_view(View.MATCHUP), MatchupsView)
        if matchup_period is None:
            return read
        narrowed = read.data.model_copy(update={"schedule": read.data.for_period(matchup_period)})
        return EspnRead(narrowed, read.as_of, read.capture)

    def scoreboard(self, matchup_period: int, scoring_period: int | None = None) -> EspnRead[MatchupsView]:
        """``mMatchupScore`` + ``mScoreboard`` for one matchup period: live scores with each side's lineup."""
        read = self.get_view(
            View.MATCHUP_SCORE,
            View.SCOREBOARD,
            scoring_period=scoring_period,
            filter=schedule_filter([matchup_period]),
            key=f"mp{matchup_period}{_period_key(scoring_period, prefix='_')}",
        )
        return self._typed(read, MatchupsView)

    def free_agents(
        self,
        scoring_period: int | None = None,
        *,
        statuses: Sequence[str] = FREE_AGENT_STATUSES,
        slot_ids: Sequence[int] = (),
        limit: int = DEFAULT_POOL_LIMIT,
        offset: int = 0,
    ) -> EspnRead[PlayersView]:
        """``kona_player_info``: the player pool filtered by status and slot, most-owned first."""
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit!r}")
        if offset < 0:
            raise ValueError(f"offset must be >= 0, got {offset!r}")
        filter = player_filter(
            statuses=statuses, slot_ids=slot_ids, limit=limit, offset=offset, sort_percent_owned=True
        )
        slots = "-".join(str(slot) for slot in slot_ids)
        key = f"{'_'.join(status.lower() for status in statuses) or 'all'}{_period_key(scoring_period, prefix='_')}"
        key += f"_slots{slots}" if slots else ""
        key += f"_o{offset}" if offset else ""
        read = self.get_view(View.PLAYER_INFO, scoring_period=scoring_period, filter=filter, key=key)
        return self._typed(read, PlayersView)

    def player_cards(
        self,
        player_ids: Sequence[int],
        *,
        scoring_period: int | None = None,
        top_scoring_periods: int | None = None,
    ) -> EspnRead[PlayersView]:
        """``kona_playercard`` for up to :data:`PLAYER_CARD_BATCH` players: ownership, ranks and stat lines.

        Asks for the season actual and projected lines, plus the actual and projected lines of ``scoring_period``
        when given. ``top_scoring_periods`` (default: ``scoring_period``, else 1) is ESPN's "most recent N periods"
        knob.
        """
        ids = _player_ids(player_ids)
        if len(ids) > PLAYER_CARD_BATCH:
            raise ValueError(f"at most {PLAYER_CARD_BATCH} player ids per player-card request, got {len(ids)}")
        stat_ids = [stat_entry_id(self.season), stat_entry_id(self.season, projected=True)]
        if scoring_period:
            stat_ids.append(stat_entry_id(self.season, scoring_period, projected=True))
            stat_ids.append(stat_entry_id(self.season, scoring_period))
        top = top_scoring_periods if top_scoring_periods is not None else (scoring_period or 1)
        filter = player_filter(ids=ids, top_scoring_periods=top, stat_ids=stat_ids)
        digest = hashlib.sha256(",".join(str(player_id) for player_id in ids).encode()).hexdigest()[:8]
        key = f"{len(ids)}ids_{digest}{_period_key(scoring_period, prefix='_')}"
        read = self.get_view(View.PLAYER_CARD, scoring_period=scoring_period, filter=filter, key=key)
        return self._typed(read, PlayersView)

    def projections(self, player_ids: Sequence[int], scoring_period: int) -> dict[int, PlayerStats]:
        """ESPN's projected stat lines for one scoring period, by player id, fetched in card batches.

        Players ESPN has no projection for (no game, not yet published) are left out.
        """
        ids = _player_ids(player_ids)
        projections: dict[int, PlayerStats] = {}
        for start in range(0, len(ids), PLAYER_CARD_BATCH):
            batch = ids[start : start + PLAYER_CARD_BATCH]
            for entry in self.player_cards(batch, scoring_period=scoring_period).data.players:
                line = entry.player.projection(self.season, scoring_period)
                if line is not None:
                    projections[entry.id] = line
        return projections

    def transactions(
        self,
        scoring_period: int | None = None,
        *,
        types: Sequence[str] = DEFAULT_TRANSACTION_TYPES,
        statuses: Sequence[str] | None = None,
    ) -> EspnRead[TransactionsView]:
        """``mTransactions2`` for a scoring period, filtered by type on ESPN's side and by ``statuses`` on ours.

        Waiver claims carry ``bid_amount`` whether they won or failed.
        """
        if not types:
            raise ValueError("at least one transaction type is required")
        key = f"{'_'.join(t.lower() for t in types) if len(types) <= 3 else f'{len(types)}types'}"
        key += _period_key(scoring_period, prefix="_")
        read = self.get_view(
            View.TRANSACTIONS, scoring_period=scoring_period, filter=transaction_filter(types), key=key
        )
        typed = self._typed(read, TransactionsView)
        if statuses is None:
            return typed
        wanted = set(statuses)
        kept = tuple(t for t in typed.data.transactions if t.status in wanted)
        return EspnRead(typed.data.model_copy(update={"transactions": kept}), typed.as_of, typed.capture)

    def pending_transactions(self, scoring_period: int | None = None) -> EspnRead[TransactionsView]:
        """``mPendingTransactions``: the first candidate view for open offers and claims (unconfirmed, see #14)."""
        read = self.get_view(
            View.PENDING_TRANSACTIONS, scoring_period=scoring_period, key=_period_key(scoring_period) or "current"
        )
        return self._typed(read, TransactionsView)

    def pending_offers(self, scoring_period: int | None = None) -> tuple[Transaction, ...]:
        """Open trade offers and waiver claims from both candidate views, once each, oldest first.

        Reads ``mPendingTransactions`` and ``mTransactions2`` (waivers and trade proposals, ``PENDING`` only) and
        merges them by transaction id. Both bodies are captured, so the real-league spike can compare them.
        """
        merged: dict[str, Transaction] = {}
        for transaction in self.pending_transactions(scoring_period).data.transactions:
            if transaction.pending or transaction.status is None:
                merged.setdefault(transaction.id, transaction)
        filtered = self.transactions(scoring_period, types=PENDING_OFFER_TYPES, statuses=(TRANSACTION_PENDING,))
        for transaction in filtered.data.transactions:
            merged.setdefault(transaction.id, transaction)
        return tuple(sorted(merged.values(), key=_transaction_order))

    def pro_schedule(self) -> EspnRead[ProSchedule]:
        """``proTeamSchedules_wl``: every pro team's games by scoring period (lock times, byes, first tips)."""
        return self._typed(self.get_game_view(View.PRO_SCHEDULES), ProSchedule)

    # --- raw views --------------------------------------------------------------------------------------------------

    def get_view(
        self,
        *views: str,
        scoring_period: int | None = None,
        filter: Mapping[str, Any] | None = None,
        key: str = "",
    ) -> EspnRead[dict[str, Any]]:
        """GET league ``views`` (repeated ``view=`` params) and return the JSON object, captured.

        ``filter`` goes in the ``X-Fantasy-Filter`` header; ``key`` names the capture file. Anyone needing a view this
        module does not type (the real-league capture script) goes through here.
        """
        return self._read(
            self.league_url, views, scoring_period=scoring_period, filter=filter, league_id=self.league_id, key=key
        )

    def get_game_view(
        self, *views: str, filter: Mapping[str, Any] | None = None, key: str = ""
    ) -> EspnRead[dict[str, Any]]:
        """GET season-level ``views`` (``/seasons/{season}?view=...``), which are not scoped to a league."""
        return self._read(self.season_url, views, scoring_period=None, filter=filter, league_id=None, key=key)

    # --- internals --------------------------------------------------------------------------------------------------

    def _read(
        self,
        url: str,
        views: Sequence[str],
        *,
        scoring_period: int | None,
        filter: Mapping[str, Any] | None,
        league_id: int | None,
        key: str,
    ) -> EspnRead[dict[str, Any]]:
        if not views:
            raise ValueError("at least one view is required")
        view_names = [str(view) for view in views]
        kind = "+".join(view_names)
        params: QueryPairs = [("view", view) for view in view_names]
        record: dict[str, Any] = {"view": view_names}
        if scoring_period is not None:
            params.append(("scoringPeriodId", scoring_period))
            record["scoringPeriodId"] = scoring_period
        headers: dict[str, str] = {}
        if filter:
            headers.update(filter_header(filter))
            record["filter"] = json.loads(json.dumps(filter))  # a plain copy for the meta file
        if self._cookie_header is not None:
            headers["Cookie"] = self._cookie_header

        response = self._request(url, params=params, headers=headers)
        fetched_at = self._clock()
        body = response.content
        capture = (
            self._capture(kind, body, response, record, fetched_at, league_id, scoring_period, key)
            if self.capture_enabled
            else None
        )
        try:
            data = json.loads(body)
        except ValueError as exc:
            raise EspnSchemaError(f"{kind}: response is not JSON ({exc}){_saved_at(capture)}") from exc
        if isinstance(data, list) and len(data) == 1 and isinstance(data[0], dict):
            data = data[0]  # the leagueHistory endpoint wraps the league in a list
        if not isinstance(data, dict):
            raise EspnSchemaError(f"{kind}: expected a JSON object, got {type(data).__name__}{_saved_at(capture)}")
        return EspnRead(data, fetched_at, capture)

    def _request(self, url: str, *, params: QueryPairs, headers: Mapping[str, str]) -> httpx.Response:
        """GET with pacing, bounded retries on transport errors / 429 / 5xx, and no retry on other errors."""
        self._pace()
        last_failure = ""
        last_status: int | None = None
        for attempt in range(1, self.max_attempts + 1):
            self.requests += 1
            try:
                response = self.client.get(url, params=params, headers=headers)
            except httpx.TransportError as exc:
                last_failure = f"{type(exc).__name__}: {exc}".strip(": ")
                last_status = None
                delay: float | None = None
                logger.warning("espn: %s (attempt %d/%d)", last_failure, attempt, self.max_attempts)
            else:
                status = response.status_code
                shown_url = str(response.request.url)
                if status >= 400 and is_auth_failure(status, response.content):
                    raise EspnAuthError(
                        f"ESPN rejected the session (HTTP {status}: {_error_detail(response)}); run `fm login`"
                    )
                if status not in RETRY_STATUSES:
                    if not response.is_success:
                        raise EspnHttpError(status, shown_url, _error_detail(response))
                    logger.debug("espn: GET %s -> %d (%d bytes)", shown_url, status, len(response.content))
                    return response
                last_failure = f"HTTP {status}"
                last_status = status
                delay = _retry_after(response)
                logger.warning("espn: HTTP %d for %s (attempt %d/%d)", status, shown_url, attempt, self.max_attempts)
            if attempt < self.max_attempts:
                self._sleep(delay if delay is not None else self.backoff_s * 2 ** (attempt - 1))
        shown = str(httpx.URL(url, params=httpx.QueryParams(params)))
        raise EspnHttpError(last_status, shown, f"gave up after {self.max_attempts} attempts: {last_failure}")

    def _pace(self) -> None:
        now = self._monotonic()
        if self._last_request is not None:
            wait = self.min_interval_s - (now - self._last_request)
            if wait > 0:
                self._sleep(wait)
                now += wait
        self._last_request = now

    def _capture(
        self,
        kind: str,
        body: bytes,
        response: httpx.Response,
        record: Mapping[str, Any],
        fetched_at: datetime,
        league_id: int | None,
        scoring_period: int | None,
        key: str,
    ) -> RawCapture | None:
        folder = (
            self.cache_root
            / CAPTURE_SUBDIR
            / self.game.value
            / str(self.season)
            / (str(league_id) if league_id is not None else "game")
            / _safe(kind)
        )
        stem = fetched_at.astimezone(UTC).strftime(_STAMP) + (f"_{_safe(key)}" if key else "")
        path = folder / f"{stem}.json"
        counter = 1
        while path.exists():
            path = folder / f"{stem}-{counter}.json"
            counter += 1
        sha = hashlib.sha256(body).hexdigest()
        meta = {
            "kind": kind,
            "game": self.game.value,
            "season": self.season,
            "league_id": league_id,
            "scoring_period_id": scoring_period,
            "url": str(response.request.url),
            "params": dict(record),
            "status_code": response.status_code,
            "fetched_at": fetched_at.isoformat(),
            "sha256": sha,
            "bytes": len(body),
        }
        try:
            folder.mkdir(parents=True, exist_ok=True)
            _atomic_write(path, body)
            _atomic_write(_meta_path(path), json.dumps(meta, indent=1, sort_keys=True).encode("utf-8"))
        except OSError as exc:
            logger.warning("espn: could not save %s capture to %s: %s", kind, path, exc)
            return None
        return RawCapture(
            kind=kind,
            path=path,
            root=self.cache_root,
            url=str(response.request.url),
            params=dict(record),
            status_code=response.status_code,
            sha256=sha,
            size_bytes=len(body),
            fetched_at=fetched_at,
            league_id=league_id,
            scoring_period_id=scoring_period,
        )

    def _typed[M: BaseModel](self, read: EspnRead[dict[str, Any]], model: type[M]) -> EspnRead[M]:
        kind = read.capture.kind if read.capture is not None else model.__name__
        try:
            parsed = model.model_validate(read.data)
        except ValidationError as exc:
            problems = _describe_validation(exc)
            raise EspnSchemaError(
                f"{kind}: response did not parse as {model.__name__} ({problems}){self._saved_at(read)}"
            ) from exc
        return EspnRead(parsed, read.as_of, read.capture)

    @staticmethod
    def _saved_at(read: EspnRead[Any]) -> str:
        return _saved_at(read.capture)


def _saved_at(capture: RawCapture | None) -> str:
    return f"; raw body saved at {capture.path}" if capture is not None else ""


def _period_key(scoring_period: int | None, *, prefix: str = "") -> str:
    return f"{prefix}sp{scoring_period}" if scoring_period is not None else ""


def _player_ids(player_ids: Sequence[int]) -> list[int]:
    ids: list[int] = []
    for player_id in player_ids:
        value = int(player_id)
        if value not in ids:
            ids.append(value)
    if not ids:
        raise ValueError("at least one player id is required")
    return ids


def _transaction_order(transaction: Transaction) -> tuple[int, datetime, str]:
    date = transaction.date
    return (0 if date is not None else 1, date or datetime.min.replace(tzinfo=UTC), transaction.id)
