"""Row models for the v1 schema: one frozen pydantic model per table, with field names equal to column names.

Conventions
-----------
- Timestamps are ``UtcDatetime``: timezone-aware only (naive values are rejected), normalised to UTC on validation and
  stored as fixed-width ISO-8601 text (``2026-10-04T13:00:00.000000Z``), so text order is time order and SQLite's date
  functions can read them. ``as_of`` is the source's freshness stamp; ``created_at`` / ``fetched_at`` are our clock.
- JSON columns (``JsonDict``, ``StatLine``, ``IntList``, ``StrList``) are dicts and lists on the model and compact JSON
  text in the database; they parse themselves when a row is read back.
- ``sport`` is ``fm.config.Sport`` (``"nfl"`` or ``"nba"``; ESPN games ``ffl`` / ``fba``). ESPN ids are ints.
  ``scoring_period_id`` is a week in NFL and a day in NBA; ``0`` means the full season.
- Models are frozen. Change a row with ``model_copy(update=...)`` and hand it back to its repository. Rows with a
  store-assigned ``id`` (``Identified``) expose ``row_id`` for the common "I know this came from the store" case.
- Every table row is named ``<Table>Row`` so it never shadows a config model (``fm.config.League``) or a parsed
  ESPN model; a module may import ``fm.config``, ``fm.store`` and ``fm.espn`` together without aliasing.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Annotated, Any, Literal

from pydantic import (
    AfterValidator,
    AwareDatetime,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PlainSerializer,
)

from fm.config import Sport

TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


def utc_now() -> datetime:
    """The current time, timezone-aware in UTC."""
    return datetime.now(UTC)


def format_timestamp(value: datetime) -> str:
    """Encode an aware datetime the way the database stores it: fixed width, UTC, microseconds, trailing ``Z``."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value.astimezone(UTC).strftime(TIMESTAMP_FORMAT)


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


def _parse_json(value: Any) -> Any:
    if isinstance(value, str | bytes | bytearray):
        return json.loads(value)
    return value


type UtcDatetime = Annotated[
    AwareDatetime, AfterValidator(_to_utc), PlainSerializer(format_timestamp, return_type=str, when_used="json")
]
type JsonDict = Annotated[dict[str, Any], BeforeValidator(_parse_json)]
type StatLine = Annotated[dict[str, float], BeforeValidator(_parse_json)]
type IntList = Annotated[list[int], BeforeValidator(_parse_json)]
type StrList = Annotated[list[str], BeforeValidator(_parse_json)]

type ProjectionKind = Literal["projected", "actual"]
type ProposalPolicy = Literal["off", "approve", "auto"]  # the approval policy a proposal was made under
type ProposalStatus = Literal["proposed", "approved", "rejected", "expired", "executing", "verified", "failed"]
type ExecutionMode = Literal["api", "ui"]
type ExecutionStatus = Literal["running", "verified", "failed", "unknown", "dry_run"]
type SignalKind = Literal["injury", "role", "rest", "suspension", "other"]

SPORTS: tuple[Sport, ...] = ("nfl", "nba")
OPEN_PROPOSAL_STATUSES: tuple[ProposalStatus, ...] = ("proposed", "approved", "executing")


class Row(BaseModel):
    """Base for every table row: frozen, and unknown fields are an error, so schema/model drift fails on read."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Identified(Row):
    """A row whose ``id`` the store assigns on insert. ``id`` is ``None`` until then."""

    id: int | None = None

    @property
    def row_id(self) -> int:
        """The assigned ``id``; raises ``ValueError`` for a row that has not been stored yet."""
        if self.id is None:
            raise ValueError(f"{type(self).__name__} has no id yet; store it first")
        return self.id


class LeagueRow(Identified):
    """A configured league for one season (``leagues``). ``key`` is the config key (``nfl``, ``nba``) and, with
    ``season``, the league's identity: correcting ``espn_league_id`` in config.toml updates the row in place.
    ``team_id`` is our team. (sport, espn_league_id, season) is unique as well."""

    key: str
    sport: Sport
    espn_league_id: int
    season: int
    team_id: int
    name: str | None = None
    as_of: UtcDatetime


class LeagueSettingsRow(Row):
    """Parsed ESPN ``mSettings`` for a league (``league_settings``): scoring items, slot counts, lock type, waiver and
    acquisition rules, trade deadline, playoff weeks. ``settings`` is the parser's output as a dict, so the store does
    not depend on the parser's model; ``raw_snapshot_id`` points at the response it was parsed from."""

    league_id: int
    settings: JsonDict
    raw_snapshot_id: int | None = None
    as_of: UtcDatetime


class TeamRow(Row):
    """An ESPN team in a league with its standings and FAAB state (``teams``)."""

    league_id: int
    team_id: int
    name: str
    abbrev: str | None = None
    division_id: int | None = None
    wins: int = 0
    losses: int = 0
    ties: int = 0
    points_for: float = 0.0
    points_against: float = 0.0
    playoff_seed: int | None = None
    waiver_rank: int | None = None
    acquisition_budget_spent: int = 0
    as_of: UtcDatetime


class PlayerRow(Row):
    """The canonical player row, keyed by ESPN id per sport (``players``). ``position`` / ``pro_team`` are the decoded
    names of ``default_position_id`` / ``pro_team_id``; ``eligible_slot_ids`` are ESPN lineup slot ids."""

    sport: Sport
    espn_id: int
    full_name: str
    default_position_id: int | None = None
    position: str | None = None
    pro_team_id: int | None = None
    pro_team: str | None = None
    eligible_slot_ids: IntList = Field(default_factory=list)
    injury_status: str | None = None
    injured: bool = False
    active: bool = True
    as_of: UtcDatetime


class PlayerIdRow(Row):
    """One crosswalk entry: the player's id in another source (``player_ids``). ``source`` names the id system
    (``gsis``, ``sleeper``, ``nba``, ...); ``origin`` says where the mapping came from (``ff_playerids``, ``override``,
    ``name_match``). A source id maps to one ESPN player."""

    sport: Sport
    espn_id: int
    source: str
    source_id: str
    origin: str
    as_of: UtcDatetime


class RosterEntryRow(Row):
    """One player on one team's roster for one scoring period (``roster_snapshots``). A team's rows for a period are
    replaced together, so they always describe the latest read of that roster."""

    league_id: int
    scoring_period_id: int
    team_id: int
    espn_id: int
    lineup_slot_id: int
    acquisition_type: str | None = None
    acquisition_date: UtcDatetime | None = None
    lineup_locked: bool = False
    as_of: UtcDatetime


class ProjectionRow(Row):
    """A stat line for one player, period and source (``projections``), never points: each league turns stats into
    points with its own scoring items. ``kind`` separates projections from actuals; period ``0`` is the season."""

    sport: Sport
    espn_id: int
    source: str
    kind: ProjectionKind = "projected"
    season: int
    scoring_period_id: int
    stats: StatLine
    as_of: UtcDatetime


class AvailabilityRow(Row):
    """Probability a player is active in a scoring period (``availability``), with the designation and the inputs
    behind the number (practice trend, injury report, applied news signals) so the decision is replayable."""

    sport: Sport
    espn_id: int
    season: int
    scoring_period_id: int
    designation: str | None = None
    p_active: float = Field(ge=0.0, le=1.0)
    has_game: bool = True
    game_time: UtcDatetime | None = None
    inputs: JsonDict = Field(default_factory=dict)
    as_of: UtcDatetime


class NewsItemRow(Identified):
    """A deduplicated news item (``news_items``): (source, external_id) is unique. ``triaged_at`` is set once the
    advisor has processed it."""

    source: str
    external_id: str
    sport: Sport
    title: str
    body: str | None = None
    url: str | None = None
    espn_ids: IntList = Field(default_factory=list)
    published_at: UtcDatetime
    fetched_at: UtcDatetime
    triaged_at: UtcDatetime | None = None


class NewsSignalRow(Identified):
    """The advisor's structured reading of a news item for one player (``news_signals``, DESIGN section 10).
    ``p_active_delta`` is the proposed adjustment; the availability model clamps and logs what it applies."""

    news_item_id: int
    sport: Sport
    espn_id: int
    kind: SignalKind
    severity: str
    games_out: int | None = None
    p_active_delta: float = 0.0
    confidence: float = Field(ge=0.0, le=1.0)
    summary: str | None = None
    source_url: str | None = None
    published_at: UtcDatetime
    created_at: UtcDatetime


class MarketValueRow(Row):
    """What league-mates believe a player is worth (``market_values``): redraft trade value, ranks and ownership
    trends per source. Used only to model trade acceptance."""

    sport: Sport
    espn_id: int
    source: str
    value: float | None = None
    rank: int | None = None
    position_rank: int | None = None
    trend: float | None = None
    percent_owned: float | None = None
    percent_started: float | None = None
    details: JsonDict = Field(default_factory=dict)
    as_of: UtcDatetime


class ProposalRow(Identified):
    """A proposed move (``proposals``, DESIGN section 11): ``proposed -> approved | rejected | expired -> executing ->
    verified | failed``. ``execution_token`` is issued on approval and consumed exactly once; ``dedupe_key`` lets a
    decision module recognise a move it already proposed."""

    league_id: int
    kind: str
    status: ProposalStatus = "proposed"
    policy: ProposalPolicy
    scoring_period_id: int | None = None
    payload: JsonDict
    engine_numbers: JsonDict = Field(default_factory=dict)
    rationale: str | None = None
    deadline: UtcDatetime | None = None
    created_by: str
    created_at: UtcDatetime
    decided_by: str | None = None
    decided_at: UtcDatetime | None = None
    execution_token: str | None = None
    token_consumed_at: UtcDatetime | None = None
    dedupe_key: str | None = None


class ExecutionRow(Identified):
    """One attempt to carry out a proposal (``executions``): the request and response, the API re-read that verified
    it, and audit artifact paths. ``unknown`` means a timeout; state must be re-read before anything else happens."""

    proposal_id: int
    mode: ExecutionMode
    status: ExecutionStatus = "running"
    started_at: UtcDatetime
    finished_at: UtcDatetime | None = None
    request: JsonDict | None = None
    response: JsonDict | None = None
    error: str | None = None
    verification: JsonDict | None = None
    artifacts: StrList = Field(default_factory=list)
    espn_transaction_id: str | None = None


class DecisionEvalRow(Identified):
    """A replayable decision (``decision_evals``): its inputs with their ``as_of`` stamps, what was decided, and, once
    results are in, the outcome and metrics (lineup efficiency, regret, pickup value)."""

    league_id: int
    kind: str
    season: int
    scoring_period_id: int | None = None
    proposal_id: int | None = None
    decided_at: UtcDatetime
    inputs: JsonDict
    decision: JsonDict
    outcome: JsonDict | None = None
    metrics: JsonDict | None = None
    evaluated_at: UtcDatetime | None = None


class LlmUsageRow(Identified):
    """One Claude API call (``llm_usage``), for the daily budget cap and cost reporting."""

    called_at: UtcDatetime
    worker: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    cost_usd: float = 0.0
    batch: bool = False
    stop_reason: str | None = None
    request_id: str | None = None
    league_id: int | None = None


class RawSnapshotRow(Identified):
    """Index entry for a raw response saved under the cache dir (``raw_snapshots``). ``path`` is relative to
    ``paths.cache_dir()``; ``params`` holds the query and filters, never cookies or tokens."""

    source: str
    kind: str
    league_id: int | None = None
    scoring_period_id: int | None = None
    url: str | None = None
    params: JsonDict = Field(default_factory=dict)
    path: str
    sha256: str | None = None
    size_bytes: int | None = None
    status_code: int | None = None
    fetched_at: UtcDatetime
