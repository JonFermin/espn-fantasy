"""SQLite state: migration runner, the v1 schema, and typed repositories (DESIGN section 14).

Usage::

    from fm.store import Store

    with Store.open() as store:                 # paths.state_db(), migrated on open
        league = store.leagues.by_key("nfl")
        for proposal in store.proposals.open():
            ...

``Store.open(path)`` opens any file (or ``":memory:"``) and brings it to the current schema; ``Store(db)`` wraps an
already-open ``Database`` without migrating. Repositories take and return the frozen row models in
``fm.store.models``; modules outside this package go through them rather than writing SQL. Several writes become one
atomic unit inside ``with store.db.transaction():``.
"""

from __future__ import annotations

from pathlib import Path

from fm import paths
from fm.store.db import (
    Database,
    Migration,
    SchemaError,
    applied_migrations,
    available_migrations,
    migrate,
    schema_version,
)
from fm.store.models import (
    OPEN_PROPOSAL_STATUSES,
    SPORTS,
    AvailabilityRow,
    DecisionEvalRow,
    ExecutionMode,
    ExecutionRow,
    ExecutionStatus,
    Identified,
    IntList,
    JsonDict,
    LeagueRow,
    LeagueSettingsRow,
    LlmUsageRow,
    MarketValueRow,
    NewsItemRow,
    NewsSignalRow,
    PlayerIdRow,
    PlayerRow,
    ProjectionKind,
    ProjectionRow,
    ProposalPolicy,
    ProposalRow,
    ProposalStatus,
    RawSnapshotRow,
    RosterEntryRow,
    Row,
    SignalKind,
    Sport,
    StatLine,
    StrList,
    TeamRow,
    UtcDatetime,
    format_timestamp,
    utc_now,
)
from fm.store.repos import (
    REPOSITORIES,
    AvailabilityRepo,
    DecisionEvalRepo,
    ExecutionRepo,
    LeagueRepo,
    LeagueSettingsRepo,
    LlmUsageRepo,
    MarketValueRepo,
    NewsRepo,
    NewsSignalRepo,
    PlayerIdRepo,
    PlayerRepo,
    ProjectionRepo,
    ProposalRepo,
    RawSnapshotRepo,
    Repository,
    RosterRepo,
    TeamRepo,
)

__all__ = [
    "OPEN_PROPOSAL_STATUSES",
    "REPOSITORIES",
    "SPORTS",
    "AvailabilityRow",
    "AvailabilityRepo",
    "Database",
    "DecisionEvalRow",
    "DecisionEvalRepo",
    "ExecutionRow",
    "ExecutionMode",
    "ExecutionRepo",
    "ExecutionStatus",
    "Identified",
    "IntList",
    "JsonDict",
    "LeagueRow",
    "LeagueRepo",
    "LeagueSettingsRow",
    "LeagueSettingsRepo",
    "LlmUsageRow",
    "LlmUsageRepo",
    "MarketValueRow",
    "MarketValueRepo",
    "Migration",
    "NewsItemRow",
    "NewsRepo",
    "NewsSignalRow",
    "NewsSignalRepo",
    "PlayerRow",
    "PlayerIdRow",
    "PlayerIdRepo",
    "PlayerRepo",
    "ProposalPolicy",
    "ProjectionRow",
    "ProjectionKind",
    "ProjectionRepo",
    "ProposalRow",
    "ProposalRepo",
    "ProposalStatus",
    "RawSnapshotRow",
    "RawSnapshotRepo",
    "Repository",
    "RosterEntryRow",
    "RosterRepo",
    "Row",
    "SchemaError",
    "SignalKind",
    "Sport",
    "StatLine",
    "Store",
    "StrList",
    "TeamRow",
    "TeamRepo",
    "UtcDatetime",
    "applied_migrations",
    "available_migrations",
    "format_timestamp",
    "migrate",
    "schema_version",
    "utc_now",
]


class Store:
    """Every repository over one database connection."""

    def __init__(self, db: Database) -> None:
        self.db = db
        self.leagues = LeagueRepo(db)
        self.settings = LeagueSettingsRepo(db)
        self.teams = TeamRepo(db)
        self.players = PlayerRepo(db)
        self.player_ids = PlayerIdRepo(db)
        self.rosters = RosterRepo(db)
        self.projections = ProjectionRepo(db)
        self.availability = AvailabilityRepo(db)
        self.news = NewsRepo(db)
        self.news_signals = NewsSignalRepo(db)
        self.market_values = MarketValueRepo(db)
        self.proposals = ProposalRepo(db)
        self.executions = ExecutionRepo(db)
        self.decision_evals = DecisionEvalRepo(db)
        self.llm_usage = LlmUsageRepo(db)
        self.raw_snapshots = RawSnapshotRepo(db)

    @classmethod
    def open(cls, path: str | Path | None = None) -> Store:
        """Open the state database (``paths.state_db()`` by default) and apply any pending migrations."""
        db = Database.open(paths.state_db() if path is None else path)
        migrate(db)
        return cls(db)

    def close(self) -> None:
        self.db.close()

    def __enter__(self) -> Store:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
