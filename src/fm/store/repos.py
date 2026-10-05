"""Typed repositories: one class per table, taking and returning the row models in ``fm.store.models``.

Natural-key tables (players, projections, ...) expose ``upsert``: insert, or replace the row with that key. Id tables
(proposals, executions, ...) expose ``insert``, which returns the row with its new ``id``, and ``update``, which writes
the whole row by id. Every write runs in a transaction; wrap several calls in ``with store.db.transaction():`` to make
them atomic together. Queries beyond simple lookups belong here too, so modules outside ``fm.store`` do not write SQL.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from datetime import datetime
from typing import Any, ClassVar

from fm.store.db import Database
from fm.store.models import (
    OPEN_PROPOSAL_STATUSES,
    AvailabilityRow,
    DecisionEvalRow,
    ExecutionRow,
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
    ProposalRow,
    ProposalStatus,
    RawSnapshotRow,
    RosterEntryRow,
    Row,
    Sport,
    TeamRow,
    format_timestamp,
)

IN_CHUNK = 500


def encode_json(value: object) -> str:
    """Compact, key-sorted JSON text, the form every JSON column stores."""
    return json.dumps(value, separators=(",", ":"), sort_keys=True, ensure_ascii=False)


def _chunks(values: Sequence[Any], size: int = IN_CHUNK) -> Iterator[Sequence[Any]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _marks(count: int) -> str:
    return ", ".join("?" * count)


class Repository[M: Row]:
    """Shared encoding and statement builders. Subclasses set ``table``, ``model`` and, for upserts, ``key``."""

    table: ClassVar[str]
    key: ClassVar[tuple[str, ...]] = ()
    model: type[M]

    def __init__(self, db: Database) -> None:
        self.db = db

    @classmethod
    def columns(cls) -> tuple[str, ...]:
        """Column names, which are exactly the model's field names in declaration order."""
        return tuple(cls.model.model_fields)

    def to_row(self, item: M) -> dict[str, Any]:
        """Model -> column values: datetimes become fixed-width UTC text, dicts and lists become JSON text."""
        data = item.model_dump(mode="json")
        return {name: encode_json(value) if isinstance(value, dict | list) else value for name, value in data.items()}

    def from_row(self, row: sqlite3.Row) -> M:
        return self.model.model_validate(dict(row))

    def _insert_values(self, item: M) -> dict[str, Any]:
        row = self.to_row(item)
        if "id" in row and row["id"] is None:
            del row["id"]
        return row

    def _insert_statement(self, columns: Sequence[str], conflict: str | None = None) -> str:
        parts = [f"INSERT INTO {self.table} ({', '.join(columns)}) VALUES ({_marks(len(columns))})"]
        if conflict:
            parts.append(conflict)
        parts.append("RETURNING *")
        return " ".join(parts)

    def _returned(self, row: sqlite3.Row | None) -> M:
        if row is None:
            raise RuntimeError(f"{self.table}: statement returned no row")
        return self.from_row(row)

    def _insert(self, item: M) -> M:
        row = self._insert_values(item)
        return self._returned(self.db.one(self._insert_statement(list(row)), list(row.values())))

    def _upsert(self, item: M) -> M:
        row = self._insert_values(item)
        updates = ", ".join(f"{name} = excluded.{name}" for name in row if name not in self.key)
        conflict = f"ON CONFLICT ({', '.join(self.key)}) DO UPDATE SET {updates}"
        return self._returned(self.db.one(self._insert_statement(list(row), conflict), list(row.values())))

    def _upsert_many(self, items: Iterable[M]) -> int:
        count = 0
        with self.db.transaction():
            for item in items:
                self._upsert(item)
                count += 1
        return count

    def _insert_or_ignore(self, item: M) -> M | None:
        row = self._insert_values(item)
        conflict = f"ON CONFLICT ({', '.join(self.key)}) DO NOTHING"
        fetched = self.db.one(self._insert_statement(list(row), conflict), list(row.values()))
        return None if fetched is None else self.from_row(fetched)

    def _update(self, item: M) -> M:
        row = self.to_row(item)
        ident = row.pop("id", None)
        if ident is None:
            raise ValueError(f"{self.table}: update needs a row with an id")
        assignments = ", ".join(f"{name} = ?" for name in row)
        fetched = self.db.one(f"UPDATE {self.table} SET {assignments} WHERE id = ? RETURNING *", [*row.values(), ident])
        if fetched is None:
            raise LookupError(f"{self.table}: no row with id {ident}")
        return self.from_row(fetched)

    def _select(
        self, where: str = "", params: Sequence[Any] = (), *, order: str = "", limit: int | None = None
    ) -> list[M]:
        sql = f"SELECT * FROM {self.table}"
        if where:
            sql += f" WHERE {where}"
        if order:
            sql += f" ORDER BY {order}"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return [self.from_row(row) for row in self.db.all(sql, params)]

    def _select_one(self, where: str, params: Sequence[Any], *, order: str = "") -> M | None:
        rows = self._select(where, params, order=order, limit=1)
        return rows[0] if rows else None


class LeagueRepo(Repository[LeagueRow]):
    """``leagues``: the config key is a league's identity, one row per (key, season)."""

    table = "leagues"
    key = ("key", "season")
    model = LeagueRow

    def upsert(self, league: LeagueRow) -> LeagueRow:
        """Insert, or update the row for (key, season) in place. Returns the stored row; ``league.id`` is ignored.

        Correcting ``espn_league_id`` (or ``sport``) in config.toml keeps the row and its id but points it at another
        ESPN league, so what was read from the old one goes: ``league_settings`` and ``teams`` (with their
        ``roster_snapshots``) are deleted and ``raw_snapshots`` are detached, and the next sync starts clean.
        Proposals and decision history stay with the key. A row for the same ESPN league under a key that is no
        longer in use is taken over (renamed) rather than duplicated: one ESPN league is managed under one key.
        """
        with self.db.transaction():
            current = self._select_one("key = ? AND season = ?", (league.key, league.season))
            if current is None:
                current = self._select_one(
                    "sport = ? AND espn_league_id = ? AND season = ?",
                    (league.sport, league.espn_league_id, league.season),
                )
            if current is None:
                return self._insert(league.model_copy(update={"id": None}))
            if (current.sport, current.espn_league_id) != (league.sport, league.espn_league_id):
                self._forget_espn_state(current.row_id)
            return self._update(league.model_copy(update={"id": current.id}))

    def _forget_espn_state(self, league_id: int) -> None:
        """Drop what was read from the ESPN league this row used to point at."""
        self.db.execute("DELETE FROM league_settings WHERE league_id = ?", (league_id,))
        self.db.execute("DELETE FROM teams WHERE league_id = ?", (league_id,))  # cascades to roster_snapshots
        self.db.execute("UPDATE raw_snapshots SET league_id = NULL WHERE league_id = ?", (league_id,))

    def get(self, league_id: int) -> LeagueRow | None:
        return self._select_one("id = ?", (league_id,))

    def by_key(self, key: str) -> LeagueRow | None:
        """The league configured under ``key`` (``nfl``, ``nba``), latest season first."""
        return self._select_one("key = ?", (key,), order="season DESC")

    def all(self) -> list[LeagueRow]:
        return self._select(order="season DESC, key")


class LeagueSettingsRepo(Repository[LeagueSettingsRow]):
    table = "league_settings"
    key = ("league_id",)
    model = LeagueSettingsRow

    def upsert(self, record: LeagueSettingsRow) -> LeagueSettingsRow:
        with self.db.transaction():
            return self._upsert(record)

    def get(self, league_id: int) -> LeagueSettingsRow | None:
        return self._select_one("league_id = ?", (league_id,))


class TeamRepo(Repository[TeamRow]):
    table = "teams"
    key = ("league_id", "team_id")
    model = TeamRow

    def upsert(self, team: TeamRow) -> TeamRow:
        with self.db.transaction():
            return self._upsert(team)

    def upsert_many(self, teams: Iterable[TeamRow]) -> int:
        return self._upsert_many(teams)

    def get(self, league_id: int, team_id: int) -> TeamRow | None:
        return self._select_one("league_id = ? AND team_id = ?", (league_id, team_id))

    def for_league(self, league_id: int) -> list[TeamRow]:
        return self._select("league_id = ?", (league_id,), order="team_id")


class PlayerRepo(Repository[PlayerRow]):
    table = "players"
    key = ("sport", "espn_id")
    model = PlayerRow

    def upsert(self, player: PlayerRow) -> PlayerRow:
        with self.db.transaction():
            return self._upsert(player)

    def upsert_many(self, players: Iterable[PlayerRow]) -> int:
        return self._upsert_many(players)

    def get(self, sport: Sport, espn_id: int) -> PlayerRow | None:
        return self._select_one("sport = ? AND espn_id = ?", (sport, espn_id))

    def many(self, sport: Sport, espn_ids: Iterable[int]) -> list[PlayerRow]:
        """The players with these ids that are stored, ordered by ``espn_id``; missing ids are simply absent."""
        ids = sorted(set(espn_ids))
        found: list[PlayerRow] = []
        for chunk in _chunks(ids):
            found.extend(
                self._select(f"sport = ? AND espn_id IN ({_marks(len(chunk))})", (sport, *chunk), order="espn_id")
            )
        return found


class PlayerIdRepo(Repository[PlayerIdRow]):
    table = "player_ids"
    key = ("sport", "espn_id", "source")
    model = PlayerIdRow

    def upsert(self, mapping: PlayerIdRow) -> PlayerIdRow:
        with self.db.transaction():
            return self._upsert(mapping)

    def upsert_many(self, mappings: Iterable[PlayerIdRow]) -> int:
        return self._upsert_many(mappings)

    def for_player(self, sport: Sport, espn_id: int) -> list[PlayerIdRow]:
        return self._select("sport = ? AND espn_id = ?", (sport, espn_id), order="source")

    def for_sport(self, sport: Sport) -> list[PlayerIdRow]:
        """Every mapping of ``sport`` by (``espn_id``, ``source``): what a crosswalk loads."""
        return self._select("sport = ?", (sport,), order="espn_id, source")

    def replace_sport(self, sport: Sport, mappings: Iterable[PlayerIdRow]) -> int:
        """Replace every mapping of ``sport`` with ``mappings`` (each must be that sport) in one transaction.

        A replace rather than an upsert: when a mapping re-points a source id to another player, the old row would
        otherwise collide with the one-owner-per-source-id constraint. Returns the number of rows written.
        """
        rows = list(mappings)
        for row in rows:
            if row.sport != sport:
                raise ValueError(f"player id mapping for ESPN {row.espn_id} is {row.sport}, not {sport}")
        with self.db.transaction():
            self.db.execute("DELETE FROM player_ids WHERE sport = ?", (sport,))
            for row in rows:
                self._insert(row)
        return len(rows)

    def lookup(self, sport: Sport, source: str, source_id: str) -> PlayerIdRow | None:
        """The ESPN mapping for a source id (``sleeper`` ``"4046"`` -> espn_id ...)."""
        return self._select_one("sport = ? AND source = ? AND source_id = ?", (sport, source, source_id))

    def unmapped(self, sport: Sport, source: str, espn_ids: Iterable[int]) -> set[int]:
        """The subset of ``espn_ids`` with no ``source`` mapping: the rostered-player gate's input."""
        wanted = set(espn_ids)
        mapped: set[int] = set()
        for chunk in _chunks(sorted(wanted)):
            rows = self.db.all(
                f"SELECT espn_id FROM player_ids WHERE sport = ? AND source = ? AND espn_id IN ({_marks(len(chunk))})",
                (sport, source, *chunk),
            )
            mapped.update(row["espn_id"] for row in rows)
        return wanted - mapped


class RosterRepo(Repository[RosterEntryRow]):
    table = "roster_snapshots"
    key = ("league_id", "scoring_period_id", "team_id", "espn_id")
    model = RosterEntryRow

    def replace(self, league_id: int, scoring_period_id: int, team_id: int, entries: Iterable[RosterEntryRow]) -> int:
        """Replace one team's roster for one period with ``entries`` (each must belong to that team and period)."""
        rows = list(entries)
        for entry in rows:
            if (entry.league_id, entry.scoring_period_id, entry.team_id) != (league_id, scoring_period_id, team_id):
                raise ValueError(
                    f"roster entry for player {entry.espn_id} belongs to league {entry.league_id}, period "
                    f"{entry.scoring_period_id}, team {entry.team_id}, not {league_id}/{scoring_period_id}/{team_id}"
                )
        with self.db.transaction():
            self.db.execute(
                "DELETE FROM roster_snapshots WHERE league_id = ? AND scoring_period_id = ? AND team_id = ?",
                (league_id, scoring_period_id, team_id),
            )
            for entry in rows:
                self._insert(entry)
        return len(rows)

    def team(self, league_id: int, scoring_period_id: int, team_id: int) -> list[RosterEntryRow]:
        return self._select(
            "league_id = ? AND scoring_period_id = ? AND team_id = ?",
            (league_id, scoring_period_id, team_id),
            order="lineup_slot_id, espn_id",
        )

    def league(self, league_id: int, scoring_period_id: int) -> list[RosterEntryRow]:
        return self._select(
            "league_id = ? AND scoring_period_id = ?", (league_id, scoring_period_id), order="team_id, lineup_slot_id"
        )

    def rostered_ids(self, league_id: int, scoring_period_id: int) -> set[int]:
        """Every player on any roster in the league for the period."""
        rows = self.db.all(
            "SELECT DISTINCT espn_id FROM roster_snapshots WHERE league_id = ? AND scoring_period_id = ?",
            (league_id, scoring_period_id),
        )
        return {row["espn_id"] for row in rows}

    def latest_period(self, league_id: int) -> int | None:
        """The most recent scoring period with a snapshot for the league, or ``None`` before the first sync."""
        row = self.db.one(
            "SELECT MAX(scoring_period_id) AS period FROM roster_snapshots WHERE league_id = ?", (league_id,)
        )
        return None if row is None else row["period"]


class ProjectionRepo(Repository[ProjectionRow]):
    table = "projections"
    key = ("sport", "espn_id", "source", "kind", "season", "scoring_period_id")
    model = ProjectionRow

    def upsert(self, projection: ProjectionRow) -> ProjectionRow:
        with self.db.transaction():
            return self._upsert(projection)

    def upsert_many(self, projections: Iterable[ProjectionRow]) -> int:
        return self._upsert_many(projections)

    def get(
        self,
        sport: Sport,
        espn_id: int,
        source: str,
        season: int,
        scoring_period_id: int,
        kind: ProjectionKind = "projected",
    ) -> ProjectionRow | None:
        return self._select_one(
            "sport = ? AND espn_id = ? AND source = ? AND kind = ? AND season = ? AND scoring_period_id = ?",
            (sport, espn_id, source, kind, season, scoring_period_id),
        )

    def for_period(
        self,
        sport: Sport,
        season: int,
        scoring_period_id: int,
        *,
        source: str | None = None,
        kind: ProjectionKind = "projected",
    ) -> list[ProjectionRow]:
        """All stat lines for a period, optionally from one source."""
        where = "sport = ? AND season = ? AND scoring_period_id = ? AND kind = ?"
        params: list[Any] = [sport, season, scoring_period_id, kind]
        if source is not None:
            where += " AND source = ?"
            params.append(source)
        return self._select(where, params, order="espn_id, source")

    def for_player(
        self, sport: Sport, espn_id: int, season: int, *, kind: ProjectionKind = "projected"
    ) -> list[ProjectionRow]:
        return self._select(
            "sport = ? AND espn_id = ? AND season = ? AND kind = ?",
            (sport, espn_id, season, kind),
            order="scoring_period_id, source",
        )

    def delete(
        self,
        sport: Sport,
        season: int,
        scoring_period_id: int,
        source: str,
        espn_ids: Iterable[int],
        *,
        kind: ProjectionKind = "projected",
    ) -> int:
        """Delete one source's stat lines for these players in one period (a blended row no source feeds any more),
        in one transaction. Returns the number of rows deleted; ids without a row are skipped."""
        ids = sorted(set(espn_ids))
        deleted = 0
        with self.db.transaction():
            for chunk in _chunks(ids):
                cursor = self.db.execute(
                    "DELETE FROM projections WHERE sport = ? AND season = ? AND scoring_period_id = ? AND source = ? "
                    f"AND kind = ? AND espn_id IN ({_marks(len(chunk))})",
                    (sport, season, scoring_period_id, source, kind, *chunk),
                )
                deleted += cursor.rowcount
        return deleted


class AvailabilityRepo(Repository[AvailabilityRow]):
    table = "availability"
    key = ("sport", "espn_id", "season", "scoring_period_id")
    model = AvailabilityRow

    def upsert(self, availability: AvailabilityRow) -> AvailabilityRow:
        with self.db.transaction():
            return self._upsert(availability)

    def upsert_many(self, rows: Iterable[AvailabilityRow]) -> int:
        return self._upsert_many(rows)

    def get(self, sport: Sport, espn_id: int, season: int, scoring_period_id: int) -> AvailabilityRow | None:
        return self._select_one(
            "sport = ? AND espn_id = ? AND season = ? AND scoring_period_id = ?",
            (sport, espn_id, season, scoring_period_id),
        )

    def for_period(self, sport: Sport, season: int, scoring_period_id: int) -> list[AvailabilityRow]:
        return self._select(
            "sport = ? AND season = ? AND scoring_period_id = ?", (sport, season, scoring_period_id), order="espn_id"
        )


class NewsRepo(Repository[NewsItemRow]):
    table = "news_items"
    key = ("source", "external_id")
    model = NewsItemRow

    def ingest(self, item: NewsItemRow) -> NewsItemRow | None:
        """Store a news item unless (source, external_id) is already known; returns ``None`` for a duplicate."""
        with self.db.transaction():
            return self._insert_or_ignore(item)

    def get(self, news_item_id: int) -> NewsItemRow | None:
        return self._select_one("id = ?", (news_item_id,))

    def untriaged(self, limit: int = 100) -> list[NewsItemRow]:
        """Items the advisor has not processed yet, oldest first."""
        return self._select("triaged_at IS NULL", order="published_at, id", limit=limit)

    def mark_triaged(self, news_item_ids: Iterable[int], at: datetime) -> int:
        ids = sorted(set(news_item_ids))
        updated = 0
        with self.db.transaction():
            for chunk in _chunks(ids):
                cursor = self.db.execute(
                    f"UPDATE news_items SET triaged_at = ? WHERE id IN ({_marks(len(chunk))})",
                    (format_timestamp(at), *chunk),
                )
                updated += cursor.rowcount
        return updated

    def published_since(self, since: datetime, *, sport: Sport | None = None) -> list[NewsItemRow]:
        where = "published_at >= ?"
        params: list[Any] = [format_timestamp(since)]
        if sport is not None:
            where += " AND sport = ?"
            params.append(sport)
        return self._select(where, params, order="published_at, id")


class NewsSignalRepo(Repository[NewsSignalRow]):
    table = "news_signals"
    model = NewsSignalRow

    def insert(self, signal: NewsSignalRow) -> NewsSignalRow:
        with self.db.transaction():
            return self._insert(signal)

    def for_player(self, sport: Sport, espn_id: int, *, since: datetime | None = None) -> list[NewsSignalRow]:
        where = "sport = ? AND espn_id = ?"
        params: list[Any] = [sport, espn_id]
        if since is not None:
            where += " AND published_at >= ?"
            params.append(format_timestamp(since))
        return self._select(where, params, order="published_at, id")

    def for_item(self, news_item_id: int) -> list[NewsSignalRow]:
        return self._select("news_item_id = ?", (news_item_id,), order="id")


class MarketValueRepo(Repository[MarketValueRow]):
    table = "market_values"
    key = ("sport", "espn_id", "source")
    model = MarketValueRow

    def upsert(self, value: MarketValueRow) -> MarketValueRow:
        with self.db.transaction():
            return self._upsert(value)

    def upsert_many(self, values: Iterable[MarketValueRow]) -> int:
        return self._upsert_many(values)

    def get(self, sport: Sport, espn_id: int, source: str) -> MarketValueRow | None:
        return self._select_one("sport = ? AND espn_id = ? AND source = ?", (sport, espn_id, source))

    def for_source(self, sport: Sport, source: str) -> list[MarketValueRow]:
        return self._select("sport = ? AND source = ?", (sport, source), order="rank, espn_id")


class ProposalRepo(Repository[ProposalRow]):
    table = "proposals"
    model = ProposalRow

    def insert(self, proposal: ProposalRow) -> ProposalRow:
        with self.db.transaction():
            return self._insert(proposal)

    def update(self, proposal: ProposalRow) -> ProposalRow:
        with self.db.transaction():
            return self._update(proposal)

    def get(self, proposal_id: int) -> ProposalRow | None:
        return self._select_one("id = ?", (proposal_id,))

    def find(
        self,
        *,
        league_id: int | None = None,
        statuses: Iterable[ProposalStatus] | None = None,
        kinds: Iterable[str] | None = None,
        scoring_period_id: int | None = None,
    ) -> list[ProposalRow]:
        """Proposals matching every given filter, oldest first."""
        clauses: list[str] = []
        params: list[Any] = []
        if league_id is not None:
            clauses.append("league_id = ?")
            params.append(league_id)
        if statuses is not None:
            wanted = list(statuses)
            clauses.append(f"status IN ({_marks(len(wanted))})")
            params.extend(wanted)
        if kinds is not None:
            wanted_kinds = list(kinds)
            clauses.append(f"kind IN ({_marks(len(wanted_kinds))})")
            params.extend(wanted_kinds)
        if scoring_period_id is not None:
            clauses.append("scoring_period_id = ?")
            params.append(scoring_period_id)
        return self._select(" AND ".join(clauses), params, order="created_at, id")

    def open(self, league_id: int | None = None) -> list[ProposalRow]:
        """Proposals still in flight: proposed, approved or executing."""
        return self.find(league_id=league_id, statuses=OPEN_PROPOSAL_STATUSES)

    def consume_execution_token(self, proposal_id: int, token: str, at: datetime) -> bool:
        """Atomically mark the proposal's single-use execution token as used. ``False`` if the token does not match
        or was already consumed, so a second execution attempt cannot proceed."""
        with self.db.transaction():
            cursor = self.db.execute(
                "UPDATE proposals SET token_consumed_at = ? "
                "WHERE id = ? AND execution_token = ? AND token_consumed_at IS NULL",
                (format_timestamp(at), proposal_id, token),
            )
            return cursor.rowcount == 1


class ExecutionRepo(Repository[ExecutionRow]):
    table = "executions"
    model = ExecutionRow

    def insert(self, execution: ExecutionRow) -> ExecutionRow:
        with self.db.transaction():
            return self._insert(execution)

    def update(self, execution: ExecutionRow) -> ExecutionRow:
        with self.db.transaction():
            return self._update(execution)

    def get(self, execution_id: int) -> ExecutionRow | None:
        return self._select_one("id = ?", (execution_id,))

    def for_proposal(self, proposal_id: int) -> list[ExecutionRow]:
        return self._select("proposal_id = ?", (proposal_id,), order="started_at, id")


class DecisionEvalRepo(Repository[DecisionEvalRow]):
    table = "decision_evals"
    model = DecisionEvalRow

    def insert(self, evaluation: DecisionEvalRow) -> DecisionEvalRow:
        with self.db.transaction():
            return self._insert(evaluation)

    def update(self, evaluation: DecisionEvalRow) -> DecisionEvalRow:
        with self.db.transaction():
            return self._update(evaluation)

    def get(self, eval_id: int) -> DecisionEvalRow | None:
        return self._select_one("id = ?", (eval_id,))

    def find(
        self,
        league_id: int,
        *,
        kind: str | None = None,
        season: int | None = None,
        scoring_period_id: int | None = None,
    ) -> list[DecisionEvalRow]:
        where = "league_id = ?"
        params: list[Any] = [league_id]
        if kind is not None:
            where += " AND kind = ?"
            params.append(kind)
        if season is not None:
            where += " AND season = ?"
            params.append(season)
        if scoring_period_id is not None:
            where += " AND scoring_period_id = ?"
            params.append(scoring_period_id)
        return self._select(where, params, order="decided_at, id")

    def unevaluated(self, league_id: int) -> list[DecisionEvalRow]:
        """Decisions whose outcome has not been recorded yet."""
        return self._select("league_id = ? AND outcome IS NULL", (league_id,), order="decided_at, id")


class LlmUsageRepo(Repository[LlmUsageRow]):
    table = "llm_usage"
    model = LlmUsageRow

    def insert(self, usage: LlmUsageRow) -> LlmUsageRow:
        with self.db.transaction():
            return self._insert(usage)

    def cost_since(self, since: datetime) -> float:
        """Total spend in USD for calls at or after ``since``: the input to the daily budget cap."""
        row = self.db.one(
            "SELECT COALESCE(SUM(cost_usd), 0) AS cost FROM llm_usage WHERE called_at >= ?", (format_timestamp(since),)
        )
        return float(row["cost"]) if row is not None else 0.0

    def since(self, since: datetime) -> list[LlmUsageRow]:
        return self._select("called_at >= ?", (format_timestamp(since),), order="called_at, id")


class RawSnapshotRepo(Repository[RawSnapshotRow]):
    table = "raw_snapshots"
    model = RawSnapshotRow

    def insert(self, snapshot: RawSnapshotRow) -> RawSnapshotRow:
        with self.db.transaction():
            return self._insert(snapshot)

    def get(self, snapshot_id: int) -> RawSnapshotRow | None:
        return self._select_one("id = ?", (snapshot_id,))

    def latest(
        self, source: str, kind: str, *, league_id: int | None = None, scoring_period_id: int | None = None
    ) -> RawSnapshotRow | None:
        """The newest snapshot of ``kind`` from ``source``, narrowed to a league and period when given."""
        where, params = self._filters(source, kind, league_id, scoring_period_id)
        return self._select_one(where, params, order="fetched_at DESC, id DESC")

    def find(
        self,
        source: str,
        kind: str | None = None,
        *,
        league_id: int | None = None,
        scoring_period_id: int | None = None,
    ) -> list[RawSnapshotRow]:
        where, params = self._filters(source, kind, league_id, scoring_period_id)
        return self._select(where, params, order="fetched_at, id")

    @staticmethod
    def _filters(
        source: str, kind: str | None, league_id: int | None, scoring_period_id: int | None
    ) -> tuple[str, list[Any]]:
        where = "source = ?"
        params: list[Any] = [source]
        if kind is not None:
            where += " AND kind = ?"
            params.append(kind)
        if league_id is not None:
            where += " AND league_id = ?"
            params.append(league_id)
        if scoring_period_id is not None:
            where += " AND scoring_period_id = ?"
            params.append(scoring_period_id)
        return where, params


REPOSITORIES: tuple[type[Repository[Any]], ...] = (
    LeagueRepo,
    LeagueSettingsRepo,
    TeamRepo,
    PlayerRepo,
    PlayerIdRepo,
    RosterRepo,
    ProjectionRepo,
    AvailabilityRepo,
    NewsRepo,
    NewsSignalRepo,
    MarketValueRepo,
    ProposalRepo,
    ExecutionRepo,
    DecisionEvalRepo,
    LlmUsageRepo,
    RawSnapshotRepo,
)
