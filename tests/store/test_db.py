"""Database connection, transactions, and the migration runner: fresh migrate, idempotent re-migrate, guard rails."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from fm import paths
from fm.store import Store
from fm.store.db import (
    MIGRATION_FILE,
    Database,
    Migration,
    SchemaError,
    applied_migrations,
    available_migrations,
    migrate,
    schema_version,
)

V1_TABLES = frozenset(
    {
        "leagues",
        "league_settings",
        "teams",
        "players",
        "player_ids",
        "roster_snapshots",
        "projections",
        "availability",
        "news_items",
        "news_signals",
        "market_values",
        "proposals",
        "executions",
        "decision_evals",
        "llm_usage",
        "raw_snapshots",
    }
)
STAMP = "2026-10-04T13:30:15.123456Z"
INSERT_USAGE = "INSERT INTO llm_usage (called_at, worker, model) VALUES (?, ?, ?)"


@pytest.fixture
def db(tmp_path: Path) -> Iterator[Database]:
    with Database.open(tmp_path / "state.db") as database:
        yield database


def user_tables(db: Database) -> set[str]:
    rows = db.all("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")
    return {row["name"] for row in rows}


def schema_dump(db: Database) -> list[tuple[object, ...]]:
    return [tuple(row) for row in db.all("SELECT type, name, tbl_name, sql FROM sqlite_master ORDER BY type, name")]


def pragma(db: Database, name: str) -> object:
    row = db.one(f"PRAGMA {name}")
    assert row is not None
    return row[0]


def workers(db: Database) -> list[str]:
    return [row["worker"] for row in db.all("SELECT worker FROM llm_usage ORDER BY id")]


def test_fresh_database_migrates_to_v1(db: Database) -> None:
    assert schema_version(db) == 0
    assert applied_migrations(db) == []
    applied = migrate(db)
    assert [(m.version, m.name) for m in applied] == [(1, "v1_schema")]
    assert schema_version(db) == 1
    assert applied_migrations(db) == [(1, "v1_schema")]
    assert user_tables(db) == V1_TABLES | {"schema_migrations"}


def test_remigrate_is_idempotent(db: Database) -> None:
    migrate(db)
    before = schema_dump(db)
    recorded = db.all("SELECT * FROM schema_migrations")
    assert migrate(db) == []
    assert migrate(db) == []
    assert schema_dump(db) == before
    assert [tuple(row) for row in db.all("SELECT * FROM schema_migrations")] == [tuple(row) for row in recorded]


def test_reopening_an_existing_database_applies_nothing(tmp_path: Path) -> None:
    path = tmp_path / "state.db"
    with Database.open(path) as first:
        migrate(first)
        first.execute(INSERT_USAGE, (STAMP, "a", "m"))
    with Database.open(path) as second:
        assert migrate(second) == []
        assert schema_version(second) == 1
        assert workers(second) == ["a"]


def test_every_table_is_strict(db: Database) -> None:
    migrate(db)
    for table in V1_TABLES | {"schema_migrations"}:
        row = db.one("SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (table,))
        assert row is not None and row["sql"].rstrip().endswith("STRICT"), table


def test_connection_pragmas(db: Database) -> None:
    assert pragma(db, "foreign_keys") == 1
    assert pragma(db, "journal_mode") == "wal"
    assert pragma(db, "busy_timeout") == 5000
    assert not db.in_transaction


def test_migration_files_are_well_formed() -> None:
    migrations = available_migrations()
    assert [m.version for m in migrations] == list(range(1, len(migrations) + 1))
    assert migrations[0].name == "v1_schema"
    for migration in migrations:
        assert MIGRATION_FILE.match(f"{migration.version:04d}_{migration.name}.sql")
        assert migration.sql.strip()


def test_newer_database_is_refused(db: Database) -> None:
    migrate(db)
    db.execute(
        "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)", (2, "from_the_future", STAMP)
    )
    with pytest.raises(SchemaError, match="newer"):
        migrate(db)


def test_mismatched_migration_name_is_refused(db: Database) -> None:
    migrate(db)
    db.execute("UPDATE schema_migrations SET name = 'something_else' WHERE version = 1")
    with pytest.raises(SchemaError, match="does not match"):
        migrate(db)


def test_failed_migration_rolls_back_completely(db: Database, monkeypatch: pytest.MonkeyPatch) -> None:
    (v1,) = available_migrations()
    broken = Migration(2, "broken", "CREATE TABLE half_done (id INTEGER PRIMARY KEY) STRICT;\nCREATE TABLE nope (;")
    monkeypatch.setattr("fm.store.db.available_migrations", lambda: (v1, broken))
    with pytest.raises(sqlite3.OperationalError):
        migrate(db)
    assert schema_version(db) == 1
    assert "half_done" not in user_tables(db)
    assert not db.in_transaction
    # The good migration is in place and usable.
    db.execute(INSERT_USAGE, (STAMP, "a", "m"))
    assert workers(db) == ["a"]


def test_transaction_commits_on_success_and_rolls_back_on_error(db: Database) -> None:
    migrate(db)
    with db.transaction():
        db.execute(INSERT_USAGE, (STAMP, "a", "m"))
        assert db.in_transaction
    with pytest.raises(RuntimeError, match="boom"):
        with db.transaction():
            db.execute(INSERT_USAGE, (STAMP, "b", "m"))
            raise RuntimeError("boom")
    assert workers(db) == ["a"]
    assert not db.in_transaction


def test_nested_transactions_become_savepoints(db: Database) -> None:
    migrate(db)
    with db.transaction():
        db.execute(INSERT_USAGE, (STAMP, "outer", "m"))
        with pytest.raises(ValueError, match="inner"):
            with db.transaction():
                db.execute(INSERT_USAGE, (STAMP, "inner", "m"))
                raise ValueError("inner fails")
        assert db.in_transaction
        with db.transaction():
            db.execute(INSERT_USAGE, (STAMP, "nested ok", "m"))
    assert workers(db) == ["outer", "nested ok"]
    assert not db.in_transaction


def test_constraints_are_enforced(db: Database) -> None:
    migrate(db)
    proposal = (
        "INSERT INTO proposals (league_id, kind, policy, payload, created_by, created_at) VALUES (?, ?, ?, ?, ?, ?)"
    )
    with pytest.raises(sqlite3.IntegrityError, match="FOREIGN KEY"):
        db.execute(proposal, (999, "lineup", "approve", "{}", "t", STAMP))
    league = "INSERT INTO leagues (key, sport, espn_league_id, season, team_id, as_of) VALUES (?, ?, ?, ?, ?, ?)"
    with pytest.raises(sqlite3.IntegrityError, match="TEXT value in INTEGER"):
        db.execute(league, ("nfl", "nfl", "not-an-id", 2026, 1, STAMP))
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        db.execute(league, ("nhl", "nhl", 1, 2026, 1, STAMP))
    db.execute(league, ("nfl", "nfl", 1, 2026, 1, STAMP))
    with pytest.raises(sqlite3.IntegrityError, match="json_valid"):
        db.execute(proposal, (1, "lineup", "approve", "{not json", "t", STAMP))
    with pytest.raises(sqlite3.IntegrityError, match="CHECK constraint failed"):
        db.execute(proposal, (1, "lineup", "whenever", "{}", "t", STAMP))


def test_store_open_uses_state_db_under_config_dir(tmp_path: Path) -> None:
    with Store.open() as store:
        assert store.db.path is not None
        assert store.db.path == paths.state_db() == tmp_path / "config" / "state.db"
        assert store.db.path.exists()
        assert schema_version(store.db) == 1
    with Store.open() as reopened:
        assert migrate(reopened.db) == []


def test_store_open_in_memory() -> None:
    with Store.open(":memory:") as store:
        assert store.db.path is None
        assert schema_version(store.db) == 1
        assert user_tables(store.db) == V1_TABLES | {"schema_migrations"}
