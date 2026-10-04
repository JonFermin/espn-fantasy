"""SQLite connection and migration runner.

``Database`` wraps one ``sqlite3`` connection opened in autocommit mode with WAL, foreign keys and a busy timeout, and
gives explicit transactions: ``with db.transaction(): ...`` starts ``BEGIN IMMEDIATE`` (so a second process never
deadlocks on a lock upgrade) and nested calls become savepoints. Statements outside a transaction commit on their own.

Migrations are ``NNNN_name.sql`` files in ``fm.store.migrations``. ``migrate`` applies the ones not yet recorded in
``schema_migrations``, each inside its own transaction, in version order; on an up-to-date database it does nothing. A
database recorded at a newer version than this build knows, or whose recorded names disagree with the files, is
refused rather than guessed at.
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

from fm.store.models import format_timestamp, utc_now

MEMORY = ":memory:"
MIGRATIONS_PACKAGE = "fm.store.migrations"
MIGRATION_FILE = re.compile(r"^(?P<version>\d{4})_(?P<name>[a-z0-9_]+)\.sql$")
SCHEMA_MIGRATIONS_DDL = (
    "CREATE TABLE IF NOT EXISTS schema_migrations ("
    "version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at TEXT NOT NULL) STRICT"
)

type Params = Sequence[Any] | Mapping[str, Any]


class SchemaError(RuntimeError):
    """The migration files or the database's recorded schema state are inconsistent."""


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    sql: str


class Database:
    """One SQLite connection with explicit transactions and small query helpers."""

    def __init__(self, connection: sqlite3.Connection, path: Path | None = None) -> None:
        self.connection = connection
        self.path = path
        self._savepoints = 0

    @classmethod
    def open(cls, path: str | Path = MEMORY) -> Database:
        """Open (creating if needed) the database at ``path``, or an in-memory one for ``":memory:"``."""
        file = None if str(path) == MEMORY else Path(path)
        if file is not None:
            file.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(MEMORY if file is None else file, autocommit=True)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.execute("PRAGMA synchronous = NORMAL")
        return cls(connection, file)

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def execute(self, sql: str, params: Params = ()) -> sqlite3.Cursor:
        return self.connection.execute(sql, params)

    def one(self, sql: str, params: Params = ()) -> sqlite3.Row | None:
        """The first row, or ``None``. Reads every returned row so ``RETURNING`` statements complete."""
        rows = self.connection.execute(sql, params).fetchall()
        return rows[0] if rows else None

    def all(self, sql: str, params: Params = ()) -> list[sqlite3.Row]:
        return self.connection.execute(sql, params).fetchall()

    @property
    def in_transaction(self) -> bool:
        return self.connection.in_transaction

    @contextmanager
    def transaction(self) -> Iterator[None]:
        """Run the block atomically: ``BEGIN IMMEDIATE`` at the top level, a savepoint when already inside one."""
        if self.connection.in_transaction:
            self._savepoints += 1
            name = f"fm_savepoint_{self._savepoints}"
            self.connection.execute(f"SAVEPOINT {name}")
            try:
                yield
            except BaseException:
                self.connection.execute(f"ROLLBACK TO {name}")
                self.connection.execute(f"RELEASE {name}")
                raise
            else:
                self.connection.execute(f"RELEASE {name}")
            finally:
                self._savepoints -= 1
            return
        self.connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            if self.connection.in_transaction:
                self.connection.execute("ROLLBACK")
            raise
        self.connection.execute("COMMIT")


def available_migrations() -> tuple[Migration, ...]:
    """Every migration file in ``fm.store.migrations``, in version order. Versions must run 1..N with no gaps."""
    found: list[Migration] = []
    for entry in resources.files(MIGRATIONS_PACKAGE).iterdir():
        match = MIGRATION_FILE.match(entry.name)
        if match is None or not entry.is_file():
            continue
        found.append(Migration(int(match["version"]), match["name"], entry.read_text(encoding="utf-8")))
    found.sort(key=lambda migration: migration.version)
    versions = [migration.version for migration in found]
    if versions != list(range(1, len(found) + 1)):
        raise SchemaError(f"migration files must be numbered 1..N without gaps or duplicates, found {versions}")
    return tuple(found)


def applied_migrations(db: Database) -> list[tuple[int, str]]:
    """``(version, name)`` of every applied migration, oldest first; empty for a fresh database."""
    if db.one("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'schema_migrations'") is None:
        return []
    return [
        (row["version"], row["name"]) for row in db.all("SELECT version, name FROM schema_migrations ORDER BY version")
    ]


def schema_version(db: Database) -> int:
    """The highest applied migration version, ``0`` for a fresh database."""
    applied = applied_migrations(db)
    return applied[-1][0] if applied else 0


def migrate(db: Database) -> list[Migration]:
    """Apply every pending migration in order and return the ones applied now (empty when already current)."""
    available = available_migrations()
    db.execute(SCHEMA_MIGRATIONS_DDL)
    applied = applied_migrations(db)
    if len(applied) > len(available):
        raise SchemaError(
            f"database schema version {applied[-1][0]} is newer than this build knows ({len(available)}); "
            "update espn-fantasy"
        )
    for (version, name), migration in zip(applied, available, strict=False):
        if (version, name) != (migration.version, migration.name):
            raise SchemaError(
                f"applied migration {version:04d}_{name} does not match file "
                f"{migration.version:04d}_{migration.name}.sql"
            )
    pending = list(available[len(applied) :])
    for migration in pending:
        with db.transaction():
            db.connection.executescript(migration.sql)
            db.execute(
                "INSERT INTO schema_migrations (version, name, applied_at) VALUES (?, ?, ?)",
                (migration.version, migration.name, format_timestamp(utc_now())),
            )
    return pending
