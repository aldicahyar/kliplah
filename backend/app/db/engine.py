"""SQLite engine (WAL + foreign keys), schema bootstrap/migrations and session helper."""
import re
import sqlite3
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, event
from sqlmodel import Session, SQLModel, create_engine

from app.core.config import Settings
from app.core.errors import AppError

DB_FILENAME = "kliplah.db"
SCHEMA_VERSION = 2

# Statements that bring a database from version ``v - 1`` to ``v``. Fresh databases are created
# directly at SCHEMA_VERSION by ``create_all`` and never run these.
MIGRATIONS: dict[int, list[str]] = {
    2: ["ALTER TABLE clip ADD COLUMN signal_bonus INTEGER NOT NULL DEFAULT 0"],
}

_ADD_COLUMN = re.compile(r"ALTER TABLE (\w+) ADD COLUMN (\w+)", re.IGNORECASE)

_engines: dict[Path, Engine] = {}
_lock = threading.Lock()


def _on_connect(dbapi_conn, _record) -> None:
    cur = dbapi_conn.cursor()
    try:
        cur.execute("PRAGMA busy_timeout=5000")
        cur.execute("PRAGMA journal_mode=WAL")
        cur.execute("PRAGMA foreign_keys=ON")
    finally:
        cur.close()


def get_engine(settings: Settings) -> Engine:
    """One engine per resolved data_dir; creates the directory if missing."""
    data_dir = Path(settings.data_dir).resolve()
    with _lock:
        engine = _engines.get(data_dir)
        if engine is None:
            data_dir.mkdir(parents=True, exist_ok=True)
            engine = create_engine(f"sqlite:///{(data_dir / DB_FILENAME).as_posix()}",
                                   connect_args={"check_same_thread": False})
            event.listen(engine, "connect", _on_connect)
            _engines[data_dir] = engine
        return engine


@contextmanager
def _immediate(dbapi: sqlite3.Connection) -> Iterator[None]:
    """Real write transaction on an autocommit sqlite3 connection (DDL included).

    IMMEDIATE takes the write lock up front, so a concurrent init_db (API and worker start together)
    waits on busy_timeout instead of failing on a read->write lock upgrade.
    """
    dbapi.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        if dbapi.in_transaction:  # SQLite may already have rolled back on its own
            dbapi.execute("ROLLBACK")
        raise
    dbapi.execute("COMMIT")


def _stored_version(dbapi: sqlite3.Connection) -> int | None:
    row = dbapi.execute("SELECT version FROM schema_version").fetchone()
    return None if row is None else row[0]


def _migrate(conn: sqlite3.Connection, from_version: int) -> None:
    """Apply the statements that move the schema from ``from_version`` to ``from_version + 1``.

    ``conn`` is a raw sqlite3 connection inside an open transaction (see ``_immediate``).
    ADD COLUMN statements are skipped when the column already exists, so a re-run is harmless.
    """
    to_version = from_version + 1
    for stmt in MIGRATIONS.get(to_version, []):
        m = _ADD_COLUMN.match(stmt)
        if m:
            table, column = m.groups()
            columns = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
            if column in columns:
                continue
        conn.execute(stmt)
    conn.execute("UPDATE schema_version SET version = ?", (to_version,))


def init_db(engine: Engine) -> None:
    """Create all tables, migrate an older schema to SCHEMA_VERSION. Safe to call repeatedly and concurrently.

    pysqlite's default transaction handling autocommits DDL, so this runs on an AUTOCOMMIT connection
    with explicit BEGIN IMMEDIATE/COMMIT: one transaction for the bootstrap, then one per version.
    Each step re-reads the stored version under the lock, so a process that waited skips work already done.
    """
    import app.db.tables  # noqa: F401  registers tables on SQLModel.metadata

    with engine.connect() as conn:
        conn.execution_options(isolation_level="AUTOCOMMIT")
        dbapi = conn.connection.driver_connection
        with _immediate(dbapi):
            existing = dbapi.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_version'").fetchone() is not None
            SQLModel.metadata.create_all(conn)
            dbapi.execute("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)")
            stored = _stored_version(dbapi)
            if stored is None:
                # Fresh database: create_all already built the current schema, so record it as-is.
                # (An existing but empty schema_version is treated as v1; guarded ALTERs keep that safe.)
                stored = 1 if existing else SCHEMA_VERSION
                dbapi.execute("INSERT INTO schema_version (version) VALUES (?)", (stored,))
        if stored > SCHEMA_VERSION:
            raise AppError("INVALID_STATE",
                           f"Database dibuat oleh versi Kliplah yang lebih baru (skema {stored}, "
                           f"aplikasi ini mendukung sampai {SCHEMA_VERSION}). Perbarui aplikasi.",
                           status=409)
        while stored < SCHEMA_VERSION:
            with _immediate(dbapi):
                stored = _stored_version(dbapi)
                if stored < SCHEMA_VERSION:
                    _migrate(dbapi, stored)
                    stored += 1


@contextmanager
def session_scope(engine: Engine) -> Iterator[Session]:
    """Session that commits on success and rolls back on error."""
    session = Session(engine, expire_on_commit=False)
    try:
        yield session
        session.commit()
    except BaseException:
        session.rollback()
        raise
    finally:
        session.close()
