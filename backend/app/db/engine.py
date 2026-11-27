"""SQLite engine (WAL + foreign keys), schema bootstrap and session helper."""
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import Engine, event, text
from sqlmodel import Session, SQLModel, create_engine

from app.core.config import Settings

DB_FILENAME = "kliplah.db"
SCHEMA_VERSION = 1

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


def init_db(engine: Engine) -> None:
    """Create all tables and the schema_version row. Safe to call repeatedly."""
    import app.db.tables  # noqa: F401  registers tables on SQLModel.metadata

    SQLModel.metadata.create_all(engine)
    with engine.begin() as conn:
        conn.execute(text("CREATE TABLE IF NOT EXISTS schema_version (version INTEGER NOT NULL)"))
        if conn.execute(text("SELECT COUNT(*) FROM schema_version")).scalar_one() == 0:
            conn.execute(text("INSERT INTO schema_version (version) VALUES (:v)"), {"v": SCHEMA_VERSION})


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
