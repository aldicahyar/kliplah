import sqlite3
import threading

import pytest
from sqlalchemy import event, text
from sqlmodel import create_engine

from app.core.config import Settings
from app.core.errors import AppError
from app.db import engine as engine_mod
from app.db.engine import DB_FILENAME, MIGRATIONS, SCHEMA_VERSION, get_engine, init_db, session_scope
from app.db.tables import Clip

V1_CLIP_DDL = """
CREATE TABLE clip (
    id VARCHAR NOT NULL,
    project_id VARCHAR,
    kind VARCHAR NOT NULL,
    origin VARCHAR NOT NULL,
    rank INTEGER NOT NULL,
    score INTEGER NOT NULL,
    llm_score INTEGER,
    heatmap_bonus INTEGER NOT NULL,
    category VARCHAR NOT NULL,
    reason VARCHAR NOT NULL,
    pieces JSON NOT NULL,
    layout_segments JSON NOT NULL,
    style_override JSON NOT NULL,
    style JSON,
    overlays JSON NOT NULL,
    hook_text VARCHAR NOT NULL,
    hook_options JSON NOT NULL,
    subtitles JSON NOT NULL,
    captions JSON NOT NULL,
    status VARCHAR NOT NULL,
    needs_rerender BOOLEAN NOT NULL,
    output_path VARCHAR,
    created_at DATETIME NOT NULL,
    updated_at DATETIME NOT NULL,
    PRIMARY KEY (id)
)
"""


def _clip_columns(engine) -> set[str]:
    with engine.connect() as conn:
        return {r[1] for r in conn.execute(text("PRAGMA table_info(clip)"))}


def _versions(engine) -> list[int]:
    with engine.connect() as conn:
        return [r[0] for r in conn.execute(text("SELECT version FROM schema_version"))]


def _make_v1_db(data_dir) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(data_dir / DB_FILENAME)
    try:
        con.execute("PRAGMA journal_mode=WAL")  # S1 databases are WAL (persistent mode)
        con.execute(V1_CLIP_DDL)
        con.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
        con.execute("INSERT INTO schema_version (version) VALUES (1)")
        con.execute("INSERT INTO clip VALUES ('c1', NULL, 'single', 'ai', 1, 80, NULL, 5, '', '', "
                    "'[]', '[]', '{}', NULL, '[]', '', '[]', '[]', '{}', 'proposed', 0, NULL, "
                    "'2026-11-26 00:00:00', '2026-11-26 00:00:00')")
        con.commit()
    finally:
        con.close()


def test_schema_version_constants():
    assert SCHEMA_VERSION == 2
    assert MIGRATIONS[2] == ["ALTER TABLE clip ADD COLUMN signal_bonus INTEGER NOT NULL DEFAULT 0"]


def test_new_db_created_at_v2(tmp_path):
    engine = get_engine(Settings(data_dir=tmp_path))
    init_db(engine)
    assert _versions(engine) == [2]
    assert "signal_bonus" in _clip_columns(engine)


def test_v1_db_migrates_to_v2_keeping_rows(tmp_path):
    _make_v1_db(tmp_path)
    engine = get_engine(Settings(data_dir=tmp_path))
    init_db(engine)
    assert _versions(engine) == [2]
    with engine.connect() as conn:
        row = conn.execute(text("SELECT id, score, heatmap_bonus, signal_bonus FROM clip")).one()
    assert tuple(row) == ("c1", 80, 5, 0)
    with session_scope(engine) as s:
        clip = s.get(Clip, "c1")
        assert (clip.score, clip.heatmap_bonus, clip.signal_bonus) == (80, 5, 0)


def test_migration_is_idempotent(tmp_path):
    _make_v1_db(tmp_path)
    engine = get_engine(Settings(data_dir=tmp_path))
    init_db(engine)
    init_db(engine)
    assert _versions(engine) == [2]
    assert "signal_bonus" in _clip_columns(engine)


def test_failed_migration_rolls_back_column_and_version(tmp_path, monkeypatch):
    _make_v1_db(tmp_path)
    monkeypatch.setitem(engine_mod.MIGRATIONS, 2, [
        "ALTER TABLE clip ADD COLUMN signal_bonus INTEGER NOT NULL DEFAULT 0",
        "THIS IS NOT SQL",
    ])
    engine = get_engine(Settings(data_dir=tmp_path))
    with pytest.raises(sqlite3.OperationalError):
        init_db(engine)
    assert "signal_bonus" not in _clip_columns(engine)
    assert _versions(engine) == [1]


def test_newer_schema_version_is_rejected(tmp_path):
    engine = get_engine(Settings(data_dir=tmp_path))
    init_db(engine)
    with engine.begin() as conn:
        conn.execute(text("UPDATE schema_version SET version = :v"), {"v": SCHEMA_VERSION + 1})
    with pytest.raises(AppError) as exc:
        init_db(engine)
    assert exc.value.code == "INVALID_STATE"
    assert "versi Kliplah yang lebih baru" in exc.value.message
    assert _versions(engine) == [SCHEMA_VERSION + 1]


def test_concurrent_init_db_on_v1_both_succeed(tmp_path):
    _make_v1_db(tmp_path)
    url = f"sqlite:///{(tmp_path / DB_FILENAME).as_posix()}"
    engines = []
    for _ in range(2):  # separate engines = separate connections, like the API and worker processes
        e = create_engine(url, connect_args={"check_same_thread": False})
        event.listen(e, "connect", engine_mod._on_connect)
        engines.append(e)
    barrier = threading.Barrier(2)
    errors: list[BaseException] = []

    def run(e):
        try:
            barrier.wait()
            init_db(e)
        except BaseException as exc:  # noqa: BLE001  collected and asserted below
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(e,)) for e in engines]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    try:
        assert errors == []
        assert _versions(engines[0]) == [2]
        cols = []
        with engines[0].connect() as conn:
            cols = [r[1] for r in conn.execute(text("PRAGMA table_info(clip)"))]
        assert cols.count("signal_bonus") == 1
    finally:
        for e in engines:
            e.dispose()
