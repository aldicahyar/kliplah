from sqlalchemy import text

from app.core.config import Settings
from app.db.engine import get_engine, init_db, session_scope
from app.db.tables import Clip

EXPECTED_TABLES = {"project", "clip", "job", "styletemplate", "asset", "appsetting", "schema_version"}


def _engine(tmp_path):
    return get_engine(Settings(data_dir=tmp_path))


def test_init_db_creates_tables_and_schema_version(tmp_path):
    engine = _engine(tmp_path)
    init_db(engine)
    with engine.connect() as conn:
        names = {r[0] for r in conn.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))}
        assert EXPECTED_TABLES <= names
        assert conn.execute(text("SELECT version FROM schema_version")).scalar_one() == 2
    assert (tmp_path / "kliplah.db").exists()


def test_wal_and_foreign_keys_enabled(tmp_path):
    engine = _engine(tmp_path)
    init_db(engine)
    with engine.connect() as conn:
        assert conn.execute(text("PRAGMA journal_mode")).scalar_one().lower() == "wal"
        assert conn.execute(text("PRAGMA foreign_keys")).scalar_one() == 1


def test_init_db_is_idempotent(tmp_path):
    engine = _engine(tmp_path)
    init_db(engine)
    init_db(engine)
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT version FROM schema_version")).all()
    assert [r[0] for r in rows] == [2]


def test_json_columns_roundtrip(tmp_path):
    engine = _engine(tmp_path)
    init_db(engine)
    pieces = [{"project_id": "p", "start": 1.5, "end": 9.25}]
    captions = {"tiktok": "halo", "tags": ["a", "b"]}
    with session_scope(engine) as s:
        clip = Clip(
            origin="ai", rank=1, score=80, category="lucu", reason="r",
            pieces=pieces, layout_segments=[{"start": 0.0, "end": 7.75, "layout": "full"}],
            hook_text="hook", subtitles=[{"start": 0.0, "end": 1.0, "text": "x"}], captions=captions,
            status="proposed",
        )
        s.add(clip)
        clip_id = clip.id
    with session_scope(engine) as s:
        got = s.get(Clip, clip_id)
        assert got.pieces == pieces
        assert got.captions == captions
        assert got.style_override == {}
        assert got.overlays == []
        assert got.project_id is None
        assert got.created_at.utcoffset() is not None  # timezone-aware UTC after reload


def test_get_engine_cached_per_data_dir(tmp_path):
    a = get_engine(Settings(data_dir=tmp_path / "a"))
    assert get_engine(Settings(data_dir=tmp_path / "a")) is a
    assert get_engine(Settings(data_dir=tmp_path / "b")) is not a
    assert (tmp_path / "a").is_dir()


def test_app_error_fields():
    from app.core.errors import AppError

    e = AppError("NOT_FOUND", "Proyek tidak ditemukan", status=404)
    assert (e.code, e.message, e.status, e.retryable) == ("NOT_FOUND", "Proyek tidak ditemukan", 404, False)
    assert str(e) == "Proyek tidak ditemukan"
    assert AppError("X", "m").status == 400
