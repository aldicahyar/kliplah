from app.core.fs import atomic_write_json, project_dir, read_json
from app.core.config import Settings


def test_atomic_write_then_read(tmp_path):
    p = tmp_path / "a" / "x.json"
    p.parent.mkdir()
    atomic_write_json(p, {"k": 1})
    assert read_json(p) == {"k": 1}
    assert not p.with_suffix(".json.tmp").exists()


def test_read_json_ignores_torn_file(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('{"k": ', encoding="utf-8")
    assert read_json(p) is None


def test_read_json_missing(tmp_path):
    assert read_json(tmp_path / "nope.json") is None


def test_project_dir_created(tmp_path):
    s = Settings(_env_file=None, data_dir=tmp_path)
    d = project_dir(s, "abc")
    assert d == tmp_path / "projects" / "abc" and d.is_dir()
