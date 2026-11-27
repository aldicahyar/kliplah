import json

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session

from app.api.main import create_app
from app.core.config import REPO_DIR, Settings
from app.core.models import VideoMeta
from app.db.engine import get_engine
from app.db.tables import Clip, Project
from app.domain.style import default_style
from tests.helpers import make_transcript

VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path)


@pytest.fixture
def client(settings):
    with TestClient(create_app(settings)) as c:  # `with` runs startup (init_db + default template)
        yield c


@pytest.fixture
def engine(client, settings):
    return get_engine(settings)


def _add(engine, row):
    with Session(engine, expire_on_commit=False) as s:
        s.add(row)
        s.commit()
    return row


def _clip_body(s: int, e: int) -> dict:
    return {"start_sentence": s, "end_sentence": e, "score": 80, "category": "hot_take", "reason": "r",
            "hook": "hook", **{p: {"title": "Judul", "caption": "c", "hashtags": ["#a"]}
                               for p in ("tiktok", "reels", "shorts")}}


def test_create_project_returns_201_and_queued_job(client):
    r = client.post("/api/projects", json={"url": VIDEO_URL, "content_type": "podcast"})
    assert r.status_code == 201
    body = r.json()
    assert body["project"]["status"] == "queued"
    assert body["project"]["video_id"] == "dQw4w9WgXcQ"
    assert body["job"]["type"] == "process" and body["job"]["status"] == "queued"
    assert body["job"]["project_id"] == body["project"]["id"]
    got = client.get(f"/api/projects/{body['project']['id']}").json()
    assert got["project"]["id"] == body["project"]["id"] and got["active_job"]["id"] == body["job"]["id"]
    assert [p["id"] for p in client.get("/api/projects").json()] == [body["project"]["id"]]


def test_upload_project_moves_file_into_uploads(client, settings):
    r = client.post("/api/projects/upload", files={"file": ("rekaman.mp4", b"\x00" * 1000, "video/mp4")},
                    data={"content_type": "general", "max_clips": "3"})
    assert r.status_code == 201, r.text
    p = r.json()["project"]
    assert p["source_type"] == "upload" and p["original_filename"] == "rekaman.mp4" and p["max_clips"] == 3
    assert (settings.data_dir / "uploads" / p["id"] / "rekaman.mp4").stat().st_size == 1000
    assert list((settings.data_dir / "uploads" / "_incoming").iterdir()) == []


def test_error_format_for_invalid_url(client):
    r = client.post("/api/projects", json={"url": "https://example.com/video"})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["code"] == "INVALID_URL" and err["message"]


@pytest.mark.parametrize("method, url, kwargs, expected", [
    ("post", "/api/projects/x/clips", {"json": {"start": "awal", "end": 5}},
     "Data permintaan tidak valid pada 'start': harus berupa angka."),
    ("post", "/api/projects", {"json": {}}, "Data permintaan tidak valid pada 'url': wajib diisi."),
    ("post", "/api/projects/x/moments-import?part=satu", {"content": "x"},
     "Data permintaan tidak valid pada 'part': harus bilangan bulat."),
    ("post", "/api/clips/bulk", {"json": {"ids": "a", "action": "approve"}},
     "Data permintaan tidak valid pada 'ids': harus berupa daftar."),
    ("post", "/api/projects", {"content": "{rusak", "headers": {"Content-Type": "application/json"}},
     "Data permintaan tidak valid pada 'permintaan': JSON tidak valid."),
])
def test_request_validation_error_is_indonesian_envelope(client, method, url, kwargs, expected):
    r = getattr(client, method)(url, **kwargs)
    assert r.status_code == 422
    assert r.json() == {"error": {"code": "VALIDATION", "message": expected}}


def test_validation_phrases_cover_bounds_literal_extra_and_fallback():
    from app.api.main import _phrase

    assert _phrase({"type": "greater_than", "ctx": {"gt": 0}}) == "harus lebih besar dari 0"
    assert _phrase({"type": "less_than_equal", "ctx": {"le": 5}}) == "maksimal 5"
    assert _phrase({"type": "string_too_short"}) == "terlalu pendek"
    assert _phrase({"type": "literal_error", "msg": "Input should be 'a'"}) == "bukan salah satu pilihan yang diizinkan"
    assert _phrase({"type": "extra_forbidden"}) == "field tidak dikenal"
    assert _phrase({"type": "uuid_parsing"}) == "tipe datanya tidak sesuai"
    assert _phrase({"type": "something_new", "msg": "English text"}) == "nilai tidak valid"


def test_unknown_clip_returns_404_not_found_code(client):
    r = client.get("/api/clips/tidak-ada")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "NOT_FOUND"


def test_moments_flow_export_import_finish(client, engine, settings):
    p = _add(engine, Project(status="awaiting_moments", last_completed_stage="transcribe", whisper_model="small",
                             max_clips=8, min_clip_sec=30, max_clip_sec=90))
    t = make_transcript(20, each=10.0)
    d = settings.data_dir / "projects" / p.id
    (d / "transcripts" / "small").mkdir(parents=True)
    (d / "transcripts" / "small" / "transcript.json").write_text(t.model_dump_json(), encoding="utf-8")
    (d / "meta.json").write_text(VideoMeta(source_type="youtube", project_id=p.id, title="Judul",
                                           duration=t.duration).model_dump_json(), encoding="utf-8")

    parts = client.get(f"/api/projects/{p.id}/moments-export").json()
    assert len(parts) == 1 and parts[0]["part"] == 1 and parts[0]["text"]

    bad = client.post(f"/api/projects/{p.id}/moments-import?part=1", content="maaf, saya tidak bisa",
                      headers={"Content-Type": "text/plain"})
    assert bad.status_code == 400 and bad.json()["error"]["code"] == "INVALID_IMPORT"

    reply = "Berikut:\n```json\n" + json.dumps({"clips": [_clip_body(0, 4)]}) + "\n```"
    r = client.post(f"/api/projects/{p.id}/moments-import", content=reply.encode("utf-8"),
                    headers={"Content-Type": "text/plain; charset=utf-8"})
    assert r.status_code == 200 and r.json() == {"imported": [1], "missing": []}

    job = client.post(f"/api/projects/{p.id}/moments-finish").json()
    assert job["type"] == "process" and job["status"] == "queued"
    assert client.get(f"/api/projects/{p.id}").json()["project"]["status"] == "processing"
    assert client.get(f"/api/projects/{p.id}/transcript").json()["duration"] == t.duration


def test_patch_clip_style_override_and_get_effective(client, engine):
    p = _add(engine, Project(status="ready", style=default_style()))
    c = _add(engine, Clip(project_id=p.id, origin="ai", rank=1, score=70))
    r = client.patch(f"/api/clips/{c.id}", json={"style_override": {"subtitle": {"size": 80}}})
    assert r.status_code == 200, r.text
    assert r.json()["effective_style"]["subtitle"]["size"] == 80
    got = client.get(f"/api/clips/{c.id}").json()
    assert got["style_override"] == {"subtitle": {"size": 80}}
    assert got["effective_style"]["subtitle"]["size"] == 80
    assert got["effective_style"]["hook"]["size"] == default_style()["hook"]["size"]


def test_clip_actions_bulk_and_list(client, engine):
    p = _add(engine, Project(status="ready", style=default_style()))
    a = _add(engine, Clip(project_id=p.id, origin="ai", rank=1, score=70))
    b = _add(engine, Clip(project_id=p.id, origin="ai", rank=2, score=60))
    assert client.post(f"/api/clips/{a.id}/reject").json()["status"] == "rejected"
    assert [c["id"] for c in client.get(f"/api/projects/{p.id}/clips").json()] == [b.id]
    assert len(client.get(f"/api/projects/{p.id}/clips?include_rejected=true").json()) == 2
    assert client.post(f"/api/clips/{a.id}/restore").json()["status"] == "proposed"
    r = client.post("/api/clips/bulk", json={"ids": [a.id, "hilang"], "action": "approve"}).json()
    assert r == {"ok": [a.id], "errors": {"hilang": "NOT_FOUND"}}


def test_proxy_supports_range_requests(client, engine, settings):
    p = _add(engine, Project(status="ready"))
    d = settings.data_dir / "projects" / p.id
    d.mkdir(parents=True)
    (d / "proxy.mp4").write_bytes(bytes(range(256)) * 4)
    r = client.get(f"/api/projects/{p.id}/proxy", headers={"Range": "bytes=0-99"})
    assert r.status_code == 206
    assert len(r.content) == 100 and r.content == bytes(range(100))
    assert r.headers["content-range"] == "bytes 0-99/1024"
    assert client.get(f"/api/projects/{p.id}/proxy").status_code == 200


def test_missing_files_return_404_not_found(client, engine):
    p = _add(engine, Project(status="ready"))
    c = _add(engine, Clip(project_id=p.id, origin="ai", rank=1, score=70))
    for url in (f"/api/projects/{p.id}/proxy", f"/api/projects/{p.id}/transcript", f"/api/clips/{c.id}/thumbnail",
                f"/api/clips/{c.id}/download", f"/api/clips/{c.id}/captions"):
        r = client.get(url)
        assert r.status_code == 404, url
        assert r.json()["error"]["code"] == "NOT_FOUND"


def test_download_and_captions_of_rendered_clip(client, engine, settings):
    p = _add(engine, Project(status="ready"))
    out = settings.data_dir / "projects" / p.id / "clips" / "c1"
    out.mkdir(parents=True)
    (out / "final.mp4").write_bytes(b"mp4")
    (out / "captions.txt").write_text("TikTok: halo", encoding="utf-8")
    c = _add(engine, Clip(project_id=p.id, origin="ai", rank=3, score=70, status="rendered",
                          output_path=str(out / "final.mp4")))
    r = client.get(f"/api/clips/{c.id}/download")
    assert r.status_code == 200 and r.content == b"mp4"
    assert r.headers["content-disposition"].startswith("attachment;")
    assert f"3_{c.id}.mp4" in r.headers["content-disposition"]
    assert client.get(f"/api/clips/{c.id}/captions").text == "TikTok: halo"


def test_style_templates_crud_and_save_clip_as_template(client, engine):
    templates = client.get("/api/style-templates").json()
    assert [t["name"] for t in templates] == ["Default"] and templates[0]["is_default"]
    r = client.post("/api/style-templates", json={"name": "Tebal", "style": default_style()})
    assert r.status_code == 201
    tid = r.json()["id"]
    assert client.patch(f"/api/style-templates/{tid}", json={"name": "Tebal 2"}).json()["name"] == "Tebal 2"
    assert client.post(f"/api/style-templates/{tid}/default").json()["is_default"] is True
    assert client.delete(f"/api/style-templates/{tid}").json()["error"]["code"] == "INVALID_STATE"
    p = _add(engine, Project(status="ready", style=default_style()))
    c = _add(engine, Clip(project_id=p.id, origin="ai", rank=1, score=70))
    r = client.post(f"/api/clips/{c.id}/save-as-template", json={"name": "Dari klip"})
    assert r.status_code == 201 and r.json()["name"] == "Dari klip"
    default_id = templates[0]["id"]
    assert client.delete(f"/api/style-templates/{default_id}").status_code == 204


def test_style_schema_endpoint_matches_shared_files(client):
    body = client.get("/api/system/style-schema").json()
    shared = REPO_DIR / "shared"
    assert body["schema"] == json.loads((shared / "style_schema.json").read_text(encoding="utf-8"))
    assert body["default"] == json.loads((shared / "style_default.json").read_text(encoding="utf-8"))


def test_system_profiles_settings_and_health(client, monkeypatch):
    from app.api import routes_system

    assert "general" in client.get("/api/system/profiles").json()
    assert client.put("/api/system/settings", json={"max_clips": 5}).json()["max_clips"] == 5
    assert client.get("/api/system/settings").json()["max_clips"] == 5
    bad = client.put("/api/system/settings", json={"max_clips": 0})
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "VALIDATION"

    monkeypatch.setattr(routes_system.check_setup, "_ollama_tags", lambda url: None)
    h = client.get("/api/system/health").json()
    assert set(h) == {"ffmpeg", "ffmpeg_has_libass", "ollama", "ollama_model", "whisper_model_cached",
                      "disk_free_gb", "data_dir", "db"}
    assert h["db"] is True and h["ollama"] is False and h["ollama_model"] is False


def test_app_settings_content_type_and_injected_default_apply_to_projects(tmp_path):
    custom = Settings(_env_file=None, data_dir=tmp_path, default_moment_source="ollama")
    with TestClient(create_app(custom)) as c:
        assert c.get("/api/system/settings").json()["moment_source"] == "ollama"
        assert c.put("/api/system/settings", json={"content_type": "podcast"}).json()["content_type"] == "podcast"
        created = c.post("/api/projects", json={"url": VIDEO_URL}).json()["project"]
        assert (created["content_type"], created["min_clip_sec"], created["max_clip_sec"]) == ("podcast", 30, 90)
        uploaded = c.post("/api/projects/upload",
                          files={"file": ("v.mp4", b"\x00" * 16, "video/mp4")}).json()["project"]
        assert (uploaded["content_type"], uploaded["min_clip_sec"], uploaded["max_clip_sec"]) == ("podcast", 30, 90)


def test_cors_allows_only_vite_origin(client):
    ok = client.options("/api/projects", headers={"Origin": "http://localhost:5173",
                                                   "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:5173"
    bad = client.options("/api/projects", headers={"Origin": "http://evil.example",
                                                    "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in bad.headers
    plain = client.get("/api/projects", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in plain.headers


def test_unexpected_error_returns_500_unknown(settings, monkeypatch):
    from app.api import routes_projects

    def boom(engine):
        raise RuntimeError("rusak")

    monkeypatch.setattr(routes_projects.projects, "list_projects", boom)
    with TestClient(create_app(settings)) as c:  # handled inside the app: nothing reaches the server layer
        r = c.get("/api/projects", headers={"Origin": "http://localhost:5173"})
    assert r.status_code == 500
    assert r.json() == {"error": {"code": "UNKNOWN", "message": "Terjadi kesalahan tak terduga."}}
    assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"


def test_error_envelopes_carry_cors_header(client):
    origin = {"Origin": "http://localhost:5173"}
    for r in (client.get("/api/clips/tidak-ada", headers=origin),
              client.post("/api/projects", json={}, headers=origin)):
        assert r.status_code in (404, 422)
        assert r.headers.get("access-control-allow-origin") == "http://localhost:5173"
