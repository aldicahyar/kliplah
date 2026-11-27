import json
from pathlib import Path

import pytest
from sqlmodel import Session, select

from app import cli, migrate_m1
from app.core.config import Settings
from app.core.models import (Candidate, ClipPlan, Materials, PlatformCaption, ReframeSegment, SubtitleLine,
                             VideoMeta, Word)
from app.db.engine import get_engine, init_db
from app.db.tables import Clip, Project, StyleTemplate
from app.pipeline.report import clip_folder

PID = "W_-dyVlXbLA"
RUN = "small__claude-manual"


def _plan(clip_id: str, rank: int, start: float, end: float, hook: str) -> ClipPlan:
    cap = PlatformCaption(title=f"judul {clip_id}", caption="isi", hashtags=["#a"])
    words = [Word(w="halo", start=0.2, end=0.6), Word(w="semua.", start=0.7, end=1.2)]
    return ClipPlan(
        id=clip_id, rank=rank,
        candidate=Candidate(start_sentence=0, end_sentence=5, start=start, end=end, llm_score=82,
                            category="hot_take", reason="alasan", hook=hook, heatmap_bonus=3, score=85 - rank),
        materials=Materials(hook=hook, tiktok=cap, reels=cap, shorts=cap),
        subtitles=[SubtitleLine(id=0, start=0.2, end=1.2, text="halo semua.", words=words)],
        reframe_segments=[ReframeSegment(start=0.0, end=10.0, mode="track", x=0.3),
                          ReframeSegment(start=10.0, end=end - start, mode="wide", x=0.5)],
        layout="face_crop")


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path / "data")


@pytest.fixture
def m1_project(settings):
    """An M1 project folder: meta.json, source.mp4 and one run with 2 clips, the first one rendered."""
    d = Path(settings.data_dir) / "projects" / PID
    run = d / "runs" / RUN
    run.mkdir(parents=True)
    (d / "source.mp4").write_bytes(b"v")
    meta = VideoMeta(source_type="youtube", project_id=PID, title="Feige datangi RDJ", channel="Kanal",
                     duration=600.0, video_id=PID, heatmap=[{"start": 0, "end": 1, "value": 1.0}])
    (d / "meta.json").write_text(meta.model_dump_json(), encoding="utf-8")
    plans = [_plan("aaaa1111", 1, 100.0, 140.0, "Feige datangi RDJ"), _plan("bbbb2222", 2, 300.0, 330.5, "Kedua")]
    (run / "clips.json").write_text(json.dumps([p.model_dump() for p in plans]), encoding="utf-8")
    final = run / "clips" / clip_folder(plans[0]) / "final.mp4"
    final.parent.mkdir(parents=True)
    final.write_bytes(b"mp4")
    return d, plans, final


def _engine(settings):
    eng = get_engine(settings)
    init_db(eng)
    return eng


def test_migrate_m1_is_idempotent(settings, m1_project):
    _, plans, final = m1_project
    first = migrate_m1.run(settings)
    assert first["projects"] == 1 and first["clips"] == 2

    second = migrate_m1.run(settings)
    assert second["projects"] == 0 and second["clips"] == 0 and second["skipped"] == 3  # 1 project + 2 clips

    with Session(_engine(settings)) as s:
        assert len(s.exec(select(Project)).all()) == 1
        clips = {c.id: c for c in s.exec(select(Clip)).all()}
    assert set(clips) == {"aaaa1111", "bbbb2222"}
    rendered, proposed = clips["aaaa1111"], clips["bbbb2222"]
    assert rendered.status == "rendered" and Path(rendered.output_path) == final
    assert proposed.status == "proposed" and proposed.output_path is None


def test_migrate_m1_maps_project_and_clip_fields(settings, m1_project):
    migrate_m1.run(settings)
    with Session(_engine(settings)) as s:
        p = s.get(Project, PID)
        tpl = s.exec(select(StyleTemplate).where(StyleTemplate.is_default)).one()
        c = s.get(Clip, "aaaa1111")
    assert (p.source_type, p.video_id, p.title, p.channel, p.duration_sec, p.has_heatmap) == (
        "youtube", PID, "Feige datangi RDJ", "Kanal", 600.0, True)
    assert p.youtube_url == f"https://www.youtube.com/watch?v={PID}"
    assert (p.status, p.last_completed_stage, p.moment_source, p.whisper_model, p.content_type) == (
        "ready", "prepare", "claude_manual", "small", "general")
    assert p.style == tpl.style and p.style_template_id == tpl.id

    assert (c.project_id, c.origin, c.kind, c.rank, c.score, c.llm_score, c.heatmap_bonus) == (
        PID, "m1", "single", 1, 84, 82, 3)
    assert (c.category, c.reason, c.hook_text, c.hook_options) == (
        "hot_take", "alasan", "Feige datangi RDJ", ["Feige datangi RDJ"])
    assert c.pieces == [{"project_id": PID, "start": 100.0, "end": 140.0}]
    assert [(seg["start"], seg["end"], seg["layout"]) for seg in c.layout_segments] == [
        (0.0, 10.0, "face_crop"), (10.0, 40.0, "fit_blur")]
    assert c.layout_segments[0]["params"]["x"] == 0.3
    assert c.subtitles[0]["text"] == "halo semua." and c.subtitles[0]["source"] == "whisper"
    assert c.subtitles[0]["words"][1]["w"] == "semua."
    assert c.captions["tiktok"]["title"] == "judul aaaa1111"


def test_migrate_m1_ollama_run_and_several_runs(settings, m1_project):
    d, _, _ = m1_project
    other = d / "runs" / "medium__qwen3-4b"
    other.mkdir(parents=True)
    (other / "clips.json").write_text(json.dumps([_plan("cccc3333", 1, 50.0, 80.0, "Lain").model_dump()]),
                                      encoding="utf-8")
    result = migrate_m1.run(settings)
    assert result == {"projects": 1, "clips": 3, "skipped": 0}
    with Session(_engine(settings)) as s:
        assert {c.id for c in s.exec(select(Clip)).all()} == {"aaaa1111", "bbbb2222", "cccc3333"}


def test_migrate_m1_ignores_folders_without_meta(settings, m1_project):
    (Path(settings.data_dir) / "projects" / "kosong").mkdir()
    migrate_m1.run(settings)
    with Session(_engine(settings)) as s:
        assert [p.id for p in s.exec(select(Project)).all()] == [PID]


def test_cli_migrate_m1_command(settings, m1_project, monkeypatch, capsys):
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    assert cli.main(["migrate-m1"]) == 0
    out = capsys.readouterr().out
    assert "1 project" in out and "2 klip" in out


def test_migrate_skips_folders_without_runs(settings, m1_project):
    # e.g. a v2 project folder, or an M1 project never clipped: meta.json but no runs/*/clips.json
    bare = Path(settings.data_dir) / "projects" / "abc123def456"
    bare.mkdir()
    (bare / "meta.json").write_text(VideoMeta(source_type="upload", project_id="abc123def456", title="t",
                                              duration=10.0).model_dump_json(), encoding="utf-8")
    assert migrate_m1.run(settings) == {"projects": 1, "clips": 2, "skipped": 1}
    with Session(_engine(settings)) as s:
        assert [p.id for p in s.exec(select(Project)).all()] == [PID]


def test_migrate_skips_malformed_clips_json(settings, m1_project, caplog):
    broken = Path(settings.data_dir) / "projects" / "rusak"
    (broken / "runs" / "small__qwen3-4b").mkdir(parents=True)
    (broken / "meta.json").write_text(VideoMeta(source_type="youtube", project_id="rusak", title="t",
                                                duration=10.0).model_dump_json(), encoding="utf-8")
    (broken / "runs" / "small__qwen3-4b" / "clips.json").write_text('[{"id": "x"}]', encoding="utf-8")
    with caplog.at_level("WARNING", logger="kliplah.migrate"):
        result = migrate_m1.run(settings)
    assert result == {"projects": 1, "clips": 2, "skipped": 1}
    assert "rusak" in caplog.text
    with Session(_engine(settings)) as s:
        assert s.get(Project, "rusak") is None


def test_migrate_skips_invalid_clip_span_without_aborting(settings, m1_project, caplog):
    broken = Path(settings.data_dir) / "projects" / "rusak_span"
    (broken / "runs" / "small__qwen3-4b").mkdir(parents=True)
    (broken / "meta.json").write_text(VideoMeta(source_type="youtube", project_id="rusak_span", title="t",
                                                duration=10.0).model_dump_json(), encoding="utf-8")
    bad = _plan("bad1", 1, 20.0, 20.0, "Zero length")
    (broken / "runs" / "small__qwen3-4b" / "clips.json").write_text(json.dumps([bad.model_dump()]), encoding="utf-8")
    with caplog.at_level("WARNING", logger="kliplah.migrate"):
        result = migrate_m1.run(settings)
    assert result == {"projects": 1, "clips": 2, "skipped": 1}
    assert "rusak_span" in caplog.text
    with Session(_engine(settings)) as s:
        assert s.get(Project, "rusak_span") is None and s.get(Clip, "bad1") is None
