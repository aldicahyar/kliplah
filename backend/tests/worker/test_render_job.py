import json
import shutil
import subprocess
from pathlib import Path

import pytest
from sqlmodel import Session

from app.core.config import Settings
from app.core.errors import AppError
from app.db.engine import get_engine, init_db
from app.db.tables import Clip, Job, Project
from app.pipeline.render import RenderError
from app.worker import queue
from app.worker.render_job import FONTS_DIR, run_render_job


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path)


@pytest.fixture
def engine(settings):
    eng = get_engine(settings)
    init_db(eng)
    return eng


def _setup(engine, settings, *, status="approved", pieces=None, source: Path | None = None) -> tuple[str, str, str]:
    """Project + clip + claimed render job; returns (project_id, clip_id, job_id)."""
    with Session(engine) as s:
        p = Project(youtube_url="https://youtu.be/dQw4w9WgXcQ")
        s.add(p)
        s.commit()
        pid = p.id
        clip = Clip(project_id=pid, origin="ai", rank=1, score=80, hook_text="Hook keren",
                    pieces=pieces or [{"project_id": pid, "start": 2.0, "end": 8.0}],
                    layout_segments=[{"start": 0.0, "end": 6.0, "layout": "fit_blur"}],
                    subtitles=[{"id": 0, "start": 0.5, "end": 2.0, "text": "halo semua",
                                "words": [{"w": "halo", "start": 0.5, "end": 1.0},
                                          {"w": "semua", "start": 1.1, "end": 2.0}]}],
                    status=status, needs_rerender=True)
        s.add(clip)
        s.commit()
        cid = clip.id
        queue.enqueue(s, "render", pid, cid)
        job = queue.claim_next(s)
        jid = job.id
    d = Path(settings.data_dir) / "projects" / pid
    d.mkdir(parents=True, exist_ok=True)
    if source is None:
        (d / "source.mp4").write_bytes(b"v")
    else:
        shutil.copy(source, d / "source.mp4")
    return pid, cid, jid


def _get(engine, model, id_):
    with Session(engine) as s:
        return s.get(model, id_)


class FakeRender:
    def __init__(self, error: Exception | None = None):
        self.error = error
        self.calls = []

    def __call__(self, source, plan, out_dir, presets, fonts_dir, on_progress=None):
        self.calls.append((source, plan, out_dir, presets, fonts_dir))
        on_progress(3.0)
        if self.error:
            raise self.error
        out_dir.mkdir(parents=True, exist_ok=True)
        final = out_dir / "final.mp4"
        final.write_bytes(b"mp4")
        return final


def test_render_job_sets_rendered_and_output_path(engine, settings):
    pid, cid, jid = _setup(engine, settings)
    fake = FakeRender()
    run_render_job(engine, jid, render=fake, settings=settings)

    pdir = Path(settings.data_dir) / "projects" / pid
    source, plan, out_dir, presets, fonts_dir = fake.calls[0]
    assert source == pdir / "source.mp4"
    assert out_dir == pdir / "clips" / cid
    assert fonts_dir == FONTS_DIR
    assert (plan.candidate.start, plan.candidate.end) == (2.0, 8.0)
    assert plan.materials.hook == "Hook keren"
    assert presets["presets"][plan.caption_preset]["size"] == 104

    clip = _get(engine, Clip, cid)
    assert clip.status == "rendered"
    assert clip.output_path == str(pdir / "clips" / cid / "final.mp4")
    assert clip.needs_rerender is False
    job = _get(engine, Job, jid)
    assert job.status == "succeeded" and job.stage == "render" and job.progress == 1.0


def test_rerender_from_rendered_clears_needs_rerender(engine, settings):
    _, cid, jid = _setup(engine, settings, status="rendered")  # _setup marks needs_rerender=True
    run_render_job(engine, jid, render=FakeRender(), settings=settings)

    clip = _get(engine, Clip, cid)
    assert clip.status == "rendered" and clip.needs_rerender is False
    assert _get(engine, Job, jid).status == "succeeded"


def test_render_job_reports_fractional_progress(engine, settings):
    _, _, jid = _setup(engine, settings)
    seen = []

    def render(source, plan, out_dir, presets, fonts_dir, on_progress=None):
        on_progress(3.0)  # 3 s of a 6 s clip
        with Session(engine) as s:
            seen.append(s.get(Job, jid).progress)
        raise RenderError("RENDER_FAILED", "boom")

    run_render_job(engine, jid, render=render, settings=settings)
    assert seen == [0.5]


def test_render_failure_marks_render_failed(engine, settings):
    _, cid, jid = _setup(engine, settings)
    run_render_job(engine, jid, render=FakeRender(RenderError("RENDER_FAILED", "Render gagal: x")), settings=settings)

    assert _get(engine, Clip, cid).status == "render_failed"
    job = _get(engine, Job, jid)
    assert job.status == "failed" and job.error_code == "RENDER_FAILED"


def test_retryable_render_error_returns_clip_to_approved(engine, settings):
    _, cid, jid = _setup(engine, settings)
    err = AppError("DISK_FULL", "Disk penuh", retryable=True)
    run_render_job(engine, jid, render=FakeRender(err), settings=settings)

    assert _get(engine, Job, jid).status == "queued"
    assert _get(engine, Clip, cid).status == "approved"


def test_render_multi_piece_rejected_until_s4(engine, settings):
    with Session(engine) as s:
        p = Project()
        s.add(p)
        s.commit()
        other = p.id
    pieces = [{"project_id": other, "start": 0.0, "end": 3.0}, {"project_id": other, "start": 5.0, "end": 8.0}]
    _, cid, jid = _setup(engine, settings, pieces=pieces)
    fake = FakeRender()
    run_render_job(engine, jid, render=fake, settings=settings)

    assert fake.calls == []
    assert _get(engine, Clip, cid).status == "render_failed"
    job = _get(engine, Job, jid)
    assert job.status == "failed" and job.error_code == "VALIDATION" and "S4" in job.error_message


def test_render_missing_source_fails_invalid_state(engine, settings):
    pid, cid, jid = _setup(engine, settings)
    (Path(settings.data_dir) / "projects" / pid / "source.mp4").unlink()
    run_render_job(engine, jid, render=FakeRender(), settings=settings)

    job = _get(engine, Job, jid)
    assert job.status == "failed" and job.error_code == "INVALID_STATE"
    assert job.error_message == "Video sumber sudah dihapus"
    assert _get(engine, Clip, cid).status == "render_failed"


def test_render_rejects_unapproved_clip(engine, settings):
    _, cid, jid = _setup(engine, settings, status="proposed")
    fake = FakeRender()
    run_render_job(engine, jid, render=fake, settings=settings)

    assert fake.calls == []
    assert _get(engine, Job, jid).error_code == "INVALID_STATE"
    assert _get(engine, Clip, cid).status == "proposed"


@pytest.mark.slow
def test_render_job_real_ffmpeg(engine, settings, synthetic_video):
    pid, cid, jid = _setup(engine, settings, source=synthetic_video)
    run_render_job(engine, jid, settings=settings)

    job = _get(engine, Job, jid)
    assert job.status == "succeeded", job.error_message
    final = Path(settings.data_dir) / "projects" / pid / "clips" / cid / "final.mp4"
    assert _get(engine, Clip, cid).output_path == str(final)
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "json", str(final)], capture_output=True, text=True)
    stream = json.loads(out.stdout)["streams"][0]
    assert (stream["width"], stream["height"]) == (1080, 1920)


def test_render_uses_range_file_and_local_times(engine, settings):
    from app.core.models import VideoMeta

    pid, cid, jid = _setup(engine, settings, pieces=[{"project_id": "placeholder", "start": 1805.0, "end": 1812.0}])
    with Session(engine) as s:
        clip = s.get(Clip, cid)
        clip.pieces = [{"project_id": pid, "start": 1805.0, "end": 1812.0}]
        clip.layout_segments = [{"start": 0.0, "end": 7.0, "layout": "fit_blur"}]
        s.add(clip)
        s.commit()
    pdir = Path(settings.data_dir) / "projects" / pid
    (pdir / "source_r1.mp4").write_bytes(b"v1")
    meta = VideoMeta(
        source_type="youtube", project_id=pid, title="T", duration=3600.0,
        ranges=[{"index": 0, "start": 0.0, "end": 100.0, "file": "source_r0.mp4"},
                {"index": 1, "start": 1800.0, "end": 1900.0, "file": "source_r1.mp4"}],
    )
    (pdir / "meta.json").write_text(meta.model_dump_json(), encoding="utf-8")

    fake = FakeRender()
    run_render_job(engine, jid, render=fake, settings=settings)
    source, plan, _, _, _ = fake.calls[0]
    assert source == pdir / "source_r1.mp4"
    assert (plan.candidate.start, plan.candidate.end) == (5.0, 12.0)
    assert plan.subtitles[0].start == 0.5 and plan.subtitles[0].end == 2.0


def test_render_legacy_project_still_works(engine, settings):
    from app.core.models import VideoMeta

    pid, cid, jid = _setup(engine, settings)
    pdir = Path(settings.data_dir) / "projects" / pid
    (pdir / "meta.json").write_text(
        VideoMeta(source_type="youtube", project_id=pid, title="T", duration=120.0).model_dump_json(),
        encoding="utf-8",
    )
    fake = FakeRender()
    run_render_job(engine, jid, render=fake, settings=settings)
    source, plan, _, _, _ = fake.calls[0]
    assert source == pdir / "source.mp4"
    assert (plan.candidate.start, plan.candidate.end) == (2.0, 8.0)
    assert _get(engine, Clip, cid).status == "rendered"


@pytest.mark.slow
def test_render_range_project_real_ffmpeg(engine, settings, synthetic_video):
    from app.core.models import VideoMeta

    pid, cid, jid = _setup(engine, settings)
    with Session(engine) as s:
        clip = s.get(Clip, cid)
        clip.pieces = [{"project_id": pid, "start": 105.0, "end": 112.0}]
        clip.layout_segments = [{"start": 0.0, "end": 7.0, "layout": "fit_blur"}]
        s.add(clip)
        s.commit()
    pdir = Path(settings.data_dir) / "projects" / pid
    shutil.copy(synthetic_video, pdir / "source_r0.mp4")
    shutil.copy(synthetic_video, pdir / "source_r1.mp4")
    meta = VideoMeta(
        source_type="youtube", project_id=pid, title="T", duration=300.0,
        ranges=[{"index": 0, "start": 0.0, "end": 20.0, "file": "source_r0.mp4"},
                {"index": 1, "start": 100.0, "end": 120.0, "file": "source_r1.mp4"}],
    )
    (pdir / "meta.json").write_text(meta.model_dump_json(), encoding="utf-8")

    run_render_job(engine, jid, settings=settings)
    job = _get(engine, Job, jid)
    assert job.status == "succeeded", job.error_message
    final = pdir / "clips" / cid / "final.mp4"
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height:format=duration",
                          "-of", "json", str(final)], capture_output=True, text=True, check=True)
    info = json.loads(out.stdout)
    stream = next(s for s in info["streams"] if s.get("width"))
    assert (stream["width"], stream["height"]) == (1080, 1920)
    assert float(info["format"]["duration"]) == pytest.approx(7.0, abs=1.0)
