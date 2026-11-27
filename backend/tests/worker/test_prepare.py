import json
import subprocess
from pathlib import Path

import pytest
from sqlmodel import Session, select

from app.core.config import Settings
from app.core.models import Candidate, Materials, PlatformCaption, ReframeSegment, VideoMeta
from app.db.engine import get_engine, init_db
from app.db.tables import Clip, Project
from app.worker.prepare import PrepareDeps, build_clips_from_picks, make_proxy, make_thumbnail, prepare_project_clips
from tests.helpers import make_transcript


def _pick(start: float, end: float, score: int, no_ai: bool = False) -> tuple[Candidate, Materials]:
    cap = PlatformCaption(title="judul", caption="isi", hashtags=["#a"])
    c = Candidate(start_sentence=0, end_sentence=1, start=start, end=end, llm_score=None if no_ai else score,
                  category="cerita", reason="alasan", hook=f"hook {score}", heatmap_bonus=5, score=score,
                  no_ai=no_ai)
    return c, Materials(hook=f"hook {score}", tiktok=cap, reels=cap, shorts=cap)


class FakeMedia:
    def __init__(self):
        self.reframed: list[tuple[float, float]] = []
        self.thumbs: list[tuple[float, Path]] = []

    def reframe(self, video, start, end):
        self.reframed.append((start, end))
        return [ReframeSegment(start=0.0, end=round(end - start, 3), mode="track", x=0.4)], "face_crop"

    def thumbnail(self, src, t, out):
        self.thumbs.append((t, out))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"jpg")


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path)


@pytest.fixture
def engine(settings):
    eng = get_engine(settings)
    init_db(eng)
    return eng


@pytest.fixture
def project_id(engine, settings):
    with Session(engine) as s:
        p = Project(youtube_url="https://youtu.be/dQw4w9WgXcQ", style={"subtitle": {"words_per_line": 2}})
        s.add(p)
        s.commit()
        pid = p.id
    d = Path(settings.data_dir) / "projects" / pid
    tdir = d / "transcripts" / "small"
    tdir.mkdir(parents=True)
    (d / "source.mp4").write_bytes(b"v")
    t = make_transcript(30)
    (tdir / "transcript.json").write_text(t.model_dump_json(), encoding="utf-8")
    (d / "silences.json").write_text("[]", encoding="utf-8")
    meta = VideoMeta(source_type="youtube", project_id=pid, title="Judul", duration=t.duration)
    (d / "meta.json").write_text(meta.model_dump_json(), encoding="utf-8")
    return pid


def _clips(engine, pid) -> list[Clip]:
    with Session(engine) as s:
        return list(s.exec(select(Clip).where(Clip.project_id == pid).order_by(Clip.rank)))


def test_prepare_creates_ranked_v2_clips(engine, settings, project_id):
    media = FakeMedia()
    deps = PrepareDeps(settings=settings, reframe=media.reframe, thumbnail=media.thumbnail)
    ids = build_clips_from_picks(engine, project_id, [_pick(0.0, 40.0, 70), _pick(60.0, 100.0, 90)], deps)
    clips = _clips(engine, project_id)
    assert sorted(ids) == sorted(c.id for c in clips) and len(clips) == 2
    top = clips[0]
    assert (top.rank, top.score, clips[1].rank, clips[1].score) == (1, 90, 2, 70)
    assert top.origin == "ai" and top.status == "proposed" and top.kind == "single"
    assert (top.llm_score, top.heatmap_bonus, top.category, top.reason) == (90, 5, "cerita", "alasan")
    assert len(top.pieces) == 1
    piece = top.pieces[0]
    assert piece["project_id"] == project_id and 55.0 <= piece["start"] < piece["end"] <= 105.0
    assert top.layout_segments[0]["layout"] in {"face_crop", "fit_blur"}
    assert top.subtitles and top.subtitles[0]["words"]
    assert all(len(line["words"]) <= 2 for line in top.subtitles)  # words_per_line from project style
    assert top.subtitles[0]["start"] < 5.0  # output timeline
    assert top.hook_text == "hook 90" and top.hook_options == ["hook 90"]
    assert set(top.captions) == {"tiktok", "reels", "shorts"}
    assert top.captions["tiktok"] == {"title": "judul", "caption": "isi", "hashtags": ["#a"]}
    thumb = Path(settings.data_dir) / "projects" / project_id / "clips" / top.id / "thumb.jpg"
    assert thumb.exists()
    t, _ = next(x for x in media.thumbs if x[1] == thumb)
    assert t == pytest.approx(piece["start"] + 1.0)


def test_prepare_marks_fallback_origin(engine, settings, project_id):
    media = FakeMedia()
    deps = PrepareDeps(settings=settings, reframe=media.reframe, thumbnail=media.thumbnail)
    build_clips_from_picks(engine, project_id, [_pick(0.0, 40.0, 60, no_ai=True)], deps)
    (clip,) = _clips(engine, project_id)
    assert clip.origin == "fallback" and clip.llm_score is None


def test_prepare_keeps_approved_clips_on_rerun(engine, settings, project_id):
    media = FakeMedia()
    deps = PrepareDeps(settings=settings, reframe=media.reframe, thumbnail=media.thumbnail)
    first = build_clips_from_picks(engine, project_id, [_pick(0.0, 40.0, 70), _pick(60.0, 100.0, 90)], deps)
    with Session(engine) as s:
        kept = s.get(Clip, first[0])
        kept.status = "approved"
        s.add(kept)
        s.commit()
        kept_rank = kept.rank
    dropped = next(i for i in first if i != first[0])
    pdir = Path(settings.data_dir) / "projects" / project_id
    assert (pdir / "clips" / dropped).exists()

    second = build_clips_from_picks(engine, project_id, [_pick(120.0, 160.0, 50)], deps)
    clips = {c.id: c for c in _clips(engine, project_id)}
    assert set(clips) == {first[0], *second}
    assert clips[first[0]].status == "approved" and clips[first[0]].rank == kept_rank
    assert clips[second[0]].rank == 1 and clips[second[0]].status == "proposed"
    assert not (pdir / "clips" / dropped).exists()
    assert (pdir / "clips" / first[0] / "thumb.jpg").exists()


@pytest.mark.slow
def test_make_proxy_and_thumbnail_real_ffmpeg(synthetic_video, tmp_path):
    proxy = tmp_path / "proxy.mp4"
    make_proxy(synthetic_video, proxy)
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height", "-of", "json", str(proxy)],
                         capture_output=True, text=True, check=True).stdout
    stream = json.loads(out)["streams"][0]
    assert stream["height"] == 480 and stream["width"] % 2 == 0
    assert not (tmp_path / "proxy.partial.mp4").exists()
    mtime = proxy.stat().st_mtime_ns
    make_proxy(synthetic_video, proxy)  # already there: skipped
    assert proxy.stat().st_mtime_ns == mtime

    thumb = tmp_path / "clips" / "abc" / "thumb.jpg"
    make_thumbnail(synthetic_video, 2.0, thumb)
    assert thumb.exists() and thumb.stat().st_size > 0
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width", "-of", "json", str(thumb)],
                         capture_output=True, text=True, check=True).stdout
    assert json.loads(out)["streams"][0]["width"] == 360


def test_prepare_project_clips_forwards_settings_and_normalizes_layouts(engine, settings, project_id, monkeypatch):
    from app.worker import prepare

    seen = {}

    def fake_reframe(video, start, end, models_dir=None):
        seen["models_dir"] = models_dir
        # intentional small gap (0.2 s) and unrounded end to verify normalize_layout_segments runs
        return [ReframeSegment(start=0.0, end=5.0, mode="track", x=0.4),
                ReframeSegment(start=5.2, end=round(end - start - 0.1, 3), mode="wide", x=0.5)], "face_crop"

    monkeypatch.setattr(prepare, "reframe_clip", fake_reframe)
    monkeypatch.setattr(prepare, "make_thumbnail", lambda src, t, out: (out.parent.mkdir(parents=True, exist_ok=True),
                                                                        out.write_bytes(b"jpg")))
    monkeypatch.setattr(prepare, "make_proxy", lambda src, out: out.write_bytes(b"proxy"))

    prepare_project_clips(engine, project_id, [_pick(10.0, 40.0, 85)], settings=settings)
    assert seen["models_dir"] == settings.data_dir / "models"
    [clip] = _clips(engine, project_id)
    dur = round(clip.pieces[0]["end"] - clip.pieces[0]["start"], 3)
    assert clip.layout_segments[0]["start"] == 0.0
    assert clip.layout_segments[0]["end"] == clip.layout_segments[1]["start"]
    assert clip.layout_segments[-1]["end"] == dur
    assert (Path(settings.data_dir) / "projects" / project_id / "proxy.mp4").read_bytes() == b"proxy"


@pytest.mark.slow
def test_e2e_api_worker_process_prepare_and_render(tmp_path, synthetic_video, monkeypatch):
    import shutil
    from fastapi.testclient import TestClient
    from app.api.main import create_app
    from app.pipeline import reframe
    from app.services import clips as clip_service
    from app.worker.main import worker_main
    from app.worker.process import ProcessDeps

    scratch = Settings(_env_file=None, data_dir=tmp_path / "scratch")
    monkeypatch.setattr(reframe, "ensure_face_model", lambda models_dir: (_ for _ in ()).throw(OSError("offline")))

    with TestClient(create_app(scratch)) as client:
        r = client.post("/api/projects", json={"url": "https://youtu.be/dQw4w9WgXcQ", "moment_source": "ollama"})
        assert r.status_code == 201
        pid = r.json()["project"]["id"]

    t = make_transcript(12, each=1.5)  # 18 s within synthetic_video (20 s)

    def fake_ingest(project, s):
        pdir = Path(s.data_dir) / "projects" / project.id
        pdir.mkdir(parents=True, exist_ok=True)
        shutil.copy(synthetic_video, pdir / "source.mp4")
        return pdir / "source.mp4", VideoMeta(source_type="youtube", project_id=project.id, title="E2E",
                                             duration=18.0, video_id="dQw4w9WgXcQ")

    def fake_transcribe(wav, duration, tdir, model, corrections, on_progress=None):
        (tdir / "transcript.json").write_text(t.model_dump_json(), encoding="utf-8")
        return t

    deps = ProcessDeps(
        settings=scratch,
        ingest=fake_ingest,
        extract_audio=lambda src, wav: wav.write_bytes(b"wav"),
        detect_silences=lambda wav: [],
        transcribe=fake_transcribe,
        analyze=lambda tr, meta, max_clips, min_sec, max_sec, ck, s: [_pick(1.0, 9.0, 95)],
    )
    eng = get_engine(scratch)
    worker_main(scratch, once=True, process_deps=deps)  # runs real default_prepare -> prepare_project_clips

    clips = clip_service.list_clips(eng, pid)
    assert len(clips) == 1 and clips[0].status == "proposed"
    pdir = Path(scratch.data_dir) / "projects" / pid
    assert (pdir / "proxy.mp4").is_file()
    assert (pdir / "clips" / clips[0].id / "thumb.jpg").is_file()

    clip_service.approve(eng, clips[0].id)
    worker_main(scratch, once=True)  # runs real run_render_job -> ffmpeg final.mp4
    done = clip_service.get_clip(eng, clips[0].id)
    assert done.status == "rendered" and done.output_path and Path(done.output_path).is_file()
