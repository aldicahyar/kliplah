import json
from pathlib import Path

import pytest
from sqlmodel import Session

from app.core.config import Settings
from app.core.models import Candidate, Materials, PlatformCaption, VideoMeta
from app.db.engine import get_engine, init_db
from app.db.tables import Job, Project
from app.pipeline.ingest import IngestError
from app.worker import process, queue
from app.worker.process import STAGES, ProcessDeps, load_picks, run_process_job, save_picks
from tests.helpers import make_transcript


def _pick(start=0.0, end=40.0) -> tuple[Candidate, Materials]:
    cap = PlatformCaption(title="judul", caption="isi", hashtags=["#a"])
    c = Candidate(start_sentence=0, end_sentence=6, start=start, end=end, llm_score=80, category="cerita",
                  reason="alasan", hook="hook", score=80)
    return c, Materials(hook="hook", tiktok=cap, reels=cap, shorts=cap)


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path)


@pytest.fixture
def engine(settings):
    eng = get_engine(settings)
    init_db(eng)
    return eng


class Recorder:
    """Mocked pipeline that records the order of calls."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.calls: list[str] = []
        self.after = None  # optional hook(name) run after each mocked call
        self.transcript = make_transcript(20)
        self.picks = [_pick()]
        self.prepared = None
        self.progress: list[float] = []

    def _hit(self, name):
        self.calls.append(name)
        if self.after:
            self.after(name)

    def ingest(self, project, settings):
        self._hit("ingest")
        d = Path(settings.data_dir) / "projects" / project.id
        d.mkdir(parents=True, exist_ok=True)
        (d / "source.mp4").write_bytes(b"v")
        return d / "source.mp4", VideoMeta(source_type="youtube", project_id=project.id, title="Judul",
                                           channel="Kanal", duration=120.0, video_id="dQw4w9WgXcQ",
                                           heatmap=[{"start": 0, "end": 1, "value": 1.0}])

    def extract_audio(self, src, wav):
        self._hit("extract_audio")
        wav.write_bytes(b"w")

    def detect_silences(self, wav):
        self._hit("detect_silences")
        return [(1.0, 1.5)]

    def transcribe(self, wav, duration, tdir, model, corrections, on_progress=None):
        on_progress(0.5)
        self._hit("transcribe")
        (tdir / "transcript.json").write_text(self.transcript.model_dump_json(), encoding="utf-8")
        return self.transcript

    def analyze(self, t, meta, max_clips, min_sec, max_sec, checkpoint_dir, settings):
        self._hit("analyze")
        return self.picks

    def prepare(self, engine, project_id, picks):
        self._hit("prepare")
        self.prepared = picks

    def deps(self) -> ProcessDeps:
        return ProcessDeps(settings=self.settings, ingest=self.ingest, extract_audio=self.extract_audio,
                           detect_silences=self.detect_silences, transcribe=self.transcribe,
                           analyze=self.analyze, prepare=self.prepare)


def _setup(engine, **fields) -> tuple[str, str]:
    """A project plus its claimed (running) process job."""
    fields.setdefault("youtube_url", "https://youtu.be/dQw4w9WgXcQ")
    with Session(engine, expire_on_commit=False) as s:
        p = Project(**fields)
        s.add(p)
        s.commit()
        job = queue.enqueue(s, "process", p.id)
        assert queue.claim_next(s).id == job.id
        return p.id, job.id


def _rows(engine, pid, jid) -> tuple[Project, Job]:
    with Session(engine) as s:
        return s.get(Project, pid), s.get(Job, jid)


def test_stages_constant():
    assert STAGES == ("ingest", "audio", "transcribe", "moments", "prepare")


def test_process_runs_all_stages_with_ollama_source(engine, settings):
    pid, jid = _setup(engine, moment_source="ollama")
    rec = Recorder(settings)
    assert run_process_job(engine, jid, rec.deps()) == "done"
    assert rec.calls == ["ingest", "extract_audio", "detect_silences", "transcribe", "analyze", "prepare"]
    p, j = _rows(engine, pid, jid)
    assert p.status == "ready" and p.last_completed_stage == "prepare"
    assert j.status == "succeeded" and j.progress == 1.0
    assert (p.title, p.channel, p.duration_sec, p.video_id, p.has_heatmap) == (
        "Judul", "Kanal", 120.0, "dQw4w9WgXcQ", True)
    d = settings.data_dir / "projects" / pid
    assert json.loads((d / "silences.json").read_text(encoding="utf-8")) == [[1.0, 1.5]]
    assert json.loads((d / "meta.json").read_text(encoding="utf-8"))["project_id"] == pid
    assert load_picks(d / "picks.json") == rec.picks
    assert rec.prepared == rec.picks


def test_transcribe_progress_is_reported(engine, settings):
    pid, jid = _setup(engine)
    rec = Recorder(settings)
    seen = []
    rec.after = lambda name: name == "transcribe" and seen.append(_rows(engine, pid, jid)[1])
    run_process_job(engine, jid, rec.deps())
    assert (seen[0].stage, seen[0].progress) == ("transcribe", 0.5)


def test_claude_manual_pauses_after_transcribe(engine, settings):
    pid, jid = _setup(engine, moment_source="claude_manual")
    rec = Recorder(settings)
    assert run_process_job(engine, jid, rec.deps()) == "paused"
    assert "analyze" not in rec.calls and "prepare" not in rec.calls
    p, j = _rows(engine, pid, jid)
    assert p.status == "awaiting_moments" and p.last_completed_stage == "transcribe"
    assert j.status == "succeeded"


def test_resume_skips_completed_stages(engine, settings):
    pid, jid = _setup(engine, moment_source="ollama", last_completed_stage="transcribe", whisper_model="org/m:v3")
    rec = Recorder(settings)
    d = settings.data_dir / "projects" / pid
    tdir = d / "transcripts" / "org-m-v3"
    tdir.mkdir(parents=True)
    (tdir / "transcript.json").write_text(rec.transcript.model_dump_json(), encoding="utf-8")
    meta = VideoMeta(source_type="youtube", project_id=pid, title="Judul", duration=120.0)
    (d / "meta.json").write_text(meta.model_dump_json(), encoding="utf-8")
    seen = {}

    def analyze(t, m, max_clips, min_sec, max_sec, checkpoint_dir, s):
        rec._hit("analyze")
        seen.update(t=t, meta=m, ck=checkpoint_dir)
        return rec.picks

    deps = rec.deps()
    deps.analyze = analyze
    assert run_process_job(engine, jid, deps) == "done"
    assert rec.calls == ["analyze", "prepare"]
    assert seen == {"t": rec.transcript, "meta": meta, "ck": tdir / "checkpoints"}
    assert _rows(engine, pid, jid)[0].status == "ready"


def test_resume_prepare_reads_saved_picks(engine, settings):
    pid, jid = _setup(engine, moment_source="claude_manual", last_completed_stage="moments")
    rec = Recorder(settings)
    picks = [_pick(0, 40), _pick(60, 100)]
    d = settings.data_dir / "projects" / pid
    d.mkdir(parents=True)
    save_picks(d / "picks.json", picks)
    assert run_process_job(engine, jid, rec.deps()) == "done"
    assert rec.calls == ["prepare"] and rec.prepared == picks


def test_yields_between_stages_when_render_queued(engine, settings):
    pid, jid = _setup(engine, moment_source="ollama")
    rec = Recorder(settings)

    def queue_render(name):
        if name == "detect_silences":
            with Session(engine) as s:
                queue.enqueue(s, "render", pid)

    rec.after = queue_render
    assert run_process_job(engine, jid, rec.deps()) == "yield"
    assert "transcribe" not in rec.calls
    p, j = _rows(engine, pid, jid)
    assert p.last_completed_stage == "audio" and p.status == "processing"
    assert j.status == "running"  # the worker loop requeues it


def test_ingest_error_maps_to_app_error(engine, settings):
    pid, jid = _setup(engine)
    deps = Recorder(settings).deps()

    def boom(project, s):
        raise IngestError("VIDEO_UNAVAILABLE", "Video ini privat, dihapus, atau dibatasi umur.")

    deps.ingest = boom
    run_process_job(engine, jid, deps)
    p, j = _rows(engine, pid, jid)
    assert j.status == "failed" and j.error_code == "VIDEO_UNAVAILABLE"
    assert j.error_message == "Video ini privat, dihapus, atau dibatasi umur."
    assert p.status == "failed" and p.last_completed_stage is None


def test_retryable_error_requeues_job_and_keeps_project_processing(engine, settings):
    pid, jid = _setup(engine)
    deps = Recorder(settings).deps()

    def flaky(project, s):
        raise IngestError("NETWORK", "Gagal mengunduh", retryable=True)

    deps.ingest = flaky
    run_process_job(engine, jid, deps)
    p, j = _rows(engine, pid, jid)
    assert j.status == "queued" and p.status == "processing"


def test_unknown_exception_fails_job_with_unknown(engine, settings):
    pid, jid = _setup(engine)
    deps = Recorder(settings).deps()
    deps.detect_silences = lambda wav: 1 / 0
    run_process_job(engine, jid, deps)
    p, j = _rows(engine, pid, jid)
    assert j.status == "failed" and j.error_code == "UNKNOWN"
    assert p.status == "failed" and p.last_completed_stage == "ingest"


def test_clip_durations_default_from_profile(engine, settings):
    pid, jid = _setup(engine, moment_source="ollama", content_type="stream", min_clip_sec=0, max_clip_sec=0)
    rec = Recorder(settings)
    seen = {}

    def analyze(t, meta, max_clips, min_sec, max_sec, checkpoint_dir, s):
        seen.update(min=min_sec, max=max_sec, n=max_clips)
        return rec.picks

    deps = rec.deps()
    deps.analyze = analyze
    assert run_process_job(engine, jid, deps) == "done"
    assert seen == {"min": 15, "max": 35, "n": 8}


def test_default_ingest_upload_path_from_project_row(engine, settings, monkeypatch):
    pid, jid = _setup(engine, source_type="upload", youtube_url=None, original_filename="rekaman.mp4")
    seen = {}

    def fake_ingest_file(path, s, project_id=None):
        seen.update(path=path, project_id=project_id)
        raise IngestError("INVALID_URL", "stop")

    monkeypatch.setattr(process, "ingest_file", fake_ingest_file)
    deps = Recorder(settings).deps()
    deps.ingest = process.default_ingest
    run_process_job(engine, jid, deps)
    assert seen == {"path": settings.data_dir / "uploads" / pid / "rekaman.mp4", "project_id": pid}


def test_default_ingest_youtube_uses_project_id(engine, settings, monkeypatch):
    pid, _ = _setup(engine)
    seen = {}
    monkeypatch.setattr(process, "ingest_youtube", lambda url, s, project_id=None: seen.update(
        url=url, project_id=project_id))
    with Session(engine) as s:
        process.default_ingest(s.get(Project, pid), settings)
    assert seen == {"url": "https://youtu.be/dQw4w9WgXcQ", "project_id": pid}


def test_picks_roundtrip(tmp_path):
    picks = [_pick(0, 40), _pick(50, 95)]
    save_picks(tmp_path / "picks.json", picks)
    raw = json.loads((tmp_path / "picks.json").read_text(encoding="utf-8"))
    assert set(raw[0]) == {"candidate", "materials"}
    assert load_picks(tmp_path / "picks.json") == picks


def test_default_prepare_delegates_to_prepare_stage(monkeypatch):
    from app.worker import prepare

    seen = []
    monkeypatch.setattr(prepare, "prepare_project_clips",
                        lambda engine, pid, picks, **kw: seen.append((engine, pid, picks, kw.get("settings"))))
    ProcessDeps().prepare("eng", "x", [])
    custom = Settings(_env_file=None, data_dir=Path("custom-dir"))
    process.default_prepare("eng", "y", [], settings=custom)
    assert seen == [("eng", "x", [], None), ("eng", "y", [], custom)]


def test_run_process_job_forwards_custom_settings_to_default_prepare(engine, settings, monkeypatch):
    from app.worker import prepare

    pid, jid = _setup(engine, moment_source="claude_manual", last_completed_stage="moments")
    d = settings.data_dir / "projects" / pid
    d.mkdir(parents=True)
    save_picks(d / "picks.json", [_pick(0, 40)])
    seen = {}

    def fake_prepare(eng, project_id, picks, *, settings=None):
        seen.update(engine=eng, project_id=project_id, count=len(picks), settings=settings)

    monkeypatch.setattr(prepare, "prepare_project_clips", fake_prepare)
    assert run_process_job(engine, jid, ProcessDeps(settings=settings)) == "done"
    assert seen == {"engine": engine, "project_id": pid, "count": 1, "settings": settings}
