"""Ranged projects: audio, silences and transcript per range file, merged on the original timeline."""
import json
from pathlib import Path

import pytest
from sqlmodel import Session

from app.core.config import Settings
from app.core.models import Transcript, VideoMeta, Word
from app.db.engine import get_engine, init_db
from app.db.tables import Project
from app.services.settings_store import put_app_settings
from app.worker import process
from app.worker.process import run_process_job
from tests.worker.test_process import Recorder, _rows, _setup

RANGES = [{"index": 0, "start": 1800.0, "end": 1830.0, "file": "source_r0.mp4"},
          {"index": 1, "start": 3600.0, "end": 3640.0, "file": "source_r1.mp4"}]


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path)


@pytest.fixture
def engine(settings):
    eng = get_engine(settings)
    init_db(eng)
    return eng


class RangeRecorder(Recorder):
    """Recorder whose ingest yields two range files and whose mocks record their arguments."""

    def __init__(self, settings, engine, ids):
        super().__init__(settings)
        self.engine, self.ids = engine, ids
        self.audio_args: list[tuple[str, str]] = []
        self.silence_wavs: list[str] = []
        self.transcribe_args: list[tuple[str, float, Path]] = []
        self.progress_seen: list[float] = []

    def ingest(self, project, settings):
        self._hit("ingest")
        d = Path(settings.data_dir) / "projects" / project.id
        d.mkdir(parents=True, exist_ok=True)
        for r in RANGES:
            (d / r["file"]).write_bytes(b"v")
        return d / "source_r0.mp4", VideoMeta(source_type="youtube", project_id=project.id, title="Judul",
                                              duration=7200.0, video_id="dQw4w9WgXcQ", ranges=RANGES)

    def extract_audio(self, src, wav):
        self.audio_args.append((src.name, wav.name))
        super().extract_audio(src, wav)

    def detect_silences(self, wav):
        self._hit("detect_silences")
        self.silence_wavs.append(wav.name)
        return [(1.0, 1.5)] if wav.name == "audio_r0.wav" else [(0.25, 2.0), (5.0, 6.0)]

    def transcribe(self, wav, duration, tdir, model, corrections, on_progress=None):
        on_progress(0.5)
        self._hit("transcribe")
        self.progress_seen.append(_rows(self.engine, *self.ids)[1].progress)
        self.transcribe_args.append((wav.name, duration, tdir))
        text = "satu dua tiga" if wav.name == "audio_r0.wav" else "empat lima"
        words = [Word(w=w, start=round(0.5 + i * 0.3, 3), end=round(0.75 + i * 0.3, 3))
                 for i, w in enumerate(text.split())]
        t = Transcript(language="id", duration=duration, words=words, sentences=[])
        (tdir / "transcript.json").write_text(t.model_dump_json(), encoding="utf-8")
        return t


def test_process_ranges_creates_audio_and_transcript_per_range(engine, settings):
    pid, jid = _setup(engine, moment_source="claude_manual", ranges=[[1800, 1830], [3600, 3640]],
                      whisper_model="small")
    rec = RangeRecorder(settings, engine, (pid, jid))
    assert run_process_job(engine, jid, rec.deps()) == "paused"
    d = settings.data_dir / "projects" / pid
    tdir = d / "transcripts" / "small"
    assert rec.audio_args == [("source_r0.mp4", "audio_r0.wav"), ("source_r1.mp4", "audio_r1.wav")]
    assert rec.silence_wavs == ["audio_r0.wav", "audio_r1.wav"]
    assert rec.transcribe_args == [("audio_r0.wav", 30.0, tdir / "r0"), ("audio_r1.wav", 40.0, tdir / "r1")]
    # overall progress weighted by range length: 0.5 of 30 s, then 30 s + 0.5 of 40 s, out of 70 s
    assert rec.progress_seen == [pytest.approx(15 / 70, abs=1e-4), pytest.approx(50 / 70, abs=1e-4)]

    silences = json.loads((d / "silences.json").read_text(encoding="utf-8"))
    assert silences == [[1801.0, 1801.5], [3600.25, 3602.0], [3605.0, 3606.0]]

    t = Transcript.model_validate_json((tdir / "transcript.json").read_text(encoding="utf-8"))
    assert [w.w for w in t.words] == ["satu", "dua", "tiga", "empat", "lima"]
    assert [w.start for w in t.words] == [1800.5, 1800.8, 1801.1, 3600.5, 3600.8]
    assert [(s.i, s.word_range) for s in t.sentences] == [(0, (0, 3)), (1, (3, 5))]
    assert t.duration == 7200.0
    assert _rows(engine, pid, jid)[0].last_completed_stage == "signals"


def test_process_legacy_single_source_unchanged(engine, settings):
    pid, jid = _setup(engine, moment_source="claude_manual", whisper_model="small")
    rec = Recorder(settings)
    seen = {}

    def extract_audio(src, wav):
        seen["audio"] = (src.name, wav.name)
        Recorder.extract_audio(rec, src, wav)

    def transcribe(wav, duration, tdir, model, corrections, on_progress=None):
        seen["transcribe"] = (wav.name, duration, tdir)
        return Recorder.transcribe(rec, wav, duration, tdir, model, corrections, on_progress=on_progress)

    deps = rec.deps()
    deps.extract_audio, deps.transcribe = extract_audio, transcribe
    assert run_process_job(engine, jid, deps) == "paused"
    d = settings.data_dir / "projects" / pid
    tdir = d / "transcripts" / "small"
    assert seen == {"audio": ("source.mp4", "audio.wav"), "transcribe": ("audio.wav", 120.0, tdir)}
    assert json.loads((d / "silences.json").read_text(encoding="utf-8")) == [[1.0, 1.5]]
    saved = Transcript.model_validate_json((tdir / "transcript.json").read_text(encoding="utf-8"))
    assert saved == rec.transcript
    assert not (tdir / "r0").exists() and not (d / "audio_r0.wav").exists()


def test_default_ingest_passes_ranges_and_cookies_browser(engine, settings, monkeypatch):
    pid, _ = _setup(engine, ranges=[[1800.0, 1830.0]])
    with Session(engine) as s:
        put_app_settings(s, {"cookies_browser": "firefox", "encode_threads": 2, "range_max_height": 480},
                         settings=settings)
    seen = {}

    def fake_ingest_youtube(url, s, project_id=None, ranges=None, cookies_browser="",
                            encode_threads=None, range_max_height=None):
        seen.update(url=url, project_id=project_id, ranges=ranges, cookies_browser=cookies_browser,
                    encode_threads=encode_threads, range_max_height=range_max_height)

    monkeypatch.setattr(process, "ingest_youtube", fake_ingest_youtube)
    with Session(engine) as s:
        process.default_ingest(s.get(Project, pid), settings)
    assert seen == {
        "url": "https://youtu.be/dQw4w9WgXcQ", "project_id": pid, "ranges": [[1800.0, 1830.0]],
        "cookies_browser": "firefox", "encode_threads": 2, "range_max_height": 480,
    }
