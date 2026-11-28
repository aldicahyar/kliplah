"""Tests for source language detection/propagation in worker transcribe stage and transcript_dir."""
import pytest

from app.core.config import Settings
from app.core.models import Transcript, VideoMeta, Word
from app.db.engine import get_engine, init_db
from app.db.tables import Project
from app.worker import process
from app.worker.process import run_process_job, transcript_dir
from tests.helpers import make_transcript
from tests.worker.test_process import Recorder, _rows, _setup
from tests.worker.test_process_ranges import RangeRecorder


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path)


@pytest.fixture
def engine(settings):
    eng = get_engine(settings)
    init_db(eng)
    return eng


def test_whisper_detection_saves_source_language_on_project_and_meta(engine, settings):
    pid, jid = _setup(engine, moment_source="claude_manual", source_language=None, target_language="id")
    rec = Recorder(settings)
    seen: list[tuple[str | None, str]] = []

    def fake_transcribe(wav, duration, tdir, model, corrections, on_progress=None,
                        language=None, target_language="id"):
        seen.append((language, target_language))
        t = make_transcript(10).model_copy(update={"language": "en"})
        (tdir / "transcript.json").write_text(t.model_dump_json(), encoding="utf-8")
        return t

    deps = rec.deps()
    deps.transcribe = fake_transcribe
    assert run_process_job(engine, jid, deps) == "paused"
    assert seen == [(None, "id")]

    p, _ = _rows(engine, pid, jid)
    assert p.source_language == "en"
    meta = VideoMeta.model_validate_json((settings.data_dir / "projects" / pid / "meta.json").read_text(encoding="utf-8"))
    assert meta.source_language == "en"


def test_later_ranges_reuse_detected_language(engine, settings):
    pid, jid = _setup(engine, moment_source="claude_manual", ranges=[[1800, 1830], [3600, 3640]],
                      whisper_model="small", source_language=None, target_language="id")
    rec = RangeRecorder(settings, engine, (pid, jid))
    seen: list[tuple[str, str | None, str]] = []

    def fake_transcribe(wav, duration, tdir, model, corrections, on_progress=None,
                        language=None, target_language="id"):
        seen.append((wav.name, language, target_language))
        text = "hello world" if wav.name == "audio_r0.wav" else "second range"
        words = [Word(w=w, start=round(0.5 + i * 0.3, 3), end=round(0.75 + i * 0.3, 3))
                 for i, w in enumerate(text.split())]
        t = Transcript(language=language or "en", duration=duration, words=words, sentences=[])
        (tdir / "transcript.json").write_text(t.model_dump_json(), encoding="utf-8")
        return t

    deps = rec.deps()
    deps.transcribe = fake_transcribe
    assert run_process_job(engine, jid, deps) == "paused"

    assert seen == [("audio_r0.wav", None, "id"), ("audio_r1.wav", "en", "id")]
    p, _ = _rows(engine, pid, jid)
    assert p.source_language == "en"

    d = settings.data_dir / "projects" / pid
    meta = VideoMeta.model_validate_json((d / "meta.json").read_text(encoding="utf-8"))
    assert meta.source_language == "en"
    merged = Transcript.model_validate_json((d / "transcripts" / "small" / "transcript.json").read_text(encoding="utf-8"))
    assert merged.language == "en"


def test_explicit_source_language_is_passed_through(engine, settings):
    pid, jid = _setup(engine, moment_source="claude_manual", source_language="ko", target_language="id")
    rec = Recorder(settings)
    seen: list[tuple[str | None, str]] = []

    def fake_transcribe(wav, duration, tdir, model, corrections, on_progress=None,
                        language=None, target_language="id"):
        seen.append((language, target_language))
        t = make_transcript(10).model_copy(update={"language": language or "id"})
        (tdir / "transcript.json").write_text(t.model_dump_json(), encoding="utf-8")
        return t

    deps = rec.deps()
    deps.transcribe = fake_transcribe
    assert run_process_job(engine, jid, deps) == "paused"
    assert seen == [("ko", "id")]

    p, _ = _rows(engine, pid, jid)
    assert p.source_language == "ko"
    meta = VideoMeta.model_validate_json((settings.data_dir / "projects" / pid / "meta.json").read_text(encoding="utf-8"))
    assert meta.source_language == "ko"


def test_default_transcribe_passes_language_and_target_language(tmp_path, monkeypatch):
    seen: dict = {}

    class FakeFW:
        def __init__(self, model_name, language=None, target_language="id"):
            seen["init"] = (model_name, language, target_language)

    def fake_transcribe(wav, duration, tdir, transcriber, model_name, corrections,
                        on_progress=None, language=None):
        seen["transcribe"] = (wav, duration, tdir, model_name, corrections, on_progress, language)
        return make_transcript(5).model_copy(update={"language": "ko"})

    monkeypatch.setattr(process, "FasterWhisperTranscriber", FakeFW)
    monkeypatch.setattr(process, "transcribe", fake_transcribe)

    def cb(p):
        return None
    wav = tmp_path / "audio.wav"
    tdir = tmp_path / "tdir"
    out = process.default_transcribe(wav, 12.0, tdir, "small", [("a", "b")],
                                     on_progress=cb, language="ko", target_language="en")
    assert out.language == "ko"
    assert seen["init"] == ("small", "ko", "en")
    assert seen["transcribe"] == (wav, 12.0, tdir, "small", [("a", "b")], cb, "ko")


def test_transcript_dir_uses_source_language_for_youtube_sources(tmp_path):
    s = Settings(_env_file=None, data_dir=tmp_path)
    base = tmp_path / "projects" / "p1" / "transcripts"
    assert transcript_dir(
        s, Project(id="p1", transcript_source="youtube_manual", source_language="en", target_language="id")
    ) == base / "youtube-manual-en"
    assert transcript_dir(
        s, Project(id="p1", transcript_source="youtube_auto", source_language="ko", target_language="id")
    ) == base / "youtube-auto-ko"


def test_transcript_dir_s2_project_unchanged(tmp_path):
    s = Settings(_env_file=None, data_dir=tmp_path)
    base = tmp_path / "projects" / "p1" / "transcripts"
    assert transcript_dir(
        s, Project(id="p1", transcript_source="youtube_manual", source_language=None, target_language="id")
    ) == base / "youtube-manual-id"
    assert transcript_dir(
        s, Project(id="p1", transcript_source="youtube_auto", source_language=None, target_language="en")
    ) == base / "youtube-auto-en"
    assert transcript_dir(
        s, Project(id="p1", transcript_source="whisper", source_language="en", whisper_model="small")
    ) == base / "small"
    assert transcript_dir(
        s, Project(id="p1", transcript_source="auto", source_language=None, whisper_model="small")
    ) == base / "small"
