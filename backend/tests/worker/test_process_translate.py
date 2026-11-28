"""Tests for worker `translate` stage and Ollama moments analysis from translated transcripts."""
import logging
from pathlib import Path

import pytest
from sqlmodel import Session

from app.core.config import Settings
from app.core.errors import AppError
from app.core.models import Transcript, VideoMeta, Word
from app.db.engine import get_engine, init_db
from app.domain.translation import Translation, load_translation, save_translation, translation_path
from app.providers.llm import LLMUnavailable
from app.worker import process, queue
from app.worker.process import ProcessDeps, run_process_job
from tests.helpers import make_transcript, sents
from tests.worker.test_process import Recorder, _rows, _setup


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path)


@pytest.fixture
def engine(settings):
    eng = get_engine(settings)
    init_db(eng)
    return eng


def _foreign_transcript(n: int = 4, lang: str = "en") -> Transcript:
    t = make_transcript(n, each=6.0)
    # Give sentences readable foreign text while keeping valid word_ranges/timings
    custom = sents([f"english sentence number {i}" for i in range(n)], each=6.0)
    words = []
    for i, s in enumerate(custom):
        toks = s.text.split()
        step = 6.0 / len(toks)
        a = len(words)
        for j, tok in enumerate(toks):
            st = i * 6.0 + j * step
            words.append(Word(w=tok, start=round(st, 3), end=round(st + step * 0.8, 3)))
        custom[i] = s.model_copy(update={"start": words[a].start, "end": words[-1].end, "word_range": (a, len(words))})
    return t.model_copy(update={"language": lang, "words": words, "sentences": custom})


def test_legacy_project_skips_translate(engine, settings):
    pid, jid = _setup(engine, moment_source="ollama", target_language="id")
    rec = Recorder(settings)
    rec.transcript = make_transcript(10)  # language="id"
    seen_t: list[Transcript] = []

    def fake_translate(*args, **kwargs):
        raise AssertionError("translate must not be called for target-language transcript")

    def fake_analyze(t, meta, max_clips, min_sec, max_sec, ck, s):
        rec._hit("analyze")
        seen_t.append(t)
        return rec.picks

    deps = rec.deps()
    deps.translate = fake_translate
    deps.analyze = fake_analyze

    assert run_process_job(engine, jid, deps) == "done"
    assert rec.calls == ["ingest", "extract_audio", "detect_silences", "transcribe", "signals", "analyze", "prepare"]
    tdir = settings.data_dir / "projects" / pid / "transcripts" / "small"
    assert not translation_path(tdir).exists()
    assert seen_t == [rec.transcript]


def test_translate_stage_runs_for_foreign_transcript_with_ollama(engine, settings):
    pid, jid = _setup(engine, moment_source="ollama", target_language="id")
    rec = Recorder(settings)
    rec.transcript = _foreign_transcript(4, lang="en")
    seen_calls: list[tuple] = []
    seen_progress: list[tuple[str | None, float]] = []

    def fake_translate(sentences, source, target, checkpoint_dir, s, on_progress=None):
        rec._hit("translate")
        seen_calls.append(([x.i for x in sentences], source, target, checkpoint_dir))
        if on_progress is not None:
            on_progress(0.5)
            seen_progress.append((_rows(engine, pid, jid)[1].stage, _rows(engine, pid, jid)[1].progress))
            on_progress(1.0)
            seen_progress.append((_rows(engine, pid, jid)[1].stage, _rows(engine, pid, jid)[1].progress))
        return {x.i: f"kalimat indonesia nomor {x.i}" for x in sentences}, []

    deps = rec.deps()
    deps.translate = fake_translate

    assert run_process_job(engine, jid, deps) == "done"
    assert rec.calls == [
        "ingest", "extract_audio", "detect_silences", "transcribe", "translate", "signals", "analyze", "prepare"
    ]
    tdir = settings.data_dir / "projects" / pid / "transcripts" / "small"
    assert seen_calls == [([0, 1, 2, 3], "en", "id", tdir / "checkpoints")]
    assert seen_progress == [("translate", 0.5), ("translate", 1.0)]

    saved = load_translation(tdir, "id", "en")
    assert saved.lines == {
        "0": "kalimat indonesia nomor 0",
        "1": "kalimat indonesia nomor 1",
        "2": "kalimat indonesia nomor 2",
        "3": "kalimat indonesia nomor 3",
    }
    p, j = _rows(engine, pid, jid)
    assert p.status == "ready" and p.last_completed_stage == "prepare"
    assert j.status == "succeeded"


def test_translate_stage_skipped_for_claude_manual_source(engine, settings):
    pid, jid = _setup(engine, moment_source="claude_manual", target_language="id")
    rec = Recorder(settings)
    rec.transcript = _foreign_transcript(4, lang="en")

    deps = rec.deps()
    deps.translate = lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("translate must not run for claude_manual")
    )

    assert run_process_job(engine, jid, deps) == "paused"
    assert "translate" not in rec.calls
    tdir = settings.data_dir / "projects" / pid / "transcripts" / "small"
    assert not translation_path(tdir).exists()
    p, j = _rows(engine, pid, jid)
    assert p.status == "awaiting_moments" and p.last_completed_stage == "signals"
    assert j.status == "succeeded"


def test_translate_stage_resumes_only_missing_sentences(engine, settings):
    pid, jid = _setup(
        engine,
        moment_source="ollama",
        target_language="id",
        last_completed_stage="transcribe",
        whisper_model="small",
    )
    rec = Recorder(settings)
    rec.transcript = _foreign_transcript(5, lang="en")
    d = settings.data_dir / "projects" / pid
    tdir = d / "transcripts" / "small"
    tdir.mkdir(parents=True)
    (tdir / "transcript.json").write_text(rec.transcript.model_dump_json(), encoding="utf-8")
    (d / "meta.json").write_text(
        VideoMeta(source_type="youtube", project_id=pid, title="Judul", duration=30.0).model_dump_json(),
        encoding="utf-8",
    )
    save_translation(
        tdir,
        Translation(language="id", source_language="en", lines={"0": "nol", "1": "satu", "3": "tiga"}),
    )

    asked_indices: list[list[int]] = []

    def fake_translate(sentences, source, target, checkpoint_dir, s, on_progress=None):
        asked_indices.append([x.i for x in sentences])
        return {x.i: f"terjemahan {x.i}" for x in sentences}, []

    deps = rec.deps()
    deps.translate = fake_translate

    assert run_process_job(engine, jid, deps) == "done"
    assert asked_indices == [[2, 4]]
    saved = load_translation(tdir, "id", "en")
    assert saved.lines == {
        "0": "nol",
        "1": "satu",
        "2": "terjemahan 2",
        "3": "tiga",
        "4": "terjemahan 4",
    }


def test_ollama_unavailable_in_translate_is_retryable_app_error(engine, settings):
    # First attempt: retryable AppError requeues the job and keeps project processing
    pid, jid = _setup(engine, moment_source="ollama", target_language="id")
    rec = Recorder(settings)
    rec.transcript = _foreign_transcript(3, lang="en")

    def down_translate(*args, **kwargs):
        raise LLMUnavailable("Ollama tidak berjalan")

    deps = rec.deps()
    deps.translate = down_translate
    assert run_process_job(engine, jid, deps) == "done"
    p, j = _rows(engine, pid, jid)
    assert j.status == "queued"
    assert p.status == "processing"
    assert p.last_completed_stage == "transcribe"

    # Final attempt (attempts == MAX_ATTEMPTS): job fails with OLLAMA_UNAVAILABLE in Indonesian
    with Session(engine) as s:
        job = s.get(type(j), jid)
        job.status = "running"
        job.attempts = queue.MAX_ATTEMPTS
        s.add(job)
        s.commit()

    assert run_process_job(engine, jid, deps) == "done"
    p2, j2 = _rows(engine, pid, jid)
    assert j2.status == "failed"
    assert j2.error_code == "OLLAMA_UNAVAILABLE"
    assert j2.error_message is not None
    assert "claude_manual" in j2.error_message and "Ollama" in j2.error_message
    assert p2.status == "failed"


def test_analyze_receives_translated_view_for_foreign_transcript(engine, settings):
    pid, jid = _setup(engine, moment_source="ollama", target_language="id")
    rec = Recorder(settings)
    original = _foreign_transcript(4, lang="en")
    rec.transcript = original
    captured: list[Transcript] = []

    def fake_translate(sentences, source, target, checkpoint_dir, s, on_progress=None):
        return {x.i: f"arti kalimat {x.i}" for x in sentences}, []

    def fake_analyze(t, meta, max_clips, min_sec, max_sec, ck, s):
        rec._hit("analyze")
        captured.append(t)
        return rec.picks

    deps = rec.deps()
    deps.translate = fake_translate
    deps.analyze = fake_analyze

    assert run_process_job(engine, jid, deps) == "done"
    assert len(captured) == 1
    analyzed_t = captured[0]
    assert analyzed_t.language == "id"
    assert analyzed_t.duration == original.duration
    assert [s.i for s in analyzed_t.sentences] == [s.i for s in original.sentences]
    assert [(s.start, s.end) for s in analyzed_t.sentences] == [(s.start, s.end) for s in original.sentences]
    assert [s.text for s in analyzed_t.sentences] == [f"arti kalimat {i}" for i in range(4)]

    # Original transcript.json on disk stays in source language
    tdir = settings.data_dir / "projects" / pid / "transcripts" / "small"
    disk_t = Transcript.model_validate_json((tdir / "transcript.json").read_text(encoding="utf-8"))
    assert disk_t.language == "en"
    assert [s.text for s in disk_t.sentences] == [s.text for s in original.sentences]


def test_s2_project_at_signals_stage_resumes_without_translate(engine, settings):
    pid, jid = _setup(
        engine,
        moment_source="ollama",
        last_completed_stage="signals",
        whisper_model="small",
        target_language="id",
    )
    rec = Recorder(settings)
    d = settings.data_dir / "projects" / pid
    tdir = d / "transcripts" / "small"
    tdir.mkdir(parents=True)
    (tdir / "transcript.json").write_text(rec.transcript.model_dump_json(), encoding="utf-8")
    (d / "meta.json").write_text(
        VideoMeta(source_type="youtube", project_id=pid, title="S2", duration=120.0).model_dump_json(),
        encoding="utf-8",
    )
    (d / "signals.json").write_text('{"chat": [], "audio": []}', encoding="utf-8")

    deps = rec.deps()
    deps.translate = lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("translate must not run when resuming after signals")
    )

    assert run_process_job(engine, jid, deps) == "done"
    assert rec.calls == ["analyze", "prepare"]
    p, j = _rows(engine, pid, jid)
    assert p.status == "ready" and p.last_completed_stage == "prepare"


def test_translate_stage_saves_partial_results_and_warns_on_failed_sentences(engine, settings, caplog):
    pid, jid = _setup(engine, moment_source="ollama", target_language="id")
    rec = Recorder(settings)
    rec.transcript = _foreign_transcript(3, lang="en")
    captured: list[Transcript] = []

    def partial_translate(sentences, source, target, checkpoint_dir, s, on_progress=None):
        return {0: "kalimat nol", 2: "kalimat dua"}, [1]

    def fake_analyze(t, meta, max_clips, min_sec, max_sec, ck, s):
        captured.append(t)
        return rec.picks

    deps = rec.deps()
    deps.translate = partial_translate
    deps.analyze = fake_analyze

    with caplog.at_level(logging.WARNING, logger="kliplah.process"):
        assert run_process_job(engine, jid, deps) == "done"

    tdir = settings.data_dir / "projects" / pid / "transcripts" / "small"
    saved = load_translation(tdir, "id", "en")
    assert saved.lines == {"0": "kalimat nol", "2": "kalimat dua"}
    assert [s.text for s in captured[0].sentences] == [
        "kalimat nol",
        "english sentence number 1",
        "kalimat dua",
    ]
    assert any("1" in r.message for r in caplog.records)


def test_default_translate_uses_ollama_client_and_translate_sentences(tmp_path, monkeypatch):
    seen: dict = {}

    class FakeOllama:
        def __init__(self, url, model):
            seen["ollama"] = (url, model)

    def fake_translate_sentences(sentences, source, target, llm, checkpoint_dir, *, on_progress=None):
        seen["call"] = (sentences, source, target, llm, checkpoint_dir, on_progress)
        return {0: "halo"}, []

    monkeypatch.setattr(process, "OllamaClient", FakeOllama)
    monkeypatch.setattr(process, "translate_sentences", fake_translate_sentences)

    s = Settings(_env_file=None, data_dir=tmp_path, ollama_url="http://127.0.0.1:11434", ollama_model="qwen3:4b")
    cb = lambda p: None  # noqa: E731
    sents_in = _foreign_transcript(1, lang="en").sentences
    ck = tmp_path / "ck"
    out = process.default_translate(sents_in, "en", "id", ck, s, on_progress=cb)

    assert out == ({0: "halo"}, [])
    assert seen["ollama"] == ("http://127.0.0.1:11434", "qwen3:4b")
    assert seen["call"][:3] == (sents_in, "en", "id")
    assert isinstance(seen["call"][3], FakeOllama)
    assert seen["call"][4:] == (ck, cb)


def test_translate_stage_skips_llm_when_all_sentences_already_translated(engine, settings, monkeypatch):
    pid, jid = _setup(
        engine,
        moment_source="ollama",
        target_language="id",
        last_completed_stage="transcribe",
        whisper_model="small",
    )
    rec = Recorder(settings)
    rec.transcript = _foreign_transcript(3, lang="en")
    d = settings.data_dir / "projects" / pid
    tdir = d / "transcripts" / "small"
    tdir.mkdir(parents=True)
    (tdir / "transcript.json").write_text(rec.transcript.model_dump_json(), encoding="utf-8")
    (d / "meta.json").write_text(
        VideoMeta(source_type="youtube", project_id=pid, title="Judul", duration=18.0).model_dump_json(),
        encoding="utf-8",
    )
    save_translation(
        tdir,
        Translation(language="id", source_language="en", lines={"0": "nol", "1": "satu", "2": "dua"}),
    )

    progress_events: list[tuple[str, float]] = []
    orig_set_progress = queue.set_progress

    def spy_progress(session, job_id, stage, progress, cursor=None):
        progress_events.append((stage, progress))
        return orig_set_progress(session, job_id, stage, progress, cursor=cursor)

    monkeypatch.setattr(queue, "set_progress", spy_progress)

    deps = rec.deps()
    deps.translate = lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("translate should not run when all lines exist")
    )

    assert run_process_job(engine, jid, deps) == "done"
    assert ("translate", 1.0) in progress_events


def test_legacy_regional_target_language_code_skips_translate(engine, settings):
    pid, jid = _setup(engine, moment_source="ollama", target_language="id")
    rec = Recorder(settings)
    rec.transcript = make_transcript(5).model_copy(update={"language": "id-ID"})

    deps = rec.deps()
    deps.translate = lambda *a, **kw: (_ for _ in ()).throw(
        AssertionError("translate must not run for id-ID when target is id")
    )

    assert run_process_job(engine, jid, deps) == "done"
    tdir = settings.data_dir / "projects" / pid / "transcripts" / "small"
    assert not translation_path(tdir).exists()
