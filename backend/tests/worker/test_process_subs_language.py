"""Tests for S3 non-target YouTube subtitle plan integration in worker process stages."""
import json

import pytest

from app.core.config import Settings
from app.core.models import Transcript, VideoMeta
from app.db.engine import get_engine, init_db
from app.worker.process import ProcessDeps, run_process_job
from tests.helpers import make_transcript
from tests.worker.test_process import Recorder, _rows, _setup

SUBS_MANUAL_SAMPLE = {
    "events": [
        {"tStartMs": 1000, "dDurationMs": 2000, "segs": [{"utf8": "hello world"}]},
        {"tStartMs": 4500, "dDurationMs": 1500, "segs": [{"utf8": "how are you"}]},
    ]
}


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path)


@pytest.fixture
def engine(settings):
    eng = get_engine(settings)
    init_db(eng)
    return eng


def test_process_downloads_non_target_subtitle_and_labels_language(engine, settings):
    pid, jid = _setup(engine, moment_source="claude_manual", transcript_source="auto",
                      source_language=None, target_language="id")
    d = settings.data_dir / "projects" / pid
    rec = Recorder(settings)
    sub_calls: list[tuple] = []

    def fake_ingest(project, s):
        rec._hit("ingest")
        d.mkdir(parents=True, exist_ok=True)
        (d / "source.mp4").write_bytes(b"v")
        return d / "source.mp4", VideoMeta(
            source_type="youtube", project_id=project.id, title="English Talk", channel="Ch",
            duration=20.0, video_id="dQw4w9WgXcQ", subtitles={"manual": ["en"], "auto": []},
        )

    def fake_download_subs(vid, lang, kind, out_dir, s):
        rec._hit("download_subs")
        sub_calls.append((vid, lang, kind, out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"{vid}.{lang}.json3"
        dest.write_text(json.dumps(SUBS_MANUAL_SAMPLE), encoding="utf-8")
        return dest

    deps = rec.deps()
    deps.ingest = fake_ingest
    deps.download_subs = fake_download_subs
    deps.transcribe = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Whisper must not be called"))

    assert run_process_job(engine, jid, deps) == "paused"
    assert sub_calls == [("dQw4w9WgXcQ", "en", "manual", d / "subs")]

    p, _ = _rows(engine, pid, jid)
    assert p.transcript_source == "youtube_manual"
    assert p.source_language == "en"

    meta = VideoMeta.model_validate_json((d / "meta.json").read_text(encoding="utf-8"))
    assert meta.transcript_source == "youtube_manual"
    assert meta.source_language == "en"

    tpath = d / "transcripts" / "youtube-manual-en" / "transcript.json"
    assert tpath.exists()
    t = Transcript.model_validate_json(tpath.read_text(encoding="utf-8"))
    assert t.language == "en"
    assert [s.text for s in t.sentences] == ["hello world", "how are you"]


def test_s2_project_with_target_subs_unchanged(engine, settings):
    pid, jid = _setup(engine, moment_source="claude_manual", transcript_source="auto",
                      source_language=None, target_language="id")
    d = settings.data_dir / "projects" / pid
    rec = Recorder(settings)
    sub_calls: list[tuple] = []

    def fake_ingest(project, s):
        rec._hit("ingest")
        d.mkdir(parents=True, exist_ok=True)
        (d / "source.mp4").write_bytes(b"v")
        return d / "source.mp4", VideoMeta(
            source_type="youtube", project_id=project.id, title="Video ID", channel="Ch",
            duration=20.0, video_id="dQw4w9WgXcQ", subtitles={"manual": ["id"], "auto": ["id"]},
        )

    def fake_download_subs(vid, lang, kind, out_dir, s):
        rec._hit("download_subs")
        sub_calls.append((vid, lang, kind, out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"{vid}.{lang}.json3"
        dest.write_text(json.dumps(SUBS_MANUAL_SAMPLE), encoding="utf-8")
        return dest

    deps = rec.deps()
    deps.ingest = fake_ingest
    deps.download_subs = fake_download_subs
    deps.transcribe = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Whisper must not be called"))

    assert run_process_job(engine, jid, deps) == "paused"
    assert sub_calls == [("dQw4w9WgXcQ", "id", "manual", d / "subs")]

    p, _ = _rows(engine, pid, jid)
    assert p.transcript_source == "youtube_manual"
    assert p.source_language == "id"
    tpath = d / "transcripts" / "youtube-manual-id" / "transcript.json"
    assert tpath.exists()
    t = Transcript.model_validate_json(tpath.read_text(encoding="utf-8"))
    assert t.language == "id"


def test_process_explicit_youtube_auto_downloads_spoken_language(engine, settings):
    pid, jid = _setup(engine, moment_source="claude_manual", transcript_source="youtube_auto",
                      source_language=None, target_language="id")
    d = settings.data_dir / "projects" / pid
    rec = Recorder(settings)
    sub_calls: list[tuple] = []

    meta_ko = VideoMeta(
        source_type="youtube", project_id=pid, title="Korean Stream", channel="Ch",
        duration=20.0, video_id="dQw4w9WgXcQ", subtitles={"manual": [], "auto": ["ko"]},
    )

    def fake_ingest(project, s):
        rec._hit("ingest")
        d.mkdir(parents=True, exist_ok=True)
        (d / "source.mp4").write_bytes(b"v")
        return d / "source.mp4", meta_ko

    def fake_download_subs(vid, lang, kind, out_dir, s):
        rec._hit("download_subs")
        sub_calls.append((vid, lang, kind, out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"{vid}.{lang}.json3"
        dest.write_text(json.dumps(SUBS_MANUAL_SAMPLE), encoding="utf-8")
        return dest

    deps = ProcessDeps(
        settings=settings,
        probe=lambda proj, s: meta_ko,
        ingest=fake_ingest,
        extract_audio=rec.extract_audio,
        detect_silences=rec.detect_silences,
        download_subs=fake_download_subs,
        signals=rec.signals,
        transcribe=lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Whisper must not be called")),
    )

    assert run_process_job(engine, jid, deps) == "paused"
    assert sub_calls == [("dQw4w9WgXcQ", "ko", "auto", d / "subs")]

    p, _ = _rows(engine, pid, jid)
    assert p.transcript_source == "youtube_auto"
    assert p.source_language == "ko"
    tpath = d / "transcripts" / "youtube-auto-ko" / "transcript.json"
    assert tpath.exists()
    assert Transcript.model_validate_json(tpath.read_text(encoding="utf-8")).language == "ko"


def test_process_auto_fallback_to_whisper_preserves_explicit_source_language(engine, settings):
    pid, jid = _setup(engine, moment_source="claude_manual", transcript_source="auto",
                      source_language="ko", target_language="id")
    d = settings.data_dir / "projects" / pid
    rec = Recorder(settings)
    seen_lang: list[str | None] = []

    def fake_ingest(project, s):
        rec._hit("ingest")
        d.mkdir(parents=True, exist_ok=True)
        (d / "source.mp4").write_bytes(b"v")
        return d / "source.mp4", VideoMeta(
            source_type="youtube", project_id=project.id, title="No Manual", channel="Ch",
            duration=20.0, video_id="dQw4w9WgXcQ", subtitles={"manual": [], "auto": ["ko"]},
        )

    def fake_transcribe(wav, duration, tdir, model, corrections, on_progress=None,
                        language=None, target_language="id"):
        seen_lang.append(language)
        t = make_transcript(10).model_copy(update={"language": language or "id"})
        (tdir / "transcript.json").write_text(t.model_dump_json(), encoding="utf-8")
        return t

    deps = rec.deps()
    deps.ingest = fake_ingest
    deps.transcribe = fake_transcribe

    assert run_process_job(engine, jid, deps) == "paused"
    assert seen_lang == ["ko"]
    p, _ = _rows(engine, pid, jid)
    assert p.transcript_source == "whisper"
    assert p.source_language == "ko"


def test_process_auto_prefers_requested_source_language_in_manual_subs(engine, settings):
    pid, jid = _setup(engine, moment_source="claude_manual", transcript_source="auto",
                      source_language="ja", target_language="id")
    d = settings.data_dir / "projects" / pid
    rec = Recorder(settings)
    sub_calls: list[tuple] = []

    def fake_ingest(project, s):
        rec._hit("ingest")
        d.mkdir(parents=True, exist_ok=True)
        (d / "source.mp4").write_bytes(b"v")
        return d / "source.mp4", VideoMeta(
            source_type="youtube", project_id=project.id, title="JP Video", channel="Ch",
            duration=20.0, video_id="dQw4w9WgXcQ", subtitles={"manual": ["en", "ja"], "auto": []},
        )

    def fake_download_subs(vid, lang, kind, out_dir, s):
        rec._hit("download_subs")
        sub_calls.append((vid, lang, kind, out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"{vid}.{lang}.json3"
        dest.write_text(json.dumps(SUBS_MANUAL_SAMPLE), encoding="utf-8")
        return dest

    deps = rec.deps()
    deps.ingest = fake_ingest
    deps.download_subs = fake_download_subs
    deps.transcribe = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Whisper must not be called"))

    assert run_process_job(engine, jid, deps) == "paused"
    assert sub_calls == [("dQw4w9WgXcQ", "ja", "manual", d / "subs")]
    p, _ = _rows(engine, pid, jid)
    assert p.source_language == "ja"
    tpath = d / "transcripts" / "youtube-manual-ja" / "transcript.json"
    assert tpath.exists()
    assert Transcript.model_validate_json(tpath.read_text(encoding="utf-8")).language == "ja"
