"""Tests for the worker `signals` stage and signal wiring into analyze, finish_moments, and prepare."""
import json
from pathlib import Path

import pytest
from sqlmodel import Session, select

from app.core.config import Settings
from app.core.errors import AppError
from app.core.models import Candidate, Materials, PlatformCaption, ReframeSegment, VideoMeta
from app.db.engine import get_engine, init_db
from app.db.tables import Clip, Project
from app.pipeline.signals import Signals
from app.services import moments
from app.worker import process
from app.worker.prepare import PrepareDeps, build_clips_from_picks
from app.worker.process import STAGES, ProcessDeps, run_process_job
from tests.helpers import make_transcript
from tests.worker.test_process import Recorder, _pick, _rows, _setup


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path)


@pytest.fixture
def engine(settings):
    eng = get_engine(settings)
    init_db(eng)
    return eng


def test_signals_stage_runs_between_transcribe_and_moments(engine, settings):
    assert STAGES == ("ingest", "audio", "transcribe", "signals", "moments", "prepare")
    pid, jid = _setup(engine, moment_source="ollama", content_type="stream", focus_prompt="cari kaget")
    rec = Recorder(settings)
    seen_analyze = {}

    def fake_signals(project, meta, smap, s):
        rec._hit("signals")
        return Signals(chat=[{"start": 10.0, "end": 20.0, "value": 1.0}], audio=[])

    def fake_analyze(t, meta, max_clips, min_sec, max_sec, ck, s, *, signals=None, content_type=None,
                     focus_prompt=None, source_map=None):
        rec._hit("analyze")
        seen_analyze.update(signals=signals, content_type=content_type, focus_prompt=focus_prompt,
                            source_map=source_map)
        return rec.picks

    deps = rec.deps()
    deps.signals = fake_signals
    deps.analyze = fake_analyze
    assert run_process_job(engine, jid, deps) == "done"
    assert rec.calls == ["ingest", "extract_audio", "detect_silences", "transcribe", "signals", "analyze", "prepare"]
    assert seen_analyze["signals"] == Signals(chat=[{"start": 10.0, "end": 20.0, "value": 1.0}], audio=[])
    assert seen_analyze["content_type"] == "stream" and seen_analyze["focus_prompt"] == "cari kaget"
    assert seen_analyze["source_map"] is not None
    p, _ = _rows(engine, pid, jid)
    assert p.signals_available == {"heatmap": True, "live_chat": True, "audio_peak": False}
    assert (settings.data_dir / "projects" / pid / "signals.json").exists()


def test_stream_profile_downloads_chat_podcast_does_not(engine, settings, monkeypatch):
    chat_calls = []
    loud_calls = []

    def fake_dl_chat(vid, out, s, cookies_browser="", until_sec=None):
        chat_calls.append((vid, until_sec))
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text('{"replayChatItemAction":{"videoOffsetTimeMsec":"12000",'
                       '"actions":[{"addChatItemAction":{"item":{"liveChatTextMessageRenderer":'
                       '{"message":{"runs":[{"text":"wkwk"}]}}}}}]}}\n', encoding="utf-8")
        return out

    monkeypatch.setattr(process, "download_live_chat", fake_dl_chat)
    monkeypatch.setattr(process, "audio_loudness", lambda wav: loud_calls.append(wav.name) or [
        (0.5, -24.0), (1.5, -24.0), (2.5, -24.0), (3.5, -10.0), (4.5, -24.0)])

    def make_meta(project):
        return VideoMeta(source_type="youtube", project_id=project.id, title="T", duration=120.0,
                         video_id="dQw4w9WgXcQ", has_live_chat=True,
                         ranges=[{"index": 0, "start": 100.0, "end": 160.0, "file": "source_r0.mp4"}])

    # Stream: downloads chat + computes audio peaks shifted by range start
    pid_s, jid_s = _setup(engine, moment_source="claude_manual", content_type="stream")
    rec_s = Recorder(settings)
    deps_s = rec_s.deps()
    deps_s.signals = process.default_signals
    deps_s.ingest = lambda proj, s: (
        ((d := settings.data_dir / "projects" / proj.id).mkdir(parents=True, exist_ok=True) or True)
        and ((d / "source_r0.mp4").write_bytes(b"v") or True)
        and (d / "source_r0.mp4", make_meta(proj))
    )
    assert run_process_job(engine, jid_s, deps_s) == "paused"
    assert chat_calls == [("dQw4w9WgXcQ", 160.0)] and loud_calls == ["audio_r0.wav"]
    sigs = json.loads((settings.data_dir / "projects" / pid_s / "signals.json").read_text(encoding="utf-8"))
    assert sigs["audio"] and sigs["audio"][0]["start"] == pytest.approx(103.5)
    assert _rows(engine, pid_s, jid_s)[0].signals_available["audio_peak"] is True

    # Podcast: does NOT download chat or run audio_loudness
    chat_calls.clear()
    loud_calls.clear()
    pid_p, jid_p = _setup(engine, moment_source="claude_manual", content_type="podcast")
    deps_p = Recorder(settings).deps()
    deps_p.signals = process.default_signals
    deps_p.ingest = lambda proj, s: (
        ((d := settings.data_dir / "projects" / proj.id).mkdir(parents=True, exist_ok=True) or True)
        and ((d / "source_r0.mp4").write_bytes(b"v") or True)
        and (d / "source_r0.mp4", make_meta(proj))
    )
    assert run_process_job(engine, jid_p, deps_p) == "paused"
    assert chat_calls == [] and loud_calls == []
    assert _rows(engine, pid_p, jid_p)[0].signals_available == {
        "heatmap": False, "live_chat": False, "audio_peak": False
    }


def test_chat_download_failure_does_not_fail_project(engine, settings, monkeypatch):
    def boom(*a, **kw):
        raise AppError("NETWORK", "chat error", retryable=True)

    monkeypatch.setattr(process, "download_live_chat", boom)
    monkeypatch.setattr(process, "audio_loudness", lambda wav: [(0.5, -24.0), (1.5, -24.0), (2.5, -24.0)])

    pid, jid = _setup(engine, moment_source="ollama", content_type="stream")
    rec = Recorder(settings)
    deps = rec.deps()
    deps.signals = process.default_signals
    deps.ingest = lambda proj, s: (
        ((d := settings.data_dir / "projects" / proj.id).mkdir(parents=True, exist_ok=True) or True)
        and ((d / "source.mp4").write_bytes(b"v") or True)
        and (d / "source.mp4", VideoMeta(source_type="youtube", project_id=proj.id, title="T",
                                         duration=120.0, video_id="dQw4w9WgXcQ", has_live_chat=True))
    )
    assert run_process_job(engine, jid, deps) == "done"
    p, _ = _rows(engine, pid, jid)
    assert p.status == "ready"
    assert p.signals_available["live_chat"] is False


def test_s1_project_at_transcribe_resumes_with_signals(engine, settings):
    # S1 project stopped after transcribe: signals runs before moments; project after moments skips signals
    pid1, jid1 = _setup(engine, moment_source="ollama", last_completed_stage="transcribe", whisper_model="small")
    d1 = settings.data_dir / "projects" / pid1
    (d1 / "transcripts" / "small").mkdir(parents=True)
    (d1 / "transcripts" / "small" / "transcript.json").write_text(make_transcript(20).model_dump_json(), "utf-8")
    (d1 / "meta.json").write_text(VideoMeta(source_type="youtube", project_id=pid1, title="T",
                                            duration=120.0).model_dump_json(), "utf-8")
    rec = Recorder(settings)
    deps = rec.deps()
    deps.signals = lambda p, m, smap, s: (rec._hit("signals") or Signals())
    assert run_process_job(engine, jid1, deps) == "done"
    assert rec.calls == ["signals", "analyze", "prepare"]

    pid2, jid2 = _setup(engine, moment_source="claude_manual", last_completed_stage="moments")
    d2 = settings.data_dir / "projects" / pid2
    d2.mkdir(parents=True)
    process.save_picks(d2 / "picks.json", [_pick()])
    rec2 = Recorder(settings)
    deps2 = rec2.deps()
    deps2.signals = lambda p, m, smap, s: (rec2._hit("signals") or Signals())
    assert run_process_job(engine, jid2, deps2) == "done"
    assert rec2.calls == ["prepare"]


def test_prepare_stores_signal_bonus(engine, settings):
    with Session(engine) as s:
        p = Project(youtube_url="https://youtu.be/dQw4w9WgXcQ", content_type="stream")
        s.add(p)
        s.commit()
        pid = p.id
    d = settings.data_dir / "projects" / pid
    (d / "transcripts" / "small").mkdir(parents=True)
    (d / "source.mp4").write_bytes(b"v")
    t = make_transcript(20)
    (d / "transcripts" / "small" / "transcript.json").write_text(t.model_dump_json(), "utf-8")
    (d / "silences.json").write_text("[]", "utf-8")
    (d / "meta.json").write_text(VideoMeta(source_type="youtube", project_id=pid, title="T",
                                           duration=t.duration).model_dump_json(), "utf-8")
    cap = PlatformCaption(title="j", caption="c", hashtags=["#a"])
    cand = Candidate(start_sentence=0, end_sentence=2, start=0.0, end=18.0, llm_score=70,
                     category="hook", reason="r", hook="h", heatmap_bonus=4, signal_bonus=15, score=89)
    deps = PrepareDeps(settings=settings,
                       reframe=lambda v, a, b: ([ReframeSegment(start=0, end=b - a, mode="track", x=0.5)], "face_crop"),
                       thumbnail=lambda src, tm, out: (out.parent.mkdir(parents=True, exist_ok=True), out.write_bytes(b"j")))
    build_clips_from_picks(engine, pid, [(cand, Materials(hook="h", tiktok=cap, reels=cap, shorts=cap))], deps)
    with Session(engine) as s:
        [clip] = s.exec(select(Clip).where(Clip.project_id == pid)).all()
    assert clip.signal_bonus == 15 and clip.score == 89


def test_finish_moments_passes_signals_to_import_external(engine, settings):
    with Session(engine, expire_on_commit=False) as s:
        p = Project(status="awaiting_moments", last_completed_stage="signals", content_type="stream",
                    whisper_model="small", max_clips=4, min_clip_sec=15, max_clip_sec=40)
        s.add(p)
        s.commit()
        pid = p.id
    d = settings.data_dir / "projects" / pid
    (d / "transcripts" / "small").mkdir(parents=True)
    t = make_transcript(20, each=6.0)
    (d / "transcripts" / "small" / "transcript.json").write_text(t.model_dump_json(), "utf-8")
    (d / "meta.json").write_text(VideoMeta(source_type="youtube", project_id=pid, title="T",
                                           duration=t.duration).model_dump_json(), "utf-8")
    (d / "signals.json").write_text(Signals(chat=[{"start": 0.0, "end": 20.0, "value": 1.0}]).model_dump_json(),
                                      "utf-8")
    reply = json.dumps({"clips": [{"start_sentence": 0, "end_sentence": 4, "score": 70, "category": "hook",
                                   "reason": "r", "hook": "h",
                                   **{k: {"title": "j", "caption": "c", "hashtags": ["#a"]}
                                      for k in ("tiktok", "reels", "shorts")}}]})
    moments.import_part(engine, pid, 1, reply, settings=settings)
    moments.finish_moments(engine, pid, settings=settings)
    picks = process.load_picks(d / "picks.json")
    assert picks[0][0].signal_bonus == 15 and picks[0][0].score == 85


def test_audio_loudness_failure_does_not_fail_project(engine, settings, monkeypatch):
    def boom_loudness(wav):
        raise RuntimeError("ffmpeg gagal membaca wav")

    monkeypatch.setattr(process, "audio_loudness", boom_loudness)
    pid, jid = _setup(engine, moment_source="ollama", content_type="stream")
    rec = Recorder(settings)
    deps = rec.deps()
    deps.signals = process.default_signals
    deps.ingest = lambda proj, s: (
        ((d := settings.data_dir / "projects" / proj.id).mkdir(parents=True, exist_ok=True) or True)
        and ((d / "source.mp4").write_bytes(b"v") or True)
        and (d / "source.mp4", VideoMeta(source_type="youtube", project_id=proj.id, title="T",
                                         duration=120.0, video_id="dQw4w9WgXcQ", has_live_chat=False))
    )
    assert run_process_job(engine, jid, deps) == "done"
    p, _ = _rows(engine, pid, jid)
    assert p.status == "ready"
    assert p.signals_available["audio_peak"] is False


def test_stage_moments_passes_legacy_sourcemap_extended_by_transcript(engine, settings):
    pid, jid = _setup(engine, moment_source="ollama", last_completed_stage="signals", whisper_model="small")
    d = settings.data_dir / "projects" / pid
    (d / "transcripts" / "small").mkdir(parents=True)
    t = make_transcript(20, each=6.0)
    t.sentences[-1] = t.sentences[-1].model_copy(update={"end": 120.8})
    (d / "transcripts" / "small" / "transcript.json").write_text(t.model_dump_json(), "utf-8")
    (d / "meta.json").write_text(VideoMeta(source_type="youtube", project_id=pid, title="T",
                                           duration=120.0).model_dump_json(), "utf-8")
    (d / "signals.json").write_text(Signals().model_dump_json(), "utf-8")
    rec = Recorder(settings)
    seen = {}

    def fake_analyze(tr, meta, max_clips, min_sec, max_sec, ck, s, *, signals=None, content_type=None,
                     focus_prompt=None, source_map=None):
        seen["source_map"] = source_map
        return rec.picks

    deps = rec.deps()
    deps.analyze = fake_analyze
    assert run_process_job(engine, jid, deps) == "done"
    assert seen["source_map"].files[0].end == pytest.approx(120.8)


def test_default_signals_passes_last_range_end_or_none(engine, settings, monkeypatch):
    from app.domain.ranges import RangeFile, SourceMap

    seen = []

    def fake_dl_chat(vid, out, s, cookies_browser="", until_sec=None):
        seen.append(until_sec)
        return None

    monkeypatch.setattr(process, "download_live_chat", fake_dl_chat)
    pid, _ = _setup(engine, content_type="stream")
    with Session(engine) as s:
        proj = s.get(Project, pid)

    pdir = settings.data_dir / "projects" / pid
    ranged_meta = VideoMeta(
        source_type="youtube", project_id=pid, title="R", duration=10800.0, video_id="dQw4w9WgXcQ",
        has_live_chat=True,
        ranges=[{"index": 0, "start": 100.0, "end": 200.0, "file": "source_r0.mp4"},
                {"index": 1, "start": 1800.0, "end": 3600.0, "file": "source_r1.mp4"}],
    )
    process.default_signals(proj, ranged_meta, SourceMap.from_meta(ranged_meta, pdir), settings)
    assert seen == [3600.0]

    seen.clear()
    full_meta = VideoMeta(
        source_type="youtube", project_id=pid, title="F", duration=10800.0, video_id="dQw4w9WgXcQ",
        has_live_chat=True, ranges=[],
    )
    process.default_signals(proj, full_meta, SourceMap.from_meta(full_meta, pdir), settings)
    assert seen == [None]
