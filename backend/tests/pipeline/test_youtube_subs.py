"""Tests for YouTube json3 subtitle downloading, parsing, range clipping, source resolution, and process job."""
import json
from pathlib import Path

import pytest
from sqlmodel import Session

from app.core.config import Settings
from app.core.errors import AppError
from app.core.models import Transcript, VideoMeta, Word
from app.db.engine import get_engine, init_db
from app.db.tables import Project
from app.domain.ranges import RangeFile
from app.pipeline import youtube_subs
from app.pipeline.youtube_subs import (
    SubtitlePlan,
    clip_words_to_ranges,
    distribute_words,
    download_subs,
    parse_json3,
    resolve_subtitle_plan,
    resolve_transcript_source,
)
from app.services.settings_store import put_app_settings
from app.worker import process
from app.worker.process import ProcessDeps, run_process_job, transcript_dir
from tests.worker.test_process import Recorder, _rows, _setup

FIXTURES = Path(__file__).parents[1] / "fixtures"


def _fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_parse_json3_auto_uses_word_offsets():
    words = parse_json3(_fixture("subs_auto.json3"))
    assert [(w.w, w.start, w.end, w.approx) for w in words] == [
        ("selamat", 0.5, 0.9, False),
        ("pagi", 0.9, 1.4, False),
        ("teman", 1.4, 3.5, False),
        ("ayo", 3.5, 4.2, False),
        ("mulai", 4.2, 5.5, False),
    ]


def test_parse_json3_manual_distributes_by_chars():
    words = parse_json3(_fixture("subs_manual.json3"))
    assert [w.w for w in words] == ["halo", "semuanya", "apa", "kabar", "sampai", "jumpa"]
    assert all(w.approx is True for w in words)
    # event 1 (1.0..3.0): "halo" (4) < "semuanya" (8)
    assert words[0].start == 1.0 and words[1].end == 3.0
    assert words[1].end - words[1].start > words[0].end - words[0].start
    assert words[0].end == words[1].start
    # event 2 (4.5..6.0) & event 3 (6.0..8.0) keep total duration and clean newlines
    assert words[2].start == 4.5 and words[3].end == 6.0
    assert words[4].start == 6.0 and words[5].end == 8.0


def test_parse_json3_skips_empty_append_and_tags():
    data = {
        "events": [
            {"tStartMs": 0, "dDurationMs": 1000, "segs": [{"utf8": "<b></b> \n "}]},
            {"tStartMs": 500, "dDurationMs": 1000, "aAppend": 1, "segs": [{"utf8": "skip"}]},
            {"tStartMs": 1000, "dDurationMs": 1000, "segs": [{"utf8": "  {\\an8}<font>satu</font>   dua  "}]},
        ]
    }
    words = parse_json3(data)
    assert [w.w for w in words] == ["satu", "dua"]
    assert words[0].start == 1.0 and words[1].end == 2.0


def test_clip_words_to_ranges_drops_outside_words():
    words = [
        Word(w="sebelum", start=50.0, end=51.0),
        Word(w="dalam1", start=105.0, end=106.0, approx=True),
        Word(w="jeda", start=150.0, end=151.0),
        Word(w="dalam2", start=205.0, end=206.0, approx=False),
        Word(w="sesudah", start=300.0, end=301.0),
    ]
    r0 = RangeFile(0, 100.0, 120.0, "source_r0.mp4")
    r1 = RangeFile(1, 200.0, 220.0, "source_r1.mp4")
    clipped = clip_words_to_ranges(words, [r0, r1])
    assert [(rf.index, [w.w for w in ws]) for rf, ws in clipped] == [
        (0, ["dalam1"]),
        (1, ["dalam2"]),
    ]


def test_clip_words_to_ranges_returns_local_times():
    words = [
        Word(w="awal", start=1800.5, end=1801.25, approx=True),
        Word(w="ujung", start=1829.6, end=1830.4, approx=True),
        Word(w="dua", start=3601.0, end=3601.5, approx=False),
    ]
    r0 = RangeFile(0, 1800.0, 1830.0, "source_r0.mp4")
    r1 = RangeFile(1, 3600.0, 3640.0, "source_r1.mp4")
    clipped = clip_words_to_ranges(words, [r0, r1])
    assert [(w.w, w.start, w.end, w.approx) for w in clipped[0][1]] == [
        ("awal", 0.5, 1.25, True),
        ("ujung", 29.6, 30.0, True),
    ]
    assert [(w.w, w.start, w.end, w.approx) for w in clipped[1][1]] == [
        ("dua", 1.0, 1.5, False),
    ]
    # legacy single full-video range is a no-op
    legacy = RangeFile(0, 0.0, 4000.0, "source.mp4")
    assert clip_words_to_ranges([words[0]], [legacy])[0][1] == [words[0]]


def test_resolve_auto_prefers_manual_target_subs():
    meta = VideoMeta(source_type="youtube", project_id="p1", title="T", duration=100.0,
                     subtitles={"manual": ["en", "id"], "auto": ["id"]})
    assert resolve_transcript_source("auto", meta, "id") == "youtube_manual"


def test_resolve_auto_falls_back_to_whisper():
    meta = VideoMeta(source_type="youtube", project_id="p1", title="T", duration=100.0,
                     subtitles={"manual": [], "auto": ["id"]})
    assert resolve_transcript_source("auto", meta, "id") == "whisper"
    upload = VideoMeta(source_type="upload", project_id="p2", title="T", duration=100.0)
    assert resolve_transcript_source("auto", upload, "id") == "whisper"
    assert resolve_transcript_source("youtube_manual", upload, "id") == "whisper"


def test_explicit_youtube_source_without_target_subs_rejected():
    empty = VideoMeta(source_type="youtube", project_id="p1", title="T", duration=100.0,
                      subtitles={"manual": [], "auto": []})
    with pytest.raises(AppError) as e1:
        resolve_transcript_source("youtube_manual", empty, "id")
    assert e1.value.code == "VALIDATION"
    with pytest.raises(AppError) as e2:
        resolve_transcript_source("youtube_auto", empty, "id")
    assert e2.value.code == "VALIDATION"
    meta = VideoMeta(source_type="youtube", project_id="p1", title="T", duration=100.0,
                     subtitles={"manual": ["en"], "auto": ["en"]})
    assert resolve_transcript_source("youtube_manual", meta, "id") == "youtube_manual"
    assert resolve_transcript_source("youtube_auto", meta, "id") == "youtube_auto"
    assert resolve_transcript_source("youtube_manual", meta, "en") == "youtube_manual"
    assert resolve_transcript_source("youtube_auto", meta, "en") == "youtube_auto"
    assert resolve_transcript_source("whisper", meta, "id") == "whisper"


def test_plan_prefers_target_then_requested_source_then_english():
    meta_all = VideoMeta(source_type="youtube", project_id="p1", title="T", duration=100.0,
                         subtitles={"manual": ["ja", "en", "id"], "auto": []})
    assert resolve_subtitle_plan("auto", meta_all, "id") == SubtitlePlan("youtube_manual", "id", False)

    meta_ja_en = VideoMeta(source_type="youtube", project_id="p1", title="T", duration=100.0,
                           subtitles={"manual": ["ja", "en"], "auto": []})
    assert resolve_subtitle_plan("auto", meta_ja_en, "id") == SubtitlePlan("youtube_manual", "en", True)

    meta_ja_th = VideoMeta(source_type="youtube", project_id="p1", title="T", duration=100.0,
                           subtitles={"manual": ["ja", "th"], "auto": []})
    assert resolve_subtitle_plan("auto", meta_ja_th, "id") == SubtitlePlan("youtube_manual", "ja", True)
    assert resolve_subtitle_plan("auto", meta_ja_th, "id", source_language="th") == SubtitlePlan("youtube_manual", "th", True)
    assert resolve_subtitle_plan("youtube_manual", meta_ja_th, "id", source_language="th") == SubtitlePlan(
        "youtube_manual", "th", True
    )

    # Normalized match works and exact match is preferred when both exist
    meta_region = VideoMeta(source_type="youtube", project_id="p1", title="T", duration=100.0,
                            subtitles={"manual": ["en-US", "ja"], "auto": []})
    assert resolve_subtitle_plan("auto", meta_region, "id") == SubtitlePlan("youtube_manual", "en-US", True)
    assert resolve_subtitle_plan("auto", meta_region, "en") == SubtitlePlan("youtube_manual", "en-US", False)

    meta_exact = VideoMeta(source_type="youtube", project_id="p1", title="T", duration=100.0,
                           subtitles={"manual": ["en-US", "en", "ja"], "auto": []})
    assert resolve_subtitle_plan("auto", meta_exact, "id") == SubtitlePlan("youtube_manual", "en", True)


def test_plan_auto_never_picks_youtube_auto():
    meta = VideoMeta(source_type="youtube", project_id="p1", title="T", duration=100.0,
                     subtitles={"manual": [], "auto": ["id", "en"]})
    assert resolve_subtitle_plan("auto", meta, "id") == SubtitlePlan("whisper", None, None)
    assert resolve_subtitle_plan("auto", meta, "id", source_language="ko") == SubtitlePlan("whisper", "ko", None)


def test_plan_auto_without_manual_subs_uses_whisper():
    meta = VideoMeta(source_type="youtube", project_id="p1", title="T", duration=100.0,
                     subtitles={"manual": [], "auto": []})
    assert resolve_subtitle_plan("auto", meta, "id") == SubtitlePlan("whisper", None, None)
    assert resolve_subtitle_plan("whisper", meta, "id", source_language="EN") == SubtitlePlan("whisper", "en", None)

    upload = VideoMeta(source_type="upload", project_id="p2", title="T", duration=100.0)
    assert resolve_subtitle_plan("auto", upload, "id", source_language="ko") == SubtitlePlan("whisper", None, None)
    assert resolve_subtitle_plan("youtube_manual", upload, "id") == SubtitlePlan("whisper", None, None)


def test_plan_explicit_youtube_auto_uses_spoken_language():
    meta_ko = VideoMeta(source_type="youtube", project_id="p1", title="T", duration=100.0,
                        subtitles={"manual": [], "auto": ["ko"]})
    assert resolve_subtitle_plan("youtube_auto", meta_ko, "id") == SubtitlePlan("youtube_auto", "ko", True)

    meta_both = VideoMeta(source_type="youtube", project_id="p1", title="T", duration=100.0,
                          subtitles={"manual": [], "auto": ["ko", "id"]})
    assert resolve_subtitle_plan("youtube_auto", meta_both, "id") == SubtitlePlan("youtube_auto", "id", False)


def test_plan_explicit_sources_without_subs_rejected():
    no_manual = VideoMeta(source_type="youtube", project_id="p1", title="T", duration=100.0,
                          subtitles={"manual": [], "auto": ["id"]})
    with pytest.raises(AppError) as e_m:
        resolve_subtitle_plan("youtube_manual", no_manual, "id")
    assert e_m.value.code == "VALIDATION" and e_m.value.status == 422

    no_auto = VideoMeta(source_type="youtube", project_id="p1", title="T", duration=100.0,
                        subtitles={"manual": ["id"], "auto": []})
    with pytest.raises(AppError) as e_a:
        resolve_subtitle_plan("youtube_auto", no_auto, "id")
    assert e_a.value.code == "VALIDATION" and e_a.value.status == 422


def test_resolve_transcript_source_still_returns_string():
    meta_en = VideoMeta(source_type="youtube", project_id="p1", title="T", duration=100.0,
                        subtitles={"manual": ["en"], "auto": []})
    assert resolve_transcript_source("auto", meta_en, "id") == "youtube_manual"

    meta_none = VideoMeta(source_type="youtube", project_id="p1", title="T", duration=100.0,
                          subtitles={"manual": [], "auto": ["id"]})
    assert resolve_transcript_source("auto", meta_none, "id") == "whisper"


def test_distribute_words_public_and_legacy_alias():
    words = distribute_words(["halo", "semuanya"], 1.0, 3.0, approx=True)
    assert [w.w for w in words] == ["halo", "semuanya"]
    assert words[0].start == 1.0 and words[1].end == 3.0
    assert all(w.approx is True for w in words)
    assert youtube_subs._distribute_words is distribute_words


def test_transcript_dir_per_source(tmp_path):
    s = Settings(_env_file=None, data_dir=tmp_path)
    base = tmp_path / "projects" / "p1" / "transcripts"
    assert transcript_dir(s, Project(id="p1", transcript_source="auto", whisper_model="org/m:v3")) == base / "org-m-v3"
    assert transcript_dir(s, Project(id="p1", transcript_source="whisper", whisper_model="small")) == base / "small"
    assert transcript_dir(s, Project(id="p1", transcript_source="youtube_manual", target_language="id")) == (
        base / "youtube-manual-id"
    )
    assert transcript_dir(s, Project(id="p1", transcript_source="youtube_auto", target_language="en")) == (
        base / "youtube-auto-en"
    )


def test_download_subs_options_reuse_and_missing_error(monkeypatch, tmp_path):
    s = Settings(_env_file=None, data_dir=tmp_path)
    subs_dir = tmp_path / "projects" / "p1" / "subs"
    calls = []

    class _SubsYDL:
        write_file = True

        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def download(self, urls):
            calls.append((self.opts, urls))
            if self.write_file:
                p = Path(self.opts["outtmpl"].replace("%(ext)s", "id.json3"))
                p.write_text(json.dumps(_fixture("subs_manual.json3")), encoding="utf-8")

    monkeypatch.setattr(youtube_subs, "YoutubeDL", _SubsYDL)
    out = download_subs("dQw4w9WgXcQ", "id", "manual", subs_dir, s, cookies_browser="chrome")
    assert out == subs_dir / "dQw4w9WgXcQ.id.json3" and out.exists()
    assert calls[0][0]["skip_download"] is True
    assert calls[0][0]["writesubtitles"] is True and calls[0][0]["writeautomaticsub"] is False
    assert calls[0][0]["subtitleslangs"] == ["id"] and calls[0][0]["subtitlesformat"] == "json3"
    assert calls[0][0]["cookiesfrombrowser"] == ("chrome",)

    # existing file is reused without calling yt-dlp
    monkeypatch.setattr(youtube_subs, "YoutubeDL", None)
    assert download_subs("dQw4w9WgXcQ", "id", "manual", subs_dir, s) == out

    # missing output file raises INVALID_STATE
    _SubsYDL.write_file = False
    monkeypatch.setattr(youtube_subs, "YoutubeDL", _SubsYDL)
    with pytest.raises(AppError) as e:
        download_subs("dQw4w9WgXcQ", "en", "auto", subs_dir, s)
    assert e.value.code == "INVALID_STATE" and "en" in e.value.message


def test_process_youtube_manual_source_skips_whisper(engine_and_settings):
    engine, settings = engine_and_settings
    pid, jid = _setup(engine, moment_source="claude_manual", transcript_source="auto", target_language="id")
    d = settings.data_dir / "projects" / pid
    (settings.data_dir / "corrections.json").write_text(
        json.dumps([{"from": "semuanya", "to": "Semuanya"}]), encoding="utf-8"
    )
    rec = Recorder(settings)
    sub_calls = []

    def fake_ingest(project, s):
        rec._hit("ingest")
        (d).mkdir(parents=True, exist_ok=True)
        (d / "source.mp4").write_bytes(b"v")
        return d / "source.mp4", VideoMeta(
            source_type="youtube", project_id=project.id, title="Judul", channel="Kanal",
            duration=20.0, video_id="dQw4w9WgXcQ", subtitles={"manual": ["id"], "auto": ["id"]},
        )

    def fake_download_subs(vid, lang, kind, out_dir, s):
        rec._hit("download_subs")
        sub_calls.append((vid, lang, kind, out_dir))
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"{vid}.{lang}.json3"
        dest.write_text(json.dumps(_fixture("subs_manual.json3")), encoding="utf-8")
        return dest

    deps = rec.deps()
    deps.ingest = fake_ingest
    deps.download_subs = fake_download_subs
    deps.transcribe = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Whisper must not be called"))

    assert run_process_job(engine, jid, deps) == "paused"
    assert rec.calls == ["ingest", "extract_audio", "detect_silences", "download_subs", "signals"]
    assert sub_calls == [("dQw4w9WgXcQ", "id", "manual", d / "subs")]

    p, _ = _rows(engine, pid, jid)
    assert p.transcript_source == "youtube_manual"
    meta = VideoMeta.model_validate_json((d / "meta.json").read_text(encoding="utf-8"))
    assert meta.transcript_source == "youtube_manual"
    assert (d / "silences.json").exists()

    tpath = d / "transcripts" / "youtube-manual-id" / "transcript.json"
    assert tpath.exists()
    t = Transcript.model_validate_json(tpath.read_text(encoding="utf-8"))
    assert t.language == "id"
    # correction applied, and 1.5 s pause between event 0 (end 3.0) and event 1 (start 4.5) splits sentences
    assert [s.text for s in t.sentences] == ["halo Semuanya", "apa kabar sampai jumpa"]
    assert all(w.approx is True for w in t.words)


def test_default_download_subs_reads_cookies_browser(engine_and_settings, monkeypatch):
    engine, settings = engine_and_settings
    with Session(engine) as s:
        put_app_settings(s, {"cookies_browser": "edge"}, settings=settings)
    seen = {}

    def fake_dl(vid, lang, kind, out, s, cookies_browser=""):
        seen.update(vid=vid, lang=lang, kind=kind, out=out, cookies_browser=cookies_browser)
        return out / f"{vid}.{lang}.json3"

    monkeypatch.setattr(process, "download_subs", fake_dl)
    out = settings.data_dir / "projects" / "p1" / "subs"
    process.default_download_subs("dQw4w9WgXcQ", "id", "auto", out, settings)
    assert seen == {"vid": "dQw4w9WgXcQ", "lang": "id", "kind": "auto", "out": out, "cookies_browser": "edge"}


def test_process_ranged_youtube_auto_keeps_explicit_source_and_original_timeline(engine_and_settings):
    engine, settings = engine_and_settings
    pid, jid = _setup(engine, moment_source="claude_manual", transcript_source="youtube_auto",
                      target_language="id", ranges=[[1800, 1830], [3600, 3640]])
    d = settings.data_dir / "projects" / pid
    (settings.data_dir / "corrections.json").write_text(
        json.dumps([{"from": "halo dunia", "to": "Halo Dunia"}]), encoding="utf-8"
    )
    rec = Recorder(settings)
    ranges_meta = [
        {"index": 0, "start": 1800.0, "end": 1830.0, "file": "source_r0.mp4"},
        {"index": 1, "start": 3600.0, "end": 3640.0, "file": "source_r1.mp4"},
    ]

    def fake_ingest(project, s):
        d.mkdir(parents=True, exist_ok=True)
        for r in ranges_meta:
            (d / r["file"]).write_bytes(b"v")
        return d / "source_r0.mp4", VideoMeta(
            source_type="youtube", project_id=project.id, title="Judul", duration=7200.0,
            video_id="dQw4w9WgXcQ", ranges=ranges_meta, subtitles={"manual": [], "auto": ["id"]},
        )

    subs_data = {
        "events": [
            {"tStartMs": 10000, "dDurationMs": 2000, "segs": [{"utf8": "dibuang"}]},
            {"tStartMs": 1805000, "dDurationMs": 2000, "segs": [{"utf8": "halo"}, {"utf8": " dunia", "tOffsetMs": 800}]},
            {"tStartMs": 3602000, "dDurationMs": 1500, "segs": [{"utf8": "lanjut lagi"}]},
        ]
    }

    def fake_dl(vid, lang, kind, out_dir, s):
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / f"{vid}.{lang}.json3"
        p.write_text(json.dumps(subs_data), encoding="utf-8")
        return p

    deps = rec.deps()
    deps.ingest = fake_ingest
    deps.download_subs = fake_dl
    deps.transcribe = lambda *a, **kw: (_ for _ in ()).throw(AssertionError("Whisper must not be called"))

    assert run_process_job(engine, jid, deps) == "paused"
    p, _ = _rows(engine, pid, jid)
    assert p.transcript_source == "youtube_auto"
    t = Transcript.model_validate_json((d / "transcripts" / "youtube-auto-id" / "transcript.json").read_text("utf-8"))
    assert [(w.w, w.start, w.end, w.approx) for w in t.words] == [
        ("Halo Dunia", 1805.0, 1807.0, False),
        ("lanjut", 3602.0, 3602.9, True),
        ("lagi", 3602.9, 3603.5, True),
    ]
    assert [s.text for s in t.sentences] == ["Halo Dunia", "lanjut lagi"]


def test_download_subs_auto_requests_orig_lang_when_available(monkeypatch, tmp_path):
    s = Settings(_env_file=None, data_dir=tmp_path)
    subs_dir = tmp_path / "projects" / "p1" / "subs"
    seen_langs: list[list[str]] = []

    class _OrigYDL:
        caps = {"en-orig": [{"ext": "json3"}], "en": [{"ext": "json3"}], "id": [{"ext": "json3"}]}

        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            return {"automatic_captions": self.caps}

        def download(self, urls):
            langs = list(self.opts["subtitleslangs"])
            seen_langs.append(langs)
            p = Path(self.opts["outtmpl"].replace("%(ext)s", f"{langs[0]}.json3"))
            p.write_text(json.dumps(_fixture("subs_auto.json3")), encoding="utf-8")

    monkeypatch.setattr(youtube_subs, "YoutubeDL", _OrigYDL)
    out = download_subs("dQw4w9WgXcQ", "en", "auto", subs_dir, s)
    assert seen_langs == [["en-orig"]]
    assert out == subs_dir / "dQw4w9WgXcQ.en.json3" and out.exists()

    # Fallback to exact lang when no `<lang>-orig` exists
    _OrigYDL.caps = {"id": [{"ext": "json3"}]}
    seen_langs.clear()
    out2 = download_subs("dQw4w9WgXcQ", "id", "auto", subs_dir, s)
    assert seen_langs == [["id"]]
    assert out2 == subs_dir / "dQw4w9WgXcQ.id.json3" and out2.exists()


def test_explicit_youtube_sub_source_validates_before_download(engine_and_settings, monkeypatch):
    from app.pipeline import ingest

    engine, settings = engine_and_settings
    pid, jid = _setup(engine, moment_source="claude_manual", transcript_source="youtube_auto", target_language="id")

    class _NoDownloadYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def extract_info(self, url, download=False):
            assert download is False
            return {
                "id": "dQw4w9WgXcQ",
                "title": "Video Tanpa Sub",
                "duration": 120.0,
                "subtitles": {},
                "automatic_captions": {},
            }

        def download(self, urls):
            raise AssertionError("video download must not run when requested subtitles are unavailable")

    monkeypatch.setattr(ingest, "YoutubeDL", _NoDownloadYDL)
    assert run_process_job(engine, jid, ProcessDeps(settings=settings)) == "done"
    p, j = _rows(engine, pid, jid)
    assert p.status == "failed" and p.last_completed_stage is None
    assert j.status == "failed" and j.error_code == "VALIDATION"

    # Explicit youtube_manual with custom probe + ingest also rejects before calling ingest
    pid_m, jid_m = _setup(engine, moment_source="claude_manual", transcript_source="youtube_manual",
                          target_language="id")
    deps_m = ProcessDeps(
        settings=settings,
        probe=lambda proj, s: VideoMeta(source_type="youtube", project_id=proj.id, title="Video Tanpa Manual",
                                        duration=120.0, subtitles={"manual": [], "auto": ["en"]}),
        ingest=lambda *a, **kw: (_ for _ in ()).throw(AssertionError("ingest must not run")),
    )
    assert run_process_job(engine, jid_m, deps_m) == "done"
    pm, jm = _rows(engine, pid_m, jid_m)
    assert pm.status == "failed" and jm.error_code == "VALIDATION"


@pytest.fixture
def engine_and_settings(tmp_path):
    s = Settings(_env_file=None, data_dir=tmp_path)
    eng = get_engine(s)
    init_db(eng)
    return eng, s
