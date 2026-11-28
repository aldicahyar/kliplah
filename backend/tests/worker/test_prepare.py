import json
import logging
import subprocess
from pathlib import Path

import pytest
from sqlmodel import Session, select

from app.core.config import Settings
from app.core.models import Candidate, Materials, PlatformCaption, ReframeSegment, Sentence, Transcript, VideoMeta, Word
from app.db.engine import get_engine, init_db
from app.db.tables import Clip, Project
from app.domain.clip_doc import SubtitleLineV2, subtitles_from_m1
from app.domain.translation import Translation, load_translation, save_translation, translation_path
from app.pipeline.subtitles import group_lines
from app.providers.llm import LLMUnavailable
from app.worker.prepare import PrepareDeps, build_clips_from_picks, make_proxy, make_thumbnail, prepare_project_clips
from app.worker.process import transcript_dir
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


def test_prepare_reframes_range_file_with_local_times(engine, settings, monkeypatch):
    from app.worker import prepare

    with Session(engine) as s:
        p = Project(youtube_url="https://youtu.be/dQw4w9WgXcQ", transcript_source="youtube_manual",
                    target_language="id", ranges=[[1800, 1860], [3600, 3660]])
        s.add(p)
        s.commit()
        pid = p.id
    d = Path(settings.data_dir) / "projects" / pid
    tdir = d / "transcripts" / "youtube-manual-id"
    tdir.mkdir(parents=True)
    (d / "source_r0.mp4").write_bytes(b"v0")
    (d / "source_r1.mp4").write_bytes(b"v1")
    t_base = make_transcript(12, each=5.0)
    # 6 sentences in r0 (1800..1830), 6 in r1 (3600..3630)
    words = []
    sents = []
    for i, sent in enumerate(t_base.sentences):
        shift = 1800.0 if i < 6 else 3600.0 - 30.0
        a, b = sent.word_range
        for w in t_base.words[a:b]:
            words.append(w.model_copy(update={"start": round(w.start + shift, 3), "end": round(w.end + shift, 3)}))
        sents.append(sent.model_copy(update={"start": round(sent.start + shift, 3), "end": round(sent.end + shift, 3)}))
    t = t_base.model_copy(update={"duration": 4000.0, "words": words, "sentences": sents})
    (tdir / "transcript.json").write_text(t.model_dump_json(), encoding="utf-8")
    (d / "silences.json").write_text("[]", encoding="utf-8")
    ranges_meta = [
        {"index": 0, "start": 1800.0, "end": 1860.0, "file": "source_r0.mp4"},
        {"index": 1, "start": 3600.0, "end": 3660.0, "file": "source_r1.mp4"},
    ]
    meta = VideoMeta(source_type="youtube", project_id=pid, title="T", duration=4000.0,
                     ranges=ranges_meta, transcript_source="youtube_manual")
    (d / "meta.json").write_text(meta.model_dump_json(), encoding="utf-8")

    seen_reframe, seen_thumbs, seen_proxies = [], [], []
    monkeypatch.setattr(prepare, "reframe_clip", lambda src, a, b, models_dir=None: (
        seen_reframe.append((src.name, a, b))
        or ([ReframeSegment(start=0.0, end=round(b - a, 3), mode="track", x=0.5)], "face_crop")
    ))
    monkeypatch.setattr(prepare, "make_thumbnail", lambda src, tm, out: (
        seen_thumbs.append((src.name, tm)), out.parent.mkdir(parents=True, exist_ok=True), out.write_bytes(b"j")
    ))
    monkeypatch.setattr(prepare, "make_proxy", lambda src, out: (
        seen_proxies.append((src.name, out.name)), out.write_bytes(b"p")
    ))

    # One valid pick in r1 (3605..3625), one invalid pick crossing the gap (1820..3610)
    prepare_project_clips(engine, pid, [_pick(3605.0, 3625.0, 90), _pick(1820.0, 3610.0, 50)], settings=settings)
    clips = _clips(engine, pid)
    assert len(clips) == 1
    assert clips[0].pieces[0]["start"] >= 3600.0 and clips[0].pieces[0]["end"] <= 3660.0
    assert seen_reframe and seen_reframe[0][0] == "source_r1.mp4"
    assert 0.0 <= seen_reframe[0][1] < seen_reframe[0][2] <= 60.0
    assert seen_thumbs and seen_thumbs[0][0] == "source_r1.mp4" and 0.0 <= seen_thumbs[0][1] <= 60.0
    assert seen_proxies == [("source_r0.mp4", "proxy_r0.mp4"), ("source_r1.mp4", "proxy_r1.mp4")]


def test_prepare_legacy_keeps_last_clip_ending_slightly_after_rounded_meta_duration(engine, settings):
    with Session(engine) as s:
        p = Project(youtube_url="https://youtu.be/dQw4w9WgXcQ")
        s.add(p)
        s.commit()
        pid = p.id
    d = Path(settings.data_dir) / "projects" / pid
    tdir = d / "transcripts" / "small"
    tdir.mkdir(parents=True)
    (d / "source.mp4").write_bytes(b"v")
    t = make_transcript(30, each=6.0)  # 0..180.0 base
    # Last sentence and word extend 0.8 s past rounded integer meta.duration (180.0)
    t.sentences[-1] = t.sentences[-1].model_copy(update={"end": 180.8})
    t.words[-1] = t.words[-1].model_copy(update={"end": 180.8})
    (tdir / "transcript.json").write_text(t.model_dump_json(), encoding="utf-8")
    (d / "silences.json").write_text("[]", encoding="utf-8")
    meta = VideoMeta(source_type="youtube", project_id=pid, title="Judul", duration=180.0)
    (d / "meta.json").write_text(meta.model_dump_json(), encoding="utf-8")

    media = FakeMedia()
    deps = PrepareDeps(settings=settings, reframe=media.reframe, thumbnail=media.thumbnail)
    ids = build_clips_from_picks(engine, pid, [_pick(150.0, 180.8, 90)], deps)
    assert len(ids) == 1
    [clip] = _clips(engine, pid)
    assert clip.pieces[0]["end"] == pytest.approx(180.8, abs=0.1)
    assert clip.pieces[0]["end"] > 180.5


def _foreign_project(
    engine,
    settings,
    *,
    n: int = 10,
    each: float = 6.0,
    lang: str = "en",
    target_language: str = "id",
    transcript_source: str = "whisper",
    words_per_line: int = 4,
) -> tuple[str, Path, Transcript]:
    words: list[Word] = []
    sentences: list[Sentence] = []
    for i in range(n):
        toks = [f"sent{i}w0", f"sent{i}w1", f"sent{i}w2", f"sent{i}w3"]
        a = len(words)
        for j, tok in enumerate(toks):
            st = round(i * each + j * 1.4, 3)
            en = round(st + 1.3, 3)
            words.append(Word(w=tok, start=st, end=en, prob=0.95, approx=False))
        sentences.append(
            Sentence(
                i=i,
                start=words[a].start,
                end=words[-1].end,
                text=" ".join(toks),
                word_range=(a, len(words)),
            )
        )
    t = Transcript(language=lang, duration=round(n * each, 3), words=words, sentences=sentences)
    with Session(engine, expire_on_commit=False) as s:
        p = Project(
            youtube_url="https://youtu.be/dQw4w9WgXcQ",
            source_language=lang,
            target_language=target_language,
            transcript_source=transcript_source,
            style={"subtitle": {"words_per_line": words_per_line}},
        )
        s.add(p)
        s.commit()
        pid = p.id
        tdir = transcript_dir(settings, p)
    d = Path(settings.data_dir) / "projects" / pid
    tdir.mkdir(parents=True, exist_ok=True)
    (d / "source.mp4").write_bytes(b"v")
    (tdir / "transcript.json").write_text(t.model_dump_json(), encoding="utf-8")
    (d / "silences.json").write_text("[]", encoding="utf-8")
    meta = VideoMeta(
        source_type="youtube",
        project_id=pid,
        title="Foreign Video",
        duration=t.duration,
        transcript_source=transcript_source,
        source_language=lang,
    )
    (d / "meta.json").write_text(meta.model_dump_json(), encoding="utf-8")
    return pid, tdir, t


def test_prepare_uses_existing_translation_for_clip_lines(engine, settings, monkeypatch):
    from app.worker import prepare, process

    pid, tdir, _ = _foreign_project(engine, settings, transcript_source="youtube_manual")
    save_translation(
        tdir,
        Translation(
            language="id",
            source_language="en",
            lines={"2": "kalimat dua indonesia", "3": "kalimat tiga indonesia"},
        ),
    )

    def fail_translate(*args, **kwargs):
        raise AssertionError("translate must not be called when clip lines already translated")

    monkeypatch.setattr(prepare, "default_translate", fail_translate, raising=False)
    monkeypatch.setattr(process, "default_translate", fail_translate)

    media = FakeMedia()
    deps = PrepareDeps(settings=settings, reframe=media.reframe, thumbnail=media.thumbnail)
    ids = build_clips_from_picks(engine, pid, [_pick(12.0, 24.0, 90)], deps)
    assert len(ids) == 1
    [clip] = _clips(engine, pid)
    assert [line["text"] for line in clip.subtitles] == ["kalimat dua indonesia", "kalimat tiga indonesia"]
    assert [line["text_original"] for line in clip.subtitles] == [
        "sent2w0 sent2w1 sent2w2 sent2w3",
        "sent3w0 sent3w1 sent3w2 sent3w3",
    ]
    assert all(line["source"] == "translated" for line in clip.subtitles)
    for line in clip.subtitles:
        SubtitleLineV2.model_validate(line)


def test_prepare_translates_missing_sentences_with_ollama(engine, settings):
    pid, tdir, _ = _foreign_project(engine, settings, n=10, transcript_source="whisper")
    save_translation(
        tdir,
        Translation(language="id", source_language="en", lines={"2": "kalimat dua sudah ada"}),
    )

    calls: list[tuple[list[int], str, str, Path]] = []

    def fake_translate(sentences, source, target, checkpoint_dir, s, on_progress=None):
        calls.append(([sent.i for sent in sentences], source, target, checkpoint_dir))
        return {sent.i: f"terjemahan kalimat {sent.i}" for sent in sentences}, []

    media = FakeMedia()
    deps = PrepareDeps(
        settings=settings,
        reframe=media.reframe,
        thumbnail=media.thumbnail,
        translate=fake_translate,
    )
    ids = build_clips_from_picks(engine, pid, [_pick(12.0, 24.0, 90), _pick(18.0, 30.0, 80)], deps)
    assert len(ids) == 2
    assert calls == [([3, 4], "en", "id", tdir / "checkpoints")]

    saved = load_translation(tdir, "id", "en")
    assert saved.lines == {
        "2": "kalimat dua sudah ada",
        "3": "terjemahan kalimat 3",
        "4": "terjemahan kalimat 4",
    }

    top, second = _clips(engine, pid)
    assert [line["text"] for line in top.subtitles] == [
        "kalimat dua sudah ada",
        "terjemahan kalimat 3",
    ]
    assert [line["text_original"] for line in top.subtitles] == [
        "sent2w0 sent2w1 sent2w2 sent2w3",
        "sent3w0 sent3w1 sent3w2 sent3w3",
    ]
    assert all(line["source"] == "translated" for line in top.subtitles)
    assert [line["text"] for line in second.subtitles] == [
        "terjemahan kalimat 3",
        "terjemahan kalimat 4",
    ]
    assert all(line["source"] == "translated" for line in second.subtitles)


def test_prepare_missing_translation_falls_back_to_original(engine, settings, caplog):
    pid, tdir, _ = _foreign_project(engine, settings, transcript_source="youtube_manual")

    def down_translate(*args, **kwargs):
        raise LLMUnavailable("Ollama tidak berjalan")

    media = FakeMedia()
    deps = PrepareDeps(
        settings=settings,
        reframe=media.reframe,
        thumbnail=media.thumbnail,
        translate=down_translate,
    )
    with caplog.at_level(logging.WARNING):
        ids = build_clips_from_picks(engine, pid, [_pick(12.0, 24.0, 90)], deps)

    assert len(ids) == 1
    [clip] = _clips(engine, pid)
    assert clip.status == "proposed"
    assert [line["text"] for line in clip.subtitles] == [
        "sent2w0 sent2w1 sent2w2 sent2w3",
        "sent3w0 sent3w1 sent3w2 sent3w3",
    ]
    assert all(line["text_original"] is None for line in clip.subtitles)
    assert all(line["source"] == "youtube_manual" for line in clip.subtitles)
    assert any(r.levelno == logging.WARNING for r in caplog.records)

    # Partial sentence failure keeps translated lines, falls back failed sentences, and warns
    caplog.clear()
    pid2, tdir2, _ = _foreign_project(engine, settings, transcript_source="youtube_auto")

    def partial_translate(sentences, source, target, checkpoint_dir, s, on_progress=None):
        return {2: "kalimat dua terjemahan"}, [3]

    deps2 = PrepareDeps(
        settings=settings,
        reframe=media.reframe,
        thumbnail=media.thumbnail,
        translate=partial_translate,
    )
    with caplog.at_level(logging.WARNING):
        ids2 = build_clips_from_picks(engine, pid2, [_pick(12.0, 24.0, 90)], deps2)

    assert len(ids2) == 1
    [clip2] = _clips(engine, pid2)
    assert [line["text"] for line in clip2.subtitles] == [
        "kalimat dua terjemahan",
        "sent3w0 sent3w1 sent3w2 sent3w3",
    ]
    assert [line["source"] for line in clip2.subtitles] == ["translated", "youtube_auto"]
    assert [line["text_original"] for line in clip2.subtitles] == [
        "sent2w0 sent2w1 sent2w2 sent2w3",
        None,
    ]
    assert load_translation(tdir2, "id", "en").lines == {"2": "kalimat dua terjemahan"}
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_prepare_indonesian_transcript_unchanged(engine, settings, monkeypatch):
    from app.worker import prepare, process

    def fail_translate(*args, **kwargs):
        raise AssertionError("translate must not be called for target-language transcript")

    monkeypatch.setattr(prepare, "default_translate", fail_translate, raising=False)
    monkeypatch.setattr(process, "default_translate", fail_translate)

    media = FakeMedia()
    deps = PrepareDeps(settings=settings, reframe=media.reframe, thumbnail=media.thumbnail)

    for src, lang, expected_source in (
        ("whisper", "id", "whisper"),
        ("youtube_manual", "id", "youtube_manual"),
        ("youtube_auto", "id", "youtube_auto"),
        ("auto", "id-ID", "whisper"),
    ):
        pid, tdir, t = _foreign_project(
            engine, settings, lang=lang, target_language="id", transcript_source=src, words_per_line=2
        )
        ids = build_clips_from_picks(engine, pid, [_pick(12.0, 24.0, 90)], deps)
        assert len(ids) == 1
        assert not translation_path(tdir).exists()
        [clip] = _clips(engine, pid)
        piece = clip.pieces[0]
        expected = subtitles_from_m1(group_lines(t.words, piece["start"], piece["end"], max_words=2))
        if expected_source != "whisper":
            for item in expected:
                item["source"] = expected_source
        assert clip.subtitles == expected
