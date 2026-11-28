"""Tests for manual translation export, import, and clip subtitle refresh service."""
import json

import pytest
from sqlmodel import Session

from app.core.config import Settings
from app.core.errors import AppError
from app.core.models import Sentence, Transcript, VideoMeta, Word
from app.db.engine import get_engine, init_db
from app.db.tables import Clip, Project
from app.domain.translation import Translation, load_translation, save_translation
from app.pipeline.subtitle_lines import clip_subtitle_lines
from app.services.translation import (
    PART_LINES,
    export_translation,
    import_translation,
    refresh_clip_subtitles,
)


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, data_dir=tmp_path)


@pytest.fixture
def engine(settings):
    eng = get_engine(settings)
    init_db(eng)
    return eng


def _english_transcript(n: int = 12, each: float = 6.0) -> Transcript:
    words: list[Word] = []
    sentences: list[Sentence] = []
    for i in range(n):
        toks = [f"word{i}a", f"word{i}b", f"word{i}c"]
        a = len(words)
        step = each / len(toks)
        for j, tok in enumerate(toks):
            st = round(i * each + j * step, 3)
            en = round(st + step * 0.8, 3)
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
    return Transcript(language="en", duration=round(n * each, 3), words=words, sentences=sentences)


def _setup_project(
    engine,
    settings,
    transcript: Transcript,
    *,
    status: str = "ready",
    target_language: str = "id",
    source_language: str | None = None,
    transcript_source: str = "whisper",
    whisper_model: str = "small",
) -> tuple[str, object]:
    src_lang = source_language if source_language is not None else transcript.language
    with Session(engine, expire_on_commit=False) as s:
        p = Project(
            status=status,
            last_completed_stage="prepare",
            whisper_model=whisper_model,
            source_language=src_lang,
            target_language=target_language,
            transcript_source=transcript_source,
        )
        s.add(p)
        s.commit()
        pid = p.id
    d = settings.data_dir / "projects" / pid
    d.mkdir(parents=True, exist_ok=True)
    meta = VideoMeta(
        source_type="youtube",
        project_id=pid,
        title="Video Asing",
        duration=transcript.duration,
        source_language=src_lang,
        transcript_source=transcript_source,
    )
    (d / "meta.json").write_text(meta.model_dump_json(), encoding="utf-8")

    if transcript_source == "youtube_manual":
        sub = f"youtube-manual-{src_lang}"
    elif transcript_source == "youtube_auto":
        sub = f"youtube-auto-{src_lang}"
    else:
        sub = whisper_model.replace(":", "-").replace("/", "-")
    tdir = d / "transcripts" / sub
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / "transcript.json").write_text(transcript.model_dump_json(), encoding="utf-8")
    return pid, tdir


def _add_clip(
    engine,
    pid: str,
    pieces: list[dict],
    subtitles: list[dict],
    *,
    rank: int = 1,
    status: str = "proposed",
) -> str:
    with Session(engine, expire_on_commit=False) as s:
        c = Clip(
            project_id=pid,
            origin="manual",
            rank=rank,
            score=70,
            pieces=pieces,
            subtitles=subtitles,
            status=status,
        )
        s.add(c)
        s.commit()
        return c.id


def test_export_translation_clips_scope_only_needed_sentences(engine, settings):
    t = _english_transcript(12, each=6.0)
    pid, tdir = _setup_project(engine, settings, t)
    # Clip 1 covers [12.0, 24.0] -> overlaps sentences 2, 3 -> expanded +-1 = {1, 2, 3, 4}
    _add_clip(engine, pid, [{"project_id": pid, "start": 12.0, "end": 24.0}], [], rank=1, status="proposed")
    # Clip 2 covers [54.0, 60.0] -> overlaps sentence 9 -> expanded +-1 = {8, 9, 10}
    _add_clip(engine, pid, [{"project_id": pid, "start": 54.0, "end": 60.0}], [], rank=2, status="approved")
    # Clip 3 is rejected -> [30.0, 42.0] (sentences 5, 6) must be ignored
    _add_clip(engine, pid, [{"project_id": pid, "start": 30.0, "end": 42.0}], [], rank=3, status="rejected")
    # Sentence 2 already translated -> skipped in export
    save_translation(tdir, Translation(language="id", source_language="en", lines={"2": "sudah diterjemahkan"}))

    parts = export_translation(engine, pid, scope="clips", settings=settings)
    assert len(parts) == 1
    p = parts[0]
    assert (p["part"], p["of"], p["filename"]) == (1, 1, "terjemahan_claude.md")
    text = p["text"]
    assert "Inggris" in text and "Indonesia" in text
    assert '"lines"' in text

    for needed_i in (1, 3, 4, 8, 9, 10):
        assert f"[{needed_i}] word{needed_i}a" in text
    for unneeded_i in (0, 2, 5, 6, 7, 11):
        assert f"[{unneeded_i}]" not in text

    export_dir = tdir / "translation_export"
    assert (export_dir / "terjemahan_claude.md").read_text(encoding="utf-8") == text
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest == {"parts": {"1": [1, 3, 4, 8, 9, 10]}}


def test_export_translation_all_scope_splits_parts_and_clears_old_export(engine, settings):
    assert PART_LINES == 300
    t = _english_transcript(310, each=2.0)
    pid, tdir = _setup_project(engine, settings, t)
    save_translation(
        tdir,
        Translation(language="id", source_language="en", lines={str(i): f"sudah {i}" for i in range(5)}),
    )
    export_dir = tdir / "translation_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    stale = export_dir / "stale.md"
    stale.write_text("old", encoding="utf-8")

    parts = export_translation(engine, pid, scope="all", settings=settings)
    assert not stale.exists()
    assert [(p["part"], p["of"], p["filename"]) for p in parts] == [
        (1, 2, "terjemahan_claude_part1of2.md"),
        (2, 2, "terjemahan_claude_part2of2.md"),
    ]
    assert "[5] " in parts[0]["text"] and "[304] " in parts[0]["text"] and "[305] " not in parts[0]["text"]
    assert "[305] " in parts[1]["text"] and "[309] " in parts[1]["text"] and "[304] " not in parts[1]["text"]

    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["parts"]["1"] == list(range(5, 305))
    assert manifest["parts"]["2"] == list(range(305, 310))

    # Validation error for invalid scope
    with pytest.raises(AppError) as e:
        export_translation(engine, pid, scope="wrong", settings=settings)  # type: ignore[arg-type]
    assert (e.value.code, e.value.status) == ("VALIDATION", 422)

    # When nothing is missing, returns [] and writes empty manifest
    save_translation(
        tdir,
        Translation(language="id", source_language="en", lines={str(i): f"id {i}" for i in range(310)}),
    )
    assert export_translation(engine, pid, scope="all", settings=settings) == []
    assert json.loads((export_dir / "manifest.json").read_text(encoding="utf-8")) == {"parts": {}}


def test_export_translation_rejects_target_language_transcript(engine, settings):
    t_id = _english_transcript(4, each=5.0).model_copy(update={"language": "id"})
    pid, _ = _setup_project(engine, settings, t_id, target_language="id")
    with pytest.raises(AppError) as e:
        export_translation(engine, pid, scope="all", settings=settings)
    assert (e.value.code, e.value.status) == ("INVALID_STATE", 409)
    assert "sudah berbahasa target" in e.value.message.lower()

    # Regional code id-ID also matches target id
    t_reg = _english_transcript(4, each=5.0).model_copy(update={"language": "id-ID"})
    pid_reg, _ = _setup_project(engine, settings, t_reg, target_language="id")
    with pytest.raises(AppError) as e2:
        export_translation(engine, pid_reg, scope="clips", settings=settings)
    assert (e2.value.code, e2.value.status) == ("INVALID_STATE", 409)

    with pytest.raises(AppError) as e3:
        import_translation(engine, pid, 1, '{"lines": [{"n": 0, "text": "x"}]}', settings=settings)
    assert (e3.value.code, e3.value.status) == ("INVALID_STATE", 409)


def test_import_translation_rejects_number_outside_part(engine, settings):
    t = _english_transcript(305, each=2.0)
    pid, tdir = _setup_project(engine, settings, t)
    export_translation(engine, pid, scope="all", settings=settings)

    # Part 2 only allows 300..304; sending 10 must fail with INVALID_IMPORT
    with pytest.raises(AppError) as e:
        import_translation(
            engine,
            pid,
            2,
            '```json\n{"lines": [{"n": 300, "text": "ok"}, {"n": 10, "text": "salah bagian"}]}\n```',
            settings=settings,
        )
    assert (e.value.code, e.value.status) == ("INVALID_IMPORT", 400)
    # Atomic rejection: valid line 300 from same invalid reply must NOT be saved
    assert load_translation(tdir, "id", "en").lines == {}

    # Part 1 only allows 0..299; sending 300 fails
    with pytest.raises(AppError) as e2:
        import_translation(engine, pid, 1, '{"lines": [{"n": 300, "text": "luar"}]}', settings=settings)
    assert (e2.value.code, e2.value.status) == ("INVALID_IMPORT", 400)

    # Unknown part number / bad JSON / bool n / missing lines
    for bad_part, body in (
        (0, '{"lines": [{"n": 0, "text": "x"}]}'),
        (3, '{"lines": [{"n": 0, "text": "x"}]}'),
        (1, "bukan json"),
        (1, '{"clips": []}'),
        (1, '{"lines": [{"n": true, "text": "x"}]}'),
        (1, '{"lines": [{"n": 0, "text": 123}]}'),
    ):
        with pytest.raises(AppError) as eb:
            import_translation(engine, pid, bad_part, body, settings=settings)
        assert (eb.value.code, eb.value.status) == ("INVALID_IMPORT", 400)


def test_import_translation_refreshes_untranslated_clips(engine, settings):
    t = _english_transcript(4, each=6.0)
    pid, _ = _setup_project(engine, settings, t, transcript_source="youtube_manual")
    orig_subs = clip_subtitle_lines(t, None, 0.0, 12.0, max_words=4, source="youtube_manual")
    assert all(line["source"] == "youtube_manual" and line["text_original"] is None for line in orig_subs)

    cid = _add_clip(
        engine,
        pid,
        [{"project_id": pid, "start": 0.0, "end": 12.0}],
        orig_subs,
        rank=1,
        status="rendered",
    )
    export_translation(engine, pid, scope="clips", settings=settings)
    reply = json.dumps(
        {
            "lines": [
                {"n": 0, "text": "kalimat pertama indonesia"},
                {"n": 1, "text": "kalimat kedua indonesia"},
            ]
        }
    )
    res = import_translation(engine, pid, 1, f"Ini terjemahannya:\n```json\n{reply}\n```", settings=settings)
    assert res == {"merged": 2, "refreshed": [cid], "skipped": []}

    with Session(engine) as s:
        clip = s.get(Clip, cid)
        assert clip is not None
        assert clip.needs_rerender is True
        assert [line["text"] for line in clip.subtitles] == [
            "kalimat pertama indonesia",
            "kalimat kedua indonesia",
        ]
        assert all(line["source"] == "translated" for line in clip.subtitles)
        assert [line["text_original"] for line in clip.subtitles] == [
            "word0a word0b word0c",
            "word1a word1b word1c",
        ]
        assert all(w["approx"] is True for line in clip.subtitles for w in line["words"])


def test_refresh_skips_clips_with_edited_lines(engine, settings):
    t = _english_transcript(4, each=6.0)
    pid, tdir = _setup_project(engine, settings, t, transcript_source="whisper")
    save_translation(
        tdir,
        Translation(
            language="id",
            source_language="en",
            lines={"0": "terjemahan nol", "1": "terjemahan satu"},
        ),
    )

    clean_subs = clip_subtitle_lines(t, None, 0.0, 12.0, max_words=4, source="whisper")
    edited_subs = clip_subtitle_lines(t, None, 0.0, 12.0, max_words=4, source="whisper")
    edited_subs[0]["text"] = "teks editan manual"
    edited_subs[0]["edited"] = True

    cid_edited = _add_clip(engine, pid, [{"project_id": pid, "start": 0.0, "end": 12.0}], edited_subs, rank=1)
    cid_clean = _add_clip(engine, pid, [{"project_id": pid, "start": 0.0, "end": 12.0}], clean_subs, rank=2)

    refreshed, skipped = refresh_clip_subtitles(engine, pid, settings=settings)
    assert refreshed == [cid_clean]
    assert skipped == [cid_edited]

    with Session(engine) as s:
        c_edited = s.get(Clip, cid_edited)
        c_clean = s.get(Clip, cid_clean)
        assert c_edited.subtitles[0]["text"] == "teks editan manual"
        assert c_edited.subtitles[0]["edited"] is True
        assert [line["text"] for line in c_clean.subtitles] == ["terjemahan nol", "terjemahan satu"]
        assert all(line["source"] == "translated" for line in c_clean.subtitles)
        assert [line["text_original"] for line in c_clean.subtitles] == [
            "word0a word0b word0c",
            "word1a word1b word1c",
        ]


def test_refresh_handles_multi_piece_clip_offsets(engine, settings):
    t = _english_transcript(4, each=6.0)
    pid, tdir = _setup_project(engine, settings, t, transcript_source="youtube_auto")
    save_translation(
        tdir,
        Translation(language="id", source_language="en", lines={"0": "bagian awal", "2": "bagian kedua"}),
    )
    cid = _add_clip(
        engine,
        pid,
        [
            {"project_id": pid, "start": 0.0, "end": 6.0},
            {"project_id": pid, "start": 12.0, "end": 18.0},
        ],
        [],
        rank=1,
    )
    refreshed, skipped = refresh_clip_subtitles(engine, pid, settings=settings)
    assert (refreshed, skipped) == ([cid], [])
    with Session(engine) as s:
        clip = s.get(Clip, cid)
        assert [line["id"] for line in clip.subtitles] == [1, 2]
        assert [line["text"] for line in clip.subtitles] == ["bagian awal", "bagian kedua"]
        assert clip.subtitles[0]["start"] < 6.0
        assert clip.subtitles[1]["start"] >= 6.0
        assert clip.subtitles[1]["text_original"] == "word2a word2b word2c"


def test_export_translation_boundary_clamping_and_errors(engine, settings, monkeypatch):
    from app.services import translation as tr_svc

    with pytest.raises(AppError) as e_nf:
        export_translation(engine, "tidak-ada", settings=settings)
    assert (e_nf.value.code, e_nf.value.status) == ("NOT_FOUND", 404)

    t = _english_transcript(4, each=6.0)
    pid, tdir = _setup_project(engine, settings, t)
    # Clip spanning first & last sentence clamps cleanly to [0, 3]
    _add_clip(engine, pid, [{"project_id": pid, "start": 0.0, "end": 24.0}], [], rank=1)
    parts = export_translation(engine, pid, scope="clips", settings=settings)
    assert len(parts) == 1
    assert json.loads((tdir / "translation_export" / "manifest.json").read_text(encoding="utf-8")) == {
        "parts": {"1": [0, 1, 2, 3]}
    }

    # OSError clearing translation_export raises INVALID_STATE 409
    monkeypatch.setattr(tr_svc.shutil, "rmtree", lambda _p: (_ for _ in ()).throw(PermissionError("locked")))
    with pytest.raises(AppError) as e_lock:
        export_translation(engine, pid, scope="clips", settings=settings)
    assert (e_lock.value.code, e_lock.value.status) == ("INVALID_STATE", 409)
    monkeypatch.undo()

    # Missing transcript.json raises INVALID_STATE 409
    (tdir / "transcript.json").unlink()
    with pytest.raises(AppError) as e_missing:
        export_translation(engine, pid, scope="all", settings=settings)
    assert (e_missing.value.code, e_missing.value.status) == ("INVALID_STATE", 409)


def test_import_translation_fallback_without_manifest_and_unchanged_rerender(engine, settings):
    t = _english_transcript(305, each=2.0)
    pid, tdir = _setup_project(engine, settings, t)
    cid = _add_clip(
        engine,
        pid,
        [{"project_id": pid, "start": 0.0, "end": 4.0}],
        clip_subtitle_lines(t, None, 0.0, 4.0, max_words=4, source="whisper"),
        status="rendered",
    )

    # Without manifest.json, fallback splits missing sentences by PART_LINES (0..299, 300..304)
    with pytest.raises(AppError) as e_out:
        import_translation(engine, pid, 2, '{"lines": [{"n": 5, "text": "salah"}]}', settings=settings)
    assert (e_out.value.code, e_out.value.status) == ("INVALID_IMPORT", 400)

    res = import_translation(engine, pid, 1, '{"lines": [{"n": 0, "text": "nol indonesia"}]}', settings=settings)
    assert res == {"merged": 1, "refreshed": [cid], "skipped": []}
    with Session(engine) as s:
        c = s.get(Clip, cid)
        assert c.needs_rerender is True
        c.needs_rerender = False
        s.add(c)
        s.commit()

    # Re-running refresh when subtitles are unchanged must NOT set needs_rerender
    refreshed, skipped = refresh_clip_subtitles(engine, pid, settings=settings)
    assert (refreshed, skipped) == ([cid], [])
    with Session(engine) as s:
        assert s.get(Clip, cid).needs_rerender is False


def test_refresh_skips_rendering_clips(engine, settings):
    t = _english_transcript(4, each=6.0)
    pid, tdir = _setup_project(engine, settings, t, transcript_source="whisper")
    save_translation(
        tdir,
        Translation(language="id", source_language="en", lines={"0": "terjemahan nol"}),
    )
    orig_subs = clip_subtitle_lines(t, None, 0.0, 6.0, max_words=4, source="whisper")
    cid_rendering = _add_clip(
        engine, pid, [{"project_id": pid, "start": 0.0, "end": 6.0}], orig_subs, rank=1, status="rendering"
    )
    cid_ready = _add_clip(
        engine, pid, [{"project_id": pid, "start": 0.0, "end": 6.0}], orig_subs, rank=2, status="proposed"
    )

    refreshed, skipped = refresh_clip_subtitles(engine, pid, settings=settings)
    assert refreshed == [cid_ready]
    assert skipped == [cid_rendering]
    with Session(engine) as s:
        c_rendering = s.get(Clip, cid_rendering)
        assert c_rendering.subtitles == orig_subs
        assert c_rendering.needs_rerender is False

