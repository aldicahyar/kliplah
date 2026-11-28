from app.core.models import Sentence, Transcript, Word
from app.domain.clip_doc import SubtitleLineV2, subtitles_from_m1
from app.domain.translation import Translation
from app.pipeline.subtitle_lines import clip_subtitle_lines
from app.pipeline.subtitles import group_lines


def _indonesian_transcript() -> Transcript:
    words = [
        Word(w="halo", start=10.2, end=10.5, prob=0.9, approx=False),
        Word(w="semua", start=10.5, end=10.9, prob=0.45, approx=False),
        Word(w="apa", start=10.9, end=11.2, prob=0.8, approx=False),
        Word(w="kabar", start=11.2, end=11.6, prob=0.85, approx=False),
        Word(w="hari", start=11.6, end=11.9, prob=0.75, approx=False),
        Word(w="ini?", start=11.9, end=12.4, prob=0.95, approx=False),
    ]
    sentences = [
        Sentence(i=0, start=10.2, end=12.4, text="halo semua apa kabar hari ini?", word_range=(0, 6)),
    ]
    return Transcript(language="id", duration=15.0, words=words, sentences=sentences)


def _english_transcript() -> Transcript:
    words = [
        # Sentence 0: [10.0, 12.0]
        Word(w="first", start=10.0, end=10.8, prob=0.9, approx=False),
        Word(w="line", start=10.9, end=12.0, prob=0.92, approx=False),
        # Sentence 1: [12.5, 14.5]
        Word(w="second", start=12.5, end=13.4, prob=0.88, approx=False),
        Word(w="line", start=13.5, end=14.5, prob=0.91, approx=False),
        # Sentence 2: [15.0, 17.0]
        Word(w="third", start=15.0, end=15.9, prob=0.84, approx=False),
        Word(w="line", start=16.0, end=17.0, prob=0.87, approx=False),
    ]
    sentences = [
        Sentence(i=0, start=10.0, end=12.0, text="first line", word_range=(0, 2)),
        Sentence(i=1, start=12.5, end=14.5, text="second line", word_range=(2, 4)),
        Sentence(i=2, start=15.0, end=17.0, text="third line", word_range=(4, 6)),
    ]
    return Transcript(language="en", duration=20.0, words=words, sentences=sentences)


def test_clip_subtitle_lines_without_translation_matches_m1():
    t = _indonesian_transcript()
    expected = subtitles_from_m1(group_lines(t.words, 10.0, 13.0, max_words=4))

    assert clip_subtitle_lines(t, None, 10.0, 13.0, max_words=4) == expected

    same_lang = Translation(language="id", source_language="id", lines={"0": "teks diabaikan"})
    assert clip_subtitle_lines(t, same_lang, 10.0, 13.0, max_words=4) == expected

    yt_lines = clip_subtitle_lines(t, None, 10.0, 13.0, max_words=4, source="youtube_manual")
    assert all(line["source"] == "youtube_manual" and line["text_original"] is None for line in yt_lines)


def test_clip_subtitle_lines_translated_fills_text_original_by_overlap():
    t = _english_transcript()
    tr = Translation(
        language="id",
        source_language="en",
        lines={
            "0": "baris pertama",
            "1": "baris kedua",
            "2": "baris ketiga",
        },
    )

    lines = clip_subtitle_lines(t, tr, 10.0, 18.0, max_words=4, source="whisper")
    assert len(lines) == 3
    assert [line["id"] for line in lines] == [1, 2, 3]
    assert [line["text"] for line in lines] == ["baris pertama", "baris kedua", "baris ketiga"]
    assert [line["text_original"] for line in lines] == ["first line", "second line", "third line"]
    assert all(line["source"] == "translated" for line in lines)
    assert all(w["approx"] is True for line in lines for w in line["words"])
    assert lines[0]["start"] == 0.0 and lines[0]["end"] == 2.0
    assert lines[1]["start"] == 2.5 and lines[1]["end"] == 4.5
    assert lines[2]["start"] == 5.0 and lines[2]["end"] == 7.0
    for line in lines:
        SubtitleLineV2.model_validate(line)


def test_clip_subtitle_lines_partial_and_empty_translation_fallback():
    t = _english_transcript()

    empty_tr = Translation(language="id", source_language="en", lines={})
    fallback = clip_subtitle_lines(t, empty_tr, 10.0, 18.0, max_words=4, source="youtube_auto")
    assert len(fallback) == 3
    assert all(line["source"] == "youtube_auto" and line["text_original"] is None for line in fallback)

    partial_tr = Translation(language="id", source_language="en", lines={"0": "baris satu", "2": "baris tiga"})
    mixed = clip_subtitle_lines(t, partial_tr, 10.0, 18.0, max_words=4, source="youtube_manual")
    assert [line["source"] for line in mixed] == ["translated", "youtube_manual", "translated"]
    assert [line["text"] for line in mixed] == ["baris satu", "second line", "baris tiga"]
    assert [line["text_original"] for line in mixed] == ["first line", None, "third line"]


def test_clip_subtitle_lines_empty_original_overlap_becomes_none():
    t = Transcript(
        language="en",
        duration=3.0,
        words=[],
        sentences=[Sentence(i=0, start=0.0, end=2.0, text="silent caption", word_range=(0, 0))],
    )
    tr = Translation(language="id", source_language="en", lines={"0": "teks terjemahan"})
    lines = clip_subtitle_lines(t, tr, 0.0, 3.0, max_words=4)
    assert len(lines) == 1
    assert lines[0]["source"] == "translated"
    assert lines[0]["text"] == "teks terjemahan"
    assert lines[0]["text_original"] is None


def test_clip_subtitle_lines_splits_single_sentence_across_lines():
    words = [
        Word(w="one", start=10.0, end=10.9),
        Word(w="two", start=11.0, end=11.9),
        Word(w="three", start=12.0, end=12.9),
        Word(w="four", start=13.0, end=13.9),
        Word(w="five", start=14.0, end=14.9),
        Word(w="six", start=15.0, end=16.0),
    ]
    t = Transcript(
        language="en",
        duration=18.0,
        words=words,
        sentences=[Sentence(i=0, start=10.0, end=16.0, text="one two three four five six", word_range=(0, 6))],
    )
    tr = Translation(language="id", source_language="en", lines={"0": "satu dua, tiga empat, lima enam"})

    lines = clip_subtitle_lines(t, tr, 10.0, 16.0, max_words=4)
    assert [line["id"] for line in lines] == [1, 2, 3]
    assert [line["text"] for line in lines] == ["satu dua,", "tiga empat,", "lima enam"]
    assert all(line["source"] == "translated" for line in lines)
    # Every original word is assigned to its overlapping translated line without loss or duplication
    joined_original = " ".join(line["text_original"] or "" for line in lines)
    assert joined_original == "one two three four five six"
    assert all(line["text_original"] for line in lines)


def test_clip_subtitle_lines_rounds_line_bounds_against_float_overshoot():
    # 0.1 + 0.2 == 0.30000000000000004 in IEEE-754 float; unrounded bounds steal/drop a midpoint at 0.3
    assert 0.1 + 0.2 > 0.3
    words = [
        Word(w="zero", start=0.15, end=0.25),   # midpoint 0.2
        Word(w="first", start=0.25, end=0.35),  # midpoint 0.3
        Word(w="second", start=0.4, end=0.6),   # midpoint 0.5
    ]
    t = Transcript(
        language="en",
        duration=1.0,
        words=words,
        sentences=[
            Sentence(i=0, start=0.1, end=0.3, text="zero", word_range=(0, 1)),
            Sentence(i=1, start=0.3, end=0.9, text="first second", word_range=(1, 3)),
        ],
    )
    tr = Translation(language="id", source_language="en", lines={"0": "nol.", "1": "satu dua"})

    lines = clip_subtitle_lines(t, tr, 0.1, 1.0, max_words=4)
    assert [line["text_original"] for line in lines] == ["zero", "first second"]

