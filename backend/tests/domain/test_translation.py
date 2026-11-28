from app.core.models import Sentence, Transcript, Word
from app.domain.translation import (
    Translation,
    load_translation,
    merge_lines,
    missing_sentences,
    original_text_between,
    save_translation,
    translated_view,
    translation_path,
)


def _sample_transcript() -> Transcript:
    words = [
        Word(w="hello", start=1.0, end=1.4, prob=0.95, approx=False),
        Word(w="world.", start=1.4, end=2.0, prob=0.9, approx=False),
        Word(w="how", start=2.5, end=2.8, prob=0.88, approx=False),
        Word(w="are", start=2.8, end=3.1, prob=0.85, approx=False),
        Word(w="you?", start=3.1, end=3.5, prob=0.92, approx=False),
        Word(w="good", start=4.0, end=4.5, prob=0.8, approx=False),
        Word(w="morning.", start=4.5, end=5.2, prob=0.82, approx=False),
    ]
    sentences = [
        Sentence(i=0, start=1.0, end=2.0, text="hello world.", word_range=(0, 2)),
        Sentence(i=1, start=2.5, end=3.5, text="how are you?", word_range=(2, 5)),
        Sentence(i=2, start=4.0, end=5.2, text="good morning.", word_range=(5, 7)),
    ]
    return Transcript(language="en", duration=6.0, words=words, sentences=sentences)


def test_translation_roundtrip_and_mismatch_is_empty(tmp_path):
    assert translation_path(tmp_path) == tmp_path / "translation.json"

    missing = load_translation(tmp_path, "id", "en")
    assert missing == Translation(language="id", source_language="en", lines={})

    tr = Translation(language="id", source_language="en", lines={"0": "halo dunia.", "2": "selamat pagi."})
    save_translation(tmp_path, tr)
    assert translation_path(tmp_path).exists()

    loaded = load_translation(tmp_path, "id", "en")
    assert loaded == tr

    # Target or source language mismatch returns an empty Translation for the requested pair
    assert load_translation(tmp_path, "ja", "en") == Translation(language="ja", source_language="en", lines={})
    assert load_translation(tmp_path, "id", "ko") == Translation(language="id", source_language="ko", lines={})

    # Corrupt JSON or invalid schema returns empty
    translation_path(tmp_path).write_text("{not valid json", encoding="utf-8")
    assert load_translation(tmp_path, "id", "en") == Translation(language="id", source_language="en", lines={})

    translation_path(tmp_path).write_text('["not", "an", "object"]', encoding="utf-8")
    assert load_translation(tmp_path, "id", "en") == Translation(language="id", source_language="en", lines={})

    translation_path(tmp_path).write_text('{"language": "id", "source_language": "en", "lines": "bad"}',
                                         encoding="utf-8")
    assert load_translation(tmp_path, "id", "en") == Translation(language="id", source_language="en", lines={})


def test_merge_lines_cleans_skips_empty_and_invalid():
    tr = Translation(language="id", source_language="en", lines={"0": "lama", "1": "tetap sama"})

    added_or_changed = merge_lines(
        tr,
        {
            0: "",
            1: "  tetap   sama\n",
            2: "  baris   baru\tini  ",
            3: "   \n  ",
            4: None,  # type: ignore[dict-item]
            99: "luar batas",
        },
        valid={0, 1, 2, 3, 4},
    )

    assert added_or_changed == 1
    assert tr.lines == {"0": "lama", "1": "tetap sama", "2": "baris baru ini"}

    assert merge_lines(tr, {0: "  versi   baru  "}, valid=set()) == 0
    assert merge_lines(tr, {0: "  versi   baru  "}) == 1
    assert tr.lines["0"] == "versi baru"


def test_missing_sentences_respects_indices():
    t = _sample_transcript()
    tr = Translation(language="id", source_language="en", lines={"1": "apa kabar?", "2": "   "})

    assert [s.i for s in missing_sentences(t, tr)] == [0, 2]
    assert [s.i for s in missing_sentences(t, tr, indices={0, 1})] == [0]
    assert [s.i for s in missing_sentences(t, tr, indices={1})] == []
    assert missing_sentences(t, tr, indices=set()) == []


def test_translated_view_keeps_timing_and_marks_words_approx():
    t = _sample_transcript()
    tr = Translation(
        language="id",
        source_language="en",
        lines={"0": "halo dunia.", "2": "selamat pagi semua."},
    )

    view = translated_view(t, tr)

    assert view.language == "id"
    assert view.duration == t.duration
    assert len(view.sentences) == len(t.sentences)
    assert [(s.i, s.start, s.end) for s in view.sentences] == [(s.i, s.start, s.end) for s in t.sentences]

    # Sentence 0 translated: approx=True words and updated word_range
    s0 = view.sentences[0]
    assert s0.text == "halo dunia."
    w0 = view.words[s0.word_range[0]:s0.word_range[1]]
    assert [w.w for w in w0] == ["halo", "dunia."]
    assert all(w.approx is True and w.prob is None for w in w0)
    assert w0[0].start == 1.0 and w0[-1].end == 2.0

    # Sentence 1 untranslated: original text and original words/flags kept
    s1 = view.sentences[1]
    assert s1.text == "how are you?"
    w1 = view.words[s1.word_range[0]:s1.word_range[1]]
    assert [w.w for w in w1] == ["how", "are", "you?"]
    assert all(w.approx is False for w in w1)
    assert [w.prob for w in w1] == [0.88, 0.85, 0.92]

    # Sentence 2 translated
    s2 = view.sentences[2]
    assert s2.text == "selamat pagi semua."
    w2 = view.words[s2.word_range[0]:s2.word_range[1]]
    assert [w.w for w in w2] == ["selamat", "pagi", "semua."]
    assert all(w.approx is True for w in w2)
    assert s2.word_range[1] == len(view.words)

    # Original transcript is not mutated
    assert t.language == "en"
    assert t.sentences[0].text == "hello world."


def test_original_text_between_uses_word_midpoints():
    words = [
        Word(w="zero", start=0.0, end=0.8),   # mid = 0.4 -> before 1.0
        Word(w="left", start=0.5, end=1.5),   # mid = 1.0 -> included at start boundary
        Word(w="mid", start=1.2, end=1.6),    # mid = 1.4 -> included
        Word(w="right", start=1.5, end=2.5),  # mid = 2.0 -> excluded at end boundary
    ]
    t = Transcript(language="en", duration=3.0, words=words, sentences=[])

    assert original_text_between(t, 1.0, 2.0) == "left mid"
    assert original_text_between(t, 2.0, 3.0) == "right"
    assert original_text_between(t, 5.0, 6.0) == ""
    assert original_text_between(t, 2.0, 2.0) == ""
