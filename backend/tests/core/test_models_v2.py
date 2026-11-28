from app.core.errors import KNOWN_CODES
from app.core.models import Candidate, Transcript, VideoMeta, Word
from app.db.tables import Clip
from app.services._common import err

S1_TRANSCRIPT = {
    "language": "id",
    "duration": 4.0,
    "words": [{"w": "halo", "start": 0.0, "end": 0.5}, {"w": "semua", "start": 0.5, "end": 1.0}],
    "sentences": [{"i": 0, "start": 0.0, "end": 1.0, "text": "halo semua", "word_range": [0, 2]}],
}

S1_META = {
    "source_type": "youtube",
    "project_id": "p1",
    "title": "Judul",
    "channel": "Kanal",
    "duration": 120.0,
    "video_id": "abc123",
    "heatmap": [{"start": 0.0, "end": 1.0, "value": 0.5}],
}


def test_old_transcript_json_still_loads():
    t = Transcript.model_validate(S1_TRANSCRIPT)
    assert t.words[0].approx is False
    assert t.words[0].prob is None


def test_word_accepts_new_fields():
    w = Word(w="x", start=0.0, end=0.1, approx=True, prob=0.9)
    assert (w.approx, w.prob) == (True, 0.9)


def test_videometa_defaults_for_legacy_meta():
    m = VideoMeta.model_validate(S1_META)
    assert m.ranges == []
    assert m.chapters == []
    assert m.subtitles == {}
    assert m.has_live_chat is False
    assert m.transcript_source is None
    assert VideoMeta.model_validate(m.model_dump()) == m


def test_videometa_new_fields_roundtrip():
    m = VideoMeta.model_validate({
        **S1_META,
        "ranges": [{"index": 0, "start": 10.0, "end": 40.0, "file": "range_0.mp4"}],
        "chapters": [{"start": 0.0, "end": 60.0, "title": "Intro"}],
        "subtitles": {"manual": ["id"], "auto": ["en"]},
        "has_live_chat": True,
        "transcript_source": "youtube_manual",
    })
    assert VideoMeta.model_validate(m.model_dump()) == m
    assert m.ranges[0]["file"] == "range_0.mp4"


def test_candidate_and_clip_signal_bonus_default():
    c = Candidate(start_sentence=0, end_sentence=1, start=0.0, end=1.0, llm_score=None,
                  category=None, reason="", hook="")
    assert c.signal_bonus == 0
    assert Clip(origin="ai", rank=1, score=50).signal_bonus == 0


def test_age_restricted_error_code():
    assert "AGE_RESTRICTED" in KNOWN_CODES
    e = err("AGE_RESTRICTED", "Video dibatasi usia.")
    assert (e.status, e.retryable) == (400, False)


def test_videometa_language_defaults_for_legacy_meta():
    s2_meta = {
        **S1_META,
        "ranges": [{"index": 0, "start": 10.0, "end": 40.0, "file": "range_0.mp4"}],
        "chapters": [{"start": 0.0, "end": 60.0, "title": "Intro"}],
        "subtitles": {"manual": ["id"], "auto": ["en"]},
        "has_live_chat": True,
        "transcript_source": "youtube_manual",
    }
    m = VideoMeta.model_validate(s2_meta)
    assert m.language is None
    assert m.source_language is None
    assert VideoMeta.model_validate(m.model_dump()) == m

    m_with_lang = VideoMeta.model_validate({**s2_meta, "language": "ko", "source_language": "ko"})
    assert m_with_lang.language == "ko"
    assert m_with_lang.source_language == "ko"
    assert VideoMeta.model_validate(m_with_lang.model_dump()) == m_with_lang
