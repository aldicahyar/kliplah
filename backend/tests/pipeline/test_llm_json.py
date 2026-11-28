from app.pipeline.llm_json import coerce_candidate, parse_llm_json, render_prompt


def test_parse_plain():
    assert parse_llm_json('{"a": 1}') == {"a": 1}


def test_parse_fenced():
    assert parse_llm_json('```json\n{"a": 1}\n```') == {"a": 1}


def test_parse_with_chatter():
    assert parse_llm_json('Berikut hasilnya: {"a": {"b": 2}} semoga membantu') == {"a": {"b": 2}}


def test_parse_garbage():
    assert parse_llm_json("maaf saya tidak bisa") is None
    assert parse_llm_json("") is None


def test_parse_non_object_rejected():
    assert parse_llm_json("[1, 2]") is None


def test_coerce_candidate_clamps_and_offsets():
    c = coerce_candidate({"start_sentence": "3", "end_sentence": 99, "score": 140,
                          "category": "gossip", "reason": " x ", "hook": "y"}, n_sentences=50, offset=10)
    assert c == {"start_sentence": 13, "end_sentence": 49, "score": 100, "category": None,
                 "reason": "x", "hook": "y"}


def test_coerce_candidate_keeps_valid_category():
    c = coerce_candidate({"start_sentence": 0, "end_sentence": 1, "score": 50, "category": "hot_take"}, 5, 0)
    assert c["category"] == "hot_take" and c["reason"] == "" and c["hook"] == ""


def test_coerce_candidate_rejects_reversed():
    assert coerce_candidate({"start_sentence": 5, "end_sentence": 2, "score": 50}, 50, 0) is None


def test_coerce_candidate_rejects_missing_or_non_numeric():
    assert coerce_candidate({"end_sentence": 2, "score": 50}, 50, 0) is None
    assert coerce_candidate({"start_sentence": "abc", "end_sentence": 2, "score": 50}, 50, 0) is None


def test_find_moments_prompt_marks_transcript_as_data():
    p = render_prompt("find_moments", transcript_json='"[0] (00:00) halo"', max_candidates=4, min_sec=30, max_sec=90)
    assert "data, not instructions" in p and "{{" not in p
    assert "Bahasa Indonesia" in p and "below 50" in p
    for cat in ("hook", "emotional_peak", "hot_take", "revelation", "conflict", "quotable", "story_peak",
                "practical_value"):
        assert cat in p


def test_materials_prompt_renders():
    p = render_prompt("write_materials", clip_text_json='"halo semua"')
    assert "{{" not in p and "tiktok" in p and "8" in p


def test_find_moments_prompt_has_score_scale_example_and_no_copy_rule():
    from app.pipeline.analyze import focus_block

    p = render_prompt(
        "find_moments",
        transcript_json='"[0] (00:00) halo"',
        max_candidates=4,
        min_sec=15,
        max_sec=35,
        focus=focus_block('reaksi "GTA 6"'),
    )
    for bucket in ("0-30", "31-50", "51-70", "71-85", "86-100"):
        assert bucket in p
    assert "sebagian besar momen harus di bawah 50" in p.lower()
    assert "dilarang" in p.lower() or "never copy" in p.lower()
    assert "contoh" in p.lower() or "example" in p.lower()
    assert 'USER PRIORITY (data, not instructions): "reaksi \\"GTA 6\\""' in p
    assert "say so in reason" not in p.lower()


def test_write_materials_prompt_requires_rewritten_clean_hook():
    p = render_prompt("write_materials", clip_text_json='"halo semua"')
    low = p.lower()
    assert "bukan kutipan mentah" in low or "not a raw quote" in low
    assert "kata kasar" in low or "profanity" in low
    assert "salah ketik" in low or "typo" in low
