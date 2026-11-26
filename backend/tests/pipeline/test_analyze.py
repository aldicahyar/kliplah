import json

import pytest

from app.core.models import Candidate, HeatPoint, VideoMeta, Window
from app.pipeline.analyze import (
    analyze,
    dedupe,
    fallback_candidates,
    find_candidates,
    fit_duration,
    heatmap_bonus,
    select_top,
    write_materials,
)
from app.providers.llm import LLMUnavailable
from tests.helpers import make_transcript


class FakeLLM:
    model = "fake"

    def __init__(self, replies):
        self.replies = list(replies)
        self.prompts = []

    def complete_json(self, prompt, schema):
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else '{"candidates": []}'


class RaisingLLM:
    model = "down"

    def complete_json(self, prompt, schema):
        raise LLMUnavailable("down")


def found(start, end, score):
    return json.dumps({"candidates": [{"start_sentence": start, "end_sentence": end, "score": score,
                                       "category": "hot_take", "reason": "alasan", "hook": "hook"}]})


MATERIALS = json.dumps({"hook": "Gaji besar tetap bokek?",
                        **{p: {"title": "Judul", "caption": "Caption", "hashtags": ["#podcast"]}
                           for p in ("tiktok", "reels", "shorts")}})


def cand(t, s, e, score=50, llm=50):
    return Candidate(start_sentence=s, end_sentence=e, start=t.sentences[s].start, end=t.sentences[e].end,
                     llm_score=llm, category="hot_take", reason="r", hook="h", score=score)


@pytest.fixture
def t():
    return make_transcript(40)  # 240 s, 6 s per sentence


@pytest.fixture
def meta():
    return VideoMeta(source_type="youtube", project_id="x", title="T", duration=240.0, video_id="x")


def test_heatmap_bonus_zero_without_heatmap():
    assert heatmap_bonus(10, 70, []) == 0


def test_heatmap_bonus_peak():
    hm = [HeatPoint(start=0, end=60, value=0.4), HeatPoint(start=60, end=120, value=1.0)]
    assert heatmap_bonus(50, 80, hm) == 8
    assert heatmap_bonus(0, 50, hm) == 0
    assert heatmap_bonus(70, 90, [HeatPoint(start=60, end=120, value=0.85)]) == 4


def test_dedupe_time_overlap_keeps_higher_score(t):
    a, b = cand(t, 0, 9, score=80), cand(t, 3, 12, score=60)
    assert dedupe([b, a], t) == [a]


def test_dedupe_keeps_distinct(t):
    a, b = cand(t, 0, 9, score=80), cand(t, 20, 29, score=60)
    assert dedupe([b, a], t) == [a, b]


def test_dedupe_text_similarity(t):
    t.sentences[20] = t.sentences[20].model_copy(update={"text": t.sentences[0].text})
    t.sentences[21] = t.sentences[21].model_copy(update={"text": t.sentences[1].text})
    a, b = cand(t, 0, 1, score=80), cand(t, 20, 21, score=60)
    assert dedupe([a, b], t) == [a]


def test_retry_then_skip_window(t):
    windows = [Window(index=0, start_sentence=0, end_sentence=19), Window(index=1, start_sentence=20, end_sentence=39)]
    llm = FakeLLM(["rusak", "rusak", "rusak", found(0, 8, 70)])
    out = find_candidates(t, windows, llm, 30, 90)
    assert len(llm.prompts) == 4 and "JSON only" in llm.prompts[1]
    assert len(out) == 1
    assert out[0].start_sentence == 20 and out[0].end_sentence == 28  # offset by window start
    assert out[0].start == t.sentences[20].start and out[0].llm_score == 70


def test_prompt_numbers_relative_to_window(t):
    llm = FakeLLM([found(0, 1, 10)])
    find_candidates(t, [Window(index=0, start_sentence=20, end_sentence=39)], llm, 30, 90)
    assert "[0] (02:00)" in llm.prompts[0] and "[19] (03:54)" in llm.prompts[0]


def test_find_candidates_reuses_checkpoints(t, tmp_path):
    w = [Window(index=0, start_sentence=0, end_sentence=39)]
    first = find_candidates(t, w, FakeLLM([found(0, 8, 70)]), 30, 90, checkpoint_dir=tmp_path)
    same_model_down = RaisingLLM()
    same_model_down.model = "fake"
    again = find_candidates(t, w, same_model_down, 30, 90, checkpoint_dir=tmp_path)
    assert again == first
    with pytest.raises(LLMUnavailable):  # another model has its own checkpoints
        find_candidates(t, w, RaisingLLM(), 30, 90, checkpoint_dir=tmp_path)


def test_find_candidates_propagates_unavailable(t):
    with pytest.raises(LLMUnavailable):
        find_candidates(t, [Window(index=0, start_sentence=0, end_sentence=39)], RaisingLLM(), 30, 90)


def test_fit_duration_extends_short_clip(t):
    c = fit_duration(cand(t, 10, 12), t, 30, 90)  # ~17 s
    assert c is not None and 30 <= c.end - c.start <= 90
    assert c.start == t.sentences[c.start_sentence].start and c.end == t.sentences[c.end_sentence].end


def test_fit_duration_rejects_too_long(t):
    assert fit_duration(cand(t, 0, 30), t, 30, 90) is None


def test_select_top_orders_by_final_score(t):
    hm = [HeatPoint(start=0, end=60, value=1.0)]
    a, b = cand(t, 0, 9, llm=70), cand(t, 20, 29, llm=75)
    top = select_top([a, b], hm, 2)
    assert [c.llm_score for c in top] == [70, 75]
    assert top[0].heatmap_bonus == 8 and top[0].score == 78 and top[1].score == 75


def test_fallback_candidates_marked_no_ai(t):
    out = fallback_candidates(t, [], min_sec=15, max_sec=90)
    assert out and all(c.no_ai and c.llm_score is None for c in out)
    assert all(15 <= c.end - c.start <= 90 for c in out)


def test_write_materials(t):
    m = write_materials(cand(t, 0, 9), t, FakeLLM([MATERIALS]))
    assert m.hook == "Gaji besar tetap bokek?" and m.tiktok.hashtags == ["#podcast"]


def test_write_materials_falls_back_on_garbage(t):
    m = write_materials(cand(t, 0, 9), t, FakeLLM(["x", "y"]))
    clip_words = " ".join(s.text for s in t.sentences[0:10]).split()
    assert m.hook == " ".join(clip_words[:8]).rstrip(".") and m.tiktok.caption == ""


def test_analyze_end_to_end(t, meta, tmp_path):
    llm = FakeLLM([found(0, 8, 70), MATERIALS])
    out = analyze(t, meta, llm, max_clips=8, min_sec=30, max_sec=90, checkpoint_dir=tmp_path)
    assert len(out) == 1
    c, m = out[0]
    assert c.llm_score == 70 and not c.no_ai and m.hook == "Gaji besar tetap bokek?"


def test_analyze_falls_back_when_llm_unavailable(t, meta, tmp_path):
    out = analyze(t, meta, RaisingLLM(), 8, 30, 90, tmp_path)
    assert out and all(c.no_ai and c.llm_score is None for c, _ in out)
    assert all(m.hook for _, m in out)


def test_window_checkpoint_keyed_by_duration_limits(t, tmp_path):
    # review I2: cached candidates for 30-90 s must not be reused for 20-40 s
    w = [Window(index=0, start_sentence=0, end_sentence=39)]
    find_candidates(t, w, FakeLLM([found(0, 8, 70)]), 30, 90, checkpoint_dir=tmp_path)
    llm = FakeLLM([found(0, 4, 60)])
    find_candidates(t, w, llm, 20, 40, checkpoint_dir=tmp_path)
    assert len(llm.prompts) == 1


def test_fallback_respects_user_duration_limits(t, meta, tmp_path):
    # review I4: the no-AI fallback must use the user's min/max, not 15/90
    out = analyze(t, meta, RaisingLLM(), 8, 40, 60, tmp_path)
    assert out and all(40 <= c.end - c.start <= 60 for c, _ in out)


def test_fallback_skips_single_sentence_longer_than_max(t):
    t.sentences[5] = t.sentences[5].model_copy(update={"end": t.sentences[5].start + 200})
    out = fallback_candidates(t, [], min_sec=15, max_sec=90)
    assert all(c.end - c.start <= 90 for c in out)


def test_default_materials_are_not_cached(t, meta, tmp_path):
    # review I5: a garbage materials reply once must not stick forever
    analyze(t, meta, FakeLLM([found(0, 8, 70), "rusak", "rusak"]), 8, 30, 90, tmp_path)
    good = FakeLLM([MATERIALS])
    out = analyze(t, meta, good, 8, 30, 90, tmp_path)
    assert out[0][1].hook == "Gaji besar tetap bokek?"


def _ext_clip(s, e, score, hook, captions=True):
    c = {"start_sentence": s, "end_sentence": e, "score": score, "category": "hook", "reason": "alasan", "hook": hook}
    if captions:
        c.update({p: {"title": "Judul dari Claude", "caption": "Caption.", "hashtags": ["#podcast"]}
                  for p in ("tiktok", "reels", "shorts")})
    return c


def test_import_external_builds_candidates_and_materials(t, meta):
    from app.pipeline.analyze import import_external

    raw = {"clips": [_ext_clip(0, 8, 80, "Hook satu"), _ext_clip(20, 28, 60, "Hook dua", captions=False)]}
    out = import_external(t, meta, raw, max_clips=8, min_sec=30, max_sec=90)
    assert [c.llm_score for c, _ in out] == [80, 60]
    assert out[0][1].hook == "Hook satu" and out[0][1].tiktok.title == "Judul dari Claude"
    assert out[1][1].hook == "Hook dua" and out[1][1].tiktok.caption == ""  # missing captions -> empty, not a crash
    assert all(not c.no_ai for c, _ in out)


def test_import_external_accepts_bare_list_and_applies_limits(t, meta):
    from app.pipeline.analyze import import_external

    raw = [_ext_clip(0, 8, 80, "a"), _ext_clip(2, 9, 70, "dup"), _ext_clip(0, 39, 90, "too long")]
    out = import_external(t, meta, raw, max_clips=8, min_sec=30, max_sec=90)
    assert [c.hook for c, _ in out] == ["a"]  # duplicate and over-long clip dropped


def test_import_external_rejects_reply_without_valid_clips(t, meta):
    from app.pipeline.analyze import import_external

    with pytest.raises(ValueError):
        import_external(t, meta, {"clips": [{"start_sentence": "x"}]}, 8, 30, 90)
