import logging

import pytest

from app.core.errors import AppError
from app.core.models import SubtitleLine, Word
from app.db.tables import Clip
from app.domain.clip_doc import Overlay, SubtitleLineV2, WordV2, _m1_line, subtitles_from_m1, to_clip_plan
from app.domain.style import deep_merge, default_style
from app.domain.timeline import Piece, output_duration
from app.pipeline.subtitles import build_ass, load_presets

M1_LINES = [
    SubtitleLine(id=1, start=0.2, end=1.4, text="halo semua apa kabar",
                 words=[Word(w="halo", start=0.2, end=0.5), Word(w="semua", start=0.5, end=0.8),
                        Word(w="apa", start=0.8, end=1.0), Word(w="kabar", start=1.0, end=1.4)]),
    SubtitleLine(id=2, start=1.6, end=2.4, text="baik sekali", edited=True,
                 words=[Word(w="baik", start=1.6, end=1.9), Word(w="sekali", start=1.9, end=2.4)]),
]

# Style that mimics the M1 caption presets (shared/caption_presets.json).
M1_LIKE = {"subtitle": {"size": 72, "outline_width": 6, "position": {"anchor": "bottom", "offset_y": 420}},
           "hook": {"size": 84, "outline_width": 7, "duration": 3.0, "position": {"anchor": "top", "offset_y": 260}}}


def make_clip(**kw) -> Clip:
    base = dict(id="c1", project_id="p1", origin="ai", rank=1, score=77, llm_score=8, category="hot_take",
                reason="opini tajam", pieces=[{"project_id": "p1", "start": 10.0, "end": 40.0}],
                layout_segments=[{"start": 0, "end": 12, "layout": "face_crop", "params": {"x": 0.4, "zoom": 1.0}},
                                 {"start": 12, "end": 30, "layout": "fit_blur"}],
                hook_text="Ini rahasianya", subtitles=subtitles_from_m1(M1_LINES),
                captions={"tiktok": {"title": "T", "caption": "cap tt", "hashtags": ["#a"]},
                          "reels": {"title": "R", "caption": "cap r", "hashtags": ["#b"]},
                          "shorts": {"title": "S", "caption": "cap s", "hashtags": ["#c"]}})
    base.update(kw)
    return Clip(**base)


def test_subtitles_from_m1_shape():
    subs = subtitles_from_m1(M1_LINES)
    assert subs[0]["id"] == 1 and subs[0]["source"] == "whisper" and subs[0]["edited"] is False
    assert subs[0]["text_original"] is None and subs[0]["speaker_emoji"] is None
    assert subs[0]["words"][1] == {"w": "semua", "start": 0.5, "end": 0.8, "prob": None, "approx": False}
    assert subs[1]["edited"] is True
    # round-trips through the v2 model
    assert SubtitleLineV2.model_validate(subs[0]).words[0].w == "halo"


def test_subtitles_from_m1_copies_prob_and_approx():
    line = SubtitleLine(
        id=1, start=0.2, end=1.0, text="aku pergi",
        words=[Word(w="aku", start=0.2, end=0.5, prob=0.35, approx=False),
               Word(w="pergi", start=0.5, end=1.0, prob=None, approx=True)],
    )
    subs = subtitles_from_m1([line])
    assert subs[0]["words"][0]["prob"] == 0.35 and subs[0]["words"][0]["approx"] is False
    assert subs[0]["words"][1]["prob"] is None and subs[0]["words"][1]["approx"] is True


def test_overlay_model():
    o = Overlay(id="o1", type="label", start=1.0, end=2.0, x=0.5, y=0.3, text="Wah")
    assert o.rotation == 0 and o.emoji is None and o.style == {}
    with pytest.raises(ValueError):
        Overlay(id="o2", type="sticker", start=0, end=1, x=0, y=0)


def test_to_clip_plan_basic_fields():
    plan, presets = to_clip_plan(make_clip(), default_style())
    c = plan.candidate
    assert (plan.id, plan.rank) == ("c1", 1)
    assert (c.start, c.end, c.llm_score, c.category, c.reason, c.hook, c.score) == \
        (10.0, 40.0, 8, "hot_take", "opini tajam", "Ini rahasianya", 77)
    assert plan.materials.hook == "Ini rahasianya"
    assert plan.materials.reels.caption == "cap r" and plan.materials.shorts.hashtags == ["#c"]
    assert [s.mode for s in plan.reframe_segments] == ["track", "wide"]
    assert plan.reframe_segments[0].x == 0.4
    assert plan.layout == "face_crop"
    assert plan.hook_duration_sec == 3.0
    assert plan.caption_preset == "klasik"
    assert presets["presets"]["klasik"]["highlight"] is None
    assert presets["presets"]["klasik"]["size"] == 104
    assert presets["font"] == {"family": "Montserrat ExtraBold", "file": "Montserrat-ExtraBold.ttf"}
    assert [l.text for l in plan.subtitles] == ["halo semua apa kabar", "baik sekali"]
    assert plan.subtitles[1].edited is True


def test_to_clip_plan_uses_effective_style():
    clip = make_clip(style_override={"subtitle": {"size": 80, "mode": "highlight_word"}})
    plan, presets = to_clip_plan(clip, default_style())
    name = plan.caption_preset
    assert name == "sorot_kata"
    assert presets["presets"][name]["size"] == 80
    assert presets["presets"][name]["highlight"] == "#FFE14D"


def test_to_clip_plan_disabled_subtitle_and_full_hook():
    clip = make_clip(style_override={"subtitle": {"enabled": False}, "hook": {"duration": "full"}})
    plan, _ = to_clip_plan(clip, default_style())
    assert plan.subtitles == []
    assert plan.hook_duration_sec == output_duration([Piece(**p) for p in clip.pieces]) == 30.0


def test_to_clip_plan_disabled_hook_empties_hook_text():
    plan, presets = to_clip_plan(make_clip(style_override={"hook": {"enabled": False}}), default_style())
    assert plan.materials.hook == "" and plan.candidate.hook == ""
    ass = build_ass(plan.subtitles, plan.materials.hook or plan.candidate.hook, plan.hook_duration_sec,
                    plan.caption_preset, presets)
    assert ",Hook,," not in ass


def test_to_clip_plan_uppercase():
    plan, _ = to_clip_plan(make_clip(style_override={"subtitle": {"uppercase": True}}), default_style())
    assert plan.subtitles[0].text == "HALO SEMUA APA KABAR"
    assert plan.subtitles[0].words[0].w == "HALO"


def test_to_clip_plan_rejects_multi_piece():
    clip = make_clip(pieces=[{"project_id": "p1", "start": 0, "end": 5}, {"project_id": "p1", "start": 9, "end": 12}])
    with pytest.raises(AppError) as exc:
        to_clip_plan(clip, default_style())
    assert exc.value.code == "VALIDATION" and "S4" in exc.value.message


def test_to_clip_plan_rejects_s4_layout_and_font():
    with pytest.raises(AppError) as exc:
        to_clip_plan(make_clip(layout_segments=[{"start": 0, "end": 30, "layout": "stack"}]), default_style())
    assert exc.value.code == "VALIDATION"
    with pytest.raises(AppError) as exc:
        to_clip_plan(make_clip(style_override={"font_family": "Inter Bold"}), default_style())
    assert exc.value.code == "VALIDATION" and "Inter Bold" in exc.value.message and "S4" in exc.value.message


def test_to_clip_plan_warns_about_ignored_fields(caplog):
    override = {"subtitle": {"box": {"enabled": True}, "position": {"anchor": "middle"}},
                "background": {"type": "solid"}}
    with caplog.at_level(logging.WARNING, logger="app.domain.clip_doc"):
        to_clip_plan(make_clip(style_override=override), default_style())
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 1
    for path in ("subtitle.box", "subtitle.position.anchor", "background.type"):
        assert path in msgs[0]


def test_to_clip_plan_no_warning_for_default_style(caplog):
    with caplog.at_level(logging.WARNING, logger="app.domain.clip_doc"):
        to_clip_plan(make_clip(), default_style())
    assert caplog.records == []


@pytest.mark.parametrize("mode,m1_preset", [("line", "klasik"), ("highlight_word", "sorot_kata")])
def test_render_adapter_matches_m1_ass(mode, m1_preset):
    project_style = deep_merge(default_style(), M1_LIKE)
    clip = make_clip(style_override={"subtitle": {"mode": mode}})
    plan, presets = to_clip_plan(clip, project_style)
    adapter_ass = build_ass(plan.subtitles, plan.materials.hook, plan.hook_duration_sec, plan.caption_preset, presets)
    m1_ass = build_ass(M1_LINES, "Ini rahasianya", 3.0, m1_preset, load_presets())
    assert adapter_ass == m1_ass


def _translated_subtitles() -> list[dict]:
    return [
        {
            "id": 1,
            "start": 0.2,
            "end": 1.4,
            "text": "halo semua",
            "edited": False,
            "source": "translated",
            "text_original": "hello everyone",
            "speaker_emoji": None,
            "words": [
                {"w": "halo", "start": 0.2, "end": 0.8, "prob": None, "approx": True},
                {"w": "semua", "start": 0.8, "end": 1.4, "prob": None, "approx": True},
            ],
        },
        {
            "id": 2,
            "start": 1.6,
            "end": 2.4,
            "text": "tanpa terjemahan",
            "edited": False,
            "source": "whisper",
            "text_original": None,
            "speaker_emoji": None,
            "words": [
                {"w": "tanpa", "start": 1.6, "end": 2.0, "prob": 0.9, "approx": False},
                {"w": "terjemahan", "start": 2.0, "end": 2.4, "prob": 0.88, "approx": False},
            ],
        },
    ]


@pytest.mark.parametrize("mode,m1_preset", [("line", "klasik"), ("highlight_word", "sorot_kata")])
def test_target_mode_ass_matches_m1(mode, m1_preset):
    project_style = deep_merge(default_style(), M1_LIKE)
    clip = make_clip(style_override={"subtitle": {"mode": mode, "language_mode": "target"}})
    plan, presets = to_clip_plan(clip, project_style)
    adapter_ass = build_ass(plan.subtitles, plan.materials.hook, plan.hook_duration_sec, plan.caption_preset, presets)
    m1_ass = build_ass(M1_LINES, "Ini rahasianya", 3.0, m1_preset, load_presets())
    assert adapter_ass == m1_ass


def test_bilingual_mode_puts_original_above_translation():
    clip = make_clip(
        subtitles=_translated_subtitles(),
        style_override={"subtitle": {"language_mode": "bilingual", "mode": "highlight_word"}},
    )
    plan, presets = to_clip_plan(clip, default_style())
    assert plan.subtitles[0].text == "halo semua"
    assert plan.subtitles[0].original == "hello everyone"
    assert plan.subtitles[0].words == []
    assert plan.subtitles[1].text == "tanpa terjemahan"
    assert plan.subtitles[1].original is None
    assert plan.subtitles[1].words == []

    ass = build_ass(plan.subtitles, plan.materials.hook, plan.hook_duration_sec, plan.caption_preset, presets)
    subs = [l for l in ass.splitlines() if l.startswith("Dialogue") and ",Sub," in l]
    assert len(subs) == 2
    assert r"{\fs75}hello everyone{\r}\Nhalo semua" in subs[0]
    assert subs[0].index("hello everyone") < subs[0].index(r"\N") < subs[0].index("halo semua")
    assert subs[1].endswith(",tanpa terjemahan")

    clip_upper = make_clip(
        subtitles=_translated_subtitles(),
        style_override={"subtitle": {"language_mode": "bilingual", "uppercase": True}},
    )
    plan_upper, presets_upper = to_clip_plan(clip_upper, default_style())
    assert plan_upper.subtitles[0].text == "HALO SEMUA"
    assert plan_upper.subtitles[0].original == "HELLO EVERYONE"
    ass_upper = build_ass(
        plan_upper.subtitles,
        plan_upper.materials.hook,
        plan_upper.hook_duration_sec,
        plan_upper.caption_preset,
        presets_upper,
    )
    assert r"{\fs75}HELLO EVERYONE{\r}\NHALO SEMUA" in ass_upper


def test_original_mode_uses_text_original_with_fallback_to_text():
    subs_data = _translated_subtitles() + [
        {
            "id": 3,
            "start": 2.6,
            "end": 3.2,
            "text": "teks kosong",
            "edited": False,
            "source": "translated",
            "text_original": "   ",
            "speaker_emoji": None,
            "words": [{"w": "teks", "start": 2.6, "end": 2.9, "approx": True},
                      {"w": "kosong", "start": 2.9, "end": 3.2, "approx": True}],
        }
    ]
    clip = make_clip(
        subtitles=subs_data,
        style_override={"subtitle": {"language_mode": "original", "mode": "highlight_word"}},
    )
    plan, _ = to_clip_plan(clip, default_style())
    assert [l.text for l in plan.subtitles] == ["hello everyone", "tanpa terjemahan", "teks kosong"]
    assert all(l.original is None for l in plan.subtitles)
    assert all(l.words == [] for l in plan.subtitles)

    clip_upper = make_clip(
        subtitles=subs_data,
        style_override={"subtitle": {"language_mode": "original", "uppercase": True}},
    )
    plan_upper, _ = to_clip_plan(clip_upper, default_style())
    assert [l.text for l in plan_upper.subtitles] == ["HELLO EVERYONE", "TANPA TERJEMAHAN", "TEKS KOSONG"]


def test_highlight_mode_falls_back_to_whole_line_for_approx_words():
    clip = make_clip(
        subtitles=_translated_subtitles(),
        style_override={"subtitle": {"mode": "highlight_word", "language_mode": "target"}},
    )
    plan, presets = to_clip_plan(clip, default_style())
    assert plan.subtitles[0].words == []
    assert [w.w for w in plan.subtitles[1].words] == ["tanpa", "terjemahan"]

    ass = build_ass(plan.subtitles, plan.materials.hook, plan.hook_duration_sec, plan.caption_preset, presets)
    subs = [l for l in ass.splitlines() if l.startswith("Dialogue") and ",Sub," in l]
    assert len(subs) == 3  # 1 whole line for approx line + 2 per-word highlights for non-approx line
    assert subs[0].endswith(",halo semua")
    assert r"{\c&H004DE1FF&}" not in subs[0]
    assert r"{\c&H004DE1FF&}tanpa{\r} terjemahan" in subs[1]
    assert r"tanpa {\c&H004DE1FF&}terjemahan{\r}" in subs[2]


@pytest.mark.parametrize("lang_mode", ["bilingual", "original"])
def test_language_mode_no_longer_warned_as_ignored(caplog, lang_mode):
    with caplog.at_level(logging.WARNING, logger="app.domain.clip_doc"):
        to_clip_plan(make_clip(style_override={"subtitle": {"language_mode": lang_mode}}), default_style())
    assert caplog.records == []


def test_highlight_mode_falls_back_for_translated_source_even_when_words_not_approx():
    line = SubtitleLineV2(
        id=1,
        start=0.2,
        end=1.4,
        text="halo semua",
        source="translated",
        text_original="hello everyone",
        words=[
            WordV2(w="halo", start=0.2, end=0.8, approx=False),
            WordV2(w="semua", start=0.8, end=1.4, approx=False),
        ],
    )
    m1 = _m1_line(line, uppercase=False, language_mode="target")
    assert m1.words == []
