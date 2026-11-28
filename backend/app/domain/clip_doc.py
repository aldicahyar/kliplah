"""v2 clip document shapes (spec 4.9, 4.10) and the adapter to the M1 renderer.

``to_clip_plan`` turns a v2 ``Clip`` row plus its project style into the M1 ``ClipPlan`` and
a presets dict shaped like ``shared/caption_presets.json``, so ``render_clip``/``build_ass``
can render it unchanged. Until S4 only single-piece clips, the ``face_crop``/``fit_blur``
layouts and the bundled font are supported; style fields the M1 renderer cannot express
are ignored with one warning.
"""
import logging
from typing import Literal

from pydantic import BaseModel

from app.core.errors import AppError
from app.core.models import Candidate, ClipPlan, Materials, PlatformCaption, SubtitleLine, Word
from app.db.tables import Clip
from app.domain.layouts import LayoutSegment, to_reframe_segments
from app.domain.style import effective_style
from app.domain.timeline import Piece, output_duration

log = logging.getLogger(__name__)

CANVAS = {"width": 1080, "height": 1920}
FONT_FILES = {"Montserrat ExtraBold": "Montserrat-ExtraBold.ttf"}
HOOK_MAX_CHARS_PER_LINE = 18


class WordV2(BaseModel):
    w: str
    start: float
    end: float
    prob: float | None = None
    approx: bool = False


class SubtitleLineV2(BaseModel):
    id: int
    start: float
    end: float
    text: str
    edited: bool = False
    source: Literal["whisper", "youtube_manual", "youtube_auto", "translated"] = "whisper"
    text_original: str | None = None
    speaker_emoji: str | None = None
    words: list[WordV2] = []


class Overlay(BaseModel):
    id: str
    type: Literal["label", "arrow", "emoji"]
    start: float
    end: float
    x: float  # normalised to the 1080x1920 canvas
    y: float
    rotation: float = 0.0
    text: str | None = None
    emoji: str | None = None
    style: dict = {}


def subtitles_from_m1(lines: list[SubtitleLine]) -> list[dict]:
    return [SubtitleLineV2(id=line.id, start=line.start, end=line.end, text=line.text, edited=line.edited,
                           source="whisper",
                           words=[WordV2(w=w.w, start=w.start, end=w.end, prob=w.prob, approx=w.approx)
                                  for w in line.words]).model_dump()
            for line in lines]


def _ignored_fields(style: dict, clip: Clip) -> list[str]:
    sub, hook = style["subtitle"], style["hook"]
    checks = {
        "subtitle.position.anchor": sub["position"]["anchor"] != "bottom",
        "subtitle.box": sub["box"]["enabled"],
        "subtitle.speaker_emoji": sub["speaker_emoji"],
        "hook.position.anchor": hook["position"]["anchor"] != "top",
        "hook.box": hook["box"]["enabled"],
        "logo": style["logo"] is not None and style["logo"]["enabled"],
        "background.type": style["background"]["type"] != "blur",
        "overlays": bool(clip.overlays),
    }
    return [path for path, ignored in checks.items() if ignored]


def _presets(style: dict) -> tuple[str, dict]:
    family = style["font_family"]
    if family not in FONT_FILES:
        raise AppError("VALIDATION", f"Font {family} baru tersedia di S4")
    sub, hook = style["subtitle"], style["hook"]
    highlight = sub["mode"] == "highlight_word"
    name = "sorot_kata" if highlight else "klasik"
    return name, {
        "canvas": dict(CANVAS),
        "font": {"family": family, "file": FONT_FILES[family]},
        "hook": {"size": hook["size"], "color": hook["color"], "outline": hook["outline_color"],
                 "outline_width": hook["outline_width"], "margin_top": hook["position"]["offset_y"],
                 "max_chars_per_line": HOOK_MAX_CHARS_PER_LINE},
        "presets": {name: {"size": sub["size"], "color": sub["color"],
                           "highlight": sub["highlight_color"] if highlight else None,
                           "outline": sub["outline_color"], "outline_width": sub["outline_width"],
                           "margin_bottom": sub["position"]["offset_y"]}},
    }


def _m1_line(line: SubtitleLineV2, uppercase: bool, language_mode: str) -> SubtitleLine:
    def up(s: str) -> str:
        return s.upper() if uppercase else s

    orig_src = line.text_original if (line.text_original and line.text_original.strip()) else None
    if language_mode == "original":
        shown = up(orig_src) if orig_src else up(line.text)
        orig = None
    elif language_mode == "bilingual":
        shown = up(line.text)
        orig = up(orig_src) if orig_src else None
    else:
        shown = up(line.text)
        orig = None

    # Highlight-word mode relies on exact per-word timing, so approx, translated, or non-target modes fall back to whole-line.
    has_approx = line.source == "translated" or any(w.approx for w in line.words)
    words = (
        []
        if (has_approx or language_mode != "target")
        else [Word(w=up(w.w), start=w.start, end=w.end, approx=w.approx, prob=w.prob) for w in line.words]
    )
    return SubtitleLine(
        id=line.id, start=line.start, end=line.end, text=shown, words=words, edited=line.edited, original=orig
    )


def _m1_lines(subtitles: list[dict], uppercase: bool, language_mode: str = "target") -> list[SubtitleLine]:
    return [_m1_line(SubtitleLineV2.model_validate(s), uppercase, language_mode) for s in subtitles]


def _caption(captions: dict, key: str) -> PlatformCaption:
    return PlatformCaption.model_validate({"title": "", "caption": "", "hashtags": [], **captions.get(key, {})})


def to_clip_plan(clip: Clip, project_style: dict) -> tuple[ClipPlan, dict]:
    pieces = [Piece.model_validate(p) for p in clip.pieces]
    if not pieces:
        raise AppError("VALIDATION", "Klip tidak punya potongan.")
    if len(pieces) > 1:
        raise AppError("VALIDATION", "Render multi-piece tersedia di S4")
    style = effective_style(project_style, clip.style_override)
    preset_name, presets = _presets(style)
    segments = to_reframe_segments([LayoutSegment.model_validate(s) for s in clip.layout_segments])

    ignored = _ignored_fields(style, clip)
    if ignored:
        log.warning("Klip %s: field gaya diabaikan oleh renderer S1: %s", clip.id, ", ".join(ignored))

    sub, hook = style["subtitle"], style["hook"]
    hook_text = clip.hook_text if hook["enabled"] else ""
    hook_duration = output_duration(pieces) if hook["duration"] == "full" else float(hook["duration"])
    subtitles = _m1_lines(clip.subtitles, sub["uppercase"], sub["language_mode"]) if sub["enabled"] else []

    piece = pieces[0]
    candidate = Candidate(start_sentence=0, end_sentence=0, start=piece.start, end=piece.end,
                          llm_score=clip.llm_score, category=clip.category, reason=clip.reason,
                          hook=hook_text, score=clip.score)
    materials = Materials(hook=hook_text, tiktok=_caption(clip.captions, "tiktok"),
                          reels=_caption(clip.captions, "reels"), shorts=_caption(clip.captions, "shorts"))
    plan = ClipPlan(id=clip.id, rank=clip.rank, candidate=candidate, materials=materials, subtitles=subtitles,
                    reframe_segments=segments,
                    layout="face_crop" if any(s.mode == "track" for s in segments) else "fit_blur",
                    caption_preset=preset_name, hook_duration_sec=hook_duration)
    return plan, presets
