"""Turning small-model output into trustworthy data, plus prompt templates."""
import json
import re
from pathlib import Path

from app.core.models import CATEGORIES

PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"
_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

_CAPTION = {"type": "object", "properties": {"title": {"type": "string"}, "caption": {"type": "string"},
                                             "hashtags": {"type": "array", "items": {"type": "string"}}},
            "required": ["title", "caption", "hashtags"]}

FIND_MOMENTS_SCHEMA = {
    "type": "object",
    "properties": {"candidates": {"type": "array", "items": {
        "type": "object",
        "properties": {"start_sentence": {"type": "integer"}, "end_sentence": {"type": "integer"},
                       "score": {"type": "integer"}, "category": {"type": "string", "enum": list(CATEGORIES)},
                       "reason": {"type": "string"}, "hook": {"type": "string"}},
        "required": ["start_sentence", "end_sentence", "score", "category", "reason", "hook"]}}},
    "required": ["candidates"],
}

MATERIALS_SCHEMA = {
    "type": "object",
    "properties": {"hook": {"type": "string"}, "tiktok": _CAPTION, "reels": _CAPTION, "shorts": _CAPTION},
    "required": ["hook", "tiktok", "reels", "shorts"],
}


def parse_llm_json(text: str) -> dict | None:
    for attempt in (text, _FENCE.sub("", text.strip())):
        try:
            v = json.loads(attempt)
            return v if isinstance(v, dict) else None
        except (json.JSONDecodeError, TypeError):
            pass
    a, b = text.find("{"), text.rfind("}")
    if a != -1 and b > a:
        try:
            v = json.loads(text[a: b + 1])
            return v if isinstance(v, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def coerce_candidate(raw: dict, n_sentences: int, offset: int) -> dict | None:
    try:
        s = int(raw["start_sentence"]) + offset
        e = int(raw["end_sentence"]) + offset
        score = int(float(raw.get("score", 0)))
    except (KeyError, TypeError, ValueError):
        return None
    s = min(max(s, 0), n_sentences - 1)
    e = min(max(e, 0), n_sentences - 1)
    if e < s:
        return None
    cat = raw.get("category")
    return {"start_sentence": s, "end_sentence": e, "score": min(max(score, 0), 100),
            "category": cat if cat in CATEGORIES else None,
            "reason": str(raw.get("reason") or "").strip(), "hook": str(raw.get("hook") or "").strip()}


def render_prompt(name: str, **values) -> str:
    text = (PROMPTS_DIR / f"{name}.md").read_text(encoding="utf-8")
    defaults = {"criteria": "", "focus": "", "language_note": "", "lines_field": ""}
    for k, v in {**defaults, **values}.items():
        text = text.replace("{{" + k + "}}", str(v))
    return text
