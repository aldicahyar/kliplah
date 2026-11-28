"""Language catalog, normalization, and detection aliases (S3)."""
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from app.core.errors import AppError

LANGUAGES_PATH = Path(__file__).resolve().parents[1] / "data" / "languages.json"

LANGUAGE_ALIASES: dict[str, frozenset[str]] = {"id": frozenset({"ms"})}
LOW_CONFIDENCE_PROB = 0.6

_LANG_RE = re.compile(r"^[a-z]{2,3}(?:[-_][a-z0-9]+)*$")


class Language(BaseModel):
    code: str
    name_id: str
    name_en: str
    whisper: bool = True
    weak_local_translation: bool = False


@lru_cache
def _load() -> dict[str, Language]:
    raw = json.loads(LANGUAGES_PATH.read_text(encoding="utf-8"))
    return {code: Language(code=code, **entry) for code, entry in raw.items()}


def load_languages() -> dict[str, Language]:
    """Return a deep copy of all languages from the catalog."""
    return {code: lang.model_copy(deep=True) for code, lang in _load().items()}


def normalize_language(code: str | None) -> str | None:
    """Normalize a language code (strip region and ``-orig``); ``None``/empty/``auto`` -> ``None``."""
    if code is None:
        return None
    cleaned = code.strip().lower()
    if not cleaned or cleaned == "auto":
        return None
    cleaned = cleaned.removesuffix("-orig").removesuffix("_orig")
    if not _LANG_RE.match(cleaned):
        raise AppError("VALIDATION", f"Kode bahasa '{code}' tidak dikenal.", status=422)
    base = re.split(r"[-_]", cleaned, maxsplit=1)[0]
    if base not in _load():
        raise AppError("VALIDATION", f"Kode bahasa '{code}' tidak dikenal.", status=422)
    return base


def _lookup(code: str) -> Language | None:
    try:
        norm = normalize_language(code)
    except AppError:
        return None
    return _load().get(norm) if norm else None


def language_name(code: str, lang: Literal["en", "id"] = "en") -> str:
    """Return display name for ``code`` in ``lang`` (``en`` or ``id``), falling back to ``code``."""
    if lang not in ("en", "id"):
        raise AppError("VALIDATION", f"Bahasa tampilan '{lang}' tidak dikenal.", status=422)
    entry = _lookup(code)
    if entry is None:
        return code
    return entry.name_id if lang == "id" else entry.name_en


def is_weak_local_translation(code: str) -> bool:
    """True when local LLM translation from ``code`` to Indonesian is marked weak."""
    entry = _lookup(code)
    return entry.weak_local_translation if entry is not None else False
