"""Content profiles (spec 4.1), stored as data in ``app/data/content_profiles.json``."""
import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel

from app.core.errors import AppError

PROFILES_PATH = Path(__file__).resolve().parents[1] / "data" / "content_profiles.json"


class ContentProfile(BaseModel):
    key: str
    label: str
    min_sec: int
    max_sec: int
    default_layouts: list[str]
    criteria: list[str]
    signals: list[str]


@lru_cache
def _load() -> dict[str, ContentProfile]:
    raw = json.loads(PROFILES_PATH.read_text(encoding="utf-8"))
    return {key: ContentProfile.model_validate(value) for key, value in raw.items()}


def load_profiles() -> dict[str, ContentProfile]:
    return {key: p.model_copy(deep=True) for key, p in _load().items()}


def profile_for(key: str) -> ContentProfile:
    profile = _load().get(key)
    if profile is None:
        raise AppError("VALIDATION", f"Jenis konten '{key}' tidak dikenal.")
    return profile.model_copy(deep=True)
