"""System routes: health, style schema, content profiles and app settings."""
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy import Engine, text

from app.api.deps import get_engine_dep, get_settings_dep
from app.core.config import REPO_DIR, Settings
from app.db.engine import session_scope
from app.domain.languages import LOW_CONFIDENCE_PROB, load_languages
from app.domain.profiles import load_profiles
from app.services.settings_store import get_app_settings, put_app_settings
from scripts import check_setup

log = logging.getLogger(__name__)

router = APIRouter(prefix="/system", tags=["system"])

SHARED_DIR = REPO_DIR / "shared"


def _whisper_model_cached(model: str) -> bool:
    """True when faster-whisper can load ``model`` without downloading (local folder or Hugging Face cache)."""
    try:
        if Path(model).is_dir():
            return True
        from faster_whisper.utils import _MODELS
        from huggingface_hub import try_to_load_from_cache

        repo = _MODELS.get(model, model if "/" in model else None)
        return repo is not None and isinstance(try_to_load_from_cache(repo, "model.bin"), str)
    except Exception:
        return False


def _db_ok(engine: Engine) -> bool:
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def _tools(settings: Settings) -> dict[str, Any]:
    try:
        return check_setup.collect(settings)
    except Exception:
        log.exception("Health probe failed")
        return {}


@router.get("/health")
def health(engine: Engine = Depends(get_engine_dep), settings: Settings = Depends(get_settings_dep)) -> dict:
    tools = _tools(settings)
    return {
        "ffmpeg": bool(tools.get("ffmpeg", False)),
        "ffmpeg_has_libass": bool(tools.get("ffmpeg_has_libass", False)),
        "ollama": bool(tools.get("ollama", False)),
        "ollama_model": bool(tools.get("ollama_model", False)),
        "whisper_model_cached": _whisper_model_cached(settings.whisper_model),
        "disk_free_gb": tools.get("disk_free_gb", 0.0),
        "data_dir": str(Path(settings.data_dir).resolve()),
        "db": _db_ok(engine),
    }


@router.get("/style-schema")
def style_schema() -> dict:
    def read(name: str) -> Any:
        return json.loads((SHARED_DIR / name).read_text(encoding="utf-8"))

    return {"schema": read("style_schema.json"), "default": read("style_default.json")}


@router.get("/profiles")
def profiles() -> dict:
    return {key: p.model_dump() for key, p in load_profiles().items()}


@router.get("/languages")
def languages() -> dict:
    return {
        "languages": [
            {"code": code, "name_id": lang.name_id, "name_en": lang.name_en, "whisper": lang.whisper}
            for code, lang in load_languages().items()
        ],
        "low_confidence_prob": LOW_CONFIDENCE_PROB,
    }


@router.get("/settings")
def get_settings_route(engine: Engine = Depends(get_engine_dep),
                       settings: Settings = Depends(get_settings_dep)) -> dict:
    with session_scope(engine) as s:
        return get_app_settings(s, settings=settings)


@router.put("/settings")
def put_settings_route(patch: dict[str, Any], engine: Engine = Depends(get_engine_dep),
                       settings: Settings = Depends(get_settings_dep)) -> dict:
    with session_scope(engine) as s:
        return put_app_settings(s, patch, settings=settings)
