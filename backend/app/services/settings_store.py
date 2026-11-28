"""App-wide settings kept in the `appsetting` table, one row per key: `value = {"value": <setting>}`."""
from typing import Any

from sqlmodel import Session

from app.core.config import DEFAULT_RANGE_MAX_HEIGHT, RANGE_MAX_HEIGHTS, Settings, cpu_count, default_encode_threads, get_settings
from app.core.errors import AppError
from app.db.tables import AppSetting
from app.domain.profiles import profile_for

MOMENT_SOURCES = ("claude_manual", "ollama")
COOKIE_BROWSERS = ("", "chrome", "edge", "firefox", "brave", "opera", "vivaldi")


def _defaults(settings: Settings | None = None) -> dict[str, Any]:
    cfg = settings or get_settings()
    return {"moment_source": cfg.default_moment_source, "max_clips": 8, "content_type": "general",
            "cookies_browser": "", "encode_threads": default_encode_threads(),
            "range_max_height": DEFAULT_RANGE_MAX_HEIGHT}


def _validate(key: str, value: Any) -> None:
    if key == "moment_source" and value not in MOMENT_SOURCES:
        raise AppError("VALIDATION", f"Sumber momen harus salah satu dari: {', '.join(MOMENT_SOURCES)}.")
    if key == "max_clips" and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
        raise AppError("VALIDATION", "Jumlah klip maksimal harus bilangan bulat minimal 1.")
    if key == "content_type":
        profile_for(value)  # raises VALIDATION for an unknown content type
    if key == "cookies_browser" and (not isinstance(value, str) or value not in COOKIE_BROWSERS):
        raise AppError("VALIDATION", "Browser cookies harus kosong atau salah satu dari: "
                                     f"{', '.join(b for b in COOKIE_BROWSERS if b)}.")
    if key == "encode_threads":
        cores = cpu_count()
        if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= cores:
            raise AppError("VALIDATION", f"encode_threads harus bilangan bulat antara 1 sampai {cores}.")
    if key == "range_max_height":
        if not isinstance(value, int) or isinstance(value, bool) or value not in RANGE_MAX_HEIGHTS:
            opts = ", ".join(str(h) for h in RANGE_MAX_HEIGHTS)
            raise AppError("VALIDATION", f"range_max_height harus salah satu dari: {opts}.")


def get_app_settings(session: Session, *, settings: Settings | None = None) -> dict[str, Any]:
    out = _defaults(settings)
    for key in out:
        row = session.get(AppSetting, key, populate_existing=True)
        if row is not None and "value" in row.value:
            out[key] = row.value["value"]
    return out


def put_app_settings(session: Session, patch: dict[str, Any], *, settings: Settings | None = None) -> dict[str, Any]:
    """Validate the whole patch first, then save it; returns the merged settings."""
    known = _defaults(settings)
    unknown = sorted(set(patch) - set(known))
    if unknown:
        raise AppError("VALIDATION", f"Pengaturan tidak dikenal: {', '.join(unknown)}.")
    for key, value in patch.items():
        _validate(key, value)
    for key, value in patch.items():
        row = session.get(AppSetting, key) or AppSetting(key=key)
        row.value = {"value": value}  # JSON column is not mutation-tracked: always reassign
        session.add(row)
    session.commit()
    return get_app_settings(session, settings=settings)
