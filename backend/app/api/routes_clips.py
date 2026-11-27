"""Clip routes: get/patch (with `effective_style`), status actions, bulk, files and save-as-template."""
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import Engine

from app.api.deps import dump, file_or_404, get_engine_dep, get_settings_dep
from app.core.config import Settings
from app.db.engine import session_scope
from app.db.tables import Clip
from app.domain.style import effective_style
from app.services import clips, styles
from app.services._common import domain_errors

router = APIRouter(prefix="/clips", tags=["clips"])


class BulkAction(BaseModel):
    ids: list[str]
    action: str


class TemplateName(BaseModel):
    name: str


def _with_style(engine: Engine, clip: Clip) -> dict:
    with session_scope(engine) as s:
        base = styles.clip_base_style(s, clip)
    with domain_errors("style"):
        eff = effective_style(base, clip.style_override)
    return {**dump(clip), "effective_style": eff}


@router.post("/bulk")
def bulk(body: BulkAction, engine: Engine = Depends(get_engine_dep)) -> dict:
    return clips.bulk(engine, body.ids, body.action)


@router.get("/{clip_id}")
def get_clip(clip_id: str, engine: Engine = Depends(get_engine_dep)) -> dict:
    return _with_style(engine, clips.get_clip(engine, clip_id))


@router.patch("/{clip_id}")
def patch_clip(clip_id: str, patch: dict[str, Any], engine: Engine = Depends(get_engine_dep),
               settings: Settings = Depends(get_settings_dep)) -> dict:
    return _with_style(engine, clips.patch_clip(engine, clip_id, patch, settings=settings))


@router.post("/{clip_id}/approve")
def approve(clip_id: str, engine: Engine = Depends(get_engine_dep)) -> dict:
    return dump(clips.approve(engine, clip_id))


@router.post("/{clip_id}/reject")
def reject(clip_id: str, engine: Engine = Depends(get_engine_dep)) -> dict:
    return dump(clips.reject(engine, clip_id))


@router.post("/{clip_id}/restore")
def restore(clip_id: str, engine: Engine = Depends(get_engine_dep)) -> dict:
    return dump(clips.restore(engine, clip_id))


@router.post("/{clip_id}/save-as-template", status_code=201)
def save_as_template(clip_id: str, body: TemplateName, engine: Engine = Depends(get_engine_dep)) -> dict:
    return dump(styles.save_clip_as_template(engine, clip_id, body.name))


# ---------- files ----------

@router.get("/{clip_id}/thumbnail")
def thumbnail(clip_id: str, engine: Engine = Depends(get_engine_dep), settings: Settings = Depends(get_settings_dep)):
    clip = clips.get_clip(engine, clip_id)
    path = None
    if clip.project_id is not None:
        path = Path(settings.data_dir) / "projects" / clip.project_id / "clips" / clip.id / "thumb.jpg"
    return file_or_404(path, "thumbnail", media_type="image/jpeg")


@router.get("/{clip_id}/download")
def download(clip_id: str, engine: Engine = Depends(get_engine_dep)):
    clip = clips.get_clip(engine, clip_id)
    path = Path(clip.output_path) if clip.output_path else None
    return file_or_404(path, "video klip", media_type="video/mp4", filename=f"{clip.rank}_{clip.id}.mp4")


@router.get("/{clip_id}/captions")
def captions(clip_id: str, engine: Engine = Depends(get_engine_dep)):
    clip = clips.get_clip(engine, clip_id)
    path = Path(clip.output_path).parent / "captions.txt" if clip.output_path else None
    return file_or_404(path, "caption", media_type="text/plain; charset=utf-8")
