"""Project routes: create (link or upload), list/get/update/delete/retry, Claude-manual moments, files, clips."""
import json
import shutil
import uuid
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, File, Form, Query, Request, Response, UploadFile
from pydantic import BaseModel
from sqlalchemy import Engine

from app.api.deps import dump, file_or_404, get_engine_dep, get_settings_dep
from app.core.config import Settings
from app.core.errors import AppError
from app.services import clips, moments, projects
from app.worker.process import transcript_dir

router = APIRouter(prefix="/projects", tags=["projects"])


class ProjectOptions(BaseModel):
    content_type: str | None = None
    ranges: list | None = None
    focus_prompt: str | None = None
    target_language: str = "id"
    transcript_source: str = "auto"
    moment_source: str | None = None
    max_clips: int | None = None
    min_clip_sec: int | None = None
    max_clip_sec: int | None = None
    style_template_id: str | None = None


class ProjectCreate(ProjectOptions):
    url: str


class ManualClip(BaseModel):
    start: float
    end: float


def _created(engine: Engine, project) -> dict:
    _, job = projects.get_project(engine, project.id)
    return {"project": dump(project), "job": dump(job)}


@router.post("", status_code=201)
def create_project(body: ProjectCreate, engine: Engine = Depends(get_engine_dep),
                   settings: Settings = Depends(get_settings_dep)) -> dict:
    project = projects.create_project(engine, **body.model_dump(), settings=settings)
    return _created(engine, project)


def _parse_ranges(raw: str | None) -> list | None:
    if raw is None or not raw.strip():
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = None
    if not isinstance(value, list):
        raise AppError("VALIDATION", "'ranges' harus berupa daftar JSON.", status=422)
    return value


@router.post("/upload", status_code=201)
def upload_project(file: UploadFile = File(...), content_type: str | None = Form(None),
                   ranges: str | None = Form(None), focus_prompt: str | None = Form(None),
                   target_language: str = Form("id"), transcript_source: str = Form("auto"),
                   moment_source: str | None = Form(None), max_clips: int | None = Form(None),
                   min_clip_sec: int | None = Form(None), max_clip_sec: int | None = Form(None),
                   style_template_id: str | None = Form(None), engine: Engine = Depends(get_engine_dep),
                   settings: Settings = Depends(get_settings_dep)) -> dict:
    """Stream the file to `uploads/_incoming/`; `create_project` moves it into `uploads/<id>/`."""
    options = ProjectOptions(content_type=content_type, ranges=_parse_ranges(ranges), focus_prompt=focus_prompt,
                             target_language=target_language, transcript_source=transcript_source,
                             moment_source=moment_source, max_clips=max_clips, min_clip_sec=min_clip_sec,
                             max_clip_sec=max_clip_sec, style_template_id=style_template_id)
    incoming = Path(settings.data_dir) / "uploads" / "_incoming"
    incoming.mkdir(parents=True, exist_ok=True)
    tmp = incoming / uuid.uuid4().hex
    try:
        with open(tmp, "wb") as out:
            shutil.copyfileobj(file.file, out, length=1024 * 1024)
        project = projects.create_project(engine, upload_path=tmp, original_filename=file.filename,
                                          **options.model_dump(), settings=settings)
    finally:
        tmp.unlink(missing_ok=True)  # already moved on success
    return _created(engine, project)


@router.get("")
def list_projects(engine: Engine = Depends(get_engine_dep)) -> list[dict]:
    return [dump(p) for p in projects.list_projects(engine)]


@router.get("/{project_id}")
def get_project(project_id: str, engine: Engine = Depends(get_engine_dep)) -> dict:
    project, job = projects.get_project(engine, project_id)
    return {"project": dump(project), "active_job": dump(job)}


@router.patch("/{project_id}")
def update_project(project_id: str, patch: dict[str, Any], engine: Engine = Depends(get_engine_dep)) -> dict:
    return dump(projects.update_project(engine, project_id, patch))


@router.post("/{project_id}/retry")
def retry_project(project_id: str, engine: Engine = Depends(get_engine_dep)) -> dict:
    return _created(engine, projects.retry_project(engine, project_id))


@router.delete("/{project_id}", status_code=204)
def delete_project(project_id: str, engine: Engine = Depends(get_engine_dep),
                   settings: Settings = Depends(get_settings_dep)) -> Response:
    projects.delete_project(engine, project_id, settings=settings)
    return Response(status_code=204)


# ---------- Claude-manual moments ----------

@router.get("/{project_id}/moments-export")
def moments_export(project_id: str, engine: Engine = Depends(get_engine_dep),
                   settings: Settings = Depends(get_settings_dep)) -> list[dict]:
    return moments.export_parts(engine, project_id, settings=settings)


@router.post("/{project_id}/moments-import")
async def moments_import(project_id: str, request: Request, part: int = Query(1),
                         engine: Engine = Depends(get_engine_dep),
                         settings: Settings = Depends(get_settings_dep)) -> dict:
    """The body is Claude's reply as-is (plain text or JSON), decoded as UTF-8."""
    try:
        text = (await request.body()).decode("utf-8")
    except UnicodeDecodeError:
        raise AppError("VALIDATION", "Balasan harus berupa teks UTF-8.", status=422) from None
    return moments.import_part(engine, project_id, part, text, settings=settings)


@router.post("/{project_id}/moments-finish")
def moments_finish(project_id: str, allow_missing: bool = Query(False), engine: Engine = Depends(get_engine_dep),
                   settings: Settings = Depends(get_settings_dep)) -> dict:
    return dump(moments.finish_moments(engine, project_id, allow_missing=allow_missing, settings=settings))


# ---------- files ----------

@router.get("/{project_id}/proxy")
def proxy(project_id: str, engine: Engine = Depends(get_engine_dep), settings: Settings = Depends(get_settings_dep)):
    projects.get_project(engine, project_id)  # NOT_FOUND for an unknown project
    path = Path(settings.data_dir) / "projects" / project_id / "proxy.mp4"
    return file_or_404(path, "proxy video", media_type="video/mp4")


@router.get("/{project_id}/transcript")
def transcript(project_id: str, engine: Engine = Depends(get_engine_dep),
               settings: Settings = Depends(get_settings_dep)):
    project, _ = projects.get_project(engine, project_id)
    return file_or_404(transcript_dir(settings, project) / "transcript.json", "transkrip", media_type="application/json")


# ---------- clips of a project ----------

@router.get("/{project_id}/clips")
def list_clips(project_id: str, include_rejected: bool = Query(False),
               engine: Engine = Depends(get_engine_dep)) -> list[dict]:
    return [dump(c) for c in clips.list_clips(engine, project_id, include_rejected=include_rejected)]


@router.post("/{project_id}/clips", status_code=201)
def create_manual_clip(project_id: str, body: ManualClip, engine: Engine = Depends(get_engine_dep),
                       settings: Settings = Depends(get_settings_dep)) -> dict:
    return dump(clips.create_manual(engine, project_id, body.start, body.end, settings=settings))
