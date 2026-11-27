"""Project service: create (YouTube link or upload), list, get, delete, retry and update."""
import copy
import shutil
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import Engine
from sqlmodel import Session, col, select

from app.core.config import Settings, get_settings
from app.db.engine import session_scope
from app.db.tables import Clip, Job, Project, StyleTemplate, utcnow
from app.domain.profiles import profile_for
from app.pipeline.ingest import IngestError, parse_youtube_id
from app.services import styles
from app.services._common import domain_errors, err, fetch
from app.services.settings_store import MOMENT_SOURCES, get_app_settings
from app.worker import queue

UPDATABLE = ("style", "facecam_box", "keep_source", "content_type")
FACECAM_LAYOUTS = ("facecam_gameplay", "facecam_full")


def _content_profile(content_type: str):
    with domain_errors("content_type"):
        if not isinstance(content_type, str):
            raise err("VALIDATION", "Jenis konten harus berupa teks.")
        return profile_for(content_type)


def _check_positive_int(name: str, value: Any) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise err("VALIDATION", f"'{name}' harus bilangan bulat minimal 1.")


def _upload_filename(upload_path: Path, original_filename: str | None) -> str:
    name = Path(original_filename or upload_path.name).name  # no directories from the client
    if not name or name in (".", ".."):
        raise err("VALIDATION", "Nama file unggahan tidak valid.")
    return name


def create_project(engine: Engine, *, url: str | None = None, upload_path: Path | None = None,
                   original_filename: str | None = None, content_type: str | None = None, ranges: list | None = None,
                   focus_prompt: str | None = None, target_language: str = "id", transcript_source: str = "auto",
                   moment_source: str | None = None, max_clips: int | None = None, min_clip_sec: int | None = None,
                   max_clip_sec: int | None = None, style_template_id: str | None = None,
                   settings: Settings | None = None) -> Project:
    """Create a `queued` project and enqueue its `process` job. An upload is moved into `uploads/<id>/`."""
    settings = settings or get_settings()
    if (url is None) == (upload_path is None):
        raise err("VALIDATION", "Isi tepat satu sumber: link YouTube atau file unggahan.")
    video_id = filename = None
    if url is not None:
        try:
            video_id = parse_youtube_id(url)
        except IngestError as e:
            raise err("INVALID_URL", str(e)) from e
    else:
        upload_path = Path(upload_path)
        if not upload_path.is_file():
            raise err("VALIDATION", "File unggahan tidak ditemukan.")
        filename = _upload_filename(upload_path, original_filename)

    if content_type is not None:
        _content_profile(content_type)
    if min_clip_sec is not None:
        _check_positive_int("min_clip_sec", min_clip_sec)
    if max_clip_sec is not None:
        _check_positive_int("max_clip_sec", max_clip_sec)
    if min_clip_sec is not None and max_clip_sec is not None and max_clip_sec < min_clip_sec:
        raise err("VALIDATION", "Durasi maksimal klip tidak boleh lebih kecil dari durasi minimal.")
    if moment_source is not None and moment_source not in MOMENT_SOURCES:
        raise err("VALIDATION", f"Sumber momen harus salah satu dari: {', '.join(MOMENT_SOURCES)}.")
    if max_clips is not None:
        _check_positive_int("max_clips", max_clips)

    default = styles.ensure_default_template(engine)
    project_id = uuid.uuid4().hex
    dest_dir = Path(settings.data_dir) / "uploads" / project_id
    with session_scope(engine) as s:
        template = default if style_template_id is None else fetch(s, StyleTemplate, style_template_id, "Template")
        app = get_app_settings(s, settings=settings)
        resolved_type = content_type if content_type is not None else app["content_type"]
        profile = _content_profile(resolved_type)
        min_sec = profile.min_sec if min_clip_sec is None else min_clip_sec
        max_sec = profile.max_sec if max_clip_sec is None else max_clip_sec
        _check_positive_int("min_clip_sec", min_sec)
        _check_positive_int("max_clip_sec", max_sec)
        if max_sec < min_sec:
            raise err("VALIDATION", "Durasi maksimal klip tidak boleh lebih kecil dari durasi minimal.")
        project = Project(
            id=project_id, source_type="youtube" if url is not None else "upload",
            youtube_url=url.strip() if url is not None else None, video_id=video_id, original_filename=filename,
            content_type=resolved_type, ranges=copy.deepcopy(ranges), focus_prompt=focus_prompt,
            target_language=target_language, transcript_source=transcript_source,
            moment_source=moment_source or app["moment_source"],
            max_clips=max_clips if max_clips is not None else app["max_clips"],
            min_clip_sec=min_sec, max_clip_sec=max_sec, style=copy.deepcopy(template.style),
            style_template_id=template.id, whisper_model=settings.whisper_model, status="queued")
        s.add(project)
        if upload_path is not None:
            dest_dir.mkdir(parents=True, exist_ok=True)
            try:
                shutil.move(str(upload_path), str(dest_dir / filename))
            except OSError as e:
                shutil.rmtree(dest_dir, ignore_errors=True)
                raise err("VALIDATION", f"File unggahan tidak bisa dipindahkan: {e}") from e
        try:
            queue.enqueue(s, "process", project_id)
        except BaseException:
            shutil.rmtree(dest_dir, ignore_errors=True)
            raise
        return project


def list_projects(engine: Engine) -> list[Project]:
    with session_scope(engine) as s:
        return list(s.exec(select(Project).order_by(col(Project.created_at).desc())).all())


def _active_job(session: Session, project_id: str) -> Job | None:
    jobs = session.exec(select(Job).where(Job.project_id == project_id, col(Job.status).in_(queue.ACTIVE_STATUSES))
                        .order_by(Job.created_at)).all()
    return next((j for j in jobs if j.status == "running"), jobs[0] if jobs else None)


def get_project(engine: Engine, project_id: str) -> tuple[Project, Job | None]:
    with session_scope(engine) as s:
        project = fetch(s, Project, project_id, "Project")
        return project, _active_job(s, project_id)


def delete_project(engine: Engine, project_id: str, *, settings: Settings | None = None) -> None:
    """Delete the project's jobs, clips and row, then its `projects/<id>` and `uploads/<id>` folders."""
    settings = settings or get_settings()
    with session_scope(engine) as s:
        fetch(s, Project, project_id, "Project")
        jobs = s.exec(select(Job).where(Job.project_id == project_id)).all()
        if any(j.status == "running" for j in jobs):
            raise err("INVALID_STATE", "Project sedang diproses; tunggu sampai job selesai sebelum menghapus.")
        clips = s.exec(select(Clip).where(Clip.project_id == project_id)).all()
        clip_ids = [c.id for c in clips]
        # render jobs of this project's clips (normally carry the same project_id; catch any that do not)
        other = s.exec(select(Job).where(col(Job.clip_id).in_(clip_ids))).all() if clip_ids else []
        if any(j.status == "running" for j in other):
            raise err("INVALID_STATE", "Klip project ini sedang dirender; tunggu sampai selesai sebelum menghapus.")
        for job in {j.id: j for j in [*jobs, *other]}.values():
            s.delete(job)
        s.flush()
        for clip in clips:
            s.delete(clip)
        s.flush()
        s.delete(s.get(Project, project_id))
    root = Path(settings.data_dir)
    for d in (root / "projects" / project_id, root / "uploads" / project_id):
        shutil.rmtree(d, ignore_errors=True)


def retry_project(engine: Engine, project_id: str) -> Project:
    """A `failed` project goes back to `queued` and its `process` job resumes after the last completed stage."""
    with session_scope(engine) as s:
        project = fetch(s, Project, project_id, "Project")
        if project.status != "failed":
            raise err("INVALID_STATE", f"Hanya project yang gagal yang bisa diulang (status: '{project.status}').")
        project.status = "queued"
        project.updated_at = utcnow()
        s.add(project)
        queue.enqueue(s, "process", project_id)
        return project


def _validate_facecam(box: Any) -> dict | None:
    if box is None:
        return None
    if not isinstance(box, dict) or set(box) != {"x", "y", "w", "h"}:
        raise err("VALIDATION", "Kotak facecam harus berisi tepat x, y, w, h.")
    for k, v in box.items():
        if not isinstance(v, (int, float)) or isinstance(v, bool) or not 0 <= v <= 1:
            raise err("VALIDATION", f"Nilai '{k}' kotak facecam harus angka 0 sampai 1.")
    if box["w"] <= 0 or box["h"] <= 0 or box["x"] + box["w"] > 1 or box["y"] + box["h"] > 1:
        raise err("VALIDATION", "Kotak facecam harus berada di dalam frame dan punya luas.")
    return {k: float(box[k]) for k in ("x", "y", "w", "h")}


def update_project(engine: Engine, project_id: str, patch: dict) -> Project:
    """Change `style` (flags affected rendered clips), `facecam_box`, `keep_source` or `content_type`."""
    if not isinstance(patch, dict):
        raise err("VALIDATION", "Perubahan project harus berupa objek.")
    unknown = sorted(set(patch) - set(UPDATABLE))
    if unknown:
        raise err("VALIDATION", f"Field project tidak bisa diubah: {', '.join(unknown)}.")
    fields: dict[str, Any] = {}
    if "facecam_box" in patch:
        fields["facecam_box"] = _validate_facecam(patch["facecam_box"])
    if "keep_source" in patch:
        if not isinstance(patch["keep_source"], bool):
            raise err("VALIDATION", "'keep_source' harus bernilai true atau false.")
        fields["keep_source"] = patch["keep_source"]
    if "content_type" in patch:
        _content_profile(patch["content_type"])
        fields["content_type"] = patch["content_type"]

    with session_scope(engine) as s:  # one transaction: nothing is saved if any part fails
        project = fetch(s, Project, project_id, "Project")
        if "style" in patch:
            styles.apply_project_style(s, project, patch["style"])
        if "facecam_box" in fields and fields["facecam_box"] != project.facecam_box:
            for clip in s.exec(select(Clip).where(Clip.project_id == project_id, Clip.status == "rendered")).all():
                if any(seg.get("layout") in FACECAM_LAYOUTS for seg in clip.layout_segments):
                    clip.needs_rerender = True
                    clip.updated_at = utcnow()
                    s.add(clip)
        for k, v in fields.items():
            setattr(project, k, v)
        project.updated_at = utcnow()
        s.add(project)
        return project
