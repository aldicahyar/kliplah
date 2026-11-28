"""The `render` job: one approved v2 clip -> `projects/<pid>/clips/<clip_id>/final.mp4` via the M1 renderer.

The v2 clip document is turned into an M1 `ClipPlan` by `to_clip_plan` (single-piece only until S4).
"""
import logging
from collections.abc import Callable
from pathlib import Path

from sqlalchemy import Engine
from sqlmodel import Session

from app.core.config import BACKEND_DIR, Settings, get_settings
from app.core.errors import AppError
from app.core.fs import read_json
from app.core.models import VideoMeta
from app.db.tables import Clip, Job, Project, utcnow
from app.domain.clip_doc import to_clip_plan
from app.domain.ranges import SourceMap
from app.pipeline.render import render_clip
from app.worker import queue
from app.worker.process import to_app_error

log = logging.getLogger("kliplah.render")

FONTS_DIR = BACKEND_DIR / "assets" / "fonts"
RENDERABLE = ("approved", "rendered", "render_failed")  # rendered = re-render, render_failed = manual retry
PROGRESS_STEP = 0.01

Render = Callable[..., Path]


def _set_clip(session: Session, clip: Clip, **fields) -> None:
    for k, v in fields.items():
        setattr(clip, k, v)
    clip.updated_at = utcnow()
    session.add(clip)
    session.commit()


def _progress_writer(engine: Engine, job_id: str, duration: float) -> Callable[[float], None]:
    """ffmpeg reports output seconds from a reader thread: write the fraction with its own session, throttled."""
    last = {"value": -1.0}

    def on_progress(seconds: float) -> None:
        frac = min(max(seconds / duration, 0.0), 1.0) if duration > 0 else 0.0
        if frac - last["value"] < PROGRESS_STEP:
            return
        last["value"] = frac
        try:  # best-effort: raising here would kill ffmpeg's reader thread and look like a stall
            with Session(engine) as s:
                queue.set_progress(s, job_id, "render", round(frac, 3))
        except Exception as e:
            log.warning("progress render job %s tidak tersimpan: %s", job_id, e)

    return on_progress


def run_render_job(engine: Engine, job_id: str, render: Render = render_clip, *,
                   settings: Settings | None = None) -> None:
    """Render the claimed job's clip. The job ends succeeded, failed, or queued again for a retry."""
    settings = settings or get_settings()
    with Session(engine, expire_on_commit=False) as session:
        job = session.get(Job, job_id)
        if job is None or job.clip_id is None:
            raise AppError("NOT_FOUND", f"Job {job_id} tidak ditemukan", status=404)
        clip = session.get(Clip, job.clip_id)
        if clip is None:
            queue.fail_job(session, job_id, AppError("NOT_FOUND", f"Klip {job.clip_id} tidak ditemukan", status=404))
            return
        if clip.status not in RENDERABLE:
            queue.fail_job(session, job_id, AppError(
                "INVALID_STATE", f"Klip berstatus '{clip.status}' tidak bisa dirender; setujui klip dulu."))
            return

        try:
            _set_clip(session, clip, status="rendering")
            queue.set_progress(session, job_id, "render", 0.0)
            project = session.get(Project, clip.project_id) if clip.project_id else None
            if project is None:
                raise AppError("VALIDATION", "Render klip kompilasi tersedia di S4")
            pdir = Path(settings.data_dir) / "projects" / project.id
            plan, presets = to_clip_plan(clip, project.style or {})
            meta_raw = read_json(pdir / "meta.json")
            if meta_raw and meta_raw.get("ranges"):
                smap = SourceMap.from_meta(VideoMeta.model_validate(meta_raw), pdir)
                rf = smap.locate_span(plan.candidate.start, plan.candidate.end)
                if rf is None:
                    raise AppError("VALIDATION", "Bagian video ini belum diunduh (di luar rentang project).")
                source = smap.path(rf)
                ls = max(0.0, smap.to_local(rf, plan.candidate.start))
                le = max(ls, smap.to_local(rf, plan.candidate.end))
                plan = plan.model_copy(update={"candidate": plan.candidate.model_copy(update={"start": ls, "end": le})})
            else:
                source = pdir / "source.mp4"
            if not source.exists():
                raise AppError("INVALID_STATE", "Video sumber sudah dihapus")
            duration = plan.candidate.end - plan.candidate.start
            out = render(source, plan, pdir / "clips" / clip.id, presets, FONTS_DIR,
                         on_progress=_progress_writer(engine, job_id, duration))
            _set_clip(session, clip, status="rendered", output_path=str(out), needs_rerender=False)
            queue.finish_job(session, job_id)
        except Exception as e:
            log.exception("render job %s failed", job_id)
            session.rollback()
            queue.fail_job(session, job_id, to_app_error(e))
            retrying = session.get(Job, job_id, populate_existing=True).status == "queued"
            clip = session.get(Clip, clip.id, populate_existing=True)
            # output_path is kept: a failed re-render leaves the previous final.mp4 valid in place
            _set_clip(session, clip, status="approved" if retrying else "render_failed")
