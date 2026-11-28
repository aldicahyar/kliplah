"""Project service: create (YouTube link or upload), list, get, delete, retry and update."""
import copy
import math
import shutil
import uuid
from pathlib import Path
from typing import Any

from sqlalchemy import Engine
from sqlmodel import Session, col, select

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.db.engine import session_scope
from app.db.tables import Clip, Job, Project, StyleTemplate, utcnow
from app.domain.languages import (
    is_weak_local_translation,
    language_name,
    normalize_language,
    safe_normalize_language,
)
from app.domain.profiles import profile_for
from app.domain.ranges import normalize_ranges
from app.pipeline.ingest import IngestError, parse_youtube_id, probe_youtube
from app.pipeline.youtube_subs import TRANSCRIPT_SOURCES, resolve_subtitle_plan
from app.services import styles
from app.services._common import domain_errors, err, fetch
from app.services.settings_store import MOMENT_SOURCES, get_app_settings
from app.worker import queue

UPDATABLE = ("style", "facecam_box", "keep_source", "content_type")
FACECAM_LAYOUTS = ("facecam_gameplay", "facecam_full")
# perkiraan dari uji nyata 2026-11-27 (i7-1355U, tanpa GPU)
DOWNLOAD_SEC_PER_HOUR = 120.0
ENCODE_FACTOR = 0.35
WHISPER_FACTOR = 0.85
CHAT_SEC_PER_HOUR = 50.0
LLM_SEC_PER_WINDOW = 480.0
LLM_WINDOW_SEC = 300.0
LLM_SEC_PER_CLIP = 25.0
TRANSLATE_SEC_PER_MIN = 20.0  # perkiraan awal, dikalibrasi di Task 12


def _breakdown_minutes(secs: dict[str, float]) -> tuple[int, dict[str, int]]:
    """Convert stage seconds to integer minutes whose sum equals `max(1, round(total_sec / 60))`."""
    total_sec = sum(secs.values())
    if total_sec <= 0:
        return 0, {k: 0 for k in secs}
    target = max(1, round(total_sec / 60.0))
    exact = {k: v / 60.0 for k, v in secs.items()}
    mins = {k: math.floor(v) for k, v in exact.items()}
    diff = target - sum(mins.values())
    if diff > 0:
        order = sorted((k for k in secs if secs[k] > 0), key=lambda k: (exact[k] - mins[k], exact[k]), reverse=True)
        for k in order[:diff]:
            mins[k] += 1
    return sum(mins.values()), mins


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


def _normalize_source(source_language: Any) -> str | None:
    with domain_errors("source_language"):
        if source_language is not None and not isinstance(source_language, str):
            raise err("VALIDATION", "Bahasa sumber harus berupa teks.")
        return normalize_language(source_language)


def _normalize_target(target_language: Any) -> str:
    with domain_errors("target_language"):
        if target_language is not None and not isinstance(target_language, str):
            raise err("VALIDATION", "Bahasa target harus berupa teks.")
        if isinstance(target_language, str) and target_language.strip().lower() == "auto":
            raise err("VALIDATION", "Bahasa target tidak boleh 'auto'.")
        return normalize_language(target_language) or "id"


def _probe_warnings(effective_dur: float, plan, cand_lang: str | None, needs_tr: bool,
                    resolved_moment: str) -> list[str]:
    warnings: list[str] = []
    if effective_dur > 7200:
        warnings.append("Durasi video lebih dari 2 jam; pilih rentang waktu atau sumber transkrip youtube_auto "
                        "agar proses lebih cepat dan hemat ruang disk.")
    if needs_tr and cand_lang is not None and is_weak_local_translation(cand_lang) and resolved_moment == "ollama":
        warnings.append(
            f"Bahasa sumber {language_name(cand_lang, 'id')} kurang akurat diterjemahkan dengan Ollama lokal; "
            "pertimbangkan jalur Claude manual."
        )
    if plan.source == "whisper" and cand_lang is not None and cand_lang not in ("id", "en"):
        warnings.append(
            f"Bahasa sumber {language_name(cand_lang, 'id')} bukan Indonesia/Inggris; "
            "pertimbangkan whisper_model 'medium' agar transkrip lebih akurat."
        )
    return warnings


def probe(engine: Engine, url: str, *, ranges: list | None = None, source_language: str | None = None,
          target_language: str = "id", content_type: str | None = None, moment_source: str | None = None,
          max_clips: int | None = None, settings: Settings | None = None) -> dict:
    """Inspect a YouTube link without downloading: metadata, subtitle/chat flags, resource estimates, warnings."""
    settings = settings or get_settings()
    norm_source = _normalize_source(source_language)
    norm_target = _normalize_target(target_language)
    with session_scope(engine) as s:
        app = get_app_settings(s, settings=settings)
    cookies_browser = app["cookies_browser"]
    resolved_type = content_type if content_type is not None else app["content_type"]
    profile = _content_profile(resolved_type)
    resolved_moment = moment_source if moment_source is not None else app["moment_source"]
    if resolved_moment not in MOMENT_SOURCES:
        raise err("VALIDATION", f"Sumber momen harus salah satu dari: {', '.join(MOMENT_SOURCES)}.")
    resolved_clips = max_clips if max_clips is not None else app["max_clips"]
    _check_positive_int("max_clips", resolved_clips)
    try:
        meta = probe_youtube(url, settings, cookies_browser=cookies_browser)
    except IngestError as e:
        raise err(e.code, str(e)) from e
    norm = normalize_ranges(ranges, meta.duration or None)
    effective_dur = sum(e - s for s, e in norm) if norm else meta.duration
    plan = resolve_subtitle_plan("auto", meta, norm_target, source_language=norm_source)
    cand_lang = safe_normalize_language(plan.language) or norm_source or safe_normalize_language(meta.language)
    needs_tr = (
        plan.needs_translation
        if plan.needs_translation is not None
        else (cand_lang is not None and cand_lang != norm_target)
    )
    video_rate = 0.6 * 1024**2 if norm else 1.5 * 1024**2
    disk_bytes = effective_dur * (video_rate + 32 * 1024)
    disk_gb = round(disk_bytes / 1024**3, 1)
    download_sec = (effective_dur / 3600.0) * DOWNLOAD_SEC_PER_HOUR
    encode_sec = effective_dur * ENCODE_FACTOR if norm else 0.0
    whisper_sec = effective_dur * WHISPER_FACTOR if plan.source == "whisper" else 0.0
    chat_dur = norm[-1][1] if norm else meta.duration
    chat_sec = (
        (chat_dur / 3600.0) * CHAT_SEC_PER_HOUR
        if meta.has_live_chat and "live_chat" in profile.signals
        else 0.0
    )
    translate_sec = (
        (effective_dur / 60.0) * TRANSLATE_SEC_PER_MIN
        if effective_dur > 0 and needs_tr and resolved_moment == "ollama"
        else 0.0
    )
    llm_sec = (
        (effective_dur / LLM_WINDOW_SEC) * LLM_SEC_PER_WINDOW + resolved_clips * LLM_SEC_PER_CLIP
        if effective_dur > 0 and resolved_moment == "ollama"
        else 0.0
    )
    process_minutes, breakdown = _breakdown_minutes({
        "download_min": download_sec,
        "encode_min": encode_sec,
        "transcribe_min": whisper_sec,
        "chat_min": chat_sec,
        "translate_min": translate_sec,
        "llm_min": llm_sec,
    })
    if effective_dur <= 0:
        process_minutes = 0
        breakdown = {k: 0 for k in breakdown}
    warnings = _probe_warnings(effective_dur, plan, cand_lang, needs_tr, resolved_moment)
    return {
        "title": meta.title,
        "channel": meta.channel,
        "duration": meta.duration,
        "chapters": meta.chapters,
        "has_heatmap": bool(meta.heatmap),
        "subtitles": meta.subtitles or {"manual": [], "auto": []},
        "has_live_chat": meta.has_live_chat,
        "transcript_source_auto": plan.source,
        "source_language_hint": meta.language,
        "subtitle_plan": {
            "source": plan.source,
            "language": plan.language,
            "needs_translation": plan.needs_translation,
        },
        "translation_engine": "ollama" if resolved_moment == "ollama" else "claude",
        "estimate": {"process_minutes": process_minutes, "breakdown": breakdown, "disk_gb": disk_gb},
        "warnings": warnings,
    }


def create_project(engine: Engine, *, url: str | None = None, upload_path: Path | None = None,
                   original_filename: str | None = None, content_type: str | None = None, ranges: list | None = None,
                   focus_prompt: str | None = None, source_language: str | None = None,
                   target_language: str = "id", transcript_source: str = "auto",
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
    norm_source = _normalize_source(source_language)
    norm_target = _normalize_target(target_language)
    if min_clip_sec is not None:
        _check_positive_int("min_clip_sec", min_clip_sec)
    if max_clip_sec is not None:
        _check_positive_int("max_clip_sec", max_clip_sec)
    if min_clip_sec is not None and max_clip_sec is not None and max_clip_sec < min_clip_sec:
        raise err("VALIDATION", "Durasi maksimal klip tidak boleh lebih kecil dari durasi minimal.")
    if moment_source is not None and moment_source not in MOMENT_SOURCES:
        raise err("VALIDATION", f"Sumber momen harus salah satu dari: {', '.join(MOMENT_SOURCES)}.")
    if transcript_source not in TRANSCRIPT_SOURCES:
        raise err("VALIDATION", f"Sumber transkrip harus salah satu dari: {', '.join(TRANSCRIPT_SOURCES)}.")
    norm_ranges = normalize_ranges(ranges, None)
    if upload_path is not None and norm_ranges is not None:
        raise err("VALIDATION", "Rentang untuk file unggahan belum didukung.")
    stored_ranges = [[s, e] for s, e in norm_ranges] if norm_ranges else None
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
            content_type=resolved_type, ranges=stored_ranges, focus_prompt=focus_prompt,
            source_language=norm_source, target_language=norm_target, transcript_source=transcript_source,
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
