"""One-time migration of M1 CLI results into the database (spec 10.6). Safe to run again.

M1 layout: `DATA_DIR/projects/<id>/meta.json` (Project.id = folder name) and `projects/<id>/runs/<whisper>__<picker>/clips.json`
(a list of `ClipPlan`), rendered files at `runs/<run>/clips/<clip_folder>/final.mp4`.
A project folder whose `Project` row exists is not created again; a clip whose id exists is skipped.
Folders without any `runs/*/clips.json` (v2 projects, unclipped M1 projects) and folders whose files fail
validation are skipped (the latter with a warning naming the folder).
"""
import logging
from pathlib import Path

from pydantic import ValidationError

from app.core.config import Settings
from app.core.errors import AppError
from app.core.fs import read_json
from app.core.models import ClipPlan, VideoMeta
from app.db.engine import get_engine, init_db, session_scope
from app.db.tables import Clip, Project
from app.domain.clip_doc import subtitles_from_m1
from app.domain.layouts import from_reframe_segments, normalize_layout_segments
from app.domain.timeline import Piece
from app.pipeline.report import clip_folder
from app.services import styles

log = logging.getLogger("kliplah.migrate")

MANUAL_RUN_SUFFIX = "__claude-manual"


def _runs(d: Path) -> list[tuple[Path, list[ClipPlan]]]:
    """Run folders holding a clips.json, oldest first (the newest decides the project's whisper model)."""
    out = []
    for clips_json in (d / "runs").glob("*/clips.json"):
        raw = read_json(clips_json)
        if isinstance(raw, list):
            out.append((clips_json.parent, [ClipPlan.model_validate(c) for c in raw]))
    return sorted(out, key=lambda r: ((r[0] / "clips.json").stat().st_mtime, r[0].name))


def _project(project_id: str, meta: VideoMeta, runs: list[tuple[Path, list[ClipPlan]]], template) -> Project:
    names = [run.name for run, _ in runs]
    whisper = names[-1].split("__", 1)[0]
    manual = any(n.endswith(MANUAL_RUN_SUFFIX) for n in names)
    return Project(
        id=project_id,
        source_type=meta.source_type,
        youtube_url=f"https://www.youtube.com/watch?v={meta.video_id}" if meta.video_id else None,
        video_id=meta.video_id, title=meta.title, channel=meta.channel, duration_sec=meta.duration,
        has_heatmap=bool(meta.heatmap), content_type="general",
        moment_source="claude_manual" if manual else "ollama", style=dict(template.style),
        style_template_id=template.id, status="ready", last_completed_stage="prepare", whisper_model=whisper)


def _clip(project_id: str, run: Path, plan: ClipPlan) -> Clip:
    cand, mats = plan.candidate, plan.materials
    duration = cand.end - cand.start
    final = run / "clips" / clip_folder(plan) / "final.mp4"
    layout = [seg.model_dump() for seg in from_reframe_segments(plan.reframe_segments)]
    return Clip(
        id=plan.id, project_id=project_id, kind="single", origin="m1", rank=plan.rank, score=cand.score,
        llm_score=cand.llm_score, heatmap_bonus=cand.heatmap_bonus, category=cand.category or "",
        reason=cand.reason, pieces=[Piece(project_id=project_id, start=cand.start, end=cand.end).model_dump()],
        layout_segments=normalize_layout_segments(layout, duration), hook_text=mats.hook, hook_options=[mats.hook],
        subtitles=subtitles_from_m1(plan.subtitles),
        captions={"tiktok": mats.tiktok.model_dump(), "reels": mats.reels.model_dump(),
                  "shorts": mats.shorts.model_dump()},
        status="rendered" if final.is_file() else "proposed", output_path=str(final) if final.is_file() else None)


def run(settings: Settings) -> dict:
    """Migrate every M1 project folder: `{"projects": n, "clips": m, "skipped": k}`.

    `k` counts project rows and clips already in the database plus folders skipped (no runs, invalid files).
    """
    engine = get_engine(settings)
    init_db(engine)
    template = styles.ensure_default_template(engine)
    counts = {"projects": 0, "clips": 0, "skipped": 0}
    root = Path(settings.data_dir) / "projects"
    folders = sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []
    for d in folders:
        raw = read_json(d / "meta.json")
        if raw is None:
            continue
        try:
            meta, runs = VideoMeta.model_validate(raw), _runs(d)
        except (ValidationError, ValueError) as e:
            log.warning("folder %s dilewati: meta.json atau clips.json tidak valid (%s)", d.name, e)
            counts["skipped"] += 1
            continue
        if not runs:
            counts["skipped"] += 1
            continue
        try:
            with session_scope(engine) as s:
                added_projects = 0
                added_clips = 0
                skipped_items = 0
                if s.get(Project, d.name) is None:
                    s.add(_project(d.name, meta, runs, template))
                    s.flush()
                    added_projects += 1
                else:
                    skipped_items += 1
                for run_dir, plans in runs:
                    for plan in plans:
                        if s.get(Clip, plan.id) is not None:
                            skipped_items += 1
                            continue
                        s.add(_clip(d.name, run_dir, plan))
                        s.flush()
                        added_clips += 1
            counts["projects"] += added_projects
            counts["clips"] += added_clips
            counts["skipped"] += skipped_items
        except (ValidationError, ValueError, AppError) as e:
            log.warning("folder %s dilewati: data klip tidak valid (%s)", d.name, e)
            counts["skipped"] += 1
            continue
        log.info("migrasi %s selesai", d.name)
    return counts
