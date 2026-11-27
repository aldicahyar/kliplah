"""The `prepare` stage: chosen moments (picks) -> v2 Clip rows, thumbnails and the 480p preview proxy.

Edge snapping, reframing and subtitle grouping are lifted from the M1 CLI (`finish_clips`).
`pieces` are on the source timeline; `layout_segments` and `subtitles` on the clip's output timeline.
"""
import shutil
import subprocess
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy import Engine
from sqlmodel import Session, select

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.fs import project_dir, read_json
from app.core.models import ReframeSegment, Transcript, VideoMeta
from app.db.tables import Clip, Project
from app.domain.clip_doc import subtitles_from_m1
from app.domain.layouts import from_reframe_segments, normalize_layout_segments
from app.domain.profiles import profile_for
from app.domain.style import effective_style
from app.domain.timeline import Piece
from app.pipeline.reframe import reframe_clip
from app.pipeline.snap import neighbor_bounds, snap_edges
from app.pipeline.subtitles import group_lines
from app.worker.process import Pick, _safe

SNAP_SLACK = 3.0  # edge refinement may stretch a clip slightly past max_sec (same as M1)
THUMB_WIDTH = 360
PROXY_HEIGHT = 480


def make_thumbnail(src: Path, t: float, out: Path) -> None:
    """One JPEG frame at `t` seconds, 360 px wide."""
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".partial.jpg")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(src), "-frames:v", "1",
                    "-vf", f"scale={THUMB_WIDTH}:-2", "-q:v", "3", str(tmp)], check=True)
    tmp.replace(out)


def make_proxy(src: Path, out: Path) -> None:
    """480p H.264/AAC preview with faststart. Skipped when `out` already exists (checkpoint)."""
    if out.exists():
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.stem + ".partial.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vf", f"scale=-2:{PROXY_HEIGHT}",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", "-pix_fmt", "yuv420p",
                    "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", str(tmp)], check=True)
    tmp.replace(out)


@dataclass
class PrepareDeps:
    """Media functions used by the stage; every one can be replaced in tests."""
    settings: Settings = field(default_factory=get_settings)
    reframe: Callable[[Path, float, float], tuple[list[ReframeSegment], str]] | None = None
    thumbnail: Callable[[Path, float, Path], None] | None = None
    proxy: Callable[[Path, Path], None] | None = None


def _load(d: Path, whisper_model: str) -> tuple[VideoMeta, Transcript, list[tuple[float, float]]]:
    meta = read_json(d / "meta.json")
    transcript = read_json(d / "transcripts" / _safe(whisper_model) / "transcript.json")
    silences = read_json(d / "silences.json")
    if meta is None or transcript is None or silences is None:
        raise AppError("INVALID_STATE", "Data video, transkrip, atau jeda tidak ditemukan; proses ulang project ini.")
    return VideoMeta.model_validate(meta), Transcript.model_validate(transcript), [tuple(s) for s in silences]


def build_clips_from_picks(engine: Engine, project_id: str, picks: list[Pick], deps: PrepareDeps) -> list[str]:
    """Replace the project's `proposed` clips with clips built from `picks`; returns the new clip ids.

    Clips the user already acted on (approved, rendering, rendered, render_failed, rejected) are kept as they are.
    New clips are ranked 1..N by score among themselves.
    """
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if project is None:
            raise AppError("NOT_FOUND", f"Project {project_id} tidak ditemukan", status=404)
        whisper_model = project.whisper_model
        max_sec = project.max_clip_sec or profile_for(project.content_type).max_sec
        words_per_line = effective_style(project.style or {}, {})["subtitle"]["words_per_line"]

    d = project_dir(deps.settings, project_id)
    src = d / "source.mp4"
    meta, t, silences = _load(d, whisper_model)

    by_time = sorted(picks, key=lambda p: p[0].start)
    spans = [(c.start, c.end) for c, _ in by_time]
    clips: list[Clip] = []
    for i, (cand, mats) in enumerate(by_time):
        lower, upper = neighbor_bounds(spans, i, meta.duration)
        s, e = snap_edges(cand.start, cand.end, t.words, t.sentences, silences, lower=lower, upper=upper,
                          max_len=max_sec + SNAP_SLACK)
        if deps.reframe is None:
            segs, _ = reframe_clip(src, s, e, models_dir=Path(deps.settings.data_dir) / "models")
        else:
            segs, _ = deps.reframe(src, s, e)
        piece = Piece(project_id=project_id, start=s, end=e)
        clips.append(Clip(
            id=uuid.uuid4().hex, project_id=project_id, kind="single", origin="fallback" if cand.no_ai else "ai",
            rank=0, score=cand.score, llm_score=cand.llm_score, heatmap_bonus=cand.heatmap_bonus,
            category=cand.category or "", reason=cand.reason, pieces=[piece.model_dump()],
            layout_segments=normalize_layout_segments(
                [seg.model_dump() for seg in from_reframe_segments(segs)], round(e - s, 3)),
            hook_text=mats.hook, hook_options=[mats.hook],
            subtitles=subtitles_from_m1(group_lines(t.words, s, e, max_words=words_per_line)),
            captions={"tiktok": mats.tiktok.model_dump(), "reels": mats.reels.model_dump(),
                      "shorts": mats.shorts.model_dump()},
            status="proposed"))
    clips.sort(key=lambda c: c.score, reverse=True)
    for rank, c in enumerate(clips, 1):
        c.rank = rank
    thumbs = [(c.id, c.pieces[0]["start"], c.pieces[0]["end"]) for c in clips]

    with Session(engine) as session:
        stale = session.exec(select(Clip).where(Clip.project_id == project_id, Clip.status == "proposed")).all()
        stale_ids = [c.id for c in stale]
        for c in stale:
            session.delete(c)
        session.add_all(clips)
        session.commit()

    for cid in stale_ids:
        shutil.rmtree(d / "clips" / cid, ignore_errors=True)
    thumb_fn = make_thumbnail if deps.thumbnail is None else deps.thumbnail
    for cid, start, end in thumbs:
        thumb_fn(src, start + min(1.0, (end - start) / 2), d / "clips" / cid / "thumb.jpg")
    return [cid for cid, _, _ in thumbs]


def prepare_project_clips(engine: Engine, project_id: str, picks: list[Pick],
                          settings: Settings | None = None) -> None:
    """Default `ProcessDeps.prepare`: build the clips, then the 480p preview proxy."""
    deps = PrepareDeps(settings=settings or get_settings())
    build_clips_from_picks(engine, project_id, picks, deps)
    d = project_dir(deps.settings, project_id)
    proxy_fn = make_proxy if deps.proxy is None else deps.proxy
    proxy_fn(d / "source.mp4", d / "proxy.mp4")
