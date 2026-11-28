"""The `prepare` stage: chosen moments (picks) -> v2 Clip rows, thumbnails and the 480p preview proxy.

Edge snapping, reframing and subtitle grouping are lifted from the M1 CLI (`finish_clips`).
`pieces` are on the source timeline; `layout_segments` and `subtitles` on the clip's output timeline.
"""
import inspect
import logging
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
from app.core.models import ReframeSegment, Sentence, Transcript, VideoMeta
from app.db.tables import Clip, Project
from app.domain.layouts import from_reframe_segments, normalize_layout_segments
from app.domain.profiles import profile_for
from app.domain.ranges import RangeFile, SourceMap
from app.domain.style import effective_style
from app.domain.timeline import Piece
from app.domain.translation import Translation, load_translation, merge_lines, missing_sentences, save_translation
from app.pipeline.reframe import reframe_clip
from app.pipeline.snap import neighbor_bounds, snap_edges
from app.pipeline.subtitle_lines import clip_subtitle_lines, resolve_subtitle_source
from app.pipeline.subtitles import EPS
from app.providers.llm import LLMUnavailable
from app.services.translation import needs_translation
from app.worker.process import Pick, default_translate, transcript_dir

log = logging.getLogger(__name__)

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
    translate: Callable[
        [list[Sentence], str, str, Path, Settings, Callable[[float], None] | None],
        tuple[dict[int, str], list[int]],
    ] | None = None


def _load(d: Path, tdir: Path) -> tuple[VideoMeta, Transcript, list[tuple[float, float]]]:
    meta = read_json(d / "meta.json")
    transcript = read_json(tdir / "transcript.json")
    silences = read_json(d / "silences.json")
    if meta is None or transcript is None or silences is None:
        raise AppError("INVALID_STATE", "Data video, transkrip, atau jeda tidak ditemukan; proses ulang project ini.")
    return VideoMeta.model_validate(meta), Transcript.model_validate(transcript), [tuple(s) for s in silences]


def _overlapping_sentence_indices(sentences: list[Sentence], spans: list[tuple[float, float]]) -> set[int]:
    """Return indices of sentences overlapping any snapped clip span `[s, e]`."""
    return {
        sent.i
        for sent in sentences
        if any(sent.end > s + EPS and sent.start < e - EPS for s, e in spans)
    }


def _call_translate(
    translate_fn: Callable[..., tuple[dict[int, str], list[int]]],
    sentences: list[Sentence],
    source: str,
    target: str,
    checkpoint_dir: Path,
    settings: Settings,
) -> tuple[dict[int, str], list[int]]:
    """Invoke `translate_fn`, supplying `None` for `on_progress` only when the 6th arg is required."""
    params = inspect.signature(translate_fn).parameters
    positional = [
        p for p in params.values()
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]
    if len(positional) >= 6 and positional[5].default is inspect.Parameter.empty:
        return translate_fn(sentences, source, target, checkpoint_dir, settings, None)
    return translate_fn(sentences, source, target, checkpoint_dir, settings)


def _ensure_clip_translations(
    t: Transcript,
    tdir: Path,
    project_id: str,
    target_language: str,
    spans: list[tuple[float, float]],
    deps: PrepareDeps,
) -> Translation | None:
    """Translate missing sentences overlapping `spans` and persist them; fall back gracefully on failure."""
    if not needs_translation(t.language, target_language):
        return None
    tr = load_translation(tdir, target_language, t.language)
    needed = _overlapping_sentence_indices(t.sentences, spans)
    missing = missing_sentences(t, tr, indices=needed)
    if not missing:
        return tr
    translate_fn = default_translate if deps.translate is None else deps.translate
    try:
        translated, failed = _call_translate(
            translate_fn, missing, t.language, target_language, tdir / "checkpoints", deps.settings
        )
    except LLMUnavailable as e:
        log.warning(
            "Ollama tidak tersedia saat menerjemahkan subtitle klip untuk project %s (%s); teks asli dipakai.",
            project_id,
            e,
        )
        return tr
    if merge_lines(tr, translated, valid={s.i for s in missing}) > 0:
        save_translation(tdir, tr)
    if failed:
        log.warning(
            "%d kalimat gagal diterjemahkan untuk project %s; teks asli dipertahankan.",
            len(failed),
            project_id,
        )
    return tr


def build_clips_from_picks(engine: Engine, project_id: str, picks: list[Pick], deps: PrepareDeps) -> list[str]:
    """Replace the project's `proposed` clips with clips built from `picks`; returns the new clip ids.

    Clips the user already acted on (approved, rendering, rendered, render_failed, rejected) are kept as they are.
    New clips are ranked 1..N by score among themselves.
    """
    with Session(engine) as session:
        project = session.get(Project, project_id)
        if project is None:
            raise AppError("NOT_FOUND", f"Project {project_id} tidak ditemukan", status=404)
        tdir = transcript_dir(deps.settings, project)
        target_language = project.target_language
        project_source = project.transcript_source
        max_sec = project.max_clip_sec or profile_for(project.content_type).max_sec
        words_per_line = effective_style(project.style or {}, {})["subtitle"]["words_per_line"]

    d = project_dir(deps.settings, project_id)
    meta, t, silences = _load(d, tdir)
    smap = SourceMap.from_meta(meta, d, t)
    sub_source = resolve_subtitle_source(project_source, meta.transcript_source)

    by_time = sorted(picks, key=lambda p: p[0].start)
    spans = [(c.start, c.end) for c, _ in by_time]
    prepared: list[tuple[Pick, RangeFile, float, float]] = []
    for i, (cand, mats) in enumerate(by_time):
        rf = smap.locate_span(cand.start, cand.end)
        if rf is None:
            log.warning("Kandidat %.3f-%.3f melewati batas rentang; dilewati.", cand.start, cand.end)
            continue
        lower, upper = neighbor_bounds(spans, i, max(meta.duration, rf.end))
        lower, upper = max(lower, rf.start), min(upper, rf.end)
        s, e = snap_edges(cand.start, cand.end, t.words, t.sentences, silences, lower=lower, upper=upper,
                          max_len=max_sec + SNAP_SLACK)
        prepared.append(((cand, mats), rf, s, e))

    tr = _ensure_clip_translations(t, tdir, project_id, target_language, [(s, e) for _, _, s, e in prepared], deps)

    clips: list[Clip] = []
    thumb_loc: dict[str, tuple[Path, float]] = {}
    for (cand, mats), rf, s, e in prepared:
        src = smap.path(rf)
        ls = max(0.0, smap.to_local(rf, s))
        le = max(ls, smap.to_local(rf, e))
        if deps.reframe is None:
            segs, _ = reframe_clip(src, ls, le, models_dir=Path(deps.settings.data_dir) / "models")
        else:
            segs, _ = deps.reframe(src, ls, le)
        cid = uuid.uuid4().hex
        thumb_loc[cid] = (src, max(0.0, smap.to_local(rf, s + min(1.0, (e - s) / 2))))
        piece = Piece(project_id=project_id, start=s, end=e)
        clips.append(Clip(
            id=cid, project_id=project_id, kind="single", origin="fallback" if cand.no_ai else "ai",
            rank=0, score=cand.score, llm_score=cand.llm_score, heatmap_bonus=cand.heatmap_bonus,
            signal_bonus=cand.signal_bonus,
            category=cand.category or "", reason=cand.reason, pieces=[piece.model_dump()],
            layout_segments=normalize_layout_segments(
                [seg.model_dump() for seg in from_reframe_segments(segs)], round(e - s, 3)),
            hook_text=mats.hook, hook_options=[mats.hook],
            subtitles=clip_subtitle_lines(t, tr, s, e, max_words=words_per_line, source=sub_source),
            captions={"tiktok": mats.tiktok.model_dump(), "reels": mats.reels.model_dump(),
                      "shorts": mats.shorts.model_dump()},
            status="proposed"))
    clips.sort(key=lambda c: c.score, reverse=True)
    for rank, c in enumerate(clips, 1):
        c.rank = rank
    thumbs = [(c.id, *thumb_loc[c.id]) for c in clips]

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
    for cid, src, local_t in thumbs:
        thumb_fn(src, local_t, d / "clips" / cid / "thumb.jpg")
    return [cid for cid, _, _ in thumbs]


def prepare_project_clips(engine: Engine, project_id: str, picks: list[Pick],
                          settings: Settings | None = None) -> None:
    """Default `ProcessDeps.prepare`: build the clips, then the 480p preview proxy."""
    deps = PrepareDeps(settings=settings or get_settings())
    build_clips_from_picks(engine, project_id, picks, deps)
    d = project_dir(deps.settings, project_id)
    meta_raw = read_json(d / "meta.json")
    proxy_fn = make_proxy if deps.proxy is None else deps.proxy
    if meta_raw and meta_raw.get("ranges"):
        smap = SourceMap.from_meta(VideoMeta.model_validate(meta_raw), d)
        for rf in smap.files:
            proxy_fn(smap.path(rf), d / f"proxy_r{rf.index}.mp4")
    else:
        proxy_fn(d / "source.mp4", d / "proxy.mp4")
