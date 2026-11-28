"""Clip service: list/get, PATCH of the clip document, status actions (approve/reject/restore/bulk), manual clips.

`pieces` are on the source timeline; `subtitles`, `overlays` and `layout_segments` on the clip's output timeline.
A change of pieces moves the output-timeline items with `remap_output_items` (spec 10.5) and fills the newly
covered output ranges with subtitle lines grouped from the transcript.
"""
import copy
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel
from sqlalchemy import Engine
from sqlmodel import Session, select

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.fs import project_dir, read_json
from app.core.models import PlatformCaption, Transcript, VideoMeta
from app.db.engine import session_scope
from app.db.tables import Clip, Job, Project, utcnow
from app.domain.clip_doc import Overlay, SubtitleLineV2
from app.domain.layouts import LayoutSegment, normalize_layout_segments
from app.domain.ranges import RangeFile, SourceMap
from app.domain.style import effective_style, style_diff
from app.domain.timeline import Piece, output_duration, output_to_source, remap_output_items
from app.domain.translation import Translation, load_translation
from app.pipeline.subtitle_lines import clip_subtitle_lines, resolve_subtitle_source
from app.services._common import domain_errors, err, fetch
from app.services.styles import clip_base_style
from app.services.translation import needs_translation
from app.worker import queue
from app.worker.process import transcript_dir

PATCHABLE = ("pieces", "layout_segments", "style_override", "overlays", "subtitles", "hook_text", "captions")
RERENDER_FIELDS = ("pieces", "layout_segments", "style_override", "overlays", "subtitles", "hook_text")
PLATFORMS = ("tiktok", "reels", "shorts")
APPROVABLE = ("proposed", "rendered", "render_failed", "approved")
RENDERABLE = ("approved", "rendered", "render_failed")
_EPS = 0.0005


# ---------- reads ----------

def list_clips(engine: Engine, project_id: str, include_rejected: bool = False) -> list[Clip]:
    with session_scope(engine) as s:
        fetch(s, Project, project_id, "Project")
        q = select(Clip).where(Clip.project_id == project_id)
        if not include_rejected:
            q = q.where(Clip.status != "rejected")
        return list(s.exec(q.order_by(Clip.rank, Clip.created_at)).all())


def get_clip(engine: Engine, clip_id: str) -> Clip:
    with session_scope(engine) as s:
        return fetch(s, Clip, clip_id, "Klip")


# ---------- transcript-based subtitles ----------


def _transcript_loader(
    session: Session, settings: Settings
) -> Callable[[str], tuple[Transcript, Translation | None, str] | None]:
    cache: dict[str, tuple[Transcript, Translation | None, str] | None] = {}

    def load(project_id: str) -> tuple[Transcript, Translation | None, str] | None:
        if project_id not in cache:
            project = session.get(Project, project_id)
            if project is None:
                cache[project_id] = None
            else:
                tdir = transcript_dir(settings, project)
                raw = read_json(tdir / "transcript.json")
                if raw is None:
                    cache[project_id] = None
                else:
                    t = Transcript.model_validate(raw)
                    tr = (
                        load_translation(tdir, project.target_language, t.language)
                        if needs_translation(t.language, project.target_language)
                        else None
                    )
                    cache[project_id] = (t, tr, resolve_subtitle_source(project.transcript_source))
        return cache[project_id]

    return load


def _piece_lines(
    t: Transcript,
    piece: Piece,
    offset: float,
    max_words: int,
    *,
    tr: Translation | None = None,
    source: str = "whisper",
) -> list[dict]:
    """Subtitle lines of one piece from the transcript, shifted to the piece's output offset."""
    lines = clip_subtitle_lines(t, tr, piece.start, piece.end, max_words=max_words, source=source)
    for line in lines:
        for item in (line, *line["words"]):
            item["start"] = round(item["start"] + offset, 3) + 0.0
            item["end"] = round(item["end"] + offset, 3) + 0.0
    return lines


def _overlaps(a: dict, b: dict) -> bool:
    return a["start"] < b["end"] - _EPS and a["end"] > b["start"] + _EPS


def _fill_subtitles(
    existing: list[dict],
    pieces: list[Piece],
    load: Callable[[str], tuple[Transcript, Translation | None, str] | None],
    max_words: int,
) -> list[dict]:
    """Add transcript lines that overlap no existing line; renumber ids 0..n-1 in time order."""
    out = list(existing)
    offset = 0.0
    for piece in pieces:
        loaded = load(piece.project_id)
        if loaded is not None:
            t, tr, source = loaded
            out += [
                line for line in _piece_lines(t, piece, offset, max_words, tr=tr, source=source)
                if not any(_overlaps(line, old) for old in existing)
            ]
        offset += piece.end - piece.start
    return _renumber(out)


def _renumber(lines: list[dict]) -> list[dict]:
    """Sort subtitle lines by time and number them 0..n-1."""
    lines.sort(key=lambda line: (line["start"], line["end"]))
    for i, line in enumerate(lines):
        line["id"] = i
    return lines


def _r(t: float) -> float:
    return round(t, 3) + 0.0


def _fit_items(items: list[dict], duration: float) -> list[dict]:
    """Round output-timeline items (and their words) to 0.001 s, clip them to [0, duration], drop empty ones."""
    out = []
    for item in copy.deepcopy(items):
        start, end = _r(max(item["start"], 0.0)), _r(min(item["end"], duration))
        if end - start <= _EPS:
            continue
        item["start"], item["end"] = start, end
        if "words" in item:
            words = []
            for w in item["words"]:
                ws, we = _r(max(w["start"], 0.0)), _r(min(w["end"], duration))
                if we - ws > _EPS:
                    w["start"], w["end"] = ws, we
                    words.append(w)
            item["words"] = words
        out.append(item)
    return out


def _layout_for(segments: list[dict], pieces: list[Piece]) -> list[dict]:
    """Segments tiling the whole output timeline, each with the index of the piece it starts in."""
    segs = normalize_layout_segments(segments, output_duration(pieces))
    for seg in segs:
        hit = output_to_source(pieces, seg["start"])
        seg["piece_index"] = hit[0] if hit else 0
    return segs


# ---------- PATCH ----------

def _validated_list(patch: dict, field: str, model: type[BaseModel]) -> list[dict]:
    value = patch[field]
    with domain_errors(field):
        if not isinstance(value, list):
            raise err("VALIDATION", f"Data '{field}' harus berupa daftar.")
        return [model.model_validate(item).model_dump() for item in value]


def _check_piece_in_ranges(project: Project, start: float, end: float, settings: Settings) -> None:
    pdir = project_dir(settings, project.id)
    meta_raw = read_json(pdir / "meta.json")
    smap: SourceMap | None = None
    if meta_raw and meta_raw.get("ranges"):
        smap = SourceMap.from_meta(VideoMeta.model_validate(meta_raw), pdir)
    elif project.ranges:
        files = [RangeFile(i, float(r[0]), float(r[1]), f"source_r{i}.mp4") for i, r in enumerate(project.ranges)]
        smap = SourceMap(files, pdir)
    if smap is not None and smap.locate_span(start, end) is None:
        raise err("VALIDATION", "Bagian video ini belum diunduh (di luar rentang project).")


def _validated_pieces(patch: dict, clip: Clip, session: Session, settings: Settings) -> list[dict]:
    pieces = _validated_list(patch, "pieces", Piece)
    if not pieces:
        raise err("VALIDATION", "Klip harus punya minimal satu potongan.")
    for p in pieces:
        if clip.kind == "single" and p["project_id"] != clip.project_id:
            raise err("VALIDATION", "Potongan klip biasa harus berasal dari project klip itu sendiri.")
        project = session.get(Project, p["project_id"])
        if project is None:
            raise err("VALIDATION", f"Project {p['project_id']} pada potongan tidak ditemukan.")
        if project.duration_sec and p["end"] > project.duration_sec + _EPS:
            raise err("VALIDATION", "Akhir potongan melewati durasi video.")
        _check_piece_in_ranges(project, p["start"], p["end"], settings)
    return pieces


def _validated_captions(value: Any, current: dict) -> dict:
    """Merge per-platform caption fields into the current captions and validate the result."""
    if not isinstance(value, dict) or set(value) - set(PLATFORMS) or not all(isinstance(v, dict)
                                                                              for v in value.values()):
        raise err("VALIDATION", f"Data 'captions' harus objek dengan kunci {', '.join(PLATFORMS)}.")
    out = dict(current)
    with domain_errors("captions"):
        for k, v in value.items():
            merged = {"title": "", "caption": "", "hashtags": [], **current.get(k, {}), **v}
            out[k] = PlatformCaption.model_validate(merged).model_dump()
    return out


def patch_clip(engine: Engine, clip_id: str, patch: dict, *, settings: Settings | None = None) -> Clip:
    """Apply a partial clip document. Items sent together with new `pieces` are taken as already on the new
    output timeline; items not sent are remapped. Rendered clips that really changed get `needs_rerender`."""
    settings = settings or get_settings()
    if not isinstance(patch, dict):
        raise err("VALIDATION", "Perubahan klip harus berupa objek.")
    unknown = sorted(set(patch) - set(PATCHABLE))
    if unknown:
        raise err("VALIDATION", f"Field klip tidak bisa diubah: {', '.join(unknown)}.")

    with session_scope(engine) as s:
        clip = fetch(s, Clip, clip_id, "Klip")
        if clip.status == "rendering":
            raise err("INVALID_STATE", "Klip sedang dirender; tunggu sampai selesai sebelum mengubahnya.")
        base_style = clip_base_style(s, clip)
        before = {f: copy.deepcopy(getattr(clip, f)) for f in PATCHABLE}
        new = copy.deepcopy(before)

        if "hook_text" in patch:
            if not isinstance(patch["hook_text"], str):
                raise err("VALIDATION", "Data 'hook_text' harus berupa teks.")
            new["hook_text"] = patch["hook_text"]
        if "captions" in patch:
            new["captions"] = _validated_captions(patch["captions"], before["captions"])
        if "style_override" in patch:
            override = patch["style_override"]
            with domain_errors("style_override"):
                if not isinstance(override, dict):
                    raise err("VALIDATION", "Data 'style_override' harus berupa objek.")
                new["style_override"] = style_diff(base_style, effective_style(base_style, override))
        for field, model in (("subtitles", SubtitleLineV2), ("overlays", Overlay),
                             ("layout_segments", LayoutSegment)):
            if field in patch:
                new[field] = _validated_list(patch, field, model)
        if "pieces" in patch:
            new["pieces"] = _validated_pieces(patch, clip, s, settings)

        pieces_changed = new["pieces"] != before["pieces"]
        new_p = [Piece.model_validate(p) for p in new["pieces"]]
        if not new_p and (pieces_changed or {"subtitles", "overlays", "layout_segments"} & set(patch)):
            raise err("VALIDATION", "Klip tidak punya potongan.")
        duration = output_duration(new_p) if new_p else 0.0
        if pieces_changed:
            old_p = [Piece.model_validate(p) for p in before["pieces"]]
            for field in ("subtitles", "overlays", "layout_segments"):
                if field not in patch:
                    new[field] = remap_output_items(before[field], old_p, new_p)
        for field in ("subtitles", "overlays"):
            if pieces_changed or field in patch:
                new[field] = _fit_items(new[field], duration)
        if pieces_changed or "layout_segments" in patch:
            new["layout_segments"] = _layout_for(new["layout_segments"], new_p)
        if pieces_changed and "subtitles" not in patch:
            with domain_errors("style_override"):
                max_words = effective_style(base_style, new["style_override"])["subtitle"]["words_per_line"]
            new["subtitles"] = _fill_subtitles(new["subtitles"], new_p, _transcript_loader(s, settings), max_words)

        changed = [f for f in PATCHABLE if new[f] != before[f]]
        if not changed:
            return clip
        for f in changed:
            setattr(clip, f, new[f])  # JSON columns are not mutation-tracked: assign new values
        if clip.status == "rendered" and any(f in RERENDER_FIELDS for f in changed):
            clip.needs_rerender = True
        clip.updated_at = utcnow()
        s.add(clip)
        return clip


# ---------- status actions ----------

def _set_status(session: Session, clip: Clip, status: str) -> None:
    clip.status = status
    clip.updated_at = utcnow()
    session.add(clip)


def _approve(session: Session, clip: Clip) -> None:
    if clip.status not in APPROVABLE:
        raise err("INVALID_STATE", f"Klip berstatus '{clip.status}' tidak bisa disetujui.")
    _set_status(session, clip, "approved")
    queue.enqueue(session, "render", clip.project_id, clip.id)


def _reject(session: Session, clip: Clip) -> None:
    """Reject the clip and cancel its queued render job, if any."""
    if clip.status == "rendering":
        raise err("INVALID_STATE", "Klip sedang dirender dan tidak bisa ditolak.")
    _set_status(session, clip, "rejected")
    for job in session.exec(select(Job).where(Job.clip_id == clip.id, Job.type == "render",
                                              Job.status == "queued")).all():
        job.status = "cancelled"
        job.finished_at = queue.now()
        session.add(job)


def _restore(session: Session, clip: Clip) -> None:
    if clip.status != "rejected":
        raise err("INVALID_STATE", f"Hanya klip yang ditolak yang bisa dikembalikan (status: '{clip.status}').")
    _set_status(session, clip, "proposed")


def _render(session: Session, clip: Clip) -> None:
    if clip.status not in RENDERABLE:
        raise err("INVALID_STATE", f"Klip berstatus '{clip.status}' tidak bisa dirender; setujui klip dulu.")
    queue.enqueue(session, "render", clip.project_id, clip.id)


ACTIONS: dict[str, Callable[[Session, Clip], None]] = {
    "approve": _approve, "reject": _reject, "restore": _restore, "render": _render}


def _act(engine: Engine, clip_id: str, action: Callable[[Session, Clip], None]) -> Clip:
    with session_scope(engine) as s:
        clip = fetch(s, Clip, clip_id, "Klip")
        action(s, clip)
        return clip


def approve(engine: Engine, clip_id: str) -> Clip:
    return _act(engine, clip_id, _approve)


def reject(engine: Engine, clip_id: str) -> Clip:
    return _act(engine, clip_id, _reject)


def restore(engine: Engine, clip_id: str) -> Clip:
    return _act(engine, clip_id, _restore)


def bulk(engine: Engine, ids: list[str], action: str) -> dict:
    """Apply one action per clip in its own transaction: `{"ok": [ids], "errors": {id: code}}`."""
    fn = ACTIONS.get(action)
    if fn is None:
        raise err("VALIDATION", f"Aksi massal harus salah satu dari: {', '.join(ACTIONS)}.")
    ok: list[str] = []
    errors: dict[str, str] = {}
    for clip_id in ids:
        try:
            _act(engine, clip_id, fn)
            ok.append(clip_id)
        except AppError as e:  # one bad clip must not stop the others
            errors[clip_id] = e.code
    return {"ok": ok, "errors": errors}


# ---------- manual clips ----------

def create_manual(engine: Engine, project_id: str, start: float, end: float, *,
                  settings: Settings | None = None) -> Clip:
    """A `proposed` clip from a source range chosen by the user (no AI score, empty materials)."""
    settings = settings or get_settings()
    with session_scope(engine) as s:
        project = fetch(s, Project, project_id, "Project")
        if project.status != "ready":
            raise err("INVALID_STATE", f"Klip manual hanya bisa dibuat saat project siap (status: "
                                       f"'{project.status}').")
        with domain_errors("pieces"):
            piece = Piece(project_id=project_id, start=start, end=end)
        if project.duration_sec and piece.end > project.duration_sec + _EPS:
            raise err("VALIDATION", "Akhir klip melewati durasi video.")
        _check_piece_in_ranges(project, piece.start, piece.end, settings)
        loaded = _transcript_loader(s, settings)(project_id)
        if loaded is None:
            raise err("INVALID_STATE", "Transkrip tidak ditemukan; proses ulang project ini.")
        t, tr, source = loaded
        with domain_errors("style"):
            max_words = effective_style(project.style, {})["subtitle"]["words_per_line"]
        ranks = s.exec(select(Clip.rank).where(Clip.project_id == project_id)).all()
        clip = Clip(project_id=project_id, kind="single", origin="manual", rank=max(ranks, default=0) + 1,
                    score=0, llm_score=None, pieces=[piece.model_dump()], layout_segments=_layout_for([], [piece]),
                    hook_text="", hook_options=[],
                    subtitles=_renumber(_piece_lines(t, piece, 0.0, max_words, tr=tr, source=source)),
                    captions={k: {"title": "", "caption": "", "hashtags": []} for k in PLATFORMS},
                    status="proposed")
        s.add(clip)
        return clip
