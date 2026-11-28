"""Claude-manual moment selection (spec 10.1): export the transcript in parts, import a reply per part, finish.

Parts are at most PART_MAX_SEC of audio and recomputed from the transcript on every call (deterministic).
Files under `transcripts/<model>/`: prompts `untuk_claude[_part<k>of<n>].md`, replies `moments/part_<k>.json`.
Export starts a fresh round (old replies deleted); a successful finish deletes the replies too.
`finish_moments` writes `picks.json`, marks the `moments` stage done and queues `process` to run `prepare`.
"""
import shutil
from pathlib import Path

from sqlalchemy import Engine
from sqlmodel import Session

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.fs import atomic_write_json, project_dir, read_json
from app.core.models import Transcript, VideoMeta
from app.db.engine import session_scope
from app.db.tables import Job, Project, utcnow
from app.domain.profiles import profile_for
from app.domain.ranges import SourceMap
from app.pipeline.analyze import focus_block, import_external, profile_criteria_line, transcript_lines
from app.pipeline.llm_json import parse_llm_json, render_prompt
from app.pipeline.signals import load_signals
from app.worker import queue
from app.worker.process import save_picks, transcript_dir

PART_MAX_SEC = 5400.0
OPEN_STATES = ("awaiting_moments", "ready")  # ready = choosing moments again after a re-export


def split_parts(t: Transcript, max_sec: float = PART_MAX_SEC) -> list[tuple[int, int]]:
    """Inclusive sentence ranges; a part's audio span (last.end - first.start) stays within `max_sec`.

    Greedy: a new part starts when the next sentence would push the span over the limit, so a single
    sentence longer than the limit is a part of its own.
    """
    s = t.sentences
    parts: list[tuple[int, int]] = []
    a = 0
    for i in range(1, len(s) + 1):
        if i == len(s) or s[i].end - s[a].start > max_sec:
            parts.append((a, i - 1))
            a = i
    return parts


def _filename(k: int, n: int) -> str:
    return "untuk_claude.md" if n == 1 else f"untuk_claude_part{k}of{n}.md"


class _Ctx:
    """Project row plus its files, loaded for one call."""

    def __init__(self, session: Session, project_id: str, settings: Settings):
        project = session.get(Project, project_id)
        if project is None:
            raise AppError("NOT_FOUND", f"Project {project_id} tidak ditemukan", status=404)
        if project.status not in OPEN_STATES:
            raise AppError("INVALID_STATE", f"Project berstatus '{project.status}'; pemilihan momen manual hanya "
                                            "bisa dilakukan saat project menunggu momen atau sudah siap.",
                           status=409)
        self.project = project
        self.dir = project_dir(settings, project_id)
        self.tdir = transcript_dir(settings, project)
        raw = read_json(self.tdir / "transcript.json")
        if raw is None:
            raise AppError("INVALID_STATE", "Transkrip tidak ditemukan; proses ulang project ini.", status=409)
        self.transcript = Transcript.model_validate(raw)
        self.parts = split_parts(self.transcript)
        if not self.parts:
            raise AppError("INVALID_STATE", "Transkrip kosong; tidak ada kalimat untuk dipilih.", status=409)

    @property
    def min_sec(self) -> int:
        return self.project.min_clip_sec or profile_for(self.project.content_type).min_sec

    @property
    def max_sec(self) -> int:
        return self.project.max_clip_sec or profile_for(self.project.content_type).max_sec

    @property
    def moments_dir(self) -> Path:
        """Replies index into this model's transcript, so they live next to it; one dir = one round."""
        return self.tdir / "moments"

    def part_path(self, k: int) -> Path:
        return self.moments_dir / f"part_{k}.json"

    def status(self) -> dict:
        n = len(self.parts)
        have = [k for k in range(1, n + 1) if self.part_path(k).exists()]
        return {"imported": have, "missing": [k for k in range(1, n + 1) if k not in have]}


def export_parts(engine: Engine, project_id: str, settings: Settings | None = None) -> list[dict]:
    settings = settings or get_settings()
    with Session(engine) as session:
        ctx = _Ctx(session, project_id, settings)
    if ctx.moments_dir.exists():
        try:
            shutil.rmtree(ctx.moments_dir)  # a new export starts a new round of replies
        except OSError as e:
            raise AppError("INVALID_STATE", f"Folder balasan momen tidak bisa dibersihkan ({e}).",
                           status=409) from e
    n = len(ctx.parts)
    out = []
    criteria = profile_criteria_line(ctx.project.content_type)
    focus = focus_block(ctx.project.focus_prompt)
    for k, (a, b) in enumerate(ctx.parts, start=1):
        note = f"Ini bagian {k} dari {n}; hanya pakai kalimat [{a}]–[{b}]." if n > 1 else ""
        text = render_prompt("claude_manual", max_candidates=ctx.project.max_clips, min_sec=ctx.min_sec,
                             max_sec=ctx.max_sec, part_note=note, criteria=criteria, focus=focus,
                             transcript=transcript_lines(ctx.transcript, a, b + 1))
        name = _filename(k, n)
        ctx.tdir.mkdir(parents=True, exist_ok=True)
        (ctx.tdir / name).write_text(text, encoding="utf-8")
        out.append({"part": k, "of": n, "filename": name, "text": text})
    return out


def _check_rows(data: dict, a: int, b: int) -> None:
    rows = data.get("clips")
    if not isinstance(rows, list) or not rows:
        raise AppError("INVALID_IMPORT", "Balasan tidak berisi daftar \"clips\" atau daftarnya kosong.")
    for i, r in enumerate(rows, start=1):
        try:
            s, e = int(r["start_sentence"]), int(r["end_sentence"])
        except (KeyError, TypeError, ValueError):
            msg = f"Klip ke-{i} tidak punya start_sentence/end_sentence yang valid."
            raise AppError("INVALID_IMPORT", msg) from None
        if not (a <= s <= b and a <= e <= b):
            raise AppError("INVALID_IMPORT", f"Klip ke-{i} memakai kalimat [{s}]–[{e}], padahal bagian ini "
                                             f"hanya berisi kalimat [{a}]–[{b}].")


def import_part(engine: Engine, project_id: str, part: int, reply_text: str,
                settings: Settings | None = None) -> dict:
    """Validate and store one part's reply (replacing an earlier one); returns imported/missing part numbers."""
    settings = settings or get_settings()
    with Session(engine) as session:
        ctx = _Ctx(session, project_id, settings)
    n = len(ctx.parts)
    if not 1 <= part <= n:
        raise AppError("INVALID_IMPORT", f"Bagian {part} tidak ada; transkrip ini terdiri dari {n} bagian.")
    data = parse_llm_json(reply_text)
    if data is None:
        raise AppError("INVALID_IMPORT", "Balasan bukan JSON. Pastikan menyalin seluruh balasan Claude.")
    _check_rows(data, *ctx.parts[part - 1])
    ctx.part_path(part).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(ctx.part_path(part), data)
    return ctx.status()


def finish_moments(engine: Engine, project_id: str, allow_missing: bool = False,
                   settings: Settings | None = None) -> Job:
    """Merge the imported parts, keep the top clips by the M1 rules and queue `process` to continue at prepare."""
    settings = settings or get_settings()
    with session_scope(engine) as session:
        ctx = _Ctx(session, project_id, settings)
        st = ctx.status()
        if st["missing"] and not (allow_missing and st["imported"]):
            missing = ", ".join(map(str, st["missing"]))
            raise AppError("INVALID_STATE", f"Balasan untuk bagian {missing} belum diimpor.", status=409)
        rows = [r for k in st["imported"] for r in (read_json(ctx.part_path(k)) or {}).get("clips", [])]
        meta_raw = read_json(ctx.dir / "meta.json")
        if meta_raw is None:
            raise AppError("INVALID_STATE", "Data video (meta.json) tidak ditemukan; proses ulang project ini.",
                           status=409)
        p = ctx.project
        meta = VideoMeta.model_validate(meta_raw)
        signals = load_signals(ctx.dir / "signals.json")
        smap = SourceMap.from_meta(meta, ctx.dir, ctx.transcript)
        try:
            picks = import_external(ctx.transcript, meta, {"clips": rows}, p.max_clips, ctx.min_sec, ctx.max_sec,
                                    signals=signals, content_type=p.content_type, source_map=smap)
        except ValueError as e:
            raise AppError("INVALID_IMPORT", str(e)) from e
        save_picks(ctx.dir / "picks.json", picks)
        p.last_completed_stage = "moments"
        p.status = "processing"
        p.updated_at = utcnow()
        session.add(p)
        job = queue.enqueue(session, "process", project_id)
    shutil.rmtree(ctx.moments_dir, ignore_errors=True)  # this round is used up; picks.json now holds the result
    return job
