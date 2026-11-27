"""Piece timeline of a clip and remapping of items on its output timeline.

A clip is a list of ``Piece`` (ranges on original videos) played end to end. Subtitles,
overlays and layout segments live on the clip's output timeline, so every change of
pieces must move them with ``remap_output_items`` (the single remapping function).
All times are rounded to 0.001 s.
"""
import copy
from typing import Any

from pydantic import BaseModel, model_validator

from app.core.errors import AppError

_EPS = 0.0005  # half a millisecond: two times closer than this are the same time


def _r(t: float) -> float:
    return round(t, 3) + 0.0  # "+ 0.0" turns -0.0 into 0.0


class Piece(BaseModel):
    project_id: str
    start: float
    end: float

    @model_validator(mode="after")
    def _round_and_check(self) -> "Piece":
        self.start = _r(self.start)
        self.end = _r(self.end)
        if self.start < 0:
            raise AppError("VALIDATION", "Awal potongan tidak boleh negatif.")
        if self.end <= self.start:
            raise AppError("VALIDATION", "Akhir potongan harus lebih besar dari awalnya.")
        return self


def _offsets(pieces: list[Piece]) -> list[float]:
    """Output-timeline start of each piece."""
    offsets, acc = [], 0.0
    for p in pieces:
        offsets.append(acc)
        acc += p.end - p.start
    return offsets


def output_duration(pieces: list[Piece]) -> float:
    return _r(sum(p.end - p.start for p in pieces))


def output_to_source(pieces: list[Piece], t: float) -> tuple[int, float] | None:
    """(piece index, source time) for output time ``t``; ``None`` outside [0, duration)."""
    if t < 0:
        return None
    for i, (p, off) in enumerate(zip(pieces, _offsets(pieces))):
        if t < off + (p.end - p.start) - _EPS:
            return i, _r(p.start + (t - off))
    return None


def source_to_output(pieces: list[Piece], piece_index: int, src_t: float) -> float | None:
    """Output time of source time ``src_t`` in piece ``piece_index``; ``None`` outside it."""
    p = pieces[piece_index]
    if src_t < p.start - _EPS or src_t > p.end + _EPS:
        return None
    return _r(_offsets(pieces)[piece_index] + (src_t - p.start))


def _map_range(start: float, end: float, old: list[Piece], new: list[Piece]) -> tuple[float, float] | None:
    """Map an output range of ``old`` onto the output timeline of ``new``.

    The range is split per old piece; each source part is cut into elementary segments at
    the boundaries of ``new`` pieces and every segment goes to the first new piece with the
    same project that contains it. Contiguous output segments are merged; if several
    disjoint ranges remain, the longest one (earliest on a tie) is returned.
    """
    segments: list[tuple[float, float]] = []
    new_offsets = _offsets(new)
    for p, off in zip(old, _offsets(old)):
        lo, hi = max(start, off), min(end, off + (p.end - p.start))
        if hi - lo <= _EPS:
            continue
        a, b = p.start + (lo - off), p.start + (hi - off)
        cuts = sorted({a, b, *(t for q in new if q.project_id == p.project_id
                               for t in (q.start, q.end) if a < t < b)})
        for x, y in zip(cuts, cuts[1:]):
            mid = (x + y) / 2
            for q, qoff in zip(new, new_offsets):
                if q.project_id == p.project_id and q.start <= mid <= q.end:
                    segments.append((_r(qoff + x - q.start), _r(qoff + y - q.start)))
                    break
    segments = [s for s in segments if s[1] - s[0] > _EPS]
    if not segments:
        return None
    segments.sort()
    merged = [segments[0]]
    for s, e in segments[1:]:
        ms, me = merged[-1]
        if s <= me + _EPS:
            merged[-1] = (ms, max(me, e))
        else:
            merged.append((s, e))
    return max(merged, key=lambda r: (r[1] - r[0], -r[0]))


def remap_output_items(
    items: list[dict[str, Any]], old: list[Piece], new: list[Piece], min_len: float = 0.05
) -> list[dict[str, Any]]:
    """Move items (``start``/``end``, optional ``words``) from the ``old`` to the ``new`` timeline.

    Items whose remaining part is shorter than ``min_len`` are dropped, words that fall
    outside the remapped item are dropped, other fields are kept. Inputs are not mutated;
    the result is sorted by ``start``.
    """
    out: list[dict[str, Any]] = []
    for item in items:
        mapped = _map_range(item["start"], item["end"], old, new)
        if mapped is None or mapped[1] - mapped[0] < min_len - _EPS:
            continue
        start, end = mapped
        new_item = copy.deepcopy(item)
        new_item["start"], new_item["end"] = start, end
        if "words" in item:
            words = []
            for w in new_item["words"]:
                wm = _map_range(w["start"], w["end"], old, new)
                if wm is None:
                    continue
                ws, we = max(wm[0], start), min(wm[1], end)
                if we - ws <= _EPS:
                    continue
                w["start"], w["end"] = ws, we
                words.append(w)
            new_item["words"] = words
        out.append(new_item)
    out.sort(key=lambda i: i["start"])
    return out
