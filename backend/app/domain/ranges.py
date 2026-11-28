"""Time ranges of a project and the map between the original video timeline and range files.

A project may download only some ranges of a video; each range is stored as its own file
whose local time 0 is the range start. Everything outside ffmpeg/MediaPipe calls stays on
the original video timeline, and ``SourceMap`` converts at that boundary. A project without
ranges maps to a single range covering ``source.mp4``. All times are rounded to 0.001 s.
"""
import math
import re
from dataclasses import dataclass
from pathlib import Path

from app.core.errors import AppError
from app.core.models import Transcript, VideoMeta

MAX_RANGES = 10
MIN_RANGE_LEN = 10.0  # seconds, after clamping and merging
MERGE_GAP = 1.0  # ranges at most this far apart are merged
SPAN_TOLERANCE = 0.05  # seconds a span may stick out of its range

_SECONDS = re.compile(r"^(\d+(?:\.\d+)?)$")
_MM_SS = re.compile(r"^(\d+):([0-5]\d(?:\.\d+)?)$")
_HH_MM_SS = re.compile(r"^(\d{1,2}):([0-5]\d):([0-5]\d(?:\.\d+)?)$")


def _r(t: float) -> float:
    return round(t, 3) + 0.0  # "+ 0.0" turns -0.0 into 0.0


def _invalid(message: str) -> AppError:
    return AppError("VALIDATION", message, status=422)


def _parse_time(part: str) -> float | None:
    part = part.strip()
    if m := _SECONDS.match(part):
        return float(m.group(1))
    if m := _MM_SS.match(part):
        return int(m.group(1)) * 60 + float(m.group(2))
    if m := _HH_MM_SS.match(part):
        return int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    return None


def _is_real(v: object) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def parse_range(text: str) -> tuple[float, float]:
    """Parse ``HH:MM:SS-HH:MM:SS``, ``MM:SS-MM:SS`` or ``1800-3600`` into (start, end) seconds."""
    parts = text.split("-")
    if len(parts) != 2:
        raise _invalid(f"Format rentang '{text}' tidak valid. Contoh: 00:30:00-01:00:00.")
    start, end = (_parse_time(p) for p in parts)
    if start is None or end is None:
        raise _invalid(f"Format rentang '{text}' tidak valid. Contoh: 00:30:00-01:00:00.")
    start, end = _r(start), _r(end)
    if start >= end:
        raise _invalid(f"Rentang '{text}': waktu mulai harus sebelum waktu selesai.")
    return start, end


def normalize_ranges(ranges: list[list[float]] | None,
                     duration: float | None) -> list[tuple[float, float]] | None:
    """Sort, clamp to [0, duration], merge overlapping or ≤ 1 s apart, drop ranges < 10 s.

    ``None`` or empty means the whole video (returns ``None``).
    """
    if not ranges:
        return None
    if len(ranges) > MAX_RANGES:
        raise _invalid(f"Maksimal {MAX_RANGES} rentang waktu.")
    items: list[tuple[float, float]] = []
    for i, rng in enumerate(ranges):
        if not (isinstance(rng, (list, tuple)) and len(rng) == 2 and all(_is_real(v) for v in rng)):
            raise _invalid(f"Rentang ke-{i + 1} tidak valid: harus berisi [mulai, selesai] berupa angka detik.")
        start, end = max(0.0, _r(float(rng[0]))), _r(float(rng[1]))
        if duration is not None:
            end = min(end, _r(duration))
        if end > start:
            items.append((start, end))
    items.sort()
    merged: list[tuple[float, float]] = []
    for start, end in items:
        if merged and start - merged[-1][1] <= MERGE_GAP + 0.0005:  # exact at 0.001 s
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    kept = [(s, e) for s, e in merged if e - s >= MIN_RANGE_LEN - 0.0005]
    if not kept:
        raise _invalid(f"Tidak ada rentang yang tersisa: setiap rentang minimal {MIN_RANGE_LEN:g} detik "
                       "di dalam durasi video.")
    return kept


@dataclass(frozen=True)
class RangeFile:
    index: int
    start: float  # original video time where this file starts
    end: float
    file: str  # relative to the project dir


class SourceMap:
    """Maps original-video time to (range file, local time) and back."""

    def __init__(self, files: list[RangeFile], project_dir: Path):
        self._files = sorted(files, key=lambda f: f.start)
        self._project_dir = Path(project_dir)

    @classmethod
    def from_meta(cls, meta: VideoMeta, project_dir: Path,
                  transcript: Transcript | None = None) -> "SourceMap":
        if not meta.ranges:
            t_end = 0.0
            if transcript is not None:
                t_end = (transcript.sentences[-1].end if transcript.sentences
                         else (transcript.words[-1].end if transcript.words else transcript.duration))
            end = _r(max(meta.duration, t_end) if meta.duration else t_end)
            return cls([RangeFile(0, 0.0, end, "source.mp4")], project_dir)
        files = [RangeFile(int(r["index"]), _r(float(r["start"])), _r(float(r["end"])), str(r["file"]))
                 for r in meta.ranges]
        return cls(files, project_dir)

    @property
    def files(self) -> list[RangeFile]:
        return list(self._files)

    def locate(self, t: float) -> RangeFile | None:
        """Range containing ``t``: [start, end), the last range's end inclusive."""
        last = len(self._files) - 1
        for i, rf in enumerate(self._files):
            if rf.start <= t < rf.end or (i == last and t == rf.end):
                return rf
        return None

    def locate_span(self, start: float, end: float) -> RangeFile | None:
        """The single range containing the whole span, with 0.05 s tolerance."""
        if start > end:
            return None
        for rf in self._files:
            if rf.start - SPAN_TOLERANCE <= start and end <= rf.end + SPAN_TOLERANCE:
                return rf
        return None

    def path(self, rf: RangeFile) -> Path:
        return self._project_dir / rf.file

    def to_local(self, rf: RangeFile, t: float) -> float:
        return _r(t - rf.start)

    def to_original(self, rf: RangeFile, local_t: float) -> float:
        return _r(local_t + rf.start)
