"""Layout segments of a clip (spec 4.7) and their mapping to M1 reframe segments.

Layout segments live on the clip's output timeline. Until S4 the renderer is the M1 one,
which only knows ``face_crop`` (M1 ``track``) and ``fit_blur`` (M1 ``wide``/``general``).
"""
import copy
from typing import Literal, get_args

from pydantic import BaseModel

from app.core.errors import AppError
from app.core.models import ReframeSegment

Layout = Literal["face_crop", "fit_blur", "fit_solid", "facecam_gameplay", "gameplay_full", "facecam_full", "stack"]
LAYOUTS: tuple[str, ...] = get_args(Layout)
MIN_SEGMENT_SEC = 0.05


class LayoutSegment(BaseModel):
    start: float
    end: float
    layout: Layout
    params: dict = {}
    piece_index: int = 0
    user_override: bool = False


def from_reframe_segments(segs: list[ReframeSegment]) -> list[LayoutSegment]:
    out = []
    for s in segs:
        if s.mode == "track":
            out.append(LayoutSegment(start=s.start, end=s.end, layout="face_crop",
                                     params={"x": s.x, "zoom": s.zoom}, user_override=s.user_override))
        else:
            out.append(LayoutSegment(start=s.start, end=s.end, layout="fit_blur", user_override=s.user_override))
    return out


def to_reframe_segments(segs: list[LayoutSegment]) -> list[ReframeSegment]:
    out = []
    for s in segs:
        if s.layout == "face_crop":
            out.append(ReframeSegment(start=s.start, end=s.end, mode="track", x=s.params.get("x", 0.5),
                                      zoom=s.params.get("zoom", 1.0), user_override=s.user_override))
        elif s.layout == "fit_blur":
            out.append(ReframeSegment(start=s.start, end=s.end, mode="wide", x=0.5, user_override=s.user_override))
        else:
            raise AppError("VALIDATION", f"Layout {s.layout} baru tersedia di S4")
    return out


def _r(t: float) -> float:
    return round(t, 3) + 0.0


def normalize_layout_segments(segments: list[dict], duration: float) -> list[dict]:
    """Make segments tile the output timeline [0, duration] exactly.

    Sorted by start, clamped to [0, duration] and rounded to 0.001 s; segments shorter than 0.05 s are dropped,
    an overlap is cut off the later segment, a gap is closed by extending the preceding segment, the first
    segment starts at 0 and the last ends at ``duration``. No segment left -> one ``fit_blur`` segment.
    """
    duration = _r(max(duration, 0.0))
    out: list[dict] = []
    for seg in sorted(segments, key=lambda s: (s["start"], s["end"])):
        start = _r(min(max(seg["start"], 0.0), duration))
        end = _r(min(max(seg["end"], 0.0), duration))
        if out:
            start = max(start, out[-1]["end"])
        if end - start < MIN_SEGMENT_SEC - 0.0005:
            continue
        if out:
            out[-1]["end"] = start  # close the gap before this segment
        out.append({**copy.deepcopy(seg), "start": start, "end": end})
    if not out:
        return [LayoutSegment(start=0.0, end=duration, layout="fit_blur").model_dump()]
    out[0]["start"] = 0.0
    out[-1]["end"] = duration
    return out
