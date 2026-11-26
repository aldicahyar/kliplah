"""Three-stage clip edge refinement: word, then sentence, then silence.

Approach adapted from fralapo/clippyme (https://github.com/fralapo/clippyme), MIT-licensed.
Each stage is kept only if it stays inside the bounds and never cuts a word in half.
"""
import math

from app.core.models import Sentence, Word

WORD_SNAP = 0.6
PAD_BEFORE, PAD_AFTER = 0.05, 0.08
SENT_BACK, SENT_FWD = 2.5, 1.5
SIL_REACH, SIL_LEAD, SIL_TAIL = 0.35, 0.04, 0.06
EPS = 1e-6


def _inside_word(t: float, words: list[Word]) -> bool:
    return any(w.start + EPS < t < w.end - EPS for w in words)


def _pad_start(anchor: float, words: list[Word]) -> float:
    prev_end = max((w.end for w in words if w.end <= anchor + EPS), default=-math.inf)
    return max(anchor - PAD_BEFORE, prev_end, 0.0)


def _pad_end(anchor: float, words: list[Word]) -> float:
    next_start = min((w.start for w in words if w.start >= anchor - EPS), default=math.inf)
    return min(anchor + PAD_AFTER, next_start)


def neighbor_bounds(spans: list[tuple[float, float]], idx: int, total: float) -> tuple[float, float]:
    """Edges a clip may not cross: the nearest clips that do NOT overlap it (overlapping clips are separate cuts)."""
    s, e = spans[idx]
    others = [x for i, x in enumerate(spans) if i != idx]
    lower = max((b for a, b in others if b <= s), default=0.0)
    upper = min((a for a, b in others if a >= e), default=total)
    return lower, upper


def snap_edges(start: float, end: float, words: list[Word], sentences: list[Sentence],
               silences: list[tuple[float, float]], lower: float = 0.0, upper: float = math.inf,
               max_len: float = 90.0) -> tuple[float, float]:
    def ok(s: float, e: float) -> bool:
        return (lower - EPS <= s < e <= upper + EPS and e - s <= max_len + EPS
                and not _inside_word(s, words) and not _inside_word(e, words))

    # Stage 1: nearest word edges, plus a little breathing room.
    s_anchor, e_anchor = start, end
    s, e = start, end
    if words:
        ws = min((w.start for w in words), key=lambda x: abs(x - start))
        if abs(ws - start) <= WORD_SNAP:
            s_anchor, s = ws, _pad_start(ws, words)
        we = min((w.end for w in words), key=lambda x: abs(x - end))
        if abs(we - end) <= WORD_SNAP:
            e_anchor, e = we, _pad_end(we, words)
    if not ok(s, e):
        s, e = max(start, lower), min(end, upper)

    # Stage 2: extend to whole sentences within small budgets.
    back = [x.start for x in sentences if x.start <= s_anchor + EPS and s_anchor - x.start <= SENT_BACK]
    fwd = [x.end for x in sentences if x.end >= e_anchor - EPS and x.end - e_anchor <= SENT_FWD]
    new_s = _pad_start(max(back), words) if back and max(back) < s_anchor - EPS else s
    new_e = _pad_end(min(fwd), words) if fwd and min(fwd) > e_anchor + EPS else e
    for cs, ce in ((new_s, new_e), (new_s, e), (s, new_e)):
        if (cs, ce) != (s, e) and ok(cs, ce):
            s, e = cs, ce
            break

    # Stage 3: nudge edges into adjacent silence.
    before = [sil for sil in silences if sil[1] <= s + EPS and s - sil[1] <= SIL_REACH]
    if before:
        a, b = max(before, key=lambda x: x[1])
        cs = max(a, b - SIL_LEAD)
        if ok(cs, e):
            s = cs
    after = [sil for sil in silences if sil[0] >= e - EPS and sil[0] - e <= SIL_REACH]
    if after:
        a, b = min(after, key=lambda x: x[0])
        ce = min(b, a + SIL_TAIL)
        if ok(s, ce):
            e = ce
    return round(s, 3), round(e, 3)
