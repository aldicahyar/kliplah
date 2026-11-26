"""Split a transcript into overlapping ~5 min windows for the LLM, ending at topic changes when close."""
from app.core.models import Sentence, Window

SNAP_RANGE_SEC = 60.0


def make_windows(sentences: list[Sentence], boundaries: list[int], size_sec: float = 300.0,
                 overlap_sec: float = 45.0) -> list[Window]:
    if not sentences:
        return []
    last = len(sentences) - 1
    out: list[Window] = []
    start_idx = 0
    while True:
        t0 = sentences[start_idx].start
        if sentences[last].end - t0 <= size_sec:
            out.append(Window(index=len(out), start_sentence=start_idx, end_sentence=last))
            return out
        target = t0 + size_sec
        end_idx = start_idx
        while end_idx + 1 <= last and sentences[end_idx + 1].end <= target:
            end_idx += 1
        near = [b for b in boundaries if start_idx + 1 < b <= last and abs(sentences[b].start - target) <= SNAP_RANGE_SEC]
        if near:
            end_idx = min(near, key=lambda b: abs(sentences[b].start - target)) - 1
        out.append(Window(index=len(out), start_sentence=start_idx, end_sentence=end_idx))
        cut = sentences[end_idx].end - overlap_sec
        nxt = next(i for i in range(start_idx + 1, last + 1) if sentences[i].start >= cut or i == end_idx + 1)
        start_idx = nxt
