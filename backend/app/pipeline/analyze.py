"""Find the moments worth clipping: LLM per window, then code-side validation and ranking."""
import json
import logging
from difflib import SequenceMatcher
from pathlib import Path

from pydantic import ValidationError

from app.core.fs import atomic_write_json, read_json
from app.core.models import Candidate, HeatPoint, Materials, PlatformCaption, Transcript, VideoMeta, Window
from app.pipeline.llm_json import (
    FIND_MOMENTS_SCHEMA,
    MATERIALS_SCHEMA,
    coerce_candidate,
    parse_llm_json,
    render_prompt,
)
from app.pipeline.texttiling import topic_boundaries
from app.pipeline.windows import make_windows
from app.providers.llm import LLMClient, LLMUnavailable

log = logging.getLogger(__name__)

RETRY_SUFFIX = "\n\nRespond with JSON only. No explanations, no markdown."
MAX_BONUS = 8
TIME_OVERLAP_MAX = 0.40
TEXT_SIMILARITY_MAX = 0.70


def _mmss(sec: float) -> str:
    return f"{int(sec) // 60:02d}:{int(sec) % 60:02d}"


def _clip_text(t: Transcript, s: int, e: int) -> str:
    return " ".join(x.text for x in t.sentences[s: e + 1])


def _model_name(llm: LLMClient) -> str:
    return str(getattr(llm, "model", "llm")).replace(":", "-").replace("/", "-")


def _ask_json(llm: LLMClient, prompt: str, schema: dict, attempts: int) -> dict | None:
    for i in range(attempts):
        data = parse_llm_json(llm.complete_json(prompt + (RETRY_SUFFIX if i else ""), schema))
        if data is not None:
            return data
    return None


def find_candidates(t: Transcript, windows: list[Window], llm: LLMClient, min_sec: int, max_sec: int,
                    max_per_window: int = 4, checkpoint_dir: Path | None = None) -> list[Candidate]:
    n = len(t.sentences)
    out: list[Candidate] = []
    for w in windows:
        key = f"analyze_{_model_name(llm)}_{min_sec}-{max_sec}_{w.index}.json"
        ck = checkpoint_dir / key if checkpoint_dir else None
        cached = read_json(ck) if ck else None
        if cached is not None:
            out.extend(Candidate.model_validate(c) for c in cached)
            continue
        lines = [f"[{s.i - w.start_sentence}] ({_mmss(s.start)}) {s.text}"
                 for s in t.sentences[w.start_sentence: w.end_sentence + 1]]
        prompt = render_prompt("find_moments", transcript_json=json.dumps("\n".join(lines), ensure_ascii=False),
                               max_candidates=max_per_window, min_sec=min_sec, max_sec=max_sec)
        data = _ask_json(llm, prompt, FIND_MOMENTS_SCHEMA, attempts=3)
        if data is None:
            log.warning("window %d skipped: model never returned valid JSON", w.index)
            continue
        found: list[Candidate] = []
        for raw in (data.get("candidates") or [])[:max_per_window]:
            c = coerce_candidate(raw, n, w.start_sentence) if isinstance(raw, dict) else None
            if c is None or c["end_sentence"] > w.end_sentence:
                continue
            found.append(Candidate(start=t.sentences[c["start_sentence"]].start,
                                   end=t.sentences[c["end_sentence"]].end, llm_score=c.pop("score"), **c))
        if ck:
            atomic_write_json(ck, [c.model_dump() for c in found])
        out.extend(found)
    return out


def _span(t: Transcript, c: Candidate, s: int, e: int) -> Candidate:
    return c.model_copy(update={"start_sentence": s, "end_sentence": e,
                                "start": t.sentences[s].start, "end": t.sentences[e].end})


def fit_duration(c: Candidate, t: Transcript, min_sec: int, max_sec: int) -> Candidate | None:
    s, e = c.start_sentence, c.end_sentence
    last = len(t.sentences) - 1
    dur = lambda a, b: t.sentences[b].end - t.sentences[a].start  # noqa: E731
    go_right = True
    while dur(s, e) < min_sec:
        if go_right and e < last:
            e += 1
        elif s > 0:
            s -= 1
        elif e < last:
            e += 1
        else:
            return None
        go_right = not go_right
    if dur(s, e) > max_sec:
        return None
    return _span(t, c, s, e)


def heatmap_bonus(start: float, end: float, heatmap: list[HeatPoint]) -> int:
    vals = [h.value for h in heatmap if h.start < end and h.end > start]
    if not vals:
        return 0
    return round(MAX_BONUS * min(max((max(vals) - 0.7) / 0.3, 0.0), 1.0))


def dedupe(cands: list[Candidate], t: Transcript) -> list[Candidate]:
    kept: list[Candidate] = []
    for c in sorted(cands, key=lambda x: x.score, reverse=True):
        text = _clip_text(t, c.start_sentence, c.end_sentence)
        dup = False
        for k in kept:
            inter = min(c.end, k.end) - max(c.start, k.start)
            shorter = min(c.end - c.start, k.end - k.start) or 1e-9
            if inter > 0 and inter / shorter > TIME_OVERLAP_MAX:
                dup = True
            elif SequenceMatcher(None, text, _clip_text(t, k.start_sentence, k.end_sentence)).ratio() > TEXT_SIMILARITY_MAX:
                dup = True
            if dup:
                break
        if not dup:
            kept.append(c)
    return kept


def select_top(cands: list[Candidate], heatmap: list[HeatPoint], n: int) -> list[Candidate]:
    scored = []
    for c in cands:
        bonus = heatmap_bonus(c.start, c.end, heatmap)
        scored.append(c.model_copy(update={"heatmap_bonus": bonus,
                                           "score": min(max((c.llm_score or 0) + bonus, 0), 100)}))
    return sorted(scored, key=lambda c: c.score, reverse=True)[:n]


def fallback_candidates(t: Transcript, heatmap: list[HeatPoint], min_sec: int = 15, max_sec: int = 90,
                        n: int = 12) -> list[Candidate]:
    """Topic segments ranked by heatmap, for when the LLM is down or finds nothing."""
    sents = t.sentences
    cuts = set(topic_boundaries(sents))
    segs: list[tuple[int, int]] = []
    a = 0
    n_s = len(sents)
    for i in range(1, n_s + 1):
        if not (i == n_s or i in cuts or sents[i].end - sents[a].start > max_sec):
            continue
        dur = sents[i - 1].end - sents[a].start
        if dur > max_sec:
            a = i  # a single sentence longer than a clip may be
        elif dur >= min_sec:
            segs.append((a, i - 1))
            a = i
        elif i < n_s and sents[i].end - sents[a].start > max_sec:
            a = i  # too short, and growing it would exceed max: drop it
        # otherwise: too short at a topic cut, keep growing into the next topic
    out = []
    for s, e in segs:
        start, end = sents[s].start, sents[e].end
        v = max([h.value for h in heatmap if h.start < end and h.end > start] or [0.0])
        bonus = heatmap_bonus(start, end, heatmap)
        out.append(Candidate(start_sentence=s, end_sentence=e, start=start, end=end, llm_score=None,
                             category=None, reason="Dipilih dari pergantian topik (tanpa skor AI)", hook="",
                             heatmap_bonus=bonus, score=round(100 * v), no_ai=True))
    return sorted(out, key=lambda c: c.score, reverse=True)[:n]


def _default_materials(t: Transcript, c: Candidate) -> Materials:
    hook = " ".join(_clip_text(t, c.start_sentence, c.end_sentence).split()[:8]).rstrip(".,")
    empty = PlatformCaption(title="", caption="", hashtags=[])
    return Materials(hook=hook, tiktok=empty, reels=empty, shorts=empty)


def _llm_materials(c: Candidate, t: Transcript, llm: LLMClient) -> Materials | None:
    """Materials from the LLM, or None when it never produced valid ones."""
    text = json.dumps(_clip_text(t, c.start_sentence, c.end_sentence), ensure_ascii=False)
    data = _ask_json(llm, render_prompt("write_materials", clip_text_json=text), MATERIALS_SCHEMA, attempts=2)
    if data is not None:
        try:
            return Materials.model_validate(data)
        except ValidationError:
            log.warning("materials JSON did not match schema; using defaults")
    return None


def write_materials(c: Candidate, t: Transcript, llm: LLMClient) -> Materials:
    return _llm_materials(c, t, llm) or _default_materials(t, c)


def transcript_lines(t: Transcript) -> str:
    """The whole transcript as numbered lines: `[n] (mm:ss) text`."""
    return "\n".join(f"[{s.i}] ({_mmss(s.start)}) {s.text}" for s in t.sentences)


def import_external(t: Transcript, meta: VideoMeta, raw: dict | list, max_clips: int, min_sec: int,
                    max_sec: int) -> list[tuple[Candidate, Materials]]:
    """Clips chosen outside the app (e.g. by Claude via the user's subscription), held to the same rules."""
    rows = raw.get("clips") if isinstance(raw, dict) else raw
    n = len(t.sentences)
    found: list[tuple[Candidate, dict]] = []
    for r in rows if isinstance(rows, list) else []:
        c = coerce_candidate(r, n, 0) if isinstance(r, dict) and n else None
        if c is None:
            continue
        cand = Candidate(start=t.sentences[c["start_sentence"]].start, end=t.sentences[c["end_sentence"]].end,
                         llm_score=c.pop("score"), **c)
        found.append((cand, r))
    fitted = [(f, r) for c, r in found if (f := fit_duration(c, t, min_sec, max_sec)) is not None]
    if not fitted:
        raise ValueError("Balasan tidak berisi klip yang valid (cek nomor kalimat dan durasinya).")
    raw_by_span = {(c.start_sentence, c.end_sentence, c.hook): r for c, r in fitted}
    chosen = dedupe(select_top([c for c, _ in fitted], meta.heatmap, len(fitted)), t)[:max_clips]
    out = []
    for c in chosen:
        r = raw_by_span.get((c.start_sentence, c.end_sentence, c.hook), {})
        default = _default_materials(t, c)
        caps = {}
        for p in ("tiktok", "reels", "shorts"):
            try:
                caps[p] = PlatformCaption.model_validate(r.get(p))
            except ValidationError:
                caps[p] = default.tiktok
        out.append((c, Materials(hook=c.hook or default.hook, **caps)))
    return out


def analyze(t: Transcript, meta: VideoMeta, llm: LLMClient, max_clips: int, min_sec: int, max_sec: int,
            checkpoint_dir: Path) -> list[tuple[Candidate, Materials]]:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    windows = make_windows(t.sentences, topic_boundaries(t.sentences))
    llm_ok = True
    try:
        raw = find_candidates(t, windows, llm, min_sec, max_sec, checkpoint_dir=checkpoint_dir)
    except LLMUnavailable as e:
        log.warning("LLM unavailable, using topic fallback: %s", e)
        raw, llm_ok = [], False
    fitted = [f for c in raw if (f := fit_duration(c, t, min_sec, max_sec)) is not None]
    chosen = dedupe(select_top(fitted, meta.heatmap, len(fitted)), t)[:max_clips]
    if not chosen:
        chosen = fallback_candidates(t, meta.heatmap, min_sec=min_sec, max_sec=max_sec, n=max_clips)
        llm_ok = False if not raw else llm_ok
    out = []
    for c in chosen:
        if c.no_ai or not llm_ok:
            out.append((c, _default_materials(t, c)))
            continue
        ck = checkpoint_dir / f"materials_{_model_name(llm)}_{c.start_sentence}_{c.end_sentence}.json"
        cached = read_json(ck)
        if cached is not None:
            m = Materials.model_validate(cached)
        else:
            try:
                m = _llm_materials(c, t, llm)
            except LLMUnavailable:
                m = None
            if m is None:
                m = _default_materials(t, c)  # not cached: a later run retries the LLM
            else:
                atomic_write_json(ck, m.model_dump())
        out.append((c.model_copy(update={"hook": m.hook or c.hook}), m))
    return out
