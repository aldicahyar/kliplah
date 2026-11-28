"""Build v2 subtitle lines for a clip span, with optional translated text and original overlap."""
from app.core.models import Sentence, Transcript
from app.domain.clip_doc import subtitles_from_m1
from app.domain.translation import Translation, original_text_between, translated_view
from app.pipeline.subtitles import EPS, group_lines


def _has_translated_overlap(sentences: list[Sentence], tr: Translation, line_start: float, line_end: float) -> bool:
    for s in sentences:
        if s.end > line_start + EPS and s.start < line_end - EPS:
            raw = tr.lines.get(str(s.i))
            if isinstance(raw, str) and raw.strip():
                return True
    return False


def clip_subtitle_lines(
    t: Transcript,
    tr: Translation | None,
    start: float,
    end: float,
    max_words: int,
    source: str = "whisper",
) -> list[dict]:
    """Group transcript words in ``[start, end]`` into v2 subtitle lines (1-based ``id``, times relative to ``start``).

    When ``tr`` targets a different language than ``t``, translated lines use ``source="translated"`` and fill
    ``text_original`` from overlapping source words, while untranslated sentences fall back to ``source``.
    """
    if tr is None or tr.language == t.language:
        dicts = subtitles_from_m1(group_lines(t.words, start, end, max_words=max_words))
        if source != "whisper":
            for item in dicts:
                item["source"] = source
        return dicts

    view = translated_view(t, tr)
    m1_lines = group_lines(view.words, start, end, max_words=max_words)
    dicts = subtitles_from_m1(m1_lines)
    for line, item in zip(m1_lines, dicts):
        line_start = round(start + line.start, 3)
        line_end = round(start + line.end, 3)
        if _has_translated_overlap(t.sentences, tr, line_start, line_end):
            item["source"] = "translated"
            item["text_original"] = original_text_between(t, line_start, line_end) or None
        else:
            item["source"] = source
            item["text_original"] = None
    return dicts
