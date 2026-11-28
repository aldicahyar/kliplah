"""Sentence-keyed translation artifact (translation.json) and translated transcript view."""
from pathlib import Path

from pydantic import BaseModel, ValidationError

from app.core.fs import atomic_write_json, read_json
from app.core.models import Sentence, Transcript, Word
from app.pipeline.youtube_subs import distribute_words


class Translation(BaseModel):
    """Per-sentence translation stored separately from transcript.json."""

    language: str
    source_language: str
    lines: dict[str, str] = {}


def translation_path(tdir: Path) -> Path:
    """Return the path to translation.json inside a transcript directory."""
    return Path(tdir) / "translation.json"


def load_translation(tdir: Path, target: str, source: str) -> Translation:
    """Load translation.json for (source -> target), or return an empty Translation if missing/corrupt/mismatched."""
    raw = read_json(translation_path(tdir))
    if not isinstance(raw, dict):
        return Translation(language=target, source_language=source)
    try:
        tr = Translation.model_validate(raw)
    except (ValidationError, TypeError):
        return Translation(language=target, source_language=source)
    if tr.language != target or tr.source_language != source:
        return Translation(language=target, source_language=source)
    return tr


def save_translation(tdir: Path, tr: Translation) -> None:
    """Atomically write ``tr`` to ``tdir / translation.json``."""
    path = translation_path(tdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, tr.model_dump())


def merge_lines(tr: Translation, lines: dict[int, str], valid: set[int] | None = None) -> int:
    """Clean and merge translated lines into ``tr``; return count of added or changed entries."""
    changed = 0
    for k, val in lines.items():
        if valid is not None and k not in valid:
            continue
        if not isinstance(val, str):
            continue
        cleaned = " ".join(val.split())
        if not cleaned:
            continue
        key = str(k)
        if tr.lines.get(key) != cleaned:
            tr.lines[key] = cleaned
            changed += 1
    return changed


def missing_sentences(t: Transcript, tr: Translation, indices: set[int] | None = None) -> list[Sentence]:
    """Return sentences in ``t`` (optionally filtered by ``indices``) that have no translation yet."""
    return [
        s for s in t.sentences
        if (indices is None or s.i in indices) and not (tr.lines.get(str(s.i)) or "").strip()
    ]


def translated_view(t: Transcript, tr: Translation) -> Transcript:
    """Build a Transcript with translated text and approx-timed words, keeping sentence timings unchanged."""
    words: list[Word] = []
    sentences: list[Sentence] = []
    cursor = 0
    for s in t.sentences:
        a, b = s.word_range
        if a > cursor:
            words.extend(w.model_copy() for w in t.words[cursor:a])
        raw = tr.lines.get(str(s.i))
        cleaned = raw.strip() if isinstance(raw, str) else ""
        if cleaned:
            tokens = cleaned.split()
            seg_words = distribute_words(tokens, s.start, s.end, approx=True) if tokens else []
            text = cleaned
        else:
            seg_words = [w.model_copy() for w in t.words[a:b]]
            text = s.text
        start_idx = len(words)
        words.extend(seg_words)
        sentences.append(s.model_copy(update={"text": text, "word_range": (start_idx, len(words))}))
        cursor = max(cursor, b)
    if cursor < len(t.words):
        words.extend(w.model_copy() for w in t.words[cursor:])
    return Transcript(language=tr.language, duration=t.duration, words=words, sentences=sentences)


def original_text_between(t: Transcript, start: float, end: float) -> str:
    """Join original words whose midpoint falls inside ``[start, end)`` with spaces."""
    return " ".join(
        w.w for w in t.words
        if start <= (w.start + w.end) / 2.0 < end
    )
