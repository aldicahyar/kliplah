"""Chunked, resumable transcription: each ~10 min chunk is a checkpoint."""
from collections.abc import Callable
from pathlib import Path

from app.core.fs import atomic_write_json, read_json
from app.core.models import Sentence, Transcript, Word
from app.domain.ranges import RangeFile
from app.pipeline.corrections import apply_corrections
from app.pipeline.sentences import split_sentences
from app.providers.transcriber import Transcriber


def chunk_bounds(duration: float, chunk_sec: float = 600.0) -> list[tuple[float, float]]:
    out, t = [], 0.0
    while t < duration:
        out.append((t, min(t + chunk_sec, duration)))
        t += chunk_sec
    return out


def transcribe(wav: Path, duration: float, out_dir: Path, transcriber: Transcriber, model_name: str,
               corrections: list[tuple[str, str]],
               on_progress: Callable[[float], None] | None = None, language: str | None = None) -> Transcript:
    """Transcribe one audio file; word times are local to that file."""
    final = out_dir / "transcript.json"
    cached = read_json(final)
    if cached:
        return Transcript.model_validate(cached)

    ck = out_dir / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    lang_path = ck / f"transcribe_{model_name}_lang.json"
    resumed_lang: str | None = language or getattr(transcriber, "language", None)
    if resumed_lang is None:
        lang_raw = read_json(lang_path)
        if isinstance(lang_raw, dict):
            cached_lang = lang_raw.get("language")
            if isinstance(cached_lang, str) and cached_lang.strip():
                resumed_lang = cached_lang
                if hasattr(transcriber, "language"):
                    try:
                        transcriber.language = resumed_lang
                    except AttributeError:
                        pass

    words: list[Word] = []
    for idx, (a, b) in enumerate(chunk_bounds(duration)):
        path = ck / f"transcribe_{model_name}_{idx}.json"
        part = read_json(path)
        if part is None:
            part = [w.model_dump() for w in transcriber.transcribe_chunk(wav, a, b)]
            atomic_write_json(path, part)
            detected = getattr(transcriber, "language", None)
            if language is None and detected and resumed_lang is None:
                resumed_lang = detected
                atomic_write_json(lang_path, {"language": detected})
        words.extend(Word.model_validate(w) for w in part)
        if on_progress:
            on_progress(round(b / duration, 4))

    words = apply_corrections(words, corrections)
    used_lang = (
        getattr(transcriber, "language", None)
        or resumed_lang
        or language
        or getattr(transcriber, "target_language", None)
        or "id"
    )
    t = Transcript(language=used_lang, duration=duration, words=words, sentences=split_sentences(words))
    atomic_write_json(final, t.model_dump())
    return t


def merge_range_transcripts(parts: list[tuple[RangeFile, list[Word]]], pause_sec: float, language: str,
                            duration: float) -> Transcript:
    """One transcript on the original video timeline from per-range words (local times).

    Words are shifted by their range start; sentences are split per range, so no sentence spans two
    ranges, then numbered globally with `word_range` indexing the merged word list.
    """
    words: list[Word] = []
    sentences: list[Sentence] = []
    for rf, local in parts:
        shifted = [w.model_copy(update={"start": round(w.start + rf.start, 3), "end": round(w.end + rf.start, 3)})
                   for w in local]
        offset = len(words)
        for s in split_sentences(shifted, pause_sec):
            a, b = s.word_range
            sentences.append(s.model_copy(update={"i": len(sentences), "word_range": (a + offset, b + offset)}))
        words.extend(shifted)
    return Transcript(language=language, duration=duration, words=words, sentences=sentences)
