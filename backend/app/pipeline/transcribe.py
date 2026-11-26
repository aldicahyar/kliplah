"""Chunked, resumable transcription: each ~10 min chunk is a checkpoint."""
from collections.abc import Callable
from pathlib import Path

from app.core.fs import atomic_write_json, read_json
from app.core.models import Transcript, Word
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
               on_progress: Callable[[float], None] | None = None) -> Transcript:
    final = out_dir / "transcript.json"
    cached = read_json(final)
    if cached:
        return Transcript.model_validate(cached)

    ck = out_dir / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    words: list[Word] = []
    for idx, (a, b) in enumerate(chunk_bounds(duration)):
        path = ck / f"transcribe_{model_name}_{idx}.json"
        part = read_json(path)
        if part is None:
            part = [w.model_dump() for w in transcriber.transcribe_chunk(wav, a, b)]
            atomic_write_json(path, part)
        words.extend(Word.model_validate(w) for w in part)
        if on_progress:
            on_progress(round(b / duration, 4))

    words = apply_corrections(words, corrections)
    t = Transcript(language="id", duration=duration, words=words, sentences=split_sentences(words))
    atomic_write_json(final, t.model_dump())
    return t
