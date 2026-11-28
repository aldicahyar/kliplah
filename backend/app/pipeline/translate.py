"""Batch LLM translation with strict reply validation, recursive batch splitting, and atomic checkpoints."""
from collections.abc import Callable
import hashlib
import json
import logging
import re
from pathlib import Path

from app.core.fs import atomic_write_json, read_json
from app.core.models import Sentence
from app.domain.languages import language_name, safe_normalize_language
from app.pipeline.llm_json import parse_llm_json, render_prompt
from app.providers.llm import LLMClient, LLMTimeout

log = logging.getLogger(__name__)

BATCH_SIZE = 40
MAX_EXPANSION = 4.0
MAX_EXPANSION_PAD = 20
_BRACKET_TAG_RE = re.compile(r"\[[^\]]*\]|\([^)]*\)|\{[^}]*\}")

RETRY_SUFFIX = "\n\nRespond with JSON only. No explanations, no markdown. Keep the exact same line count and n numbers."

TRANSLATE_SCHEMA = {
    "type": "object",
    "properties": {
        "lines": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"n": {"type": "integer"}, "text": {"type": "string"}},
                "required": ["n", "text"],
            },
        }
    },
    "required": ["lines"],
}


def _model_name(llm: LLMClient) -> str:
    return str(getattr(llm, "model", None) or "llm").replace(":", "-").replace("/", "-")


def _batch_hash(source: str, target: str, batch: list[Sentence]) -> str:
    payload = json.dumps(
        {"source": source, "target": target, "lines": [[s.i, s.text] for s in batch]},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def _checkpoint_path(checkpoint_dir: Path, llm: LLMClient, source: str, target: str, batch: list[Sentence]) -> Path:
    digest = _batch_hash(source, target, batch)
    return checkpoint_dir / f"translate_{_model_name(llm)}_{batch[0].i}_{batch[-1].i}_{digest}.json"


def _save_batch_checkpoint(ck: Path, batch: list[Sentence], translated: dict[int, str]) -> None:
    atomic_write_json(ck, {"lines": [{"n": s.i, "text": translated[s.i]} for s in batch]})


def validate_reply(reply: dict | None, expected: list[Sentence]) -> dict[int, str] | None:
    """Validate LLM translation JSON against expected sentence indices, emptiness, and length bounds."""
    if not isinstance(reply, dict):
        return None
    lines = reply.get("lines")
    if not isinstance(lines, list) or len(lines) != len(expected):
        return None

    expected_by_i = {s.i: s for s in expected}
    if len(expected_by_i) != len(expected):
        return None

    out: dict[int, str] = {}
    for item in lines:
        if not isinstance(item, dict):
            return None
        raw_n = item.get("n")
        if isinstance(raw_n, bool) or not isinstance(raw_n, int):
            return None
        if raw_n in out or raw_n not in expected_by_i:
            return None
        raw_text = item.get("text")
        if not isinstance(raw_text, str):
            return None
        cleaned = " ".join(raw_text.split())
        if not cleaned:
            return None
        orig_len = len(expected_by_i[raw_n].text)
        if len(cleaned) > MAX_EXPANSION * orig_len + MAX_EXPANSION_PAD:
            return None
        out[raw_n] = cleaned

    if set(out) != set(expected_by_i):
        return None
    return out


def _is_speech_verbatim(orig_raw: str, translated: str) -> bool:
    """Return True when a non-trivial speech line was echoed verbatim instead of translated."""
    orig = " ".join(orig_raw.split())
    if translated != orig:
        return False
    stripped = _BRACKET_TAG_RE.sub("", orig)
    letters = [ch for ch in stripped if ch.isalpha()]
    if any(ord(ch) > 0x024F for ch in letters):
        return len(letters) >= 2
    return len(letters) >= 12


def _has_untranslated_echo(out: dict[int, str], batch: list[Sentence], source: str, target: str) -> bool:
    if (safe_normalize_language(source) or source) == (safe_normalize_language(target) or target):
        return False
    return any(_is_speech_verbatim(s.text, out.get(s.i, "")) for s in batch)


def _ask_valid_batch(
    batch: list[Sentence],
    source: str,
    target: str,
    source_name: str,
    target_name: str,
    llm: LLMClient,
    attempts: int = 2,
) -> dict[int, str] | None:
    lines_json = json.dumps([{"n": s.i, "text": s.text} for s in batch], ensure_ascii=False)
    prompt = render_prompt(
        "translate",
        source_name=source_name,
        target_name=target_name,
        lines_json=lines_json,
    )
    for attempt in range(attempts):
        raw = llm.complete_json(prompt + (RETRY_SUFFIX if attempt else ""), TRANSLATE_SCHEMA)
        validated = validate_reply(parse_llm_json(raw), batch)
        if validated is not None and not _has_untranslated_echo(validated, batch, source, target):
            return validated
    return None


def translate_sentences(
    sentences: list[Sentence],
    source: str,
    target: str,
    llm: LLMClient,
    checkpoint_dir: Path,
    *,
    on_progress: Callable[[float], None] | None = None,
) -> tuple[dict[int, str], list[int]]:
    """Translate ``sentences`` from ``source`` to ``target`` in batches with validation and checkpoints.

    Returns ``(translations_by_index, failed_indices)``. Propagates ``LLMUnavailable`` directly.
    """
    if not sentences:
        return {}, []

    ck_dir = Path(checkpoint_dir)
    ck_dir.mkdir(parents=True, exist_ok=True)
    source_name = language_name(source, "en")
    target_name = language_name(target, "en")

    total = len(sentences)
    done = 0

    def advance(count: int) -> None:
        nonlocal done
        done += count
        if on_progress is not None:
            on_progress(min(1.0, round(done / total, 4)))

    def run_batch(batch: list[Sentence]) -> tuple[dict[int, str], list[int]]:
        ck = _checkpoint_path(ck_dir, llm, source, target, batch)
        cached = read_json(ck)
        if cached is not None:
            validated_cache = validate_reply(cached, batch)
            if validated_cache is not None and not _has_untranslated_echo(validated_cache, batch, source, target):
                advance(len(batch))
                return validated_cache, []

        try:
            validated = _ask_valid_batch(batch, source, target, source_name, target_name, llm, attempts=2)
        except LLMTimeout:
            if len(batch) <= 1:
                raise
            log.warning(
                "Terjemahan batch %d-%d melewati batas waktu; memecah batch menjadi dua bagian.",
                batch[0].i,
                batch[-1].i,
            )
            validated = None
        if validated is not None:
            _save_batch_checkpoint(ck, batch, validated)
            advance(len(batch))
            return validated, []

        if len(batch) > 1:
            mid = len(batch) // 2
            left_ok, left_fail = run_batch(batch[:mid])
            right_ok, right_fail = run_batch(batch[mid:])
            merged = {**left_ok, **right_ok}
            failed = left_fail + right_fail
            if not failed:
                _save_batch_checkpoint(ck, batch, merged)
            return merged, failed

        log.warning("Kalimat %d gagal diterjemahkan setelah dicoba ulang; teks asli dipertahankan.", batch[0].i)
        advance(1)
        return {}, [batch[0].i]

    all_translations: dict[int, str] = {}
    all_failed: list[int] = []
    for start in range(0, total, BATCH_SIZE):
        ok, fail = run_batch(sentences[start : start + BATCH_SIZE])
        all_translations.update(ok)
        all_failed.extend(fail)

    return all_translations, all_failed
