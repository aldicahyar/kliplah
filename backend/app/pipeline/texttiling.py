"""Word-overlap TextTiling: find where the conversation changes topic, without an LLM."""
import math
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

from app.core.models import Sentence

_TOKEN = re.compile(r"[a-zA-ZÀ-ɏ]+")
MIN_DEPTH = 0.1


@lru_cache
def stopwords() -> frozenset[str]:
    path = Path(__file__).resolve().parents[1] / "data" / "stopwords_id.txt"
    return frozenset(path.read_text(encoding="utf-8").split())


def _bag(sentences: list[Sentence]) -> Counter:
    sw = stopwords()
    c: Counter = Counter()
    for s in sentences:
        c.update(t for t in _TOKEN.findall(s.text.lower()) if t not in sw and len(t) > 1)
    return c


def _cosine(a: Counter, b: Counter) -> float:
    dot = sum(a[k] * b[k] for k in a.keys() & b.keys())
    na = math.sqrt(sum(v * v for v in a.values()))
    nb = math.sqrt(sum(v * v for v in b.values()))
    return dot / (na * nb) if na and nb else 0.0


def _smooth(xs: list[float], width: int) -> list[float]:
    h = width // 2
    return [sum(xs[max(0, i - h): i + h + 1]) / len(xs[max(0, i - h): i + h + 1]) for i in range(len(xs))]


def topic_boundaries(sentences: list[Sentence], k: int = 7, smooth: int = 3) -> list[int]:
    """Indices of sentences that start a new topic."""
    n = len(sentences)
    if n < 2 * k:
        return []
    gaps = list(range(1, n))  # gap g sits before sentence g
    scores = _smooth([_cosine(_bag(sentences[max(0, g - k): g]), _bag(sentences[g: g + k])) for g in gaps], smooth)

    depths = []
    for i, v in enumerate(scores):
        left = v
        for j in range(i - 1, -1, -1):
            if scores[j] < left:
                break
            left = scores[j]
        right = v
        for j in range(i + 1, len(scores)):
            if scores[j] < right:
                break
            right = scores[j]
        depths.append((left - v) + (right - v))

    mean = sum(depths) / len(depths)
    std = math.sqrt(sum((d - mean) ** 2 for d in depths) / len(depths))
    cut = max(mean + std, MIN_DEPTH)  # floor ignores float noise on uniform text
    out = []
    for i, d in enumerate(depths):
        is_peak = d >= (depths[i - 1] if i else 0) and d >= (depths[i + 1] if i + 1 < len(depths) else 0)
        if d > cut and is_peak and (not out or gaps[i] - out[-1] > 1):
            out.append(gaps[i])
    return out
