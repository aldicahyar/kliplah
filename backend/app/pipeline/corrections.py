import json
import re
from pathlib import Path

from app.core.models import Word

_NON_WORD = re.compile(r"[^\w]", re.UNICODE)
_TRAILING_PUNCT = re.compile(r"[.,!?;:]+$")


def load_corrections(path: Path) -> list[tuple[str, str]]:
    try:
        rows = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return []
    return [(r["from"], r["to"]) for r in rows]


def _norm(s: str) -> str:
    return _NON_WORD.sub("", s.lower())


def apply_corrections(words: list[Word], rules: list[tuple[str, str]]) -> list[Word]:
    """Replace misheard phrases (case/punctuation-insensitive), longest phrase first."""
    compiled = [([_norm(t) for t in src.split()], dst) for src, dst in rules]
    compiled = [r for r in compiled if r[0]]
    compiled.sort(key=lambda r: len(r[0]), reverse=True)
    norm = [_norm(w.w) for w in words]
    out: list[Word] = []
    i = 0
    while i < len(words):
        for toks, dst in compiled:
            n = len(toks)
            if norm[i:i + n] == toks:
                punct = _TRAILING_PUNCT.search(words[i + n - 1].w)
                out.append(Word(w=dst + (punct.group(0) if punct else ""),
                                start=words[i].start, end=words[i + n - 1].end))
                i += n
                break
        else:
            out.append(words[i])
            i += 1
    return out
