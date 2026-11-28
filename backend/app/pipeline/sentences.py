from app.core.models import Sentence, Word

ABBREVIATIONS = {"dll.", "dsb.", "dst.", "dr.", "bpk.", "ibu.", "no.", "hlm.", "sdr.", "yth."}
PAUSE_SPLIT_SEC = 2.0


def _ends_sentence(token: str) -> bool:
    t = token.strip().lower()
    return t.endswith((".", "?", "!")) and t not in ABBREVIATIONS


def split_sentences(words: list[Word], pause_sec: float = PAUSE_SPLIT_SEC) -> list[Sentence]:
    """Group words into numbered sentences by end punctuation or a pause of at least `pause_sec`."""
    out: list[Sentence] = []
    begin = 0
    for idx, w in enumerate(words):
        last = idx == len(words) - 1
        pause = not last and words[idx + 1].start - w.end >= pause_sec
        if last or pause or _ends_sentence(w.w):
            chunk = words[begin: idx + 1]
            out.append(Sentence(i=len(out), start=chunk[0].start, end=chunk[-1].end,
                                text=" ".join(x.w for x in chunk), word_range=(begin, idx + 1)))
            begin = idx + 1
    return out
