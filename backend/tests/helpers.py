"""Test-only builders."""
from app.core.models import Sentence, Transcript, Word


def W(text: str, start: float = 0.0, step: float = 0.3, dur: float = 0.25) -> list[Word]:
    """Words from text, each `step` seconds apart starting at `start`."""
    return [Word(w=t, start=round(start + i * step, 3), end=round(start + i * step + dur, 3))
            for i, t in enumerate(text.split())]


def sents(texts: list[str], each: float = 6.0) -> list[Sentence]:
    """One sentence per text, back to back, `each` seconds long."""
    out, wi = [], 0
    for i, t in enumerate(texts):
        n = len(t.split())
        out.append(Sentence(i=i, start=i * each, end=(i + 1) * each, text=t, word_range=(wi, wi + n)))
        wi += n
    return out


def timed_sents(n: int, each: float = 6.0) -> list[Sentence]:
    return sents([f"kalimat nomor {i}" for i in range(n)], each)


_VOCAB = ("bisnis modal untung rugi karyawan gaji kantor kopi warung pasar saham emas rumah mobil motor "
          "keluarga anak istri teman sekolah kampus dosen buku musik film game makan resep pedas manis "
          "jakarta bandung surabaya bali hujan panas pagi malam kerja libur mimpi takut berani gagal sukses").split()


def make_transcript(n: int = 40, each: float = 6.0, words_per: int = 5) -> Transcript:
    """n distinct sentences, `each` seconds apart, words spread evenly inside each sentence."""
    import random

    words, sentences = [], []
    step = each / words_per
    for i in range(n):
        rng = random.Random(i)
        toks = [rng.choice(_VOCAB) for _ in range(words_per)]
        toks[-1] += "."
        a = len(words)
        for j, t in enumerate(toks):
            st = i * each + j * step
            words.append(Word(w=t, start=round(st, 3), end=round(st + step * 0.8, 3)))
        sentences.append(Sentence(i=i, start=words[a].start, end=words[-1].end, text=" ".join(toks),
                                  word_range=(a, len(words))))
    return Transcript(language="id", duration=n * each, words=words, sentences=sentences)
