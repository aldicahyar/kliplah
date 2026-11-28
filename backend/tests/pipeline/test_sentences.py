import pytest

from app.pipeline.sentences import split_sentences
from tests.helpers import W


def test_split_on_terminal_punctuation():
    ws = W("Halo semua. Apa kabar? Baik!")
    assert [s.text for s in split_sentences(ws)] == ["Halo semua.", "Apa kabar?", "Baik!"]


def test_abbreviations_do_not_split():
    ws = W("Ada buku, pensil, dll. yang dibawa dr. Andi ke sana.")
    assert len(split_sentences(ws)) == 1


def test_split_without_punctuation_uses_pauses():
    a = W("jadi gini aku mau cerita")
    b = W("terus dia bilang oke", start=a[-1].end + 2.1)
    assert [s.text for s in split_sentences(a + b)] == ["jadi gini aku mau cerita", "terus dia bilang oke"]


def test_short_pause_does_not_split():
    a = W("jadi gini aku")
    b = W("mau cerita", start=a[-1].end + 1.0)
    assert len(split_sentences(a + b)) == 1


def test_word_range_and_times_consistent():
    s = split_sentences(W("Satu dua. Tiga empat."))
    assert s[1].word_range == (2, 4)
    assert s[1].start == pytest.approx(0.6)
    assert s[1].end == pytest.approx(0.9 + 0.25)
    assert [x.i for x in s] == [0, 1]


def test_empty_words():
    assert split_sentences([]) == []


def test_split_sentences_custom_pause():
    a = W("jadi gini aku")
    b = W("mau cerita", start=a[-1].end + 1.2)
    assert len(split_sentences(a + b)) == 1
    assert [s.text for s in split_sentences(a + b, pause_sec=1.0)] == ["jadi gini aku", "mau cerita"]
