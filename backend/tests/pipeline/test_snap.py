import pytest

from app.core.models import Sentence, Word
from app.pipeline.snap import snap_edges


def wd(t, s, e):
    return Word(w=t, start=s, end=e)


def test_word_snap_and_padding():
    s, e = snap_edges(1.02, 2.97, words=[wd("a", 1.0, 1.4), wd("b.", 2.5, 3.0)], sentences=[], silences=[])
    assert s == pytest.approx(0.95) and e == pytest.approx(3.08)


def test_word_snap_ignored_beyond_budget():
    s, _ = snap_edges(2.0, 3.0, words=[wd("a", 1.0, 1.3), wd("b", 2.7, 3.0)], sentences=[], silences=[])
    assert s == pytest.approx(2.0)  # nearest word start is 0.7 s away


def test_padding_never_overlaps_previous_word():
    s, _ = snap_edges(1.0, 2.0, words=[wd("x", 0.5, 0.98), wd("a", 1.0, 2.0)], sentences=[], silences=[])
    assert s == pytest.approx(0.98)


def _story():
    words = [wd("Jadi", 8.0, 8.4), wd("aku", 8.5, 8.9), wd("mau", 9.0, 9.4), wd("cerita", 10.0, 10.5),
             wd("sesuatu.", 10.6, 11.0), wd("Oke.", 12.0, 12.4)]
    sents = [Sentence(i=0, start=8.0, end=11.0, text="Jadi aku mau cerita sesuatu.", word_range=(0, 5)),
             Sentence(i=1, start=12.0, end=12.4, text="Oke.", word_range=(5, 6))]
    return words, sents


def test_sentence_snap_moves_start_back_within_budget():
    words, sents = _story()
    s, _ = snap_edges(10.0, 11.0, words, sents, [])
    assert s == pytest.approx(7.95)


def test_sentence_snap_rejected_beyond_budget():
    words, sents = _story()
    words[0] = wd("Jadi", 7.0, 7.4)
    sents[0] = sents[0].model_copy(update={"start": 7.0})
    s, _ = snap_edges(10.0, 11.0, words, sents, [])
    assert s == pytest.approx(9.95)


def test_sentence_snap_moves_end_forward_within_budget():
    words, sents = _story()
    _, e = snap_edges(8.0, 10.5, words, sents, [])
    assert e == pytest.approx(11.08)


def test_sentence_snap_end_rejected_beyond_budget():
    words, sents = _story()
    _, e = snap_edges(8.0, 9.4, words, sents, [])  # sentence ends 1.6 s later
    assert e == pytest.approx(9.48)


def test_never_crosses_neighbor_bound():
    words, sents = _story()
    s, _ = snap_edges(10.0, 11.0, words, sents, [], lower=8.5)
    assert s == pytest.approx(9.95)


def test_max_len_respected():
    words, sents = _story()
    s, e = snap_edges(10.0, 11.0, words, sents, [], max_len=2.0)
    assert e - s <= 2.0 and s == pytest.approx(9.95)


def test_silence_refine_moves_into_silence():
    s, _ = snap_edges(1.0, 2.97, words=[wd("a", 1.0, 1.4), wd("b.", 2.5, 3.0)], sentences=[],
                      silences=[(0.5, 0.9)])
    assert s == pytest.approx(0.86)


def test_silence_refine_end():
    _, e = snap_edges(1.0, 2.97, words=[wd("a", 1.0, 1.4), wd("b.", 2.5, 3.0)], sentences=[],
                      silences=[(3.2, 4.0)])
    assert e == pytest.approx(3.26)


def test_never_cuts_into_word():
    # a quiet word sits inside a detected silence; moving the start there would cut it in half
    s, _ = snap_edges(1.0, 2.97, words=[wd("hm", 0.7, 0.88), wd("a", 1.0, 1.4), wd("b.", 2.5, 3.0)],
                      sentences=[], silences=[(0.5, 0.9)])
    assert s == pytest.approx(0.95)


def test_neighbor_bounds_ignore_overlapping_clips():
    # review I1: overlapping neighbours must not cut each other's shared part away
    from app.pipeline.snap import neighbor_bounds

    spans = [(10.0, 50.4), (40.0, 90.4), (120.0, 150.0)]
    assert neighbor_bounds(spans, 0, 300.0) == (0.0, 120.0)
    assert neighbor_bounds(spans, 1, 300.0) == (0.0, 120.0)
    assert neighbor_bounds(spans, 2, 300.0) == (90.4, 300.0)
