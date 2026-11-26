from app.core.models import Window
from app.pipeline.windows import make_windows
from tests.helpers import timed_sents


def test_windows_short_transcript_single_window():
    s = timed_sents(n=30, each=6.0)  # 180 s
    assert make_windows(s, []) == [Window(index=0, start_sentence=0, end_sentence=29)]


def test_windows_overlap_and_cover_everything():
    s = timed_sents(n=200, each=6.0)  # 1200 s
    w = make_windows(s, [])
    assert w[0].start_sentence == 0 and w[-1].end_sentence == 199
    assert [x.index for x in w] == list(range(len(w)))
    for a, b in zip(w, w[1:]):
        overlap = s[a.end_sentence].end - s[b.start_sentence].start
        assert 30 <= overlap <= 60


def test_window_end_snaps_to_topic_boundary():
    s = timed_sents(n=200, each=6.0)
    w = make_windows(s, [55])  # sentence 55 starts at 330 s, within 60 s of 300 s
    assert w[0].end_sentence == 54


def test_boundary_too_far_is_ignored():
    s = timed_sents(n=200, each=6.0)
    w = make_windows(s, [70])  # 420 s: more than 60 s away
    assert w[0].end_sentence == 49


def test_empty():
    assert make_windows([], []) == []
