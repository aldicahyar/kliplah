from app.pipeline.texttiling import topic_boundaries
from tests.helpers import sents


def test_topic_boundary_between_two_topics():
    s = sents(["kita bahas saham dividen investasi saham"] * 12 + ["resep masak nasi goreng bumbu"] * 12)
    b = topic_boundaries(s)
    assert any(10 <= i <= 14 for i in b)


def test_stopwords_ignored():
    # identical content words, different filler words: no boundary
    s = sents(["yang dan di saham dividen"] * 12 + ["gitu sih kan saham dividen"] * 12)
    assert topic_boundaries(s) == []


def test_topic_boundaries_too_short_returns_empty():
    assert topic_boundaries(sents(["a b c"] * 5)) == []
