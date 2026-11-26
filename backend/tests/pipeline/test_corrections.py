import json

import pytest

from app.pipeline.corrections import apply_corrections, load_corrections
from tests.helpers import W


def test_corrections_multiword_case_insensitive():
    ws = W("kemarin gue ketemu De ni sumargo di jakarta")
    out = apply_corrections(ws, [("de ni sumargo", "Deny Sumargo")])
    assert " ".join(w.w for w in out) == "kemarin gue ketemu Deny Sumargo di jakarta"


def test_merged_word_spans_original_times():
    ws = W("ketemu de ni sumargo")
    out = apply_corrections(ws, [("de ni sumargo", "Deny Sumargo")])
    merged = out[1]
    assert merged.start == pytest.approx(ws[1].start) and merged.end == pytest.approx(ws[3].end)


def test_keeps_trailing_punctuation():
    out = apply_corrections(W("namanya de ni."), [("de ni", "Deny")])
    assert out[-1].w == "Deny."


def test_longest_phrase_first():
    out = apply_corrections(W("open ai gpt"), [("open", "Buka"), ("open ai", "OpenAI")])
    assert [w.w for w in out] == ["OpenAI", "gpt"]


def test_load_corrections(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps([{"from": "de ni", "to": "Deny"}]), encoding="utf-8")
    assert load_corrections(p) == [("de ni", "Deny")]
    assert load_corrections(tmp_path / "missing.json") == []
