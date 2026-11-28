from pathlib import Path

import pytest

from app.core.errors import AppError
from app.core.models import VideoMeta
from app.domain.ranges import RangeFile, SourceMap, normalize_ranges, parse_range


def _meta(duration: float, ranges: list[dict] | None = None) -> VideoMeta:
    return VideoMeta(source_type="youtube", project_id="p1", title="t", duration=duration,
                     ranges=ranges or [])


def test_parse_range_formats():
    assert parse_range("00:30:00-01:00:00") == (1800.0, 3600.0)
    assert parse_range("5:00-7:30") == (300.0, 450.0)
    assert parse_range("1800-3600") == (1800.0, 3600.0)
    assert parse_range(" 1:00:00 - 1:30:00 ") == (3600.0, 5400.0)
    assert parse_range("10.5-20") == (10.5, 20.0)
    for bad in ("abc", "60-30", "30-30", "1:60-2:00", "1-2-3", "", "-5-10"):
        with pytest.raises(AppError) as e:
            parse_range(bad)
        assert e.value.code == "VALIDATION"


def test_normalize_ranges_clamps_merges_and_rejects_empty():
    assert normalize_ranges(None, 100.0) is None
    assert normalize_ranges([], 100.0) is None
    with pytest.raises(AppError) as e:
        normalize_ranges([[18000, 21600]], 14400)
    assert e.value.code == "VALIDATION"
    assert normalize_ranges([[600, 1200], [900, 1500]], None) == [(600.0, 1500.0)]
    assert normalize_ranges([[100.5, 200], [0, 100]], None) == [(0.0, 200.0)]
    # gap of exactly 1.0 s merges, larger gap does not
    assert normalize_ranges([[0, 100], [101, 200]], None) == [(0.0, 200.0)]
    assert normalize_ranges([[0, 100], [101.5, 200]], None) == [(0.0, 100.0), (101.5, 200.0)]
    # clamped remainder of exactly 10 s is kept, 9.999 s rejected
    assert normalize_ranges([[14390, 15000]], 14400) == [(14390.0, 14400.0)]
    with pytest.raises(AppError) as e:
        normalize_ranges([[14390.001, 15000]], 14400)
    assert e.value.code == "VALIDATION"
    # short ranges are dropped silently when others survive
    assert normalize_ranges([[0, 5], [100, 200]], None) == [(100.0, 200.0)]
    # rounding to 0.001
    assert normalize_ranges([[0.00049, 20.1234]], None) == [(0.0, 20.123)]


def test_more_than_ten_ranges_rejected():
    eleven = [[i * 100, i * 100 + 50] for i in range(11)]
    with pytest.raises(AppError) as e:
        normalize_ranges(eleven, None)
    assert e.value.code == "VALIDATION"
    assert len(normalize_ranges(eleven[:10], None)) == 10


def test_legacy_project_maps_to_single_range(tmp_path: Path):
    sm = SourceMap.from_meta(_meta(120.0), tmp_path)
    assert sm.files == [RangeFile(0, 0.0, 120.0, "source.mp4")]
    rf = sm.files[0]
    assert sm.path(rf) == tmp_path / "source.mp4"
    assert sm.to_local(rf, 12.5) == 12.5
    assert sm.to_original(rf, 12.5) == 12.5
    assert sm.locate(120.0) == rf


def test_locate_span_requires_single_range(tmp_path: Path):
    meta = _meta(400.0, [
        {"index": 0, "start": 0.0, "end": 100.0, "file": "ranges/r0.mp4"},
        {"index": 1, "start": 200.0, "end": 300.0, "file": "ranges/r1.mp4"},
    ])
    sm = SourceMap.from_meta(meta, tmp_path)
    assert [f.index for f in sm.files] == [0, 1]
    assert sm.locate_span(250, 260).index == 1
    assert sm.locate_span(90, 210) is None
    assert sm.locate_span(199.96, 300.04).index == 1  # 0.05 s tolerance
    assert sm.locate_span(199.9, 260) is None
    assert sm.locate(50).index == 0
    assert sm.locate(100.0) is None  # [start, end) except the last range
    assert sm.locate(150) is None
    assert sm.locate(300.0).index == 1
    rf = sm.files[1]
    assert sm.path(rf) == tmp_path / "ranges/r1.mp4"
    assert sm.to_local(rf, 250.0) == 50.0
    assert sm.to_original(rf, 50.0) == 250.0


@pytest.mark.parametrize("ranges", [
    [["a", 20]],
    [[0, None]],
    [5],
    [[float("nan"), 100]],
    [[0, float("inf")]],
    [[True, 20]],
    [[0, 20, 30]],
])
def test_normalize_ranges_rejects_malformed_elements(ranges):
    with pytest.raises(AppError) as e:
        normalize_ranges(ranges, None)
    assert e.value.code == "VALIDATION"
    assert "ke-1" in e.value.message


def test_rounding_happens_before_comparing():
    with pytest.raises(AppError) as e:
        parse_range("10.0001-10.0004")
    assert e.value.code == "VALIDATION"
    # 100.0004 rounds to 100.0, so the gap to 101.0004 (-> 101.0) is exactly 1.0 s and merges
    assert normalize_ranges([[0, 100.0004], [101.0004, 200]], None) == [(0.0, 200.0)]
    assert normalize_ranges([[0, 100], [101.001, 200]], None) == [(0.0, 100.0), (101.001, 200.0)]


def test_locate_span_rejects_inverted_span(tmp_path: Path):
    sm = SourceMap.from_meta(_meta(120.0), tmp_path)
    assert sm.locate_span(60, 50) is None
    assert sm.locate_span(50, 50).index == 0


def test_legacy_sourcemap_extends_to_last_sentence_end_and_handles_zero_duration(tmp_path: Path):
    from tests.helpers import make_transcript

    tr = make_transcript(20, each=6.0)
    tr.sentences[-1] = tr.sentences[-1].model_copy(update={"end": 120.8})
    sm = SourceMap.from_meta(_meta(120.0), tmp_path, tr)
    assert sm.files == [RangeFile(0, 0.0, 120.8, "source.mp4")]
    assert sm.locate_span(100.0, 120.8) == sm.files[0]

    # When meta.duration is 0, use transcript end
    sm_zero = SourceMap.from_meta(_meta(0.0), tmp_path, tr)
    assert sm_zero.files == [RangeFile(0, 0.0, 120.8, "source.mp4")]
