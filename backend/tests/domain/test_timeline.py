import copy

import pytest

from app.core.errors import AppError
from app.domain.timeline import (
    Piece,
    output_duration,
    output_to_source,
    remap_output_items,
    source_to_output,
)


def P(start: float, end: float, project_id: str = "p1") -> Piece:
    return Piece(project_id=project_id, start=start, end=end)


def test_output_duration_sums_pieces():
    assert output_duration([P(10, 20), P(50, 55)]) == 15.0


def test_output_to_source_across_pieces():
    pieces = [P(10, 20), P(50, 55)]
    assert output_to_source(pieces, 12.0) == (1, 52.0)
    assert output_to_source(pieces, 15.0) is None
    assert output_to_source(pieces, 0.0) == (0, 10.0)
    assert output_to_source(pieces, 10.0) == (1, 50.0)
    assert output_to_source(pieces, -0.5) is None


def test_source_to_output_inside_and_outside_piece():
    pieces = [P(10, 20), P(50, 55)]
    assert source_to_output(pieces, 1, 52.0) == 12.0
    assert source_to_output(pieces, 1, 55.0) == 15.0
    assert source_to_output(pieces, 1, 49.0) is None
    assert source_to_output(pieces, 0, 52.0) is None


def test_piece_rounds_and_rejects_end_not_after_start():
    p = P(1.23456, 2.00049)
    assert (p.start, p.end) == (1.235, 2.0)
    with pytest.raises(AppError) as exc:
        P(5, 5)
    assert exc.value.code == "VALIDATION"
    with pytest.raises(AppError):
        P(6, 5)


def test_remap_trim_start_shifts_items():
    out = remap_output_items([{"start": 3, "end": 5}], [P(10, 20)], [P(12, 20)])
    assert out == [{"start": 1.0, "end": 3.0}]


def test_remap_keeps_edited_lines_inside_and_trims_partial():
    items = [
        {"id": "A", "start": 0, "end": 2, "text": "a", "edited": True},
        {"id": "B", "start": 5, "end": 8, "text": "b", "edited": True},
        {"id": "C", "start": 7, "end": 9, "text": "c"},
    ]
    before = copy.deepcopy(items)
    out = remap_output_items(items, [P(10, 20)], [P(10, 16)])
    assert out == [
        {"id": "A", "start": 0.0, "end": 2.0, "text": "a", "edited": True},
        {"id": "B", "start": 5.0, "end": 6.0, "text": "b", "edited": True},
    ]
    assert items == before  # inputs never mutated


def test_remap_smart_cut_gap_drops_and_shifts():
    items = [
        {"id": "gap", "start": 4.2, "end": 5.8},
        {"id": "late", "start": 7, "end": 8},
        {
            "id": "cut",
            "start": 3,
            "end": 5.8,
            "words": [
                {"w": "a", "start": 3, "end": 3.5},
                {"w": "w", "start": 5, "end": 5.5},
            ],
        },
    ]
    out = remap_output_items(items, [P(0, 10)], [P(0, 4), P(6, 10)])
    assert out == [
        {"id": "cut", "start": 3.0, "end": 4.0, "words": [{"w": "a", "start": 3.0, "end": 3.5}]},
        {"id": "late", "start": 5.0, "end": 6.0},
    ]


def test_remap_item_spanning_piece_boundary_merges_contiguous_parts():
    old = [P(10, 20), P(50, 55)]
    new = [P(15, 20), P(50, 55)]
    out = remap_output_items([{"id": "x", "start": 8, "end": 12}], old, new)
    assert out == [{"id": "x", "start": 3.0, "end": 7.0}]


def test_remap_non_contiguous_parts_keep_longest():
    old = [P(0, 10)]
    new = [P(5, 10), P(0, 4)]
    out = remap_output_items([{"start": 3, "end": 6}], old, new)
    # parts: source 3-4 -> out 8-9 (1.0 s), source 5-6 -> out 0-1 (1.0 s); a tie keeps the earliest in output
    assert out == [{"start": 0.0, "end": 1.0}]
    out = remap_output_items([{"start": 3, "end": 7}], old, new)
    assert out == [{"start": 0.0, "end": 2.0}]


def test_remap_uses_matching_project_and_first_containing_piece():
    old = [P(0, 10, "a")]
    new = [P(0, 10, "b"), P(2, 8, "a"), P(0, 10, "a")]
    out = remap_output_items([{"start": 3, "end": 4}], old, new)
    assert out == [{"start": 11.0, "end": 12.0}]


def test_remap_drops_items_shorter_than_min_len_and_sorts():
    old = [P(0, 10)]
    new = [P(0, 5.03)]
    items = [{"start": 5, "end": 6}, {"start": 2, "end": 3}, {"start": 1, "end": 1.5}]
    out = remap_output_items(items, old, new)
    assert [(i["start"], i["end"]) for i in out] == [(1.0, 1.5), (2.0, 3.0)]


def test_remap_rounds_to_millisecond():
    old = [P(0.1, 10)]
    new = [P(0.3, 9.7)]
    items = [{"start": 0.7, "end": 2.9, "words": [{"w": "a", "start": 0.7, "end": 1.1}]}]
    out = remap_output_items(items, old, new)
    values = [out[0]["start"], out[0]["end"], out[0]["words"][0]["start"], out[0]["words"][0]["end"]]
    assert values == [0.5, 2.7, 0.5, 0.9]
    for v in values:
        assert round(v, 3) == v
