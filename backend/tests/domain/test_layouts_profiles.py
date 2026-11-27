import pytest

from app.core.errors import AppError
from app.core.models import ReframeSegment
from app.domain.layouts import LAYOUTS, LayoutSegment, from_reframe_segments, to_reframe_segments
from app.domain.profiles import load_profiles, profile_for


def test_profiles_match_spec_table():
    profiles = load_profiles()
    assert set(profiles) == {"podcast", "stream", "variety", "news", "general"}
    durations = {k: (p.min_sec, p.max_sec) for k, p in profiles.items()}
    assert durations == {"podcast": (30, 90), "stream": (15, 35), "variety": (15, 35),
                         "news": (30, 60), "general": (20, 60)}
    assert profile_for("stream").min_sec == 15 and profile_for("stream").max_sec == 35
    assert profile_for("news").max_sec == 60
    for p in profiles.values():
        assert p.key in profiles
        assert p.label and p.criteria and p.signals
        assert p.default_layouts and set(p.default_layouts) <= set(LAYOUTS)


def test_profile_for_unknown_key_is_validation_error():
    with pytest.raises(AppError) as exc:
        profile_for("cooking")
    assert exc.value.code == "VALIDATION"


def test_reframe_roundtrip():
    segs = [ReframeSegment(start=0, end=2.5, mode="track", x=0.3, zoom=1.2, user_override=True),
            ReframeSegment(start=2.5, end=5, mode="wide", x=0.5),
            ReframeSegment(start=5, end=6, mode="general", x=0.5)]
    layout = from_reframe_segments(segs)
    assert [s.layout for s in layout] == ["face_crop", "fit_blur", "fit_blur"]
    assert layout[0].params == {"x": 0.3, "zoom": 1.2}
    assert layout[0].user_override is True and layout[0].piece_index == 0
    assert (layout[1].start, layout[1].end) == (2.5, 5)

    back = to_reframe_segments(layout)
    assert back[0].mode == "track" and back[0].x == 0.3 and back[0].zoom == 1.2 and back[0].user_override
    assert back[1].mode in ("wide", "general") and back[2].mode in ("wide", "general")
    assert [(s.start, s.end) for s in back] == [(0, 2.5), (2.5, 5), (5, 6)]


def test_to_reframe_rejects_s4_layouts():
    with pytest.raises(AppError) as exc:
        to_reframe_segments([LayoutSegment(start=0, end=3, layout="facecam_full")])
    assert exc.value.code == "VALIDATION"
    assert "facecam_full" in exc.value.message and "S4" in exc.value.message


def _seg(start, end, layout="face_crop"):
    return {"start": start, "end": end, "layout": layout, "params": {}, "piece_index": 0, "user_override": False}


def _tiles(segs, duration):
    return (segs[0]["start"] == 0.0 and segs[-1]["end"] == duration
            and all(a["end"] == b["start"] for a, b in zip(segs, segs[1:])))


def test_normalize_layout_segments_closes_gaps_and_edges():
    from app.domain.layouts import normalize_layout_segments

    segs = [_seg(12.0, 18.0, "fit_blur"), _seg(6.0, 10.0)]
    out = normalize_layout_segments(segs, 24.0)
    assert [(s["start"], s["end"], s["layout"]) for s in out] == [(0.0, 12.0, "face_crop"), (12.0, 24.0, "fit_blur")]
    assert segs[0]["start"] == 12.0  # input not mutated


def test_normalize_layout_segments_clamps_rounds_drops_and_overlaps():
    from app.domain.layouts import normalize_layout_segments

    segs = [_seg(-1.0, 4.00049), _seg(3.0, 8.0, "fit_blur"), _seg(8.0, 8.03), _seg(9.0, 50.0)]
    out = normalize_layout_segments(segs, 20.0)
    assert [(s["start"], s["end"]) for s in out] == [(0.0, 4.0), (4.0, 9.0), (9.0, 20.0)]
    assert _tiles(out, 20.0)


def test_normalize_layout_segments_empty_is_fit_blur():
    from app.domain.layouts import normalize_layout_segments

    assert normalize_layout_segments([], 10.0) == [_seg(0.0, 10.0, "fit_blur")]
    assert normalize_layout_segments([_seg(30.0, 40.0)], 10.0) == [_seg(0.0, 10.0, "fit_blur")]
