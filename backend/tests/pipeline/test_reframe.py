import pytest

from app.core.models import ReframeSegment
from app.pipeline.reframe import Face, collapse_scene, crop_x_px, merge_segments, reframe_clip


def frames(*faces_per_frame):
    return [list(f) for f in faces_per_frame]


def steady(cx, n=9, width=0.2, mouth=lambda i: 0.1):
    return [[Face(cx, width, mouth(i))] for i in range(n)]


def test_no_faces_general():
    seg = collapse_scene(0, 5, [[]] * 9, None)
    assert seg.mode == "general" and seg.x == 0.5 and (seg.start, seg.end) == (0, 5)


def test_mostly_no_faces_general():
    s = steady(0.3, n=4) + [[]] * 5
    assert collapse_scene(0, 5, s, None).mode == "general"


def test_single_speaker_track_median():
    seg = collapse_scene(0, 5, steady(0.31, mouth=lambda i: 0.1 * (i % 2)), None)
    assert seg.mode == "track" and seg.x == pytest.approx(0.31)


def test_near_center_snaps():
    assert collapse_scene(0, 5, steady(0.56), None).x == 0.5


def test_ignores_tiny_faces():
    s = [[Face(0.3, 0.2, 0.1), Face(0.8, 0.015, 0.5 * (i % 2))] for i in range(9)]
    seg = collapse_scene(0, 5, s, None)
    assert seg.mode == "track" and seg.x == pytest.approx(0.3)


def two_people(left_mouth, right_mouth, n=9):
    return [[Face(0.25, 0.2, left_mouth(i)), Face(0.75, 0.2, right_mouth(i))] for i in range(n)]


def test_active_speaker_by_mouth_movement():
    seg = collapse_scene(0, 5, two_people(lambda i: 0.1, lambda i: 0.5 * (i % 2)), None)
    assert seg.mode == "track" and seg.x == pytest.approx(0.75)


def test_hysteresis_keeps_previous_speaker():
    s = two_people(lambda i: 0.45 * (i % 2), lambda i: 0.5 * (i % 2))  # left activity = 0.9 x right
    assert collapse_scene(0, 5, s, prev_x=0.25).x == pytest.approx(0.25)
    assert collapse_scene(0, 5, s, prev_x=None).mode == "wide"


def test_clear_new_speaker_takes_over():
    s = two_people(lambda i: 0.1 * (i % 2), lambda i: 0.5 * (i % 2))
    assert collapse_scene(0, 5, s, prev_x=0.25).x == pytest.approx(0.75)


def test_two_equal_speakers_wide():
    s = two_people(lambda i: 0.5 * (i % 2), lambda i: 0.5 * (i % 2))
    assert collapse_scene(0, 5, s, None).mode == "wide"


def test_moving_subject_wide():
    s = [[Face(0.2 + 0.07 * i, 0.2, 0.1)] for i in range(9)]  # sweeps across the frame
    assert collapse_scene(0, 5, s, None).mode == "wide"


def test_merge_segments():
    a = ReframeSegment(start=0, end=3, mode="track", x=0.30)
    b = ReframeSegment(start=3, end=6, mode="track", x=0.32)
    c = ReframeSegment(start=6, end=9, mode="wide", x=0.5)
    out = merge_segments([a, b, c])
    assert [(s.start, s.end, s.mode, s.x) for s in out] == [(0, 6, "track", 0.30), (6, 9, "wide", 0.5)]


def test_crop_x_px_clamped():
    assert crop_x_px(0.0, 1920, 1080) == 0
    assert crop_x_px(1.0, 1920, 1080) == 1920 - 608
    assert crop_x_px(0.5, 1920, 1080) == 656


@pytest.mark.slow
def test_reframe_synthetic_has_no_faces(synthetic_video):
    segs, layout = reframe_clip(synthetic_video, 2.0, 8.0)
    assert segs and all(s.mode == "general" for s in segs) and layout == "fit_blur"
    assert segs[0].start == 0 and segs[-1].end == pytest.approx(6.0, abs=0.1)


def test_jumping_face_without_stable_track_is_wide():
    # review C1: a face that moves > TRACK_JOIN every sample leaves no track; must not crash
    s = [[Face(0.1 + 0.1 * i, 0.2, 0.1)] for i in range(9)]
    assert collapse_scene(0, 5, s, None).mode == "wide"


def test_reframe_clip_falls_back_when_face_model_unavailable(monkeypatch, tmp_path):
    # review M3: first run offline must not crash after a long transcription
    from app.pipeline import reframe

    def offline(models_dir):
        raise OSError("no internet")

    monkeypatch.setattr(reframe, "ensure_face_model", offline)
    monkeypatch.setattr(reframe, "detect_scenes", lambda v, a, b: [(0.0, b - a)])
    segs, layout = reframe.reframe_clip(tmp_path / "x.mp4", 10.0, 40.0, models_dir=tmp_path)
    assert layout == "fit_blur" and [(s.start, s.end, s.mode) for s in segs] == [(0.0, 30.0, "general")]
