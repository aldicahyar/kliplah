"""Tests for ebur128 audio loudness extraction and peak detection."""
from pathlib import Path

import pytest

from app.pipeline.audio import extract_audio
from app.pipeline.audio_peaks import audio_loudness, audio_peaks, parse_ebur128

FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_audio_loudness_parses_ebur128():
    series = parse_ebur128((FIXTURES / "ebur128_stderr.txt").read_text(encoding="utf-8"))
    assert len(series) == 5
    by_sec = {int(t): m for t, m in series}
    assert by_sec[0] == -24.0
    assert by_sec[3] == -12.0
    assert by_sec[4] == -24.0


def test_audio_peaks_threshold_and_offset():
    # median = -24.0; at t=3.1 -> -12.0 (+12 LU -> value = (12-8)/8 = 0.5)
    # at t=10 -> jump from -26.0 (t=9) to -16.0 (+10 LU in 1 s, +8 LU over median) -> peak
    series = [
        (0.5, -24.0),
        (1.5, -24.0),
        (2.5, -24.0),
        (3.1, -12.0),
        (4.5, -24.0),
        (8.5, -24.0),
        (9.5, -26.0),
        (10.5, -16.0),
        (11.5, -24.0),
    ]
    peaks = audio_peaks(series, offset=1800.0)
    assert len(peaks) == 2
    assert peaks[0]["start"] == pytest.approx(1803.1)
    assert peaks[0]["value"] == pytest.approx(0.5)
    assert peaks[1]["start"] == pytest.approx(1810.5)
    assert peaks[1]["value"] == pytest.approx(0.25)


@pytest.mark.slow
def test_audio_loudness_real_ffmpeg(synthetic_video, tmp_path):
    wav = tmp_path / "audio.wav"
    extract_audio(synthetic_video, wav)
    series = audio_loudness(wav)
    assert len(series) >= 15
    quiet = [m for t, m in series if 8.5 <= t <= 10.0]
    loud = [m for t, m in series if 2.0 <= t <= 7.0]
    assert quiet and loud
    assert max(quiet) < min(loud) - 10.0


def test_audio_peaks_30000_samples_under_1s():
    import time

    series = [(i + 0.5, -10.0 if i == 15000 else -24.0) for i in range(30000)]
    t0 = time.perf_counter()
    peaks = audio_peaks(series)
    elapsed = time.perf_counter() - t0
    assert elapsed < 1.0
    assert len(peaks) == 1 and peaks[0]["start"] == pytest.approx(15000.5)
