from pathlib import Path

import pytest

from app.pipeline.audio import detect_silences, extract_audio, parse_silencedetect

FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_parse_silencedetect():
    text = (FIXTURES / "silencedetect_stderr.txt").read_text(encoding="utf-8")
    assert parse_silencedetect(text) == [(1.504, 2.31), (7.95, 10.02)]


def test_parse_silencedetect_unterminated_silence_at_eof():
    assert parse_silencedetect("[silencedetect @ 0x1] silence_start: 5.0\n") == []


def test_parse_silencedetect_negative_start_clamped():
    s = "[silencedetect @ 1] silence_start: -0.01\n[silencedetect @ 1] silence_end: 0.5 | silence_duration: 0.51\n"
    assert parse_silencedetect(s) == [(0.0, 0.5)]


@pytest.mark.slow
def test_detect_silences_on_synthetic(synthetic_video, tmp_path):
    wav = tmp_path / "a.wav"
    extract_audio(synthetic_video, wav)
    assert wav.stat().st_size > 0
    sil = detect_silences(wav)
    assert any(7.8 <= a <= 8.3 and 9.7 <= b <= 10.3 for a, b in sil)


@pytest.mark.slow
def test_extract_audio_skips_existing(synthetic_video, tmp_path):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"cached")
    extract_audio(synthetic_video, wav)
    assert wav.read_bytes() == b"cached"
