import wave
from pathlib import Path

import numpy as np
import pytest

from app.providers.transcriber import SAMPLE_RATE, read_wav_chunk


def _write_wav(path: Path, seconds: float) -> np.ndarray:
    # a ramp so each sample position is identifiable
    n = int(seconds * SAMPLE_RATE)
    samples = (np.arange(n) % 30000).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(samples.tobytes())
    return samples


def test_read_wav_chunk_returns_float32_slice(tmp_path):
    p = tmp_path / "a.wav"
    samples = _write_wav(p, 3.0)
    chunk = read_wav_chunk(p, 1.0, 2.0)
    assert chunk.dtype == np.float32 and len(chunk) == SAMPLE_RATE
    expected = samples[SAMPLE_RATE: 2 * SAMPLE_RATE].astype(np.float32) / 32768.0
    assert np.allclose(chunk, expected)


def test_read_wav_chunk_clamps_past_end(tmp_path):
    p = tmp_path / "a.wav"
    _write_wav(p, 1.5)
    assert len(read_wav_chunk(p, 1.0, 10.0)) == SAMPLE_RATE // 2


def test_read_wav_chunk_rejects_wrong_format(tmp_path):
    p = tmp_path / "stereo.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(44100)
        w.writeframes(b"\x00" * 400)
    with pytest.raises(ValueError):
        read_wav_chunk(p, 0.0, 1.0)
