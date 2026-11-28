import logging
import wave
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from app.core.errors import AppError
from app.providers.transcriber import SAMPLE_RATE, FasterWhisperTranscriber, read_wav_chunk


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


class _FakeModel:
    def __init__(self, responses):
        self._responses = list(responses)
        self.languages: list[str | None] = []

    def transcribe(self, audio, *, language=None, **kwargs):
        self.languages.append(language)
        segments, info = self._responses.pop(0)
        return (s for s in segments), info


def _seg(n_words: int, prob: float | None = 0.9):
    words = [
        SimpleNamespace(word=f" kata{i}", start=i * 0.1, end=i * 0.1 + 0.08,
                        **({} if prob is None else {"probability": prob}))
        for i in range(n_words)
    ]
    return [SimpleNamespace(words=words)]


def test_words_carry_probability(tmp_path):
    p = tmp_path / "a.wav"
    _write_wav(p, 2.0)
    seg = SimpleNamespace(words=[
        SimpleNamespace(word=" halo", start=0.1, end=0.5, probability=0.4321),
        SimpleNamespace(word=" dunia", start=0.6, end=1.0),
    ])
    t = FasterWhisperTranscriber("small")
    t._model = _FakeModel([([seg], SimpleNamespace(language="id", language_probability=0.95))])
    words = t.transcribe_chunk(p, 1.0, 2.0)
    assert [w.w for w in words] == ["halo", "dunia"]
    assert words[0].start == 1.1 and words[0].end == 1.5
    assert words[0].prob == 0.432
    assert words[1].prob is None


def test_detected_malay_is_treated_as_indonesian_target(tmp_path):
    p = tmp_path / "a.wav"
    _write_wav(p, 2.0)
    model = _FakeModel([(_seg(12), SimpleNamespace(language="ms", language_probability=0.9))])
    t = FasterWhisperTranscriber("small", target_language="id")
    t._model = model
    t.transcribe_chunk(p, 0.0, 1.0)
    assert model.languages == [None]
    assert t.language == "id"
    assert t.language_probability == 0.9


def test_low_confidence_detection_falls_back_to_target(tmp_path):
    from app.domain.languages import LOW_CONFIDENCE_PROB

    assert LOW_CONFIDENCE_PROB == 0.6
    p = tmp_path / "a.wav"
    _write_wav(p, 2.0)
    t = FasterWhisperTranscriber("small", target_language="id")
    t._model = _FakeModel([(_seg(12), SimpleNamespace(language="ko", language_probability=0.4))])
    t.transcribe_chunk(p, 0.0, 1.0)
    assert t.language == "id"
    assert t.language_probability == 0.4


def test_detection_waits_for_speech_chunk(tmp_path):
    p = tmp_path / "a.wav"
    _write_wav(p, 4.0)
    model = _FakeModel([
        (_seg(3), SimpleNamespace(language="ko", language_probability=0.9)),
        (_seg(40), SimpleNamespace(language="en", language_probability=0.95)),
        (_seg(15), SimpleNamespace(language="fr", language_probability=0.99)),
    ])
    t = FasterWhisperTranscriber("small")
    t._model = model

    t.transcribe_chunk(p, 0.0, 1.0)
    assert t.language is None
    assert t.language_probability is None

    t.transcribe_chunk(p, 1.0, 2.0)
    assert t.language == "en"
    assert t.language_probability == 0.95

    t.transcribe_chunk(p, 2.0, 3.0)
    assert model.languages == [None, None, "en"]
    assert t.language == "en"


def test_explicit_language_skips_detection(tmp_path):
    p = tmp_path / "a.wav"
    _write_wav(p, 2.0)
    model = _FakeModel([(_seg(15), SimpleNamespace(language="en", language_probability=0.99))])
    t = FasterWhisperTranscriber("small", language="ko")
    t._model = model
    t.transcribe_chunk(p, 0.0, 1.0)
    assert model.languages == ["ko"]
    assert t.language == "ko"
    assert t.language_probability is None


def test_unknown_detected_language_falls_back_to_target(tmp_path, caplog):
    p = tmp_path / "a.wav"
    _write_wav(p, 2.0)
    t = FasterWhisperTranscriber("small", target_language="id")
    t._model = _FakeModel([(_seg(12), SimpleNamespace(language="zz", language_probability=0.9))])
    with caplog.at_level(logging.WARNING, logger="app.providers.transcriber"):
        t.transcribe_chunk(p, 0.0, 1.0)
    assert t.language == "id"
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_explicit_unknown_language_raises_validation():
    with pytest.raises(AppError) as exc:
        FasterWhisperTranscriber("small", language="xx")
    assert exc.value.code == "VALIDATION" and exc.value.status == 422
