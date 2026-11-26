import wave
from pathlib import Path
from typing import Protocol

import numpy as np

from app.core.models import Word

SAMPLE_RATE = 16000


class Transcriber(Protocol):
    def transcribe_chunk(self, wav: Path, start: float, end: float) -> list[Word]:
        """Words for [start, end) of the audio, with absolute times."""
        ...


def read_wav_chunk(wav: Path, start: float, end: float) -> np.ndarray:
    """Read [start, end) seconds of the 16 kHz mono PCM WAV made by extract_audio, as float32.

    Reads only that slice, so long podcasts never sit in memory whole, and it does not depend on
    PyAV (faster-whisper's decode_audio broke with newer PyAV releases).
    """
    with wave.open(str(wav), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2 or w.getframerate() != SAMPLE_RATE:
            raise ValueError(f"{wav} harus WAV 16 kHz mono 16-bit (dibuat oleh extract_audio).")
        first = min(int(start * SAMPLE_RATE), w.getnframes())
        w.setpos(first)
        raw = w.readframes(max(0, int(end * SAMPLE_RATE) - first))
    return np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0


class FasterWhisperTranscriber:
    """faster-whisper on CPU (int8), Indonesian, word timestamps, Silero VAD."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from faster_whisper import WhisperModel

            self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        return self._model

    def transcribe_chunk(self, wav: Path, start: float, end: float) -> list[Word]:
        piece = read_wav_chunk(wav, start, end)
        if len(piece) == 0:
            return []
        segments, _ = self._load_model().transcribe(
            piece, language="id", word_timestamps=True, vad_filter=True,
            condition_on_previous_text=False, beam_size=2)
        words: list[Word] = []
        for seg in segments:
            for w in seg.words or []:
                text = w.word.strip()
                if text:
                    words.append(Word(w=text, start=round(start + w.start, 3), end=round(start + w.end, 3)))
        return words
