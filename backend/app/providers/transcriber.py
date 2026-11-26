from pathlib import Path
from typing import Protocol

from app.core.models import Word

SAMPLE_RATE = 16000


class Transcriber(Protocol):
    def transcribe_chunk(self, wav: Path, start: float, end: float) -> list[Word]:
        """Words for [start, end) of the audio, with absolute times."""
        ...


class FasterWhisperTranscriber:
    """faster-whisper on CPU (int8), Indonesian, word timestamps, Silero VAD."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self._model = None
        self._audio = None
        self._audio_path: Path | None = None

    def _load(self, wav: Path):
        from faster_whisper import WhisperModel, decode_audio

        if self._model is None:
            self._model = WhisperModel(self.model_name, device="cpu", compute_type="int8")
        if self._audio_path != wav:
            self._audio = decode_audio(str(wav), sampling_rate=SAMPLE_RATE)
            self._audio_path = wav

    def transcribe_chunk(self, wav: Path, start: float, end: float) -> list[Word]:
        self._load(wav)
        piece = self._audio[int(start * SAMPLE_RATE): int(end * SAMPLE_RATE)]
        segments, _ = self._model.transcribe(
            piece, language="id", word_timestamps=True, vad_filter=True,
            condition_on_previous_text=False, beam_size=2)
        words: list[Word] = []
        for seg in segments:
            for w in seg.words or []:
                text = w.word.strip()
                if text:
                    words.append(Word(w=text, start=round(start + w.start, 3), end=round(start + w.end, 3)))
        return words
