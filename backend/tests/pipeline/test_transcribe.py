import json
from pathlib import Path

from app.core.models import Word
from app.pipeline.transcribe import chunk_bounds, transcribe


class FakeTranscriber:
    def __init__(self):
        self.calls: list[float] = []

    def transcribe_chunk(self, wav, start, end):
        self.calls.append(start)
        return [Word(w=f"k{int(start)}.", start=start + 1, end=start + 1.5)]


def test_chunk_bounds():
    assert chunk_bounds(1500, 600) == [(0, 600), (600, 1200), (1200, 1500)]
    assert chunk_bounds(30, 600) == [(0, 30)]


def test_transcribe_resumes_from_checkpoints(tmp_path):
    f = FakeTranscriber()
    transcribe(Path("a.wav"), 1500, tmp_path, f, "small", [])
    assert f.calls == [0, 600, 1200]
    (tmp_path / "transcript.json").unlink()
    (tmp_path / "checkpoints" / "transcribe_small_2.json").unlink()
    f2 = FakeTranscriber()
    t = transcribe(Path("a.wav"), 1500, tmp_path, f2, "small", [])
    assert f2.calls == [1200]
    assert [w.w for w in t.words] == ["k0.", "k600.", "k1200."]
    assert t.language == "id" and t.duration == 1500 and len(t.sentences) == 3


def test_checkpoints_are_per_model(tmp_path):
    transcribe(Path("a.wav"), 600, tmp_path, FakeTranscriber(), "small", [])
    (tmp_path / "transcript.json").unlink()
    f = FakeTranscriber()
    transcribe(Path("a.wav"), 600, tmp_path, f, "medium", [])
    assert f.calls == [0]


def test_final_transcript_reused(tmp_path):
    transcribe(Path("a.wav"), 600, tmp_path, FakeTranscriber(), "small", [])
    f = FakeTranscriber()
    transcribe(Path("a.wav"), 600, tmp_path, f, "small", [])
    assert f.calls == []


def test_corrections_applied(tmp_path):
    t = transcribe(Path("a.wav"), 600, tmp_path, FakeTranscriber(), "small", [("k0", "Halo")])
    assert t.words[0].w == "Halo."
    saved = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert saved["words"][0]["w"] == "Halo."


def test_progress_reported(tmp_path):
    seen = []
    transcribe(Path("a.wav"), 1200, tmp_path, FakeTranscriber(), "small", [], on_progress=seen.append)
    assert seen == [0.5, 1.0]
