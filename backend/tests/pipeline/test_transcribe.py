import json
from pathlib import Path

from app.core.models import Word
from app.domain.ranges import RangeFile
from app.pipeline.transcribe import chunk_bounds, merge_range_transcripts, transcribe
from tests.helpers import W


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


def test_transcribe_language_param(tmp_path):
    t = transcribe(Path("a.wav"), 600, tmp_path, FakeTranscriber(), "small", [], language="en")
    assert t.language == "en"


def test_transcribe_labels_transcript_with_detected_language(tmp_path):
    class DetectingTranscriber(FakeTranscriber):
        def __init__(self):
            super().__init__()
            self.language: str | None = None

        def transcribe_chunk(self, wav, start, end):
            self.language = "en"
            return super().transcribe_chunk(wav, start, end)

    f = DetectingTranscriber()
    t = transcribe(Path("a.wav"), 600, tmp_path, f, "small", [])
    assert t.language == "en"
    saved = json.loads((tmp_path / "transcript.json").read_text(encoding="utf-8"))
    assert saved["language"] == "en"


def test_transcribe_defaults_language_to_id_when_none(tmp_path):
    t = transcribe(Path("a.wav"), 600, tmp_path, FakeTranscriber(), "small", [], language=None)
    assert t.language == "id"


def test_transcribe_persists_detected_language_across_checkpoint_resume(tmp_path):
    class PartialDetectingTranscriber:
        def __init__(self, target_language="id"):
            self.language: str | None = None
            self.target_language = target_language
            self.calls: list[tuple[float, str | None]] = []

        def transcribe_chunk(self, wav, start, end):
            self.calls.append((start, self.language))
            if start == 0:
                self.language = "ko"
                return [Word(w=f"w{i}", start=start + i, end=start + i + 0.5) for i in range(12)]
            raise RuntimeError("simulated crash on chunk 1")

    f1 = PartialDetectingTranscriber()
    try:
        transcribe(Path("a.wav"), 1200, tmp_path, f1, "small", [])
    except RuntimeError:
        pass
    assert (tmp_path / "checkpoints" / "transcribe_small_0.json").exists()
    assert not (tmp_path / "transcript.json").exists()

    class ResumedShortTranscriber:
        def __init__(self, target_language="id"):
            self.language: str | None = None
            self.target_language = target_language
            self.calls: list[tuple[float, str | None]] = []

        def transcribe_chunk(self, wav, start, end):
            # Chunk 1 has < 10 words so it cannot detect on its own; it must inherit "ko" before running.
            self.calls.append((start, self.language))
            return [Word(w="akhir.", start=start + 1.0, end=start + 1.5)]

    f2 = ResumedShortTranscriber()
    t = transcribe(Path("a.wav"), 1200, tmp_path, f2, "small", [], language=None)
    assert f2.calls == [(600.0, "ko")]
    assert f2.language == "ko"
    assert t.language == "ko"

    # Also verify resuming when every chunk is already cached before transcript.json exists
    (tmp_path / "transcript.json").unlink()
    f3 = ResumedShortTranscriber()
    t3 = transcribe(Path("a.wav"), 1200, tmp_path, f3, "small", [], language=None)
    assert f3.calls == []
    assert t3.language == "ko"


def test_transcribe_falls_back_to_transcriber_target_language_when_undetected(tmp_path):
    class UndetectingTranscriber(FakeTranscriber):
        def __init__(self, target_language="en"):
            super().__init__()
            self.language: str | None = None
            self.target_language = target_language

    t = transcribe(Path("a.wav"), 600, tmp_path, UndetectingTranscriber("en"), "small", [], language=None)
    assert t.language == "en"
    assert not (tmp_path / "checkpoints" / "transcribe_small_lang.json").exists()


def test_transcribe_resume_ignores_readonly_language_on_slotted_fake(tmp_path):
    (tmp_path / "checkpoints").mkdir(parents=True)
    (tmp_path / "checkpoints" / "transcribe_small_lang.json").write_text('{"language": "ko"}', encoding="utf-8")

    class SlottedReadOnlyTranscriber:
        __slots__ = ("calls",)

        @property
        def language(self):
            return None

        def __init__(self):
            self.calls = []

        def transcribe_chunk(self, wav, start, end):
            self.calls.append(start)
            return [Word(w="halo.", start=start + 1, end=start + 1.5)]

    t = transcribe(Path("a.wav"), 600, tmp_path, SlottedReadOnlyTranscriber(), "small", [], language=None)
    assert t.language == "ko"


def _rf(index, start, end):
    return RangeFile(index, start, end, f"source_r{index}.mp4")


def test_merge_range_transcripts_shifts_to_original_timeline():
    words = [Word(w="halo.", start=0.5, end=0.9, prob=0.8)]
    t = merge_range_transcripts([(_rf(0, 1800.0, 1830.0), words)], 2.0, "id", 7200.0)
    assert (t.words[0].start, t.words[0].end, t.words[0].prob) == (1800.5, 1800.9, 0.8)
    assert t.sentences[0].start == 1800.5 and t.sentences[0].word_range == (0, 1)
    assert t.language == "id" and t.duration == 7200.0


def test_transcript_never_spans_two_ranges():
    """Adjacent ranges, no punctuation, no long pause across the seam: sentences still break at the seam."""
    r0, r1 = _rf(0, 100.0, 110.0), _rf(1, 111.0, 121.0)
    w0 = W("jadi gini aku mau cerita soal kerja", start=7.5)  # local 7.5..9.55 -> 107.5..109.55
    w1 = W("terus dia bilang oke aja gitu", start=0.0)  # local 0 -> 111.0, gap 1.45 s < 2.0 s
    t = merge_range_transcripts([(r0, w0), (r1, w1)], 2.0, "id", 300.0)
    seam = len(w0)
    assert len(t.words) == len(w0) + len(w1)
    assert len(t.sentences) == 2
    for s in t.sentences:
        a, b = s.word_range
        assert b <= seam or a >= seam
        assert (t.words[a].start, t.words[b - 1].end) == (s.start, s.end)
    assert [s.i for s in t.sentences] == [0, 1]
    assert t.sentences[-1].word_range == (seam, len(t.words))
    assert t.words[seam].start == 111.0
