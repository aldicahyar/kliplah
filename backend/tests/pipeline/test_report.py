from app.core.models import VideoMeta
from app.pipeline.report import build_report
from tests.pipeline.test_render import _clip


def test_report_contains_eval_columns():
    meta = VideoMeta(source_type="youtube", project_id="x", title="Podcast | Bisnis", duration=3600, video_id="x")
    r = build_report(meta, [_clip()], {"transcribe": 1200.0, "analyze": 65.0},
                     {"whisper": "small", "llm": "qwen3:4b"})
    assert "Layak? (ya/tidak)" in r and "qwen3:4b" in r and "small" in r
    assert "20:00" in r and "01:05" in r          # stage timings as mm:ss
    assert "00:02-00:08" in r and "6 s" in r      # clip time range and duration
    assert "Podcast \\| Bisnis" in r              # pipes escaped in markdown tables
    assert "Hook teks" in r and "01_abc12345/final.mp4" in r


def test_report_marks_no_ai_clips():
    c = _clip()
    c.candidate.no_ai = True
    c.candidate.llm_score = None
    meta = VideoMeta(source_type="upload", project_id="f", title="T", duration=60)
    r = build_report(meta, [c], {}, {"whisper": "small", "llm": "qwen3:4b"})
    assert "| tidak |" in r and "tanpa skor ai" in r.lower()
