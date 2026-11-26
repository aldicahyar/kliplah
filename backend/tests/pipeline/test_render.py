import json
import subprocess
import sys
from pathlib import Path

import pytest

from app.core.models import Candidate, ClipPlan, Materials, PlatformCaption, ReframeSegment, SubtitleLine, Word
from app.pipeline.render import RenderError, build_render_command, escape_filter_path, render_clip, run_ffmpeg
from app.pipeline.subtitles import load_presets

FONTS = Path(__file__).parents[2] / "assets" / "fonts"


def test_escape_filter_path_windows():
    p = Path(r"C:\Users\a b\OneDrive\Documents\x's\subs.ass")
    assert escape_filter_path(p) == r"C\\:/Users/a b/OneDrive/Documents/x\\\'s/subs.ass"


def test_escape_filter_path_brackets_and_commas():
    assert escape_filter_path(Path("D:/clips [1], v2;/s.ass")) == r"D\\:/clips \[1\]\, v2\;/s.ass"


def test_command_track_and_wide_segments():
    cmd = build_render_command(Path("s.mp4"), 1920, 1080, 100.0, 160.0,
                               [ReframeSegment(start=0, end=30, mode="track", x=0.3),
                                ReframeSegment(start=30, end=60, mode="wide", x=0.5)],
                               Path("subs.ass"), Path("fonts"), Path("final.partial.mp4"))
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "crop=608:1080:" in fc and "boxblur" in fc and "concat=n=2:v=1:a=1" in fc
    assert "trim=start=0:end=30" in fc and "trim=start=30:end=60" in fc
    assert "loudnorm=I=-14" in fc and "ass=filename=" in fc
    assert cmd[cmd.index("-ss") + 1] == "100.0" and cmd[cmd.index("-to") + 1] == "160.0"
    assert "+faststart" in cmd and cmd[-1].endswith("final.partial.mp4")
    assert cmd[cmd.index("-c:v") + 1] == "libx264" and "-progress" in cmd


def test_run_ffmpeg_stall_kills():
    fake = [sys.executable, "-c",
            "import time,sys; print('out_time_ms=1000', flush=True); time.sleep(30)"]
    with pytest.raises(RenderError) as e:
        run_ffmpeg(fake, stall_timeout=1.0)
    assert e.value.code == "RENDER_STALLED"


def test_run_ffmpeg_failure_reports_stderr():
    fake = [sys.executable, "-c", "import sys; sys.stderr.write('boom happened'); sys.exit(3)"]
    with pytest.raises(RenderError) as e:
        run_ffmpeg(fake, stall_timeout=5.0)
    assert e.value.code == "RENDER_FAILED" and "boom happened" in str(e.value)


def test_run_ffmpeg_reports_progress():
    fake = [sys.executable, "-c", "print('out_time_ms=1000000'); print('out_time_ms=2000000')"]
    seen = []
    run_ffmpeg(fake, stall_timeout=60.0, on_progress=seen.append)  # generous: python may start slowly under load
    assert seen == [1.0, 2.0]


def _clip() -> ClipPlan:
    cap = PlatformCaption(title="Judul klip", caption="Caption singkat.", hashtags=["#podcast", "#bisnis"])
    words = [Word(w="halo", start=0.5, end=0.9), Word(w="semua", start=1.0, end=1.4)]
    return ClipPlan(
        id="abc12345", rank=1,
        candidate=Candidate(start_sentence=0, end_sentence=1, start=2.0, end=8.0, llm_score=70, category="hook",
                            reason="r", hook="Hook {aneh} \\N teks", score=70),
        materials=Materials(hook="Hook teks", tiktok=cap, reels=cap, shorts=cap),
        subtitles=[SubtitleLine(id=1, start=0.5, end=1.4, text="halo semua", words=words)],
        reframe_segments=[ReframeSegment(start=0, end=3, mode="track", x=0.3),
                          ReframeSegment(start=3, end=6, mode="general", x=0.5)],
        layout="face_crop", caption_preset="sorot_kata")


@pytest.mark.slow
def test_render_synthetic_end_to_end(synthetic_video, tmp_path):
    out_dir = tmp_path / "dir with space's"
    final = render_clip(synthetic_video, _clip(), out_dir, load_presets(), FONTS)
    probe = json.loads(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height:format=duration", "-of", "json",
         str(final)], capture_output=True, text=True, check=True).stdout)
    video = [s for s in probe["streams"] if s["codec_type"] == "video"][0]
    assert (video["width"], video["height"]) == (1080, 1920)
    assert any(s["codec_type"] == "audio" for s in probe["streams"])
    assert abs(float(probe["format"]["duration"]) - 6.0) <= 1.0
    assert not list(out_dir.glob("*.partial.mp4"))
    captions = (out_dir / "captions.txt").read_text(encoding="utf-8")
    assert "TIKTOK" in captions and "#podcast #bisnis" in captions


def test_portrait_source_never_crops_wider_than_frame():
    # review M5: a phone video narrower than 9:16 (1080x2400) cannot be cropped to 9:16 of its height
    cmd = build_render_command(Path("s.mp4"), 1080, 2400, 0.0, 10.0,
                               [ReframeSegment(start=0, end=10, mode="track", x=0.3)],
                               Path("subs.ass"), Path("fonts"), Path("o.partial.mp4"))
    fc = cmd[cmd.index("-filter_complex") + 1]
    assert "crop=1350:2400" not in fc and "boxblur" in fc
