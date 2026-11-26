"""Final render: per-scene crop or fit+blur, burned .ass subtitles, loudness normalization."""
import json
import os
import shutil
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

from app.core.models import ClipPlan, ReframeSegment
from app.pipeline.reframe import crop_x_px
from app.pipeline.subtitles import build_ass

OUT_W, OUT_H = 1080, 1920


class RenderError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _escape(s: str, chars: str) -> str:
    return "".join("\\" + ch if ch in chars else ch for ch in s)


def escape_filter_path(p: Path) -> str:
    """Escape a path for a filter option inside -filter_complex (two escaping levels)."""
    option_level = _escape(str(p).replace("\\", "/"), "\\':")
    return _escape(option_level, "\\'[],;")


def _num(x: float) -> str:
    return f"{x:g}"


def _layout(i: int, seg: ReframeSegment, src_w: int, src_h: int) -> str:
    cw = int(round(src_h * 9 / 16 / 2) * 2)
    if seg.mode == "track" and cw <= src_w:  # portrait/square sources cannot be cropped this way
        return f"crop={cw}:{src_h}:{crop_x_px(seg.x, src_w, src_h)}:0,scale={OUT_W}:{OUT_H}"
    return (f"split[bg{i}][fg{i}];"
            f"[bg{i}]scale={OUT_W}:{OUT_H}:force_original_aspect_ratio=increase,crop={OUT_W}:{OUT_H},boxblur=20:2[bgb{i}];"
            f"[fg{i}]scale={OUT_W}:-2[fgs{i}];[bgb{i}][fgs{i}]overlay=(W-w)/2:(H-h)/2")


def build_render_command(source: Path, src_w: int, src_h: int, clip_start: float, clip_end: float,
                         segments: list[ReframeSegment], ass_path: Path, fonts_dir: Path,
                         out_partial: Path) -> list[str]:
    parts, pairs = [], []
    for i, seg in enumerate(segments):
        a, b = _num(seg.start), _num(seg.end)
        parts.append(f"[0:v]trim=start={a}:end={b},setpts=PTS-STARTPTS,{_layout(i, seg, src_w, src_h)},setsar=1[v{i}]")
        parts.append(f"[0:a]atrim=start={a}:end={b},asetpts=PTS-STARTPTS[a{i}]")
        pairs.append(f"[v{i}][a{i}]")
    parts.append(f"{''.join(pairs)}concat=n={len(segments)}:v=1:a=1[vc][ac]")
    parts.append(f"[vc]fps=30,ass=filename={escape_filter_path(ass_path)}:fontsdir={escape_filter_path(fonts_dir)}[vo]")
    parts.append("[ac]loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[ao]")
    return ["ffmpeg", "-hide_banner", "-y", "-ss", str(clip_start), "-to", str(clip_end), "-i", str(source),
            "-filter_complex", ";".join(parts), "-map", "[vo]", "-map", "[ao]",
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-movflags", "+faststart",
            "-progress", "pipe:1", "-nostats", str(out_partial)]


def run_ffmpeg(cmd: list[str], stall_timeout: float = 120.0,
               on_progress: Callable[[float], None] | None = None) -> None:
    if cmd[0] == "ffmpeg" and shutil.which("ffmpeg") is None:
        raise RenderError("FFMPEG_MISSING", "ffmpeg tidak ditemukan. Jalankan scripts/check_setup.py.")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            encoding="utf-8", errors="replace")
    state = {"last": time.monotonic(), "value": None}
    tail: deque[str] = deque(maxlen=20)

    def read_out():
        for line in proc.stdout:
            if line.startswith("out_time_ms="):
                try:
                    v = int(line.split("=", 1)[1])
                except ValueError:
                    continue
                if v != state["value"]:
                    state["value"], state["last"] = v, time.monotonic()
                    if on_progress:
                        on_progress(v / 1_000_000)

    def read_err():
        for line in proc.stderr:
            tail.append(line.rstrip())

    threads = [threading.Thread(target=read_out, daemon=True), threading.Thread(target=read_err, daemon=True)]
    for t in threads:
        t.start()
    while proc.poll() is None:
        if time.monotonic() - state["last"] > stall_timeout:
            proc.kill()
            proc.wait()
            raise RenderError("RENDER_STALLED", f"ffmpeg tidak bergerak selama {stall_timeout:.0f} detik.")
        time.sleep(0.2)
    for t in threads:
        t.join(timeout=2)
    if proc.returncode != 0:
        raise RenderError("RENDER_FAILED", "Render gagal:\n" + "\n".join(tail))


def _probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height:format=duration",
                          "-of", "json", str(path)], capture_output=True, text=True)
    return json.loads(out.stdout or "{}")


def finalize_output(partial: Path, final: Path, expected_duration: float) -> None:
    info = _probe(partial) if partial.exists() else {}
    types = {s.get("codec_type") for s in info.get("streams", [])}
    dur = float(info.get("format", {}).get("duration", -1))
    if not {"video", "audio"} <= types or abs(dur - expected_duration) > 1.0:
        partial.unlink(missing_ok=True)
        raise RenderError("RENDER_FAILED", f"Hasil render tidak valid (stream {sorted(t for t in types if t)}, "
                                           f"durasi {dur:.1f} s, harusnya {expected_duration:.1f} s).")
    os.replace(partial, final)


def _captions_text(clip: ClipPlan) -> str:
    out = []
    for name, cap in (("TIKTOK", clip.materials.tiktok), ("INSTAGRAM REELS", clip.materials.reels),
                      ("YOUTUBE SHORTS", clip.materials.shorts)):
        out += [f"=== {name} ===", f"Judul: {cap.title}", cap.caption, " ".join(cap.hashtags), ""]
    return "\n".join(out)


def render_clip(source: Path, clip: ClipPlan, out_dir: Path, presets: dict, fonts_dir: Path,
                on_progress: Callable[[float], None] | None = None) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    c = clip.candidate
    streams = _probe(source).get("streams", [])
    video = next(s for s in streams if s.get("codec_type") == "video")
    ass = out_dir / "subs.ass"
    ass.write_text(build_ass(clip.subtitles, clip.materials.hook or c.hook, clip.hook_duration_sec,
                             clip.caption_preset, presets), encoding="utf-8")
    partial, final = out_dir / "final.partial.mp4", out_dir / "final.mp4"
    segments = clip.reframe_segments or [ReframeSegment(start=0, end=c.end - c.start, mode="general", x=0.5)]
    cmd = build_render_command(source, int(video["width"]), int(video["height"]), c.start, c.end, segments,
                               ass, fonts_dir, partial)
    try:
        run_ffmpeg(cmd, on_progress=on_progress)
    except RenderError:
        partial.unlink(missing_ok=True)
        raise
    finalize_output(partial, final, segments[-1].end - segments[0].start)
    (out_dir / "captions.txt").write_text(_captions_text(clip), encoding="utf-8")
    return final
