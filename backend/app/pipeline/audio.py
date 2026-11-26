import re
import subprocess
from pathlib import Path

_START = re.compile(r"silence_start:\s*(-?[\d.]+)")
_END = re.compile(r"silence_end:\s*(-?[\d.]+)")


def extract_audio(src: Path, dst: Path) -> None:
    """16 kHz mono PCM WAV for Whisper. Skipped when dst already exists (checkpoint)."""
    if dst.exists():
        return
    tmp = dst.with_name(dst.stem + ".partial.wav")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le", str(tmp)], check=True)
    tmp.replace(dst)


def parse_silencedetect(stderr: str) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    start: float | None = None
    for line in stderr.splitlines():
        if m := _START.search(line):
            start = max(0.0, float(m.group(1)))
        elif (m := _END.search(line)) and start is not None:
            out.append((start, float(m.group(1))))
            start = None
    return out


def detect_silences(wav: Path, noise_db: float = -35.0, min_dur: float = 0.3) -> list[tuple[float, float]]:
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(wav), "-af",
                        f"silencedetect=noise={noise_db}dB:d={min_dur}", "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace", check=True)
    return parse_silencedetect(r.stderr)
