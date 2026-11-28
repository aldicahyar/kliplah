"""Extract ~1 Hz momentary loudness via ffmpeg `ebur128` and detect audio peaks."""
import re
import statistics
import subprocess
from pathlib import Path

_EBU_RE = re.compile(r"\bt:\s*(-?[\d.]+)\b.*?\bM:\s*(-?[\d.]+)")
_SILENCE_FLOOR = -65.0


def _r(t: float) -> float:
    return round(t, 3) + 0.0


def parse_ebur128(stderr: str) -> list[tuple[float, float]]:
    """Parse ffmpeg `ebur128` verbose lines into one `(t, momentary_lufs)` sample per ~1 s bin."""
    buckets: dict[int, tuple[float, float]] = {}
    for line in stderr.splitlines():
        m = _EBU_RE.search(line)
        if not m:
            continue
        t = float(m.group(1))
        lufs = float(m.group(2))
        if t < 0:
            continue
        sec = int(t)
        prev = buckets.get(sec)
        if prev is None or lufs > prev[1]:
            buckets[sec] = (_r(t), lufs)
    return [buckets[k] for k in sorted(buckets)]


def audio_loudness(wav: Path) -> list[tuple[float, float]]:
    """Run ffmpeg `ebur128=framelog=verbose` on `wav` and return `(t, lufs)` per ~1 s."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "verbose", "-i", str(wav),
         "-af", "ebur128=framelog=verbose", "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
    )
    return parse_ebur128(r.stderr)


def audio_peaks(series: list[tuple[float, float]], offset: float = 0.0) -> list[dict]:
    """Detect loud peaks (`>= median + 8 LU` or `+10 LU` jump within 2 s), shifted by `offset`."""
    if not series:
        return []
    active = [m for _, m in series if m > _SILENCE_FLOOR]
    med = statistics.median(active if active else [m for _, m in series])
    peaks: list[dict] = []
    for idx, (t, m) in enumerate(series):
        diff_med = m - med
        prev_vals = [pm for pt, pm in series[max(0, idx - 2):idx]
                     if 0.0 < t - pt <= 2.0 + 1e-6 and pm > _SILENCE_FLOOR]
        jump = (m - min(prev_vals)) if (prev_vals and m > _SILENCE_FLOOR) else 0.0
        if diff_med < 8.0 and jump < 10.0:
            continue
        selisih = max(diff_med, jump)
        val = round(min(1.0, max(0.0, (selisih - 8.0) / 8.0)), 4)
        start = _r(t + offset)
        end_local = series[idx + 1][0] if idx + 1 < len(series) else t + 1.0
        end = _r(end_local + offset)
        if peaks and start <= peaks[-1]["end"] + 0.05:
            peaks[-1]["end"] = max(peaks[-1]["end"], end)
            peaks[-1]["value"] = max(peaks[-1]["value"], val)
        else:
            peaks.append({"start": start, "end": end, "value": val})
    return peaks
