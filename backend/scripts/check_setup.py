"""Report whether the tools Kliplah needs are installed. Exit 1 if anything is missing."""
import shutil
import subprocess
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import get_settings  # noqa: E402


def _ollama_tags(url: str) -> list[str] | None:
    try:
        r = httpx.get(f"{url}/api/tags", timeout=3)
        r.raise_for_status()
        return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        return None


def _ffmpeg_has_libass(ffmpeg: str | None) -> bool:
    if not ffmpeg:
        return False
    out = subprocess.run([ffmpeg, "-hide_banner", "-version"], capture_output=True, text=True).stdout
    return "--enable-libass" in out


def collect() -> dict[str, bool | float]:
    s = get_settings()
    ffmpeg = shutil.which("ffmpeg")
    tags = _ollama_tags(s.ollama_url)
    data = Path(s.data_dir).resolve()
    probe = data if data.exists() else Path(data.anchor)
    return {
        "ffmpeg": ffmpeg is not None,
        "ffmpeg_has_libass": _ffmpeg_has_libass(ffmpeg),
        "ffprobe": shutil.which("ffprobe") is not None,
        "ollama": tags is not None,
        "ollama_model": tags is not None and any(t in (s.ollama_model, f"{s.ollama_model}:latest") for t in tags),
        "disk_free_gb": round(shutil.disk_usage(probe).free / 1024**3, 1),
    }


def main() -> int:
    r = collect()
    for k, v in r.items():
        mark = "OK" if v is not False else "MISSING"
        print(f"{mark:8} {k}: {v}")
    return 1 if any(v is False for v in r.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
