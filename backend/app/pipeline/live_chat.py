"""Download, parse, and detect activity bursts in YouTube live chat."""
import json
import math
import statistics
from pathlib import Path

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from app.core.config import Settings
from app.core.errors import AppError
from app.pipeline.ingest import _map_download_error, _watch_url, ytdl_base_opts

KEYWORDS_PATH = Path(__file__).resolve().parents[1] / "data" / "chat_keywords.json"


def _r(t: float) -> float:
    return round(t, 3) + 0.0


def load_chat_keywords(path: Path = KEYWORDS_PATH) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [str(w).lower() for words in data.values() for w in words]


def download_live_chat(video_id: str, out: Path, settings: Settings, cookies_browser: str = "") -> Path | None:
    """Download yt-dlp `live_chat` jsonl to `out`, returning None if live chat is unavailable."""
    out = Path(out)
    dest = out if out.suffix else out / f"{video_id}.live_chat.json"
    if dest.exists():
        return dest
    work_dir = dest.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        **ytdl_base_opts(settings, cookies_browser),
        "skip_download": True,
        "writesubtitles": True,
        "subtitleslangs": ["live_chat"],
        "outtmpl": str(work_dir / f"{video_id}.%(ext)s"),
    }
    try:
        with YoutubeDL(opts) as ydl:
            ydl.download([_watch_url(video_id)])
    except DownloadError as e:
        mapped = _map_download_error(e)
        raise AppError(mapped.code, str(mapped), retryable=mapped.retryable) from e
    written = work_dir / f"{video_id}.live_chat.json"
    if not written.exists():
        return None
    if written != dest:
        written.replace(dest)
    return dest


def _extract_runs(runs: list) -> str:
    parts: list[str] = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        if "text" in run:
            parts.append(str(run["text"]))
        elif isinstance(run.get("emoji"), dict):
            em = run["emoji"]
            eid = str(em.get("emojiId") or "")
            if eid and not eid.startswith("http") and not em.get("isCustomEmoji"):
                parts.append(eid)
            elif em.get("shortcuts"):
                parts.append(str(em["shortcuts"][0]))
            elif eid:
                parts.append(eid)
    return "".join(parts).strip()


def _message_from_replay(replay: dict) -> str:
    for action in replay.get("actions") or []:
        if not isinstance(action, dict):
            continue
        item = (action.get("addChatItemAction") or {}).get("item")
        if not isinstance(item, dict):
            continue
        for renderer in item.values():
            if not isinstance(renderer, dict):
                continue
            for key in ("message", "headerSubtext"):
                runs = (renderer.get(key) or {}).get("runs")
                if isinstance(runs, list):
                    text = _extract_runs(runs)
                    if text:
                        return text
    return ""


def parse_live_chat(path: Path) -> list[tuple[float, str]]:
    """Parse yt-dlp `live_chat` jsonl into sorted `(time_sec, text)` pairs, skipping malformed lines."""
    out: list[tuple[float, str]] = []
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(obj, dict):
            continue
        replay = obj.get("replayChatItemAction")
        if not isinstance(replay, dict):
            continue
        try:
            offset_ms = float(replay["videoOffsetTimeMsec"])
        except (KeyError, TypeError, ValueError):
            continue
        if offset_ms < 0:
            continue
        text = _message_from_replay(replay)
        if text:
            out.append((_r(offset_ms / 1000.0), text))
    out.sort(key=lambda item: item[0])
    return out


def chat_peaks(messages: list[tuple[float, str]], duration: float, keywords: list[str],
               window: float = 10.0, baseline: float = 300.0, z_min: float = 2.0) -> list[dict]:
    """Detect chat bursts (`z >= z_min` vs ±baseline/2 moving median/MAD) and merge adjacent windows."""
    if duration <= 0 or window <= 0 or not messages:
        return []
    n_win = max(1, math.ceil(duration / window))
    scores = [0.0] * n_win
    kws = [k.lower() for k in keywords if k]
    for t, text in messages:
        if 0.0 <= t <= duration:
            idx = min(int(t // window), n_win - 1)
            low = text.lower()
            scores[idx] += 2.0 if (kws and any(kw in low for kw in kws)) else 1.0

    half = max(0, int(round((baseline / 2.0) / window)))
    peaks: list[dict] = []
    for i in range(n_win):
        hood = scores[max(0, i - half): min(n_win, i + half + 1)]
        med = statistics.median(hood)
        mad = statistics.median(abs(x - med) for x in hood)
        scale = max(1.0, mad * 1.4826)
        z = (scores[i] - med) / scale
        if z < z_min:
            continue
        val = round(min(1.0, max(0.0, (z - 2.0) / 3.0)), 4)
        start = _r(i * window)
        end = _r(min(duration, (i + 1) * window))
        if peaks and abs(peaks[-1]["end"] - start) <= 0.001:
            peaks[-1]["end"] = end
            peaks[-1]["value"] = max(peaks[-1]["value"], val)
        else:
            peaks.append({"start": start, "end": end, "value": val})
    return peaks
