"""Download, parse, and detect activity bursts in YouTube live chat."""
import gc
import json
import math
import statistics
import traceback
from pathlib import Path

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadCancelled, DownloadError

from app.core.config import Settings
from app.core.errors import AppError
from app.pipeline.ingest import _map_download_error, _watch_url, ytdl_base_opts

KEYWORDS_PATH = Path(__file__).resolve().parents[1] / "data" / "chat_keywords.json"
CHAT_CUTOFF_PAD_SEC = 120.0


def _r(t: float) -> float:
    return round(t, 3) + 0.0


def load_chat_keywords(path: Path = KEYWORDS_PATH) -> list[str]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return [str(w).lower() for words in data.values() for w in words]


def _extract_offset_ms(line: str) -> float | None:
    line = line.strip()
    if not line:
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    replay = obj.get("replayChatItemAction")
    raw = replay.get("videoOffsetTimeMsec") if isinstance(replay, dict) else None
    if raw is None:
        raw = obj.get("videoOffsetTimeMsec")
    try:
        return float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None


def _make_until_hook(cutoff_ms: float, default_paths: tuple[Path, ...]):
    scanned: dict[Path, int] = {}
    state: dict[str, Path | None] = {"tmp_path": None}

    def hook(status: dict) -> None:
        if status.get("status") not in ("downloading", "finished"):
            return
        candidates: list[Path] = []
        for key in ("tmpfilename", "filename"):
            raw = status.get(key)
            if isinstance(raw, str) and raw and raw != "-":
                p = Path(raw)
                if key == "tmpfilename" and p.suffix == ".part":
                    state["tmp_path"] = p
                candidates.append(p)
        for p in default_paths:
            if p not in candidates:
                candidates.append(p)
        for path in candidates:
            if not path.is_file():
                continue
            pos = scanned.get(path, 0)
            with open(path, "rb") as fh:
                fh.seek(pos)
                chunk = fh.read()
            last_nl = chunk.rfind(b"\n")
            if last_nl == -1:
                continue
            chunk = chunk[: last_nl + 1]
            scanned[path] = pos + len(chunk)
            for line in chunk.decode("utf-8", errors="replace").splitlines():
                offset_ms = _extract_offset_ms(line)
                if offset_ms is not None and offset_ms > cutoff_ms:
                    raise DownloadCancelled(f"Live chat melewati batas {cutoff_ms / 1000.0:.0f} dtk")

    return hook, state


def _run_chat_download(opts: dict, url: str) -> bool:
    """Run yt-dlp download; returns False when stopped early via DownloadCancelled."""
    try:
        with YoutubeDL(opts) as ydl:
            ydl.download([url])
        return True
    except DownloadCancelled as exc:
        # Clear traceback frames so yt-dlp's open `.part` file handle is released on Windows.
        traceback.clear_frames(exc.__traceback__)
        return False
    except DownloadError as e:
        mapped = _map_download_error(e)
        raise AppError(mapped.code, str(mapped), retryable=mapped.retryable) from e


def _cleanup_chat_partials(work_dir: Path, video_id: str) -> None:
    (work_dir / f"{video_id}.live_chat.json.ytdl").unlink(missing_ok=True)
    for frag in work_dir.glob(f"{video_id}.live_chat.json.part-Frag*"):
        frag.unlink(missing_ok=True)


def download_live_chat(
    video_id: str,
    out: Path,
    settings: Settings,
    cookies_browser: str = "",
    until_sec: float | None = None,
) -> Path | None:
    """Download yt-dlp `live_chat` jsonl to `out`, returning None if live chat is unavailable.

    When `until_sec` is set, stops downloading once a message past `until_sec + 120` s is written
    so long streams do not fetch chat hours past the last requested range.
    """
    out = Path(out)
    dest = out if out.suffix else out / f"{video_id}.live_chat.json"
    if dest.exists():
        return dest
    work_dir = dest.parent
    work_dir.mkdir(parents=True, exist_ok=True)
    written = work_dir / f"{video_id}.live_chat.json"
    default_part = work_dir / f"{video_id}.live_chat.json.part"
    opts = {
        **ytdl_base_opts(settings, cookies_browser),
        "skip_download": True,
        "writesubtitles": True,
        "subtitleslangs": ["live_chat"],
        "outtmpl": str(work_dir / f"{video_id}.%(ext)s"),
    }
    hook_state: dict[str, Path | None] = {"tmp_path": None}
    if until_sec is not None:
        hook, hook_state = _make_until_hook((until_sec + CHAT_CUTOFF_PAD_SEC) * 1000.0, (default_part, written))
        opts["progress_hooks"] = [hook]

    completed = _run_chat_download(opts, _watch_url(video_id))
    if not completed:
        gc.collect()
        _cleanup_chat_partials(work_dir, video_id)

    if written.exists():
        if written != dest:
            written.replace(dest)
        return dest
    part = hook_state["tmp_path"] or default_part
    if not completed and part.exists():
        if part.stat().st_size == 0:
            part.unlink(missing_ok=True)
            return None
        part.replace(dest)
        return dest
    return None


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
