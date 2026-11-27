"""Get a source video into the project folder: from YouTube (yt-dlp) or a local file."""
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from app.core.config import Settings
from app.core.fs import atomic_write_json, project_dir, read_json
from app.core.models import HeatPoint, VideoMeta

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YT_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}
SAFETY_MARGIN = 2 * 1024**3


class IngestError(Exception):
    def __init__(self, code: str, message: str, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.retryable = retryable


def parse_youtube_id(url: str) -> str:
    u = urlparse(url.strip())
    host = (u.hostname or "").lower()
    vid = None
    if host == "youtu.be":
        vid = u.path.lstrip("/").split("/")[0]
    elif host in _YT_HOSTS:
        if u.path == "/watch":
            vid = parse_qs(u.query).get("v", [None])[0]
        elif u.path.startswith(("/shorts/", "/live/", "/embed/")):
            vid = u.path.split("/")[2]
    if not vid or not _ID_RE.match(vid):
        raise IngestError("INVALID_URL", "Link ini bukan link video YouTube yang valid.")
    return vid


def meta_from_info(info: dict) -> VideoMeta:
    heat = [HeatPoint(start=h["start_time"], end=h["end_time"], value=h["value"])
            for h in (info.get("heatmap") or [])]
    return VideoMeta(source_type="youtube", project_id=info["id"], title=info.get("title") or info["id"],
                     channel=info.get("channel") or info.get("uploader"), duration=float(info.get("duration") or 0),
                     video_id=info["id"], heatmap=heat)


def ensure_disk_space(path: Path, needed_bytes: int) -> None:
    free = shutil.disk_usage(path)[2]
    if free < needed_bytes + SAFETY_MARGIN:
        raise IngestError("DISK_FULL", f"Ruang disk tidak cukup: sisa {free / 1024**3:.1f} GB, "
                                       f"butuh sekitar {(needed_bytes + SAFETY_MARGIN) / 1024**3:.1f} GB.")


def _map_download_error(err: Exception) -> IngestError:
    msg = str(err)
    low = msg.lower()
    if "not a bot" in low:
        return IngestError("DOWNLOAD_BLOCKED", "YouTube memblokir unduhan sementara. Coba lagi nanti, "
                                               "atau unggah file videonya langsung.")
    if any(k in low for k in ("private video", "video unavailable", "confirm your age", "removed", "members-only")):
        return IngestError("VIDEO_UNAVAILABLE", "Video ini privat, dihapus, atau dibatasi umur.")
    return IngestError("NETWORK", f"Gagal mengunduh: {msg[-200:]}", retryable=True)


def ingest_youtube(url: str, settings: Settings, project_id: str | None = None) -> tuple[Path, VideoMeta]:
    """`project_id` names the project folder and `meta.project_id`; default is the YouTube video id."""
    vid = parse_youtube_id(url)
    d = project_dir(settings, project_id or vid)
    src, meta_path = d / "source.mp4", d / "meta.json"
    cached = read_json(meta_path)
    if cached and src.exists():
        return src, VideoMeta.model_validate(cached)

    h = settings.max_video_height
    opts = {
        "format": f"bv*[height<={h}][ext=mp4]+ba[ext=m4a]/b[height<={h}]",
        "merge_output_format": "mp4",
        "outtmpl": str(d / "source.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "noprogress": True,
    }
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={vid}", download=False)
            size = info.get("filesize_approx") or int((info.get("duration") or 3600) * 1.5 * 1024**2)
            ensure_disk_space(d, size)
            ydl.download([f"https://www.youtube.com/watch?v={vid}"])
    except DownloadError as e:
        raise _map_download_error(e) from e
    if not src.exists():
        raise IngestError("NETWORK", "Unduhan selesai tapi file video tidak ditemukan.", retryable=True)
    meta = meta_from_info(info)
    if project_id:
        meta = meta.model_copy(update={"project_id": project_id})
    atomic_write_json(meta_path, meta.model_dump())
    return src, meta


def probe_duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)["format"]["duration"])


def _file_id(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(10 * 1024**2))
    h.update(str(path.stat().st_size).encode())
    return "file-" + h.hexdigest()[:12]


def ingest_file(path: Path, settings: Settings, project_id: str | None = None) -> tuple[Path, VideoMeta]:
    """`project_id` names the project folder and `meta.project_id`; default is a content hash of the file."""
    path = Path(path)
    if not path.is_file():
        raise IngestError("INVALID_URL", f"File tidak ditemukan: {path}")
    pid = project_id or _file_id(path)
    d = project_dir(settings, pid)
    src, meta_path = d / "source.mp4", d / "meta.json"
    cached = read_json(meta_path)
    if cached and src.exists():
        return src, VideoMeta.model_validate(cached)
    ensure_disk_space(d, path.stat().st_size)
    if path.suffix.lower() == ".mp4":
        shutil.copyfile(path, src)
    else:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(path), "-c", "copy", str(src)], check=True)
    meta = VideoMeta(source_type="upload", project_id=pid, title=path.stem, duration=probe_duration(src))
    atomic_write_json(meta_path, meta.model_dump())
    return src, meta
