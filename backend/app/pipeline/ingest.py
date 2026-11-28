"""Get a source video into the project folder: from YouTube (yt-dlp) or a local file."""
import hashlib
import json
import logging
import re
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError, download_range_func

from app.core.config import DEFAULT_RANGE_MAX_HEIGHT, Settings, default_encode_threads
from app.core.errors import AppError
from app.core.fs import atomic_write_json, project_dir, read_json
from app.core.models import HeatPoint, VideoMeta
from app.domain.languages import normalize_language
from app.domain.ranges import RangeFile, normalize_ranges

log = logging.getLogger(__name__)

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
_YT_HOSTS = {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}
SAFETY_MARGIN = 2 * 1024**3
RANGE_DURATION_TOLERANCE_SEC = 2.0


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
    chapters = [{"start": float(c["start_time"]), "end": float(c["end_time"]), "title": c.get("title") or ""}
                for c in (info.get("chapters") or [])]
    subs = info.get("subtitles") or {}
    auto_caps = info.get("automatic_captions") or {}
    orig = sorted(k.removesuffix("-orig") for k in auto_caps if k.endswith("-orig") and k != "-orig")
    subtitles = {"manual": sorted(lang for lang in subs if lang != "live_chat"),
                 "auto": orig if orig else sorted(auto_caps)}
    raw_lang = info.get("language")
    try:
        lang = normalize_language(raw_lang) if isinstance(raw_lang, str) else None
    except AppError:
        lang = None
    return VideoMeta(source_type="youtube", project_id=info["id"], title=info.get("title") or info["id"],
                     channel=info.get("channel") or info.get("uploader"), duration=float(info.get("duration") or 0),
                     video_id=info["id"], heatmap=heat, chapters=chapters, subtitles=subtitles,
                     has_live_chat="live_chat" in subs, language=lang)


def ytdl_base_opts(settings: Settings, cookies_browser: str = "") -> dict:
    """yt-dlp options shared by probing and downloading."""
    opts: dict = {"quiet": True, "noprogress": True, "noplaylist": True}
    if cookies_browser:
        opts["cookiesfrombrowser"] = (cookies_browser,)
    return opts


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
    if ("could not copy" in low and "cookie" in low) or "database is locked" in low:
        return IngestError("DOWNLOAD_BLOCKED", "Cookie browser terkunci. Tutup browser itu atau pilih browser "
                                               "lain di pengaturan cookies_browser.")
    if any(k in low for k in ("confirm your age", "age-restricted", "sign in to confirm")):
        return IngestError("AGE_RESTRICTED", "Video ini dibatasi umur. Isi pengaturan cookies_browser "
                                             "(mis. chrome) lalu coba lagi.")
    if any(k in low for k in ("private video", "video unavailable", "removed", "members-only")):
        return IngestError("VIDEO_UNAVAILABLE", "Video ini privat, dihapus, atau tidak tersedia.")
    return IngestError("NETWORK", f"Gagal mengunduh: {msg[-200:]}", retryable=True)


def _watch_url(vid: str) -> str:
    return f"https://www.youtube.com/watch?v={vid}"


def probe_youtube(url: str, settings: Settings, cookies_browser: str = "") -> VideoMeta:
    """Read video metadata (duration, heatmap, chapters, subtitles, live chat) without downloading."""
    vid = parse_youtube_id(url)
    try:
        with YoutubeDL(ytdl_base_opts(settings, cookies_browser)) as ydl:
            info = ydl.extract_info(_watch_url(vid), download=False)
    except DownloadError as e:
        raise _map_download_error(e) from e
    return meta_from_info(info)


def _range_files(ranges: list[tuple[float, float]]) -> list[RangeFile]:
    return [RangeFile(i, s, e, f"source_r{i}.mp4") for i, (s, e) in enumerate(ranges)]


def _cached_meta(meta_path: Path, d: Path, ranges: list | None) -> VideoMeta | None:
    """The saved meta when it matches the requested ranges and all its files exist."""
    cached = read_json(meta_path)
    if not cached:
        return None
    meta = VideoMeta.model_validate(cached)
    if not ranges:
        return meta if not meta.ranges and (d / "source.mp4").exists() else None
    wanted = [(f.start, f.end) for f in _range_files(normalize_ranges(ranges, meta.duration or None))]
    plan = read_json(d / ".ranges_plan.json")
    source = plan if isinstance(plan, list) else meta.ranges
    have = [(float(r["start"]), float(r["end"])) for r in source]
    if have == wanted and len(meta.ranges) == len(wanted) and all((d / r["file"]).exists() for r in meta.ranges):
        return meta
    return None


def _adjust_range_to_file_duration(rf: RangeFile, path: Path) -> RangeFile:
    """Adjust `rf.end` to `start + real_duration` if ffprobe differs by > 2 s from requested length."""
    expected = rf.end - rf.start
    try:
        real = probe_duration(path)
    except (subprocess.SubprocessError, OSError, ValueError, KeyError, TypeError) as e:
        log.warning("Gagal membaca durasi %s dengan ffprobe: %s", rf.file, e)
        return rf
    if real <= 0:
        log.warning("Durasi ffprobe untuk %s tidak valid (%.3f dtk); memakai rentang diminta.", rf.file, real)
        return rf
    if abs(real - expected) > RANGE_DURATION_TOLERANCE_SEC:
        new_end = round(rf.start + real, 3) + 0.0
        log.warning(
            "Durasi %s (%.2f dtk) berbeda > %.0f dtk dari rentang diminta (%.2f dtk); "
            "end disesuaikan ke %.3f.",
            rf.file, real, RANGE_DURATION_TOLERANCE_SEC, expected, new_end,
        )
        return RangeFile(rf.index, rf.start, new_end, rf.file)
    return rf


def ingest_youtube(url: str, settings: Settings, project_id: str | None = None,
                   ranges: list[list[float]] | None = None, cookies_browser: str = "",
                   encode_threads: int | None = None, range_max_height: int | None = None) -> tuple[Path, VideoMeta]:
    """Download a YouTube video into the project folder.

    `project_id` names the project folder and `meta.project_id`; default is the YouTube video id.
    Without `ranges` the whole video becomes `source.mp4`. With `ranges` each normalized range is
    downloaded to `source_r<i>.mp4` (keyframe-accurate cuts with fast ultrafast re-encode); files
    already present are kept, so an interrupted ingest resumes. Returns the first media file and meta.
    """
    vid = parse_youtube_id(url)
    d = project_dir(settings, project_id or vid)
    meta_path = d / "meta.json"
    cached = _cached_meta(meta_path, d, ranges)
    if cached:
        first = cached.ranges[0]["file"] if cached.ranges else "source.mp4"
        return d / first, cached

    h = settings.max_video_height
    opts = {
        **ytdl_base_opts(settings, cookies_browser),
        "format": f"bv*[height<={h}][ext=mp4]+ba[ext=m4a]/b[height<={h}]",
        "merge_output_format": "mp4",
        "outtmpl": str(d / "source.%(ext)s"),
    }
    try:
        with YoutubeDL(opts) as ydl:
            info = ydl.extract_info(_watch_url(vid), download=False)
            norm = normalize_ranges(ranges, float(info.get("duration") or 0) or None)
            if norm is None:
                size = info.get("filesize_approx") or int((info.get("duration") or 3600) * 1.5 * 1024**2)
                ensure_disk_space(d, size)
                ydl.download([_watch_url(vid)])
        files = _range_files(norm) if norm else []
        plan_path = d / ".ranges_plan.json"
        if files:
            wanted = [asdict(f) for f in files]
            saved = read_json(plan_path)
            if saved is None:
                prev_meta = read_json(meta_path)
                saved = prev_meta.get("ranges") if prev_meta else None
            if saved is not None and saved != wanted:
                for stale in d.glob("source_r*.mp4"):
                    stale.unlink(missing_ok=True)
            atomic_write_json(plan_path, wanted)
            ensure_disk_space(d, int(sum(f.end - f.start for f in files) * 1.5 * 1024**2))
        threads = max(1, encode_threads if encode_threads is not None else default_encode_threads())
        rh = range_max_height if range_max_height is not None else DEFAULT_RANGE_MAX_HEIGHT
        adjusted: list[RangeFile] = []
        for f in files:
            out_path = d / f.file
            if not out_path.exists():
                range_opts = {
                    **opts,
                    "format": f"bv*[height<={rh}][ext=mp4]+ba[ext=m4a]/b[height<={rh}]",
                    "outtmpl": str(d / f"source_r{f.index}.%(ext)s"),
                    "download_ranges": download_range_func(None, [(f.start, f.end)]),
                    "force_keyframes_at_cuts": True,
                    "external_downloader_args": {
                        "ffmpeg_o": [
                            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
                            "-threads", str(threads), "-c:a", "aac", "-b:a", "128k",
                        ],
                    },
                }
                with YoutubeDL(range_opts) as ydl:
                    ydl.download([_watch_url(vid)])
                if not out_path.exists():
                    raise IngestError("NETWORK", f"Unduhan rentang ke-{f.index + 1} selesai tapi file videonya "
                                                 "tidak ditemukan.", retryable=True)
            adjusted.append(_adjust_range_to_file_duration(f, out_path))
    except DownloadError as e:
        raise _map_download_error(e) from e
    if not adjusted and not (d / "source.mp4").exists():
        raise IngestError("NETWORK", "Unduhan selesai tapi file video tidak ditemukan.", retryable=True)
    meta = meta_from_info(info).model_copy(update={"ranges": [asdict(f) for f in adjusted]})
    if project_id:
        meta = meta.model_copy(update={"project_id": project_id})
    atomic_write_json(meta_path, meta.model_dump())
    if not adjusted:
        (d / ".ranges_plan.json").unlink(missing_ok=True)
    return d / (adjusted[0].file if adjusted else "source.mp4"), meta


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


def ingest_file(path: Path, settings: Settings, project_id: str | None = None,
                ranges: list[list[float]] | None = None) -> tuple[Path, VideoMeta]:
    """`project_id` names the project folder and `meta.project_id`; default is a content hash of the file."""
    if ranges:
        raise IngestError("VALIDATION", "Rentang untuk file unggahan belum didukung")
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
