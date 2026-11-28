"""Download, parse, and range-clip YouTube json3 subtitles."""
import re
from pathlib import Path
from typing import Literal

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from app.core.config import Settings
from app.core.errors import AppError
from app.core.models import VideoMeta, Word
from app.domain.ranges import RangeFile
from app.pipeline.ingest import _map_download_error, _watch_url, ytdl_base_opts

SUBTITLE_PAUSE_SPLIT_SEC = 1.0
TRANSCRIPT_SOURCES = ("auto", "youtube_manual", "youtube_auto", "whisper")
_TAG_RE = re.compile(r"<[^>]+>|\{\\[^}]*\}")


def _r(t: float) -> float:
    return round(t, 3) + 0.0


def _clean_words(text: str) -> list[str]:
    return [tok for tok in _TAG_RE.sub("", text).split() if tok]


def _distribute_words(toks: list[str], start: float, end: float, *, approx: bool) -> list[Word]:
    total = sum(len(t) for t in toks)
    dur = max(0.0, end - start)
    out: list[Word] = []
    acc = 0
    for tok in toks:
        w_start = _r(start + dur * (acc / total))
        acc += len(tok)
        w_end = _r(start + dur * (acc / total))
        out.append(Word(w=tok, start=w_start, end=w_end, approx=approx))
    return out


def resolve_transcript_source(requested: str, meta: VideoMeta, target_language: str) -> str:
    """Resolve `auto|youtube_manual|youtube_auto|whisper` against `meta.subtitles` for `target_language`."""
    if requested not in TRANSCRIPT_SOURCES:
        raise AppError("VALIDATION", f"Sumber transkrip harus salah satu dari: {', '.join(TRANSCRIPT_SOURCES)}.",
                       status=422)
    if meta.source_type == "upload":
        return "whisper"
    manual = meta.subtitles.get("manual") or []
    auto = meta.subtitles.get("auto") or []
    if requested == "auto":
        return "youtube_manual" if target_language in manual else "whisper"
    if requested == "whisper":
        return "whisper"
    if requested == "youtube_manual":
        if target_language not in manual:
            raise AppError("VALIDATION", f"Subtitle resmi bahasa '{target_language}' tidak tersedia.", status=422)
        return "youtube_manual"
    if target_language not in auto:
        raise AppError("VALIDATION", f"Subtitle otomatis bahasa '{target_language}' tidak tersedia.", status=422)
    return "youtube_auto"


def _auto_sub_lang(video_id: str, lang: str, settings: Settings, cookies_browser: str) -> str:
    """Ask yt-dlp for `<lang>-orig` rather than machine-translated `<lang>` when available."""
    with YoutubeDL(ytdl_base_opts(settings, cookies_browser)) as ydl:
        extract = getattr(ydl, "extract_info", None)
        if not callable(extract):
            return lang
        info = extract(_watch_url(video_id), download=False) or {}
    caps = info.get("automatic_captions") or {}
    orig = f"{lang}-orig"
    return orig if orig in caps else lang


def download_subs(video_id: str, lang: str, kind: Literal["manual", "auto"], out: Path,
                  settings: Settings, cookies_browser: str = "") -> Path:
    """Download YouTube `json3` subtitles into `out` (`projects/<id>/subs/<video_id>.<lang>.json3`), reusing cache."""
    out = Path(out)
    sub_dir = out.parent if out.suffix == ".json3" else out
    dest = out if out.suffix == ".json3" else sub_dir / f"{video_id}.{lang}.json3"
    if dest.exists():
        return dest
    sub_dir.mkdir(parents=True, exist_ok=True)
    try:
        sub_lang = _auto_sub_lang(video_id, lang, settings, cookies_browser) if kind == "auto" else lang
        opts = {
            **ytdl_base_opts(settings, cookies_browser),
            "skip_download": True,
            "writesubtitles": kind == "manual",
            "writeautomaticsub": kind == "auto",
            "subtitleslangs": [sub_lang],
            "subtitlesformat": "json3",
            "outtmpl": str(sub_dir / f"{video_id}.%(ext)s"),
        }
        with YoutubeDL(opts) as ydl:
            ydl.download([_watch_url(video_id)])
    except DownloadError as e:
        mapped = _map_download_error(e)
        raise AppError(mapped.code, str(mapped), retryable=mapped.retryable) from e
    written = sub_dir / f"{video_id}.{sub_lang}.json3"
    if not written.exists() and sub_lang != lang and (sub_dir / f"{video_id}.{lang}.json3").exists():
        written = sub_dir / f"{video_id}.{lang}.json3"
    if not written.exists():
        raise AppError("INVALID_STATE", f"Subtitle {lang} tidak tersedia")
    if written != dest:
        written.replace(dest)
    return dest


def parse_json3(data: dict) -> list[Word]:
    """Parse YouTube json3 events into Words on the original timeline."""
    words: list[Word] = []
    for ev in data.get("events") or []:
        if not isinstance(ev, dict) or ev.get("aAppend"):
            continue
        segs = ev.get("segs")
        t_start = ev.get("tStartMs")
        dur_ms = ev.get("dDurationMs")
        if not isinstance(segs, list) or not segs or t_start is None or dur_ms is None or int(dur_ms) <= 0:
            continue
        start_ms = int(t_start)
        end_ms = start_ms + int(dur_ms)
        if any(isinstance(s, dict) and "tOffsetMs" in s for s in segs):
            items: list[tuple[int, list[str]]] = []
            for s in segs:
                if not isinstance(s, dict):
                    continue
                toks = _clean_words(s.get("utf8") or "")
                if toks:
                    items.append((start_ms + int(s.get("tOffsetMs") or 0), toks))
            for idx, (seg_ms, toks) in enumerate(items):
                next_ms = items[idx + 1][0] if idx + 1 < len(items) else end_ms
                w_start = _r(seg_ms / 1000.0)
                w_end = _r(max(seg_ms, next_ms) / 1000.0)
                if len(toks) == 1:
                    words.append(Word(w=toks[0], start=w_start, end=w_end, approx=False))
                else:
                    words.extend(_distribute_words(toks, w_start, w_end, approx=False))
        else:
            raw_text = "".join(s.get("utf8") or "" for s in segs if isinstance(s, dict))
            toks = _clean_words(raw_text)
            if toks:
                words.extend(_distribute_words(toks, _r(start_ms / 1000.0), _r(end_ms / 1000.0), approx=True))
    return words


def clip_words_to_ranges(words: list[Word], ranges: list[RangeFile]) -> list[tuple[RangeFile, list[Word]]]:
    """Group original-timeline words by range and return range-local word times (w - rf.start)."""
    out: list[tuple[RangeFile, list[Word]]] = []
    for rf in sorted(ranges, key=lambda r: r.start):
        local: list[Word] = []
        for w in words:
            if rf.start <= w.start < rf.end:
                end = min(w.end, rf.end)
                local.append(w.model_copy(update={
                    "start": _r(w.start - rf.start),
                    "end": _r(max(w.start, end) - rf.start),
                }))
        out.append((rf, local))
    return out
