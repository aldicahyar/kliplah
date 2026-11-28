import json
from pathlib import Path

import pytest

from app.core.config import Settings
from app.pipeline import ingest
from app.pipeline.ingest import (
    IngestError,
    ensure_disk_space,
    ingest_youtube,
    meta_from_info,
    parse_youtube_id,
    probe_youtube,
    ytdl_base_opts,
)

FIXTURES = Path(__file__).parents[1] / "fixtures"


@pytest.fixture
def info():
    return json.loads((FIXTURES / "ytdlp_info.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("url,vid", [
    ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=10", "dQw4w9WgXcQ"),
    ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://youtu.be/dQw4w9WgXcQ?si=abc", "dQw4w9WgXcQ"),
    ("https://m.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://www.youtube.com/live/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
    ("https://youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
])
def test_parse_youtube_id(url, vid):
    assert parse_youtube_id(url) == vid


@pytest.mark.parametrize("url", ["https://vimeo.com/1", "bukan url", "https://www.youtube.com/watch?v=short"])
def test_parse_youtube_id_rejects(url):
    with pytest.raises(IngestError) as e:
        parse_youtube_id(url)
    assert e.value.code == "INVALID_URL" and e.value.retryable is False


def test_meta_from_info_with_heatmap(info):
    m = meta_from_info(info)
    assert m.project_id == info["id"] and m.video_id == info["id"]
    assert m.source_type == "youtube" and m.duration == 3612.0
    assert len(m.heatmap) == 3 and m.heatmap[0].value == 1.0


def test_meta_from_info_without_heatmap(info):
    info.pop("heatmap")
    assert meta_from_info(info).heatmap == []
    info["heatmap"] = None
    assert meta_from_info(info).heatmap == []


def test_ensure_disk_space_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest.shutil, "disk_usage", lambda p: (0, 0, 1 * 1024**3))
    with pytest.raises(IngestError) as e:
        ensure_disk_space(tmp_path, 100)
    assert e.value.code == "DISK_FULL"


def test_ensure_disk_space_ok(monkeypatch, tmp_path):
    monkeypatch.setattr(ingest.shutil, "disk_usage", lambda p: (0, 0, 10 * 1024**3))
    ensure_disk_space(tmp_path, 1024**3)


class _FakeYDL:
    error: str | None = None

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, url, download=False):
        raise ingest.DownloadError(self.error)


@pytest.mark.parametrize("msg,code,retryable", [
    ("ERROR: [youtube] x: Private video. Sign in if you've been granted access", "VIDEO_UNAVAILABLE", False),
    ("ERROR: [youtube] x: Video unavailable", "VIDEO_UNAVAILABLE", False),
    ("ERROR: Sign in to confirm your age", "AGE_RESTRICTED", False),
    ("ERROR: [youtube] x: Sign in to confirm you're not a bot", "DOWNLOAD_BLOCKED", False),
    ("ERROR: Unable to download webpage: timed out", "NETWORK", True),
])
def test_ingest_youtube_maps_errors(monkeypatch, tmp_path, msg, code, retryable):
    _FakeYDL.error = msg
    monkeypatch.setattr(ingest, "YoutubeDL", _FakeYDL)
    s = Settings(_env_file=None, data_dir=tmp_path)
    with pytest.raises(IngestError) as e:
        ingest_youtube("https://youtu.be/dQw4w9WgXcQ", s)
    assert e.value.code == code and e.value.retryable is retryable


def test_ingest_youtube_skips_when_checkpoint_exists(monkeypatch, tmp_path, info):
    s = Settings(_env_file=None, data_dir=tmp_path)
    d = tmp_path / "projects" / "dQw4w9WgXcQ"
    d.mkdir(parents=True)
    (d / "source.mp4").write_bytes(b"x")
    info["id"] = "dQw4w9WgXcQ"
    (d / "meta.json").write_text(json.dumps(meta_from_info(info).model_dump()), encoding="utf-8")
    monkeypatch.setattr(ingest, "YoutubeDL", None)  # would crash if used
    src, meta = ingest_youtube("https://youtu.be/dQw4w9WgXcQ", s)
    assert src == d / "source.mp4" and meta.project_id == "dQw4w9WgXcQ"


class _DownloadingYDL:
    """Fake yt-dlp that 'downloads' by writing the output file named by outtmpl."""
    info: dict = {}

    def __init__(self, opts):
        self.opts = opts

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def extract_info(self, url, download=False):
        return self.info

    def download(self, urls):
        Path(self.opts["outtmpl"].replace("%(ext)s", "mp4")).write_bytes(b"v")


def test_ingest_youtube_project_id_overrides_folder(monkeypatch, tmp_path, info):
    info["id"] = "dQw4w9WgXcQ"
    _DownloadingYDL.info = info
    monkeypatch.setattr(ingest, "YoutubeDL", _DownloadingYDL)
    monkeypatch.setattr(ingest, "ensure_disk_space", lambda p, n: None)
    s = Settings(_env_file=None, data_dir=tmp_path)
    src, meta = ingest_youtube("https://youtu.be/dQw4w9WgXcQ", s, project_id="proj-abc")
    d = tmp_path / "projects" / "proj-abc"
    assert src == d / "source.mp4" and src.exists()
    assert meta.project_id == "proj-abc" and meta.video_id == "dQw4w9WgXcQ"
    assert json.loads((d / "meta.json").read_text(encoding="utf-8"))["project_id"] == "proj-abc"
    assert not (tmp_path / "projects" / "dQw4w9WgXcQ").exists()


def test_ingest_file_project_id_overrides_folder(monkeypatch, tmp_path):
    video = tmp_path / "upload.mp4"
    video.write_bytes(b"tiny")
    monkeypatch.setattr(ingest, "probe_duration", lambda p: 12.5)
    s = Settings(_env_file=None, data_dir=tmp_path / "data")
    src, meta = ingest.ingest_file(video, s, project_id="proj-xyz")
    d = tmp_path / "data" / "projects" / "proj-xyz"
    assert src == d / "source.mp4" and src.read_bytes() == b"tiny"
    assert meta.project_id == "proj-xyz" and meta.source_type == "upload" and meta.duration == 12.5
    assert json.loads((d / "meta.json").read_text(encoding="utf-8"))["project_id"] == "proj-xyz"


class _RangeYDL(_DownloadingYDL):
    """Records every YoutubeDL instance so tests can inspect options per download."""
    instances: list = []
    downloads: list = []

    def __init__(self, opts):
        super().__init__(opts)
        _RangeYDL.instances.append(self)

    def download(self, urls):
        _RangeYDL.downloads.append(self.opts)
        super().download(urls)


@pytest.fixture
def range_ydl(monkeypatch, info):
    info["id"] = "dQw4w9WgXcQ"
    _RangeYDL.info, _RangeYDL.instances, _RangeYDL.downloads = info, [], []
    monkeypatch.setattr(ingest, "YoutubeDL", _RangeYDL)
    monkeypatch.setattr(ingest, "ensure_disk_space", lambda p, n: None)
    return _RangeYDL


def test_ingest_with_ranges_downloads_each_range_with_keyframe_cuts(tmp_path, range_ydl):
    s = Settings(_env_file=None, data_dir=tmp_path)
    src, meta = ingest_youtube("https://youtu.be/dQw4w9WgXcQ", s, ranges=[[1800, 1900], [60, 120.5]])
    d = tmp_path / "projects" / "dQw4w9WgXcQ"
    assert len(range_ydl.downloads) == 2
    for opts in range_ydl.downloads:
        assert opts["force_keyframes_at_cuts"] is True and callable(opts["download_ranges"])
    assert [o["outtmpl"] for o in range_ydl.downloads] == [str(d / "source_r0.%(ext)s"), str(d / "source_r1.%(ext)s")]
    sections = [list(o["download_ranges"]({}, None)) for o in range_ydl.downloads]
    assert sections[0][0]["start_time"] == 60 and sections[0][0]["end_time"] == 120.5
    assert sections[1][0]["start_time"] == 1800 and sections[1][0]["end_time"] == 1900
    assert (d / "source_r0.mp4").exists() and (d / "source_r1.mp4").exists() and not (d / "source.mp4").exists()
    assert src == d / "source_r0.mp4"
    assert meta.ranges == [{"index": 0, "start": 60.0, "end": 120.5, "file": "source_r0.mp4"},
                           {"index": 1, "start": 1800.0, "end": 1900.0, "file": "source_r1.mp4"}]
    assert json.loads((d / "meta.json").read_text(encoding="utf-8"))["ranges"] == meta.ranges


def test_ingest_ranges_resume_skips_existing_files(tmp_path, range_ydl):
    s = Settings(_env_file=None, data_dir=tmp_path)
    d = tmp_path / "projects" / "dQw4w9WgXcQ"
    d.mkdir(parents=True)
    (d / "source_r0.mp4").write_bytes(b"done")
    _, meta = ingest_youtube("https://youtu.be/dQw4w9WgXcQ", s, ranges=[[60, 120], [1800, 1900]])
    assert [o["outtmpl"] for o in range_ydl.downloads] == [str(d / "source_r1.%(ext)s")]
    assert (d / "source_r0.mp4").read_bytes() == b"done"
    assert [r["file"] for r in meta.ranges] == ["source_r0.mp4", "source_r1.mp4"]


def test_ingest_ranges_missing_file_after_download_is_retryable(monkeypatch, tmp_path, range_ydl):
    monkeypatch.setattr(range_ydl, "download", lambda self, urls: None)
    s = Settings(_env_file=None, data_dir=tmp_path)
    with pytest.raises(IngestError) as e:
        ingest_youtube("https://youtu.be/dQw4w9WgXcQ", s, ranges=[[60, 120]])
    assert e.value.code == "NETWORK" and e.value.retryable is True


def test_ingest_ranges_disk_check_uses_total_range_length(monkeypatch, tmp_path, range_ydl):
    seen = []
    monkeypatch.setattr(ingest, "ensure_disk_space", lambda p, n: seen.append(n))
    s = Settings(_env_file=None, data_dir=tmp_path)
    ingest_youtube("https://youtu.be/dQw4w9WgXcQ", s, ranges=[[60, 120], [1800, 1900]])
    assert seen == [int(160 * 1.5 * 1024**2)]


def test_ingest_without_ranges_keeps_s1_behaviour(tmp_path, range_ydl):
    s = Settings(_env_file=None, data_dir=tmp_path)
    src, meta = ingest_youtube("https://youtu.be/dQw4w9WgXcQ", s)
    d = tmp_path / "projects" / "dQw4w9WgXcQ"
    assert src == d / "source.mp4" and src.exists()
    assert meta.ranges == []
    assert len(range_ydl.downloads) == 1
    assert "download_ranges" not in range_ydl.downloads[0]
    assert range_ydl.downloads[0]["outtmpl"] == str(d / "source.%(ext)s")


def test_cookies_browser_passed_to_ytdlp(tmp_path, range_ydl):
    s = Settings(_env_file=None, data_dir=tmp_path)
    ingest_youtube("https://youtu.be/dQw4w9WgXcQ", s, ranges=[[60, 120]], cookies_browser="edge")
    assert range_ydl.instances and all(i.opts["cookiesfrombrowser"] == ("edge",) for i in range_ydl.instances)


def test_ytdl_base_opts_without_cookies(tmp_path):
    opts = ytdl_base_opts(Settings(_env_file=None, data_dir=tmp_path))
    assert opts["quiet"] is True and opts["noprogress"] is True and opts["noplaylist"] is True
    assert "cookiesfrombrowser" not in opts


@pytest.mark.parametrize("msg", [
    "ERROR: [youtube] x: Sign in to confirm your age. This video may be inappropriate for some users.",
    "ERROR: [youtube] x: This video is age-restricted",
])
def test_age_restricted_error_mentions_cookies(monkeypatch, tmp_path, msg):
    _FakeYDL.error = msg
    monkeypatch.setattr(ingest, "YoutubeDL", _FakeYDL)
    with pytest.raises(IngestError) as e:
        ingest_youtube("https://youtu.be/dQw4w9WgXcQ", Settings(_env_file=None, data_dir=tmp_path))
    assert e.value.code == "AGE_RESTRICTED" and e.value.retryable is False
    assert str(e.value) == "Video ini dibatasi umur. Isi pengaturan cookies_browser (mis. chrome) lalu coba lagi."


@pytest.mark.parametrize("msg", [
    "ERROR: Could not copy Chrome cookie database. See https://github.com/yt-dlp/yt-dlp/issues/7271",
    "sqlite3.OperationalError: database is locked",
])
def test_locked_cookie_db_message(monkeypatch, tmp_path, msg):
    _FakeYDL.error = msg
    monkeypatch.setattr(ingest, "YoutubeDL", _FakeYDL)
    with pytest.raises(IngestError) as e:
        ingest_youtube("https://youtu.be/dQw4w9WgXcQ", Settings(_env_file=None, data_dir=tmp_path))
    assert e.value.code == "DOWNLOAD_BLOCKED" and e.value.retryable is False
    assert str(e.value) == ("Cookie browser terkunci. Tutup browser itu atau pilih browser lain "
                            "di pengaturan cookies_browser.")


def test_probe_reports_subtitles_chat_chapters(monkeypatch, tmp_path, info):
    info["subtitles"] = {"live_chat": [{"ext": "json"}], "id": [{"ext": "vtt"}], "en": [{"ext": "vtt"}]}
    info["automatic_captions"] = {"id": [{"ext": "vtt"}], "de": [{"ext": "vtt"}]}
    info["chapters"] = [{"start_time": 0.0, "end_time": 600.0, "title": "Pembuka"},
                        {"start_time": 600.0, "end_time": 3612.0, "title": "Inti"}]
    calls = []

    class _ProbeYDL(_DownloadingYDL):
        def extract_info(self, url, download=False):
            calls.append((self.opts, download))
            return info

        def download(self, urls):
            raise AssertionError("probe must not download")

    monkeypatch.setattr(ingest, "YoutubeDL", _ProbeYDL)
    meta = probe_youtube("https://youtu.be/dQw4w9WgXcQ", Settings(_env_file=None, data_dir=tmp_path),
                         cookies_browser="firefox")
    assert calls and calls[0][1] is False and calls[0][0]["cookiesfrombrowser"] == ("firefox",)
    assert meta.duration == 3612.0 and len(meta.heatmap) == 3
    assert meta.subtitles == {"manual": ["en", "id"], "auto": ["de", "id"]}
    assert meta.has_live_chat is True
    assert meta.chapters == [{"start": 0.0, "end": 600.0, "title": "Pembuka"},
                             {"start": 600.0, "end": 3612.0, "title": "Inti"}]
    assert meta.ranges == []


def test_probe_without_subtitle_fields(monkeypatch, tmp_path, info):
    _DownloadingYDL.info = info
    monkeypatch.setattr(ingest, "YoutubeDL", _DownloadingYDL)
    meta = probe_youtube("https://youtu.be/dQw4w9WgXcQ", Settings(_env_file=None, data_dir=tmp_path))
    assert meta.subtitles == {"manual": [], "auto": []} and meta.has_live_chat is False and meta.chapters == []


def test_ingest_file_rejects_ranges(tmp_path):
    video = tmp_path / "upload.mp4"
    video.write_bytes(b"tiny")
    with pytest.raises(IngestError) as e:
        ingest.ingest_file(video, Settings(_env_file=None, data_dir=tmp_path / "data"), ranges=[[0, 60]])
    assert e.value.code == "VALIDATION" and str(e.value) == "Rentang untuk file unggahan belum didukung"


def test_cached_meta_with_zero_duration_uses_none(tmp_path, info):
    s = Settings(_env_file=None, data_dir=tmp_path)
    d = tmp_path / "projects" / "dQw4w9WgXcQ"
    d.mkdir(parents=True)
    (d / "source_r0.mp4").write_bytes(b"v")
    info["id"] = "dQw4w9WgXcQ"
    info["duration"] = 0.0
    meta = meta_from_info(info).model_copy(
        update={"ranges": [{"index": 0, "start": 60.0, "end": 120.0, "file": "source_r0.mp4"}]}
    )
    (d / "meta.json").write_text(meta.model_dump_json(), encoding="utf-8")
    got = ingest._cached_meta(d / "meta.json", d, [[60, 120]])
    assert got == meta


def test_video_unavailable_message_does_not_mention_age(monkeypatch, tmp_path):
    _FakeYDL.error = "ERROR: [youtube] x: Video unavailable"
    monkeypatch.setattr(ingest, "YoutubeDL", _FakeYDL)
    with pytest.raises(IngestError) as e:
        ingest_youtube("https://youtu.be/dQw4w9WgXcQ", Settings(_env_file=None, data_dir=tmp_path))
    assert e.value.code == "VIDEO_UNAVAILABLE"
    assert "umur" not in str(e.value).lower()


def test_ingest_ranges_does_not_reuse_stale_files_when_ranges_differ(tmp_path, range_ydl):
    s = Settings(_env_file=None, data_dir=tmp_path)
    d = tmp_path / "projects" / "dQw4w9WgXcQ"
    _, meta1 = ingest_youtube("https://youtu.be/dQw4w9WgXcQ", s, ranges=[[60, 120]])
    assert meta1.ranges[0]["start"] == 60.0
    (d / "source_r0.mp4").write_bytes(b"stale")
    range_ydl.downloads.clear()

    _, meta2 = ingest_youtube("https://youtu.be/dQw4w9WgXcQ", s, ranges=[[1800, 1900]])
    assert len(range_ydl.downloads) == 1
    assert meta2.ranges[0]["start"] == 1800.0
    assert (d / "source_r0.mp4").read_bytes() == b"v"


def test_probe_filters_auto_captions_to_orig_languages(monkeypatch, tmp_path, info):
    from app.core.errors import AppError
    from app.pipeline.youtube_subs import resolve_transcript_source

    info["automatic_captions"] = {
        "en-orig": [{"ext": "json3"}],
        "en": [{"ext": "json3"}],
        "id": [{"ext": "json3"}],
    }
    _DownloadingYDL.info = info
    monkeypatch.setattr(ingest, "YoutubeDL", _DownloadingYDL)
    meta = probe_youtube("https://youtu.be/dQw4w9WgXcQ", Settings(_env_file=None, data_dir=tmp_path))
    assert meta.subtitles["auto"] == ["en"]
    with pytest.raises(AppError) as e:
        resolve_transcript_source("youtube_auto", meta, "id")
    assert e.value.code == "VALIDATION"


def test_range_download_uses_fast_encode_args(tmp_path, range_ydl, monkeypatch):
    import os

    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    s = Settings(_env_file=None, data_dir=tmp_path)
    ingest_youtube("https://youtu.be/dQw4w9WgXcQ", s, ranges=[[60, 120]], encode_threads=4)
    opts = range_ydl.downloads[0]
    assert opts["force_keyframes_at_cuts"] is True
    assert opts["external_downloader_args"] == {
        "ffmpeg_o": [
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "23",
            "-threads", "4", "-c:a", "aac", "-b:a", "128k",
        ]
    }

    # Default encode_threads = max(1, cpu_count - 2) = 6 when not supplied
    range_ydl.downloads.clear()
    ingest_youtube("https://youtu.be/dQw4w9WgXcQ", s, project_id="p-default", ranges=[[60, 120]])
    ff_args = range_ydl.downloads[0]["external_downloader_args"]["ffmpeg_o"]
    assert "-preset" in ff_args and ff_args[ff_args.index("-preset") + 1] == "ultrafast"
    assert "-threads" in ff_args and ff_args[ff_args.index("-threads") + 1] == "6"


def test_full_download_has_no_encode_args(tmp_path, range_ydl):
    s = Settings(_env_file=None, data_dir=tmp_path, max_video_height=1080)
    ingest_youtube("https://youtu.be/dQw4w9WgXcQ", s)
    opts = range_ydl.downloads[0]
    assert "external_downloader_args" not in opts
    assert "download_ranges" not in opts
    assert "force_keyframes_at_cuts" not in opts
    assert "height<=1080" in opts["format"]


def test_range_format_respects_range_max_height(tmp_path, range_ydl):
    s = Settings(_env_file=None, data_dir=tmp_path, max_video_height=1080)
    ingest_youtube("https://youtu.be/dQw4w9WgXcQ", s, project_id="p-720", ranges=[[60, 120]])
    assert "height<=720" in range_ydl.downloads[0]["format"]

    range_ydl.downloads.clear()
    ingest_youtube("https://youtu.be/dQw4w9WgXcQ", s, project_id="p-480", ranges=[[60, 120]], range_max_height=480)
    assert "height<=480" in range_ydl.downloads[0]["format"]
    assert s.max_video_height == 1080


def test_range_end_adjusted_to_real_file_duration(tmp_path, range_ydl, monkeypatch, caplog):
    import logging
    import subprocess
    from app.domain.ranges import SourceMap

    monkeypatch.setattr(
        ingest.subprocess, "run",
        lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, stdout=json.dumps({"format": {"duration": "1750.0"}})),
    )
    s = Settings(_env_file=None, data_dir=tmp_path)
    with caplog.at_level(logging.WARNING):
        _, meta = ingest_youtube("https://youtu.be/dQw4w9WgXcQ", s, ranges=[[1800, 3600]])
    d = tmp_path / "projects" / "dQw4w9WgXcQ"
    assert meta.ranges == [{"index": 0, "start": 1800.0, "end": 3550.0, "file": "source_r0.mp4"}]
    assert json.loads((d / "meta.json").read_text(encoding="utf-8"))["ranges"] == meta.ranges
    assert any("source_r0.mp4" in rec.message for rec in caplog.records)

    smap = SourceMap.from_meta(meta, d)
    assert smap.locate(3549.0) is not None and smap.locate(3580.0) is None

    # Cache hit on the same requested range preserves adjusted end without re-downloading
    range_ydl.downloads.clear()
    _, cached_meta = ingest_youtube("https://youtu.be/dQw4w9WgXcQ", s, ranges=[[1800, 3600]])
    assert range_ydl.downloads == [] and cached_meta.ranges == meta.ranges

    # Small drift (<= 2 s) leaves requested end untouched without warning
    caplog.clear()
    monkeypatch.setattr(
        ingest.subprocess, "run",
        lambda *a, **kw: subprocess.CompletedProcess(a[0], 0, stdout=json.dumps({"format": {"duration": "1798.5"}})),
    )
    _, meta_ok = ingest_youtube("https://youtu.be/dQw4w9WgXcQ", s, project_id="p-tol", ranges=[[1800, 3600]])
    assert meta_ok.ranges[0]["end"] == 3600.0
    assert caplog.records == []

    # ffprobe error logs warning and does not fail
    caplog.clear()

    def boom(*a, **kw):
        raise subprocess.CalledProcessError(1, "ffprobe")

    monkeypatch.setattr(ingest.subprocess, "run", boom)
    _, meta_err = ingest_youtube("https://youtu.be/dQw4w9WgXcQ", s, project_id="p-err", ranges=[[1800, 3600]])
    assert meta_err.ranges[0]["end"] == 3600.0
    assert any("ffprobe" in rec.message.lower() for rec in caplog.records)


def test_meta_from_info_reads_language_hint(info):
    info["language"] = "ko"
    assert meta_from_info(info).language == "ko"

    info["language"] = "pt-BR"
    assert meta_from_info(info).language == "pt"

    info.pop("language", None)
    assert meta_from_info(info).language is None

    info["language"] = "zz"
    assert meta_from_info(info).language is None
