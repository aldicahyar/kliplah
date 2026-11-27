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
    ("ERROR: Sign in to confirm your age", "VIDEO_UNAVAILABLE", False),
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
