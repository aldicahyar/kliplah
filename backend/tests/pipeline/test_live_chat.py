"""Tests for live chat downloading, parsing, and peak detection."""
from pathlib import Path

import pytest

from app.core.config import Settings
from app.pipeline import live_chat
from app.pipeline.live_chat import chat_peaks, download_live_chat, load_chat_keywords, parse_live_chat

FIXTURES = Path(__file__).parents[1] / "fixtures"


def test_parse_live_chat_offsets_and_text():
    msgs = parse_live_chat(FIXTURES / "live_chat_sample.jsonl")
    assert msgs == [
        (1.5, "halo semuanya"),
        (12.3, "wkwkwk 😂"),
        (45.0, "mantap bang"),
        (60.25, "kaget anjir"),
    ]


def test_chat_peak_detects_burst_with_keywords():
    keywords = load_chat_keywords()
    assert "wkwk" in keywords
    # 30 minutes = 1800 s, 1 normal message every 10 s
    base = [(i * 10.0 + 1.0, "halo") for i in range(180)]
    assert chat_peaks(base, 1800.0, keywords) == []

    burst = base + [(720.0 + k * 0.2, "wkwkwk") for k in range(25)]
    peaks = chat_peaks(burst, 1800.0, keywords)
    assert len(peaks) == 1
    assert peaks[0]["start"] == 720.0 and peaks[0]["end"] == 730.0
    assert peaks[0]["value"] > 0.0


def test_chat_keyword_counts_double():
    keywords = ["wkwk"]
    # 300 s baseline of 2 msgs per 10 s window -> median 2, scale 1.0
    base = [(i * 10.0 + j, "biasa") for i in range(30) for j in (1.0, 2.0)]
    # 2 plain messages added in window 15 (total 4 -> z = 2.0 -> value = 0.0)
    plain = base + [(150.5, "halo"), (151.5, "iya")]
    # 2 keyword messages added in window 15 (total 2 + 2*2 = 6 -> z = 4.0 -> value = 0.667)
    kw_msgs = base + [(150.5, "wkwkwk"), (151.5, "wkwk")]
    p_plain = chat_peaks(plain, 300.0, keywords)
    p_kw = chat_peaks(kw_msgs, 300.0, keywords)
    assert p_plain[0]["value"] < p_kw[0]["value"]
    assert p_kw[0]["value"] == pytest.approx(2 / 3, abs=0.002)


def test_download_live_chat_options_reuse_and_unavailable(monkeypatch, tmp_path):
    s = Settings(_env_file=None, data_dir=tmp_path)
    out = tmp_path / "projects" / "p1" / "live_chat.json"
    calls = []

    class _ChatYDL:
        write_chat = False

        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def download(self, urls):
            calls.append((self.opts, urls))
            if self.write_chat:
                p = Path(self.opts["outtmpl"].replace("%(ext)s", "live_chat.json"))
                p.write_text((FIXTURES / "live_chat_sample.jsonl").read_text(encoding="utf-8"), encoding="utf-8")

    monkeypatch.setattr(live_chat, "YoutubeDL", _ChatYDL)
    assert download_live_chat("dQw4w9WgXcQ", out, s, cookies_browser="chrome") is None
    assert calls[0][0]["subtitleslangs"] == ["live_chat"]
    assert calls[0][0]["skip_download"] is True
    assert calls[0][0]["cookiesfrombrowser"] == ("chrome",)

    _ChatYDL.write_chat = True
    got = download_live_chat("dQw4w9WgXcQ", out, s)
    assert got == out and got.exists()

    # existing file reused
    monkeypatch.setattr(live_chat, "YoutubeDL", None)
    assert download_live_chat("dQw4w9WgXcQ", out, s) == out
