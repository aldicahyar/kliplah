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


def _chat_line(offset_sec: float, text: str) -> str:
    import json

    return json.dumps({
        "replayChatItemAction": {
            "actions": [{"addChatItemAction": {"item": {"liveChatTextMessageRenderer": {
                "message": {"runs": [{"text": text}]},
            }}}}],
            "videoOffsetTimeMsec": str(int(round(offset_sec * 1000))),
        }
    })


def test_download_live_chat_stops_after_until_sec_via_progress_hook(monkeypatch, tmp_path):
    from yt_dlp.utils import DownloadCancelled

    s = Settings(_env_file=None, data_dir=tmp_path)
    out = tmp_path / "projects" / "p_stop" / "live_chat.json"
    written_batches: list[int] = []

    class _ChunkedChatYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def download(self, urls):
            part = Path(self.opts["outtmpl"].replace("%(ext)s", "live_chat.json.part"))
            dest = Path(self.opts["outtmpl"].replace("%(ext)s", "live_chat.json"))
            hooks = self.opts.get("progress_hooks") or []
            batches = [
                (0, [_chat_line(500.0, "awal"), _chat_line(900.0, "pra-rentang")]),
                (1, [_chat_line(1100.0, "dalam rentang"), _chat_line(1350.0, "lewat batas")]),  # > 1200 + 120
                (2, [_chat_line(3000.0, "jauh setelah batas")]),
            ]
            with open(part, "wb") as fh:
                for batch_idx, lines in batches:
                    written_batches.append(batch_idx)
                    fh.write(("\n".join(lines) + "\n").encode("utf-8"))
                    fh.flush()
                    for hook in hooks:
                        hook({"status": "downloading", "tmpfilename": str(part), "filename": str(dest)})
            part.replace(dest)

    monkeypatch.setattr(live_chat, "YoutubeDL", _ChunkedChatYDL)
    got = download_live_chat("dQw4w9WgXcQ", out, s, until_sec=1200.0)
    assert got == out and got.exists()
    # Batch 2 (3000s) was never fetched because batch 1 passed 1200 + 120 = 1320s
    assert written_batches == [0, 1]
    msgs = parse_live_chat(got)
    assert [t for t, _ in msgs] == [500.0, 900.0, 1100.0, 1350.0]
    assert not (out.parent / "dQw4w9WgXcQ.live_chat.json.part").exists()
    # DownloadCancelled must not have escaped
    assert DownloadCancelled is not None


def test_download_live_chat_without_until_sec_downloads_all_and_omits_hooks(monkeypatch, tmp_path):
    s = Settings(_env_file=None, data_dir=tmp_path)
    out = tmp_path / "projects" / "p_full" / "live_chat.json"
    seen_opts = {}

    class _FullChatYDL:
        def __init__(self, opts):
            self.opts = opts
            seen_opts.update(opts)

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def download(self, urls):
            dest = Path(self.opts["outtmpl"].replace("%(ext)s", "live_chat.json"))
            dest.write_text(_chat_line(500.0, "awal") + "\n" + _chat_line(9000.0, "akhir") + "\n", encoding="utf-8")

    monkeypatch.setattr(live_chat, "YoutubeDL", _FullChatYDL)
    got = download_live_chat("dQw4w9WgXcQ", out, s)
    assert got == out
    assert "progress_hooks" not in seen_opts
    assert [t for t, _ in parse_live_chat(got)] == [500.0, 9000.0]


def test_chat_peaks_accurate_for_midstream_range_when_stopped_early(monkeypatch, tmp_path):
    s = Settings(_env_file=None, data_dir=tmp_path)
    out = tmp_path / "projects" / "p_mid" / "live_chat.json"
    duration = 10800.0  # 3 hours

    # Baseline 1 msg per 10s across entire stream, burst at 5400-5420 (middle range 5100-6000), late burst at 9000s
    full_lines = [_chat_line(i * 10.0 + 1.0, "halo") for i in range(1080)]
    full_lines += [_chat_line(5400.0 + k * 0.2, "wkwkwk") for k in range(25)]
    full_lines += [_chat_line(9000.0 + k * 0.2, "wkwkwk") for k in range(25)]
    full_path = tmp_path / "full.live_chat.json"
    full_path.write_text("\n".join(full_lines) + "\n", encoding="utf-8")
    expected_mid_peaks = [
        p for p in chat_peaks(parse_live_chat(full_path), duration, load_chat_keywords())
        if 5100.0 <= p["start"] <= 6000.0
    ]

    # Sort all lines into 600s chunks, like yt-dlp replay fragments
    parsed_pairs = []
    for raw_line in full_lines:
        import json
        ms = int(json.loads(raw_line)["replayChatItemAction"]["videoOffsetTimeMsec"])
        parsed_pairs.append((ms / 1000.0, raw_line))
    parsed_pairs.sort(key=lambda x: x[0])

    class _StreamChatYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def download(self, urls):
            part = Path(self.opts["outtmpl"].replace("%(ext)s", "live_chat.json.part"))
            dest = Path(self.opts["outtmpl"].replace("%(ext)s", "live_chat.json"))
            hooks = self.opts.get("progress_hooks") or []
            with open(part, "wb") as fh:
                for chunk_start in range(0, int(duration), 600):
                    chunk_lines = [ln for t, ln in parsed_pairs if chunk_start <= t < chunk_start + 600]
                    if chunk_lines:
                        fh.write(("\n".join(chunk_lines) + "\n").encode("utf-8"))
                        fh.flush()
                    for hook in hooks:
                        hook({"status": "downloading", "tmpfilename": str(part), "filename": str(dest)})
            part.replace(dest)

    monkeypatch.setattr(live_chat, "YoutubeDL", _StreamChatYDL)
    got = download_live_chat("dQw4w9WgXcQ", out, s, until_sec=6000.0)
    msgs = parse_live_chat(got)
    # Pre-range baseline (e.g. 4500-5100) preserved; late burst at 9000s never downloaded
    assert any(4500.0 <= t < 5100.0 for t, _ in msgs)
    assert max(t for t, _ in msgs) < 6700.0
    got_mid_peaks = [
        p for p in chat_peaks(msgs, duration, load_chat_keywords())
        if 5100.0 <= p["start"] <= 6000.0
    ]
    assert got_mid_peaks == expected_mid_peaks and len(got_mid_peaks) == 1
    assert got_mid_peaks[0]["start"] == 5400.0 and got_mid_peaks[0]["end"] == 5410.0
