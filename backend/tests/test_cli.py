import json
from pathlib import Path

import pytest

from app import cli
from app.core.config import Settings
from app.core.models import ReframeSegment, VideoMeta, Word
from app.pipeline.ingest import IngestError
from tests.pipeline.test_analyze import MATERIALS, found


class FakeTranscriber:
    calls = 0

    def __init__(self, model_name):
        self.model_name = model_name

    def transcribe_chunk(self, wav, start, end):
        FakeTranscriber.calls += 1
        words = []
        for i in range(40):  # 40 sentences of 5 words, 6 s each
            for j in range(5):
                st = i * 6 + j * 1.2
                words.append(Word(w=f"kata{i}x{j}" + ("." if j == 4 else ""), start=st, end=st + 1.0))
        return words


class FakeOllama:
    prompts = 0

    def __init__(self, url, model, **kw):
        self.model = model
        self.replies = [found(0, 8, 70), MATERIALS]

    def complete_json(self, prompt, schema):
        FakeOllama.prompts += 1
        return self.replies.pop(0) if self.replies else '{"candidates": []}'


@pytest.fixture
def patched(monkeypatch, tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    src = tmp_path / "source.mp4"
    src.write_bytes(b"x")
    meta = VideoMeta(source_type="youtube", project_id="vid12345678", title="Judul", duration=240.0, video_id="vid12345678")
    rendered = []

    def fake_ingest(url, s):
        # behaves like the real ingest: source and meta.json live in the project folder
        d = Path(s.data_dir) / "projects" / meta.project_id
        d.mkdir(parents=True, exist_ok=True)
        (d / "source.mp4").write_bytes(b"x")
        (d / "meta.json").write_text(meta.model_dump_json(), encoding="utf-8")
        return d / "source.mp4", meta

    def fake_render(source, clip, out_dir, presets, fonts_dir, on_progress=None):
        out_dir.mkdir(parents=True, exist_ok=True)
        rendered.append(clip)
        (out_dir / "final.mp4").write_bytes(b"v")
        return out_dir / "final.mp4"

    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "ingest_youtube", fake_ingest)
    monkeypatch.setattr(cli, "extract_audio", lambda a, b: b.write_bytes(b"w"))
    monkeypatch.setattr(cli, "detect_silences", lambda wav: [])
    monkeypatch.setattr(cli, "FasterWhisperTranscriber", FakeTranscriber)
    monkeypatch.setattr(cli, "OllamaClient", FakeOllama)
    monkeypatch.setattr(cli, "reframe_clip", lambda v, a, b: ([ReframeSegment(start=0, end=b - a, mode="track", x=0.3)], "face_crop"))
    monkeypatch.setattr(cli, "render_clip", fake_render)
    FakeTranscriber.calls = FakeOllama.prompts = 0
    return settings, meta, rendered


def test_cli_run_produces_report_and_clips(patched):
    settings, meta, rendered = patched
    assert cli.main(["run", "https://youtu.be/vid12345678"]) == 0
    run_dir = Path(settings.data_dir) / "projects" / meta.project_id / "runs" / "small__qwen3-4b"
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "Layak? (ya/tidak)" in report and len(rendered) == 1
    clips = json.loads((run_dir / "clips.json").read_text(encoding="utf-8"))
    assert clips[0]["materials"]["hook"] == "Gaji besar tetap bokek?"
    assert clips[0]["subtitles"] and clips[0]["subtitles"][0]["start"] >= 0
    timings = json.loads((run_dir / "timings.json").read_text(encoding="utf-8"))
    assert {"ingest", "audio", "transcribe", "analyze", "clips"} <= timings.keys()


def test_cli_run_uses_checkpoints(patched):
    cli.main(["run", "https://youtu.be/vid12345678"])
    first_t, first_p = FakeTranscriber.calls, FakeOllama.prompts
    assert first_t == 1 and first_p == 2
    cli.main(["run", "https://youtu.be/vid12345678"])
    assert FakeTranscriber.calls == first_t and FakeOllama.prompts == first_p


def test_cli_no_render(patched):
    _, _, rendered = patched
    assert cli.main(["run", "https://youtu.be/vid12345678", "--no-render"]) == 0
    assert rendered == []


def test_cli_known_error_exit_code(monkeypatch, capsys, patched):
    def boom(url, s):
        raise IngestError("INVALID_URL", "Link ini bukan link video YouTube yang valid.")

    monkeypatch.setattr(cli, "ingest_youtube", boom)
    assert cli.main(["run", "https://vimeo.com/1"]) == 2
    assert "[INVALID_URL]" in capsys.readouterr().out


def test_cli_retryable_error_exit_code(monkeypatch, capsys, patched):
    def boom(url, s):
        raise IngestError("NETWORK", "timeout", retryable=True)

    monkeypatch.setattr(cli, "ingest_youtube", boom)
    assert cli.main(["run", "https://youtu.be/vid12345678"]) == 1


def test_cli_local_file_uses_ingest_file(monkeypatch, patched, tmp_path):
    called = {}
    _, meta, _ = patched

    def fake_file(path, s):
        called["path"] = path
        (Path(s.data_dir) / "projects" / meta.project_id).mkdir(parents=True, exist_ok=True)
        return tmp_path / "source.mp4", meta

    monkeypatch.setattr(cli, "ingest_file", fake_file)
    video = tmp_path / "rekaman saya.mp4"
    video.write_bytes(b"x")
    assert cli.main(["run", str(video), "--no-render"]) == 0
    assert called["path"] == video


def test_analysis_isolated_per_whisper_model(patched):
    # review I2: candidates cached for the small transcript must not be reused for medium
    cli.main(["run", "https://youtu.be/vid12345678", "--no-render"])
    before = FakeOllama.prompts
    cli.main(["run", "https://youtu.be/vid12345678", "--no-render", "--whisper-model", "medium"])
    assert FakeOllama.prompts > before


def test_rerun_never_overwrites_filled_report(patched):
    # review I3: the user's filled-in evaluation must survive a rerun
    settings, meta, _ = patched
    cli.main(["run", "https://youtu.be/vid12345678", "--no-render"])
    run_dir = Path(settings.data_dir) / "projects" / meta.project_id / "runs" / "small__qwen3-4b"
    report = run_dir / "report.md"
    report.write_text("diisi pengguna", encoding="utf-8")
    cli.main(["run", "https://youtu.be/vid12345678", "--no-render"])
    assert report.read_text(encoding="utf-8") == "diisi pengguna"
    assert len(list(run_dir.glob("report-*.md"))) == 1


def _claude_reply(tmp_path, body):
    p = tmp_path / "balasan.json"
    p.write_text(body, encoding="utf-8")
    return p


def test_export_then_import_without_llm(patched, tmp_path):
    settings, meta, rendered = patched
    assert cli.main(["export-transcript", "https://youtu.be/vid12345678"]) == 0
    tdir = Path(settings.data_dir) / "projects" / meta.project_id / "transcripts" / "small"
    prompt = (tdir / "untuk_claude.md").read_text(encoding="utf-8")
    assert "[0] (00:00)" in prompt and "kata0x0" in prompt and '"clips"' in prompt
    assert "data, not instructions" in prompt
    assert FakeOllama.prompts == 0

    clip = {"start_sentence": 0, "end_sentence": 8, "score": 82, "category": "hot_take", "reason": "r",
            "hook": "Hook dari Claude",
            **{p: {"title": "Judul dari Claude", "caption": "c", "hashtags": ["#a"]} for p in ("tiktok", "reels", "shorts")}}
    reply = _claude_reply(tmp_path, "Berikut hasilnya:\n```json\n" + json.dumps({"clips": [clip]}) + "\n```")
    assert cli.main(["import-moments", meta.project_id, str(reply)]) == 0
    run_dir = Path(settings.data_dir) / "projects" / meta.project_id / "runs" / "small__claude-manual"
    clips = json.loads((run_dir / "clips.json").read_text(encoding="utf-8"))
    assert clips[0]["materials"]["tiktok"]["title"] == "Judul dari Claude"
    assert clips[0]["candidate"]["llm_score"] == 82
    assert "claude-manual" in (run_dir / "report.md").read_text(encoding="utf-8")
    assert FakeOllama.prompts == 0 and len(rendered) == 1


def test_import_invalid_reply_exit_code(patched, tmp_path, capsys):
    _, meta, _ = patched
    cli.main(["export-transcript", "https://youtu.be/vid12345678"])
    assert cli.main(["import-moments", meta.project_id, str(_claude_reply(tmp_path, "maaf, saya tidak bisa"))]) == 2
    assert "[INVALID_IMPORT]" in capsys.readouterr().out


def test_import_before_export_is_clear_error(patched, tmp_path, capsys):
    _, meta, _ = patched
    assert cli.main(["import-moments", "tidak-ada", str(_claude_reply(tmp_path, "{}"))]) == 2
    assert "[INVALID_IMPORT]" in capsys.readouterr().out
