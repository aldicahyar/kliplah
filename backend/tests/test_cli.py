"""CLI as a thin client: pipeline mocked through `ProcessDeps` (cli._process_deps) and the worker's render seam."""
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from sqlmodel import Session, select

from app import cli
from app.core.config import Settings
from app.core.models import ReframeSegment, VideoMeta, Word
from app.db.engine import get_engine
from app.db.tables import Clip, Job, Project
from app.pipeline.analyze import analyze
from app.pipeline.ingest import IngestError
from app.pipeline.transcribe import transcribe
from app.services import settings_store
from app.worker import main as worker_main_mod
from app.worker import queue
from app.worker.prepare import PrepareDeps, build_clips_from_picks
from app.worker.process import ProcessDeps, run_process_job
from app.worker.render_job import run_render_job
from tests.pipeline.test_analyze import MATERIALS, found

URL = "https://youtu.be/vid12345678"


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


class Pipeline:
    """Fake media around the real transcribe/analyze/prepare logic (checkpoints included)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.ingested: list[Project] = []
        self.transcribed: list[tuple[str, Path]] = []
        self.llm_models: list[str] = []
        self.rendered: list = []
        self.ingest_error: Exception | None = None

    def ingest(self, project, settings):
        if self.ingest_error is not None:
            raise self.ingest_error
        self.ingested.append(project)
        d = Path(settings.data_dir) / "projects" / project.id
        d.mkdir(parents=True, exist_ok=True)
        (d / "source.mp4").write_bytes(b"x")
        return d / "source.mp4", VideoMeta(source_type=project.source_type, project_id=project.id, title="Judul",
                                           duration=240.0, video_id="vid12345678")

    def transcribe(self, wav, duration, tdir, model, corrections, on_progress=None):
        self.transcribed.append((model, tdir))
        return transcribe(wav, duration, tdir, FakeTranscriber(model), model, corrections, on_progress=on_progress)

    def analyze(self, t, meta, max_clips, min_sec, max_sec, checkpoint_dir, settings):
        self.llm_models.append(settings.ollama_model)
        llm = FakeOllama(settings.ollama_url, settings.ollama_model)
        return analyze(t, meta, llm, max_clips, min_sec, max_sec, checkpoint_dir)

    def prepare(self, engine, project_id, picks):
        deps = PrepareDeps(settings=self.settings, thumbnail=lambda src, t, out: None, proxy=lambda src, out: None,
                           reframe=lambda v, a, b: ([ReframeSegment(start=0, end=b - a, mode="track", x=0.3)],
                                                    "face_crop"))
        build_clips_from_picks(engine, project_id, picks, deps)

    def render(self, source, plan, out_dir, presets, fonts_dir, on_progress=None):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.rendered.append(plan)
        (out_dir / "final.mp4").write_bytes(b"v")
        return out_dir / "final.mp4"

    def deps(self, settings: Settings) -> ProcessDeps:
        return ProcessDeps(settings=settings, ingest=self.ingest, extract_audio=lambda a, b: b.write_bytes(b"w"),
                           detect_silences=lambda wav: [], transcribe=self.transcribe, analyze=self.analyze,
                           prepare=self.prepare)


@pytest.fixture
def env(monkeypatch, tmp_path):
    settings = Settings(_env_file=None, data_dir=tmp_path / "data")
    pipe = Pipeline(settings)
    monkeypatch.setattr(cli, "get_settings", lambda: settings)
    monkeypatch.setattr(settings_store, "get_settings", lambda: settings)
    monkeypatch.setattr(cli, "_process_deps", pipe.deps)
    monkeypatch.setattr(worker_main_mod, "run_render_job",
                        lambda eng, job_id, *, settings: run_render_job(eng, job_id, pipe.render, settings=settings))
    FakeTranscriber.calls = FakeOllama.prompts = 0
    return settings, pipe


def _db(settings):
    return get_engine(settings)


def _projects(settings) -> list[Project]:
    with Session(_db(settings)) as s:
        return list(s.exec(select(Project).order_by(Project.created_at)).all())


def _clips(settings, project_id) -> list[Clip]:
    with Session(_db(settings)) as s:
        return list(s.exec(select(Clip).where(Clip.project_id == project_id).order_by(Clip.rank)).all())


def _jobs(settings) -> list[Job]:
    with Session(_db(settings)) as s:
        return list(s.exec(select(Job)).all())


def test_cli_run_renders_all_clips(env, capsys):
    settings, pipe = env
    assert cli.main(["run", URL, "--moments", "ollama"]) == 0
    [project] = _projects(settings)
    assert project.status == "ready" and project.moment_source == "ollama"
    clips = _clips(settings, project.id)
    assert len(clips) == 1 and len(pipe.rendered) == 1
    c = clips[0]
    assert c.status == "rendered" and Path(c.output_path).is_file()
    assert c.hook_text == "Gaji besar tetap bokek?" and c.subtitles and c.subtitles[0]["start"] >= 0
    assert all(j.status == "succeeded" for j in _jobs(settings))
    out = capsys.readouterr().out
    assert f"Project ID: {project.id}" in out and c.output_path in out
    assert FakeTranscriber.calls == 1 and FakeOllama.prompts == 2


def test_cli_no_render(env):
    settings, pipe = env
    assert cli.main(["run", URL, "--moments", "ollama", "--no-render"]) == 0
    [project] = _projects(settings)
    assert project.status == "ready" and pipe.rendered == []
    assert [c.status for c in _clips(settings, project.id)] == ["proposed"]
    assert [j.type for j in _jobs(settings)] == ["process"]


def test_cli_run_options_reach_project(env):
    settings, _ = env
    assert cli.main(["run", URL, "--moments", "ollama", "--no-render", "--type", "podcast", "--max-clips", "3",
                     "--min-sec", "25", "--max-sec", "70"]) == 0
    [p] = _projects(settings)
    assert (p.content_type, p.max_clips, p.min_clip_sec, p.max_clip_sec) == ("podcast", 3, 25, 70)


def test_cli_queue_only_does_not_process(env, capsys):
    settings, pipe = env
    assert cli.main(["run", URL, "--queue-only"]) == 0
    [project] = _projects(settings)
    assert project.status == "queued" and pipe.ingested == []
    assert [(j.type, j.status) for j in _jobs(settings)] == [("process", "queued")]
    assert project.id in capsys.readouterr().out


def test_cli_run_with_claude_moments_stops_for_export(env, capsys):
    settings, pipe = env
    assert cli.main(["run", URL, "--moments", "claude"]) == 0
    [project] = _projects(settings)
    assert project.status == "awaiting_moments" and pipe.llm_models == []
    out = capsys.readouterr().out
    assert "untuk_claude.md" in out and f"import-moments {project.id}" in out


def test_analyze_reuses_transcript_with_another_llm(env):
    # the transcript checkpoint is kept: only the moments stage runs again, with the requested model
    settings, pipe = env
    cli.main(["run", URL, "--moments", "ollama", "--no-render"])
    [project] = _projects(settings)
    assert cli.main(["analyze", project.id, "--llm-model", "gemma3:4b", "--no-render"]) == 0
    assert FakeTranscriber.calls == 1 and len(pipe.ingested) == 1
    assert pipe.llm_models == ["qwen3:4b", "gemma3:4b"]
    [p] = _projects(settings)
    assert p.status == "ready" and p.moment_source == "ollama"


def test_analyze_renders_unless_no_render(env):
    settings, pipe = env
    cli.main(["run", URL, "--moments", "claude"])
    [project] = _projects(settings)
    assert cli.main(["analyze", project.id, "--llm-model", "gemma3:4b"]) == 0
    assert [c.status for c in _clips(settings, project.id)] == ["rendered"] and len(pipe.rendered) == 1


def test_analyze_unknown_project_exit_code(env, capsys):
    assert cli.main(["analyze", "tidak-ada", "--llm-model", "gemma3:4b"]) == 2
    assert "[NOT_FOUND]" in capsys.readouterr().out


def test_cli_known_error_exit_code(env, capsys):
    assert cli.main(["run", "https://vimeo.com/1"]) == 2
    assert "[INVALID_URL]" in capsys.readouterr().out


def test_cli_job_user_error_exit_code(env, capsys):
    settings, pipe = env
    pipe.ingest_error = IngestError("VIDEO_UNAVAILABLE", "Video ini privat, dihapus, atau dibatasi umur.")
    assert cli.main(["run", URL, "--moments", "ollama"]) == 2
    assert "[VIDEO_UNAVAILABLE] Video ini privat" in capsys.readouterr().out
    assert _projects(settings)[0].status == "failed"


def test_cli_retryable_error_exit_code(env, monkeypatch, capsys):
    settings, pipe = env
    pipe.ingest_error = IngestError("NETWORK", "timeout", retryable=True)
    clock = {"t": datetime(2026, 1, 1, tzinfo=timezone.utc)}
    monkeypatch.setattr(queue, "now", lambda: clock["t"])
    waits: list[float] = []

    def fake_sleep(seconds):  # the worker's backoff wait advances the fake clock instead of blocking
        waits.append(seconds)
        clock["t"] += timedelta(seconds=seconds)

    monkeypatch.setattr(cli, "_sleep", fake_sleep)
    assert cli.main(["run", URL, "--moments", "ollama"]) == 1
    assert "[NETWORK] timeout" in capsys.readouterr().out
    [job] = _jobs(settings)
    assert job.status == "failed" and job.attempts == queue.MAX_ATTEMPTS and waits


def test_cli_local_file_is_copied_not_moved(env, tmp_path):
    settings, pipe = env
    video = tmp_path / "rekaman saya.mp4"
    video.write_bytes(b"x")
    assert cli.main(["run", str(video), "--moments", "ollama", "--no-render"]) == 0
    [p] = _projects(settings)
    assert (p.source_type, p.original_filename) == ("upload", "rekaman saya.mp4")
    assert [x.id for x in pipe.ingested] == [p.id]
    assert video.is_file()  # the user's own file stays where it was
    assert (Path(settings.data_dir) / "uploads" / p.id / "rekaman saya.mp4").is_file()


def test_cli_local_file_starting_with_youtu_is_treated_as_file(env, tmp_path, monkeypatch):
    settings, _ = env
    monkeypatch.chdir(tmp_path)
    Path("youtu_clip.mp4").write_bytes(b"x")
    assert cli.main(["run", "youtu_clip.mp4", "--queue-only"]) == 0
    [p] = _projects(settings)
    assert (p.source_type, p.original_filename) == ("upload", "youtu_clip.mp4")


def test_cli_uses_app_settings_content_type_when_type_omitted(env):
    from app.db.engine import init_db, session_scope

    settings, _ = env
    eng = get_engine(settings)
    init_db(eng)
    with session_scope(eng) as s:
        settings_store.put_app_settings(s, {"content_type": "stream"}, settings=settings)
    assert cli.main(["run", URL, "--queue-only"]) == 0
    [p] = _projects(settings)
    assert (p.content_type, p.min_clip_sec, p.max_clip_sec) == ("stream", 15, 35)


def test_cli_missing_local_file_exit_code(env, tmp_path, capsys):
    assert cli.main(["run", str(tmp_path / "tidak-ada.mp4")]) == 2
    assert "[INVALID_URL]" in capsys.readouterr().out


def test_analysis_isolated_per_whisper_model(env):
    # review I2: the medium transcript (and its analysis checkpoints) live apart from the small one
    settings, pipe = env
    assert cli.main(["run", URL, "--moments", "ollama", "--no-render", "--whisper-model", "medium"]) == 0
    [p] = _projects(settings)
    assert p.whisper_model == "medium"
    assert pipe.transcribed == [("medium", Path(settings.data_dir) / "projects" / p.id / "transcripts" / "medium")]


def test_rerun_creates_new_project_and_keeps_the_first(env):
    # review I3 (v2 form): a rerun never touches what the user already has
    settings, _ = env
    cli.main(["run", URL, "--moments", "ollama", "--no-render"])
    [first] = _projects(settings)
    before = [(c.id, c.status) for c in _clips(settings, first.id)]
    cli.main(["run", URL, "--moments", "ollama", "--no-render"])
    projects = _projects(settings)
    assert len(projects) == 2 and projects[0].id == first.id
    assert [(c.id, c.status) for c in _clips(settings, first.id)] == before


def _claude_reply(tmp_path, body):
    p = tmp_path / "balasan.json"
    p.write_text(body, encoding="utf-8")
    return p


def _clip_row(score=82):
    return {"start_sentence": 0, "end_sentence": 8, "score": score, "category": "hot_take", "reason": "r",
            "hook": "Hook dari Claude",
            **{p: {"title": "Judul dari Claude", "caption": "c", "hashtags": ["#a"]}
               for p in ("tiktok", "reels", "shorts")}}


def test_cli_export_then_import_runs_to_ready(env, tmp_path, capsys):
    settings, pipe = env
    assert cli.main(["export-transcript", URL]) == 0
    [project] = _projects(settings)
    assert project.status == "awaiting_moments" and project.moment_source == "claude_manual"
    prompt_file = Path(settings.data_dir) / "projects" / project.id / "transcripts" / "small" / "untuk_claude.md"
    prompt = prompt_file.read_text(encoding="utf-8")
    assert "[0] (00:00)" in prompt and "kata0x0" in prompt and '"clips"' in prompt
    assert "data, not instructions" in prompt
    out = capsys.readouterr().out
    assert str(prompt_file) in out and f"Project ID: {project.id}" in out
    assert f"import-moments {project.id}" in out
    assert FakeOllama.prompts == 0

    reply = _claude_reply(tmp_path, "Berikut hasilnya:\n```json\n" + json.dumps({"clips": [_clip_row()]}) + "\n```")
    assert cli.main(["import-moments", project.id, str(reply)]) == 0
    [project] = _projects(settings)
    assert project.status == "ready"
    [clip] = _clips(settings, project.id)
    assert clip.captions["tiktok"]["title"] == "Judul dari Claude" and clip.llm_score == 82
    assert clip.status == "rendered" and clip.output_path in capsys.readouterr().out
    assert FakeOllama.prompts == 0 and len(pipe.rendered) == 1 and FakeTranscriber.calls == 1


def test_import_no_render_stops_at_ready(env, tmp_path):
    settings, pipe = env
    cli.main(["export-transcript", URL])
    [project] = _projects(settings)
    reply = _claude_reply(tmp_path, json.dumps({"clips": [_clip_row()]}))
    assert cli.main(["import-moments", project.id, str(reply), "--part", "1", "--no-render"]) == 0
    assert [c.status for c in _clips(settings, project.id)] == ["proposed"] and pipe.rendered == []


def test_import_invalid_reply_exit_code(env, tmp_path, capsys):
    settings, _ = env
    cli.main(["export-transcript", URL])
    [project] = _projects(settings)
    assert cli.main(["import-moments", project.id, str(_claude_reply(tmp_path, "maaf, saya tidak bisa"))]) == 2
    assert "[INVALID_IMPORT]" in capsys.readouterr().out


def test_import_missing_reply_file_exit_code(env, tmp_path, capsys):
    settings, _ = env
    cli.main(["export-transcript", URL])
    [project] = _projects(settings)
    assert cli.main(["import-moments", project.id, str(tmp_path / "tidak-ada.json")]) == 2
    assert "[INVALID_IMPORT]" in capsys.readouterr().out


def test_import_before_export_is_clear_error(env, tmp_path, capsys):
    assert cli.main(["import-moments", "tidak-ada", str(_claude_reply(tmp_path, "{}"))]) == 2
    assert "[NOT_FOUND]" in capsys.readouterr().out


def test_bench_whisper_writes_comparison_without_db(env, monkeypatch, tmp_path):
    settings, _ = env
    meta = VideoMeta(source_type="youtube", project_id="vid12345678", title="Judul", duration=240.0)

    def fake_ingest(url, s):
        d = Path(s.data_dir) / "projects" / meta.project_id
        d.mkdir(parents=True, exist_ok=True)
        return d / "source.mp4", meta

    monkeypatch.setattr(cli, "ingest_youtube", fake_ingest)
    monkeypatch.setattr(cli, "extract_audio", lambda a, b: b.write_bytes(b"w"))
    monkeypatch.setattr(cli, "FasterWhisperTranscriber", FakeTranscriber)
    assert cli.main(["bench-whisper", URL, "--models", "small", "medium", "--seconds", "60"]) == 0
    text = (Path(settings.data_dir) / "projects" / "vid12345678" / "bench_whisper.md").read_text(encoding="utf-8")
    assert "## `small`" in text and "## `medium`" in text and "kata0x0" in text
    assert FakeTranscriber.calls == 2
    assert not (Path(settings.data_dir) / "kliplah.db").exists()


def test_bench_whisper_ingest_error_exit_code(env, monkeypatch, capsys):
    def boom(url, s):
        raise IngestError("NETWORK", "timeout", retryable=True)

    monkeypatch.setattr(cli, "ingest_youtube", boom)
    assert cli.main(["bench-whisper", URL]) == 1
    assert "[NETWORK]" in capsys.readouterr().out


# ---------- coexisting with another worker (review I1) ----------

def _claim(settings, project_id) -> str:
    """Claim the project's queued job as another worker process would."""
    with Session(_db(settings)) as s:
        return queue.claim_next(s, project_id).id


def test_cli_does_not_recover_running_jobs(env):
    settings, pipe = env
    cli.main(["run", URL, "--queue-only"])
    [other] = _projects(settings)
    other_job = _claim(settings, other.id)  # being worked on by scripts/dev.py's worker
    assert cli.main(["run", URL, "--moments", "ollama", "--no-render"]) == 0
    job = next(j for j in _jobs(settings) if j.id == other_job)
    assert job.status == "running" and job.attempts == 1
    assert [p.id for p in pipe.ingested] == [p.id for p in _projects(settings) if p.id != other.id]


def test_cli_does_not_claim_other_projects_jobs(env):
    settings, pipe = env
    cli.main(["run", URL, "--queue-only"])
    [other] = _projects(settings)
    assert cli.main(["run", URL, "--moments", "ollama", "--no-render"]) == 0
    assert next(p for p in _projects(settings) if p.id == other.id).status == "queued"
    assert other.id not in [p.id for p in pipe.ingested]


def test_cli_waits_for_job_running_elsewhere(env, monkeypatch, capsys):
    settings, pipe = env
    cli.main(["run", URL, "--moments", "ollama", "--queue-only"])
    [project] = _projects(settings)
    job_id = _claim(settings, project.id)
    waits = []

    def other_worker_finishes(seconds):
        waits.append(seconds)
        if len(waits) == 2:  # the other worker completes the job while the CLI waits
            run_process_job(_db(settings), job_id, pipe.deps(settings))

    monkeypatch.setattr(cli, "_sleep", other_worker_finishes)
    capsys.readouterr()
    result = cli._drain(_db(settings), settings, project.id)
    assert result.status == "ready" and waits == [cli.WAIT_POLL, cli.WAIT_POLL] and cli.WAIT_POLL == 2.0
    out = capsys.readouterr().out
    assert out.count("Job sedang dikerjakan worker lain") == 1 and "recover" in out
    assert len(pipe.ingested) == 1  # run once, by the other worker only


def test_cli_recover_command(env, capsys):
    settings, _ = env
    cli.main(["run", URL, "--queue-only"])
    [project] = _projects(settings)
    job_id = _claim(settings, project.id)
    assert cli.main(["recover"]) == 0
    assert "1 job" in capsys.readouterr().out
    assert next(j for j in _jobs(settings) if j.id == job_id).status == "queued"


def test_llm_model_help_says_it_is_not_saved(capsys):
    with pytest.raises(SystemExit):
        cli.main(["analyze", "--help"])
    assert "tidak disimpan" in capsys.readouterr().out


# ---------- exporting an existing project (review I2) ----------

def test_export_moments_existing_project(env, capsys):
    settings, _ = env
    cli.main(["export-transcript", URL])
    [project] = _projects(settings)
    capsys.readouterr()
    assert cli.main(["export-moments", project.id]) == 0
    out = capsys.readouterr().out
    tdir = Path(settings.data_dir) / "projects" / project.id / "transcripts" / "small"
    assert str(tdir / "untuk_claude.md") in out and f"import-moments {project.id}" in out
    assert len(_projects(settings)) == 1 and FakeTranscriber.calls == 1


def test_export_transcript_queue_only_points_to_export_moments(env, capsys):
    settings, pipe = env
    assert cli.main(["export-transcript", URL, "--queue-only"]) == 0
    [project] = _projects(settings)
    assert f"export-moments {project.id}" in capsys.readouterr().out and pipe.ingested == []


def test_export_moments_unknown_project_exit_code(env, capsys):
    assert cli.main(["export-moments", "tidak-ada"]) == 2
    assert "[NOT_FOUND]" in capsys.readouterr().out
