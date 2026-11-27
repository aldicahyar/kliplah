"""The staged `process` job: ingest -> audio -> transcribe -> moments -> prepare.

Stage logic is lifted from the M1 CLI (`prepare_project` / `process_project`) and keeps its checkpoints:
`projects/<id>/meta.json`, `silences.json`, `transcripts/<model>/` (transcript + analysis checkpoints).
A run resumes after `Project.last_completed_stage` and yields between stages when a render is waiting.
"""
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from sqlalchemy import Engine
from sqlmodel import Session

from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.fs import atomic_write_json, project_dir, read_json
from app.core.models import Candidate, Materials, Transcript, VideoMeta
from app.db.tables import Job, Project, utcnow
from app.domain.profiles import profile_for
from app.pipeline.analyze import analyze
from app.pipeline.audio import detect_silences, extract_audio
from app.pipeline.corrections import load_corrections
from app.pipeline.ingest import IngestError, ingest_file, ingest_youtube
from app.pipeline.render import RenderError
from app.pipeline.transcribe import transcribe
from app.providers.llm import OllamaClient
from app.providers.transcriber import FasterWhisperTranscriber
from app.worker import queue

log = logging.getLogger("kliplah.process")

STAGES = ("ingest", "audio", "transcribe", "moments", "prepare")
Pick = tuple[Candidate, Materials]
Outcome = Literal["done", "paused", "yield"]


def save_picks(path: Path, picks: list[Pick]) -> None:
    atomic_write_json(path, [{"candidate": c.model_dump(), "materials": m.model_dump()} for c, m in picks])


def load_picks(path: Path) -> list[Pick]:
    raw = read_json(path)
    if raw is None:
        raise AppError("INVALID_STATE", "Daftar momen (picks.json) tidak ditemukan untuk project ini.")
    return [(Candidate.model_validate(p["candidate"]), Materials.model_validate(p["materials"])) for p in raw]


def _safe(name: str) -> str:
    return name.replace(":", "-").replace("/", "-")


def transcript_dir(settings: Settings, project: Project) -> Path:
    """`projects/<id>/transcripts/<model>/`: transcript and analysis checkpoints of the project's whisper model."""
    return Path(settings.data_dir) / "projects" / project.id / "transcripts" / _safe(project.whisper_model)


def default_ingest(project: Project, settings: Settings) -> tuple[Path, VideoMeta]:
    """YouTube link, or the upload stored at `DATA_DIR/uploads/<project_id>/<original_filename>`."""
    if project.source_type == "upload":
        path = Path(settings.data_dir) / "uploads" / project.id / (project.original_filename or "")
        return ingest_file(path, settings, project_id=project.id)
    return ingest_youtube(project.youtube_url or "", settings, project_id=project.id)


def default_transcribe(wav: Path, duration: float, tdir: Path, whisper_model: str,
                       corrections: list[tuple[str, str]],
                       on_progress: Callable[[float], None] | None = None) -> Transcript:
    return transcribe(wav, duration, tdir, FasterWhisperTranscriber(whisper_model), whisper_model, corrections,
                      on_progress=on_progress)


def default_analyze(t: Transcript, meta: VideoMeta, max_clips: int, min_sec: int, max_sec: int,
                    checkpoint_dir: Path, settings: Settings) -> list[Pick]:
    llm = OllamaClient(settings.ollama_url, settings.ollama_model)
    return analyze(t, meta, llm, max_clips, min_sec, max_sec, checkpoint_dir)


def default_prepare(engine: Engine, project_id: str, picks: list[Pick],
                    settings: Settings | None = None) -> None:
    from app.worker.prepare import prepare_project_clips  # prepare imports this module

    if settings is None:
        prepare_project_clips(engine, project_id, picks)
    else:
        prepare_project_clips(engine, project_id, picks, settings=settings)


@dataclass
class ProcessDeps:
    """Pipeline functions used by the job; every one can be replaced in tests."""
    settings: Settings = field(default_factory=get_settings)
    ingest: Callable[[Project, Settings], tuple[Path, VideoMeta]] = default_ingest
    extract_audio: Callable[[Path, Path], None] = extract_audio
    detect_silences: Callable[[Path], list[tuple[float, float]]] = detect_silences
    transcribe: Callable[..., Transcript] = default_transcribe
    analyze: Callable[..., list[Pick]] = default_analyze
    prepare: Callable[[Engine, str, list[Pick]], None] = default_prepare


def to_app_error(e: Exception) -> AppError:
    if isinstance(e, AppError):
        return e
    if isinstance(e, (IngestError, RenderError)):
        return AppError(e.code, str(e), retryable=getattr(e, "retryable", False))
    return AppError("UNKNOWN", f"Terjadi kesalahan tak terduga saat memproses video ({type(e).__name__}: {e}).")


class _Run:
    """One pass over the remaining stages of a project. State not produced in this pass is read from disk."""

    def __init__(self, session: Session, engine: Engine, job_id: str, project: Project, deps: ProcessDeps):
        self.session, self.engine, self.job_id, self.project, self.deps = session, engine, job_id, project, deps
        self.settings = deps.settings
        self.dir = project_dir(self.settings, project.id)
        self.tdir = transcript_dir(self.settings, project)
        self._meta: VideoMeta | None = None
        self._transcript: Transcript | None = None

    @property
    def meta(self) -> VideoMeta:
        if self._meta is None:
            raw = read_json(self.dir / "meta.json")
            if raw is None:
                raise AppError("INVALID_STATE", "Data video (meta.json) tidak ditemukan; proses ulang project ini.")
            self._meta = VideoMeta.model_validate(raw)
        return self._meta

    @property
    def transcript(self) -> Transcript:
        if self._transcript is None:
            raw = read_json(self.tdir / "transcript.json")
            if raw is None:
                raise AppError("INVALID_STATE", "Transkrip tidak ditemukan; proses ulang project ini.")
            self._transcript = Transcript.model_validate(raw)
        return self._transcript

    def save_project(self, **fields) -> None:
        for k, v in fields.items():
            setattr(self.project, k, v)
        self.project.updated_at = utcnow()
        self.session.add(self.project)
        self.session.commit()

    def stage_ingest(self) -> None:
        _, meta = self.deps.ingest(self.project, self.settings)
        atomic_write_json(self.dir / "meta.json", meta.model_dump())
        self._meta = meta
        self.save_project(title=meta.title, channel=meta.channel, duration_sec=meta.duration,
                          video_id=meta.video_id, has_heatmap=bool(meta.heatmap))

    def stage_audio(self) -> None:
        wav = self.dir / "audio.wav"
        self.deps.extract_audio(self.dir / "source.mp4", wav)
        if read_json(self.dir / "silences.json") is None:
            atomic_write_json(self.dir / "silences.json", self.deps.detect_silences(wav))

    def stage_transcribe(self) -> None:
        self.tdir.mkdir(parents=True, exist_ok=True)
        corrections = load_corrections(Path(self.settings.data_dir) / "corrections.json")
        self._transcript = self.deps.transcribe(
            self.dir / "audio.wav", self.meta.duration, self.tdir, self.project.whisper_model, corrections,
            on_progress=lambda p: queue.set_progress(self.session, self.job_id, "transcribe", p))

    def stage_moments(self) -> None:
        p = self.project
        profile = profile_for(p.content_type)
        # analysis checkpoints live next to the transcript they index into (sentence numbers differ per model)
        picks = self.deps.analyze(self.transcript, self.meta, p.max_clips, p.min_clip_sec or profile.min_sec,
                                  p.max_clip_sec or profile.max_sec, self.tdir / "checkpoints", self.settings)
        save_picks(self.dir / "picks.json", picks)

    def stage_prepare(self) -> None:
        picks = load_picks(self.dir / "picks.json")
        if self.deps.prepare is default_prepare:
            default_prepare(self.engine, self.project.id, picks, settings=self.settings)
        else:
            self.deps.prepare(self.engine, self.project.id, picks)

    def run(self) -> Outcome:
        done = self.project.last_completed_stage
        start = STAGES.index(done) + 1 if done else 0
        for i in range(start, len(STAGES)):
            stage = STAGES[i]
            queue.set_progress(self.session, self.job_id, stage, 0.0)
            if stage == "moments" and self.project.moment_source != "ollama":
                if self.project.moment_source != "claude_manual":
                    raise AppError("VALIDATION", f"Sumber momen '{self.project.moment_source}' tidak dikenal.")
                self.save_project(status="awaiting_moments")  # picks arrive via import (Task 7)
                queue.finish_job(self.session, self.job_id)
                return "paused"
            getattr(self, f"stage_{stage}")()
            self.save_project(last_completed_stage=stage)
            if i < len(STAGES) - 1 and queue.has_queued_render(self.session):
                return "yield"
        self.save_project(status="ready")
        queue.finish_job(self.session, self.job_id)
        return "done"


def run_process_job(engine: Engine, job_id: str, deps: ProcessDeps) -> Outcome:
    """Run the remaining stages of a claimed process job.

    "done": the job's status is final (succeeded, failed, or queued again for a retry after an error);
    "paused": waiting for Claude-manual moments (job succeeded, project awaiting_moments);
    "yield": a render is queued; the job is left `running` for the worker loop to requeue.
    """
    with Session(engine, expire_on_commit=False) as session:
        job = session.get(Job, job_id)
        if job is None or job.project_id is None:
            raise AppError("NOT_FOUND", f"Job {job_id} tidak ditemukan", status=404)
        project_id = job.project_id
        try:
            project = session.get(Project, project_id)
            if project is None:
                raise AppError("NOT_FOUND", f"Project {project_id} tidak ditemukan", status=404)
            run = _Run(session, engine, job_id, project, deps)
            run.save_project(status="processing")
            return run.run()
        except Exception as e:
            log.exception("process job %s failed", job_id)
            session.rollback()
            queue.fail_job(session, job_id, to_app_error(e))
            project = session.get(Project, project_id, populate_existing=True)
            if project is not None and session.get(Job, job_id, populate_existing=True).status == "failed":
                project.status = "failed"
                project.updated_at = utcnow()
                session.add(project)
                session.commit()
            return "done"
