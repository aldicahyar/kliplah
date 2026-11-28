"""The staged `process` job: ingest -> audio -> transcribe -> signals -> moments -> prepare.

Stage logic is lifted from the M1 CLI (`prepare_project` / `process_project`) and keeps its checkpoints:
`projects/<id>/meta.json`, `silences.json`, `signals.json`, `transcripts/<model>/` (transcript + analysis checkpoints).
A project downloaded as time ranges has one `source_r<i>.mp4` / `audio_r<i>.wav` / `transcripts/<model>/r<i>/`
per range; `silences.json`, `signals.json` and the merged `transcript.json` are on the original video timeline.
A run resumes after `Project.last_completed_stage` and yields between stages when a render is waiting.
"""
import inspect
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
from app.core.models import Candidate, Materials, Transcript, VideoMeta, Word
from app.db.engine import get_engine, session_scope
from app.db.tables import Job, Project, utcnow
from app.domain.profiles import profile_for
from app.domain.ranges import RangeFile, SourceMap
from app.pipeline.analyze import analyze
from app.pipeline.audio import detect_silences, extract_audio
from app.pipeline.audio_peaks import audio_loudness, audio_peaks
from app.pipeline.corrections import apply_corrections, load_corrections
from app.pipeline.ingest import IngestError, ingest_file, ingest_youtube, probe_youtube
from app.pipeline.live_chat import chat_peaks, download_live_chat, load_chat_keywords, parse_live_chat
from app.pipeline.render import RenderError
from app.pipeline.sentences import PAUSE_SPLIT_SEC
from app.pipeline.signals import Signals, load_signals, save_signals
from app.pipeline.transcribe import merge_range_transcripts, transcribe
from app.pipeline.youtube_subs import (
    SUBTITLE_PAUSE_SPLIT_SEC,
    clip_words_to_ranges,
    download_subs,
    parse_json3,
    resolve_transcript_source,
)
from app.providers.llm import OllamaClient
from app.providers.transcriber import FasterWhisperTranscriber
from app.services.settings_store import get_app_settings
from app.worker import queue

log = logging.getLogger("kliplah.process")

STAGES = ("ingest", "audio", "transcribe", "signals", "moments", "prepare")
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
    """`projects/<id>/transcripts/<source>/`: transcript and analysis checkpoints for the project's source."""
    src = project.transcript_source
    if src == "youtube_manual":
        sub = f"youtube-manual-{_safe(project.target_language)}"
    elif src == "youtube_auto":
        sub = f"youtube-auto-{_safe(project.target_language)}"
    else:  # "auto" (S1 projects) and "whisper" share the Whisper folder
        sub = _safe(project.whisper_model)
    return Path(settings.data_dir) / "projects" / project.id / "transcripts" / sub


def _app_settings(settings: Settings) -> dict:
    with session_scope(get_engine(settings)) as session:
        return get_app_settings(session, settings=settings)


def _cookies_browser(settings: Settings) -> str:
    return _app_settings(settings)["cookies_browser"]


def default_probe(project: Project, settings: Settings) -> VideoMeta | None:
    """Probe YouTube metadata without downloading."""
    if project.source_type != "youtube":
        return None
    return probe_youtube(project.youtube_url or "", settings, cookies_browser=_cookies_browser(settings))


def default_ingest(project: Project, settings: Settings) -> tuple[Path, VideoMeta]:
    """YouTube link, or the upload stored at `DATA_DIR/uploads/<project_id>/<original_filename>`.

    Passes the project's time ranges and app settings (`cookies_browser`, `encode_threads`, `range_max_height`).
    """
    if project.source_type == "upload":
        path = Path(settings.data_dir) / "uploads" / project.id / (project.original_filename or "")
        return ingest_file(path, settings, project_id=project.id, ranges=project.ranges)
    app = _app_settings(settings)
    return ingest_youtube(project.youtube_url or "", settings, project_id=project.id, ranges=project.ranges,
                          cookies_browser=app["cookies_browser"], encode_threads=app["encode_threads"],
                          range_max_height=app["range_max_height"])


def default_download_subs(video_id: str, lang: str, kind: Literal["manual", "auto"],
                          out: Path, settings: Settings) -> Path:
    return download_subs(video_id, lang, kind, out, settings, cookies_browser=_cookies_browser(settings))


def default_transcribe(wav: Path, duration: float, tdir: Path, whisper_model: str,
                       corrections: list[tuple[str, str]],
                       on_progress: Callable[[float], None] | None = None) -> Transcript:
    return transcribe(wav, duration, tdir, FasterWhisperTranscriber(whisper_model), whisper_model, corrections,
                      on_progress=on_progress)


def default_signals(project: Project, meta: VideoMeta, smap: SourceMap, settings: Settings) -> Signals:
    """Collect profile-relevant signals (live chat and audio peaks) on the original video timeline."""
    profile = profile_for(project.content_type)
    pdir = project_dir(settings, project.id)
    chat_list: list[dict] = []
    vid = meta.video_id or project.video_id
    if "live_chat" in profile.signals and meta.has_live_chat and vid:
        try:
            until_sec = smap.files[-1].end if meta.ranges and smap.files else None
            chat_path = download_live_chat(vid, pdir / "live_chat.json", settings,
                                           cookies_browser=_cookies_browser(settings), until_sec=until_sec)
            if chat_path is not None:
                msgs = parse_live_chat(chat_path)
                chat_list = chat_peaks(msgs, meta.duration, load_chat_keywords())
        except Exception as e:
            log.warning("Gagal mengambil live chat untuk project %s: %s", project.id, e)

    audio_list: list[dict] = []
    if "audio_peak" in profile.signals:
        try:
            for rf in smap.files:
                wav = pdir / (f"audio_r{rf.index}.wav" if meta.ranges else "audio.wav")
                if wav.exists():
                    audio_list.extend(audio_peaks(audio_loudness(wav), offset=rf.start))
        except Exception as e:
            log.warning("Gagal mengambil puncak suara untuk project %s: %s", project.id, e)
            audio_list = []
    return Signals(chat=chat_list, audio=audio_list)


def default_analyze(t: Transcript, meta: VideoMeta, max_clips: int, min_sec: int, max_sec: int,
                    checkpoint_dir: Path, settings: Settings, *, signals: Signals | None = None,
                    content_type: str = "general", focus_prompt: str | None = None,
                    source_map: SourceMap | None = None) -> list[Pick]:
    llm = OllamaClient(settings.ollama_url, settings.ollama_model)
    return analyze(t, meta, llm, max_clips, min_sec, max_sec, checkpoint_dir, signals=signals,
                   content_type=content_type, focus_prompt=focus_prompt, source_map=source_map)


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
    probe: Callable[[Project, Settings], VideoMeta | None] | None = None
    ingest: Callable[[Project, Settings], tuple[Path, VideoMeta]] = default_ingest
    extract_audio: Callable[[Path, Path], None] = extract_audio
    detect_silences: Callable[[Path], list[tuple[float, float]]] = detect_silences
    transcribe: Callable[..., Transcript] = default_transcribe
    download_subs: Callable[..., Path] = default_download_subs
    signals: Callable[[Project, VideoMeta, SourceMap, Settings], Signals] = default_signals
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
        self._meta: VideoMeta | None = None
        self._transcript: Transcript | None = None

    @property
    def tdir(self) -> Path:
        return transcript_dir(self.settings, self.project)

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
        if (self.project.source_type == "youtube"
                and self.project.transcript_source in ("youtube_manual", "youtube_auto")):
            probe_fn = self.deps.probe or (default_probe if self.deps.ingest is default_ingest else None)
            if probe_fn is not None:
                probed = probe_fn(self.project, self.settings)
                if probed is not None:
                    resolve_transcript_source(self.project.transcript_source, probed, self.project.target_language)
        _, meta = self.deps.ingest(self.project, self.settings)
        source = resolve_transcript_source(self.project.transcript_source, meta, self.project.target_language)
        meta = meta.model_copy(update={"transcript_source": source})
        atomic_write_json(self.dir / "meta.json", meta.model_dump())
        self._meta = meta
        self.save_project(title=meta.title, channel=meta.channel, duration_sec=meta.duration,
                          video_id=meta.video_id, has_heatmap=bool(meta.heatmap), transcript_source=source)

    @property
    def ranged(self) -> bool:
        return bool(self.meta.ranges)

    def wav(self, rf: RangeFile) -> Path:
        """`audio_r<i>.wav` per range file; `audio.wav` for a single-source (S1) project."""
        return self.dir / (f"audio_r{rf.index}.wav" if self.ranged else "audio.wav")

    def stage_audio(self) -> None:
        smap = SourceMap.from_meta(self.meta, self.dir)
        for rf in smap.files:
            self.deps.extract_audio(smap.path(rf), self.wav(rf))
        if read_json(self.dir / "silences.json") is None:
            if not self.ranged:  # S1 layout, values unchanged
                atomic_write_json(self.dir / "silences.json", self.deps.detect_silences(self.dir / "audio.wav"))
                return
            silences: list[tuple[float, float]] = []
            for rf in smap.files:
                silences.extend((smap.to_original(rf, a), smap.to_original(rf, b))
                                for a, b in self.deps.detect_silences(self.wav(rf)))
            atomic_write_json(self.dir / "silences.json", sorted(silences))

    def _transcribe_youtube_subs(self, corrections: list[tuple[str, str]]) -> Transcript:
        cached = read_json(self.tdir / "transcript.json")
        if cached is not None:
            return Transcript.model_validate(cached)
        lang = self.project.target_language
        kind: Literal["manual", "auto"] = "manual" if self.project.transcript_source == "youtube_manual" else "auto"
        vid = self.meta.video_id or self.project.video_id
        if not vid:
            raise AppError("INVALID_STATE", "Video ID YouTube tidak ditemukan.")
        sub_path = self.deps.download_subs(vid, lang, kind, self.dir / "subs", self.settings)
        raw = read_json(sub_path)
        if not isinstance(raw, dict):
            raise AppError("INVALID_STATE", f"File subtitle {sub_path.name} tidak valid atau tidak ditemukan.")
        words = apply_corrections(parse_json3(raw), corrections)
        files = (
            SourceMap.from_meta(self.meta, self.dir).files
            if self.ranged
            else [RangeFile(0, 0.0, round(max(self.meta.duration, words[-1].end if words else 0.0), 3), "source.mp4")]
        )
        parts = clip_words_to_ranges(words, files)
        duration = self.meta.duration or (words[-1].end if words else 0.0)
        t = merge_range_transcripts(parts, SUBTITLE_PAUSE_SPLIT_SEC, lang, duration)
        atomic_write_json(self.tdir / "transcript.json", t.model_dump())
        queue.set_progress(self.session, self.job_id, "transcribe", 1.0)
        return t

    def stage_transcribe(self) -> None:
        self.tdir.mkdir(parents=True, exist_ok=True)
        corrections = load_corrections(Path(self.settings.data_dir) / "corrections.json")
        if self.project.transcript_source in ("youtube_manual", "youtube_auto"):
            self._transcript = self._transcribe_youtube_subs(corrections)
            return
        model = self.project.whisper_model
        if not self.ranged:
            self._transcript = self.deps.transcribe(
                self.dir / "audio.wav", self.meta.duration, self.tdir, model, corrections,
                on_progress=lambda p: queue.set_progress(self.session, self.job_id, "transcribe", p))
            return
        files = SourceMap.from_meta(self.meta, self.dir).files
        total = sum(rf.end - rf.start for rf in files)
        done = 0.0
        parts: list[tuple[RangeFile, list[Word]]] = []
        for rf in files:
            length = round(rf.end - rf.start, 3)
            out_dir = self.tdir / f"r{rf.index}"
            out_dir.mkdir(parents=True, exist_ok=True)

            def progress(p: float, base: float = done, length: float = length) -> None:
                queue.set_progress(self.session, self.job_id, "transcribe", round((base + p * length) / total, 4))

            part = self.deps.transcribe(self.wav(rf), length, out_dir, model, corrections, on_progress=progress)
            parts.append((rf, part.words))
            done += length
        self._transcript = merge_range_transcripts(parts, PAUSE_SPLIT_SEC, "id", self.meta.duration)
        atomic_write_json(self.tdir / "transcript.json", self._transcript.model_dump())

    def stage_signals(self) -> None:
        smap = SourceMap.from_meta(self.meta, self.dir)
        sig_path = self.dir / "signals.json"
        if read_json(sig_path) is not None:
            sigs = load_signals(sig_path)
        else:
            sigs = self.deps.signals(self.project, self.meta, smap, self.settings)
            save_signals(sig_path, sigs)
        avail = {
            "heatmap": bool(self.meta.heatmap),
            "live_chat": bool(sigs.chat or (self.dir / "live_chat.json").exists()),
            "audio_peak": bool(sigs.audio),
        }
        self.save_project(signals_available=avail)

    def stage_moments(self) -> None:
        p = self.project
        profile = profile_for(p.content_type)
        sigs = load_signals(self.dir / "signals.json")
        smap = SourceMap.from_meta(self.meta, self.dir, self.transcript)
        min_sec = p.min_clip_sec or profile.min_sec
        max_sec = p.max_clip_sec or profile.max_sec
        ck = self.tdir / "checkpoints"
        params = inspect.signature(self.deps.analyze).parameters
        accepts_kw = "signals" in params or any(v.kind == inspect.Parameter.VAR_KEYWORD for v in params.values())
        if accepts_kw:
            picks = self.deps.analyze(self.transcript, self.meta, p.max_clips, min_sec, max_sec, ck, self.settings,
                                      signals=sigs, content_type=p.content_type, focus_prompt=p.focus_prompt,
                                      source_map=smap)
        else:
            picks = self.deps.analyze(self.transcript, self.meta, p.max_clips, min_sec, max_sec, ck, self.settings)
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
