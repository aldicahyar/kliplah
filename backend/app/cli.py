"""Kliplah command line: a thin client over the same services, database and worker as the API (spec 10.6).

Every command writes to `DATA_DIR/kliplah.db` and runs its jobs synchronously through the worker
(`worker_main(once=True)` until the project's queue is empty). `--queue-only` only enqueues.

    python -m app.cli probe <youtube-url> [--type ...] [--source-language KODE]
    python -m app.cli run <youtube-url | video-file> [--source-language KODE] [--target-language KODE] [--moments ollama|claude] [--no-render]
    python -m app.cli analyze <project_id> --llm-model gemma3:4b
    python -m app.cli bench-whisper <youtube-url | video-file> --models small medium --seconds 300
    python -m app.cli migrate-m1
    python -m app.cli recover          # only when no worker is running

Without Ollama, using Claude through your own subscription (no API cost):
    python -m app.cli export-transcript <youtube-url | video-file>        # writes untuk_claude*.md
    python -m app.cli export-moments <project_id>                         # export again, same project
    python -m app.cli import-moments <project_id> <reply.json> [--part K]   # uses Claude's picks
    python -m app.cli export-translation <project_id> [--scope clips|all] # writes terjemahan_claude*.md
    python -m app.cli import-translation <project_id> <reply.json> [--part K]

Jobs run here claim only this project's jobs and never reset another worker's `running` job: while
`scripts/dev.py` is working on the project, the CLI waits for it.
"""
import argparse
import logging
import shutil
import sys
import time
import uuid
from pathlib import Path

from sqlalchemy import Engine
from sqlmodel import Session, col, select

from app import migrate_m1
from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.core.fs import project_dir
from app.db.engine import get_engine, init_db, session_scope
from app.db.tables import Clip, Job, Project, utcnow
from app.domain.ranges import parse_range
from app.pipeline.audio import extract_audio
from app.pipeline.ingest import IngestError, ingest_file, ingest_youtube
from app.pipeline.youtube_subs import TRANSCRIPT_SOURCES
from app.providers.transcriber import FasterWhisperTranscriber
from app.services import clips as clip_service
from app.services import moments, projects, translation
from app.services._common import fetch
from app.services.settings_store import get_app_settings, put_app_settings
from app.worker import queue
from app.worker.main import recover_startup, worker_main
from app.worker.process import STAGES, ProcessDeps, transcript_dir

log = logging.getLogger("kliplah")

# exit code 2: the user can fix the input; everything else (retryable or runtime) exits 1
USER_ERRORS = frozenset({"INVALID_URL", "INVALID_IMPORT", "VALIDATION", "NOT_FOUND", "INVALID_STATE",
                         "VIDEO_UNAVAILABLE", "DOWNLOAD_BLOCKED", "DISK_FULL", "AGE_RESTRICTED"})
PAUSE_STATUSES = ("awaiting_moments", "ready")
MOMENT_SOURCES = {"ollama": "ollama", "claude": "claude_manual"}
WAIT_POLL = 2.0  # seconds between status checks while another worker runs the project's job
WAITING = ("Job sedang dikerjakan worker lain; menunggu… (jika worker itu sudah mati, hentikan perintah ini "
           "lalu jalankan `python -m app.cli recover`)")
LLM_MODEL_HELP = ("model Ollama; hanya berlaku untuk job yang dijalankan CLI ini sendiri (tidak disimpan di project)")


def _process_deps(settings: Settings) -> ProcessDeps:
    """Pipeline of process jobs run by the CLI; the single seam tests replace."""
    return ProcessDeps(settings=settings)


def _sleep(seconds: float) -> None:
    time.sleep(seconds)


def _engine(settings: Settings) -> Engine:
    engine = get_engine(settings)
    init_db(engine)
    return engine


def _is_url(source: str) -> bool:
    if Path(source).is_file():
        return False
    return source.lower().startswith(("http://", "https://", "www.", "youtu"))


def _with_models(settings: Settings, args) -> Settings:
    update = {}
    if getattr(args, "whisper_model", None):
        update["whisper_model"] = args.whisper_model
    if getattr(args, "llm_model", None):
        update["ollama_model"] = args.llm_model
    return settings.model_copy(update=update) if update else settings


# ---------- running jobs synchronously ----------

def _failed_job_error(session: Session, project_id: str) -> AppError:
    job = session.exec(select(Job).where(Job.project_id == project_id, Job.status == "failed")
                       .order_by(col(Job.finished_at).desc())).first()
    if job is None:
        return AppError("UNKNOWN", f"Project {project_id} gagal diproses.")
    return AppError(job.error_code or "UNKNOWN", job.error_message or "Project gagal diproses.")


def _drain(engine: Engine, settings: Settings, project_id: str, *, stop_at_pause: bool = True) -> Project:
    """Run the project's jobs until it has no queued/running job (or pauses at awaiting_moments/ready).

    Only this project's jobs are claimed and nothing is recovered. A job `running` while the CLI is between
    jobs belongs to another worker: the CLI waits for it instead. A failed project raises its job's error.
    """
    deps = _process_deps(settings)
    noted = False
    while True:
        with Session(engine) as s:
            project = fetch(s, Project, project_id, "Project")
            if project.status == "failed":
                raise _failed_job_error(s, project_id)
            active = s.exec(select(Job).where(Job.project_id == project_id,
                                              col(Job.status).in_(queue.ACTIVE_STATUSES))).all()
            s.expunge(project)
        if not active or (stop_at_pause and project.status in PAUSE_STATUSES):
            return project
        if any(j.status == "running" for j in active):
            if not noted:
                print(WAITING, flush=True)
                noted = True
            _sleep(WAIT_POLL)
            continue
        worker_main(settings, once=True, sleep=_sleep, process_deps=deps, recover=False, project_id=project_id)


def _print_export(engine: Engine, settings: Settings, project: Project) -> list[Path]:
    parts = moments.export_parts(engine, project.id, settings)
    tdir = transcript_dir(settings, project)
    paths = [tdir / p["filename"] for p in parts]
    print(f"Project ID: {project.id}")
    print("Berikan isi file berikut ke Claude (Claude Code atau claude.ai), satu per satu:")
    for p in paths:
        print(f"  {p}")
    print("Simpan setiap balasan sebagai .json, lalu:")
    for p in parts:
        part = f" --part {p['part']}" if p["of"] > 1 else ""
        print(f"  uv run python -m app.cli import-moments {project.id} <file-balasan.json>{part}")
    return paths


def _render_all(engine: Engine, settings: Settings, project_id: str) -> list[Path]:
    """Approve every proposed clip (M1 behaviour), render, and return the output files."""
    proposed = [c.id for c in clip_service.list_clips(engine, project_id) if c.status == "proposed"]
    clip_service.bulk(engine, proposed, "approve")
    _drain(engine, settings, project_id, stop_at_pause=False)
    rows = clip_service.list_clips(engine, project_id)
    failed = [c for c in rows if c.id in proposed and c.status == "render_failed"]
    outputs = [Path(c.output_path) for c in rows if c.status == "rendered" and c.output_path]
    for path in outputs:
        print(f"  {path}")
    if failed:
        with Session(engine) as s:
            job = s.exec(select(Job).where(Job.clip_id == failed[0].id, Job.status == "failed")
                         .order_by(col(Job.finished_at).desc())).first()
        code = job.error_code if job and job.error_code else "UNKNOWN"
        msg = job.error_message if job and job.error_message else "Render gagal."
        raise AppError(code, f"{len(failed)} klip gagal dirender. {msg}")
    return outputs


def _finish(engine: Engine, settings: Settings, project: Project, render: bool) -> Path:
    """After the process job: export for Claude when awaiting moments, otherwise list (and render) the clips."""
    if project.status == "awaiting_moments":
        _print_export(engine, settings, project)
        return project_dir(settings, project.id)
    rows = clip_service.list_clips(engine, project.id)
    print(f"{len(rows)} klip untuk project {project.id}:")
    for c in rows:
        print(f"  #{c.rank} skor {c.score} [{c.status}] {c.hook_text}")
    if render:
        _render_all(engine, settings, project.id)
    return project_dir(settings, project.id)


# ---------- commands ----------

def _stage_upload(source: str, settings: Settings) -> Path:
    """Copy a local video next to the uploads so `create_project` can move the copy, never the user's file."""
    path = Path(source)
    if not path.is_file():
        raise AppError("INVALID_URL", f"File tidak ditemukan: {path}")
    tmp = Path(settings.data_dir) / "uploads" / "_incoming" / uuid.uuid4().hex / path.name
    tmp.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, tmp)
    return tmp


def _create(engine: Engine, settings: Settings, source: str, **options) -> Project:
    if _is_url(source):
        project = projects.create_project(engine, url=source, settings=settings, **options)
    else:
        tmp = _stage_upload(source, settings)
        try:
            project = projects.create_project(engine, upload_path=tmp, original_filename=Path(source).name,
                                              settings=settings, **options)
        finally:
            shutil.rmtree(tmp.parent, ignore_errors=True)
    print(f"Project ID: {project.id}")
    return project


def _parse_ranges(raw: list[str] | None) -> list[list[float]] | None:
    if not raw:
        return None
    return [[s, e] for s, e in (parse_range(r) for r in raw)]


def _read_reply_file(path: str) -> str:
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as e:
        raise AppError("INVALID_IMPORT", f"File balasan tidak bisa dibaca: {e}") from e
    except UnicodeDecodeError as e:
        raise AppError("INVALID_IMPORT", f"File balasan harus memakai encoding UTF-8: {e}") from e


def cmd_run(args, settings: Settings) -> Path:
    settings = _with_models(settings, args)
    engine = _engine(settings)
    project = _create(engine, settings, args.source, content_type=args.type,
                      ranges=_parse_ranges(args.range), focus_prompt=args.focus,
                      source_language=args.source_language, target_language=args.target_language,
                      transcript_source=args.transcript_source,
                      moment_source=MOMENT_SOURCES[args.moments] if args.moments else None,
                      max_clips=args.max_clips, min_clip_sec=args.min_sec, max_clip_sec=args.max_sec)
    if args.queue_only:
        print("Diantrekan; worker (scripts/dev.py) akan memprosesnya.")
        return project_dir(settings, project.id)
    project = _drain(engine, settings, project.id)
    return _finish(engine, settings, project, render=not args.no_render)


def cmd_export(args, settings: Settings) -> Path:
    settings = _with_models(settings, args)
    engine = _engine(settings)
    project = _create(engine, settings, args.source, content_type=args.type,
                      ranges=_parse_ranges(args.range), focus_prompt=args.focus,
                      source_language=args.source_language, target_language=args.target_language,
                      transcript_source=args.transcript_source,
                      moment_source="claude_manual", max_clips=args.max_clips, min_clip_sec=args.min_sec,
                      max_clip_sec=args.max_sec)
    if args.queue_only:
        print("Diantrekan; setelah project menunggu momen, jalankan:")
        print(f"  uv run python -m app.cli export-moments {project.id}")
        return project_dir(settings, project.id)
    project = _drain(engine, settings, project.id)
    if project.status != "awaiting_moments":
        raise AppError("INVALID_STATE", f"Project berstatus '{project.status}', bukan menunggu momen.")
    return _print_export(engine, settings, project)[0]


def cmd_export_moments(args, settings: Settings) -> Path:
    """Write the Claude prompt parts of an existing project again (no download, no transcription)."""
    engine = _engine(settings)
    project, _ = projects.get_project(engine, args.project_id)
    return _print_export(engine, settings, project)[0]


def cmd_import(args, settings: Settings) -> Path:
    engine = _engine(settings)
    reply = _read_reply_file(args.reply)
    status = moments.import_part(engine, args.project_id, args.part, reply, settings)
    if status["missing"]:
        missing = ", ".join(map(str, status["missing"]))
        print(f"Bagian {args.part} tersimpan. Masih menunggu balasan untuk bagian {missing}.")
        return project_dir(settings, args.project_id)
    moments.finish_moments(engine, args.project_id, settings=settings)
    if args.queue_only:
        print("Diantrekan; worker (scripts/dev.py) akan menyiapkan klip.")
        return project_dir(settings, args.project_id)
    project = _drain(engine, settings, args.project_id)
    return _finish(engine, settings, project, render=not args.no_render)


def cmd_export_translation(args, settings: Settings) -> Path:
    """Export untranslated sentences of a project for Claude manual translation."""
    engine = _engine(settings)
    project, _ = projects.get_project(engine, args.project_id)
    parts = translation.export_translation(engine, project.id, scope=args.scope, settings=settings)
    export_dir = transcript_dir(settings, project) / "translation_export"
    print(f"Project ID: {project.id}")
    if not parts:
        print("Semua kalimat pada cakupan ini sudah diterjemahkan.")
        return export_dir
    paths = [export_dir / p["filename"] for p in parts]
    print("Berikan isi file berikut ke Claude (Claude Code atau claude.ai), satu per satu:")
    for p in paths:
        print(f"  {p}")
    print("Simpan setiap balasan sebagai .json, lalu:")
    for p in parts:
        part = f" --part {p['part']}" if p["of"] > 1 else ""
        print(f"  uv run python -m app.cli import-translation {project.id} <file-balasan.json>{part}")
    return export_dir


def cmd_import_translation(args, settings: Settings) -> Path:
    """Import one manual translation reply part from Claude and refresh unedited clip subtitles."""
    engine = _engine(settings)
    reply = _read_reply_file(args.reply)
    res = translation.import_translation(engine, args.project_id, args.part, reply, settings=settings)
    print(
        f"Terjemahan tersimpan: {res['merged']} kalimat digabung, "
        f"{len(res['refreshed'])} klip diperbarui, {len(res['skipped'])} klip dilewati."
    )
    return project_dir(settings, args.project_id)


def cmd_analyze(args, settings: Settings) -> Path:
    """Choose moments again with Ollama, reusing the project's transcript."""
    settings = _with_models(settings, args)
    engine = _engine(settings)
    with session_scope(engine) as s:
        project = fetch(s, Project, args.project_id, "Project")
        if s.exec(select(Job).where(Job.project_id == project.id,
                                    col(Job.status).in_(queue.ACTIVE_STATUSES))).first() is not None:
            raise AppError("INVALID_STATE", "Project sedang diproses; tunggu sampai job selesai.")
        done = project.last_completed_stage
        if done is not None and STAGES.index(done) > STAGES.index("transcribe"):
            project.last_completed_stage = "transcribe"
        project.moment_source = "ollama"
        project.status = "queued"
        project.updated_at = utcnow()
        s.add(project)
        queue.enqueue(s, "process", project.id)
    if args.queue_only:
        print("Diantrekan; worker (scripts/dev.py) akan menganalisis ulang.")
        return project_dir(settings, args.project_id)
    project = _drain(engine, settings, args.project_id)
    return _finish(engine, settings, project, render=not args.no_render)


def _ingest(source: str, settings: Settings):
    if _is_url(source):
        return ingest_youtube(source, settings)
    return ingest_file(Path(source), settings)


def cmd_bench_whisper(args, settings: Settings) -> Path:
    """Compare Whisper models on the first seconds of a video (no database)."""
    src, meta = _ingest(args.source, settings)
    d = project_dir(settings, meta.project_id)
    wav = d / "audio.wav"
    extract_audio(src, wav)
    seconds = min(args.seconds, meta.duration)
    out = [f"# Perbandingan Whisper: {meta.title}", "", f"Sampel: {seconds:.0f} detik pertama", ""]
    for m in args.models:
        print(f"  whisper {m} ...", flush=True)
        t0 = time.monotonic()
        words = FasterWhisperTranscriber(m).transcribe_chunk(wav, 0.0, seconds)
        took = time.monotonic() - t0
        out += [f"## `{m}`: {took:.0f} detik ({took / seconds:.2f}x durasi audio)", "",
                " ".join(w.w for w in words), ""]
    path = d / "bench_whisper.md"
    path.write_text("\n".join(out), encoding="utf-8")
    return path


def cmd_recover(args, settings: Settings) -> Path:
    engine = _engine(settings)
    with Session(engine) as s:
        jobs, clips = recover_startup(s)
    print(f"{jobs} job dan {clips} klip tertinggal dikembalikan ke antrean.")
    return Path(settings.data_dir)


def cmd_migrate(args, settings: Settings) -> Path:
    result = migrate_m1.run(settings)
    print(f"Migrasi M1: {result['projects']} project dan {result['clips']} klip baru, "
          f"{result['skipped']} dilewati karena sudah ada.")
    return Path(settings.data_dir)


def cmd_probe(args, settings: Settings) -> Path:
    engine = _engine(settings)
    info = projects.probe(
        engine,
        args.url,
        ranges=_parse_ranges(args.range),
        source_language=args.source_language,
        target_language=args.target_language,
        content_type=args.type,
        moment_source=MOMENT_SOURCES[args.moments] if args.moments else None,
        max_clips=args.max_clips,
        settings=settings,
    )
    subs = info["subtitles"]
    print(f"Judul: {info['title']} ({info['channel'] or '-'})")
    print(f"Durasi: {info['duration']:.0f} detik | Heatmap: {'ya' if info['has_heatmap'] else 'tidak'} | "
          f"Live chat: {'ya' if info['has_live_chat'] else 'tidak'}")
    print(f"Subtitle manual: {', '.join(subs.get('manual') or []) or '-'} | "
          f"auto: {', '.join(subs.get('auto') or []) or '-'}")
    plan = info["subtitle_plan"]
    print(f"Sumber transkrip otomatis: {info['transcript_source_auto']} "
          f"(bahasa: {plan['language'] or 'deteksi'}, hint: {info['source_language_hint'] or '-'}, "
          f"terjemahan: {info['translation_engine']})")
    est = info["estimate"]
    b = est["breakdown"]
    print(f"Perkiraan: ~{est['process_minutes']} menit proses "
          f"(unduh {b['download_min']}m, encode {b['encode_min']}m, transkrip {b['transcribe_min']}m, "
          f"chat {b['chat_min']}m, terjemah {b['translate_min']}m, LLM {b['llm_min']}m), ~{est['disk_gb']} GB disk")
    for w in info["warnings"]:
        print(f"Peringatan: {w}")
    return Path(settings.data_dir)


def cmd_config(args, settings: Settings) -> Path:
    engine = _engine(settings)
    if args.action == "get":
        with session_scope(engine) as s:
            vals = get_app_settings(s, settings=settings)
        if args.key:
            if args.key not in vals:
                raise AppError("VALIDATION", f"Pengaturan tidak dikenal: {args.key}.", status=422)
            print(f"{args.key}={vals[args.key]}")
        else:
            for k, v in vals.items():
                print(f"{k}={v}")
        return Path(settings.data_dir)
    if args.value is None:
        raise AppError("VALIDATION", "Nilai pengaturan wajib diisi.", status=422)
    val: str | int = args.value
    if args.key in ("max_clips", "encode_threads", "range_max_height"):
        try:
            val = int(args.value)
        except ValueError:
            raise AppError("VALIDATION", f"Nilai '{args.key}' harus berupa bilangan bulat.", status=422) from None
    with session_scope(engine) as s:
        updated = put_app_settings(s, {args.key: val}, settings=settings)
    print(f"{args.key}={updated[args.key]}")
    return Path(settings.data_dir)


# ---------- entry point ----------

def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m app.cli", description="Kliplah: video panjang ke klip vertikal")
    sub = p.add_subparsers(dest="command", required=True)

    def jobs(sp):
        sp.add_argument("--no-render", action="store_true", help="jangan setujui dan render klip otomatis")
        sp.add_argument("--queue-only", action="store_true", help="hanya antrekan job, worker yang memproses")

    def new_project(sp):
        sp.add_argument("source", help="link YouTube atau file video")
        sp.add_argument("--type", default=None, help="jenis konten (podcast, stream, variety, news, general)")
        sp.add_argument("--range", action="append", default=None, metavar="HH:MM:SS-HH:MM:SS",
                        help="rentang waktu yang diproses (boleh diulang)")
        sp.add_argument("--focus", default=None, help="arahan momen yang dicari")
        sp.add_argument("--source-language", default=None, metavar="KODE",
                        help="kode bahasa video sumber (kosong/auto = deteksi otomatis)")
        sp.add_argument("--target-language", default="id", metavar="KODE",
                        help="kode bahasa target hasil subtitle (default: id)")
        sp.add_argument("--transcript-source", choices=TRANSCRIPT_SOURCES, default="auto",
                        help="sumber transkrip (auto, youtube_manual, youtube_auto, whisper)")
        sp.add_argument("--max-clips", type=int, default=None)
        sp.add_argument("--min-sec", type=int, default=None)
        sp.add_argument("--max-sec", type=int, default=None)
        sp.add_argument("--whisper-model", default=None)
        jobs(sp)

    run = sub.add_parser("run", help="proses link YouTube atau file video")
    new_project(run)
    run.add_argument("--moments", choices=sorted(MOMENT_SOURCES), default=None,
                     help="pemilih momen (default: pengaturan aplikasi)")
    run.add_argument("--llm-model", default=None, help=LLM_MODEL_HELP)
    an = sub.add_parser("analyze", help="pilih momen ulang dengan Ollama memakai transkrip yang ada")
    an.add_argument("project_id")
    an.add_argument("--llm-model", default=None, help=LLM_MODEL_HELP)
    jobs(an)
    ex = sub.add_parser("export-transcript", help="transkrip + instruksi untuk dipilihkan Claude (tanpa biaya API)")
    new_project(ex)
    em = sub.add_parser("export-moments", help="tulis ulang file untuk Claude dari project yang sudah ada")
    em.add_argument("project_id")
    im = sub.add_parser("import-moments", help="pakai balasan JSON dari Claude untuk membuat klip")
    im.add_argument("project_id")
    im.add_argument("reply", help="file berisi balasan Claude")
    im.add_argument("--part", type=int, default=1, help="nomor bagian transkrip (default 1)")
    jobs(im)
    et = sub.add_parser("export-translation",
                        help="ekspor kalimat belum diterjemahkan untuk Claude (tanpa biaya API)")
    et.add_argument("project_id")
    et.add_argument("--scope", choices=("clips", "all"), default="clips",
                    help="cakupan kalimat yang diekspor (default: clips)")
    it = sub.add_parser("import-translation",
                        help="pakai balasan JSON terjemahan dari Claude untuk subtitle klip")
    it.add_argument("project_id")
    it.add_argument("reply", help="file berisi balasan terjemahan Claude")
    it.add_argument("--part", type=int, default=1, help="nomor bagian terjemahan (default 1)")
    pr = sub.add_parser("probe", help="cek metadata, subtitle, live chat, dan perkiraan sumber daya link YouTube")
    pr.add_argument("url", help="link YouTube")
    pr.add_argument("--range", action="append", default=None, metavar="HH:MM:SS-HH:MM:SS",
                    help="rentang waktu untuk menghitung perkiraan (boleh diulang)")
    pr.add_argument("--type", default=None, help="jenis konten (podcast, stream, variety, news, general)")
    pr.add_argument("--source-language", default=None, metavar="KODE", help="kode bahasa video sumber (opsional)")
    pr.add_argument("--target-language", default="id", metavar="KODE",
                    help="kode bahasa target hasil subtitle (default: id)")
    pr.add_argument("--moments", choices=sorted(MOMENT_SOURCES), default=None,
                    help="pemilih momen (default: pengaturan aplikasi)")
    pr.add_argument("--max-clips", type=int, default=None, help="jumlah klip maksimal untuk perkiraan LLM")
    cfg = sub.add_parser("config", help="lihat atau ubah pengaturan aplikasi (mis. cookies_browser)")
    cfg.add_argument("action", choices=("get", "set"))
    cfg.add_argument("key", nargs="?", default=None)
    cfg.add_argument("value", nargs="?", default=None)
    bw = sub.add_parser("bench-whisper", help="bandingkan model Whisper")
    bw.add_argument("source")
    bw.add_argument("--models", nargs="+", default=["small", "medium"])
    bw.add_argument("--seconds", type=float, default=300.0)
    sub.add_parser("migrate-m1", help="pindahkan hasil CLI M1 ke database (sekali, aman diulang)")
    sub.add_parser("recover", help="kembalikan job/klip yang tertinggal worker yang mati ke antrean; "
                                   "HANYA saat tidak ada worker (scripts/dev.py) yang berjalan")
    return p


HANDLERS = {"run": cmd_run, "analyze": cmd_analyze, "bench-whisper": cmd_bench_whisper,
            "export-transcript": cmd_export, "import-moments": cmd_import, "export-moments": cmd_export_moments,
            "export-translation": cmd_export_translation, "import-translation": cmd_import_translation,
            "probe": cmd_probe, "config": cmd_config, "migrate-m1": cmd_migrate, "recover": cmd_recover}


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parser().parse_args(argv)
    try:
        out = HANDLERS[args.command](args, get_settings())
    except AppError as e:
        print(f"[{e.code}] {e.message}")
        return 2 if e.code in USER_ERRORS else 1
    except IngestError as e:  # bench-whisper calls ingest directly
        print(f"[{e.code}] {e}")
        return 1 if e.retryable else 2
    print(f"Selesai: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
