"""Kliplah command line (M1): run the whole pipeline, re-analyze with another LLM, or benchmark Whisper.

    python -m app.cli run <youtube-url | video-file> [--no-render]
    python -m app.cli analyze <project_id> --llm-model gemma3:4b
    python -m app.cli bench-whisper <youtube-url | video-file> --models small medium --seconds 300

Without Ollama, using Claude through your own subscription (no API cost):
    python -m app.cli export-transcript <youtube-url | video-file>   # writes untuk_claude.md
    python -m app.cli import-moments <project_id> <reply.json>         # renders Claude's picks
"""
import argparse
import logging
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from app.core.config import BACKEND_DIR, Settings, get_settings
from app.core.fs import atomic_write_json, project_dir, read_json
from app.core.models import ClipPlan, Transcript, VideoMeta
from app.pipeline.analyze import analyze, import_external, transcript_lines
from app.pipeline.audio import detect_silences, extract_audio
from app.pipeline.corrections import load_corrections
from app.pipeline.ingest import IngestError, ingest_file, ingest_youtube
from app.pipeline.llm_json import parse_llm_json, render_prompt
from app.pipeline.render import RenderError, render_clip
from app.pipeline.reframe import reframe_clip
from app.pipeline.report import build_report, clip_folder
from app.pipeline.snap import neighbor_bounds, snap_edges
from app.pipeline.subtitles import group_lines, load_presets
from app.pipeline.transcribe import transcribe
from app.providers.llm import OllamaClient
from app.providers.transcriber import FasterWhisperTranscriber

log = logging.getLogger("kliplah")
FONTS_DIR = BACKEND_DIR / "assets" / "fonts"
SNAP_SLACK = 3.0  # edge refinement may stretch a clip slightly past max_sec


@dataclass
class RunOptions:
    max_clips: int = 8
    min_sec: int = 30
    max_sec: int = 90
    whisper_model: str = "small"
    llm_model: str = "qwen3:4b"
    render: bool = True


def _safe(name: str) -> str:
    return name.replace(":", "-").replace("/", "-")


class _Timer:
    def __init__(self):
        self.timings: dict[str, float] = {}

    def run(self, name, fn, *a, **kw):
        t0 = time.monotonic()
        try:
            return fn(*a, **kw)
        finally:
            self.timings[name] = round(self.timings.get(name, 0) + time.monotonic() - t0, 1)


def _ingest(source: str, settings: Settings) -> tuple[Path, VideoMeta]:
    if source.lower().startswith(("http://", "https://", "www.", "youtu")):
        return ingest_youtube(source, settings)
    return ingest_file(Path(source), settings)


MANUAL_RUN = "claude-manual"
CLAUDE_PROMPT_FILE = "untuk_claude.md"


class ImportFailed(Exception):
    """The reply pasted back from Claude could not be used."""


def prepare_project(src: Path, meta: VideoMeta, opts: RunOptions, settings: Settings,
                    timer: _Timer) -> tuple[Transcript, list[tuple[float, float]], Path]:
    """Audio, silences and transcript (all checkpointed). Returns transcript, silences, transcript dir."""
    d = project_dir(settings, meta.project_id)
    wav = d / "audio.wav"
    timer.run("audio", extract_audio, src, wav)
    silences = read_json(d / "silences.json")
    if silences is None:
        silences = timer.run("silences", detect_silences, wav)
        atomic_write_json(d / "silences.json", silences)
    tdir = d / "transcripts" / _safe(opts.whisper_model)
    tdir.mkdir(parents=True, exist_ok=True)
    corrections = load_corrections(Path(settings.data_dir) / "corrections.json")
    t: Transcript = timer.run("transcribe", transcribe, wav, meta.duration, tdir,
                              FasterWhisperTranscriber(opts.whisper_model), opts.whisper_model, corrections,
                              on_progress=lambda p: print(f"  transkrip {p:.0%}", flush=True))
    return t, [tuple(s) for s in silences], tdir


def process_project(src: Path, meta: VideoMeta, opts: RunOptions, settings: Settings, timer: _Timer) -> Path:
    t, silences, tdir = prepare_project(src, meta, opts, settings, timer)
    llm = OllamaClient(settings.ollama_url, opts.llm_model)
    # analysis checkpoints live next to the transcript they index into (sentence numbers differ per model)
    picked = timer.run("analyze", analyze, t, meta, llm, opts.max_clips, opts.min_sec, opts.max_sec,
                       tdir / "checkpoints")
    return finish_clips(src, meta, t, silences, picked, opts, opts.llm_model, timer)


def finish_clips(src: Path, meta: VideoMeta, t: Transcript, silences: list[tuple[float, float]],
                 picked: list, opts: RunOptions, picker: str, timer: _Timer) -> Path:
    """Snap edges, reframe, render and write the report for already-chosen clips."""
    d = project_dir(get_settings(), meta.project_id)
    run_dir = d / "runs" / f"{_safe(opts.whisper_model)}__{_safe(picker)}"
    run_dir.mkdir(parents=True, exist_ok=True)
    presets = load_presets()
    by_time = sorted(picked, key=lambda p: p[0].start)
    spans = [(c.start, c.end) for c, _ in by_time]
    clips: list[ClipPlan] = []
    t0 = time.monotonic()
    for i, (cand, mats) in enumerate(by_time):
        lower, upper = neighbor_bounds(spans, i, meta.duration)
        s, e = snap_edges(cand.start, cand.end, t.words, t.sentences, silences, lower=lower, upper=upper,
                          max_len=opts.max_sec + SNAP_SLACK)
        cand = cand.model_copy(update={"start": s, "end": e})
        segs, layout = reframe_clip(src, s, e)
        clips.append(ClipPlan(id=uuid.uuid4().hex[:8], rank=0, candidate=cand, materials=mats,
                              subtitles=group_lines(t.words, s, e), reframe_segments=segs, layout=layout))
    clips.sort(key=lambda c: c.candidate.score, reverse=True)
    for rank, c in enumerate(clips, 1):
        c.rank = rank
    for c in clips if opts.render else []:
        print(f"  render klip {c.rank}/{len(clips)}", flush=True)
        render_clip(src, c, run_dir / "clips" / clip_folder(c), presets, FONTS_DIR)
    timer.timings["clips"] = round(time.monotonic() - t0, 1)

    atomic_write_json(run_dir / "clips.json", [c.model_dump() for c in clips])
    atomic_write_json(run_dir / "timings.json", timer.timings)
    report = run_dir / "report.md"
    if report.exists():  # never overwrite an evaluation the user may have filled in
        report = run_dir / f"report-{time.strftime('%Y%m%d-%H%M%S')}.md"
    report.write_text(build_report(meta, clips, timer.timings,
                                   {"whisper": opts.whisper_model, "llm": picker}), encoding="utf-8")
    return report


def _load_project(project_id: str, settings: Settings) -> tuple[Path, VideoMeta]:
    d = Path(settings.data_dir) / "projects" / project_id
    meta = read_json(d / "meta.json")
    if meta is None or not (d / "source.mp4").exists():
        raise ImportFailed(f"Project {project_id} tidak ditemukan di {d}. Jalankan export-transcript dulu.")
    return d / "source.mp4", VideoMeta.model_validate(meta)


def cmd_export(args, settings: Settings) -> Path:
    timer = _Timer()
    src, meta = timer.run("ingest", _ingest, args.source, settings)
    opts = _opts(args, settings)
    t, _, tdir = prepare_project(src, meta, opts, settings, timer)
    prompt = render_prompt("claude_manual", max_candidates=opts.max_clips, min_sec=opts.min_sec,
                           max_sec=opts.max_sec, transcript=transcript_lines(t))
    out = tdir / CLAUDE_PROMPT_FILE
    out.write_text(prompt, encoding="utf-8")
    print(f"Project ID: {meta.project_id}")
    print("Berikan isi file ini ke Claude (Claude Code atau claude.ai), simpan balasannya sebagai .json, lalu:")
    print(f"  uv run python -m app.cli import-moments {meta.project_id} <file-balasan.json>")
    return out


def cmd_import(args, settings: Settings) -> Path:
    src, meta = _load_project(args.project_id, settings)
    opts = _opts(args, settings)
    timer = _Timer()
    t, silences, _ = prepare_project(src, meta, opts, settings, timer)
    raw = parse_llm_json(Path(args.reply).read_text(encoding="utf-8"))
    try:
        if raw is None:
            raise ValueError("Balasan bukan JSON. Pastikan menyalin seluruh balasan Claude.")
        picked = import_external(t, meta, raw, opts.max_clips, opts.min_sec, opts.max_sec)
    except ValueError as e:
        raise ImportFailed(str(e)) from e
    return finish_clips(src, meta, t, silences, picked, opts, MANUAL_RUN, timer)


def cmd_run(args, settings: Settings) -> Path:
    timer = _Timer()
    src, meta = timer.run("ingest", _ingest, args.source, settings)
    return process_project(src, meta, _opts(args, settings), settings, timer)


def cmd_analyze(args, settings: Settings) -> Path:
    try:
        src, meta = _load_project(args.project_id, settings)
    except ImportFailed as e:
        raise IngestError("INVALID_URL", str(e)) from e
    return process_project(src, meta, _opts(args, settings), settings, _Timer())


def cmd_bench_whisper(args, settings: Settings) -> Path:
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


def _opts(args, settings: Settings) -> RunOptions:
    return RunOptions(max_clips=args.max_clips, min_sec=args.min_sec, max_sec=args.max_sec,
                      whisper_model=args.whisper_model or settings.whisper_model,
                      llm_model=args.llm_model or settings.ollama_model, render=not args.no_render)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m app.cli", description="Kliplah: podcast ke klip vertikal")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp):
        sp.add_argument("--max-clips", type=int, default=8)
        sp.add_argument("--min-sec", type=int, default=30)
        sp.add_argument("--max-sec", type=int, default=90)
        sp.add_argument("--whisper-model", default=None)
        sp.add_argument("--llm-model", default=None)
        sp.add_argument("--no-render", action="store_true")

    run = sub.add_parser("run", help="link YouTube atau file video")
    run.add_argument("source")
    common(run)
    an = sub.add_parser("analyze", help="analisis ulang project yang sudah ada")
    an.add_argument("project_id")
    common(an)
    ex = sub.add_parser("export-transcript", help="transkrip + instruksi untuk dipilihkan Claude (tanpa biaya API)")
    ex.add_argument("source")
    common(ex)
    im = sub.add_parser("import-moments", help="pakai balasan JSON dari Claude untuk membuat klip")
    im.add_argument("project_id")
    im.add_argument("reply", help="file berisi balasan Claude")
    common(im)
    bw = sub.add_parser("bench-whisper", help="bandingkan model Whisper")
    bw.add_argument("source")
    bw.add_argument("--models", nargs="+", default=["small", "medium"])
    bw.add_argument("--seconds", type=float, default=300.0)
    return p


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    args = _parser().parse_args(argv)
    settings = get_settings()
    handler = {"run": cmd_run, "analyze": cmd_analyze, "bench-whisper": cmd_bench_whisper,
               "export-transcript": cmd_export, "import-moments": cmd_import}[args.command]
    try:
        out = handler(args, settings)
    except ImportFailed as e:
        print(f"[INVALID_IMPORT] {e}")
        return 2
    except IngestError as e:
        print(f"[{e.code}] {e}")
        return 1 if e.retryable else 2
    except RenderError as e:
        print(f"[{e.code}] {e}")
        return 1
    print(f"Selesai: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
