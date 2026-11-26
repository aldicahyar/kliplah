"""Markdown report the user fills in during the M1 evaluation gate."""
from app.core.models import ClipPlan, VideoMeta


def _mmss(sec: float) -> str:
    sec = int(round(sec))
    return f"{sec // 60:02d}:{sec % 60:02d}"


def _cell(s: str | None) -> str:
    return (s or "").replace("|", "\\|").replace("\n", " ").strip()


def clip_folder(clip: ClipPlan) -> str:
    return f"{clip.rank:02d}_{clip.id}"


def build_report(meta: VideoMeta, clips: list[ClipPlan], timings: dict[str, float], models: dict[str, str]) -> str:
    out = [
        f"# Laporan evaluasi: {_cell(meta.title)}",
        "",
        f"- Sumber: {meta.source_type} `{meta.project_id}`, durasi {_mmss(meta.duration)}",
        f"- Model: Whisper `{models.get('whisper')}`, LLM `{models.get('llm')}`",
        f"- Heatmap YouTube: {'ada' if meta.heatmap else 'tidak ada'}",
        "",
        "## Waktu per tahap",
        "",
        "| Tahap | Waktu |",
        "|---|---|",
        *[f"| {k} | {_mmss(v)} |" for k, v in timings.items()],
        f"| **total** | **{_mmss(sum(timings.values()))}** |",
        "",
        "## Klip",
        "",
        "Isi kolom **Layak?** dengan `ya` jika klip bisa dipakai dengan edit kecil saja, `tidak` jika tidak.",
        "Kolom **AI?** bernilai `tidak` untuk klip cadangan tanpa skor AI (dipilih dari pergantian topik).",
        "",
        "| # | Skor | AI? | Kategori | Waktu | Durasi | Hook | Alasan | File | Layak? (ya/tidak) | Catatan |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for c in clips:
        k = c.candidate
        out.append(
            f"| {c.rank} | {k.score} | {'tidak' if k.no_ai else 'ya'} | {k.category or '-'} | "
            f"{_mmss(k.start)}-{_mmss(k.end)} | {round(k.end - k.start)} s | {_cell(c.materials.hook or k.hook)} | "
            f"{_cell(k.reason)} | clips/{clip_folder(c)}/final.mp4 |  |  |")
    out += ["", f"Klip layak: ___ dari {len(clips)}", ""]
    return "\n".join(out)
