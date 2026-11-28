"""Subtitle lines and the .ass file burned into the final clip.

ASS uses PlayRes 1080x1920, so every size and margin in shared/caption_presets.json is a real
pixel on the output frame and maps 1:1 to the browser preview (scaled).
"""
import json
from functools import lru_cache
from pathlib import Path

from app.core.config import REPO_DIR
from app.core.models import SubtitleLine, Word

CONNECTORS = {"dan", "tapi", "karena", "yang", "jadi"}
LINE_BREAK_PUNCT = (".", "?", "!", ",")
EPS = 1e-6


@lru_cache
def _load(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_presets(path: Path | None = None) -> dict:
    return _load(str(path or REPO_DIR / "shared" / "caption_presets.json"))


def group_lines(words: list[Word], clip_start: float, clip_end: float, max_words: int = 4,
                gap: float = 0.4) -> list[SubtitleLine]:
    inside = [Word(w=w.w, start=max(0.0, round(w.start - clip_start, 3)),
                   end=round(min(w.end, clip_end) - clip_start, 3), approx=w.approx, prob=w.prob)
              for w in words if w.end > clip_start + EPS and w.start < clip_end - EPS]
    lines: list[SubtitleLine] = []
    cur: list[Word] = []

    def flush():
        if cur:
            lines.append(SubtitleLine(id=len(lines) + 1, start=cur[0].start, end=cur[-1].end,
                                      text=" ".join(x.w for x in cur), words=list(cur)))
            cur.clear()

    for w in inside:
        if cur and (w.start - cur[-1].end > gap or (len(cur) >= 2 and w.w.lower().strip(".,?!") in CONNECTORS)):
            flush()
        cur.append(w)
        if len(cur) >= max_words or w.w.endswith(LINE_BREAK_PUNCT):
            flush()
    flush()
    return lines


def sanitize_ass_text(s: str) -> str:
    return s.replace("{", "").replace("}", "").replace("\\", "/").replace("\r", " ").replace("\n", " ")


def ass_color(hex_rgb: str) -> str:
    h = hex_rgb.lstrip("#").upper()
    return f"&H00{h[4:6]}{h[2:4]}{h[0:2]}"


def _ts(sec: float) -> str:
    cs = max(0, round(sec * 100))
    h, rem = divmod(cs, 360000)
    m, rem = divmod(rem, 6000)
    s, c = divmod(rem, 100)
    return f"{h}:{m:02d}:{s:02d}.{c:02d}"


def _wrap(text: str, max_chars: int) -> str:
    out, line = [], ""
    for word in text.split():
        if line and len(line) + 1 + len(word) > max_chars:
            out.append(line)
            line = word
        else:
            line = f"{line} {word}".strip()
    if line:
        out.append(line)
    return "\\N".join(out)


def _style(name: str, font: str, size: int, color: str, outline: str, width: int, align: int, margin_v: int) -> str:
    c, o = ass_color(color), ass_color(outline)
    return (f"Style: {name},{font},{size},{c},{c},{o},&H00000000,0,0,0,0,100,100,0,0,1,{width},0,"
            f"{align},60,60,{margin_v},1")


def build_ass(lines: list[SubtitleLine], hook: str, hook_duration: float, preset_name: str, presets: dict) -> str:
    font = presets["font"]["family"]
    hk, p = presets["hook"], presets["presets"][preset_name]
    out = [
        "[Script Info]", "ScriptType: v4.00+",
        f"PlayResX: {presets['canvas']['width']}", f"PlayResY: {presets['canvas']['height']}",
        "WrapStyle: 0", "ScaledBorderAndShadow: yes", "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, "
        "Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding",
        _style("Hook", font, hk["size"], hk["color"], hk["outline"], hk["outline_width"], 8, hk["margin_top"]),
        _style("Sub", font, p["size"], p["color"], p["outline"], p["outline_width"], 2, p["margin_bottom"]),
        "", "[Events]", "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    hook_text = sanitize_ass_text(hook).strip()
    if hook_text:
        out.append(f"Dialogue: 1,{_ts(0)},{_ts(hook_duration)},Hook,,0,0,0,,{_wrap(hook_text, hk['max_chars_per_line'])}")

    hl = p.get("highlight")
    for line in lines:
        if not hl or line.edited or not line.words:
            out.append(f"Dialogue: 0,{_ts(line.start)},{_ts(line.end)},Sub,,0,0,0,,{sanitize_ass_text(line.text)}")
            continue
        toks = [sanitize_ass_text(w.w) for w in line.words]
        for i, w in enumerate(line.words):
            end = line.words[i + 1].start if i + 1 < len(line.words) else line.end
            shown = " ".join(f"{{\\c{ass_color(hl)}&}}{t}{{\\r}}" if j == i else t for j, t in enumerate(toks))
            out.append(f"Dialogue: 0,{_ts(w.start)},{_ts(end)},Sub,,0,0,0,,{shown}")
    return "\n".join(out) + "\n"
