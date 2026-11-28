"""Signal storage and moment score bonuses from live chat and audio peaks."""
from pathlib import Path

from pydantic import BaseModel

from app.core.fs import atomic_write_json, read_json

CHAT_CAP_STREAM = 15
CHAT_CAP_OTHER = 8
AUDIO_CAP = 6


class Signals(BaseModel):
    chat: list[dict] = []
    audio: list[dict] = []


def save_signals(path: Path, signals: Signals) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, signals.model_dump())


def load_signals(path: Path) -> Signals:
    raw = read_json(path)
    return Signals() if raw is None else Signals.model_validate(raw)


def _max_overlap(start: float, end: float, points: list[dict]) -> float:
    vals = [float(p.get("value", 0.0)) for p in points if float(p["start"]) < end and float(p["end"]) > start]
    return min(max(max(vals), 0.0), 1.0) if vals else 0.0


def signal_bonus(start: float, end: float, signals: Signals, content_type: str) -> int:
    """Profile-capped signal bonus (chat + audio peak)."""
    chat_cap = CHAT_CAP_STREAM if content_type == "stream" else CHAT_CAP_OTHER
    chat_part = round(chat_cap * _max_overlap(start, end, signals.chat))
    audio_part = round(AUDIO_CAP * _max_overlap(start, end, signals.audio))
    return chat_part + audio_part
