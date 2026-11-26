"""Shared data shapes for the pipeline."""
from typing import Literal

from pydantic import BaseModel


class Word(BaseModel):
    w: str
    start: float
    end: float


class Sentence(BaseModel):
    i: int
    start: float
    end: float
    text: str
    word_range: tuple[int, int]  # half-open [a, b)


class Transcript(BaseModel):
    language: str
    duration: float
    words: list[Word]
    sentences: list[Sentence]


class HeatPoint(BaseModel):
    start: float
    end: float
    value: float


class VideoMeta(BaseModel):
    source_type: Literal["youtube", "upload"]
    project_id: str
    title: str
    channel: str | None = None
    duration: float
    video_id: str | None = None
    heatmap: list[HeatPoint] = []


class Window(BaseModel):
    index: int
    start_sentence: int
    end_sentence: int  # inclusive


CATEGORIES = ("hook", "emotional_peak", "hot_take", "revelation", "conflict",
              "quotable", "story_peak", "practical_value")


class Candidate(BaseModel):
    start_sentence: int
    end_sentence: int
    start: float
    end: float
    llm_score: int | None
    category: str | None
    reason: str
    hook: str
    heatmap_bonus: int = 0
    score: int = 0
    no_ai: bool = False


class PlatformCaption(BaseModel):
    title: str
    caption: str
    hashtags: list[str]


class Materials(BaseModel):
    hook: str
    tiktok: PlatformCaption
    reels: PlatformCaption
    shorts: PlatformCaption


class SubtitleLine(BaseModel):
    id: int
    start: float
    end: float
    text: str
    words: list[Word]
    edited: bool = False


class ReframeSegment(BaseModel):
    start: float
    end: float
    mode: Literal["track", "wide", "general"]
    x: float
    zoom: float = 1.0
    user_override: bool = False


class ClipPlan(BaseModel):
    id: str
    rank: int
    candidate: Candidate
    materials: Materials
    subtitles: list[SubtitleLine]
    reframe_segments: list[ReframeSegment]
    layout: Literal["face_crop", "fit_blur"]
    caption_preset: Literal["klasik", "sorot_kata"] = "klasik"
    hook_duration_sec: float = 3.0
