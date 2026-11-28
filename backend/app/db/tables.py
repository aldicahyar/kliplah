"""SQLModel tables (schema version 2). Table names are SQLModel defaults (lowercase class name)."""
import uuid
from datetime import datetime, timezone

from sqlalchemy import JSON, Column, DateTime, TypeDecorator
from sqlmodel import Field, SQLModel


def _uuid() -> str:
    return uuid.uuid4().hex


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class UTCDateTime(TypeDecorator):
    """Stores UTC as naive in SQLite, always returns timezone-aware UTC datetimes."""
    impl = DateTime
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is not None and value.tzinfo is not None:
            value = value.astimezone(timezone.utc).replace(tzinfo=None)
        return value

    def process_result_value(self, value, dialect):
        if value is not None and value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value


def _json(nullable: bool = True) -> Column:
    return Column(JSON, nullable=nullable)


def _ts():
    return Field(default_factory=utcnow, sa_type=UTCDateTime)


def _ts_opt():
    return Field(default=None, sa_type=UTCDateTime)


class Project(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    source_type: str = "youtube"  # youtube|upload
    youtube_url: str | None = None
    video_id: str | None = None
    original_filename: str | None = None
    title: str | None = None
    channel: str | None = None
    duration_sec: float | None = None
    has_heatmap: bool = False
    content_type: str = "general"
    ranges: list | None = Field(default=None, sa_column=_json())
    focus_prompt: str | None = None
    source_language: str | None = None
    target_language: str = "id"
    transcript_source: str = "auto"
    moment_source: str = "claude_manual"
    max_clips: int = 8
    min_clip_sec: int = 30
    max_clip_sec: int = 90
    style: dict = Field(default_factory=dict, sa_column=_json(nullable=False))
    style_template_id: str | None = None
    facecam_box: dict | None = Field(default=None, sa_column=_json())
    signals_available: dict = Field(default_factory=dict, sa_column=_json(nullable=False))
    status: str = Field(default="queued", index=True)  # queued|processing|awaiting_moments|ready|failed
    last_completed_stage: str | None = None
    whisper_model: str = "small"
    keep_source: bool = False
    source_deleted_at: datetime | None = _ts_opt()
    last_activity_at: datetime = _ts()
    first_opened_at: datetime | None = _ts_opt()
    created_at: datetime = _ts()
    updated_at: datetime = _ts()


class Clip(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    project_id: str | None = Field(default=None, foreign_key="project.id", index=True)  # None = compilation
    kind: str = "single"
    origin: str  # ai|manual|fallback|m1
    rank: int
    score: int
    llm_score: int | None = None
    heatmap_bonus: int = 0
    signal_bonus: int = 0
    category: str = ""
    reason: str = ""
    pieces: list = Field(default_factory=list, sa_column=_json(nullable=False))
    layout_segments: list = Field(default_factory=list, sa_column=_json(nullable=False))
    style_override: dict = Field(default_factory=dict, sa_column=_json(nullable=False))
    style: dict | None = Field(default=None, sa_column=_json())
    overlays: list = Field(default_factory=list, sa_column=_json(nullable=False))
    hook_text: str = ""
    hook_options: list = Field(default_factory=list, sa_column=_json(nullable=False))
    subtitles: list = Field(default_factory=list, sa_column=_json(nullable=False))
    captions: dict = Field(default_factory=dict, sa_column=_json(nullable=False))
    status: str = Field(default="proposed", index=True)  # proposed|approved|rejected|rendering|rendered|render_failed
    needs_rerender: bool = False
    output_path: str | None = None
    created_at: datetime = _ts()
    updated_at: datetime = _ts()


class Job(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    type: str  # process|render
    project_id: str | None = Field(default=None, foreign_key="project.id", index=True)
    clip_id: str | None = Field(default=None, foreign_key="clip.id", index=True)
    status: str = Field(default="queued", index=True)  # queued|running|succeeded|failed|cancelled
    stage: str | None = None
    progress: float = 0
    attempts: int = 0
    resume_cursor: dict | None = Field(default=None, sa_column=_json())
    error_code: str | None = None
    error_message: str | None = None
    created_at: datetime = _ts()
    started_at: datetime | None = _ts_opt()
    finished_at: datetime | None = _ts_opt()


class StyleTemplate(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    name: str = Field(unique=True)
    is_default: bool = False
    style: dict = Field(default_factory=dict, sa_column=_json(nullable=False))
    created_at: datetime = _ts()
    updated_at: datetime = _ts()


class Asset(SQLModel, table=True):
    id: str = Field(default_factory=_uuid, primary_key=True)
    type: str = "logo"
    path: str
    created_at: datetime = _ts()


class AppSetting(SQLModel, table=True):
    key: str = Field(primary_key=True)
    value: dict = Field(default_factory=dict, sa_column=_json(nullable=False))
