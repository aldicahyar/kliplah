import os
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_DIR = BACKEND_DIR.parent
RANGE_MAX_HEIGHTS = (480, 720, 1080)
DEFAULT_RANGE_MAX_HEIGHT = 720


def cpu_count() -> int:
    return max(1, os.cpu_count() or 1)


def default_encode_threads() -> int:
    return max(1, cpu_count() - 2)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    data_dir: Path = Path("./data")
    whisper_model: str = "small"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    max_video_height: int = 1080
    default_moment_source: str = "claude_manual"
    api_host: str = "127.0.0.1"
    api_port: int = 8000


@lru_cache
def get_settings() -> Settings:
    return Settings()
