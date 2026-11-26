from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_DIR = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BACKEND_DIR / ".env", extra="ignore")

    data_dir: Path = Path("./data")
    whisper_model: str = "small"
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:4b"
    max_video_height: int = 1080


@lru_cache
def get_settings() -> Settings:
    return Settings()
