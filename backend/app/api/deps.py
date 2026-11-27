"""Request dependencies and small helpers shared by the routers."""
from pathlib import Path

from fastapi import Request
from fastapi.responses import FileResponse
from sqlalchemy import Engine
from sqlmodel import SQLModel

from app.core.config import Settings
from app.core.errors import AppError


def get_engine_dep(request: Request) -> Engine:
    return request.app.state.engine


def get_settings_dep(request: Request) -> Settings:
    return request.app.state.settings


def dump(row: SQLModel | None) -> dict | None:
    return None if row is None else row.model_dump(mode="json")


def file_or_404(path: Path | None, what: str, **kwargs) -> FileResponse:
    """`FileResponse` (Range-aware) for an existing file, else NOT_FOUND 404."""
    if path is None or not Path(path).is_file():
        raise AppError("NOT_FOUND", f"File {what} tidak ditemukan.", status=404)
    return FileResponse(path, **kwargs)
