"""Helpers shared by the project, clip and style services: HTTP-aware errors and row lookup."""
from collections.abc import Iterator
from contextlib import contextmanager
from typing import TypeVar

from pydantic import ValidationError
from sqlmodel import Session, SQLModel

from app.core.errors import AppError

STATUS = {"NOT_FOUND": 404, "INVALID_STATE": 409, "VALIDATION": 422, "INVALID_URL": 400, "AGE_RESTRICTED": 400}

T = TypeVar("T", bound=SQLModel)


def err(code: str, message: str) -> AppError:
    return AppError(code, message, status=STATUS.get(code, 400))


@contextmanager
def domain_errors(field: str) -> Iterator[None]:
    """Give domain AppErrors their HTTP status and turn pydantic errors into VALIDATION naming ``field``."""
    try:
        yield
    except AppError as e:
        e.status = STATUS.get(e.code, e.status)
        raise
    except ValidationError as e:
        first = e.errors()[0]
        loc = ".".join(str(p) for p in first["loc"])
        where = f"{field}.{loc}" if loc else field
        raise err("VALIDATION", f"Data '{field}' tidak valid pada '{where}': {first['msg']}") from e


def fetch(session: Session, model: type[T], row_id: str, label: str) -> T:
    row = session.get(model, row_id, populate_existing=True)
    if row is None:
        raise err("NOT_FOUND", f"{label} {row_id} tidak ditemukan.")
    return row
