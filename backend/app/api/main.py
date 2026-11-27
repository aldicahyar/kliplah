"""FastAPI app factory: routers under `/api`, error envelope, CORS for the Vite dev server.

Run: `uv run uvicorn app.api.main:create_app --factory --host 127.0.0.1 --port 8000`
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api import routes_clips, routes_projects, routes_styles, routes_system
from app.core.config import Settings, get_settings
from app.core.errors import AppError
from app.db.engine import get_engine, init_db
from app.services._common import STATUS
from app.services.styles import ensure_default_template

log = logging.getLogger(__name__)

CORS_ORIGINS = ["http://localhost:5173"]


def _error(status: int, code: str, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": {"code": code, "message": message}})


_TYPE_PHRASES = {
    "int": "harus bilangan bulat", "float": "harus berupa angka", "bool": "harus bernilai true atau false",
    "string": "harus berupa teks", "dict": "harus berupa objek", "list": "harus berupa daftar",
}
_PHRASES = {
    "missing": "wajib diisi", "string_too_short": "terlalu pendek", "string_too_long": "terlalu panjang",
    "literal_error": "bukan salah satu pilihan yang diizinkan", "enum": "bukan salah satu pilihan yang diizinkan",
    "extra_forbidden": "field tidak dikenal", "json_invalid": "JSON tidak valid",
}
_BOUNDS = {"greater_than": ("gt", "harus lebih besar dari"), "greater_than_equal": ("ge", "minimal"),
           "less_than": ("lt", "harus lebih kecil dari"), "less_than_equal": ("le", "maksimal")}
_LOC_SOURCES = ("body", "query", "path", "header", "cookie")


def _phrase(error: dict) -> str:
    """A short Indonesian phrase for a pydantic error type (pydantic's English message is never shown)."""
    kind = str(error.get("type", ""))
    if kind in _PHRASES:
        return _PHRASES[kind]
    if kind in _BOUNDS:
        key, words = _BOUNDS[kind]
        return f"{words} {(error.get('ctx') or {}).get(key, '')}".strip()
    if kind.endswith(("_parsing", "_type")):
        return _TYPE_PHRASES.get(kind.rsplit("_", 1)[0], "tipe datanya tidak sesuai")
    return "nilai tidak valid"


def _validation_message(exc: RequestValidationError) -> str:
    errors = exc.errors()
    if not errors:
        return "Data permintaan tidak valid."
    first = errors[0]
    loc = [str(p) for p in first.get("loc", ())]
    if loc and loc[0] in _LOC_SOURCES:
        loc = loc[1:]
    if first.get("type") == "json_invalid":  # its loc is a character offset, not a field
        loc = []
    where = ".".join(loc) or "permintaan"
    return f"Data permintaan tidak valid pada '{where}': {_phrase(first)}."


class _CatchUnexpected:
    """Turn an unhandled exception into the 500 UNKNOWN envelope. Registered inside CORSMiddleware so the
    browser still gets CORS headers (Starlette's own ServerErrorMiddleware sits outside CORS)."""

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        started = False

        async def tracked_send(message):
            nonlocal started
            if message["type"] == "http.response.start":
                started = True
            await send(message)

        try:
            await self.app(scope, receive, tracked_send)
        except Exception:
            if started:  # too late for an error body; let the server close the connection
                raise
            log.exception("Unhandled error on %s %s", scope.get("method"), scope.get("path"))
            await _error(500, "UNKNOWN", "Terjadi kesalahan tak terduga.")(scope, receive, send)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    engine = get_engine(settings)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        init_db(engine)
        ensure_default_template(engine)
        try:
            yield
        finally:
            engine.dispose()

    app = FastAPI(title="Kliplah API", lifespan=lifespan)
    app.state.settings = settings
    app.state.engine = engine

    app.add_middleware(_CatchUnexpected)  # added first = runs inside CORS
    app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"],
                       expose_headers=["Content-Disposition", "Content-Range", "Accept-Ranges"])

    @app.exception_handler(AppError)
    async def _app_error(_request: Request, exc: AppError) -> JSONResponse:
        return _error(STATUS.get(exc.code, exc.status), exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def _request_validation(_request: Request, exc: RequestValidationError) -> JSONResponse:
        return _error(422, "VALIDATION", _validation_message(exc))

    for module in (routes_projects, routes_clips, routes_styles, routes_system):
        app.include_router(module.router, prefix="/api")
    return app
