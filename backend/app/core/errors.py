"""Application error carrying a stable machine code for the API error envelope."""

KNOWN_CODES = frozenset({
    "INVALID_URL", "VIDEO_UNAVAILABLE", "DOWNLOAD_BLOCKED", "FFMPEG_MISSING", "OLLAMA_UNAVAILABLE",
    "MODEL_MISSING", "DISK_FULL", "NO_MOMENTS_FOUND", "RENDER_STALLED", "INVALID_IMPORT",
    "NOT_FOUND", "INVALID_STATE", "VALIDATION", "AGE_RESTRICTED", "UNKNOWN",
})


class AppError(Exception):
    def __init__(self, code: str, message: str, status: int = 400, retryable: bool = False):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status = status
        self.retryable = retryable

    def __repr__(self) -> str:
        return f"AppError({self.code!r}, {self.message!r}, status={self.status}, retryable={self.retryable})"
