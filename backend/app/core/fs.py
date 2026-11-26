import json
import os
from pathlib import Path
from typing import Any

from app.core.config import Settings


def atomic_write_json(path: Path, data: Any) -> None:
    """Write JSON so readers never see a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_json(path: Path) -> Any | None:
    """Return parsed JSON, or None when the file is missing or torn."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError):
        return None


def project_dir(settings: Settings, project_id: str) -> Path:
    d = Path(settings.data_dir) / "projects" / project_id
    d.mkdir(parents=True, exist_ok=True)
    return d
