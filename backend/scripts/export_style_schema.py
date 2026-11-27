"""Export the style JSON schema and default style to ``shared/`` at the repo root.

Run from ``backend/``: ``uv run python scripts/export_style_schema.py``.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.core.config import REPO_DIR  # noqa: E402
from app.domain.style import Style, default_style  # noqa: E402


def main() -> None:
    shared = REPO_DIR / "shared"
    shared.mkdir(parents=True, exist_ok=True)
    for name, data in (("style_schema.json", Style.model_json_schema()), ("style_default.json", default_style())):
        path = shared / name
        # Property order follows the model definition, so output is stable across runs.
        text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8", newline="\n")
        tmp.replace(path)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
