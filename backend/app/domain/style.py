"""Three-layer style model: template -> project style (full) -> clip override (diff).

The effective style of a clip is ``deep_merge(project.style, clip.style_override)``
validated against ``Style``. The JSON schema and default values are exported to
``shared/style_schema.json`` and ``shared/style_default.json`` (see
``scripts/export_style_schema.py``) and checked by a parity test.
"""
import copy
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.core.errors import AppError

Color = Annotated[str, Field(pattern=r"^#[0-9A-Fa-f]{6}$")]
Opacity = Annotated[float, Field(ge=0.0, le=1.0)]


class _StyleModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BoxStyle(_StyleModel):
    enabled: bool = False
    color: Color = "#000000"
    opacity: Opacity = 0.6
    padding: Annotated[int, Field(ge=0)] = 16
    radius: Annotated[int, Field(ge=0)] = 12


class Position(_StyleModel):
    anchor: Literal["top", "middle", "bottom"]
    # Pixels on the 1080x1920 canvas, measured from the anchor edge.
    offset_y: int


class SubtitleStyle(_StyleModel):
    enabled: bool = True
    mode: Literal["line", "highlight_word"] = "line"
    size: Annotated[int, Field(gt=0)] = 104
    color: Color = "#FFFFFF"
    outline_color: Color = "#000000"
    outline_width: Annotated[int, Field(ge=0)] = 8
    highlight_color: Color = "#FFE14D"
    uppercase: bool = False
    words_per_line: Annotated[int, Field(ge=1, le=4)] = 4
    box: BoxStyle = Field(default_factory=BoxStyle)
    speaker_emoji: bool = False
    position: Position = Field(default_factory=lambda: Position(anchor="bottom", offset_y=420))
    language_mode: Literal["target", "bilingual", "original"] = "target"


class HookStyle(_StyleModel):
    enabled: bool = True
    duration: Annotated[float, Field(gt=0)] | Literal["full"] = 3.0
    size: Annotated[int, Field(gt=0)] = 96
    color: Color = "#FFFFFF"
    outline_color: Color = "#000000"
    outline_width: Annotated[int, Field(ge=0)] = 7
    position: Position = Field(default_factory=lambda: Position(anchor="top", offset_y=260))
    box: BoxStyle = Field(default_factory=BoxStyle)


class LogoStyle(_StyleModel):
    enabled: bool
    asset_id: str
    anchor: Literal["top-left", "top-right", "bottom-left", "bottom-right", "top-center"]
    size: Annotated[int, Field(gt=0)]
    opacity: Opacity


class Background(_StyleModel):
    type: Literal["blur", "solid"] = "blur"
    color: Color = "#000000"


class Style(_StyleModel):
    font_family: str = "Montserrat ExtraBold"
    subtitle: SubtitleStyle = Field(default_factory=SubtitleStyle)
    hook: HookStyle = Field(default_factory=HookStyle)
    logo: LogoStyle | None = None
    background: Background = Field(default_factory=Background)


def default_style() -> dict:
    """Full default style as a plain dict (fresh copy on every call)."""
    return Style().model_dump()


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge ``override`` into a copy of ``base``.

    Dicts merge key by key; any non-dict value (including lists and ``None``) replaces.
    Neither input is mutated.
    """
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def style_diff(base: dict, target: dict) -> dict:
    """Leaves of ``target`` that differ from ``base``; ``deep_merge(base, diff) == target``.

    Keys present in ``base`` but missing from ``target`` cannot be expressed as an
    override and are ignored (styles are always complete).
    """
    diff: dict[str, Any] = {}
    for key, value in target.items():
        if key not in base:
            diff[key] = copy.deepcopy(value)
            continue
        base_value = base[key]
        if isinstance(value, dict) and isinstance(base_value, dict):
            sub = style_diff(base_value, value)
            if sub:
                diff[key] = sub
        elif value != base_value or type(value) is not type(base_value):
            diff[key] = copy.deepcopy(value)
    return diff


def _validate(style: dict) -> dict:
    try:
        return Style.model_validate(style).model_dump()
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first["loc"]) or "gaya"
        raise AppError("VALIDATION", f"Gaya tidak valid pada '{loc}': {first['msg']}") from exc


def effective_style(project_style: dict, override: dict) -> dict:
    """Project style deep-merged with a clip override, validated and normalised."""
    return _validate(deep_merge(project_style, override or {}))


def validate_style_patch(patch: dict) -> dict:
    """Accept a partial style patch if it is valid on top of the defaults; return it unchanged."""
    if not isinstance(patch, dict):
        raise AppError("VALIDATION", "Gaya tidak valid: patch harus berupa objek.")
    _validate(deep_merge(default_style(), patch))
    return patch
