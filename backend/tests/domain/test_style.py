import copy
import json

import pytest

from app.core.config import REPO_DIR
from app.core.errors import AppError
from app.domain.style import (
    Style,
    deep_merge,
    default_style,
    effective_style,
    style_diff,
    validate_style_patch,
)


def test_default_style_values():
    d = default_style()
    assert d["subtitle"]["size"] == 104
    assert d["subtitle"]["outline_width"] == 8
    assert d["hook"]["size"] == 96
    assert d["hook"]["duration"] == 3.0
    assert d["font_family"] == "Montserrat ExtraBold"
    assert d["subtitle"]["color"] == "#FFFFFF"
    assert d["subtitle"]["outline_color"] == "#000000"
    assert d["subtitle"]["highlight_color"] == "#FFE14D"
    assert d["subtitle"]["words_per_line"] == 4
    assert d["subtitle"]["mode"] == "line"
    assert d["subtitle"]["position"] == {"anchor": "bottom", "offset_y": 420}
    assert d["hook"]["outline_width"] == 7
    assert d["hook"]["position"] == {"anchor": "top", "offset_y": 260}
    assert d["background"] == {"type": "blur", "color": "#000000"}
    assert d["logo"] is None
    # default_style returns a fresh copy each call
    d["subtitle"]["size"] = 1
    assert default_style()["subtitle"]["size"] == 104


def test_deep_merge_nested_and_list_replace():
    base = {"a": {"b": 1, "c": 2}, "l": [1]}
    override = {"a": {"b": 5}, "l": [9]}
    assert deep_merge(base, override) == {"a": {"b": 5, "c": 2}, "l": [9]}
    # inputs are not mutated
    assert base == {"a": {"b": 1, "c": 2}, "l": [1]}


def test_deep_merge_none_overrides():
    assert deep_merge({"logo": {"enabled": True}}, {"logo": None}) == {"logo": None}


def test_diff_roundtrip():
    base = default_style()
    target = copy.deepcopy(base)
    target["subtitle"]["size"] = 80
    target["hook"]["duration"] = "full"
    diff = style_diff(base, target)
    assert diff == {"subtitle": {"size": 80}, "hook": {"duration": "full"}}
    assert deep_merge(base, diff) == target


def test_diff_identical_is_empty():
    assert style_diff(default_style(), default_style()) == {}


def test_effective_style_applies_override():
    eff = effective_style(default_style(), {"subtitle": {"size": 80}})
    assert eff["subtitle"]["size"] == 80
    assert eff["subtitle"]["outline_width"] == 8


def test_effective_style_rejects_bad_color():
    with pytest.raises(AppError) as exc:
        effective_style(default_style(), {"subtitle": {"color": "red"}})
    assert exc.value.code == "VALIDATION"


def test_effective_style_rejects_unknown_key():
    with pytest.raises(AppError) as exc:
        effective_style(default_style(), {"subtitle": {"colour": "#FFFFFF"}})
    assert exc.value.code == "VALIDATION"


@pytest.mark.parametrize("n,ok", [(0, False), (5, False), (1, True), (4, True)])
def test_words_per_line_bounds(n, ok):
    patch = {"subtitle": {"words_per_line": n}}
    if ok:
        assert validate_style_patch(patch) == patch
    else:
        with pytest.raises(AppError) as exc:
            validate_style_patch(patch)
        assert exc.value.code == "VALIDATION"


def test_shared_schema_files_match_model():
    schema = json.loads((REPO_DIR / "shared" / "style_schema.json").read_text(encoding="utf-8"))
    default = json.loads((REPO_DIR / "shared" / "style_default.json").read_text(encoding="utf-8"))
    assert schema == Style.model_json_schema()
    assert default == default_style()
