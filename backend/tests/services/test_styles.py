import pytest
from sqlmodel import Session

from app.core.errors import AppError
from app.db.tables import Clip, Project
from app.domain.style import deep_merge, default_style
from app.services import styles


def test_ensure_default_template_creates_once(engine):
    first = styles.ensure_default_template(engine)
    assert first.name == "Default" and first.is_default
    assert first.style == default_style()
    assert styles.ensure_default_template(engine).id == first.id
    assert len(styles.list_templates(engine)) == 1


def test_create_template_duplicate_name_refused(engine):
    styles.create_template(engine, "Podcast Kuning", default_style())
    with pytest.raises(AppError) as e:
        styles.create_template(engine, "Podcast Kuning", default_style())
    assert e.value.code == "INVALID_STATE" and e.value.status == 409


def test_create_template_invalid_style(engine):
    bad = deep_merge(default_style(), {"subtitle": {"color": "kuning"}})
    with pytest.raises(AppError) as e:
        styles.create_template(engine, "Rusak", bad)
    assert e.value.code == "VALIDATION" and e.value.status == 422


def test_delete_default_template_refused(engine):
    default = styles.ensure_default_template(engine)
    with pytest.raises(AppError) as e:
        styles.delete_template(engine, default.id)
    assert e.value.code == "INVALID_STATE" and e.value.status == 409
    other = styles.create_template(engine, "Lain", default_style())
    styles.delete_template(engine, other.id)
    assert [t.id for t in styles.list_templates(engine)] == [default.id]


def test_set_default_keeps_exactly_one(engine):
    default = styles.ensure_default_template(engine)
    other = styles.create_template(engine, "Lain", default_style())
    styles.set_default(engine, other.id)
    flags = {t.id: t.is_default for t in styles.list_templates(engine)}
    assert flags == {default.id: False, other.id: True}
    styles.delete_template(engine, default.id)  # no longer default -> allowed


def test_update_template_renames_and_validates(engine):
    t = styles.create_template(engine, "A", default_style())
    new_style = deep_merge(default_style(), {"subtitle": {"size": 120}})
    out = styles.update_template(engine, t.id, name="B", style=new_style)
    assert out.name == "B" and out.style["subtitle"]["size"] == 120
    with pytest.raises(AppError) as e:
        styles.update_template(engine, "tidak-ada", name="C")
    assert e.value.code == "NOT_FOUND" and e.value.status == 404


def test_save_clip_as_template_uses_effective_style(engine, make_clip):
    clip = make_clip(style_override={"subtitle": {"color": "#FFE14D", "size": 130}})
    t = styles.save_clip_as_template(engine, clip.id, "Dari Klip")
    assert t.name == "Dari Klip" and not t.is_default
    expected = deep_merge(default_style(), {"subtitle": {"color": "#FFE14D", "size": 130}})
    assert t.style == expected


def test_project_style_change_flags_only_affected_rendered_clips(engine, ready_project, make_clip):
    project, _ = ready_project
    plain = make_clip(status="rendered", rank=1)
    overrides_color = make_clip(status="rendered", rank=2, style_override={"subtitle": {"color": "#00FF00"}})
    proposed = make_clip(status="proposed", rank=3)

    new_style = deep_merge(default_style(), {"subtitle": {"color": "#FFE14D"}})
    out = styles.set_project_style(engine, project.id, new_style)
    assert out.style == new_style

    with Session(engine) as s:
        assert s.get(Project, project.id).style == new_style
        assert s.get(Clip, plain.id).needs_rerender is True
        assert s.get(Clip, overrides_color.id).needs_rerender is False  # its override hides the change
        assert s.get(Clip, proposed.id).needs_rerender is False  # not rendered yet


def test_project_style_change_without_effect_flags_nothing(engine, ready_project, make_clip):
    project, _ = ready_project
    clip = make_clip(status="rendered")
    styles.set_project_style(engine, project.id, default_style())
    with Session(engine) as s:
        assert s.get(Clip, clip.id).needs_rerender is False


def test_set_project_style_invalid(engine, ready_project):
    project, _ = ready_project
    with pytest.raises(AppError) as e:
        styles.set_project_style(engine, project.id, {"subtitle": {"size": -1}})
    assert e.value.code == "VALIDATION" and e.value.status == 422
