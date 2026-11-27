"""Style templates (spec 4.6) and the project-style change that flags rendered clips for re-render."""
from sqlalchemy import Engine
from sqlmodel import Session, select

from app.db.engine import session_scope
from app.db.tables import Clip, Project, StyleTemplate, utcnow
from app.domain.style import deep_merge, default_style, effective_style
from app.services._common import domain_errors, err, fetch

DEFAULT_NAME = "Default"


def full_style(style: dict) -> dict:
    """Validate a complete style (missing fields take their defaults); VALIDATION 422 on error."""
    with domain_errors("style"):
        if not isinstance(style, dict):
            raise err("VALIDATION", "Gaya tidak valid: harus berupa objek.")
        return effective_style(style, {})


def _clean_name(session: Session, name: str, exclude_id: str | None = None) -> str:
    if not isinstance(name, str) or not name.strip():
        raise err("VALIDATION", "Nama template tidak boleh kosong.")
    name = name.strip()
    clash = session.exec(select(StyleTemplate).where(StyleTemplate.name == name)).first()
    if clash is not None and clash.id != exclude_id:
        raise err("INVALID_STATE", f"Template bernama '{name}' sudah ada.")
    return name


def default_template(session: Session) -> StyleTemplate | None:
    """The template marked default (the oldest one if none is marked)."""
    rows = session.exec(select(StyleTemplate).order_by(StyleTemplate.created_at)).all()
    return next((t for t in rows if t.is_default), rows[0] if rows else None)


def ensure_default_template(engine: Engine) -> StyleTemplate:
    """Create the "Default" template from `default_style()` when no template exists; return the default."""
    with session_scope(engine) as s:
        current = default_template(s)
        if current is not None:
            return current
        tpl = StyleTemplate(name=DEFAULT_NAME, is_default=True, style=default_style())
        s.add(tpl)
        return tpl


def list_templates(engine: Engine) -> list[StyleTemplate]:
    with session_scope(engine) as s:
        return list(s.exec(select(StyleTemplate).order_by(StyleTemplate.created_at)).all())


def create_template(engine: Engine, name: str, style: dict) -> StyleTemplate:
    full = full_style(style)
    with session_scope(engine) as s:
        tpl = StyleTemplate(name=_clean_name(s, name), style=full)
        s.add(tpl)
        return tpl


def update_template(engine: Engine, template_id: str, *, name: str | None = None,
                    style: dict | None = None) -> StyleTemplate:
    with session_scope(engine) as s:
        tpl = fetch(s, StyleTemplate, template_id, "Template")
        if name is not None:
            tpl.name = _clean_name(s, name, exclude_id=tpl.id)
        if style is not None:
            tpl.style = full_style(style)
        tpl.updated_at = utcnow()
        s.add(tpl)
        return tpl


def delete_template(engine: Engine, template_id: str) -> None:
    with session_scope(engine) as s:
        tpl = fetch(s, StyleTemplate, template_id, "Template")
        if tpl.is_default:
            raise err("INVALID_STATE", "Template default tidak bisa dihapus; jadikan template lain default dulu.")
        s.delete(tpl)


def set_default(engine: Engine, template_id: str) -> StyleTemplate:
    with session_scope(engine) as s:
        chosen = fetch(s, StyleTemplate, template_id, "Template")
        for tpl in s.exec(select(StyleTemplate)).all():
            if tpl.is_default != (tpl.id == chosen.id):
                tpl.is_default = tpl.id == chosen.id
                tpl.updated_at = utcnow()
                s.add(tpl)
        return chosen


def clip_base_style(session: Session, clip: Clip) -> dict:
    """The style a clip's override applies to: its project's, or its own full style (compilation)."""
    if clip.project_id is not None:
        return fetch(session, Project, clip.project_id, "Project").style
    return clip.style or default_style()


def save_clip_as_template(engine: Engine, clip_id: str, name: str) -> StyleTemplate:
    with session_scope(engine) as s:
        clip = fetch(s, Clip, clip_id, "Klip")
        with domain_errors("style"):
            style = effective_style(clip_base_style(s, clip), clip.style_override)
        tpl = StyleTemplate(name=_clean_name(s, name), style=style)
        s.add(tpl)
        return tpl


def apply_project_style(session: Session, project: Project, style_patch: dict) -> None:
    """Deep-merge ``style_patch`` onto the project's style, validate, save in ``session`` (not committed) and
    flag `rendered` clips whose effective style changed with `needs_rerender`."""
    if not isinstance(style_patch, dict):
        raise err("VALIDATION", "Gaya tidak valid: harus berupa objek.")
    old = project.style or {}
    full = full_style(deep_merge(old, style_patch))
    rendered = session.exec(select(Clip).where(Clip.project_id == project.id, Clip.status == "rendered")).all()
    for clip in rendered:
        with domain_errors("style"):
            changed = effective_style(old, clip.style_override) != effective_style(full, clip.style_override)
        if changed and not clip.needs_rerender:
            clip.needs_rerender = True
            clip.updated_at = utcnow()
            session.add(clip)
    project.style = full
    project.updated_at = utcnow()
    session.add(project)


def set_project_style(engine: Engine, project_id: str, style_patch: dict) -> Project:
    """Change the project's style (a full style or a partial patch; unspecified fields are kept)."""
    with session_scope(engine) as s:
        project = fetch(s, Project, project_id, "Project")
        apply_project_style(s, project, style_patch)
        return project
