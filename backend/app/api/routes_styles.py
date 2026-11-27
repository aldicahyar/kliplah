"""Style template routes (spec 4.6)."""
from fastapi import APIRouter, Depends, Response
from pydantic import BaseModel
from sqlalchemy import Engine

from app.api.deps import dump, get_engine_dep
from app.services import styles

router = APIRouter(prefix="/style-templates", tags=["styles"])


class TemplateCreate(BaseModel):
    name: str
    style: dict


class TemplateUpdate(BaseModel):
    name: str | None = None
    style: dict | None = None


@router.get("")
def list_templates(engine: Engine = Depends(get_engine_dep)) -> list[dict]:
    return [dump(t) for t in styles.list_templates(engine)]


@router.post("", status_code=201)
def create_template(body: TemplateCreate, engine: Engine = Depends(get_engine_dep)) -> dict:
    return dump(styles.create_template(engine, body.name, body.style))


@router.patch("/{template_id}")
def update_template(template_id: str, body: TemplateUpdate, engine: Engine = Depends(get_engine_dep)) -> dict:
    return dump(styles.update_template(engine, template_id, name=body.name, style=body.style))


@router.delete("/{template_id}", status_code=204)
def delete_template(template_id: str, engine: Engine = Depends(get_engine_dep)) -> Response:
    styles.delete_template(engine, template_id)
    return Response(status_code=204)


@router.post("/{template_id}/default")
def set_default(template_id: str, engine: Engine = Depends(get_engine_dep)) -> dict:
    return dump(styles.set_default(engine, template_id))
