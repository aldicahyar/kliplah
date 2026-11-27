import pytest
from sqlmodel import Session

from app.core.config import Settings
from app.core.errors import AppError
from app.db.engine import get_engine, init_db
from app.services.settings_store import get_app_settings, put_app_settings


@pytest.fixture
def session(tmp_path):
    eng = get_engine(Settings(_env_file=None, data_dir=tmp_path))
    init_db(eng)
    with Session(eng, expire_on_commit=False) as s:
        yield s


def test_defaults(session):
    assert get_app_settings(session) == {"moment_source": "claude_manual", "max_clips": 8, "content_type": "general"}


def test_defaults_respect_injected_settings(session):
    custom = Settings(_env_file=None, data_dir=session.bind.url.database, default_moment_source="ollama")
    assert get_app_settings(session, settings=custom)["moment_source"] == "ollama"


def test_put_app_settings_persists_patch(session):
    out = put_app_settings(session, {"moment_source": "ollama", "max_clips": 5})
    assert out == {"moment_source": "ollama", "max_clips": 5, "content_type": "general"}
    assert get_app_settings(session) == out
    assert put_app_settings(session, {"content_type": "podcast"})["max_clips"] == 5


def test_put_app_settings_rejects_unknown_key(session):
    with pytest.raises(AppError) as e:
        put_app_settings(session, {"warna": "merah", "max_clips": 3})
    assert e.value.code == "VALIDATION"
    assert get_app_settings(session)["max_clips"] == 8  # nothing from a rejected patch is saved


@pytest.mark.parametrize("patch", [{"moment_source": "gpt"}, {"max_clips": 0}, {"max_clips": "8"},
                                   {"max_clips": True}, {"content_type": "kuliner"}])
def test_put_app_settings_rejects_invalid_values(session, patch):
    with pytest.raises(AppError) as e:
        put_app_settings(session, patch)
    assert e.value.code == "VALIDATION"
