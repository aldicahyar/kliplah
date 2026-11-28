import os

import pytest
from sqlmodel import Session

from app.core.config import Settings
from app.core.errors import AppError
from app.db.engine import get_engine, init_db
from app.services.settings_store import get_app_settings, put_app_settings


def _default_threads() -> int:
    return max(1, (os.cpu_count() or 1) - 2)


@pytest.fixture
def session(tmp_path):
    eng = get_engine(Settings(_env_file=None, data_dir=tmp_path))
    init_db(eng)
    with Session(eng, expire_on_commit=False) as s:
        yield s


def test_defaults(session):
    assert get_app_settings(session) == {
        "moment_source": "claude_manual", "max_clips": 8, "content_type": "general",
        "cookies_browser": "", "encode_threads": _default_threads(), "range_max_height": 720,
    }


def test_defaults_respect_injected_settings(session):
    custom = Settings(_env_file=None, data_dir=session.bind.url.database, default_moment_source="ollama")
    assert get_app_settings(session, settings=custom)["moment_source"] == "ollama"


def test_put_app_settings_persists_patch(session):
    out = put_app_settings(session, {"moment_source": "ollama", "max_clips": 5})
    assert out == {
        "moment_source": "ollama", "max_clips": 5, "content_type": "general",
        "cookies_browser": "", "encode_threads": _default_threads(), "range_max_height": 720,
    }
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


def test_cookies_browser_setting_validated(session):
    assert put_app_settings(session, {"cookies_browser": "chrome"})["cookies_browser"] == "chrome"
    assert put_app_settings(session, {"cookies_browser": ""})["cookies_browser"] == ""
    for bad in ("safari", "Chrome", None, 1):
        with pytest.raises(AppError) as e:
            put_app_settings(session, {"cookies_browser": bad})
        assert e.value.code == "VALIDATION"


def test_encode_threads_and_range_max_height_settings_validated(session, monkeypatch):
    monkeypatch.setattr(os, "cpu_count", lambda: 8)
    assert get_app_settings(session)["encode_threads"] == 6
    assert get_app_settings(session)["range_max_height"] == 720
    assert put_app_settings(session, {"encode_threads": 1})["encode_threads"] == 1
    assert put_app_settings(session, {"encode_threads": 8})["encode_threads"] == 8
    for bad_threads in (0, -1, 9, "4", True, False, None, 2.5):
        with pytest.raises(AppError) as e:
            put_app_settings(session, {"encode_threads": bad_threads})
        assert e.value.code == "VALIDATION"

    monkeypatch.setattr(os, "cpu_count", lambda: 2)
    assert put_app_settings(session, {"encode_threads": 2})["encode_threads"] == 2
    # Reset back below 2 and verify minimum default when cpu_count is small
    monkeypatch.setattr(os, "cpu_count", lambda: 1)
    assert put_app_settings(session, {"encode_threads": 1})["encode_threads"] == 1

    for h in (480, 720, 1080):
        assert put_app_settings(session, {"range_max_height": h})["range_max_height"] == h
    for bad_h in (360, 1440, "720", True, None):
        with pytest.raises(AppError) as e:
            put_app_settings(session, {"range_max_height": bad_h})
        assert e.value.code == "VALIDATION"
