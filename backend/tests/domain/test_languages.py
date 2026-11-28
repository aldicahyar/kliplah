import pytest

from app.core.errors import AppError
from app.domain.languages import (
    LANGUAGE_ALIASES,
    is_weak_local_translation,
    language_name,
    load_languages,
    normalize_language,
)

REQUIRED_CODES = {
    "id", "ms", "en", "ko", "ja", "zh", "th", "vi", "tl",
    "es", "fr", "de", "pt", "ru", "ar", "hi", "it", "nl", "tr",
}


def test_catalog_contains_required_languages_and_fields():
    langs = load_languages()
    assert REQUIRED_CODES <= set(langs)
    for code in REQUIRED_CODES:
        entry = langs[code]
        assert entry.name_id
        assert entry.name_en
        assert isinstance(entry.whisper, bool)
        assert isinstance(entry.weak_local_translation, bool)
    assert LANGUAGE_ALIASES == {"id": frozenset({"ms"})}


def test_normalize_language_variants():
    assert normalize_language("pt-BR") == "pt"
    assert normalize_language("zh_CN") == "zh"
    assert normalize_language("ko-orig") == "ko"
    assert normalize_language("KO") == "ko"
    assert normalize_language("pt-BR-orig") == "pt"
    assert normalize_language("  en_US  ") == "en"
    assert normalize_language("auto") is None
    assert normalize_language("AUTO") is None
    assert normalize_language("") is None
    assert normalize_language("   ") is None
    assert normalize_language(None) is None

    for bad in ("xx", "-orig", "id-"):
        with pytest.raises(AppError) as exc:
            normalize_language(bad)
        assert exc.value.code == "VALIDATION"
        assert exc.value.status == 422


def test_language_names_and_weak_flags():
    assert language_name("ko", "id") == "Korea"
    assert language_name("en") == "English"
    assert language_name("pt-BR", "en") == "Portuguese"

    for weak_code in ("ko", "ja", "zh", "th"):
        assert is_weak_local_translation(weak_code) is True

    for strong_code in ("id", "ms", "en", "vi", "tl", "es", "fr", "de", "pt"):
        assert is_weak_local_translation(strong_code) is False
