from app.core.config import Settings


def test_defaults(monkeypatch):
    for k in ["DATA_DIR", "WHISPER_MODEL", "OLLAMA_MODEL", "OLLAMA_URL"]:
        monkeypatch.delenv(k, raising=False)
    s = Settings(_env_file=None)
    assert s.whisper_model == "small"
    assert s.ollama_model == "qwen3:4b"
    assert s.ollama_url == "http://localhost:11434"
    assert s.max_video_height == 1080
