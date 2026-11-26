from scripts import check_setup


def test_collect_reports_missing_ffmpeg(monkeypatch):
    monkeypatch.setattr(check_setup.shutil, "which", lambda name: None)
    monkeypatch.setattr(check_setup, "_ollama_tags", lambda url: None)
    r = check_setup.collect()
    assert r["ffmpeg"] is False
    assert r["ffmpeg_has_libass"] is False
    assert r["ollama"] is False and r["ollama_model"] is False


def test_collect_model_present(monkeypatch):
    monkeypatch.setattr(check_setup, "_ollama_tags", lambda url: ["qwen3:4b", "x:1"])
    r = check_setup.collect()
    assert r["ollama"] is True and r["ollama_model"] is True
