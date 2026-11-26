import json

import httpx
import pytest

from app.providers.llm import LLMUnavailable, OllamaClient


def test_ollama_unavailable_raises():
    c = OllamaClient("http://127.0.0.1:9", "qwen3:4b", timeout=0.5)
    with pytest.raises(LLMUnavailable):
        c.complete_json("hi", {"type": "object"})


def test_ollama_request_body_and_response():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"response": '{"ok": true}'})

    schema = {"type": "object"}
    c = OllamaClient("http://ollama", "qwen3:4b", transport=httpx.MockTransport(handler))
    assert c.complete_json("halo", schema) == '{"ok": true}'
    b = seen["body"]
    assert seen["url"] == "http://ollama/api/generate"
    assert b["model"] == "qwen3:4b" and b["prompt"] == "halo" and b["format"] == schema
    assert b["stream"] is False and b["think"] is False
    assert b["options"] == {"temperature": 0.3, "num_ctx": 8192}


def test_ollama_missing_model_is_unavailable():
    c = OllamaClient("http://ollama", "nope:1b",
                     transport=httpx.MockTransport(lambda r: httpx.Response(404, json={"error": "model not found"})))
    with pytest.raises(LLMUnavailable):
        c.complete_json("x", {})


def test_ollama_server_error_is_unavailable():
    # review I6: "model requires more system memory" comes back as HTTP 500
    c = OllamaClient("http://ollama", "qwen3:8b",
                     transport=httpx.MockTransport(lambda r: httpx.Response(500, json={"error": "requires more system memory"})))
    with pytest.raises(LLMUnavailable) as e:
        c.complete_json("x", {})
    assert "memory" in str(e.value)


def test_ollama_non_json_body_returns_empty_text():
    c = OllamaClient("http://ollama", "qwen3:4b",
                     transport=httpx.MockTransport(lambda r: httpx.Response(200, text="<html>proxy</html>")))
    assert c.complete_json("x", {}) == ""
