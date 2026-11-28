from typing import Protocol

import httpx


class LLMUnavailable(Exception):
    """Ollama is not running, unreachable, or the model is not pulled."""


class LLMTimeout(LLMUnavailable):
    """Ollama responded too slowly and a single request exceeded its timeout."""


class LLMClient(Protocol):
    def complete_json(self, prompt: str, schema: dict) -> str:
        """Raw model text for a prompt, constrained to a JSON schema."""
        ...


class OllamaClient:
    def __init__(self, url: str, model: str, timeout: float = 600.0,
                 transport: httpx.BaseTransport | None = None):
        self.model = model
        self._http = httpx.Client(base_url=url, timeout=timeout, transport=transport)

    def complete_json(self, prompt: str, schema: dict) -> str:
        body = {"model": self.model, "prompt": prompt, "format": schema, "stream": False, "think": False,
                "options": {"temperature": 0.3, "num_ctx": 8192}}
        try:
            r = self._http.post("/api/generate", json=body)
        except httpx.TimeoutException as e:
            raise LLMTimeout(f"Ollama melewati batas waktu respons: {e}") from e
        except httpx.TransportError as e:
            raise LLMUnavailable(f"Ollama tidak bisa dihubungi: {e}") from e
        if r.status_code == 404:
            raise LLMUnavailable(f"Model {self.model} belum diunduh (jalankan: ollama pull {self.model}).")
        if r.status_code >= 500:
            raise LLMUnavailable(f"Ollama error {r.status_code}: {r.text[:200]}")
        r.raise_for_status()
        try:
            return r.json().get("response", "")
        except ValueError:
            return ""  # treated by callers as an invalid reply and retried
