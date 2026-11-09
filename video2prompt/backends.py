"""Model backends. Each one returns a dict that matches the requested JSON schema."""

from __future__ import annotations

import base64
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Protocol


class BackendError(RuntimeError):
    pass


class Backend(Protocol):
    model: str

    def generate(self, system: str, text: str, images: list[Path], schema: dict) -> dict: ...


class OllamaBackend:
    def __init__(self, model: str, host: str = "http://127.0.0.1:11434") -> None:
        self.model = model
        self.host = host.rstrip("/")

    def _chat(self, payload: dict) -> dict:
        request = urllib.request.Request(
            f"{self.host}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(request, timeout=900) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise BackendError(f"Ollama a répondu {exc.code} : {detail}") from exc
        except urllib.error.URLError as exc:
            raise BackendError(
                f"Ollama est injoignable sur {self.host}. Lance « ollama serve »."
            ) from exc

    def generate(self, system: str, text: str, images: list[Path], schema: dict) -> dict:
        user: dict = {"role": "user", "content": text}
        if images:
            user["images"] = [base64.b64encode(p.read_bytes()).decode("ascii") for p in images]
        payload = {
            "model": self.model,
            "stream": False,
            "think": False,
            "format": schema,
            "messages": [{"role": "system", "content": system}, user],
            "options": {"temperature": 0.3, "num_ctx": 16384, "num_predict": 4096},
        }

        # Small models occasionally stop mid-JSON, so allow one more attempt.
        last_error: Exception | str | None = None
        for _ in range(2):
            data = self._chat(payload)
            content = data.get("message", {}).get("content", "")
            try:
                return json.loads(content)
            except json.JSONDecodeError as exc:
                last_error = exc
                if data.get("done_reason") == "length":
                    last_error = "limite de tokens atteinte avant la fin de la réponse"
        raise BackendError(f"Réponse invalide de {self.model} : {last_error}")


class ClaudeBackend:
    def __init__(self, model: str) -> None:
        try:
            import anthropic
        except ImportError as exc:
            raise BackendError("Le SDK manque. Lance : pip install -e .[claude]") from exc
        self._anthropic = anthropic
        self.model = model
        self.client = anthropic.Anthropic()

    def generate(self, system: str, text: str, images: list[Path], schema: dict) -> dict:
        content: list[dict] = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": base64.standard_b64encode(p.read_bytes()).decode("ascii"),
                },
            }
            for p in images
        ]
        content.append({"type": "text", "text": text})

        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                output_config={
                    "effort": "high",
                    "format": {"type": "json_schema", "schema": schema},
                },
                system=system,
                messages=[{"role": "user", "content": content}],
            )
        except self._anthropic.AuthenticationError as exc:
            raise BackendError("Clé Claude invalide ou absente (ANTHROPIC_API_KEY).") from exc
        except self._anthropic.APIStatusError as exc:
            raise BackendError(f"Erreur Claude {exc.status_code} : {exc.message}") from exc
        except self._anthropic.APIConnectionError as exc:
            raise BackendError("Connexion à l'API Claude impossible.") from exc

        if response.stop_reason == "refusal":
            raise BackendError("Claude a refusé d'analyser ce contenu.")
        if response.stop_reason == "max_tokens":
            raise BackendError("Réponse de Claude tronquée (max_tokens atteint).")

        body = next((b.text for b in response.content if b.type == "text"), "")
        return json.loads(body)


def ollama_vision_models(host: str = "http://127.0.0.1:11434") -> list[str] | None:
    """Installed models that accept images, or None when Ollama can't be reached."""
    try:
        with urllib.request.urlopen(f"{host}/api/tags", timeout=3) as response:
            names = [m["name"] for m in json.load(response).get("models", [])]
        vision = []
        for name in names:
            request = urllib.request.Request(
                f"{host}/api/show",
                data=json.dumps({"model": name}).encode("utf-8"),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                if "vision" in json.load(response).get("capabilities", []):
                    vision.append(name)
        return vision
    except (urllib.error.URLError, TimeoutError):
        return None


def create_backend(name: str, model: str | None) -> Backend:
    if name == "ollama":
        # The instruct variant has no thinking mode: the plain tag ignores think=false
        # and can spend its whole token budget reasoning, returning an empty answer.
        return OllamaBackend(model or "qwen3-vl:8b-instruct")
    if name == "claude":
        return ClaudeBackend(model or "claude-opus-5-5")
    raise BackendError(f"Backend inconnu : {name}")
