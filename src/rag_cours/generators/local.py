"""Générateur de réponse via un modèle local servi par Ollama."""

import os
from typing import Any, Dict, List, Optional

import httpx

from rag_cours.generators.base import SYSTEM_INSTRUCTIONS, Generator, build_user_message

DEFAULT_HOST = "http://localhost:11434"
DEFAULT_MODEL = "qwen2.5:7b"
DEFAULT_TIMEOUT = 600.0  # un modèle 7B sur CPU peut dépasser 4 min


class LocalGenerator(Generator):
    """Génération via l'API /api/chat d'Ollama (aucune donnée ne quitte la machine)."""

    name = "local"

    def __init__(
        self,
        model: Optional[str] = None,
        host: Optional[str] = None,
        client: Optional[httpx.Client] = None,
        temperature: float = 0.2,
        timeout: Optional[float] = None,
    ) -> None:
        self.model = model or os.environ.get("OLLAMA_MODEL") or DEFAULT_MODEL
        self.host = (host or os.environ.get("OLLAMA_HOST") or DEFAULT_HOST).rstrip("/")
        if timeout is None:
            timeout = float(os.environ.get("OLLAMA_TIMEOUT") or DEFAULT_TIMEOUT)
        self.client = client if client is not None else httpx.Client(timeout=timeout)
        self.temperature = temperature

    def generate(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_INSTRUCTIONS},
                {"role": "user", "content": build_user_message(question, chunks)},
            ],
            "stream": False,
            "options": {"temperature": self.temperature},
        }

        try:
            response = self.client.post(f"{self.host}/api/chat", json=payload)
        except httpx.ConnectError as err:
            raise RuntimeError(
                f"Ollama injoignable sur {self.host} : lancez `ollama serve` ou ajustez OLLAMA_HOST."
            ) from err
        except httpx.TimeoutException as err:
            raise RuntimeError(f"Ollama n'a pas répondu à temps (modèle {self.model}).") from err

        if response.status_code == 404:
            raise RuntimeError(
                f"Modèle Ollama introuvable : {self.model}. Installez-le avec `ollama pull {self.model}`."
            )
        if response.is_error:
            raise RuntimeError(f"Échec de la génération Ollama ({response.status_code}) : {response.text}")

        return response.json().get("message", {}).get("content", "")
