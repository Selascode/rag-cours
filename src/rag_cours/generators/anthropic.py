"""Générateur de réponse via l'API Claude (Anthropic)."""

import os
from typing import Any, Dict, List, Optional

import anthropic

from rag_cours.generators.base import SYSTEM_INSTRUCTIONS, Generator, build_user_message

DEFAULT_MODEL = "claude-opus-5-5"

# Modèles acceptant le repli serveur automatique en cas de refus (fallbacks="default")
_SERVER_FALLBACK_MODELS = {"claude-opus-5-5", "claude-opus-5", "claude-fable-5-1", "claude-sonnet-5-5"}
_SERVER_FALLBACK_BETA = "server-side-fallback-2026-07-01"


class AnthropicGenerator(Generator):
    """Génération via Claude. Identifiants résolus par le SDK (ANTHROPIC_API_KEY ou profil `ant auth login`)."""

    name = "anthropic"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[anthropic.Anthropic] = None,
        max_tokens: int = 16000,
    ) -> None:
        self.model = model or os.environ.get("ANTHROPIC_MODEL") or DEFAULT_MODEL
        self.client = client if client is not None else anthropic.Anthropic(api_key=api_key)
        self.max_tokens = max_tokens

    def generate(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        params: Dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM_INSTRUCTIONS,
            "messages": [{"role": "user", "content": build_user_message(question, chunks)}],
        }
        if self.model in _SERVER_FALLBACK_MODELS:
            params["betas"] = [_SERVER_FALLBACK_BETA]
            params["fallbacks"] = "default"

        try:
            response = self.client.beta.messages.create(**params)
        except anthropic.AuthenticationError as err:
            raise RuntimeError(
                "Authentification Anthropic refusée : vérifiez ANTHROPIC_API_KEY dans le fichier .env."
            ) from err
        except anthropic.NotFoundError as err:
            raise RuntimeError(f"Modèle Claude inconnu : {self.model} (voir ANTHROPIC_MODEL).") from err
        except anthropic.APIConnectionError as err:
            raise RuntimeError(f"API Anthropic injoignable : {err}") from err
        except anthropic.APIStatusError as err:
            raise RuntimeError(f"Échec de la génération Claude ({err.status_code}) : {err.message}") from err

        if response.stop_reason == "refusal":
            raise RuntimeError("Claude a refusé de répondre à cette question.")

        return "".join(block.text for block in response.content if block.type == "text")
