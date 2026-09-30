"""Générateur de réponse via l'API Google Gemini."""

import logging
import os
import time
from typing import Any, Dict, List, Optional

from google import genai
from google.genai import types

from rag_cours.generators.base import Generator, build_rag_prompt

# Suppression du warning AFC verbeux du SDK
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
logging.getLogger("google.genai.models").setLevel(logging.ERROR)

logger = logging.getLogger("rag_cours.generators.gemini")

DEFAULT_MODEL = "gemini-3.8-flash"
FALLBACK_MODELS = ["gemini-3.5-flash", "gemini-2.5-flash-lite"]


def get_gemini_client(api_key: Optional[str] = None) -> genai.Client:
    """Initialise le client Google GenAI avec la clé API résolue."""
    resolved_key = (
        api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not resolved_key:
        raise ValueError(
            "Clé API Gemini introuvable. Veuillez définir la variable d'environnement "
            "GEMINI_API_KEY (ou GOOGLE_API_KEY), ou créer un fichier .env contenant GEMINI_API_KEY=votre_clef."
        )
    return genai.Client(api_key=resolved_key)


def generate_rag_answer(
    client: genai.Client,
    prompt: str,
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_retries: int = 3,
) -> str:
    """Appelle le modèle Gemini pour générer la réponse à partir du prompt RAG avec retries."""
    candidate_models = [model_name]
    for fallback in FALLBACK_MODELS:
        if fallback not in candidate_models:
            candidate_models.append(fallback)

    last_err: Optional[Exception] = None
    for current_model in candidate_models:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=current_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                    ),
                )
                return response.text or ""
            except Exception as err:
                last_err = err
                err_str = str(err)
                if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                    wait_time = 1.5 * (attempt + 1)
                    logger.warning(
                        "Modèle %s indisponible temporairement (tentative %d/%d). Réessai dans %.1fs...",
                        current_model,
                        attempt + 1,
                        max_retries,
                        wait_time,
                    )
                    time.sleep(wait_time)
                else:
                    break

    raise RuntimeError(f"Échec de la génération RAG : {last_err}")


class GeminiGenerator(Generator):
    """Génération via Gemini, avec repli sur des modèles plus légers en cas de surcharge."""

    name = "gemini"

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        client: Optional[genai.Client] = None,
        temperature: float = 0.2,
        max_retries: int = 3,
    ) -> None:
        self.model = model or os.environ.get("GEMINI_GENERATION_MODEL") or DEFAULT_MODEL
        self.client = client if client is not None else get_gemini_client(api_key=api_key)
        self.temperature = temperature
        self.max_retries = max_retries

    def generate(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        return generate_rag_answer(
            client=self.client,
            prompt=build_rag_prompt(question, chunks),
            model_name=self.model,
            temperature=self.temperature,
            max_retries=self.max_retries,
        )
