"""Générateurs de réponse RAG interchangeables, sélectionnés par GENERATOR_BACKEND."""

import importlib
import os
from typing import Any, List, Optional

from rag_cours.generators.base import Generator, build_rag_prompt

DEFAULT_BACKEND = "gemini"

# Import paresseux : un SDK absent ne bloque que le backend qui en dépend
_BACKENDS = {
    "gemini": ("rag_cours.generators.gemini", "GeminiGenerator"),
    "anthropic": ("rag_cours.generators.anthropic", "AnthropicGenerator"),
    "local": ("rag_cours.generators.local", "LocalGenerator"),
}


def available_backends() -> List[str]:
    """Liste des noms de backends acceptés par GENERATOR_BACKEND."""
    return list(_BACKENDS)


def resolve_backend(backend: Optional[str] = None) -> str:
    """Nom du backend : argument explicite, sinon GENERATOR_BACKEND, sinon gemini."""
    name = (backend or os.environ.get("GENERATOR_BACKEND") or DEFAULT_BACKEND).strip().lower()
    if name not in _BACKENDS:
        raise ValueError(
            f"Backend de génération inconnu : '{name}'. "
            f"Valeurs possibles pour GENERATOR_BACKEND : {', '.join(_BACKENDS)}."
        )
    return name


def get_generator(backend: Optional[str] = None, **kwargs: Any) -> Generator:
    """Instancie le générateur du backend choisi ; kwargs est transmis au constructeur."""
    module_name, class_name = _BACKENDS[resolve_backend(backend)]
    generator_cls = getattr(importlib.import_module(module_name), class_name)
    return generator_cls(**kwargs)


__all__ = [
    "Generator",
    "build_rag_prompt",
    "available_backends",
    "resolve_backend",
    "get_generator",
]
