"""Configuration commune des tests."""

import pytest

_BACKEND_ENV_VARS = [
    "GENERATOR_BACKEND",
    "GEMINI_GENERATION_MODEL",
    "ANTHROPIC_MODEL",
    "OLLAMA_MODEL",
    "OLLAMA_HOST",
    "OLLAMA_TIMEOUT",
]


@pytest.fixture(autouse=True)
def isolate_backend_env(monkeypatch):
    """Empêche un .env local (ex: GENERATOR_BACKEND=local) d'influencer les tests."""
    for var in _BACKEND_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
