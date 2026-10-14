"""Tests unitaires des générateurs de réponse (gemini, anthropic, local) et de leur sélection."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import anthropic
import httpx
import httpx2
import pytest

from rag_cours.generators import available_backends, get_generator, resolve_backend
from rag_cours.generators.anthropic import AnthropicGenerator
from rag_cours.generators.base import SYSTEM_INSTRUCTIONS, Generator, build_rag_prompt, build_user_message
from rag_cours.generators.gemini import GeminiGenerator
from rag_cours.generators.local import LocalGenerator
from rag_cours.query import answer_query

CHUNKS = [
    {
        "texte": "Le polymorphisme permet de redéfinir une méthode.",
        "fichier_source": "10_Notes/Java/Polymorphisme.md",
        "titres_str": "POO > Polymorphisme",
        "matiere": "Java",
    }
]


class FakeGenerator(Generator):
    name = "fake"
    model = "fake-1"

    def __init__(self):
        self.calls = []

    def generate(self, question, chunks):
        self.calls.append((question, chunks))
        return "  Réponse factice [Source 1].  "


# --- Prompt partagé -------------------------------------------------------


def test_build_user_message_contains_sources_and_question():
    msg = build_user_message("Qu'est-ce que le polymorphisme ?", CHUNKS)
    assert "[Source 1]" in msg
    assert "10_Notes/Java/Polymorphisme.md > POO > Polymorphisme" in msg
    assert msg.rstrip().endswith("=== RÉPONSE EN FRANÇAIS ===")
    assert SYSTEM_INSTRUCTIONS not in msg


def test_build_rag_prompt_is_system_plus_user_message():
    assert build_rag_prompt("Q ?", CHUNKS) == f"{SYSTEM_INSTRUCTIONS}\n{build_user_message('Q ?', CHUNKS)}"


# --- Sélection du backend -------------------------------------------------


def test_available_backends():
    assert available_backends() == ["gemini", "anthropic", "local"]


def test_resolve_backend_defaults_to_gemini():
    assert resolve_backend() == "gemini"


def test_resolve_backend_from_env(monkeypatch):
    monkeypatch.setenv("GENERATOR_BACKEND", " Local ")
    assert resolve_backend() == "local"


def test_resolve_backend_argument_overrides_env(monkeypatch):
    monkeypatch.setenv("GENERATOR_BACKEND", "local")
    assert resolve_backend("anthropic") == "anthropic"


def test_resolve_backend_unknown():
    with pytest.raises(ValueError, match="GENERATOR_BACKEND"):
        resolve_backend("openai")


def test_get_generator_from_env(monkeypatch):
    monkeypatch.setenv("GENERATOR_BACKEND", "local")
    gen = get_generator(model="mistral:7b")
    assert isinstance(gen, LocalGenerator)
    assert gen.model == "mistral:7b"


def test_get_generator_passes_kwargs():
    gen = get_generator("anthropic", client=MagicMock(), model="claude-sonnet-5-5")
    assert isinstance(gen, AnthropicGenerator)
    assert gen.describe() == "anthropic (claude-sonnet-5-5)"


# --- Gemini ---------------------------------------------------------------


def test_gemini_generator_sends_full_prompt():
    client = MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(text="Réponse Gemini")
    gen = GeminiGenerator(client=client)

    assert gen.generate("Q ?", CHUNKS) == "Réponse Gemini"
    kwargs = client.models.generate_content.call_args.kwargs
    assert kwargs["model"] == "gemini-3.8-flash"
    assert kwargs["contents"] == build_rag_prompt("Q ?", CHUNKS)


def test_gemini_generator_model_from_env(monkeypatch):
    monkeypatch.setenv("GEMINI_GENERATION_MODEL", "gemini-3.5-flash")
    assert GeminiGenerator(client=MagicMock()).model == "gemini-3.5-flash"


def test_gemini_generator_missing_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        GeminiGenerator()


# --- Anthropic ------------------------------------------------------------


def _claude_response(text="Réponse Claude [Source 1].", stop_reason="end_turn"):
    return SimpleNamespace(
        stop_reason=stop_reason,
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
    )


def test_anthropic_generator_request_shape():
    client = MagicMock()
    client.beta.messages.create.return_value = _claude_response()
    gen = AnthropicGenerator(client=client)

    assert gen.generate("Q ?", CHUNKS) == "Réponse Claude [Source 1]."
    kwargs = client.beta.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["system"] == SYSTEM_INSTRUCTIONS
    assert kwargs["messages"] == [{"role": "user", "content": build_user_message("Q ?", CHUNKS)}]
    assert kwargs["fallbacks"] == "default"
    assert kwargs["betas"] == ["server-side-fallback-2026-07-01"]


def test_anthropic_generator_no_fallback_for_other_models(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-4-5")
    client = MagicMock()
    client.beta.messages.create.return_value = _claude_response()
    AnthropicGenerator(client=client).generate("Q ?", CHUNKS)

    kwargs = client.beta.messages.create.call_args.kwargs
    assert kwargs["model"] == "claude-haiku-4-5"
    assert "fallbacks" not in kwargs and "betas" not in kwargs


def test_anthropic_generator_refusal():
    client = MagicMock()
    client.beta.messages.create.return_value = _claude_response(text="", stop_reason="refusal")
    with pytest.raises(RuntimeError, match="refusé"):
        AnthropicGenerator(client=client).generate("Q ?", CHUNKS)


def test_anthropic_generator_auth_error():
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    error = anthropic.AuthenticationError(
        "invalid x-api-key", response=httpx2.Response(401, request=request), body=None
    )
    client = MagicMock()
    client.beta.messages.create.side_effect = error
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY"):
        AnthropicGenerator(client=client).generate("Q ?", CHUNKS)


# --- Local (Ollama) -------------------------------------------------------


def _ollama_client(handler):
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_local_generator_request_shape():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        seen["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"role": "assistant", "content": "Réponse locale"}})

    gen = LocalGenerator(client=_ollama_client(handler), host="http://ollama:11434/")
    assert gen.generate("Q ?", CHUNKS) == "Réponse locale"
    assert seen["url"] == "http://ollama:11434/api/chat"
    assert seen["body"]["model"] == "qwen2.5:7b"
    assert seen["body"]["stream"] is False
    assert seen["body"]["messages"][0] == {"role": "system", "content": SYSTEM_INSTRUCTIONS}
    assert seen["body"]["messages"][1]["content"] == build_user_message("Q ?", CHUNKS)


def test_local_generator_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:3b")
    monkeypatch.setenv("OLLAMA_HOST", "http://gpu-box:11434")
    gen = LocalGenerator(client=MagicMock())
    assert (gen.model, gen.host) == ("llama3.2:3b", "http://gpu-box:11434")


def test_local_generator_model_missing():
    client = _ollama_client(lambda r: httpx.Response(404, json={"error": "model not found"}))
    with pytest.raises(RuntimeError, match="ollama pull qwen2.5:7b"):
        LocalGenerator(client=client).generate("Q ?", CHUNKS)


def test_local_generator_unreachable():
    def handler(request):
        raise httpx.ConnectError("connexion refusée", request=request)

    with pytest.raises(RuntimeError, match="ollama serve"):
        LocalGenerator(client=_ollama_client(handler)).generate("Q ?", CHUNKS)


# --- Intégration dans answer_query ----------------------------------------


def _empty_collection():
    collection = MagicMock()
    collection.count.return_value = 0
    return collection


def test_answer_query_uses_injected_generator():
    gen = FakeGenerator()
    result = answer_query("Q ?", client=MagicMock(), collection=_empty_collection(), generator=gen)

    assert result["reponse"] == "Réponse factice [Source 1]."
    assert (result["backend"], result["modele"]) == ("fake", "fake-1")
    assert gen.calls == [("Q ?", [])]


def test_answer_query_selects_backend_from_env(monkeypatch):
    monkeypatch.setenv("GENERATOR_BACKEND", "local")
    requested = {}

    def fake_get_generator(backend, **kwargs):
        requested["backend"], requested["kwargs"] = backend, kwargs
        return FakeGenerator()

    monkeypatch.setattr("rag_cours.query.get_generator", fake_get_generator)
    answer_query("Q ?", client=MagicMock(), collection=_empty_collection(), generation_model="qwen3:8b")
    assert requested == {"backend": "local", "kwargs": {"model": "qwen3:8b"}}


def test_answer_query_gemini_reuses_retrieval_client():
    client = MagicMock()
    client.models.generate_content.return_value = SimpleNamespace(text="Réponse Gemini")
    result = answer_query("Q ?", client=client, collection=_empty_collection())

    assert result["backend"] == "gemini"
    assert result["reponse"] == "Réponse Gemini"
    client.models.generate_content.assert_called_once()
