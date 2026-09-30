"""Tests de l'interface web FastAPI (pipeline RAG simulé, aucun appel réseau)."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from rag_cours.web import app as web

FAKE_RESULT = {
    "question": "Qu'est-ce que le polymorphisme ?",
    "reponse": "Le polymorphisme... [Source 1]",
    "chunks": [
        {
            "texte": "> [!tip] Astuce\n> Redéfinir une méthode.",
            "fichier_source": "10_Notes/Java/Java - Classes, Objets et Héritage.md",
            "titres_str": "Les polymorphismes",
            "matiere": "Java",
            "distance": 0.25,
        },
        {
            "texte": "Texte",
            "fichier_source": "11_TPs/BD/TP 1.md",
            "titres_str": "",
            "matiere": "BD",
            "distance": 0.4,
        },
    ],
    "sources": [],
    "backend": "local",
    "modele": "qwen2.5:7b",
}


@pytest.fixture
def client(monkeypatch):
    calls = []

    def fake_answer_query(**kwargs):
        calls.append(kwargs)
        return FAKE_RESULT

    monkeypatch.setattr(web, "answer_query", fake_answer_query)
    monkeypatch.setattr(web, "get_embedding_client", lambda: "gemini-client")
    monkeypatch.setattr(web, "get_collection", lambda: "collection")
    monkeypatch.setattr(web, "list_matieres", lambda: ["BD", "Java"])
    monkeypatch.delenv("OBSIDIAN_VAULT_NAME", raising=False)
    test_client = TestClient(web.app)
    test_client.calls = calls
    return test_client


def test_obsidian_url_encodes_path():
    url = web.obsidian_url("10_Notes/Java/Java - Classes, Objets et Héritage.md", "Programming_travel")
    assert url == (
        "obsidian://open?vault=Programming_travel"
        "&file=10_Notes%2FJava%2FJava%20-%20Classes%2C%20Objets%20et%20H%C3%A9ritage"
    )


def test_obsidian_url_vault_from_env(monkeypatch):
    monkeypatch.setenv("OBSIDIAN_VAULT_NAME", "Mon vault")
    assert web.obsidian_url("a.md").startswith("obsidian://open?vault=Mon%20vault&file=a")


def test_index_page(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "ask-form" in res.text


def test_static_assets(client):
    assert client.get("/static/app.js").status_code == 200
    assert "callout" in client.get("/static/style.css").text


def test_config(client, monkeypatch):
    monkeypatch.setenv("GENERATOR_BACKEND", "anthropic")
    data = client.get("/api/config").json()
    assert data == {
        "backend": "anthropic",
        "backends": ["gemini", "anthropic", "local"],
        "vault": "Programming_travel",
        "matieres": ["BD", "Java"],
    }


def test_config_without_index(client, monkeypatch):
    monkeypatch.setattr(web, "list_matieres", MagicMock(side_effect=RuntimeError("pas d'index")))
    assert client.get("/api/config").json()["matieres"] == []


def test_ask_returns_numbered_sources_with_obsidian_links(client):
    res = client.post("/api/ask", json={"question": "  Qu'est-ce que le polymorphisme ?  ", "top_k": 2})
    assert res.status_code == 200
    data = res.json()

    assert data["reponse"] == FAKE_RESULT["reponse"]
    assert (data["backend"], data["modele"]) == ("local", "qwen2.5:7b")
    assert [s["index"] for s in data["sources"]] == [1, 2]
    first = data["sources"][0]
    assert first["section"] == "Les polymorphismes"
    assert first["texte"].startswith("> [!tip]")
    assert first["obsidian_url"].endswith("file=10_Notes%2FJava%2FJava%20-%20Classes%2C%20Objets%20et%20H%C3%A9ritage")

    call = client.calls[0]
    assert call["question"] == "Qu'est-ce que le polymorphisme ?"
    assert call["top_k"] == 2
    assert call["backend"] == "gemini"
    assert call["matiere"] is None
    assert (call["client"], call["collection"]) == ("gemini-client", "collection")


def test_ask_forwards_backend_and_matiere(client):
    client.post("/api/ask", json={"question": "Q ?", "backend": "local", "matiere": "BD"})
    assert client.calls[0]["backend"] == "local"
    assert client.calls[0]["matiere"] == "BD"


def test_ask_unknown_backend(client):
    res = client.post("/api/ask", json={"question": "Q ?", "backend": "openai"})
    assert res.status_code == 400
    assert "GENERATOR_BACKEND" in res.json()["detail"]


def test_ask_blank_question(client):
    assert client.post("/api/ask", json={"question": "   "}).status_code == 400
    assert client.post("/api/ask", json={"question": ""}).status_code == 422


def test_ask_generation_failure(client, monkeypatch):
    monkeypatch.setattr(web, "answer_query", MagicMock(side_effect=RuntimeError("Ollama injoignable")))
    res = client.post("/api/ask", json={"question": "Q ?"})
    assert res.status_code == 502
    assert res.json()["detail"] == "Ollama injoignable"
