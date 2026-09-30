"""Tests unitaires pour le module d'interrogation RAG (query.py)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from query import (
    answer_query,
    build_rag_prompt,
    embed_query,
    format_cli_output,
    generate_rag_answer,
    get_gemini_client,
    main,
    retrieve_relevant_chunks,
)


@pytest.fixture
def mock_gemini_client():
    """Client Gemini simulé pour les embeddings et la génération."""
    mock_client = MagicMock()

    # Simulation embedding
    mock_emb_res = MagicMock()
    emb_obj = MagicMock()
    emb_obj.values = [0.1, 0.2, 0.3]
    mock_emb_res.embeddings = [emb_obj]
    mock_client.models.embed_content.return_value = mock_emb_res

    # Simulation génération
    mock_gen_res = MagicMock()
    mock_gen_res.text = "Le polymorphisme permet à différentes classes de répondre au même message [Source 1]."
    mock_client.models.generate_content.return_value = mock_gen_res

    return mock_client


@pytest.fixture
def populated_chroma_collection(tmp_path):
    """Collection ChromaDB avec quelques documents de test."""
    import chromadb

    client = chromadb.PersistentClient(path=str(tmp_path / "test_db"))
    coll = client.get_or_create_collection(
        name="test_notes",
        metadata={"hnsw:space": "cosine"},
    )
    coll.add(
        ids=["id1", "id2"],
        documents=[
            "Le polymorphisme est un concept clé de la POO.",
            "Les listes en Python sont dynamiques.",
        ],
        embeddings=[[0.1, 0.2, 0.3], [0.8, 0.9, 0.1]],
        metadatas=[
            {
                "fichier_source": "10_Notes/Java/Polymorphisme.md",
                "titres_str": "Java > POO > Polymorphisme",
                "matiere": "Java",
                "tags_str": "poo, java",
            },
            {
                "fichier_source": "10_Notes/Python/Listes.md",
                "titres_str": "Python > Structures",
                "matiere": "Python",
                "tags_str": "python",
            },
        ],
    )
    return coll


def test_embed_query_empty():
    client = MagicMock()
    with pytest.raises(ValueError, match="ne peut pas être vide"):
        embed_query(client, "   ")


def test_embed_query_success(mock_gemini_client):
    vec = embed_query(mock_gemini_client, "Qu'est-ce que le polymorphisme ?")
    assert vec == [0.1, 0.2, 0.3]
    mock_gemini_client.models.embed_content.assert_called_once()


def test_retrieve_relevant_chunks_empty_collection(mock_gemini_client, tmp_path):
    import chromadb

    client = chromadb.PersistentClient(path=str(tmp_path / "empty_db"))
    coll = client.get_or_create_collection("empty")
    res = retrieve_relevant_chunks(mock_gemini_client, coll, "Test question")
    assert res == []


def test_retrieve_relevant_chunks_populated(mock_gemini_client, populated_chroma_collection):
    chunks = retrieve_relevant_chunks(
        client=mock_gemini_client,
        collection=populated_chroma_collection,
        question="Qu'est-ce que le polymorphisme ?",
        top_k=2,
    )
    assert len(chunks) == 2
    assert chunks[0]["fichier_source"] == "10_Notes/Java/Polymorphisme.md"
    assert chunks[0]["matiere"] == "Java"
    assert "Polymorphisme" in chunks[0]["titres_str"]


def test_retrieve_relevant_chunks_with_matiere_filter(
    mock_gemini_client, populated_chroma_collection
):
    chunks = retrieve_relevant_chunks(
        client=mock_gemini_client,
        collection=populated_chroma_collection,
        question="Structures de données",
        top_k=2,
        matiere="Python",
    )
    assert len(chunks) == 1
    assert chunks[0]["matiere"] == "Python"
    assert chunks[0]["fichier_source"] == "10_Notes/Python/Listes.md"


def test_build_rag_prompt():
    chunks = [
        {
            "fichier_source": "10_Notes/Java/Polymorphisme.md",
            "titres_str": "Java > POO",
            "matiere": "Java",
            "texte": "Contenu du cours sur le polymorphisme.",
        }
    ]
    prompt = build_rag_prompt("Qu'est-ce que le polymorphisme ?", chunks)
    assert "Tu es un assistant pédagogique" in prompt
    assert "[Source 1]" in prompt
    assert "10_Notes/Java/Polymorphisme.md" in prompt
    assert "Contenu du cours sur le polymorphisme." in prompt
    assert "Qu'est-ce que le polymorphisme ?" in prompt


def test_generate_rag_answer(mock_gemini_client):
    ans = generate_rag_answer(mock_gemini_client, "Un prompt test")
    assert "polymorphisme" in ans.lower()
    mock_gemini_client.models.generate_content.assert_called_once()


def test_answer_query_pipeline(mock_gemini_client, populated_chroma_collection):
    result = answer_query(
        question="Comment fonctionne le polymorphisme ?",
        top_k=2,
        client=mock_gemini_client,
        collection=populated_chroma_collection,
    )

    assert result["question"] == "Comment fonctionne le polymorphisme ?"
    assert "polymorphisme" in result["reponse"].lower()
    assert len(result["chunks"]) == 2
    assert len(result["sources"]) == 2
    assert result["sources"][0]["fichier"] == "10_Notes/Java/Polymorphisme.md"
    assert result["sources"][0]["section"] == "Java > POO > Polymorphisme"


def test_format_cli_output():
    result = {
        "question": "Ma question ?",
        "reponse": "Voici la réponse détaillée.",
        "sources": [
            {
                "index": 1,
                "fichier": "10_Notes/Java/Intro.md",
                "section": "Syntaxe",
                "matiere": "Java",
            }
        ],
    }
    output = format_cli_output(result)
    assert "QUESTION : Ma question ?" in output
    assert "Voici la réponse détaillée." in output
    assert "10_Notes/Java/Intro.md → Syntaxe" in output


def test_main_cli_help():
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
