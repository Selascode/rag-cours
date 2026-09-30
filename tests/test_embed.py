"""Tests unitaires pour le module de génération d'embeddings et indexation ChromaDB."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from embed import (
    ChromaDatabase,
    EmbeddingState,
    GeminiEmbedder,
    compute_source_hash,
    get_source_mtime,
    group_chunks_by_source,
    load_chunks_from_jsonl,
    main,
    run_indexing,
    sanitize_metadata,
)


@pytest.fixture
def sample_chunks():
    return [
        {
            "texte": "## Introduction à Java\nJava est orienté objet.",
            "fichier_source": "10_Notes/Java/Intro.md",
            "chemin_titres": ["Java", "Introduction à Java"],
            "matiere": "Java",
            "tags": ["java", "poo"],
            "wikilinks": ["[[Polymorphisme]]"],
        },
        {
            "texte": "## Syntaxe Java\nLes variables et types primitifs.",
            "fichier_source": "10_Notes/Java/Intro.md",
            "chemin_titres": ["Java", "Syntaxe Java"],
            "matiere": "Java",
            "tags": ["java", "syntaxe"],
            "wikilinks": [],
        },
        {
            "texte": "## List Comprehension en Python\n[x for x in range(10)]",
            "fichier_source": "10_Notes/Python/Comprehensions.md",
            "chemin_titres": ["Python", "List Comprehension"],
            "matiere": "Python",
            "tags": ["python"],
            "wikilinks": [],
        },
    ]


@pytest.fixture
def mock_embedder():
    """Crée un GeminiEmbedder avec un client simulé."""
    mock_client = MagicMock()

    def fake_embed(model, contents, config):
        res = MagicMock()
        # Générer un vecteur factice de 3 dimensions par texte
        embeddings = []
        for _ in contents:
            emb = MagicMock()
            emb.values = [0.1, 0.2, 0.3]
            embeddings.append(emb)
        res.embeddings = embeddings
        return res

    mock_client.models.embed_content.side_effect = fake_embed
    return GeminiEmbedder(client=mock_client)


def test_load_chunks_from_jsonl(tmp_path, sample_chunks):
    jsonl_file = tmp_path / "chunks.jsonl"
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for c in sample_chunks:
            f.write(json.dumps(c) + "\n")
        f.write("invalid json line\n")

    loaded = load_chunks_from_jsonl(jsonl_file)
    assert len(loaded) == 3
    assert loaded[0]["matiere"] == "Java"


def test_load_chunks_not_found():
    with pytest.raises(FileNotFoundError):
        load_chunks_from_jsonl(Path("/path/to/nonexistent/chunks.jsonl"))


def test_group_chunks_by_source(sample_chunks):
    grouped = group_chunks_by_source(sample_chunks)
    assert len(grouped) == 2
    assert "10_Notes/Java/Intro.md" in grouped
    assert len(grouped["10_Notes/Java/Intro.md"]) == 2
    assert len(grouped["10_Notes/Python/Comprehensions.md"]) == 1


def test_compute_source_hash_from_chunks(sample_chunks):
    chunks_java = [c for c in sample_chunks if c["fichier_source"] == "10_Notes/Java/Intro.md"]
    hash1 = compute_source_hash("10_Notes/Java/Intro.md", chunks_java)
    hash2 = compute_source_hash("10_Notes/Java/Intro.md", chunks_java)
    assert hash1 == hash2

    # Modification d'un texte -> le hash change
    modified_chunks = [dict(c) for c in chunks_java]
    modified_chunks[0]["texte"] = "Texte modifié"
    hash3 = compute_source_hash("10_Notes/Java/Intro.md", modified_chunks)
    assert hash1 != hash3


def test_compute_source_hash_from_disk_file(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    note_dir = vault / "10_Notes" / "Java"
    note_dir.mkdir(parents=True)
    note_file = note_dir / "Intro.md"
    note_file.write_text("Contenu original sur disque", encoding="utf-8")

    rel_path = "10_Notes/Java/Intro.md"
    hash_disk_1 = compute_source_hash(rel_path, [], vault_path=vault)
    assert isinstance(hash_disk_1, str) and len(hash_disk_1) == 64

    # Modification du fichier sur disque
    note_file.write_text("Contenu modifié sur disque", encoding="utf-8")
    hash_disk_2 = compute_source_hash(rel_path, [], vault_path=vault)
    assert hash_disk_1 != hash_disk_2

    # Vérification mtime
    mtime = get_source_mtime(rel_path, vault_path=vault)
    assert mtime is not None and mtime > 0


def test_embedding_state(tmp_path):
    state_file = tmp_path / "embed_state.json"
    state = EmbeddingState(state_file)

    assert state.is_changed("note.md", "hash123") is True
    state.update_file("note.md", "hash123", mtime=100.0, chunk_count=2)
    assert state.is_changed("note.md", "hash123", current_mtime=100.0) is False
    assert state.is_changed("note.md", "hash456", current_mtime=100.0) is True
    assert state.is_changed("note.md", "hash123", current_mtime=200.0) is True

    state.save()
    assert state_file.exists()

    # Recharger l'état
    new_state = EmbeddingState(state_file)
    assert "note.md" in new_state.entries
    assert new_state.entries["note.md"]["hash"] == "hash123"

    new_state.remove_file("note.md")
    assert "note.md" not in new_state.entries


def test_sanitize_metadata():
    chunk = {
        "fichier_source": "10_Notes/Test.md",
        "matiere": "Test",
        "chemin_titres": ["Titre"],
        "tags": [],  # Doit être géré pour éviter le ValueError de ChromaDB
        "wikilinks": ["[[AutreNote]]"],
        "texte": "12345",
    }
    meta = sanitize_metadata(chunk, chunk_index=1, file_hash="abc")

    assert meta["fichier_source"] == "10_Notes/Test.md"
    assert meta["matiere"] == "Test"
    assert meta["chunk_index"] == 1
    assert meta["file_hash"] == "abc"
    assert meta["char_count"] == 5
    assert meta["tags_str"] == ""
    assert "tags" not in meta  # Liste vide omise
    assert meta["wikilinks"] == ["[[AutreNote]]"]


def test_gemini_embedder_missing_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="Clé API Gemini introuvable"):
        GeminiEmbedder(api_key=None)


def test_chroma_database(tmp_path):
    chroma_dir = tmp_path / "chromadb"
    db = ChromaDatabase(persist_dir=chroma_dir, collection_name="test_col")

    chunks = [
        {"fichier_source": "file1.md", "matiere": "Java", "texte": "chunk 1"},
        {"fichier_source": "file1.md", "matiere": "Java", "texte": "chunk 2"},
    ]
    embeddings = [[0.1, 0.2], [0.3, 0.4]]

    db.add_chunks(chunks, embeddings, file_hash="hash1")
    assert db.count() == 2

    # Suppression ciblée des chunks d'un fichier source
    db.delete_source_chunks("file1.md")
    assert db.count() == 0


def test_run_indexing_incremental(tmp_path, sample_chunks, mock_embedder):
    jsonl_file = tmp_path / "chunks.jsonl"
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for c in sample_chunks:
            f.write(json.dumps(c) + "\n")

    chroma_dir = tmp_path / "chromadb"

    # --- 1er RUN : Indexation complète ---
    stats1 = run_indexing(
        input_file=jsonl_file,
        chroma_dir=chroma_dir,
        embedder=mock_embedder,
    )
    assert stats1["files_indexed"] == 2
    assert stats1["files_unchanged"] == 0
    assert stats1["chunks_embedded"] == 3
    assert stats1["total_in_db"] == 3

    # --- 2e RUN : Aucun changement -> 0 chunks embarqués ---
    mock_embedder.client.models.embed_content.reset_mock()
    stats2 = run_indexing(
        input_file=jsonl_file,
        chroma_dir=chroma_dir,
        embedder=mock_embedder,
    )
    assert stats2["files_indexed"] == 0
    assert stats2["files_unchanged"] == 2
    assert stats2["chunks_embedded"] == 0
    assert stats2["total_in_db"] == 3
    assert mock_embedder.client.models.embed_content.call_count == 0

    # --- 3e RUN : Modification d'un seul fichier ---
    modified_chunks = [dict(c) for c in sample_chunks]
    modified_chunks[0]["texte"] = "Introduction Java mise à jour !"
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for c in modified_chunks:
            f.write(json.dumps(c) + "\n")

    stats3 = run_indexing(
        input_file=jsonl_file,
        chroma_dir=chroma_dir,
        embedder=mock_embedder,
    )
    assert stats3["files_indexed"] == 1
    assert stats3["files_unchanged"] == 1
    assert stats3["chunks_embedded"] == 2  # Les 2 chunks de Intro.md
    assert stats3["total_in_db"] == 3

    # --- 4e RUN : Suppression d'un fichier du corpus ---
    only_python_chunks = [c for c in modified_chunks if c["fichier_source"].startswith("10_Notes/Python")]
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for c in only_python_chunks:
            f.write(json.dumps(c) + "\n")

    stats4 = run_indexing(
        input_file=jsonl_file,
        chroma_dir=chroma_dir,
        embedder=mock_embedder,
    )
    assert stats4["files_deleted"] == 1
    assert stats4["files_unchanged"] == 1
    assert stats4["chunks_embedded"] == 0
    assert stats4["total_in_db"] == 1  # Seul le chunk Python reste


def test_run_indexing_force_flag(tmp_path, sample_chunks, mock_embedder):
    jsonl_file = tmp_path / "chunks.jsonl"
    with open(jsonl_file, "w", encoding="utf-8") as f:
        for c in sample_chunks:
            f.write(json.dumps(c) + "\n")

    chroma_dir = tmp_path / "chromadb"

    # 1er run
    run_indexing(
        input_file=jsonl_file,
        chroma_dir=chroma_dir,
        embedder=mock_embedder,
    )

    # 2e run avec force=True
    stats_force = run_indexing(
        input_file=jsonl_file,
        chroma_dir=chroma_dir,
        embedder=mock_embedder,
        force=True,
    )
    assert stats_force["files_indexed"] == 2
    assert stats_force["chunks_embedded"] == 3


def test_main_cli_help(capsys):
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
