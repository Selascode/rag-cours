"""Tests unitaires pour le module d'ingestion et de chunking des notes Obsidian."""

import json
from pathlib import Path

import pytest

from ingest import (
    Block,
    chunk_document,
    extract_inline_tags,
    extract_note_tags,
    extract_wikilinks,
    ingest_vault,
    parse_blocks,
    replace_images,
)


def test_replace_images():
    text = "Voici une image Obsidian ![[Images/photo.png]] et une standard ![alt](http://site.com/pic.jpg)."
    replaced = replace_images(text)
    assert replaced == "Voici une image Obsidian [image] et une standard [image]."
    assert "![[" not in replaced
    assert "![" not in replaced


def test_extract_note_tags():
    # Format 1: Tags: #a #b
    text1 = "[[Note]]\nTags: #java #poo #syntaxe\n## Titre"
    assert extract_note_tags(text1) == ["java", "poo", "syntaxe"]

    # Format 2: **Tags** : #a #b
    text2 = "**Tags** : #python #CleanCode #PEP8\n---\n## Titre"
    assert extract_note_tags(text2) == ["python", "CleanCode", "PEP8"]

    # Format 3: tags: [html, css]
    text3 = "tags: [html, css, web]\n---\n## Titre"
    assert extract_note_tags(text3) == ["html", "css", "web"]

    # Unexpanded templater tag <% ... %> should be ignored
    text4 = "**Tags** : #<% tp.file.cursor(2) %> #syntaxe"
    assert extract_note_tags(text4) == ["syntaxe"]


def test_extract_inline_tags_ignoring_code():
    text = """
    Ceci est un texte avec #important et #a_retenir.
    
    ```python
    # Ceci est un commentaire Python, pas un tag #ignored
    val = 42 #autre_commentaire
    ```
    
    Et du code inline `#not_a_tag` avec #encore_un_tag valide.
    """
    tags = extract_inline_tags(text)
    assert "important" in tags
    assert "a_retenir" in tags
    assert "encore_un_tag" in tags
    assert "ignored" not in tags
    assert "autre_commentaire" not in tags
    assert "not_a_tag" not in tags


def test_extract_wikilinks():
    text = "Lien vers [[Java - Types et variables]] et [[Page Cible|Mon Alias]] et [image]."
    wikis = extract_wikilinks(text)
    assert wikis == ["Java - Types et variables", "Page Cible"]


def test_atomic_blocks_integrity():
    content = """## Section 1

Un paragraphe normal.

```python
def hello():
    # comment
    return "world"
```

> [!tip]+ Astuce Obsidian
> Première ligne d'astuce
> Deuxième ligne d'astuce

$$
E = mc^2
\\int x dx
$$

| Nom | Valeur |
| --- | --- |
| A | 1 |
| B | 2 |
"""
    blocks = parse_blocks(content)
    block_types = [b.type for b in blocks]
    assert block_types == ["heading", "text", "code", "callout", "latex", "table"]

    # Check that code block contains full function
    code_block = [b for b in blocks if b.type == "code"][0]
    assert "def hello():" in code_block.text
    assert 'return "world"' in code_block.text

    # Check callout
    callout_block = [b for b in blocks if b.type == "callout"][0]
    assert "[!tip]+" in callout_block.text
    assert "Deuxième ligne d'astuce" in callout_block.text

    # Check LaTeX
    latex_block = [b for b in blocks if b.type == "latex"][0]
    assert "\\int x dx" in latex_block.text

    # Check Table
    table_block = [b for b in blocks if b.type == "table"][0]
    assert "| Nom | Valeur |" in table_block.text
    assert "| B | 2 |" in table_block.text


def test_chunk_document_keeps_small_h3_and_splits_large():
    # Small H2 with H3s: should stay in 1 chunk
    small_doc = """## 1. Introduction
Un texte court.
### 1.1 Sous-partie A
Contenu A.
### 1.2 Sous-partie B
Contenu B.
"""
    chunks = chunk_document(small_doc, max_chunk_chars=1000)
    assert len(chunks) == 1
    assert "Sous-partie A" in chunks[0]["texte"]
    assert "Sous-partie B" in chunks[0]["texte"]
    assert chunks[0]["chemin_titres"] == ["1. Introduction"]

    # Large H2 exceeding max_chunk_chars: should split across H3s
    large_part_a = "Texte A " * 50
    large_part_b = "Texte B " * 50
    large_doc = f"""## 2. Grande Section
### 2.1 Partie A
{large_part_a}
### 2.2 Partie B
{large_part_b}
"""
    large_chunks = chunk_document(large_doc, max_chunk_chars=200)
    assert len(large_chunks) >= 2
    # Verify that H3 titles are in chemin_titres
    chemins = [c["chemin_titres"] for c in large_chunks]
    assert any("2.1 Partie A" in c for c in chemins)
    assert any("2.2 Partie B" in c for c in chemins)


def test_chunk_document_never_cuts_code_or_callouts():
    long_code = "print('hello')\n" * 40
    doc = f"""## Section avec grand code
```python
{long_code}
```
"""
    chunks = chunk_document(doc, max_chunk_chars=200)
    # The code block must not be broken even if it is > max_chunk_chars
    for c in chunks:
        if "```python" in c["texte"]:
            assert c["texte"].count("```python") == 1
            assert c["texte"].endswith("```") or "```\n" in c["texte"]


def test_ingest_vault(tmp_path: Path):
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()

    # 1. 00_MOCs should be excluded
    mocs_dir = vault_dir / "00_MOCs"
    mocs_dir.mkdir()
    (mocs_dir / "Java MOC.md").write_text("# MOC Java\nIndex", encoding="utf-8")

    # 2. 20_Projets should be excluded
    projets_dir = vault_dir / "20_Projets"
    projets_dir.mkdir()
    (projets_dir / "Projet1.md").write_text("# Projet Perso\nContenu", encoding="utf-8")

    # 3. Root files should be excluded
    (vault_dir / "RootNote.md").write_text("# Note Racine", encoding="utf-8")

    # 4. 10_Notes/Java should be included
    notes_java = vault_dir / "10_Notes" / "Java"
    notes_java.mkdir(parents=True)
    (notes_java / "Classes.md").write_text(
        "[[Types]]\nTags: #java #poo\n## 1. Définition\nUne classe en Java.\n![[Images/img.png]]\n",
        encoding="utf-8",
    )

    # 5. 11_TPs/TW should be included
    tps_tw = vault_dir / "11_TPs" / "TW"
    tps_tw.mkdir(parents=True)
    (tps_tw / "TP1.md").write_text(
        "tags: [web, html]\n## Exercice 1\nCréer un tableau [[LienVersNote]].\n",
        encoding="utf-8",
    )

    output_file = tmp_path / "data" / "chunks.jsonl"
    chunks = ingest_vault(str(vault_dir), str(output_file))

    assert len(chunks) == 2
    assert output_file.exists()

    # Read back JSONL
    lines = output_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    chunk1 = json.loads(lines[0])
    chunk2 = json.loads(lines[1])

    # Check fields
    expected_fields = {"texte", "fichier_source", "chemin_titres", "matiere", "tags", "wikilinks"}
    assert set(chunk1.keys()) == expected_fields
    assert set(chunk2.keys()) == expected_fields

    # Check values
    assert chunk1["matiere"] == "Java"
    assert "java" in chunk1["tags"]
    assert "poo" in chunk1["tags"]
    assert "[image]" in chunk1["texte"]
    assert "![[" not in chunk1["texte"]

    assert chunk2["matiere"] == "TW"
    assert "web" in chunk2["tags"]
    assert "html" in chunk2["tags"]
    assert "LienVersNote" in chunk2["wikilinks"]
