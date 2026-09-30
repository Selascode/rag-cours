# RAG sur notes de cours Obsidian (ISIA — INSA Rouen)

Projet de système RAG (Retrieval-Augmented Generation) sur les notes de cours Obsidian.

## Structure du projet

```text
rag-cours/
├── .python-version      # Version Python locale pyenv (3.12)
├── pyproject.toml       # Configuration du projet et dépendances uv
├── .gitignore           # Fichiers et dossiers ignorés par git
├── README.md
├── src/                 # Code source de l'application
│   ├── ingest.py        # Chunking et extraction des notes Obsidian
│   ├── embed.py         # Vectorisation Gemini et indexation ChromaDB
│   ├── query.py         # Recherche sémantique et génération RAG
│   └── rag_cours/
│       ├── __init__.py
│       ├── ingest.py
│       ├── embed.py
│       └── query.py
├── data/                # Données générées, chunks et bases ChromaDB
│   ├── chunks.jsonl
│   └── chromadb/
└── tests/               # Tests automatisés
    ├── __init__.py
    ├── test_basic.py
    ├── test_ingest.py
    └── test_embed.py
```

## Démarrage rapide

### Prérequis
- [pyenv](https://github.com/pyenv/pyenv) avec Python 3.12 installé
- [uv](https://github.com/astral-sh/uv)
- Une clé API Google Gemini (`GEMINI_API_KEY`)

### Configuration
Créer un fichier `.env` à la racine (voir `.env.example`) :

```bash
cp .env.example .env
# Renseigner votre clé GEMINI_API_KEY dans le fichier .env
```

### Installation
Synchroniser l'environnement virtuel et installer le projet avec `uv` :

```bash
uv sync
```

### Ingestion des notes (chunking)
Découper les notes de cours Obsidian en chunks JSONL :

```bash
uv run rag-ingest
# ou: uv run python src/ingest.py
```

### Indexation des embeddings (incrémentale)
Générer les embeddings avec `gemini-embedding-001` et indexer dans ChromaDB locale :

```bash
uv run rag-embed
# ou: uv run python src/embed.py
```

Pour forcer une réindexation complète de tous les chunks :
```bash
uv run rag-embed --force
```

### Interrogation RAG (query)
Poser une question sur vos cours avec recherche sémantique ChromaDB et génération Gemini :

```bash
uv run rag-query "Qu'est-ce que le polymorphisme en Java ?"
# ou: uv run python src/query.py "Qu'est-ce que le polymorphisme en Java ?"
```

Options utiles :
- `--top-k` ou `-k` : Nombre de chunks pertinents à récupérer (défaut : 5)
- `--matiere` ou `-m` : Filtrer sur une matière spécifique (ex: `-m Java` ou `-m BD`)
- `--model` : Modèle de génération Gemini (défaut : `gemini-3.8-flash`)
- `--json` : Sortie structurée au format JSON

Exemple avec filtre :
```bash
uv run rag-query "Comment modéliser une relation 1-N ?" -m BD -k 3
```

### Tests
Exécuter l'ensemble des tests automatisés :

```bash
uv run pytest
```


