# RAG sur notes de cours Obsidian  

Projet de système RAG (Retrieval-Augmented Generation) sur les notes de cours Obsidian.

## Structure du projet

```text
rag-cours/
├── .python-version      # Version Python locale pyenv (3.12)
├── pyproject.toml       # Configuration du projet et dépendances uv
├── .gitignore           # Fichiers et dossiers ignorés par git
├── README.md
├── src/
│   └── rag_cours/       # Package Python (code source de l'application)
│       ├── __init__.py
│       ├── ingest.py    # Chunking et extraction des notes Obsidian
│       ├── embed.py     # Vectorisation Gemini et indexation ChromaDB
│       ├── query.py     # Recherche sémantique et pipeline RAG (CLI rag-query)
│       ├── generators/  # Backends de génération : base, gemini, anthropic, local
│       └── web/         # Interface web FastAPI (app.py + static/)
├── data/                # Données générées, chunks et bases ChromaDB
│   ├── chunks.jsonl
│   └── chromadb/
└── tests/               # Tests automatisés
    ├── __init__.py
    ├── test_basic.py
    ├── test_ingest.py
    ├── test_embed.py
    ├── test_query.py
    ├── test_generators.py
    └── test_web.py
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
# ou: uv run python -m rag_cours.ingest
```

### Indexation des embeddings (incrémentale)
Générer les embeddings avec `gemini-embedding-001` et indexer dans ChromaDB locale :

```bash
uv run rag-embed
# ou: uv run python -m rag_cours.embed
```

Pour forcer une réindexation complète de tous les chunks :
```bash
uv run rag-embed --force
```

### Interrogation RAG (query)
Poser une question sur vos cours avec recherche sémantique ChromaDB et génération par le backend choisi :

```bash
uv run rag-query "Qu'est-ce que le polymorphisme en Java ?"
# ou: uv run python -m rag_cours.query "Qu'est-ce que le polymorphisme en Java ?"
```

Options utiles :
- `--top-k` ou `-k` : Nombre de chunks pertinents à récupérer (défaut : 5)
- `--matiere` ou `-m` : Filtrer sur une matière spécifique (ex: `-m Java` ou `-m BD`)
- `--backend` ou `-b` : Backend de génération `gemini`, `anthropic` ou `local` (défaut : `GENERATOR_BACKEND`, sinon `gemini`)
- `--model` : Modèle de génération du backend (défaut : celui du backend, voir ci-dessous)
- `--json` : Sortie structurée au format JSON

Exemple avec filtre :
```bash
uv run rag-query "Comment modéliser une relation 1-N ?" -m BD -k 3
```

### Backends de génération
Le backend est choisi par `GENERATOR_BACKEND` dans `.env` (ou `--backend` en CLI) :

| Backend | Implémentation | Variables | Modèle par défaut |
|---|---|---|---|
| `gemini` | `generators/gemini.py` | `GEMINI_API_KEY`, `GEMINI_GENERATION_MODEL` | `gemini-3.8-flash` |
| `anthropic` | `generators/anthropic.py` | `ANTHROPIC_API_KEY`, `ANTHROPIC_MODEL` | `claude-opus-5-5` |
| `local` | `generators/local.py` (Ollama) | `OLLAMA_HOST`, `OLLAMA_MODEL`, `OLLAMA_TIMEOUT` | `qwen2.5:7b` |

La recherche sémantique utilise toujours les embeddings Gemini de l'index ChromaDB :
`GEMINI_API_KEY` reste nécessaire quel que soit le backend de génération.

```bash
uv run rag-query -b local "Qu'est-ce qu'une clé étrangère ?"
```

### Interface web locale
Lancer le serveur puis ouvrir http://127.0.0.1:8000 :

```bash
uv run rag-web            # options : --host, --port
```

- Question, filtre par matière, choix du backend (défaut : `GENERATOR_BACKEND`) et nombre de sources.
- Réponse rendue en markdown façon Obsidian (thème sombre, callouts `> [!type]` stylés et
  pliables, coloration du code, wikilinks, tags) ; chaque `[Source N]` renvoie vers sa source.
- Sources numérotées, chacune avec un lien `obsidian://open?vault=...&file=...` et son extrait.
  Le nom du vault se règle avec `OBSIDIAN_VAULT_NAME` (défaut : `Programming_travel`).
- API : `POST /api/ask`, `GET /api/config` (documentation sur `/api/docs`).
- Les bibliothèques de rendu (marked, DOMPurify, highlight.js) et les polices sont chargées
  depuis un CDN ; hors ligne, la réponse s'affiche en texte brut.

### Tests
Exécuter l'ensemble des tests automatisés :

```bash
uv run pytest
```


