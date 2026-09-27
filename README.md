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
│   └── rag_cours/
│       └── __init__.py
├── data/                # Données générées et chunks (ignorés par git)
│   └── .gitkeep
└── tests/               # Tests automatisés
    ├── __init__.py
    └── test_basic.py
```

## Démarrage rapide

### Prérequis
- [pyenv](https://github.com/pyenv/pyenv) avec Python 3.12 installé
- [uv](https://github.com/astral-sh/uv)

### Installation
Synchroniser l'environnement virtuel et installer le projet avec `uv` :

```bash
uv sync
```

### Exécution
Lancer le point d'entrée :

```bash
uv run rag-cours
```

### Tests
Exécuter les tests :

```bash
uv run pytest
```
