# Projet : RAG sur notes de cours Obsidian (ISIA — INSA Rouen)

## Vault
Organisation Johnny Decimal :
Chemin vers le vault : "/run/media/selasmsi/Windows/Programming_travel/"
- 00_MOCs/           → notes d'index (une par domaine : Java MOC, Python MOC, BD MOC...)
- 10_Notes/<Matière>/ → notes de cours (cœur du corpus RAG)
- 11_TPs/<Matière>/   → notes de TP (à inclure aussi)
- 20_Projets/         → projets persos, à EXCLURE du corpus

Nom du vault pour les URI obsidian:// : probablement "Programming_travel" 
(nom du dossier racine) — à confirmer dans Obsidian (Paramètres > À propos) 
avant de générer les liens.

## Corpus à indexer
Tout ce qui est sous 10_Notes/ et 11_TPs/. La matière = nom du sous-dossier 
direct (ex: 10_Notes/Java/ → matiere="Java"). 
Ne PAS indexer 00_MOCs/ (ce sont des index, pas du contenu) ni 20_Projets/.

## Format des notes (pas de frontmatter YAML)
Une note commence en général par un wikilink (souvent vers elle-même ou une 
note liée) puis une ligne `Tags: #tag1 #tag2 ...`. Le corps utilise :
- des titres ## (sections) et ### (sous-sections)
- des blocs de code ```Java / ```Python
- des tableaux markdown
- des wikilinks [[Nom de la note]] vers d'autres notes
- des images en ![[Images/nom.png]]
- des callouts Obsidian : > [!success]+ Titre / [!tip] / [!warning] / 
  [!example] / [!note] / [!code], suivis de lignes commençant par >

## Règles de découpage (chunking)
- Découper par titres ## ; garder les ### comme sous-sections dans le même 
  chunk quand c'est possible.
- Ne JAMAIS couper au milieu d'un bloc ```code``` ni d'un callout `> [!...]` 
  (un callout peut faire plusieurs lignes, toutes préfixées par >).
- Extraire les tags depuis la ligne "Tags:" et les éventuels #hashtags 
  inline, en ignorant ceux qui apparaissent dans un bloc de code.
- Extraire la matière depuis le chemin du fichier (dossier direct sous 
  10_Notes/ ou 11_TPs/).
- Remplacer les embeds d'image ![[...]] par un simple repère "[image]" — 
  ne pas essayer de les indexer pour le MVP.
- Conserver les wikilinks [[...]] comme métadonnée "liens sortants" du 
  chunk (utile plus tard pour enrichir le retrieval, pas critique pour 
  le MVP).

## Architecture cible (multi-agents, permanent)
Le projet doit rester adaptable à plusieurs agents de génération (Gemini, 
Claude/Anthropic, un modèle local open source via Ollama), sélectionnés par 
variable d'environnement, pas en dur dans le code :
- `src/rag_cours/generators/base.py` : interface commune 
  `generate(question, chunks) -> réponse`.
- `src/rag_cours/generators/gemini.py`, `anthropic.py`, `local.py` : 
  implémentations concrètes.
- Sélection via `GENERATOR_BACKEND=gemini|anthropic|local`.
- Les embeddings restent en Gemini uniquement (pas de sélecteur 
  `EMBEDDING_BACKEND` pour l'instant). Si on en ajoute un, réutiliser le 
  pattern de sélection des générateurs. Changer de modèle d'embedding 
  impose de réindexer ChromaDB : `GEMINI_API_KEY` reste donc requise pour 
  le retrieval, quel que soit `GENERATOR_BACKEND`.

## Contraintes
- Ne jamais écrire dans le dossier du vault : lecture seule.
- Tout le code du projet est dans le package `src/rag_cours/` (exposé par 
  pyproject.toml, scripts `rag-ingest`, `rag-embed`, `rag-query`). Les 
  tests importent depuis `rag_cours.*`. Ne pas recréer de modules à la 
  racine de `src/`.
- Python 3.12 via pyenv, dépendances gérées avec uv (pyproject.toml).
- Les tests existants (`tests/test_ingest.py`, `test_embed.py`, 
  `test_query.py`, `test_basic.py`) doivent continuer à passer ; en ajouter 
  pour toute nouvelle implémentation de générateur.

## Git
- Ne jamais travailler directement sur `main` : tout le travail de l'agent 
  se fait sur une branche dédiée `agent/claude-code` (la créer si elle 
  n'existe pas : `git checkout -b agent/claude-code`).
- Après chaque modification significative qui passe les tests (un fichier 
  créé, une fonctionnalité terminée), faire un commit avec un message clair 
  et descriptif, puis `git push` sur cette branche.
- Ne jamais push sur `main`, ne jamais force-push, ne jamais réécrire 
  l'historique d'une branche déjà poussée.
- Si une autre branche `agent/...` existe déjà (travail fait via Antigravity 
  CLI), ne pas la modifier : rester sur `agent/claude-code`.
  
## Avancement
- Ingestion (`ingest.py`) : fonctionnelle, produit `data/chunks.jsonl`.
- Embeddings (`embed.py`) : fonctionnel avec ChromaDB (`data/chromadb/`).
- Génération (`query.py`) : fonctionnelle avec l'API Gemini, citations 
  fichier + section.
- Tests unitaires en place pour ingestion, embeddings et query.
- `.env` / `.env.example` déjà configurés pour les clés API.

## Objectif du jour
Implémenter l'abstraction multi-agents décrite dans "Architecture cible" 
(generators/base.py + gemini/anthropic/local) sans casser query.py existant, 
puis construire une interface web locale avec FastAPI : poser une question, 
afficher la réponse en rendant le markdown (callouts Obsidian avec leur 
style visuel), lister les sources en liens cliquables 
`obsidian://open?vault=<nom_vault>&file=<chemin_encodé>`. Thème sombre, 
typographie proche d'Obsidian. Garder query.py utilisable en CLI.