# Projet : RAG sur notes de cours Obsidian (ISIA — INSA Rouen)

## Vault
Organisation Johnny Decimal :
Chemin vers le vault :  "/run/media/selasmsi/Windows/Programming_travel/"
- 00_MOCs/           → notes d'index (une par domaine : Java MOC, Python MOC, BD MOC...)
- 10_Notes/<Matière>/ → notes de cours (cœur du corpus RAG)
- 11_TPs/<Matière>/   → notes de TP (à inclure aussi)
- 20_Projets/         → projets persos, à EXCLURE du corpus

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
  (un callout peut faire plusieurs lignes, toutes préfixées par `>`).
- Extraire les tags depuis la ligne "Tags:" et les éventuels #hashtags 
  inline, en ignorant ceux qui apparaissent dans un bloc de code.
- Extraire la matière depuis le chemin du fichier (dossier direct sous 
  10_Notes/ ou 11_TPs/).
- Remplacer les embeds d'image ![[...]] par un simple repère "[image]" — 
  ne pas essayer de les indexer pour le MVP.
- Conserver les wikilinks [[...]] comme métadonnée "liens sortants" du 
  chunk (utile plus tard pour enrichir le retrieval, pas critique pour 
  le MVP).

## Contraintes
- Ne jamais écrire dans le dossier du vault : lecture seule.
- Le code du projet vit dans ~/Workspace/rag-cours, séparé du vault.
- Python 3.12 via pyenv, dépendances gérées avec uv (pyproject.toml).

## Objectif du jour
	Écrire src/ingest.py : parcourt 10_Notes/ et 11_TPs/, extrait matière/tags, découpe par titres H2 sans casser les callouts, blocs de code ou tableaux, produit data/chunks.jsonl.