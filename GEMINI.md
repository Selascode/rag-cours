# Projet : RAG sur notes de cours Obsidian (ISIA — INSA Rouen)

## Contexte
Vault Obsidian en français, organisé par matière (Cours/<Matière>/...), 
frontmatter YAML (matiere, code_ue, chapitre, tags), notes taguées 
`rag-source` = corpus à indexer. Découpage prévu par titres H2/H3.

## Contraintes
- Ne jamais écrire dans le dossier du vault : lecture seule.
- Le code du projet vit dans ~/Workspace/rag-cours, séparé du vault.
- Python 3.12 via pyenv, dépendances gérées avec uv (pyproject.toml).

## Objectif du jour
Jour 1 : scaffolder le projet Python (structure de dossiers, dépendances, 
environnement) et vérifier que tout s'installe correctement.