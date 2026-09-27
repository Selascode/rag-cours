"""Package principal pour le projet RAG sur les notes de cours Obsidian."""

from rag_cours.ingest import ingest_vault


def main() -> None:
    print("Hello from rag-cours!")


__all__ = ["main", "ingest_vault"]
