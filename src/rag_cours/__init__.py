"""Package principal pour le projet RAG sur les notes de cours Obsidian."""

from rag_cours.embed import run_indexing
from rag_cours.ingest import ingest_vault
from rag_cours.query import answer_query


def main() -> None:
    print("Hello from rag-cours!")


__all__ = ["main", "ingest_vault", "run_indexing", "answer_query"]

