"""Point d'accès du module embed au sein du package rag_cours."""

import sys
from pathlib import Path

# Permet d'importer directement depuis src/embed.py
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

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

__all__ = [
    "load_chunks_from_jsonl",
    "group_chunks_by_source",
    "compute_source_hash",
    "get_source_mtime",
    "EmbeddingState",
    "GeminiEmbedder",
    "sanitize_metadata",
    "ChromaDatabase",
    "run_indexing",
    "main",
]

if __name__ == "__main__":
    main()
