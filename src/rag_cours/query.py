"""Point d'accès du module query au sein du package rag_cours."""

import sys
from pathlib import Path

# Permet d'importer directement depuis src/query.py
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from query import (
    answer_query,
    build_rag_prompt,
    embed_query,
    format_cli_output,
    generate_rag_answer,
    get_gemini_client,
    main,
    retrieve_relevant_chunks,
)

__all__ = [
    "get_gemini_client",
    "embed_query",
    "retrieve_relevant_chunks",
    "build_rag_prompt",
    "generate_rag_answer",
    "answer_query",
    "format_cli_output",
    "main",
]

if __name__ == "__main__":
    main()
