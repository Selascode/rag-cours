"""Point d'accès du module ingest au sein du package rag_cours."""

import sys
from pathlib import Path

# Permet d'importer directement depuis src/ingest.py
src_dir = Path(__file__).resolve().parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from ingest import (
    Block,
    chunk_document,
    extract_inline_tags,
    extract_note_tags,
    extract_wikilinks,
    get_last_text_overlap,
    ingest_vault,
    main,
    parse_blocks,
    replace_images,
)

__all__ = [
    "replace_images",
    "extract_note_tags",
    "extract_inline_tags",
    "extract_wikilinks",
    "Block",
    "parse_blocks",
    "get_last_text_overlap",
    "chunk_document",
    "ingest_vault",
    "main",
]

if __name__ == "__main__":
    main()
