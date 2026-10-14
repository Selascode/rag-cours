"""Interface web locale du RAG : API FastAPI + page statique (thème sombre façon Obsidian)."""

import argparse
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import quote

import chromadb
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_cours.generators import available_backends, resolve_backend
from rag_cours.query import answer_query, get_gemini_client

logger = logging.getLogger("rag_cours.web")

STATIC_DIR = Path(__file__).resolve().parent / "static"
DEFAULT_VAULT_NAME = "Programming_travel"
DEFAULT_DB_DIR = "data/chromadb"
DEFAULT_COLLECTION = "notes_cours"


def get_vault_name() -> str:
    """Nom du vault tel qu'affiché par Obsidian (Paramètres > À propos)."""
    return os.environ.get("OBSIDIAN_VAULT_NAME") or DEFAULT_VAULT_NAME


def obsidian_url(fichier_source: str, vault_name: Optional[str] = None) -> str:
    """Construit un lien obsidian://open vers une note, chemin relatif au vault."""
    path = fichier_source[:-3] if fichier_source.endswith(".md") else fichier_source
    vault = vault_name or get_vault_name()
    return f"obsidian://open?vault={quote(vault, safe='')}&file={quote(path, safe='')}"


@lru_cache(maxsize=1)
def get_collection() -> chromadb.Collection:
    """Collection ChromaDB ouverte une seule fois pour toute la durée du serveur."""
    db_dir = os.environ.get("RAG_DB_DIR") or DEFAULT_DB_DIR
    client = chromadb.PersistentClient(path=db_dir)
    return client.get_or_create_collection(
        name=os.environ.get("RAG_COLLECTION") or DEFAULT_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


@lru_cache(maxsize=1)
def get_embedding_client():
    """Client Gemini réutilisé pour les embeddings des questions."""
    return get_gemini_client()


@lru_cache(maxsize=1)
def list_matieres() -> List[str]:
    """Matières présentes dans l'index, pour le filtre de l'interface."""
    metadatas = get_collection().get(include=["metadatas"]).get("metadatas") or []
    return sorted({m["matiere"] for m in metadatas if m and m.get("matiere")})


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    matiere: Optional[str] = None
    top_k: int = Field(default=5, ge=1, le=20)
    backend: Optional[str] = None
    model: Optional[str] = None


def build_response(result: Dict[str, Any]) -> Dict[str, Any]:
    """Réponse JSON pour la page : sources numérotées comme les [Source N] du prompt."""
    vault = get_vault_name()
    sources = []
    for idx, chunk in enumerate(result["chunks"], start=1):
        sources.append({
            "index": idx,
            "fichier": chunk["fichier_source"],
            "section": chunk.get("titres_str") or "",
            "matiere": chunk.get("matiere", ""),
            "distance": chunk.get("distance"),
            "texte": chunk.get("texte", ""),
            "obsidian_url": obsidian_url(chunk["fichier_source"], vault),
        })
    return {
        "question": result["question"],
        "reponse": result["reponse"],
        "backend": result.get("backend"),
        "modele": result.get("modele"),
        "sources": sources,
    }


app = FastAPI(title="RAG notes de cours", docs_url="/api/docs", redoc_url=None)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/config")
def config() -> Dict[str, Any]:
    try:
        matieres = list_matieres()
    except Exception as err:  # l'index peut être absent : la page reste utilisable
        logger.warning("Impossible de lister les matières : %s", err)
        matieres = []
    return {
        "backend": resolve_backend(),
        "backends": available_backends(),
        "vault": get_vault_name(),
        "matieres": matieres,
    }


@app.post("/api/ask")
def ask(req: AskRequest) -> Dict[str, Any]:
    # Route synchrone : FastAPI l'exécute dans un thread, le pipeline bloquant ne gèle pas le serveur
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="La question est vide.")
    try:
        backend = resolve_backend(req.backend)
        result = answer_query(
            question=question,
            top_k=req.top_k,
            matiere=req.matiere or None,
            generation_model=req.model or None,
            client=get_embedding_client(),
            collection=get_collection(),
            backend=backend,
        )
    except ValueError as err:
        raise HTTPException(status_code=400, detail=str(err)) from err
    except Exception as err:
        logger.exception("Échec du pipeline RAG")
        raise HTTPException(status_code=502, detail=str(err)) from err
    return build_response(result)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def main(argv: Optional[List[str]] = None) -> None:
    """Point d'entrée CLI : lance le serveur web local."""
    import uvicorn

    parser = argparse.ArgumentParser(description="Interface web locale du RAG de cours.")
    parser.add_argument("--host", default="127.0.0.1", help="Adresse d'écoute (défaut : 127.0.0.1)")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Port d'écoute (défaut : 8000)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    print(f"Interface RAG disponible sur http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
