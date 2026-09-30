"""Module d'interrogation RAG : recherche sémantique ChromaDB et génération via le backend choisi."""

import argparse
import json
import logging
import os
import socket
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Privilégier IPv4 pour éviter les timeouts réseau lorsque l'IPv6 système n'a pas de route sortante
_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if family in (0, socket.AF_UNSPEC):
        family = socket.AF_INET
    return _orig_getaddrinfo(host, port, family, type, proto, flags)


socket.getaddrinfo = _ipv4_getaddrinfo

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from google import genai
from google.genai import types
import chromadb

from rag_cours.generators import Generator, available_backends, get_generator, resolve_backend
from rag_cours.generators.base import build_rag_prompt
from rag_cours.generators.gemini import GeminiGenerator, generate_rag_answer, get_gemini_client

logger = logging.getLogger("rag_cours.query")


def embed_query(
    client: genai.Client,
    query_text: str,
    model_name: str = "gemini-embedding-001",
) -> List[float]:
    """Génère l'embedding d'une requête avec la tâche RETRIEVAL_QUERY."""
    if not query_text.strip():
        raise ValueError("La requête ne peut pas être vide.")

    response = client.models.embed_content(
        model=model_name,
        contents=query_text,
        config=types.EmbedContentConfig(
            task_type="RETRIEVAL_QUERY",
        ),
    )
    if not response.embeddings:
        raise RuntimeError("Aucun embedding retourné par Gemini pour la requête.")
    return response.embeddings[0].values


def retrieve_relevant_chunks(
    client: genai.Client,
    collection: chromadb.Collection,
    question: str,
    top_k: int = 5,
    matiere: Optional[str] = None,
    embedding_model: str = "gemini-embedding-001",
) -> List[Dict[str, Any]]:
    """Récupère les k chunks les plus pertinents depuis ChromaDB pour une question donnée."""
    if collection.count() == 0:
        logger.warning("La collection ChromaDB '%s' est vide.", collection.name)
        return []

    # Génération du vecteur de la question
    query_vector = embed_query(
        client=client,
        query_text=question,
        model_name=embedding_model,
    )

    where_filter = {"matiere": matiere} if matiere else None

    # Interrogation de ChromaDB
    results = collection.query(
        query_embeddings=[query_vector],
        n_results=min(top_k, collection.count()),
        where=where_filter,
    )

    chunks: List[Dict[str, Any]] = []
    if not results or not results["documents"] or not results["documents"][0]:
        return chunks

    docs = results["documents"][0]
    metas = results["metadatas"][0] if results["metadatas"] else [{}] * len(docs)
    distances = results["distances"][0] if results.get("distances") else [0.0] * len(docs)
    ids = results["ids"][0] if results.get("ids") else [""] * len(docs)

    for doc_text, meta, dist, chunk_id in zip(docs, metas, distances, ids):
        chunks.append({
            "id": chunk_id,
            "texte": doc_text,
            "fichier_source": meta.get("fichier_source", "Inconnu"),
            "titres_str": meta.get("titres_str", ""),
            "matiere": meta.get("matiere", "Inconnue"),
            "tags_str": meta.get("tags_str", ""),
            "wikilinks_str": meta.get("wikilinks_str", ""),
            "distance": dist,
        })

    return chunks


def answer_query(
    question: str,
    top_k: int = 5,
    matiere: Optional[str] = None,
    generation_model: Optional[str] = None,
    embedding_model: str = "gemini-embedding-001",
    db_dir: Path = Path("data/chromadb"),
    collection_name: str = "notes_cours",
    api_key: Optional[str] = None,
    client: Optional[genai.Client] = None,
    collection: Optional[chromadb.Collection] = None,
    backend: Optional[str] = None,
    generator: Optional[Generator] = None,
) -> Dict[str, Any]:
    """Exécute le pipeline RAG complet pour une question.

    Le retrieval utilise toujours les embeddings Gemini (ceux de l'index ChromaDB).
    La génération passe par `generator` s'il est fourni, sinon par le backend
    `backend` / GENERATOR_BACKEND (gemini par défaut), avec le modèle
    `generation_model` ou celui par défaut du backend.
    """
    if client is None:
        client = get_gemini_client(api_key=api_key)

    if collection is None:
        chroma_client = chromadb.PersistentClient(path=str(db_dir))
        collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    # 1. Récupération des chunks pertinents
    chunks = retrieve_relevant_chunks(
        client=client,
        collection=collection,
        question=question,
        top_k=top_k,
        matiere=matiere,
        embedding_model=embedding_model,
    )

    # 2. Génération de la réponse avec le backend sélectionné
    if generator is None:
        backend_name = resolve_backend(backend)
        if backend_name == "gemini":
            # Réutilise le client Gemini déjà créé pour le retrieval
            generator = GeminiGenerator(model=generation_model, client=client)
        else:
            generator = get_generator(backend_name, model=generation_model)

    reponse = generator.generate(question, chunks)

    # 3. Préparation de la liste des sources uniques citées
    sources_citees = []
    seen_sources = set()
    for idx, c in enumerate(chunks, start=1):
        source_key = (c["fichier_source"], c.get("titres_str", ""))
        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources_citees.append({
                "index": idx,
                "fichier": c["fichier_source"],
                "section": c.get("titres_str") or "Général",
                "matiere": c.get("matiere", "Inconnue"),
                "distance": c.get("distance", 0.0),
            })

    return {
        "question": question,
        "reponse": reponse.strip(),
        "chunks": chunks,
        "sources": sources_citees,
        "matiere_filtre": matiere,
        "top_k": top_k,
        "backend": generator.name,
        "modele": generator.model,
    }


def format_cli_output(result: Dict[str, Any]) -> str:
    """Met en forme la réponse et les citations de sources pour l'affichage console."""
    lines: List[str] = []
    lines.append("\n" + "=" * 60)
    lines.append(f"QUESTION : {result['question']}")
    lines.append("=" * 60 + "\n")

    lines.append(result["reponse"])
    if result.get("backend"):
        modele = f" ({result['modele']})" if result.get("modele") else ""
        lines.append(f"\n🤖 Généré par : {result['backend']}{modele}")
    lines.append("\n" + "-" * 60)
    lines.append("📚 NOTES SOURCES CITÉES (Fichier + Section) :")
    lines.append("-" * 60)

    if not result["sources"]:
        lines.append("  (Aucune source trouvée dans la base ChromaDB)")
    else:
        for s in result["sources"]:
            lines.append(
                f"  [{s['index']}] {s['fichier']} → {s['section']} (Matière: {s['matiere']})"
            )

    lines.append("=" * 60 + "\n")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> None:
    """Point d'entrée CLI pour la commande query."""
    default_chroma_dir = "data/chromadb"
    default_collection = "notes_cours"
    default_embedding_model = "gemini-embedding-001"

    parser = argparse.ArgumentParser(
        description="Interroge le RAG de cours : recherche sémantique et réponse argumentée avec citations."
    )
    parser.add_argument(
        "question",
        type=str,
        nargs="?",
        default=None,
        help="Question posée sur les cours (ex: 'Qu'est-ce que le polymorphisme en Java ?')",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=5,
        help="Nombre de chunks les plus proches à récupérer dans ChromaDB (défaut : 5)",
    )
    parser.add_argument(
        "--matiere",
        "-m",
        type=str,
        default=None,
        help="Filtrer la recherche sur une matière spécifique (ex: Java, BD, Python...)",
    )
    parser.add_argument(
        "--db-dir",
        "-d",
        type=str,
        default=default_chroma_dir,
        help=f"Dossier de persistance ChromaDB (défaut : {default_chroma_dir})",
    )
    parser.add_argument(
        "--collection",
        "-c",
        type=str,
        default=default_collection,
        help=f"Nom de la collection ChromaDB (défaut : {default_collection})",
    )
    parser.add_argument(
        "--backend",
        "-b",
        type=str,
        choices=available_backends(),
        default=None,
        help="Backend de génération (défaut : GENERATOR_BACKEND, sinon gemini)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help=(
            "Modèle de génération du backend choisi (défaut : GEMINI_GENERATION_MODEL, "
            "ANTHROPIC_MODEL ou OLLAMA_MODEL selon le backend)"
        ),
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=default_embedding_model,
        help=f"Modèle d'embedding Gemini pour la question (défaut : {default_embedding_model})",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Afficher la sortie brute au format JSON.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Afficher les logs détaillés de récupération.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Récupération de la question (argument ou saisie interactive)
    question = args.question
    if not question:
        if sys.stdin.isatty():
            try:
                question = input("Entrez votre question sur les cours : ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nOpération annulée.")
                sys.exit(0)
        else:
            question = sys.stdin.read().strip()

    if not question:
        print("[ERREUR] Aucune question spécifiée.", file=sys.stderr)
        parser.print_help(file=sys.stderr)
        sys.exit(1)

    try:
        result = answer_query(
            question=question,
            top_k=args.top_k,
            matiere=args.matiere,
            generation_model=args.model,
            backend=args.backend,
            embedding_model=args.embedding_model,
            db_dir=Path(args.db_dir),
            collection_name=args.collection,
        )

        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        else:
            print(format_cli_output(result))

    except Exception as exc:
        print(f"\n[ERREUR] Une erreur est survenue lors de l'exécution RAG : {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
