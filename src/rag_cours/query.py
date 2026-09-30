"""Module d'interrogation RAG : recherche sémantique ChromaDB et génération Gemini."""

import argparse
import json
import logging
import os
import socket
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

# Privilégier IPv4 pour éviter les timeouts réseau lorsque l'IPv6 système n'a pas de route sortante
_orig_getaddrinfo = socket.getaddrinfo


def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    if family in (0, socket.AF_UNSPEC):
        family = socket.AF_INET
    return _orig_getaddrinfo(host, port, family, type, proto, flags)


socket.getaddrinfo = _ipv4_getaddrinfo

# Suppression du warning AFC verbeux du SDK
logging.getLogger("google_genai.models").setLevel(logging.ERROR)
logging.getLogger("google.genai.models").setLevel(logging.ERROR)

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from google import genai
from google.genai import types
import chromadb

logger = logging.getLogger("rag_cours.query")


def get_gemini_client(api_key: Optional[str] = None) -> genai.Client:
    """Initialise le client Google GenAI avec la clé API résolue."""
    resolved_key = (
        api_key
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
    )
    if not resolved_key:
        raise ValueError(
            "Clé API Gemini introuvable. Veuillez définir la variable d'environnement "
            "GEMINI_API_KEY (ou GOOGLE_API_KEY), ou créer un fichier .env contenant GEMINI_API_KEY=votre_clef."
        )
    return genai.Client(api_key=resolved_key)


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


def build_rag_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    """Construit un prompt enrichi combinant la question et les extraits de notes de cours."""
    system_instructions = (
        "Tu es un assistant pédagogique et tuteur pour les étudiants de l'INSA Rouen (filière ISIA).\n"
        "Ton objectif est de répondre en français aux questions posées sur les cours en t'appuyant rigoureusement "
        "sur les extraits de notes de cours fournis ci-dessous.\n\n"
        "Consignes impératives :\n"
        "1. Réponds de manière claire, structurée, didactique et directe (utilise des listes ou du markdown propre).\n"
        "2. Reste fidèle au contenu des notes de cours fournies. Si un extrait aborde un concept sous un angle précis "
        "ou avec des exemples donnés, reprends-les.\n"
        "3. Fais référence aux sources dans ta réponse sous la forme [Source 1], [Source 2], etc.\n"
        "4. Si les extraits fournis ne permettent pas de répondre entièrement à la question, dis-le clairement, "
        "puis donne les compléments utiles en précisant qu'il s'agit d'une extrapolation pédagogique.\n"
    )

    if not chunks:
        context_str = "Aucune note de cours correspondante n'a été trouvée dans la base vectorielle.\n"
    else:
        context_parts = []
        for idx, c in enumerate(chunks, start=1):
            titre_info = f" > {c['titres_str']}" if c.get("titres_str") else ""
            part = (
                f"[Source {idx}]\n"
                f"Fichier : {c['fichier_source']}{titre_info}\n"
                f"Matière : {c['matiere']}\n"
                f"Contenu :\n{c['texte']}\n"
            )
            context_parts.append(part)
        context_str = "\n".join(context_parts)

    prompt = (
        f"{system_instructions}\n"
        f"=== EXTRAITS DE COURS DISPONIBLES ===\n"
        f"{context_str}\n"
        f"=== QUESTION DE L'ÉTUDIANT ===\n"
        f"{question}\n\n"
        f"=== RÉPONSE EN FRANÇAIS ==="
    )
    return prompt


def generate_rag_answer(
    client: genai.Client,
    prompt: str,
    model_name: str = "gemini-3.8-flash",
    temperature: float = 0.2,
    max_retries: int = 3,
) -> str:
    """Appelle le modèle Gemini pour générer la réponse à partir du prompt RAG avec retries."""
    candidate_models = [model_name]
    for fallback in ["gemini-3.5-flash", "gemini-2.5-flash-lite"]:
        if fallback not in candidate_models:
            candidate_models.append(fallback)

    last_err: Optional[Exception] = None
    for current_model in candidate_models:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=current_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=temperature,
                    ),
                )
                return response.text or ""
            except Exception as err:
                last_err = err
                err_str = str(err)
                if "503" in err_str or "UNAVAILABLE" in err_str or "429" in err_str:
                    wait_time = 1.5 * (attempt + 1)
                    logger.warning(
                        "Modèle %s indisponible temporairement (tentative %d/%d). Réessai dans %.1fs...",
                        current_model,
                        attempt + 1,
                        max_retries,
                        wait_time,
                    )
                    time.sleep(wait_time)
                else:
                    break

    raise RuntimeError(f"Échec de la génération RAG : {last_err}")


def answer_query(
    question: str,
    top_k: int = 5,
    matiere: Optional[str] = None,
    generation_model: str = "gemini-3.8-flash",
    embedding_model: str = "gemini-embedding-001",
    db_dir: Path = Path("data/chromadb"),
    collection_name: str = "notes_cours",
    api_key: Optional[str] = None,
    client: Optional[genai.Client] = None,
    collection: Optional[chromadb.Collection] = None,
) -> Dict[str, Any]:
    """Exécute le pipeline RAG complet pour une question."""
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

    # 2. Construction du prompt
    prompt = build_rag_prompt(question=question, chunks=chunks)

    # 3. Génération de la réponse avec Gemini
    reponse = generate_rag_answer(
        client=client,
        prompt=prompt,
        model_name=generation_model,
    )

    # 4. Préparation de la liste des sources uniques citées
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
    }


def format_cli_output(result: Dict[str, Any]) -> str:
    """Met en forme la réponse et les citations de sources pour l'affichage console."""
    lines: List[str] = []
    lines.append("\n" + "=" * 60)
    lines.append(f"QUESTION : {result['question']}")
    lines.append("=" * 60 + "\n")

    lines.append(result["reponse"])
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
    default_generation_model = os.environ.get("GEMINI_GENERATION_MODEL", "gemini-3.8-flash")
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
        "--model",
        type=str,
        default=default_generation_model,
        help=f"Modèle Gemini pour la génération de réponse (défaut : {default_generation_model})",
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
