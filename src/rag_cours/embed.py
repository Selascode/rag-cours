"""Module de génération des embeddings et indexation persistante dans ChromaDB."""

import argparse
import hashlib
import json
import logging
import os
import re
import socket
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
from google.genai import errors, types
import chromadb

# Configuration du logging
logger = logging.getLogger("rag_cours.embed")


def load_chunks_from_jsonl(file_path: Path) -> List[Dict[str, Any]]:
    """Charge l'ensemble des chunks depuis un fichier JSONL."""
    if not file_path.exists():
        raise FileNotFoundError(f"Fichier de chunks introuvable : {file_path}")

    chunks: List[Dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                chunk = json.loads(line)
                chunks.append(chunk)
            except json.JSONDecodeError as e:
                logger.warning("Ligne %d ignorée (JSON invalide) : %s", line_num, e)

    return chunks


def group_chunks_by_source(
    chunks: List[Dict[str, Any]],
) -> Dict[str, List[Dict[str, Any]]]:
    """Groupe les chunks par fichier source."""
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in chunks:
        fichier = c.get("fichier_source", "unknown")
        grouped[fichier].append(c)
    return dict(grouped)


def compute_source_hash(
    fichier_source: str,
    chunks: List[Dict[str, Any]],
    vault_path: Optional[Path] = None,
) -> str:
    """Calcule l'empreinte sha256 d'un fichier source.

    Si le fichier physique existe dans le vault, on hache son contenu direct.
    Sinon (ex: vault démonté), on hache le contenu cumulé de ses chunks JSONL.
    """
    if vault_path:
        full_path = vault_path / fichier_source
        if full_path.is_file():
            try:
                hasher = hashlib.sha256()
                with open(full_path, "rb") as f:
                    while chunk := f.read(65536):
                        hasher.update(chunk)
                return hasher.hexdigest()
            except OSError as err:
                logger.warning(
                    "Impossible de lire le fichier disque %s (%s). Utilisation du hachage des chunks.",
                    full_path,
                    err,
                )

    # Calcul alternatif basé sur les chunks
    hasher = hashlib.sha256()
    hasher.update(fichier_source.encode("utf-8"))
    for chunk in chunks:
        hasher.update(chunk.get("texte", "").encode("utf-8"))
        hasher.update(str(chunk.get("chemin_titres", [])).encode("utf-8"))
        hasher.update(str(chunk.get("matiere", "")).encode("utf-8"))
        hasher.update(str(chunk.get("tags", [])).encode("utf-8"))
    return hasher.hexdigest()


def get_source_mtime(
    fichier_source: str,
    vault_path: Optional[Path] = None,
) -> Optional[float]:
    """Récupère la date de modification du fichier source s'il est présent sur le disque."""
    if vault_path:
        full_path = vault_path / fichier_source
        if full_path.is_file():
            try:
                return os.path.getmtime(full_path)
            except OSError:
                return None
    return None


class EmbeddingState:
    """Gère l'état d'indexation incrémentale (empreintes et métadonnées des fichiers)."""

    def __init__(self, state_file: Path) -> None:
        self.state_file = state_file
        self.entries: Dict[str, Dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        """Charge l'état depuis le fichier JSON s'il existe."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.entries = data.get("files", {})
            except Exception as e:
                logger.warning("Impossible de charger le fichier d'état %s : %s", self.state_file, e)
                self.entries = {}
        else:
            self.entries = {}

    def save(self) -> None:
        """Sauvegarde l'état actuel dans le fichier JSON."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": 1,
            "last_updated": time.time(),
            "files": self.entries,
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def is_changed(
        self,
        fichier_source: str,
        current_hash: str,
        current_mtime: Optional[float] = None,
    ) -> bool:
        """Vérifie si un fichier a été modifié depuis la dernière indexation."""
        entry = self.entries.get(fichier_source)
        if not entry:
            return True

        # Si le hash a changé, le fichier a forcément changé
        if entry.get("hash") != current_hash:
            return True

        # Vérification optionnelle de date de modif si disponible
        if (
            current_mtime is not None
            and entry.get("mtime") is not None
            and entry.get("mtime") != current_mtime
        ):
            return True

        return False

    def update_file(
        self,
        fichier_source: str,
        content_hash: str,
        mtime: Optional[float],
        chunk_count: int,
    ) -> None:
        """Met à jour l'entrée d'un fichier source."""
        self.entries[fichier_source] = {
            "hash": content_hash,
            "mtime": mtime,
            "chunk_count": chunk_count,
            "indexed_at": time.time(),
        }

    def remove_file(self, fichier_source: str) -> None:
        """Supprime un fichier source de l'état."""
        self.entries.pop(fichier_source, None)


class GeminiEmbedder:
    """Générateur d'embeddings sémantiques via l'API Google Gemini (modèle multilingue)."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-embedding-001",
        client: Optional[genai.Client] = None,
        max_retries: int = 5,
        retry_delay_base: float = 2.0,
    ) -> None:
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_delay_base = retry_delay_base

        if client is not None:
            self.client = client
        else:
            resolved_key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            if not resolved_key:
                raise ValueError(
                    "Clé API Gemini introuvable. Veuillez définir la variable d'environnement "
                    "GEMINI_API_KEY (ou GOOGLE_API_KEY), ou créer un fichier .env contenant GEMINI_API_KEY=votre_clef."
                )
            self.client = genai.Client(api_key=resolved_key)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """Génère les vecteurs d'un lot de textes avec retries exponentiels."""
        if not texts:
            return []

        # Remplacement des textes vides par un espace pour éviter un rejet d'API
        sanitized_texts = [t if t.strip() else " " for t in texts]

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.models.embed_content(
                    model=self.model_name,
                    contents=sanitized_texts,
                    config=types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                    ),
                )
                if not response.embeddings:
                    raise ValueError("Réponse Gemini vide : aucun embedding retourné.")

                return [emb.values for emb in response.embeddings]

            except Exception as err:
                last_error = err
                err_str = str(err)
                wait_time = self.retry_delay_base * (2**attempt)
                m = re.search(r"retry in (\d+(?:\.\d+)?)s", err_str, re.IGNORECASE)
                if m:
                    wait_time = max(wait_time, float(m.group(1)) + 1.0)
                elif "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    wait_time = max(wait_time, 20.0)

                logger.warning(
                    "Erreur appel Gemini embed_content (tentative %d/%d) : %s. Nouvelle tentative dans %.1fs...",
                    attempt + 1,
                    self.max_retries,
                    err,
                    wait_time,
                )
                time.sleep(wait_time)

        raise RuntimeError(
            f"Échec de génération des embeddings après {self.max_retries} tentatives : {last_error}"
        )


def sanitize_metadata(
    chunk: Dict[str, Any],
    chunk_index: int,
    file_hash: str,
) -> Dict[str, Any]:
    """Prépare et nettoie les métadonnées pour ChromaDB.

    ChromaDB accepte str, int, float, bool et list non-vide.
    Les listes vides génèrent une erreur, elles sont donc omises en format liste
    et toujours fournies sous forme de chaîne de caractères lisible.
    """
    fichier_source = str(chunk.get("fichier_source", ""))
    matiere = str(chunk.get("matiere", ""))
    chemin_titres = chunk.get("chemin_titres") or []
    tags = chunk.get("tags") or []
    wikilinks = chunk.get("wikilinks") or []
    text_content = chunk.get("texte", "")

    meta: Dict[str, Any] = {
        "fichier_source": fichier_source,
        "matiere": matiere,
        "chunk_index": int(chunk_index),
        "file_hash": file_hash,
        "char_count": len(text_content),
        "titres_str": " > ".join(str(t) for t in chemin_titres),
        "tags_str": ", ".join(str(t) for t in tags),
        "wikilinks_str": ", ".join(str(w) for w in wikilinks),
    }

    # Ajouter les listes non-vides (ChromaDB 1.5+ accepte les listes de chaînes non-vides)
    if chemin_titres:
        meta["chemin_titres"] = [str(t) for t in chemin_titres]
    if tags:
        meta["tags"] = [str(t) for t in tags]
    if wikilinks:
        meta["wikilinks"] = [str(w) for w in wikilinks]

    return meta


class ChromaDatabase:
    """Gestionnaire de la base vectorielle persistante ChromaDB locale."""

    def __init__(
        self,
        persist_dir: Path,
        collection_name: str = "notes_cours",
    ) -> None:
        self.persist_dir = persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.persist_dir))
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine", "description": "Notes de cours Obsidian"},
        )

    def delete_source_chunks(self, fichier_source: str) -> None:
        """Supprime tous les chunks associés à un fichier source."""
        try:
            self.collection.delete(where={"fichier_source": fichier_source})
        except Exception as e:
            logger.warning(
                "Impossible de supprimer les anciens chunks pour %s : %s",
                fichier_source,
                e,
            )

    def add_chunks(
        self,
        chunks: List[Dict[str, Any]],
        embeddings: List[List[float]],
        file_hash: str,
        start_index: int = 0,
    ) -> None:
        """Ajoute un lot de chunks et leurs embeddings dans la collection ChromaDB."""
        if not chunks:
            return

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for idx, chunk in enumerate(chunks, start=start_index):
            fichier = chunk.get("fichier_source", "unknown")
            chunk_id = f"{fichier}#chunk_{idx}"
            ids.append(chunk_id)
            documents.append(chunk.get("texte", ""))
            metadatas.append(sanitize_metadata(chunk, chunk_index=idx, file_hash=file_hash))

        # ChromaDB gère l'insertion par lots de sécurité
        batch_limit = 500
        for i in range(0, len(ids), batch_limit):
            end = i + batch_limit
            self.collection.add(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )

    def count(self) -> int:
        """Retourne le nombre total d'éléments dans la collection."""
        return self.collection.count()


def run_indexing(
    input_file: Path,
    chroma_dir: Path,
    collection_name: str = "notes_cours",
    model_name: str = "gemini-embedding-001",
    vault_path: Optional[Path] = None,
    api_key: Optional[str] = None,
    batch_size: int = 50,
    force: bool = False,
    embedder: Optional[GeminiEmbedder] = None,
    db: Optional[ChromaDatabase] = None,
) -> Dict[str, Any]:
    """Orchestre l'ingestion incrémentale des embeddings dans ChromaDB."""
    logger.info("Chargement des chunks depuis %s...", input_file)
    chunks = load_chunks_from_jsonl(input_file)
    grouped = group_chunks_by_source(chunks)
    logger.info("%d chunks chargés répartis sur %d fichiers sources.", len(chunks), len(grouped))

    # Initialisation de la base vectorielle
    if db is None:
        db = ChromaDatabase(persist_dir=chroma_dir, collection_name=collection_name)

    # Initialisation du fichier d'état incrémental
    state_file = chroma_dir / "embed_state.json"
    state = EmbeddingState(state_file)

    # Détection des fichiers modifiés, nouveaux ou supprimés
    to_index: List[Tuple[str, List[Dict[str, Any]], str, Optional[float]]] = []
    unchanged_count = 0

    current_files = set(grouped.keys())
    tracked_files = set(state.entries.keys())

    # Fichiers supprimés (présents dans l'état mais plus dans les chunks)
    deleted_files = tracked_files - current_files
    for deleted_file in deleted_files:
        logger.info("Suppression des chunks du fichier retiré : %s", deleted_file)
        db.delete_source_chunks(deleted_file)
        state.remove_file(deleted_file)

    # Comparaison des fichiers actuels
    for fichier_source, file_chunks in grouped.items():
        current_hash = compute_source_hash(fichier_source, file_chunks, vault_path=vault_path)
        current_mtime = get_source_mtime(fichier_source, vault_path=vault_path)

        if force or state.is_changed(fichier_source, current_hash, current_mtime):
            to_index.append((fichier_source, file_chunks, current_hash, current_mtime))
        else:
            unchanged_count += 1

    total_chunks_to_embed = sum(len(item[1]) for item in to_index)
    logger.info(
        "Bilan d'analyse incrémentale : %d fichiers à indexer (%d chunks), %d fichiers inchangés.",
        len(to_index),
        total_chunks_to_embed,
        unchanged_count,
    )

    if not to_index:
        logger.info("Tous les fichiers sont à jour dans ChromaDB. Aucun recalcul nécessaire.")
        state.save()
        return {
            "total_chunks_in_file": len(chunks),
            "files_total": len(grouped),
            "files_indexed": 0,
            "files_unchanged": unchanged_count,
            "files_deleted": len(deleted_files),
            "chunks_embedded": 0,
            "total_in_db": db.count(),
        }

    # Initialisation du générateur d'embeddings
    if embedder is None:
        embedder = GeminiEmbedder(
            api_key=api_key,
            model_name=model_name,
        )

    # Traitement des fichiers modifiés
    # 1. Supprimer les chunks existants de ces fichiers pour éviter les orphelins
    for fichier_source, _, _, _ in to_index:
        db.delete_source_chunks(fichier_source)

    # 2. Aplatir l'ensemble des chunks des fichiers à réindexer
    flat_items: List[Tuple[str, Dict[str, Any], int, str, Optional[float]]] = []
    for fichier_source, file_chunks, content_hash, mtime in to_index:
        for idx, chunk in enumerate(file_chunks):
            flat_items.append((fichier_source, chunk, idx, content_hash, mtime))

    total_chunks_to_embed = len(flat_items)
    total_batches = (total_chunks_to_embed + batch_size - 1) // batch_size if total_chunks_to_embed else 0

    # 3. Générer les embeddings par lots globaux (réduit considérablement le nombre de requêtes API)
    for b_idx in range(0, total_chunks_to_embed, batch_size):
        batch_slice = flat_items[b_idx : b_idx + batch_size]
        batch_texts = [item[1].get("texte", "") for item in batch_slice]
        batch_num = (b_idx // batch_size) + 1

        logger.info(
            "Génération embeddings lot [%d/%d] (%d chunks)...",
            batch_num,
            total_batches,
            len(batch_texts),
        )

        vectors = embedder.embed_batch(batch_texts)

        # Regrouper les résultats par fichier source pour l'insertion ordonnée
        grouped_batch: Dict[str, List[Tuple[Dict[str, Any], int, List[float], str]]] = defaultdict(list)
        for (f_src, chk, idx, c_hash, _), vec in zip(batch_slice, vectors):
            grouped_batch[f_src].append((chk, idx, vec, c_hash))

        for f_src, items in grouped_batch.items():
            sub_chunks = [it[0] for it in items]
            sub_start_idx = items[0][1]
            sub_vecs = [it[2] for it in items]
            sub_hash = items[0][3]
            db.add_chunks(
                chunks=sub_chunks,
                embeddings=sub_vecs,
                file_hash=sub_hash,
                start_index=sub_start_idx,
            )

        # Pause préventive légère pour respecter les quotas Free Tier
        time.sleep(0.1)

    # 4. Mettre à jour l'état incrémental pour chaque fichier traité
    for fichier_source, file_chunks, content_hash, mtime in to_index:
        state.update_file(
            fichier_source=fichier_source,
            content_hash=content_hash,
            mtime=mtime,
            chunk_count=len(file_chunks),
        )

    chunks_embedded = total_chunks_to_embed

    # Sauvegarde finale de l'état
    state.save()

    logger.info(
        "Indexation terminée avec succès ! Chunks indexés : %d. Total dans ChromaDB : %d.",
        chunks_embedded,
        db.count(),
    )

    return {
        "total_chunks_in_file": len(chunks),
        "files_total": len(grouped),
        "files_indexed": len(to_index),
        "files_unchanged": unchanged_count,
        "files_deleted": len(deleted_files),
        "chunks_embedded": chunks_embedded,
        "total_in_db": db.count(),
    }


def main(argv: Optional[List[str]] = None) -> None:
    """Point d'entrée CLI pour la commande embed."""
    default_vault = os.environ.get(
        "OBSIDIAN_VAULT_PATH", "/run/media/selasmsi/Windows/Programming_travel/"
    )
    default_input = "data/chunks.jsonl"
    default_chroma_dir = "data/chromadb"
    default_model = "gemini-embedding-001"
    default_collection = "notes_cours"

    parser = argparse.ArgumentParser(
        description="Génération des embeddings Gemini et indexation incrémentale dans ChromaDB."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=default_input,
        help=f"Fichier d'entrée JSONL (défaut : {default_input})",
    )
    parser.add_argument(
        "--db-dir",
        "-d",
        type=str,
        default=default_chroma_dir,
        help=f"Dossier de stockage persistant ChromaDB (défaut : {default_chroma_dir})",
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
        "-m",
        type=str,
        default=default_model,
        help=f"Modèle d'embedding Gemini (défaut : {default_model})",
    )
    parser.add_argument(
        "--vault",
        type=str,
        default=default_vault,
        help=f"Chemin vers le vault Obsidian pour la détection de mtime (défaut : {default_vault})",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Taille des lots pour l'appel d'API Gemini (défaut : 50)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Clé API Gemini (ou utiliser GEMINI_API_KEY dans l'environnement / .env)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Forcer la réindexation complète de tous les chunks, même inchangés.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Activer les messages de log détaillés.",
    )

    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    vault_path = Path(args.vault) if args.vault else None

    print(f"=== Indexation ChromaDB des notes de cours ===")
    print(f"Fichier chunks : {args.input}")
    print(f"Base ChromaDB  : {args.db_dir} (collection: {args.collection})")
    print(f"Modèle Gemini  : {args.model}")
    print(f"Mode forcé     : {args.force}")
    print("-" * 50)

    try:
        stats = run_indexing(
            input_file=Path(args.input),
            chroma_dir=Path(args.db_dir),
            collection_name=args.collection,
            model_name=args.model,
            vault_path=vault_path,
            api_key=args.api_key,
            batch_size=args.batch_size,
            force=args.force,
        )

        print("\n=== Résumé de l'opération ===")
        print(f"Fichiers analysés  : {stats['files_total']}")
        print(f"Fichiers indexés   : {stats['files_indexed']}")
        print(f"Fichiers inchangés : {stats['files_unchanged']}")
        if stats["files_deleted"] > 0:
            print(f"Fichiers supprimés : {stats['files_deleted']}")
        print(f"Chunks embarqués   : {stats['chunks_embedded']}")
        print(f"Total dans la base : {stats['total_in_db']}")
        print("Indexation achevée avec succès.")
    except Exception as exc:
        print(f"\n[ERREUR] Une erreur est survenue lors de l'indexation : {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
