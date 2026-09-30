"""Interface commune des générateurs de réponse RAG et construction du prompt partagé."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List

SYSTEM_INSTRUCTIONS = (
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


def build_context(chunks: List[Dict[str, Any]]) -> str:
    """Formate les chunks récupérés en blocs [Source N] numérotés."""
    if not chunks:
        return "Aucune note de cours correspondante n'a été trouvée dans la base vectorielle.\n"

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
    return "\n".join(context_parts)


def build_user_message(question: str, chunks: List[Dict[str, Any]]) -> str:
    """Construit la partie « utilisateur » du prompt : extraits de cours puis question."""
    return (
        f"=== EXTRAITS DE COURS DISPONIBLES ===\n"
        f"{build_context(chunks)}\n"
        f"=== QUESTION DE L'ÉTUDIANT ===\n"
        f"{question}\n\n"
        f"=== RÉPONSE EN FRANÇAIS ==="
    )


def build_rag_prompt(question: str, chunks: List[Dict[str, Any]]) -> str:
    """Construit le prompt complet en un seul bloc (consignes + extraits + question)."""
    return f"{SYSTEM_INSTRUCTIONS}\n{build_user_message(question, chunks)}"


class Generator(ABC):
    """Générateur de réponse à partir d'une question et des chunks récupérés.

    Les backends qui distinguent consignes système et message utilisateur
    utilisent SYSTEM_INSTRUCTIONS et build_user_message ; les autres
    envoient build_rag_prompt tel quel.
    """

    name: str = "base"
    model: str = ""

    @abstractmethod
    def generate(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        """Retourne la réponse en markdown, citant les sources sous la forme [Source N]."""

    def describe(self) -> str:
        """Libellé lisible du backend et du modèle, pour l'affichage."""
        return f"{self.name} ({self.model})" if self.model else self.name
