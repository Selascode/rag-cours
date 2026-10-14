"""Module d'ingestion et découpage (chunking) des notes de cours Obsidian."""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def replace_images(text: str) -> str:
    """Remplace les embeds d'images Obsidian et Markdown par le repère [image]."""
    text = re.sub(r"!\[\[.*?\]\]", "[image]", text)
    text = re.sub(r"!\[.*?\]\(.*?\)", "[image]", text)
    return text


def extract_note_tags(text: str) -> List[str]:
    """Extrait les tags déclarés en début de note (lignes Tags: #x #y, **Tags**, tags: [...])."""
    tags: List[str] = []
    lines = text.splitlines()[:25]
    for line in lines:
        m = re.search(r"(?:\*\*Tags\*\*|Tags|\*\*Tags|tags)\s*:\s*(.*)", line, re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
            if raw.startswith("[") and raw.endswith("]"):
                items = raw[1:-1].split(",")
                for it in items:
                    t = it.strip().strip("\"'").lstrip("#")
                    if t and not t.startswith("<%"):
                        tags.append(t)
            else:
                ht = re.findall(r"#([a-zA-Z0-9_\u00c0-\u017f-]+)", raw)
                for t in ht:
                    if not t.startswith("<%"):
                        tags.append(t)
    return list(dict.fromkeys(tags))


def extract_inline_tags(text: str) -> List[str]:
    """Extrait les #hashtags inline en ignorant ceux dans les blocs de code et code inline."""
    # Retirer les blocs de code
    no_code = re.sub(r"```[\s\S]*?```", "", text)
    no_code = re.sub(r"~~~[\s\S]*?~~~", "", no_code)
    # Retirer le code inline
    no_code = re.sub(r"`[^`\n]+`", "", no_code)

    inline_tags: List[str] = []
    for line in no_code.splitlines():
        stripped = line.strip()
        # Ignorer les titres markdown (# Titre, ## Titre...)
        if re.match(r"^#{1,6}\s+", stripped):
            continue
        matches = re.finditer(r"(?<![#\w])#([a-zA-Z0-9_\u00c0-\u017f][\w\-\u00c0-\u017f]*)", line)
        for m in matches:
            t = m.group(1)
            if not t.startswith("<%") and t.lower() not in ("tags", "tag"):
                inline_tags.append(t)
    return list(dict.fromkeys(inline_tags))


def extract_wikilinks(text: str) -> List[str]:
    """Extrait les wikilinks [[...]] présents dans le texte (hors embeds d'image déjà remplacés)."""
    links: List[str] = []
    matches = re.findall(r"\[\[(.*?)\]\]", text)
    for m in matches:
        target = m.split("|")[0].strip()
        if target and target not in links:
            links.append(target)
    return links


class Block:
    """Représente un bloc atomique de texte (indivisible lors du chunking)."""

    def __init__(self, block_type: str, text: str, level: int = 0, title: str = ""):
        self.type = block_type  # 'heading', 'callout', 'code', 'latex', 'table', 'text'
        self.text = text
        self.level = level  # Pour les titres (1, 2, 3...)
        self.title = title  # Titre nettoyé sans '#'

    def __repr__(self) -> str:
        return f"Block({self.type}, len={len(self.text)}, level={self.level})"


def parse_blocks(text: str) -> List[Block]:
    """Découpe un texte markdown en une séquence de blocs atomiques.

    Garantit que le code, LaTeX, tableaux et callouts ne sont jamais morcelés.
    """
    lines = text.split("\n")
    blocks: List[Block] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        stripped = line.strip()

        if not stripped:
            i += 1
            continue

        # 1. Titre Markdown (# , ## , ### ...)
        m_head = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m_head:
            blocks.append(
                Block(
                    block_type="heading",
                    text=line,
                    level=len(m_head.group(1)),
                    title=m_head.group(2).strip(),
                )
            )
            i += 1
            continue

        # 2. Callout Obsidian : > [!type]+ Titre ... suivi de lignes commençant par >
        if re.match(r"^\s*>\s*\[![^\]]+\]", line):
            callout_lines = [line]
            i += 1
            while i < n:
                next_line = lines[i]
                if re.match(r"^\s*>", next_line):
                    callout_lines.append(next_line)
                    i += 1
                else:
                    break
            blocks.append(Block(block_type="callout", text="\n".join(callout_lines)))
            continue

        # 3. Bloc de code délimité par ``` ou ~~~
        m_code = re.match(r"^\s*(`{3,}|~{3,})", line)
        if m_code:
            fence = m_code.group(1)
            fence_char = fence[0]
            fence_len = len(fence)
            code_lines = [line]
            i += 1
            while i < n:
                cur = lines[i]
                code_lines.append(cur)
                i += 1
                if re.match(rf"^\s*\{fence_char}{{{fence_len},}}\s*$", cur):
                    break
            blocks.append(Block(block_type="code", text="\n".join(code_lines)))
            continue

        # 4. Formule LaTeX en bloc : $$ ... $$
        if re.match(r"^\s*\$\$(?!\$)", line):
            latex_lines = [line]
            i += 1
            if line.count("$$") >= 2 and len(stripped) > 2:
                blocks.append(Block(block_type="latex", text=line))
                continue
            while i < n:
                cur = lines[i]
                latex_lines.append(cur)
                i += 1
                if "$$" in cur:
                    break
            blocks.append(Block(block_type="latex", text="\n".join(latex_lines)))
            continue

        # 5. Tableau Markdown
        if "|" in line and i + 1 < n and re.match(r"^\s*\|?\s*:?-+:?\s*\|", lines[i + 1]):
            table_lines = [line, lines[i + 1]]
            i += 2
            while i < n:
                cur = lines[i]
                if "|" in cur and cur.strip():
                    table_lines.append(cur)
                    i += 1
                else:
                    break
            blocks.append(Block(block_type="table", text="\n".join(table_lines)))
            continue

        # 6. Paragraphe / listes / texte régulier
        text_lines = [line]
        i += 1
        while i < n:
            cur = lines[i]
            cur_s = cur.strip()
            if not cur_s:
                i += 1
                break
            if (
                re.match(r"^#{1,6}\s+", cur)
                or re.match(r"^\s*>\s*\[!", cur)
                or re.match(r"^\s*(`{3,}|~{3,})", cur)
                or re.match(r"^\s*\$\$(?!\$)", cur)
                or ("|" in cur and i + 1 < n and re.match(r"^\s*\|?\s*:?-+:?\s*\|", lines[i + 1]))
            ):
                break
            text_lines.append(cur)
            i += 1

        blocks.append(Block(block_type="text", text="\n".join(text_lines)))

    return blocks


def is_meaningful_content(blocks: List[Block], is_preamble: bool = False) -> bool:
    """Vérifie si une liste de blocs contient du vrai contenu textuel ou multimédia."""
    for b in blocks:
        if b.type in ("code", "callout", "latex", "table"):
            return True
        if not is_preamble and b.type == "heading":
            return True
        if b.type == "text":
            cleaned = re.sub(
                r"^(?:\*\*Tags\*\*|Tags|\*\*Tags|tags)\s*:.*$",
                "",
                b.text,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            cleaned = re.sub(
                r"^(?:MOC\s*:|\*\*Navigation\*\*|\*\*MOC\*\*).*$",
                "",
                cleaned,
                flags=re.IGNORECASE | re.MULTILINE,
            )
            # En préambule, les simples wikilinks et repères [image] isolés ne constituent pas un chunk autonome
            if is_preamble:
                cleaned = re.sub(r"\[\[.*?\]\]", "", cleaned)
                cleaned = re.sub(r"\[image\]", "", cleaned)
            cleaned = cleaned.replace("---", "").strip()
            if cleaned:
                return True
    return False


def get_last_text_overlap(blocks: List[Block], max_overlap_chars: int = 150) -> str:
    """Extrait un extrait de texte en fin de chunk pour assurer la continuité sémantique."""
    for b in reversed(blocks):
        if b.type == "text":
            t = b.text.strip()
            if not t or t == "---":
                continue
            if len(t) <= max_overlap_chars:
                return t
            end_part = t[-max_overlap_chars:]
            for punct in (". ", "! ", "? ", "\n"):
                idx = end_part.find(punct)
                if idx != -1 and idx + len(punct) < len(end_part):
                    return end_part[idx + len(punct):].strip()
            return end_part.strip()
    return ""


def chunk_document(
    content: str, max_chunk_chars: int = 2000, overlap_chars: int = 150
) -> List[Dict[str, Any]]:
    """Découpe une note Obsidian par titres H2 (en conservant les H3 si possible)."""
    content = replace_images(content)
    blocks = parse_blocks(content)
    if not blocks:
        return []

    doc_h1 = None
    for b in blocks:
        if b.type == "heading" and b.level == 1:
            doc_h1 = b.title
            break

    has_h2 = any(b.type == "heading" and b.level == 2 for b in blocks)
    has_h3 = any(b.type == "heading" and b.level == 3 for b in blocks)
    has_h1 = any(b.type == "heading" and b.level == 1 for b in blocks)
    split_level = 2 if has_h2 else (3 if has_h3 else (1 if has_h1 else None))

    sections: List[Dict[str, Any]] = []
    current_section: Dict[str, Any] = {"heading": None, "blocks": []}

    for b in blocks:
        if split_level and b.type == "heading" and b.level == split_level:
            if current_section["heading"] is not None or current_section["blocks"]:
                sections.append(current_section)
            current_section = {"heading": b, "blocks": [b]}
        else:
            current_section["blocks"].append(b)

    if current_section["heading"] is not None or current_section["blocks"]:
        sections.append(current_section)

    raw_chunks: List[Dict[str, Any]] = []
    for idx, sec in enumerate(sections):
        sec_h = sec["heading"]
        sec_blocks: List[Block] = sec["blocks"]

        # Préambule avant la première section découpée
        if sec_h is None:
            if not is_meaningful_content(sec_blocks, is_preamble=True):
                if idx + 1 < len(sections):
                    next_sec = sections[idx + 1]
                    if next_sec["blocks"] and next_sec["blocks"][0].type == "heading":
                        next_sec["blocks"] = (
                            [next_sec["blocks"][0]] + sec_blocks + next_sec["blocks"][1:]
                        )
                    else:
                        next_sec["blocks"] = sec_blocks + next_sec["blocks"]
                continue

            chemin = [doc_h1] if doc_h1 else ["Introduction"]
            raw_chunks.append({
                "chemin_titres": chemin,
                "blocks": sec_blocks,
                "main_heading": doc_h1 or "Introduction",
            })
            continue

        # Si la section ne contient rien en dehors de son titre
        content_blocks = [b for b in sec_blocks if b != sec_h]
        if not is_meaningful_content(content_blocks, is_preamble=False):
            if idx + 1 < len(sections):
                sections[idx + 1]["blocks"] = sec_blocks + sections[idx + 1]["blocks"]
            continue

        h_title = sec_h.title
        chemin_base = [doc_h1, h_title] if doc_h1 and doc_h1 != h_title else [h_title]
        total_len = sum(len(b.text) for b in sec_blocks)

        # Si l'ensemble de la section H2 tient dans max_chunk_chars, conserver tous les H3 ensemble
        if total_len <= max_chunk_chars:
            raw_chunks.append({
                "chemin_titres": chemin_base,
                "blocks": sec_blocks,
                "main_heading": h_title,
            })
        else:
            # Regrouper par sous-sections H3
            intro_blocks: List[Block] = []
            h3_subsections: List[tuple[Optional[Block], List[Block]]] = []
            curr_h3: Optional[Block] = None
            curr_h3_blocks: List[Block] = []

            for b in sec_blocks:
                if b.type == "heading" and b.level == 3:
                    if curr_h3 is None:
                        intro_blocks = list(curr_h3_blocks)
                    else:
                        h3_subsections.append((curr_h3, curr_h3_blocks))
                    curr_h3 = b
                    curr_h3_blocks = [b]
                else:
                    curr_h3_blocks.append(b)
            if curr_h3 is None:
                intro_blocks = list(curr_h3_blocks)
            else:
                h3_subsections.append((curr_h3, curr_h3_blocks))

            if not h3_subsections:
                # Pas de sous-sections H3, découpage glouton par blocs atomiques
                cur_pack: List[Block] = []
                cur_len = 0
                for b in sec_blocks:
                    has_content = any(x.type != "heading" for x in cur_pack)
                    if cur_pack and has_content and (cur_len + len(b.text) > max_chunk_chars):
                        raw_chunks.append({
                            "chemin_titres": chemin_base,
                            "blocks": cur_pack,
                            "main_heading": h_title,
                        })
                        cur_pack = [b]
                        cur_len = len(b.text)
                    else:
                        cur_pack.append(b)
                        cur_len += len(b.text)
                if cur_pack:
                    raw_chunks.append({
                        "chemin_titres": chemin_base,
                        "blocks": cur_pack,
                        "main_heading": h_title,
                    })
            else:
                # Regroupement glouton des sous-sections H3
                cur_pack = list(intro_blocks)
                cur_len = sum(len(b.text) for b in cur_pack)
                cur_h3_names: List[str] = []

                for h3_head, h3_blks in h3_subsections:
                    h3_len = sum(len(b.text) for b in h3_blks)

                    if h3_len > max_chunk_chars:
                        has_content = any(x.type != "heading" for x in cur_pack)
                        if cur_pack and has_content:
                            sub_chem = list(chemin_base)
                            if cur_h3_names:
                                sub_chem.append(cur_h3_names[0])
                            raw_chunks.append({
                                "chemin_titres": sub_chem,
                                "blocks": cur_pack,
                                "main_heading": h_title,
                            })
                            cur_pack = []
                            cur_len = 0
                            cur_h3_names = []

                        # Découpage du gros bloc H3 par blocs atomiques
                        sub_cur_pack: List[Block] = []
                        sub_cur_len = 0
                        h3_title_str = h3_head.title if h3_head else ""
                        for b in h3_blks:
                            sub_has_content = any(x.type != "heading" for x in sub_cur_pack)
                            if (
                                sub_cur_pack
                                and sub_has_content
                                and (sub_cur_len + len(b.text) > max_chunk_chars)
                            ):
                                raw_chunks.append({
                                    "chemin_titres": chemin_base + ([h3_title_str] if h3_title_str else []),
                                    "blocks": sub_cur_pack,
                                    "main_heading": h_title,
                                    "h3_title": h3_title_str,
                                })
                                sub_cur_pack = [b]
                                sub_cur_len = len(b.text)
                            else:
                                sub_cur_pack.append(b)
                                sub_cur_len += len(b.text)
                        if sub_cur_pack:
                            raw_chunks.append({
                                "chemin_titres": chemin_base + ([h3_title_str] if h3_title_str else []),
                                "blocks": sub_cur_pack,
                                "main_heading": h_title,
                                "h3_title": h3_title_str,
                            })
                    else:
                        has_content = any(x.type != "heading" for x in cur_pack)
                        if cur_pack and has_content and (cur_len + h3_len > max_chunk_chars):
                            sub_chem = list(chemin_base)
                            if cur_h3_names:
                                sub_chem.append(cur_h3_names[0])
                            raw_chunks.append({
                                "chemin_titres": sub_chem,
                                "blocks": cur_pack,
                                "main_heading": h_title,
                            })
                            cur_pack = list(h3_blks)
                            cur_len = h3_len
                            cur_h3_names = [h3_head.title] if h3_head else []
                        else:
                            cur_pack.extend(h3_blks)
                            cur_len += h3_len
                            if h3_head:
                                cur_h3_names.append(h3_head.title)

                if cur_pack:
                    sub_chem = list(chemin_base)
                    if cur_h3_names:
                        sub_chem.append(cur_h3_names[0])
                    raw_chunks.append({
                        "chemin_titres": sub_chem,
                        "blocks": cur_pack,
                        "main_heading": h_title,
                    })

    # Assemblage final avec chevauchement contextuel
    final_chunks: List[Dict[str, Any]] = []
    prev_blocks: List[Block] = []

    for c in raw_chunks:
        c_blocks: List[Block] = c["blocks"]
        if not c_blocks:
            continue

        chunk_text_parts = [b.text for b in c_blocks]
        chunk_body = "\n\n".join(chunk_text_parts).strip()

        main_h = c.get("main_heading")
        first_b = c_blocks[0]

        overlap = ""
        if prev_blocks and overlap_chars > 0:
            overlap = get_last_text_overlap(prev_blocks, overlap_chars)
            if overlap and overlap in chunk_body:
                overlap = ""

        if first_b.type == "heading":
            if overlap:
                h_line = first_b.text
                rest = "\n\n".join(b.text for b in c_blocks[1:]).strip()
                if rest:
                    chunk_body = f"{h_line}\n\n... {overlap}\n\n{rest}"
                else:
                    chunk_body = f"{h_line}\n\n... {overlap}"
            else:
                chunk_body = "\n\n".join(chunk_text_parts).strip()
        else:
            prefix = f"## {main_h}\n\n" if main_h else ""
            if overlap:
                chunk_body = f"{prefix}... {overlap}\n\n{chunk_body}"
            else:
                chunk_body = f"{prefix}{chunk_body}"

        prev_blocks = c_blocks

        final_chunks.append({
            "texte": chunk_body.strip(),
            "chemin_titres": c["chemin_titres"],
            "blocks": c_blocks,
        })

    return final_chunks


def ingest_vault(
    vault_path: str,
    output_path: str,
    max_chunk_chars: int = 2000,
    overlap_chars: int = 150,
) -> List[Dict[str, Any]]:
    """Parcourt récursivement le vault, découpe les notes et enregistre dans un fichier JSONL."""
    vault_p = Path(vault_path).expanduser().resolve()
    if not vault_p.exists():
        raise FileNotFoundError(f"Le chemin du vault n'existe pas : {vault_p}")

    out_p = Path(output_path).expanduser().resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    all_chunks: List[Dict[str, Any]] = []
    processed_notes = 0

    # Collecter et trier les fichiers pour un ordre déterministe
    md_files: List[Path] = []
    for root, _, files in os.walk(vault_p):
        for f in files:
            if f.endswith(".md"):
                md_files.append(Path(root) / f)

    md_files.sort()

    for full_file_path in md_files:
        try:
            rel_path = full_file_path.relative_to(vault_p)
        except ValueError:
            continue

        parts = rel_path.parts
        # Filtre strict : uniquement sous 10_Notes/ ou 11_TPs/
        if not parts or parts[0] not in ("10_Notes", "11_TPs"):
            continue
        if len(parts) < 3:
            continue

        # Matière = sous-dossier direct sous 10_Notes/ ou 11_TPs/
        matiere = parts[1]
        fichier_source = rel_path.as_posix()

        with open(full_file_path, "r", encoding="utf-8", errors="ignore") as fp:
            content = fp.read()

        note_tags = extract_note_tags(content)
        doc_chunks = chunk_document(
            content,
            max_chunk_chars=max_chunk_chars,
            overlap_chars=overlap_chars,
        )

        for dc in doc_chunks:
            chunk_tags = extract_inline_tags(dc["texte"])
            merged_tags = list(dict.fromkeys(note_tags + chunk_tags))
            wikis = extract_wikilinks(dc["texte"])

            chunk_obj = {
                "texte": dc["texte"],
                "fichier_source": fichier_source,
                "chemin_titres": dc["chemin_titres"],
                "matiere": matiere,
                "tags": merged_tags,
                "wikilinks": wikis,
            }
            all_chunks.append(chunk_obj)

        processed_notes += 1

    # Écriture dans le fichier JSONL
    with open(out_p, "w", encoding="utf-8") as out_fp:
        for chunk in all_chunks:
            line = json.dumps(chunk, ensure_ascii=False)
            out_fp.write(line + "\n")

    return all_chunks


def main(argv: Optional[List[str]] = None) -> None:
    default_vault = os.environ.get(
        "OBSIDIAN_VAULT_PATH", "/run/media/selasmsi/Windows/Programming_travel/"
    )
    default_output = "data/chunks.jsonl"

    parser = argparse.ArgumentParser(
        description="Ingestion et découpage des notes Obsidian en chunks JSONL pour RAG."
    )
    parser.add_argument(
        "--vault",
        type=str,
        default=default_vault,
        help=f"Chemin vers le vault Obsidian (défaut : {default_vault})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=default_output,
        help=f"Fichier de sortie JSONL (défaut : {default_output})",
    )
    parser.add_argument(
        "--max-chunk-size",
        type=int,
        default=2000,
        help="Taille maximale approximative d'un chunk en caractères (défaut : 2000)",
    )
    parser.add_argument(
        "--overlap",
        type=int,
        default=150,
        help="Taille du léger chevauchement entre chunks consécutifs (défaut : 150)",
    )

    args = parser.parse_args(argv)

    print(f"Ingestion du vault depuis : {args.vault}")
    print(f"Destination : {args.output}")

    chunks = ingest_vault(
        vault_path=args.vault,
        output_path=args.output,
        max_chunk_chars=args.max_chunk_size,
        overlap_chars=args.overlap,
    )

    matieres: Dict[str, int] = {}
    for c in chunks:
        mat = c["matiere"]
        matieres[mat] = matieres.get(mat, 0) + 1

    print(f"\nIngestion terminée avec succès !")
    print(f"Total chunks générés : {len(chunks)}")
    print("Répartition par matière :")
    for mat, count in sorted(matieres.items()):
        print(f"  - {mat:15} : {count} chunks")
    print(f"Fichier sauvegardé : {args.output}")


if __name__ == "__main__":
    main()
