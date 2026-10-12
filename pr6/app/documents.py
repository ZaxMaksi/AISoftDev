"""Колекція документів: читання, метадані, поділ на фрагменти (chunking).

Це єдине місце, яке знає, як влаштовані файли в `docs/`: де в них
метадані, як розмічено текст, за якими межами його ділити. Решта
застосунку працює з готовими фрагментами (`Chunk`) і не читає файлів.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from . import config

DOCS_DIR = config.DOCS_DIR
CHUNK_SIZE = config.CHUNK_SIZE
CHUNK_OVERLAP = config.CHUNK_OVERLAP


@dataclass
class Chunk:
    """Фрагмент документа — одиниця індексування й пошуку.

    `text` — чистий текст фрагмента для відображення користувачеві.
    `source` — імʼя файлу, з якого взято фрагмент.
    `metadata` — метадані документа та розділу (title, category, product,
    audience, updated, date, status, section, heading).
    `embedding_text` — текст із контекстом документа та розділу для моделі ембедінгів.
    """

    text: str
    source: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding_text: str | None = None

    def get_embedding_text(self) -> str:
        """Повернути текст для кодування ембедінгів із контекстом заголовків."""
        if self.embedding_text:
            return self.embedding_text
        title = self.metadata.get("title", "")
        section = self.metadata.get("section", "") or self.metadata.get("heading", "")
        parts = []
        if title:
            parts.append(f"Документ: {title}")
        if section:
            parts.append(f"Розділ: {section}")
        if parts:
            return f"{'\n'.join(parts)}\n\n{self.text}"
        return self.text

    def to_dict(self) -> dict[str, Any]:
        """Серіалізація фрагмента у словник для JSON-індексу."""
        data: dict[str, Any] = {
            "text": self.text,
            "source": self.source,
            "metadata": dict(self.metadata),
        }
        for key in ("title", "section", "heading", "category", "product", "audience", "date", "updated", "status"):
            val = self.metadata.get(key)
            if val is not None and val != "":
                data[key] = val
        if self.embedding_text:
            data["embedding_text"] = self.embedding_text
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Chunk:
        """Відновлення фрагмента зі збереженого словника."""
        text = data.get("text", "")
        source = data.get("source", "")
        meta = dict(data.get("metadata", {}))

        for key in ("title", "section", "heading", "category", "product", "audience", "date", "updated", "status"):
            if key in data and key not in meta:
                meta[key] = data[key]

        if "section" in meta and "heading" not in meta:
            meta["heading"] = meta["section"]
        elif "heading" in meta and "section" not in meta:
            meta["section"] = meta["heading"]

        if "updated" in meta and "date" not in meta:
            meta["date"] = meta["updated"]
        elif "date" in meta and "updated" not in meta:
            meta["updated"] = meta["date"]

        embedding_text = data.get("embedding_text")
        return cls(text=text, source=source, metadata=meta, embedding_text=embedding_text)


def parse_front_matter(raw: str) -> tuple[dict[str, str], str]:
    """Відокремити блок метаданих від тексту документа.

    Блок — рядки `ключ: значення` між двома рядками `---` на початку файлу.
    Повертає словник метаданих і решту тексту.
    """
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw
    metadata: dict[str, str] = {}
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body = "\n".join(lines[i + 1:]).lstrip("\n")
            return metadata, body
        if ":" in line:
            key, _, value = line.partition(":")
            val = value.strip()
            metadata[key.strip()] = val
    return {}, raw


def load_documents(docs_dir: Path = DOCS_DIR) -> list[tuple[str, dict[str, str], str]]:
    """Прочитати всі документи колекції.

    Повертає список трійок (імʼя файлу, метадані, текст) для кожного `*.md`
    у папці, крім `README.md`.
    """
    documents = []
    for path in sorted(docs_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        metadata, body = parse_front_matter(path.read_text(encoding="utf-8"))
        documents.append((path.name, metadata, body))
    return documents


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Розбити довгий текст на шматки за абзацами або реченнями з перекриттям."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return [text] if text.strip() else []

    chunks: list[str] = []
    current_paras: list[str] = []
    current_len = 0

    for p in paragraphs:
        if len(p) > chunk_size:
            if current_paras:
                chunks.append("\n\n".join(current_paras))
                current_paras = []
                current_len = 0

            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", p) if s.strip()]
            sent_chunk: list[str] = []
            sent_len = 0
            for s in sentences:
                if sent_len + len(s) + 1 > chunk_size and sent_chunk:
                    chunks.append(" ".join(sent_chunk))
                    if overlap > 0 and len(sent_chunk[-1]) <= overlap:
                        sent_chunk = [sent_chunk[-1], s]
                        sent_len = len(sent_chunk[0]) + len(s) + 1
                    else:
                        sent_chunk = [s]
                        sent_len = len(s)
                else:
                    sent_chunk.append(s)
                    sent_len += len(s) + 1
            if sent_chunk:
                chunks.append(" ".join(sent_chunk))
            continue

        if current_len + len(p) + 2 > chunk_size and current_paras:
            chunks.append("\n\n".join(current_paras))
            if overlap > 0 and len(current_paras[-1]) <= overlap:
                current_paras = [current_paras[-1], p]
                current_len = len(current_paras[0]) + len(p) + 2
            else:
                current_paras = [p]
                current_len = len(p)
        else:
            current_paras.append(p)
            current_len += len(p) + 2

    if current_paras:
        chunks.append("\n\n".join(current_paras))

    return chunks


def split(
    text: str,
    source: str,
    metadata: dict[str, str],
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Поділити текст документа на змістовні фрагменти (chunks)."""
    sections = re.split(r"(?m)^(?=##? )", text)
    chunks: list[Chunk] = []
    doc_title = metadata.get("title", "")

    chunk_idx = 0
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue

        lines = sec.splitlines()
        first_line = lines[0].strip()
        body = "\n".join(lines[1:]).strip()

        if first_line.startswith("## "):
            section_title = first_line[3:].strip()
            raw_content = body if body else first_line
        elif first_line.startswith("# "):
            section_title = first_line[2:].strip()
            if not body:
                continue
            raw_content = body
        else:
            section_title = ""
            raw_content = sec

        if not raw_content:
            continue

        if len(raw_content) <= chunk_size:
            text_pieces = [raw_content]
        else:
            text_pieces = _split_long_text(raw_content, chunk_size=chunk_size, overlap=overlap)

        for piece in text_pieces:
            piece = piece.strip()
            if not piece:
                continue

            chunk_meta: dict[str, Any] = {}
            for k in ("title", "category", "product", "audience", "status"):
                val = metadata.get(k)
                if val:
                    chunk_meta[k] = val
            if metadata.get("updated"):
                chunk_meta["updated"] = metadata["updated"]
                chunk_meta["date"] = metadata["updated"]

            if section_title:
                chunk_meta["section"] = section_title
                chunk_meta["heading"] = section_title

            chunk_meta["chunk_index"] = chunk_idx

            header_lines = []
            if doc_title:
                header_lines.append(f"Документ: {doc_title}")
            if section_title:
                header_lines.append(f"Розділ: {section_title}")
            if header_lines:
                emb_text = f"{'\n'.join(header_lines)}\n\n{piece}"
            else:
                emb_text = piece

            chunks.append(
                Chunk(
                    text=piece,
                    source=source,
                    metadata=chunk_meta,
                    embedding_text=emb_text,
                )
            )
            chunk_idx += 1

    return chunks


def load_chunks(
    docs_dir: Path = DOCS_DIR,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> list[Chunk]:
    """Прочитати колекцію й повернути всі її фрагменти."""
    chunks: list[Chunk] = []
    for source, metadata, body in load_documents(docs_dir):
        chunks.extend(split(body, source, metadata, chunk_size=chunk_size, overlap=overlap))
    return chunks
