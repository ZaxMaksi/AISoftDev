"""Пошук за ключовими словами (BM25) — допоміжний метод для точних збігів.

Працює з тими самими фрагментами (Chunk), тими самими метаданими та фільтрами,
повертаючи результат у форматі Hit.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Any
from rank_bm25 import BM25Okapi
from .documents import Chunk
from .index import DEFAULT_TOP_K, Hit


@dataclass
class KeywordIndex:
    """Індекс для пошуку за словами на основі BM25."""

    chunks: list[Chunk]
    bm25: BM25Okapi
    extra: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.chunks)


def tokenize(text: str) -> list[str]:
    """Розбити текст на слова для індексування й для запиту.

    Враховує:
    - Нижній регістр (lowercasing).
    - Нормалізацію апострофів (ʼ, ’ -> ').
    - Збереження артикулів, дефісних слів та чисел (OR-X2-BLK, AX3, R10, 2025 тощо).
    """
    cleaned = text.lower().replace("’", "'").replace("ʼ", "'")
    tokens = re.findall(r"\b[\w]+(?:['\-][\w]+)*\b", cleaned, re.UNICODE)
    return tokens


def build(chunks: list[Chunk]) -> KeywordIndex:
    """Зібрати індекс за словами з тих самих фрагментів, що й векторний.

    До корпусу кожного фрагмента додаються заголовок документа та розділу,
    щоб пошук знаходив фрагменти і за назвами тем.
    """
    tokenized_corpus: list[list[str]] = []
    for c in chunks:
        doc_title = c.metadata.get("title", "")
        sec_title = c.metadata.get("section", "") or c.metadata.get("heading", "")
        full_text = f"{doc_title} {sec_title} {c.text}"
        tokenized_corpus.append(tokenize(full_text))

    bm25 = BM25Okapi(tokenized_corpus)
    return KeywordIndex(chunks=chunks, bm25=bm25)


def search(
    index: KeywordIndex,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filters: dict[str, Any] | None = None,
) -> list[Hit]:
    if len(index.chunks) == 0:
        return []

    tokens = tokenize(query)
    if not tokens:
        return []

    scores = index.bm25.get_scores(tokens)
    candidates: list[tuple[float, Chunk]] = []
    for i, chunk in enumerate(index.chunks):
        score = float(scores[i])
        if score <= 0.0:
            continue
        if filters:
            match = True
            for f_key, f_val in filters.items():
                if f_val is None or f_val == "":
                    continue
                chunk_val = chunk.metadata.get(f_key)
                if chunk_val is None:
                    match = False
                    break
                if str(chunk_val).strip().lower() != str(f_val).strip().lower():
                    match = False
                    break
            if not match:
                continue
        candidates.append((score, chunk))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [Hit(chunk=c, score=s) for s, c in candidates[:top_k]]
