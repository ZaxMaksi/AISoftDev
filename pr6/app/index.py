"""Векторний індекс: збереження векторів фрагментів та семантичний пошук.

Індекс складається з:
- Масиву векторів (vectors.npy)
- Метаданих та тексту фрагментів (chunks.json)
- Назви моделі ембедінгів та параметрів chunking.

Пошук виконується через косинусну схожість (dot product для нормалізованих векторів)
з фільтрацією за метаданими та відсіканням за порогом схожості (threshold).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import config
from .documents import Chunk

INDEX_DIR = config.INDEX_DIR
DEFAULT_TOP_K = config.TOP_K
SIMILARITY_THRESHOLD = config.SIMILARITY_THRESHOLD


@dataclass
class Hit:
    """Одне влучення пошуку: фрагмент і оцінка його схожості із запитом."""

    chunk: Chunk
    score: float


@dataclass
class SearchIndex:
    """Індекс у памʼяті.

    `vectors` — масив (кількість фрагментів × розмірність).
    `chunks` — список фрагментів документів із метаданими.
    `model_name` — назва моделі, якою збудовано вектори.
    `extra` — додаткові параметри побудови (chunk_size, overlap, dimension тощо).
    """

    chunks: list[Chunk]
    vectors: np.ndarray
    model_name: str
    extra: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.chunks)


def build(
    chunks: list[Chunk],
    vectors: np.ndarray,
    model_name: str,
    extra: dict[str, Any] | None = None,
) -> SearchIndex:
    """Зібрати індекс із фрагментів і їхніх векторів.

    Перевіряє відповідність кількості векторів та фрагментів, нормалізує вектори
    до одиничної довжини (L2 norm) для точного й швидкого скалярного добутку.
    """
    if len(chunks) != vectors.shape[0]:
        raise ValueError(
            f"Невідповідність розмірів: {len(chunks)} фрагментів і {vectors.shape[0]} векторів."
        )

    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    norm_vectors = (vectors / norms).astype(np.float32)

    return SearchIndex(
        chunks=chunks,
        vectors=norm_vectors,
        model_name=model_name,
        extra=extra or {},
    )


def save(index: SearchIndex, path: Path = INDEX_DIR) -> None:
    """Зберегти індекс на диск (vectors.npy та chunks.json)."""
    path.mkdir(parents=True, exist_ok=True)

    np.save(path / "vectors.npy", index.vectors)

    data: dict[str, Any] = {
        "model": index.model_name,
        "dimension": int(index.vectors.shape[1]),
        "chunk_size": index.extra.get("chunk_size", config.CHUNK_SIZE),
        "overlap": index.extra.get("overlap", config.CHUNK_OVERLAP),
        "total_chunks": len(index.chunks),
        "chunks": [chunk.to_dict() for chunk in index.chunks],
    }

    chunks_file = path / "chunks.json"
    chunks_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load(path: Path = INDEX_DIR, expected_model: str | None = None) -> SearchIndex:
    """Прочитати індекс із диска.

    Якщо індекс відсутній або пошкоджений — підіймає зрозумілу помилку.
    Перевіряє сумісність моделі ембедінгів із поточною конфігурацією.
    """
    vectors_file = path / "vectors.npy"
    chunks_file = path / "chunks.json"

    if not vectors_file.exists() or not chunks_file.exists():
        raise FileNotFoundError(
            "Search index is not built. Run: python ingest.py"
        )

    try:
        vectors = np.load(vectors_file)
        raw_json = chunks_file.read_text(encoding="utf-8")
        data = json.loads(raw_json)
    except Exception as exc:
        raise ValueError(
            f"Індекс пошкоджено або не вдалося прочитати ({type(exc).__name__}: {exc}). "
            f"Перебудуйте: python ingest.py"
        ) from exc

    saved_model = data.get("model", "")
    chunk_dicts = data.get("chunks", [])

    if len(chunk_dicts) != vectors.shape[0]:
        raise ValueError(
            f"Пошкоджений індекс: кількість векторів ({vectors.shape[0]}) не збігається "
            f"з кількістю фрагментів ({len(chunk_dicts)}). Перебудуйте: python ingest.py"
        )

    cur_model = expected_model or config.EMBEDDING_MODEL
    if saved_model and cur_model and saved_model != cur_model:
        raise ValueError(
            f"Невідповідність моделі ембедінгів: індекс побудовано моделлю '{saved_model}', "
            f"а поточна конфігурація вимагає '{cur_model}'. Вектори різних моделей непорівнянні. "
            f"Перебудуйте індекс: python ingest.py"
        )

    chunks = [Chunk.from_dict(cd) for cd in chunk_dicts]

    return SearchIndex(
        chunks=chunks,
        vectors=vectors,
        model_name=saved_model,
        extra={
            "dimension": data.get("dimension", vectors.shape[1]),
            "chunk_size": data.get("chunk_size"),
            "overlap": data.get("overlap"),
        },
    )


def search(
    index: SearchIndex,
    query_vector: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
    filters: dict[str, Any] | None = None,
    threshold: float | None = SIMILARITY_THRESHOLD,
) -> list[Hit]:
    """Знайти top-k фрагментів, найближчих за cosine similarity до вектора запиту.

    Алгоритм:
    1. Обчислення косинусної схожості через скалярний добуток.
    2. Фільтрація за метаданими (category, product, audience, status тощо).
    3. Відсікання результатів, нижчих за threshold.
    4. Сортування за спаданням та повернення top-k влучень.
    """
    if len(index.chunks) == 0:
        return []

    q_norm = np.linalg.norm(query_vector)
    if q_norm > 0:
        query_vector = query_vector / q_norm

    scores = index.vectors @ query_vector

    candidates: list[tuple[float, Chunk]] = []

    for i, chunk in enumerate(index.chunks):
        score = float(scores[i])

        if threshold is not None and score < threshold:
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
