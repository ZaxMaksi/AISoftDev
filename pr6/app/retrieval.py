"""Відбір фрагментів для моделі та збирання контексту.

Між пошуком із ПР5 і мовною моделлю стоїть шар, який вирішує, що саме модель побачить:
- Застосовує суворі серверні безпекові фільтри (захист від витоку внутрішніх документів);
- Відсікає нерелевантні влучення за порогом схожості (SIMILARITY_THRESHOLD);
- Окремо обробляє архівні редакції (за замовчуванням використовуються лише чинні);
- Обмежує кількість фрагментів для передачі моделі (RAG_CONTEXT_CHUNKS);
- Формує структурований пронумерований контекст із метаданими в межах бюджету токенів.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import config, embeddings, keyword
from .documents import Chunk
from .index import Hit, SearchIndex, search as search_index
from .keyword import KeywordIndex

logger = logging.getLogger(__name__)

CONTEXT_CHUNKS = config.RAG_CONTEXT_CHUNKS
CONTEXT_BUDGET = config.RAG_CONTEXT_BUDGET
SIMILARITY_THRESHOLD = config.SIMILARITY_THRESHOLD


@dataclass
class Source:
    """Джерело, показане моделі й користувачеві.

    `ref` — порядковий номер [1..N], під яким фрагмент стоїть у контексті;
    `chunk` — сам фрагмент із метаданими;
    `score` — оцінка пошуку, за якою його відібрано.
    """

    ref: int
    chunk: Chunk
    score: float


def estimate_tokens(text: str) -> int:
    """Оцінити кількість токенів для тексту українською мовою.

    Для кирилиці один токен становить у середньому ~2.9 символи,
    плюс 4 службові токени на структуру розмітки блоку.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 2.9) + 4)


def sanitize_filters(user_filters: dict[str, Any] | None) -> dict[str, Any]:
    """Очищення та валідація користувацьких фільтрів.

    Політика безпеки:
    1. Дозволені лише поля з config.ALLOWED_FILTERS (наприклад, product, category).
    2. Будь-які спроби клієнта передати audience, status тощо відхиляються або ігноруються.
    3. Сервер ЖОРСТКО встановлює:
       - audience: 'клієнти' (внутрішні документи персоналу заборонені);
       - status: 'чинний' (архівні документи не змішуються з актуальними правилами).
    """
    effective: dict[str, Any] = {
        "audience": "клієнти",
        "status": "чинний",
    }

    if user_filters:
        for k, v in user_filters.items():
            if k in config.ALLOWED_FILTERS and v:
                effective[k] = v

    return effective


def retrieve(
    query: str,
    index: SearchIndex,
    keyword_index: KeywordIndex | None = None,
    filters: dict | None = None,
    top_k: int | None = None,
    threshold: float | None = None,
    max_chunks: int | None = None,
) -> list[Hit]:
    """Знайти та відібрати фрагменти для передачі мовній моделі.

    Етапи:
    1. Формування серверних фільтрів безпеки (audience='клієнти', status='чинний').
    2. Семантичний пошук через вектори E5 з урахуванням порогу схожості.
    3. Додатковий аналіз ключових слів (BM25) для точних артикулів за наявності keyword_index.
    4. Фільтрація дублікатів (за змістом або тотожним текстом).
    5. Обмеження кількості відібраних фрагментів до max_chunks (RAG_CONTEXT_CHUNKS).
    """
    query_clean = query.strip()
    if not query_clean:
        return []

    if index is None or len(index.chunks) == 0:
        return []

    # 1. Жорсткі серверні фільтри
    safe_filters = sanitize_filters(filters)

    search_top_k = top_k or config.TOP_K
    sim_threshold = threshold if threshold is not None else SIMILARITY_THRESHOLD
    target_chunks_count = max_chunks or CONTEXT_CHUNKS

    # 2. Семантичний пошук
    query_vec = embeddings.embed_query(query_clean)
    semantic_hits = search_index(
        index=index,
        query_vector=query_vec,
        top_k=search_top_k,
        filters=safe_filters,
        threshold=sim_threshold,
    )

    # 3. Гібридний добір через BM25 (якщо доступний і потрібний для точних збігів)
    hits_by_key: dict[str, Hit] = {}
    for h in semantic_hits:
        # Гарантована вторинна перевірка безпеки
        aud = h.chunk.metadata.get("audience", "")
        stat = h.chunk.metadata.get("status", "")
        if aud and str(aud).strip().lower() == "персонал":
            continue
        if stat and str(stat).strip().lower() == "архів":
            continue

        key = f"{h.chunk.source}::{h.chunk.metadata.get('chunk_index', h.chunk.text[:50])}"
        hits_by_key[key] = h

    # Якщо семантичний пошук повернув результати, але у запиті є специфічні токени
    if keyword_index is not None:
        bm25_hits = keyword_index.bm25 and keyword.search(
            keyword_index, query_clean, top_k=3, filters=safe_filters
        ) or []
        for bh in bm25_hits:
            aud = bh.chunk.metadata.get("audience", "")
            stat = bh.chunk.metadata.get("status", "")
            if aud and str(aud).strip().lower() == "персонал":
                continue
            if stat and str(stat).strip().lower() == "архів":
                continue
            key = f"{bh.chunk.source}::{bh.chunk.metadata.get('chunk_index', bh.chunk.text[:50])}"
            if key not in hits_by_key and semantic_hits:
                # Додаємо BM25-влучення лише якщо вже є семантичне підтвердження теми,
                # щоб не тягнути спам за випадковими словами на запитах поза базою знань
                query_vec_norm = query_vec / (np.linalg.norm(query_vec) or 1.0)
                # Знаходимо його семантичний score
                chunk_idx = bh.chunk.metadata.get("chunk_index")
                if chunk_idx is not None and chunk_idx < len(index.vectors):
                    c_vec = index.vectors[chunk_idx]
                    score = float(c_vec @ query_vec_norm)
                    if sim_threshold is None or score >= sim_threshold:
                        hits_by_key[key] = Hit(chunk=bh.chunk, score=score)

    candidates = list(hits_by_key.values())
    # Сортуємо за спаданням оцінки
    candidates.sort(key=lambda x: x.score, reverse=True)

    # 4. Дедуплікація тексту фрагментів
    unique_candidates: list[Hit] = []
    seen_texts: set[str] = set()

    for h in candidates:
        norm_text = " ".join(h.chunk.text.split())
        if norm_text in seen_texts:
            continue
        seen_texts.add(norm_text)
        unique_candidates.append(h)

    # 5. Обмеження до target_chunks_count
    return unique_candidates[:target_chunks_count]


def build_context(hits: list[Hit], budget: int = CONTEXT_BUDGET) -> tuple[str, list[Source]]:
    """Зібрати з влучень текст контексту для моделі та список джерел.

    Кожен фрагмент отримує номер [1..N], назву документа, розділ,
    дату оновлення/редакції та дословний текст.

    Бюджет токенів контролюється: якщо черговий фрагмент цілком не вміщується
    в ліміт токенів, він відкидається цілком (без спотворення фактів чи умов).
    """
    if not hits:
        return "", []

    context_parts: list[str] = []
    sources: list[Source] = []
    spent_tokens = 0

    for i, hit in enumerate(hits, start=1):
        chunk = hit.chunk
        title = chunk.metadata.get("title") or chunk.source
        section = chunk.metadata.get("section") or chunk.metadata.get("heading") or "Загальне"
        updated = chunk.metadata.get("updated") or chunk.metadata.get("date") or ""

        meta_line = f"[{i}] Документ: {title}; Розділ: {section}"
        if updated:
            meta_line += f"; Редакція: {updated}"

        block = f"{meta_line}\n{chunk.text.strip()}\n"
        block_tokens = estimate_tokens(block)

        if spent_tokens + block_tokens > budget and sources:
            logger.info(
                "Фрагмент %d відкинуто через перевищення бюджету токенів: %d + %d > %d",
                i, spent_tokens, block_tokens, budget
            )
            continue

        spent_tokens += block_tokens
        context_parts.append(block)
        sources.append(Source(ref=i, chunk=chunk, score=hit.score))

    context_text = "\n".join(context_parts).strip()
    return context_text, sources
