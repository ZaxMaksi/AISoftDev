"""Веб-рівень застосунку: сторінка пошуку та JSON API.

Цей модуль не знає ані якою моделлю отримано вектори, ані як влаштований
індекс, ані як ранжує пошук за словами. Тут валідуються запити,
викликаються сервіси пошуку та формуються HTTP-відповіді.
"""

from __future__ import annotations
import time
from pathlib import Path
from typing import Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from . import config, embeddings, index, keyword

app = FastAPI(
    title="Пошук у базі знань — ПР5",
    description="Семантичний пошук та пошук за ключовими словами (BM25) магазину «Сузірʼя»",
)

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class SearchRequest(BaseModel):
    """Схема пошукового запиту."""

    query: str
    mode: str = "both"
    top_k: int = Field(default=config.TOP_K)
    filters: dict[str, Any] = Field(default_factory=dict)
    threshold: float | None = None
    # Опціональні прямі поля для фільтрів згідно зі специфікацією API
    category: str | None = None
    product: str | None = None
    audience: str | None = None
    status: str | None = None


def hit_to_dict(hit: index.Hit) -> dict[str, Any]:
    """Перетворити влучення на структуру для JSON-відповіді."""
    return {
        "score": hit.score,
        "text": hit.chunk.text,
        "source": hit.chunk.source,
        "metadata": hit.chunk.metadata,
    }


@app.on_event("startup")
def load_indexes() -> None:
    """Завантажити побудований індекс із диска при старті застосунку.

    Ембедінги НЕ перераховуються. Якщо індекс не знайдено, застосунок не падає,
    а зберігає зрозумілу підказку для користувача.
    """
    app.state.index = None
    app.state.keyword_index = None
    app.state.index_error = None

    try:
        loaded_index = index.load()
        app.state.index = loaded_index
        app.state.keyword_index = keyword.build(loaded_index.chunks)
    except FileNotFoundError:
        app.state.index_error = "Search index is not built. Run: python scripts/build_index.py"
        print(f"\n[УВАГА] {app.state.index_error}\n")
    except Exception as exc:
        app.state.index_error = f"Помилка завантаження індексу ({type(exc).__name__}): {exc}"
        print(f"\n[ПОМИЛКА] {app.state.index_error}\n")


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Віддати HTML-сторінку веб-інтерфейсу пошуку."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/status")
def api_status() -> dict[str, Any]:
    """Стан індексу: готовність, кількість фрагментів, документів та модель."""
    idx = getattr(app.state, "index", None)
    if idx is None:
        hint = getattr(app.state, "index_error", None) or "Search index is not built. Run: python scripts/build_index.py"
        return {"ready": False, "hint": hint}

    sources = {chunk.source for chunk in idx.chunks}
    return {
        "ready": True,
        "chunks": len(idx),
        "documents": len(sources),
        "model": idx.model_name,
    }


@app.post("/api/search")
@app.post("/search")
def api_search(payload: SearchRequest) -> dict[str, Any]:
    """Виконати семантичний або ключовий пошук та повернути результати."""
    idx = getattr(app.state, "index", None)
    kw_idx = getattr(app.state, "keyword_index", None)

    # 1. Перевірка наявності збудованого індексу
    if idx is None or kw_idx is None:
        hint = getattr(app.state, "index_error", None) or "Search index is not built. Run: python scripts/build_index.py"
        raise HTTPException(status_code=400, detail=hint)

    # 2. Валідація порожнього запиту
    query_text = payload.query.strip()
    if not query_text:
        raise HTTPException(status_code=400, detail="Пошуковий запит не може бути порожнім.")

    # 3. Валідація режиму
    if payload.mode not in ("semantic", "keyword", "both"):
        raise HTTPException(
            status_code=400,
            detail=f"Невідомий режим пошуку: '{payload.mode}'. Дозволені значення: 'semantic', 'keyword', 'both'.",
        )

    # 4. Валідація top_k
    if payload.top_k < 1 or payload.top_k > 100:
        raise HTTPException(status_code=400, detail="Параметр top_k повинен бути цілим числом від 1 до 100.")

    # 5. Валідація threshold
    if payload.threshold is not None:
        if payload.threshold < -1.0 or payload.threshold > 1.0:
            raise HTTPException(
                status_code=400,
                detail="Параметр threshold повинен бути числом у діапазоні від -1.0 до 1.0.",
            )

    # 6. Збирання та валідація фільтрів
    combined_filters = dict(payload.filters or {})
    for f_field in ("category", "product", "audience", "status"):
        val = getattr(payload, f_field, None)
        if val is not None and f_field not in combined_filters:
            combined_filters[f_field] = val

    # Перевірка на невідомі фільтри
    allowed_filters = config.ALLOWED_FILTERS
    for f_key in combined_filters:
        if f_key not in allowed_filters:
            raise HTTPException(
                status_code=400,
                detail=f"Невідомий фільтр: '{f_key}'. Дозволені фільтри: category, product, audience, status.",
            )

    result: dict[str, Any] = {
        "query": query_text,
        "semantic": None,
        "keyword": None,
        "elapsed": {},
    }

    # Виконання семантичного пошуку
    if payload.mode in ("semantic", "both"):
        t0 = time.perf_counter()
        q_vec = embeddings.embed_query(query_text)
        eff_threshold = payload.threshold if payload.threshold is not None else config.SEARCH_THRESHOLD
        hits_sem = index.search(
            idx,
            query_vector=q_vec,
            top_k=payload.top_k,
            filters=combined_filters or None,
            threshold=eff_threshold,
        )
        t_sem = time.perf_counter() - t0
        result["elapsed"]["semantic"] = round(t_sem, 4)
        result["semantic"] = [hit_to_dict(h) for h in hits_sem]

    # Виконання пошуку за ключовими словами (BM25)
    if payload.mode in ("keyword", "both"):
        t0 = time.perf_counter()
        hits_kw = keyword.search(
            kw_idx,
            query=query_text,
            top_k=payload.top_k,
            filters=combined_filters or None,
        )
        t_kw = time.perf_counter() - t0
        result["elapsed"]["keyword"] = round(t_kw, 4)
        result["keyword"] = [hit_to_dict(h) for h in hits_kw]

    return result
