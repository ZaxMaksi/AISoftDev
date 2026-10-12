"""Веб-рівень застосунку: сторінка помічника і JSON-ендпоінти.

Цей модуль забезпечує звʼязок між HTTP-клієнтами (веб-інтерфейсом) та RAG-конвеєром:
- віддає головну HTML-сторінку помічника;
- надає статус готовності індексу (/api/status);
- обробляє запити до помічника (/api/ask);
- забезпечує сувору валідацію вхідних даних та безпечну обробку всіх помилок
  (порожнє питання, відсутній індекс, невідомі фільтри, таймаути та ліміти LLM API).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import config, index, keyword, llm, rag

logger = logging.getLogger(__name__)

app = FastAPI(title="Помічник за базою знань — ПР6")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class AskRequest(BaseModel):
    """Модель вхідного HTTP-запиту від сторінки помічника."""

    question: str = Field(..., description="Питання користувача українською мовою")
    filters: dict[str, Any] = Field(default_factory=dict, description="Фільтри пошуку (наприклад, product)")


def hit_to_dict(hit: index.Hit) -> dict[str, Any]:
    """Серіалізація фрагмента для діагностичного блоку 'retrieved'."""
    return {
        "score": hit.score,
        "text": hit.chunk.text,
        "source": hit.chunk.source,
        "metadata": hit.chunk.metadata,
    }


def answer_to_dict(result: rag.Answer) -> dict[str, Any]:
    """Перетворити результат конвеєра на структуру, яку очікує веб-сторінка."""
    return {
        "answer": result.text,
        "found": result.found,
        "sources": [
            {
                "ref": s.ref,
                "score": s.score,
                "source": s.chunk.source,
                "metadata": s.chunk.metadata,
                "text": s.chunk.text,
            }
            for s in result.sources
        ],
        "retrieved": [hit_to_dict(h) for h in result.retrieved],
        "model": result.model,
        "elapsed": result.elapsed,
        "usage": result.usage,
    }


@app.on_event("startup")
def load_indexes() -> None:
    """Завантажити пошуковий індекс із диска при старті сервера.

    Якщо індекс не знайдено або пошкоджено — сервер все одно запускається,
    але сигналізує про необхідність запуску ingest.py через /api/status.
    """
    app.state.index = None
    app.state.keyword_index = None
    try:
        app.state.index = index.load(path=config.INDEX_DIR)
        app.state.keyword_index = keyword.build(app.state.index.chunks)
        logger.info("Індекс успішно завантажено (%d фрагментів)", len(app.state.index))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Індекс не завантажено: %s: %s", type(exc).__name__, exc)


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Віддати HTML-сторінку помічника."""
    if not INDEX_PAGE.exists():
        raise HTTPException(status_code=404, detail="Файл шаблону index.html не знайдено")
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/status")
def api_status() -> dict[str, Any]:
    """Стан індексу: готовність, кількість фрагментів, кількість документів, назва моделі."""
    idx = getattr(app.state, "index", None)
    if idx is None:
        return {
            "ready": False,
            "hint": "Індекс не збудовано — виконайте у терміналі: python ingest.py",
        }
    sources = {chunk.source for chunk in idx.chunks}
    return {
        "ready": True,
        "chunks": len(idx),
        "documents": len(sources),
        "model": idx.model_name,
    }


@app.post("/api/ask")
def api_ask(payload: AskRequest) -> dict[str, Any]:
    """Обробити питання клієнта та повернути структуровану відповідь із джерелами."""
    # 1. Перевірка наявності індексу
    idx = getattr(app.state, "index", None)
    if idx is None:
        raise HTTPException(
            status_code=503,
            detail="Пошуковий індекс не готовий або відсутній. Будь ласка, виконайте: python ingest.py",
        )

    # 2. Валідація питання
    q_clean = payload.question.strip() if payload.question else ""
    if not q_clean:
        raise HTTPException(status_code=400, detail="Питання не може бути порожнім.")

    # 3. Валідація фільтрів
    if payload.filters:
        for k in payload.filters:
            if k not in config.ALLOWED_FILTERS and k not in {"audience", "status"}:
                raise HTTPException(
                    status_code=400,
                    detail=f"Невідомий фільтр: '{k}'. Дозволені фільтри: {', '.join(sorted(config.ALLOWED_FILTERS))}",
                )

    # 4. Виклик RAG-конвеєра з перехопленням можливих збоїв
    try:
        kw_idx = getattr(app.state, "keyword_index", None)
        result = rag.answer(
            question=q_clean,
            index=idx,
            keyword_index=kw_idx,
            filters=payload.filters or None,
        )
        return answer_to_dict(result)
    except llm.LLMError as exc:
        logger.error("LLM Error під час обробки питання: %s", exc)
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
    except Exception as exc:
        logger.exception("Непередбачена помилка конвеєра: %s", exc)
        raise HTTPException(
            status_code=500,
            detail="Внутрішня помилка під час формування відповіді. Спробуйте пізніше.",
        ) from exc
