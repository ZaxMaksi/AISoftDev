"""Веб-рівень застосунку: сторінка помічника і JSON-ендпоінти.

Цей файл не знає ані які інструменти є в моделі, ані як перевіряються
аргументи, ані якою моделлю й за якою інструкцією отримано відповідь —
усе це в `app/tools.py`, `app/llm.py` і поєднується в `app/assistant.py`.
"""

import logging
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from shop import service

from . import assistant, tools
from .llm import LLMError

logger = logging.getLogger(__name__)

app = FastAPI(title="Помічник клієнта — ПР8")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class AskRequest(BaseModel):
    """Те, що надсилає сторінка.

    `customer_id` — клієнт, обраний на сторінці (імітація сесії).
    """

    customer_id: str
    question: str


def answer_to_dict(result: assistant.Answer) -> dict:
    """Перетворити результат циклу на те, що піде на сторінку."""
    return {
        "answer": result.text,
        "calls": [asdict(call) for call in result.calls],
        "rounds": result.rounds,
        "stopped": result.stopped,
        "model": result.model,
        "elapsed": result.elapsed,
        "usage": result.usage,
    }


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Віддати сторінку помічника."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/customers")
def api_customers() -> list[dict]:
    """Клієнти, від імені яких можна «увійти» на сторінці."""
    return service.list_customers()


@app.get("/api/tools")
def api_tools() -> list[dict]:
    """Описи інструментів у тому вигляді, в якому їх бачить модель."""
    return tools.specs()


@app.post("/api/reset")
def api_reset() -> dict:
    """Повернути дані магазину до початкового стану."""
    service.reset()
    return {"ok": True}


@app.post("/api/ask")
def api_ask(payload: AskRequest) -> JSONResponse:
    """Відповісти на питання клієнта й повернути відповідь із журналом викликів."""
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="Питання не може бути порожнім.")

    customer_id = (payload.customer_id or "").strip()
    if not customer_id:
        raise HTTPException(status_code=400, detail="Не обрано клієнта для сеансу.")

    # Перевірка наявності клієнта в сервісі
    all_customers = {c["customer_id"] for c in service.list_customers()}
    if customer_id not in all_customers:
        raise HTTPException(status_code=404, detail=f"Клієнта з ID {customer_id!r} не знайдено.")

    try:
        result = assistant.answer(question=question, customer_id=customer_id)
        return JSONResponse(status_code=200, content=answer_to_dict(result))
    except LLMError as e:
        logger.error("Помилка моделі: %s", e)
        # Повертаємо контрольовану відповідь клієнтові без падіння з 500
        safe_answer = assistant.Answer(
            text=f"На жаль, виникла проблема зі зв'язком із мовною моделлю: {e}. Будь ласка, спробуйте пізніше.",
            rounds=0,
            stopped="error",
            elapsed={"model": 0.0, "tools": 0.0},
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        return JSONResponse(status_code=200, content=answer_to_dict(safe_answer))
    except Exception as e:
        logger.error("Непередбачена помилка помічника: %s", e, exc_info=True)
        safe_answer = assistant.Answer(
            text="Вибачте, сталася внутрішня помилка під час обробки вашого запиту. Спробуйте пізніше.",
            rounds=0,
            stopped="error",
            elapsed={"model": 0.0, "tools": 0.0},
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )
        return JSONResponse(status_code=200, content=answer_to_dict(safe_answer))
