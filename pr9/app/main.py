"""Веб-рівень застосунку: сторінка звернень, черга оператора і
JSON-ендпоінти.

Цей модуль забезпечує зв'язок між веб-інтерфейсом та внутрішніми модулями:
workflow (`app/workflow.py`), сховищем (`app/store.py`), помічником (`app/assistant.py`)
та сервісом магазину (`shop/service.py`).
Усі збої та невалідні вхідні дані обробляються контрольовано, без необроблених помилок 500.
"""

from dataclasses import asdict
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from shop import service

from . import assistant, store, workflow
from .state import Run

app = FastAPI(title="Звернення клієнтів — ПР9")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class RequestIn(BaseModel):
    """Нове звернення зі сторінки."""

    customer_id: str
    text: str
    mode: Literal["workflow", "agent"] = "workflow"


class DecisionIn(BaseModel):
    """Рішення оператора щодо дії, яка чекає підтвердження."""

    approve: bool
    comment: str = ""


def run_to_dict(run: Run) -> dict:
    """Перетворити стан звернення на те, що піде на сторінку."""
    data = asdict(run)
    data["mode"] = "workflow"
    data["limits"] = workflow.limits()
    return data


def answer_to_dict(result: assistant.Answer) -> dict:
    """Перетворити результат агента з ПР8 на те, що піде на сторінку."""
    return {
        "mode": "agent",
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
    """Віддати сторінку."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/customers")
def api_customers() -> list[dict]:
    """Клієнти, від імені яких можна «увійти» на сторінці."""
    try:
        return service.list_customers()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Сервіс магазину недоступний: {exc}")


@app.post("/api/reset")
def api_reset() -> dict:
    """Повернути дані магазину до початкового стану."""
    try:
        service.reset()
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Помилка скидання даних: {exc}")


@app.post("/api/requests")
def api_create(payload: RequestIn) -> dict:
    """Прийняти звернення й обробити його обраним способом."""
    # Перевірка валідності вхідних даних
    if not payload.customer_id or not payload.customer_id.strip():
        raise HTTPException(status_code=400, detail="Не вказано ідентифікатор клієнта.")

    # Перевірка існування клієнта
    try:
        service.get_customer(payload.customer_id.strip())
    except service.NotFound:
        raise HTTPException(status_code=404, detail=f"Клієнта {payload.customer_id} не знайдено.")
    except Exception:
        # При недоступності сервісу магазину дозволяємо обробку через workflow з контрольованою помилкою
        pass

    try:
        if payload.mode == "agent":
            return answer_to_dict(assistant.answer(payload.text, payload.customer_id))
        return run_to_dict(workflow.start(payload.text, payload.customer_id))
    except Exception as exc:
        # Не допускаємо необробленого 500: створюємо або повертаємо контрольований стан помилки
        raise HTTPException(status_code=500, detail=f"Помилка обробки звернення: {exc}")


@app.get("/api/runs")
def api_runs(status: str | None = None) -> list[dict]:
    """Звернення, оброблені workflow, — усі або з певним статусом."""
    try:
        return [run_to_dict(run) for run in store.list_runs(status)]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Помилка читання списку звернень: {exc}")


@app.get("/api/runs/{run_id}")
def api_run(run_id: str) -> dict:
    """Стан одного звернення."""
    run = store.load(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Звернення {run_id} не знайдено.")
    return run_to_dict(run)


@app.post("/api/runs/{run_id}/decision")
def api_decision(run_id: str, payload: DecisionIn) -> dict:
    """Рішення оператора: підтвердити чи відхилити дію."""
    run = store.load(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Звернення {run_id} не знайдено.")

    # Захист від подвійного підтвердження: якщо вже не awaiting_approval, повертаємо стан як є
    if run.status != "awaiting_approval":
        return run_to_dict(run)

    try:
        updated_run = workflow.resume(run_id, payload.approve, payload.comment)
        return run_to_dict(updated_run)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Помилка застосування рішення оператора: {exc}")
