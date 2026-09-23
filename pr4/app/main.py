"""Веб-рівень застосунку: сторінка діалогу і JSON-ендпоінт.

Цей модуль відповідає за:
- віддавання веб-інтерфейсу (HTML-сторінки);
- валідацію вхідних даних HTTP-запиту (/api/chat);
- завантаження актуальних правил магазину з context.md;
- виклик модуля llm та повернення структурованої відповіді;
- централізовану обробку всіх помилок (таймаути, ліміти, невалідна відповідь тощо)
  зі зрозумілими HTTP-статусами для клієнта.
"""

import logging
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from . import llm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Помічник служби підтримки — ПР4",
    description="Діалоговий AI-помічник магазину «Сузірʼя» з підтримкою контексту розмови та structured output."
)

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"
CONTEXT_FILE = Path(__file__).parent.parent / "context.md"


class Turn(BaseModel):
    """Одна репліка розмови: `user` — клієнт, `assistant` — помічник."""
    role: str = Field(..., description="Роль: 'user' або 'assistant'")
    content: str = Field(..., description="Текстовий зміст повідомлення")


class ChatRequest(BaseModel):
    """Тіло запиту від фронтенду: нове звернення та історія попередніх реплік."""
    message: str = Field(..., description="Поточне повідомлення від користувача")
    history: list[Turn] = Field(default_factory=list, description="Історія розмови")


def load_context() -> str:
    """Прочитати правила організації, на підставі яких відповідає модель.
    
    Правила читаються з файлу context.md при кожному виклику, що дозволяє
    змінювати поведінку системи без перезапуску застосунку чи зміни коду.
    """
    try:
        return CONTEXT_FILE.read_text(encoding="utf-8")
    except FileNotFoundError:
        logger.critical("Файл правил context.md не знайдено за шляхом %s", CONTEXT_FILE)
        raise HTTPException(
            status_code=500,
            detail="Конфігураційна помилка: файл правил context.md не знайдено."
        )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Віддати сторінку діалогу."""
    if not INDEX_PAGE.exists():
        raise HTTPException(status_code=500, detail="Шаблон сторінки index.html не знайдено.")
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.post("/api/chat")
def api_chat(payload: ChatRequest):
    """Повернути структуровану відповідь помічника у форматі JSON.
    
    Приймає повідомлення та історію діалогу, повертає об'єкт із полями:
    - result: перевірена за схемою відповідь (reply, topic, based_on_rules тощо);
    - model: назва моделі;
    - elapsed: час виконання запиту в секундах;
    - usage: фактична кількість токенів (запит, відповідь, разом);
    - estimated_tokens: попередня оцінка токенів запиту.
    """
    clean_message = payload.message.strip()
    if not clean_message:
        raise HTTPException(status_code=400, detail="Звернення не може бути порожнім.")

    # Перевірка історії на коректність ролей
    for idx, turn in enumerate(payload.history):
        if turn.role not in ("user", "assistant"):
            raise HTTPException(
                status_code=400,
                detail=f"Некоректна роль у репліці #{idx + 1}: '{turn.role}'. Дозволені лише 'user' або 'assistant'."
            )

    history = [turn.model_dump() for turn in payload.history]
    context = load_context()

    try:
        return llm.ask(clean_message, history, context)
    except llm.LLMError as exc:
        logger.warning(
            "Помилка LLM під час обробки звернення (код %s, тип %s): %s",
            exc.status_code, exc.error_type, exc.message
        )
        raise HTTPException(status_code=exc.status_code, detail=exc.message)
    except Exception as exc:
        logger.exception("Непередбачена помилка сервера при обробці запиту: %s", exc)
        raise HTTPException(
            status_code=500,
            detail=f"Внутрішня помилка сервера під час обробки звернення: {str(exc)}"
        )
