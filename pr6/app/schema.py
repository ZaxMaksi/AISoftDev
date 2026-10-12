"""Контракт відповіді моделі: Pydantic-схема та програмна валідація.

Цей модуль визначає структуру даних, яку повинна повертати мовна модель,
генерує JSON Schema для запиту та забезпечує сувору валідацію сирої
відповіді перед передачею в RAG-конвейер.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)


class RAGResponse(BaseModel):
    """Схема структурованої відповіді RAG-помічника."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        ...,
        description=(
            "Вичерпна, точна та ввічлива відповідь клієнту українською мовою. "
            "Ґрунтується ВИКЛЮЧНО на наданих фрагментах контексту. "
            "Якщо інформації недостатньо або її немає в базі знань, містить пряме повідомлення про це."
        ),
    )
    found: bool = Field(
        ...,
        description=(
            "True, якщо відповідь (або її суттєва частина) знайдена в наданих фрагментах бази знань; "
            "False, якщо в наданих фрагментах немає інформації для відповіді на питання."
        ),
    )
    sources: list[int] = Field(
        default_factory=list,
        description=(
            "Список номерів джерел [1..N] із наданого контексту, на які спирається відповідь. "
            "Якщо found=false або джерел немає, список має бути порожнім []."
        ),
    )


class SchemaValidationError(Exception):
    """Помилка валідації сирої відповіді моделі за схемою."""

    def __init__(self, message: str, raw: str = ""):
        super().__init__(message)
        self.message = message
        self.raw = raw


def output_schema() -> dict[str, Any]:
    """Повернути JSON Schema відповіді моделі для передачі провайдеру."""
    return RAGResponse.model_json_schema()


def validate(raw: str) -> dict[str, Any]:
    """Перевірити сиру відповідь моделі й повернути безпечний словник даних.

    Перевіряє:
    1. Наявність непорожнього тексту.
    2. Коректність синтаксису JSON (включно зі зняттям markdown-огорож ```json ... ```).
    3. Відповідність схемі RAGResponse через Pydantic (answer, found, sources).
    """
    if not raw or not raw.strip():
        raise SchemaValidationError("Порожня відповідь від мовної моделі", raw=raw or "")

    cleaned = raw.strip()

    # Зняття можливих markdown блоків ```json ... ``` або ``` ... ```
    if "```" in cleaned:
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Спроба знайти перший валідний JSON-об'єкт всередині тексту, якщо модель додала коментарі
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                raise SchemaValidationError(f"Невалідний JSON у відповіді моделі: {exc}", raw=raw) from exc
        else:
            raise SchemaValidationError(f"Відповідь моделі не є валідним JSON: {exc}", raw=raw) from exc

    if not isinstance(data, dict):
        raise SchemaValidationError(
            f"Очікувався JSON-об'єкт (dict), отримано {type(data).__name__}",
            raw=raw,
        )

    try:
        validated_obj = RAGResponse.model_validate(data)
        return validated_obj.model_dump()
    except ValidationError as exc:
        raise SchemaValidationError(
            f"Невідповідність схемі відповіді: {exc}",
            raw=raw,
        ) from exc
