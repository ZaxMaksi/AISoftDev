"""Контракт відповіді помічника: Pydantic-схема та програмна валідація.

Цей модуль визначає структуру даних, яку повинна повертати мовна модель,
генерує JSON Schema для запиту (structured output) та забезпечує сувору
валідацію сирої відповіді перед передачею на веб-рівень.
"""

from enum import Enum
import json
import logging
import re
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError

logger = logging.getLogger(__name__)


class TopicEnum(str, Enum):
    """Фіксований перелік тем звернення клієнта.
    
    Теми відповідають розділам правил з context.md та попередньої практичної роботи.
    """
    order = "order"          # Замовлення (оформлення, зміна, скасування)
    delivery = "delivery"    # Доставка (терміни, вартість, перевізники)
    payment = "payment"      # Оплата (способи оплати, накладений платіж)
    return_ = "return"       # Повернення (14 днів, належна/неналежна якість)
    warranty = "warranty"    # Гарантія (строк 12 місяців, умови обслуговування)
    support = "support"      # Підтримка (графік роботи, контакти)
    other = "other"          # Інше (поза правилами магазину, некоректний ввід тощо)


class AssistantResponse(BaseModel):
    """Структурована відповідь помічника служби підтримки."""
    model_config = ConfigDict(extra="forbid")

    reply: str = Field(
        ...,
        description="Ввічлива, точна та вичерпна відповідь клієнту українською мовою."
    )
    topic: TopicEnum = Field(
        ...,
        description="Тема звернення клієнта з фіксованого переліку категорій."
    )
    based_on_rules: bool = Field(
        ...,
        description=(
            "True, якщо відповідь безпосередньо ґрунтується на правилах магазину з context.md; "
            "False, якщо інформації в правилах немає або питання виходить за їхні межі."
        )
    )
    needs_clarification: bool = Field(
        ...,
        description="True, якщо наданої клієнтом інформації недостатньо і потрібно уточнити деталі; інакше False."
    )
    transfer_to_operator: bool = Field(
        ...,
        description="True, якщо питання потребує передачі людині-оператору (наприклад, питання поза правилами чи специфічна проблема)."
    )
    order_number: Optional[str] = Field(
        default=None,
        description="Номер замовлення (6 цифр), якщо він згадувався користувачем у розмові; інакше null."
    )


class SchemaValidationError(Exception):
    """Помилка валідації сирої відповіді моделі за схемою."""
    def __init__(self, message: str, raw: str):
        super().__init__(message)
        self.message = message
        self.raw = raw


def output_schema() -> dict:
    """Повернути JSON Schema очікуваної структури відповіді для моделі."""
    return AssistantResponse.model_json_schema()


def validate(raw: str) -> dict:
    """Перевірити сиру відповідь моделі й повернути безпечний словник даних.
    
    Перевіряє:
    1. Наявність непорожнього тексту.
    2. Коректність синтаксису JSON (включно зі зняттям markdown-огорож або виділенням JSON-блоку).
    3. Відповідність Pydantic-моделі (наявність обов'язкових полів, валідність enum topic, типів).
    4. Коректність формату номера замовлення (якщо названо — має складатись із 6 цифр).
    """
    if not raw or not raw.strip():
        raise SchemaValidationError("Порожня відповідь від моделі", raw=raw or "")

    cleaned = raw.strip()

    # Зняття можливих markdown блоків ```json ... ```
    if "```" in cleaned:
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        # Спроба знайти перший { і останній } якщо модель додала супровідний текст
        first_brace = cleaned.find("{")
        last_brace = cleaned.rfind("}")
        if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
            sub = cleaned[first_brace : last_brace + 1]
            try:
                data = json.loads(sub)
            except json.JSONDecodeError as exc2:
                logger.warning("Синтаксична помилка JSON у відповіді моделі: %s | Raw: %r", exc2, raw)
                raise SchemaValidationError(f"Синтаксична помилка JSON: {exc2}", raw=raw) from exc2
        else:
            logger.warning("Синтаксична помилка JSON у відповіді моделі: %s | Raw: %r", exc, raw)
            raise SchemaValidationError(f"Синтаксична помилка JSON: {exc}", raw=raw) from exc

    if not isinstance(data, dict):
        raise SchemaValidationError(
            f"Очікувався JSON-об'єкт, отримано {type(data).__name__}", raw=raw
        )

    try:
        parsed = AssistantResponse.model_validate(data)
    except ValidationError as exc:
        logger.warning("Помилка Pydantic-валідації схеми відповіді: %s | Data: %r", exc, data)
        raise SchemaValidationError(f"Невідповідність схемі відповіді: {exc.errors()}", raw=raw) from exc

    result = parsed.model_dump(mode="json")

    # Додаткова перевірка формату номера замовлення (відповідно до context.md — 6 цифр)
    if result.get("order_number") is not None:
        raw_order = str(result["order_number"]).strip()
        digits = re.findall(r"\b\d{6}\b", raw_order)
        if digits:
            result["order_number"] = digits[0]
        elif re.fullmatch(r"\d{6}", raw_order):
            result["order_number"] = raw_order
        else:
            # Якщо клієнт не назвав 6-значний номер або модель підставила невалідний рядок
            logger.info("Номер замовлення %r не відповідає формату 6 цифр — скинуто в null", raw_order)
            result["order_number"] = None

    return result
