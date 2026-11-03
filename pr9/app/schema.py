"""Контракт кроку розбору: що модель має витягти зі звернення клієнта
і як це перевіряється Python-кодом.

Схема — домовленість між моделлю та рештою workflow.
Наступні кроки працюють лише з тим, що пройшло перевірку тут,
і не бачать сирого тексту відповіді моделі.
"""

import json
import re
from typing import Any

import jsonschema

# JSON Schema контракту розбору звернення
_REQUEST_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {
            "type": "string",
            "enum": ["return", "cancel", "status", "other", "multiple"],
            "description": "Вид звернення: return (повернення), cancel (скасування), status (стан замовлення), other (інше/подяка/незрозуміле), multiple (декілька дій в одному зверненні).",
        },
        "order_id": {
            "type": ["string", "null"],
            "description": "Номер замовлення, якщо клієнт його явно вказав у тексті (лише цифри). Якщо клієнт не назвав номер — суворо null.",
        },
        "item": {
            "type": ["string", "null"],
            "description": "Назва або опис товару, якщо названо (наприклад, 'робот-пилосос', 'навушники Оріон', 'роутер'). Якщо не вказано — null.",
        },
        "reason": {
            "type": ["string", "null"],
            "enum": ["not_suitable", "defect", "wrong_item", "damaged", "other", None],
            "description": "Причина звернення: not_suitable (не підійшов за розміром/характеристиками/не знадобився), defect (брак, несправність, не працює/не заряджається), wrong_item (не той товар), damaged (пошкоджено під час доставки), other (інше), або null.",
        },
        "quantity": {
            "type": ["integer", "null"],
            "description": "Кількість одиниць товару (якщо зазначено, наприклад 1 або 2; якщо не вказано — null).",
        },
        "needs_clarification": {
            "type": "boolean",
            "description": "Чи бракує важливих даних для завершення дії (наприклад, невідомо який саме товар чи замовлення).",
        },
        "needs_operator": {
            "type": "boolean",
            "description": "Чи потребує звернення обовʼязкового ручного розгляду оператором (складні/нестандартні випадки, кілька суперечливих дій, звернення від імені персоналу тощо).",
        },
        "missing_data": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Перелік назв полів, яких бракує (наприклад ['order_id'], ['item']).",
        },
        "comment": {
            "type": ["string", "null"],
            "description": "Короткий витяг суті звернення або коментар клієнта.",
        },
    },
    "required": ["kind", "order_id", "item", "reason", "quantity", "needs_clarification", "needs_operator", "missing_data"],
    "additionalProperties": False,
}


def request_schema() -> dict:
    """Повернути JSON Schema результату розбору звернення."""
    return _REQUEST_SCHEMA


def clean_json_text(raw: str) -> str:
    """Очистити сирий текст від можливих markdown-огорож на зразок ```json ... ``` або вступних фраз."""
    text = raw.strip()
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start : end + 1].strip()
    return text


def validate(raw: str, original_text: str = "") -> dict:
    """Перевірити відповідь моделі на кроці розбору та валідувати її зміст кодом.

    Перевірки:
      1. Парсинг JSON.
      2. Відповідність JSON Schema.
      3. Контроль вигадування номера замовлення (галюцинацій).
      4. Перевірка на множинні дії (r14).
      5. Нормалізація типів та видалення можливих артефактів.
    """
    cleaned = clean_json_text(raw)
    try:
        data = json.loads(cleaned)
    except Exception as exc:
        raise ValueError(f"Відповідь моделі не є валідним JSON: {exc}. Текст: {raw[:100]!r}") from exc

    if not isinstance(data, dict):
        raise ValueError("Очікувався JSON-об'єкт розбору звернення.")

    # Перевірка за JSON Schema
    try:
        jsonschema.validate(instance=data, schema=_REQUEST_SCHEMA)
    except jsonschema.ValidationError as err:
        raise ValueError(f"Розбір звернення не відповідає схемі: {err.message}") from err

    # Контроль галюцинацій номера замовлення:
    # Якщо клієнт у тексті звернення не згадував номер, але модель його згенерувала:
    parsed_order_id = data.get("order_id")
    if parsed_order_id:
        digits_in_order = re.sub(r"\D", "", str(parsed_order_id))
        # Перевіряємо, чи є ці цифри в оригінальному тексті клієнта
        if digits_in_order and (digits_in_order not in original_text):
            # Модель вигадала номер замовлення! Скидаємо в None
            data["order_id"] = None
            if "order_id" not in data["missing_data"]:
                data["missing_data"].append("order_id")
            data["needs_clarification"] = True
        else:
            data["order_id"] = digits_in_order

    # Виявлення подвійних/множинних дій у тексті (наприклад "скасуйте ... і оформіть повернення")
    orig_lower = original_text.lower()
    has_cancel = any(w in orig_lower for w in ["скасуй", "скасувати", "відмініть", "відмінити"])
    has_return = any(w in orig_lower for w in ["поверн", "повернути", "повернення"])
    if has_cancel and has_return:
        data["kind"] = "multiple"
        data["needs_operator"] = True

    # Виявлення запитів щодо повернення коштів / нарахування бонусів (адміністративні)
    if "бонус" in orig_lower or "кошт" in orig_lower or "грош" in orig_lower or "менеджер" in orig_lower:
        if not (has_cancel or has_return):
            data["kind"] = "other"
            data["needs_operator"] = True

    # Нормалізація кількості
    if data.get("quantity") is None or data.get("quantity", 0) < 1:
        data["quantity"] = 1

    return data
