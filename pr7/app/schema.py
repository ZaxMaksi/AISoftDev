"""Контракт відповіді моделі: які поля вона має повернути й як це
перевіряється.

Описуємо схему через JSON Schema і перевіряємо через jsonschema.
Також надаємо рядок-опис схеми для вставки в системне повідомлення моделі.
"""

import json

import jsonschema

# JSON Schema відповіді моделі
_SCHEMA = {
    "type": "object",
    "properties": {
        "document_type": {
            "description": "Тип документа: 'рахунок' або інший тип (наприклад, 'накладна')",
            "type": ["string", "null"]
        },
        "number": {
            "description": "Номер рахунку",
            "type": ["string", "null"]
        },
        "date": {
            "description": "Дата рахунку у форматі РРРР-ММ-ДД",
            "type": ["string", "null"]
        },
        "valid_until": {
            "description": "Строк дії рахунку у форматі РРРР-ММ-ДД. Null якщо не вказано в документі.",
            "type": ["string", "null"]
        },
        "supplier": {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
                "code": {
                    "description": "ЄДРПОУ (8 цифр) або РНОКПП (10 цифр)",
                    "type": ["string", "null"]
                },
                "iban": {
                    "description": "IBAN без пробілів",
                    "type": ["string", "null"]
                }
            }
        },
        "buyer": {
            "type": "object",
            "properties": {
                "name": {"type": ["string", "null"]},
                "code": {"type": ["string", "null"]}
            }
        },
        "items": {
            "type": ["array", "null"],
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "unit": {"type": ["string", "null"]},
                    "quantity": {"type": ["number", "null"]},
                    "price": {"type": ["string", "number", "null"]},
                    "amount": {"type": ["string", "number", "null"]}
                }
            }
        },
        "total_without_vat": {
            "description": "Сума без ПДВ",
            "type": ["string", "number", "null"]
        },
        "vat": {
            "description": "Сума ПДВ",
            "type": ["string", "number", "null"]
        },
        "total": {
            "description": "Загальна сума до оплати",
            "type": ["string", "number", "null"]
        }
    },
    "required": ["document_type"]
}


def output_schema() -> dict:
    """Повернути JSON Schema відповіді моделі."""
    return _SCHEMA


def schema_description() -> str:
    """Повернути текстовий опис схеми для вставки в інструкцію моделі."""
    return """
Поверни JSON з наступними полями:

{
  "document_type": "рахунок" | "накладна" | інший рядок | null,
  "number": "номер рахунку" | null,
  "date": "РРРР-ММ-ДД" | null,
  "valid_until": "РРРР-ММ-ДД" | null,
  "supplier": {
    "name": "назва постачальника" | null,
    "code": "ЄДРПОУ (8 цифр) або РНОКПП (10 цифр)" | null,
    "iban": "IBAN без пробілів, наприклад UA663510050000026004017723561" | null
  },
  "buyer": {
    "name": "назва покупця" | null,
    "code": "код ЄДРПОУ або РНОКПП" | null
  },
  "items": [
    {
      "name": "назва позиції" | null,
      "unit": "одиниця виміру" | null,
      "quantity": число | null,
      "price": "ціна як рядок з двома знаками після крапки" | null,
      "amount": "сума як рядок з двома знаками після крапки" | null
    }
  ],
  "total_without_vat": "сума без ПДВ як рядок" | null,
  "vat": "ПДВ як рядок" | null,
  "total": "загальна сума до оплати як рядок" | null
}
"""


class SchemaError(Exception):
    """Відповідь моделі не пройшла перевірку схемою."""


def validate(raw: str) -> dict:
    """Перевірити сиру відповідь моделі й повернути дані, яким можна
    довіряти структурно.

    Якщо текст не є JSON або не проходить схему — підняти SchemaError.
    """
    # Парсинг JSON
    # Модель іноді обгортає JSON у markdown-блок ```json ... ```
    text = raw.strip()
    if text.startswith("```"):
        # Видалити markdown-обгортку
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise SchemaError(f"Відповідь моделі не є коректним JSON: {e}")

    if not isinstance(data, dict):
        raise SchemaError(
            f"Очікувався JSON-об'єкт, отримано: {type(data).__name__}"
        )

    # Перевірка за JSON Schema
    try:
        jsonschema.validate(data, _SCHEMA)
    except jsonschema.ValidationError as e:
        raise SchemaError(f"Відповідь не відповідає схемі: {e.message}")
    except jsonschema.SchemaError as e:
        raise SchemaError(f"Помилка схеми: {e.message}")

    return data
