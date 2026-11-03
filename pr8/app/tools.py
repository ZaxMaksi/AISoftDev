"""Інструменти помічника: що модель може попросити виконати і як це
перевіряється.

Єдине місце застосунку, яке знає і про модель, і про сервіс магазину
(`shop/service.py`). Тут описано інструменти для моделі, тут перевіряють
аргументи, які вона повернула, і тут викликають сервіс. Модуль моделі
(`app/llm.py`) не знає, які інструменти існують, а сервіс не знає, що
його викликає модель.
"""

import copy
import json
import re
from dataclasses import dataclass
from typing import Any

import jsonschema
from shop import service
from shop.service import InvalidRequest, NotFound, PolicyViolation, ShopError, Unavailable


@dataclass
class Context:
    """Те, що відомо про сеанс незалежно від моделі.

    `customer_id` — клієнт, який увійшов. У справжньому застосунку його
    бере веб-рівень із сесії після автентифікації; тут його обирають на
    сторінці. Модель цього поля не бачить і змінити не може.
    """

    customer_id: str


@dataclass
class ToolResult:
    """Підсумок одного виклику інструмента.

    `status` — що сталося: `ok` — виконано; `rejected` —
    виклик не пройшов перевірок безпеки, сервіс не викликався; `error` —
    сервіс відмовив або не відповів.
    `content` — те, що піде моделі як результат виклику.
    `reason` — пояснення для журналу й сторінки.
    `arguments` — аргументи після розбору.
    """

    status: str
    content: dict | list | str | None = None
    reason: str | None = None
    arguments: dict | None = None


# ---------------------------------------------------------------------------
# Описи інструментів (JSON Schema) для моделі
# ---------------------------------------------------------------------------

_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "list_orders",
            "description": (
                "Отримати список усіх замовлень поточного клієнта (номер замовлення, "
                "дата створення, статус, сума, кількість товарів). Викликайте, коли клієнт "
                "запитує про свої замовлення загалом або не назвав номер конкретного замовлення."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": (
                "Отримати детальну інформацію про конкретне замовлення клієнта за номером: "
                "статус, товари, доставка, номер накладної (ТТН). "
                "Викликайте лише якщо клієнт вказав конкретний номер замовлення."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Номер замовлення, наприклад '10458' або '10463'."
                    }
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": (
                "Пошук товарів у каталозі магазину за ключовими словами в назві або описі. "
                "Повертає короткі картки знайдених товарів: артикул (sku), назву, категорію та ціну."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Пошуковий запит (наприклад, 'роутер Альтаїр' або 'навушники Оріон')."
                    },
                    "category": {
                        "type": "string",
                        "description": "Необов'язкова категорія (наприклад, 'Аудіо', 'Мережеве обладнання')."
                    },
                    "max_price": {
                        "type": "number",
                        "description": "Необов'язкова максимальна ціна в гривнях."
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "description": "Максимальна кількість результатів (від 1 до 50)."
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_product",
            "description": (
                "Отримати повну картку товару за його артикулом (SKU): назва, ціна, вага, "
                "гарантійний строк у місяцях, чи підлягає поверненню товар належної якості, та опис."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "Артикул товару (SKU), наприклад 'NET-ALTAIR-AX3' або 'AUD-ORION-X2-B'."
                    }
                },
                "required": ["sku"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock",
            "description": (
                "Перевірити наявність товару на складі за його артикулом (SKU): "
                "кількість доступних одиниць та очікувану дату надходження, якщо товару немає."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {
                        "type": "string",
                        "description": "Артикул товару (SKU)."
                    }
                },
                "required": ["sku"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "delivery_quote",
            "description": (
                "Розрахувати вартість і строки доставки набору товарів у вказане місто України. "
                "Магазин доставляє лише по Україні. Способи доставки: 'branch' (відділення), "
                "'courier' (кур'єр), 'pickup' (самовивіз зі складу в Кременчуці)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Місто доставки в Україні (наприклад, 'Львів', 'Полтава', 'Київ')."
                    },
                    "method": {
                        "type": "string",
                        "enum": ["branch", "courier", "pickup"],
                        "description": "Спосіб доставки: 'branch' (відділення), 'courier' (кур'єр) або 'pickup' (самовивіз)."
                    },
                    "items": {
                        "type": "array",
                        "description": "Список товарів для розрахунку.",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sku": {"type": "string", "description": "Артикул товару"},
                                "quantity": {"type": "integer", "minimum": 1, "description": "Кількість одиниць"}
                            },
                            "required": ["sku"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["city", "method", "items"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_return",
            "description": (
                "Оформити заявку на повернення товару із замовлення. "
                "Викликайте цей інструмент завжди, коли клієнт просить оформити повернення товару. "
                "Сервіс магазину самостійно здійснить усі перевірки (статус замовлення, право на повернення товару належної якості, гарантійні строки). "
                "Допустимі причини повернення: 'not_suitable' (товар належної якості не підійшов — протягом 14 днів), "
                "'defect' (виробничий дефект — протягом гарантії), "
                "'wrong_item' (товар не відповідає замовленню), 'damaged' (пошкоджено під час доставки)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Номер замовлення, наприклад '10412'."
                    },
                    "sku": {
                        "type": "string",
                        "description": "Артикул товару (SKU), який повертається."
                    },
                    "reason": {
                        "type": "string",
                        "enum": ["not_suitable", "defect", "wrong_item", "damaged"],
                        "description": "Причина повернення."
                    },
                    "quantity": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Кількість одиниць для повернення (за замовчуванням 1)."
                    },
                    "comment": {
                        "type": "string",
                        "description": "Необов'язковий коментар клієнта."
                    },
                },
                "required": ["order_id", "sku", "reason"],
                "additionalProperties": False,
            },
        },
    },
]

ALLOWED_TOOL_NAMES = {spec["function"]["name"] for spec in _SPECS}


def specs() -> list[dict]:
    """Повернути описи інструментів у форматі, який приймає API моделі."""
    return copy.deepcopy(_SPECS)


# ---------------------------------------------------------------------------
# Допоміжні функції перевірки та мінімізації даних
# ---------------------------------------------------------------------------

def normalize_order_id(raw_id: Any) -> str:
    """Звести номер замовлення до цифрового рядка: '№ 10 458' -> '10458'."""
    if raw_id is None:
        return ""
    digits = re.sub(r"\D", "", str(raw_id))
    return digits if digits else str(raw_id).strip()


def sanitize_order_for_client(order: dict) -> dict:
    """Мінімізація даних замовлення: видалення службових приміток оператора
    та внутрішніх реквізитів оплати."""
    clean = copy.deepcopy(order)
    # 1. Приховуємо службові та внутрішні примітки оператора
    clean.pop("internal_note", None)
    clean.pop("operator_notes", None)
    # 2. Приховуємо платіжні реквізити (повні номери рахунків, внутрішні id)
    if "payment" in clean and isinstance(clean["payment"], dict):
        clean["payment"] = {
            "method": clean["payment"].get("method"),
            "status": clean["payment"].get("status") or ("paid" if clean["payment"].get("paid") else "unpaid"),
            "card_last4": clean["payment"].get("card_last4"),
        }
    return clean


# ---------------------------------------------------------------------------
# Виконання та перевірка виклику інструмента
# ---------------------------------------------------------------------------

def call(name: str, raw_arguments: str, ctx: Context) -> ToolResult:
    """Виконати виклик інструмента, який запропонувала модель.

    Перевірки:
      1. Назва інструмента в реєстрі дозволених (заборона адміністративних функцій).
      2. Валідність JSON у рядку аргументів.
      3. Відповідність аргументів схемі.
      4. Приналежність замовлення поточному клієнту (рівень володіння).
      5. Виконання та обробка помилок сервісу.
    """
    # 1. Перевірка назви інструмента
    if name not in ALLOWED_TOOL_NAMES:
        return ToolResult(
            status="rejected",
            reason=f"інструмент «{name}» не дозволений для клієнтського помічника або не існує",
            content={
                "error": "forbidden_tool",
                "message": (
                    f"Операція «{name}» заборонена або не підтримується. "
                    "Клієнтський помічник не має доступу до адміністративних дій чи невідомих інструментів."
                ),
            },
        )

    # 2. Розбір JSON
    try:
        if isinstance(raw_arguments, dict):
            args = raw_arguments
        elif not raw_arguments or not raw_arguments.strip():
            args = {}
        else:
            args = json.loads(raw_arguments)
    except Exception as e:
        return ToolResult(
            status="rejected",
            reason="аргументи виклику не є валідним JSON",
            content={"error": "invalid_json", "message": f"Не вдалося розібрати JSON аргументів: {e}"},
        )

    if not isinstance(args, dict):
        return ToolResult(
            status="rejected",
            reason="аргументи мають бути JSON-обʼєктом",
            content={"error": "invalid_arguments", "message": "Очікувався JSON-об'єкт аргументів."},
            arguments={"raw": raw_arguments},
        )

    # Попередня нормалізація типів (наприклад, числовий order_id у рядок)
    if "order_id" in args and isinstance(args["order_id"], int):
        args["order_id"] = str(args["order_id"])

    # 3. Перевірка за схемою JSON Schema (типи, обовʼязкові поля, заборона зайвих полів)
    spec = next((s for s in _SPECS if s["function"]["name"] == name), None)
    if spec:
        param_schema = spec["function"].get("parameters")
        if param_schema:
            try:
                jsonschema.validate(instance=args, schema=param_schema)
            except jsonschema.ValidationError as err:
                return ToolResult(
                    status="rejected",
                    reason=f"аргументи не відповідають схемі: {err.message}",
                    content={
                        "error": "schema_validation_error",
                        "message": f"Аргументи інструмента '{name}' не відповідають схемі: {err.message}",
                    },
                    arguments=args,
                )

    # 4. Виконання конкретного інструмента з перевірками бізнес-правил та безпеки
    try:
        if name == "list_orders":
            # Клієнт визначається ТІЛЬКИ із сесії (ctx.customer_id), аргументи моделі ігноруються
            orders = service.list_orders(ctx.customer_id)
            return ToolResult(
                status="ok",
                content=orders,
                arguments={},
            )

        elif name == "get_order":
            raw_order_id = args.get("order_id")
            if not raw_order_id:
                return ToolResult(
                    status="rejected",
                    reason="відсутній обовʼязковий параметр order_id",
                    content={"error": "missing_parameter", "message": "Не вказано номер замовлення."},
                    arguments=args,
                )
            order_id = normalize_order_id(raw_order_id)

            # Перевірка існування та приналежності клієнту до виклику сервісу
            order_record = service._state.get("orders", {}).get(order_id)
            if order_record is not None and str(order_record.get("customer_id")) != str(ctx.customer_id):
                return ToolResult(
                    status="rejected",
                    reason="замовлення не належить поточному клієнтові",
                    content={
                        "error": "access_denied",
                        "message": (
                            f"Замовлення {order_id} належить іншому клієнту. "
                            "Доступ заборонено з міркувань безпеки."
                        ),
                    },
                    arguments={"order_id": order_id},
                )

            # Отримуємо замовлення через сервіс
            full_order = service.get_order(order_id)

            # Повторна перевірка власника після виклику
            if str(full_order.get("customer_id")) != str(ctx.customer_id):
                return ToolResult(
                    status="rejected",
                    reason="замовлення не належить поточному клієнтові",
                    content={
                        "error": "access_denied",
                        "message": "Замовлення належить іншому клієнту. Доступ заборонено.",
                    },
                    arguments={"order_id": order_id},
                )

            # Мінімізація даних: видаляємо службові примітки та внутрішні поля
            safe_order = sanitize_order_for_client(full_order)
            return ToolResult(
                status="ok",
                content=safe_order,
                arguments={"order_id": order_id},
            )

        elif name == "search_products":
            query = args.get("query")
            if not query or not str(query).strip():
                return ToolResult(
                    status="rejected",
                    reason="відсутній або порожній пошуковий запит query",
                    content={"error": "missing_parameter", "message": "Вкажіть пошуковий запит."},
                    arguments=args,
                )
            category = args.get("category")
            max_price = args.get("max_price")
            limit = args.get("limit", 10)
            results = service.search_products(
                query=str(query),
                category=str(category) if category else None,
                max_price=max_price,
                limit=int(limit) if limit else 10,
            )
            return ToolResult(
                status="ok",
                content=results,
                arguments={"query": query, "category": category, "max_price": max_price, "limit": limit},
            )

        elif name == "get_product":
            sku = args.get("sku")
            if not sku or not str(sku).strip():
                return ToolResult(
                    status="rejected",
                    reason="відсутній параметр sku",
                    content={"error": "missing_parameter", "message": "Вкажіть артикул товару (sku)."},
                    arguments=args,
                )
            product = service.get_product(str(sku).strip())
            return ToolResult(
                status="ok",
                content=product,
                arguments={"sku": sku},
            )

        elif name == "get_stock":
            sku = args.get("sku")
            if not sku or not str(sku).strip():
                return ToolResult(
                    status="rejected",
                    reason="відсутній параметр sku",
                    content={"error": "missing_parameter", "message": "Вкажіть артикул товару (sku)."},
                    arguments=args,
                )
            stock = service.get_stock(str(sku).strip())
            return ToolResult(
                status="ok",
                content=stock,
                arguments={"sku": sku},
            )

        elif name == "delivery_quote":
            city = args.get("city")
            method = args.get("method")
            items = args.get("items")

            if not city or not str(city).strip():
                return ToolResult(
                    status="rejected",
                    reason="відсутнє місто доставки city",
                    content={"error": "missing_parameter", "message": "Вкажіть місто доставки."},
                    arguments=args,
                )
            if not method or method not in service.DELIVERY_METHODS:
                return ToolResult(
                    status="rejected",
                    reason=f"некоректний спосіб доставки: {method!r}",
                    content={
                        "error": "invalid_method",
                        "message": f"Спосіб доставки має бути одним із: {', '.join(service.DELIVERY_METHODS)}.",
                    },
                    arguments=args,
                )
            if not items or not isinstance(items, list):
                return ToolResult(
                    status="rejected",
                    reason="не вказано список товарів items",
                    content={"error": "missing_parameter", "message": "Вкажіть товари для розрахунку доставки."},
                    arguments=args,
                )

            quote = service.delivery_quote(city=str(city).strip(), method=method, items=items)
            return ToolResult(
                status="ok",
                content=quote,
                arguments={"city": city, "method": method, "items": items},
            )

        elif name == "create_return":
            raw_order_id = args.get("order_id")
            sku = args.get("sku")
            reason = args.get("reason")
            quantity = args.get("quantity", 1)
            comment = args.get("comment", "")

            if not raw_order_id or not sku or not reason:
                return ToolResult(
                    status="rejected",
                    reason="відсутні обовʼязкові параметри order_id, sku або reason",
                    content={
                        "error": "missing_parameter",
                        "message": "Для повернення потрібні order_id, sku та reason.",
                    },
                    arguments=args,
                )

            order_id = normalize_order_id(raw_order_id)

            # Перевірка власності замовлення перед створенням повернення
            order_record = service._state.get("orders", {}).get(order_id)
            if order_record is not None and str(order_record.get("customer_id")) != str(ctx.customer_id):
                return ToolResult(
                    status="rejected",
                    reason="замовлення не належить поточному клієнтові",
                    content={
                        "error": "access_denied",
                        "message": "Неможливо оформити повернення для замовлення іншого клієнта.",
                    },
                    arguments=args,
                )

            # Запобігання повторному створенню ідентичної заявки (ідемпотентність)
            existing_returns = service.list_returns(order_id)
            for r in existing_returns:
                if r.get("sku") == sku and r.get("reason") == reason:
                    return ToolResult(
                        status="ok",
                        reason="повернення вже було оформлено раніше",
                        content={
                            "already_created": True,
                            "return_id": r.get("return_id"),
                            "order_id": order_id,
                            "sku": sku,
                            "status": r.get("status"),
                            "next_steps": r.get("next_steps"),
                            "message": f"Заявку на повернення товару {sku} вже створено раніше: {r.get('return_id')}.",
                        },
                        arguments=args,
                    )

            created = service.create_return(
                order_id=order_id,
                sku=str(sku).strip(),
                reason=reason,
                quantity=int(quantity),
                comment=str(comment),
            )
            return ToolResult(
                status="ok",
                content=created,
                arguments={"order_id": order_id, "sku": sku, "reason": reason, "quantity": quantity},
            )

    except NotFound as e:
        return ToolResult(
            status="error",
            reason=f"обʼєкт не знайдено: {e}",
            content={"error": e.code, "message": str(e)},
            arguments=args,
        )
    except InvalidRequest as e:
        return ToolResult(
            status="error",
            reason=f"некоректний запит: {e}",
            content={"error": e.code, "message": str(e)},
            arguments=args,
        )
    except PolicyViolation as e:
        return ToolResult(
            status="error",
            reason=f"правила магазину: {e}",
            content={"error": e.code, "message": str(e)},
            arguments=args,
        )
    except Unavailable as e:
        return ToolResult(
            status="error",
            reason="сервіс магазину тимчасово недоступний",
            content={"error": e.code, "message": "Сервіс магазину тимчасово недоступний. Спробуйте пізніше."},
            arguments=args,
        )
    except ShopError as e:
        return ToolResult(
            status="error",
            reason=f"помилка магазину: {e}",
            content={"error": getattr(e, "code", "error"), "message": str(e)},
            arguments=args,
        )
    except Exception as e:
        return ToolResult(
            status="error",
            reason=f"непередбачена помилка виконання: {e}",
            content={"error": "internal_error", "message": "Сталася внутрішня помилка під час виконання інструмента."},
            arguments=args,
        )

    return ToolResult(
        status="rejected",
        reason="невідомий стан виконання",
        content={"error": "unknown_state", "message": "Не вдалося виконати операцію."},
    )
