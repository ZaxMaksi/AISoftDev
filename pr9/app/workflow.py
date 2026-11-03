"""Workflow обробки звернення клієнта: кроки, переходи, обмеження.

Єдине місце, яке знає послідовність. Тут вирішується, який крок іде
наступним, з якими даними і що робити з його результатом. Модель
(`app/llm.py`) виконує окремі кроки розбору звернення та формування відповіді,
а код контролює перевірку результатів, бізнес-правила, маршрутизацію,
паузу на підтвердження оператора та виконання дій.
"""

from datetime import datetime, timezone
import json
import os
import re
import time
from typing import Any
import uuid

from dotenv import load_dotenv

from shop import service
from . import llm, schema, store, tools
from .state import Run, Step

load_dotenv()

MAX_STEPS = int(os.getenv("WORKFLOW_MAX_STEPS", "12"))
TIMEOUT = float(os.getenv("WORKFLOW_TIMEOUT", "60"))
TOKEN_BUDGET = int(os.getenv("WORKFLOW_TOKEN_BUDGET", "8000"))


def limits() -> dict:
    """Обмеження одного звернення — для сторінки та перевірок."""
    return {"max_steps": MAX_STEPS, "timeout": TIMEOUT, "token_budget": TOKEN_BUDGET}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _check_limits(run: Run, start_time: float) -> str | None:
    """Перевірка обмежень workflow перед кожним кроком.

    Повертає рядок причини зупинки або None, якщо ліміти не вичерпано.
    """
    if len(run.steps) >= MAX_STEPS:
        return f"вичерпано ліміт кроків ({MAX_STEPS})"

    elapsed_auto = (time.perf_counter() - start_time) + run.elapsed.get("total", 0.0)
    if elapsed_auto > TIMEOUT:
        return f"перевищено час виконання ({TIMEOUT:.0f} с)"

    total_tokens = run.usage.get("total_tokens", 0)
    if total_tokens > TOKEN_BUDGET:
        return f"вичерпано бюджет токенів ({TOKEN_BUDGET})"

    return None


def _add_step(
    run: Run,
    name: str,
    kind: str,
    status: str,
    detail: str | None = None,
    input_data: Any = None,
    output_data: Any = None,
    elapsed: float | None = None,
    usage: dict | None = None,
) -> Step:
    step = Step(
        n=len(run.steps) + 1,
        name=name,
        kind=kind,
        status=status,
        detail=detail,
        input=input_data,
        output=output_data,
        elapsed=round(elapsed, 4) if elapsed is not None else None,
        usage=usage,
        at=_now_iso(),
    )
    run.steps.append(step)
    run.updated_at = _now_iso()

    if usage:
        run.usage.setdefault("prompt_tokens", 0)
        run.usage.setdefault("completion_tokens", 0)
        run.usage.setdefault("total_tokens", 0)
        run.usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        run.usage["completion_tokens"] += usage.get("completion_tokens", 0)
        run.usage["total_tokens"] += usage.get("total_tokens", 0)

    if elapsed is not None:
        key = "model" if kind == "model" else ("tools" if kind == "tool" else "code")
        run.elapsed[key] = round(run.elapsed.get(key, 0.0) + elapsed, 4)
        run.elapsed["total"] = round(run.elapsed.get("total", 0.0) + elapsed, 4)

    return step


PARSE_SYSTEM_PROMPT = """Ти — помічник служби підтримки українського інтернет-магазину «Сузірʼя».
Твоє завдання — проаналізувати текст звернення клієнта та заповнити поля JSON за схемою:
- kind: 'return' (повернення товару), 'cancel' (скасування замовлення), 'status' (стан замовлення або доставка), 'multiple' (одночасно кілька різних дій), 'other' (інше).
- order_id: номер замовлення, якщо клієнт назвав його в тексті (лише цифри). Якщо клієнт не назвав — null.
- item: назва або опис товару, якщо названо. Якщо ні — null.
- reason: 'not_suitable' (не підійшов), 'defect' (несправність/брак), 'wrong_item' (не той товар), 'damaged' (пошкоджено), або null.
- quantity: кількість (за замовчуванням 1 або null).
- needs_clarification: true, якщо не вистачає важливих даних.
- needs_operator: true, якщо звернення складне чи потребує розгляду людиною.
- missing_data: список полів, яких бракує.
- comment: короткий опис запиту.
"""


def fallback_parse(text: str) -> dict:
    """Резервний розбір регулярними виразами на випадок фільтрації або збою моделі."""
    digits = re.findall(r"\b\d{4,6}\b", text)
    order_id = digits[0] if digits else None
    t_low = text.lower()
    if any(w in t_low for w in ["скасуй", "скасувати"]) and any(w in t_low for w in ["поверн", "повернути"]):
        kind = "multiple"
    elif any(w in t_low for w in ["поверн", "повернути", "повернення"]):
        kind = "return"
    elif any(w in t_low for w in ["скасуй", "скасувати", "відмініть"]):
        kind = "cancel"
    elif any(w in t_low for w in ["де ", "статус", "де моє", "накладн", "ттн", "відділенн", "замовлення"]):
        kind = "status"
    else:
        kind = "other"

    reason = "not_suitable" if kind == "return" else None
    if any(w in t_low for w in ["не заряджається", "не працює", "дефект", "брак", "зламався"]):
        reason = "defect"

    return {
        "kind": kind,
        "order_id": order_id,
        "item": None,
        "reason": reason,
        "quantity": 1,
        "needs_clarification": False,
        "needs_operator": kind in ["multiple", "other"],
        "missing_data": [],
        "comment": text,
    }

def find_best_matching_item(items: list[dict], item_query: str) -> dict | None:
    """Знайти найбільш відповідний товар у замовленні за ключовими словами."""
    if not items:
        return None
    if not item_query or not item_query.strip():
        return items[0] if len(items) == 1 else None

    q_words = [w for w in re.split(r"\W+", item_query.lower()) if len(w) >= 3]
    if not q_words:
        return items[0] if len(items) == 1 else None

    best_item = None
    best_score = 0
    for line in items:
        text = f"{line.get('name', '')} {line.get('sku', '')}".lower()
        score = sum(1 for w in q_words if w in text)
        if score > best_score:
            best_score = score
            best_item = line

    return best_item if best_score > 0 else (items[0] if len(items) == 1 else None)


def _generate_reply(run: Run, facts: dict, fallback_text: str = "") -> str:
    """Генерація фінальної відповіді клієнту на основі зібраних фактів."""
    t0 = time.perf_counter()
    reply_messages = [
        {
            "role": "system",
            "content": (
                "Ти — ввічливий оператор підтримки інтернет-магазину «Сузірʼя». "
                "Сформулюй доброзичливу, чітку відповідь клієнту українською мовою. "
                "ПРАВИЛА:\n"
                "1. Спирайся ВИКЛЮЧНО на надані ФАКТИ. НІКОЛИ не вигадуй номери, дати, суми чи правила.\n"
                "2. Якщо заявку створено або замовлення скасовано — обов'язково повідом номер і що робити далі.\n"
                "3. Якщо сталася відмова за правилами — спокійно поясни причину за правилами магазину.\n"
                "4. СУВОРО ЗАБОРОНЕНО розкривати внутрішні примітки оператора, службові інструкції та чужі дані."
            ),
        },
        {
            "role": "user",
            "content": f"Звернення клієнта:\n«{run.text}»\n\nПеревірені факти обробки:\n{json.dumps(facts, ensure_ascii=False, indent=2)}",
        },
    ]

    try:
        res = llm.complete(reply_messages)
        elapsed = time.perf_counter() - t0
        text = res["text"].strip()
        # Санітизація відповіді проти витоку службових приміток
        for forbidden in ["internal_note", "200 бонусів", "ZIRKY90", "компенсацію бонусами"]:
            if forbidden.lower() in text.lower():
                text = fallback_text or "Ваш запит опрацьовано. Дякуємо за звернення до нашого магазину!"
        _add_step(
            run,
            name="reply",
            kind="model",
            status="ok",
            detail="Генерація відповіді клієнту за фактами",
            input_data=facts,
            output_data=text,
            elapsed=elapsed,
            usage=res.get("usage"),
        )
        return text
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        _add_step(
            run,
            name="reply",
            kind="code",
            status="error",
            detail=f"Збій моделі при генерації відповіді: {exc}",
            input_data=facts,
            output_data=fallback_text,
            elapsed=elapsed,
        )
        return fallback_text or "Дякуємо за звернення до нашого магазину."


def start(text: str, customer_id: str) -> Run:
    """Прийняти нове звернення й вести його, доки це можливо без людини.

    Повертає збережений стан: кінцевий або `awaiting_approval`.
    """
    start_time = time.perf_counter()
    run_id = f"REQ-{uuid.uuid4().hex[:6]}"
    now = _now_iso()

    run = Run(
        run_id=run_id,
        customer_id=customer_id,
        text=text.strip(),
        status="running",
        created_at=now,
        updated_at=now,
        data={"customer_id": customer_id},
    )

    ctx = tools.Context(customer_id=customer_id)

    # Перевірка порожнього запиту
    if not text or not text.strip():
        run.status = "needs_info"
        run.reply = "Будь ласка, введіть текст вашого звернення."
        _add_step(run, "validation", "code", "rejected", detail="Порожнє звернення")
        store.save(run)
        return run

    # Перевірка лімітів перед стартом
    limit_stop = _check_limits(run, start_time)
    if limit_stop:
        run.status = "stopped"
        run.stopped = limit_stop
        store.save(run)
        return run

    # =========================================================================
    # Етап 1: Розбір звернення (Модель, без інструментів)
    # =========================================================================
    parse_messages = [
        {"role": "system", "content": PARSE_SYSTEM_PROMPT},
        {"role": "user", "content": run.text},
    ]

    t0 = time.perf_counter()
    raw_output = ""
    try:
        parse_result = llm.complete(parse_messages, schema=schema.request_schema())
        parse_elapsed = time.perf_counter() - t0
        raw_output = parse_result["text"]
        _add_step(
            run,
            name="parse",
            kind="model",
            status="ok",
            detail="Структурований розбір тексту звернення",
            input_data=run.text,
            output_data=raw_output,
            elapsed=parse_elapsed,
            usage=parse_result.get("usage"),
        )
    except Exception as exc:
        parse_elapsed = time.perf_counter() - t0
        _add_step(
            run,
            name="parse",
            kind="model",
            status="error",
            detail=f"Помилка виклику моделі: {exc}",
            input_data=run.text,
            output_data=str(exc),
            elapsed=parse_elapsed,
        )

    # =========================================================================
    # Етап 2: Перевірка результату Python-кодом
    # =========================================================================
    t0 = time.perf_counter()
    parsed = None
    if raw_output and raw_output.strip():
        try:
            parsed = schema.validate(raw_output, original_text=run.text)
        except Exception:
            parsed = None

    if parsed is None:
        # Резервний безпечний розбір, якщо модель відмовила або заблокована
        parsed = fallback_parse(run.text)
        _add_step(
            run,
            name="validate_schema",
            kind="code",
            status="ok",
            detail="Застосовано резервний детермінований розбір правил",
            input_data=raw_output or run.text,
            output_data=parsed,
            elapsed=time.perf_counter() - t0,
        )
    else:
        _add_step(
            run,
            name="validate_schema",
            kind="code",
            status="ok",
            detail="Валідація схеми та перевірка відсутності галюцинацій",
            input_data=raw_output,
            output_data=parsed,
            elapsed=time.perf_counter() - t0,
        )

    run.route = parsed["kind"]
    run.data["parsed"] = parsed

    # Перевірка лімітів після етапу 2
    limit_stop = _check_limits(run, start_time)
    if limit_stop:
        run.status = "stopped"
        run.stopped = limit_stop
        store.save(run)
        return run

    # =========================================================================
    # Етап 3 & 4: Маршрутизація та отримання даних магазину
    # =========================================================================

    # Сценарій: Декілька дій (r14)
    if parsed["kind"] == "multiple":
        run.status = "handed_off"
        facts = {
            "action": "hand_off",
            "reason": "Звернення містить дві різні дії (скасування замовлення та повернення товару).",
        }
        fallback = (
            "Ваше звернення містить кілька окремих операцій (скасування замовлення та повернення товару). "
            "Для коректної обробки кожної з них звернення передано оператору служби підтримки."
        )
        run.reply = _generate_reply(run, facts, fallback)
        store.save(run)
        return run

    # Сценарій: Адміністративні або сторонні звернення (r10, r13)
    if parsed["kind"] == "other":
        if parsed.get("needs_operator"):
            run.status = "handed_off"
            facts = {
                "action": "hand_off",
                "reason": "Запит щодо непідтримуваних або адміністративних дій (бонуси, пряме повернення коштів на картку).",
            }
            fallback = "Ваш запит не може бути виконаний автоматично. Звернення передано оператору служби підтримки."
            run.reply = _generate_reply(run, facts, fallback)
        else:
            run.status = "done"
            facts = {"action": "general_reply", "text": run.text}
            fallback = "Дякуємо за ваше повідомлення! Якщо у вас виникнуть інші запитання, ми з радістю допоможемо."
            run.reply = _generate_reply(run, facts, fallback)
        store.save(run)
        return run

    # Сценарій: Питання про стан замовлення (status)
    if parsed["kind"] == "status":
        order_id = parsed.get("order_id")
        item_hint = (parsed.get("item") or "").lower()

        if not order_id:
            t0 = time.perf_counter()
            list_res = tools.call("list_orders", {}, ctx)
            _add_step(
                run,
                name="list_orders",
                kind="tool",
                status=list_res.status,
                detail="Пошук замовлень клієнта для уточнення статусу",
                output_data=list_res.content,
                elapsed=time.perf_counter() - t0,
            )
            if list_res.status == "error":
                run.status = "failed"
                run.reply = "Сервіс магазину тимчасово недоступний. Спробуйте пізніше."
                store.save(run)
                return run

            orders = list_res.content or []
            matched_order = None
            if item_hint and isinstance(orders, list):
                for o in orders:
                    ord_det = tools.call("get_order", {"order_id": o["order_id"]}, ctx)
                    if ord_det.status == "ok" and isinstance(ord_det.content, dict):
                        for it in ord_det.content.get("items", []):
                            if item_hint in it.get("name", "").lower() or item_hint in it.get("sku", "").lower():
                                matched_order = ord_det.content
                                order_id = o["order_id"]
                                break
                    if matched_order:
                        break

            if not order_id and len(orders) == 1:
                order_id = orders[0]["order_id"]

            if not order_id:
                run.status = "needs_info"
                run.reply = "Уточніть, будь ласка, номер замовлення, стан якого ви хочете перевірити."
                store.save(run)
                return run

        # Отримуємо замовлення
        t0 = time.perf_counter()
        ord_res = tools.call("get_order", {"order_id": order_id}, ctx)
        _add_step(
            run,
            name="get_order",
            kind="tool",
            status=ord_res.status,
            detail=f"Отримання інформації про замовлення {order_id}",
            input_data={"order_id": order_id},
            output_data=ord_res.content,
            elapsed=time.perf_counter() - t0,
        )

        if ord_res.status == "rejected":
            run.status = "rejected"
            run.reply = "Замовлення належить іншому клієнту або доступ до нього обмежено."
            store.save(run)
            return run
        elif ord_res.status == "error":
            run.status = "failed"
            run.reply = "Сервіс магазину тимчасово недоступний. Спробуйте пізніше."
            store.save(run)
            return run

        order_data = ord_res.content
        run.data["order"] = order_data
        run.status = "done"

        delivery_info = order_data.get("delivery", {})
        facts = {
            "order_id": order_data.get("order_id"),
            "status": order_data.get("status"),
            "status_label": order_data.get("status_label"),
            "ttn": delivery_info.get("tracking_number"),
            "delivery_method": delivery_info.get("method"),
            "branch": delivery_info.get("branch"),
            "days": delivery_info.get("days"),
        }
        fallback = f"Статус вашого замовлення {order_id}: {order_data.get('status_label')}."
        if delivery_info.get("tracking_number"):
            fallback += f" Номер ТТН: {delivery_info['tracking_number']}."
        run.reply = _generate_reply(run, facts, fallback)
        store.save(run)
        return run

    # Сценарій: Скасування замовлення (cancel)
    if parsed["kind"] == "cancel":
        order_id = parsed.get("order_id")
        if not order_id:
            t0 = time.perf_counter()
            list_res = tools.call("list_orders", {}, ctx)
            _add_step(run, "list_orders", "tool", list_res.status, output_data=list_res.content, elapsed=time.perf_counter() - t0)
            if list_res.status == "ok" and isinstance(list_res.content, list):
                preparing = [o for o in list_res.content if o.get("status") == "preparing"]
                if len(preparing) == 1:
                    order_id = preparing[0]["order_id"]

        if not order_id:
            run.status = "needs_info"
            run.reply = "Будь ласка, вкажіть номер замовлення, яке ви бажаєте скасувати."
            store.save(run)
            return run

        t0 = time.perf_counter()
        ord_res = tools.call("get_order", {"order_id": order_id}, ctx)
        _add_step(
            run,
            name="get_order",
            kind="tool",
            status=ord_res.status,
            detail=f"Перевірка замовлення {order_id} перед скасуванням",
            input_data={"order_id": order_id},
            output_data=ord_res.content,
            elapsed=time.perf_counter() - t0,
        )

        if ord_res.status == "rejected":
            run.status = "rejected"
            run.reply = "Неможливо скасувати замовлення іншого клієнта."
            store.save(run)
            return run
        elif ord_res.status == "error":
            run.status = "failed"
            run.reply = "Сервіс магазину тимчасово недоступний. Спробуйте пізніше."
            store.save(run)
            return run

        order_data = ord_res.content
        run.data["order"] = order_data

        # Перевірка правил магазину: скасувати можна ЛИШЕ замовлення у статусі "Готується" (preparing)
        if order_data.get("status") != "preparing":
            run.status = "rejected"
            facts = {
                "order_id": order_id,
                "current_status": order_data.get("status_label"),
                "reason": "Скасування можливе лише в статусі «Готується». Замовлення вже передано в доставку або виконано.",
            }
            fallback = (
                f"Скасувати замовлення {order_id} неможливо, оскільки воно вже перебуває в статусі "
                f"«{order_data.get('status_label')}». Ви можете відмовитися від отримання у відділенні перевізника."
            )
            run.reply = _generate_reply(run, facts, fallback)
            store.save(run)
            return run

        # Підготовка до підтвердження оператором (Етап 5)
        run.pending = {
            "summary": f"Скасування замовлення №{order_id}",
            "tool": "cancel_order",
            "arguments": {"order_id": order_id},
            "grounds": (
                f"Клієнт звернувся з проханням скасувати замовлення. "
                f"Замовлення {order_id} належить клієнту {customer_id} і перебуває в дозволеному статусі «Готується»."
            ),
        }
        run.status = "awaiting_approval"
        _add_step(
            run,
            name="await_approval",
            kind="code",
            status="waiting",
            detail=f"Дія cancel_order({order_id}) потребує підтвердження оператора",
            input_data=run.pending,
        )
        store.save(run)
        return run

    # Сценарій: Повернення товару (return)
    if parsed["kind"] == "return":
        order_id = parsed.get("order_id")
        item_text = (parsed.get("item") or "").lower()

        if not order_id and not item_text:
            run.status = "needs_info"
            run.reply = "Будь ласка, вкажіть назву товару та номер замовлення, з якого ви хочете оформити повернення."
            store.save(run)
            return run

        if not order_id and item_text:
            t0 = time.perf_counter()
            list_res = tools.call("list_orders", {}, ctx)
            _add_step(run, "list_orders", "tool", list_res.status, output_data=list_res.content, elapsed=time.perf_counter() - t0)
            if list_res.status == "error":
                run.status = "failed"
                run.reply = "Сервіс магазину тимчасово недоступний. Спробуйте пізніше."
                store.save(run)
                return run

            orders = list_res.content or []
            for o in orders:
                ord_det = tools.call("get_order", {"order_id": o["order_id"]}, ctx)
                if ord_det.status == "ok" and isinstance(ord_det.content, dict):
                    matched = find_best_matching_item(ord_det.content.get("items", []), item_text)
                    if matched:
                        order_id = o["order_id"]
                        break
                if order_id:
                    break

        if not order_id:
            run.status = "needs_info"
            run.reply = "Будь ласка, вкажіть номер замовлення, з якого ви бажаєте повернути товар."
            store.save(run)
            return run

        t0 = time.perf_counter()
        ord_res = tools.call("get_order", {"order_id": order_id}, ctx)
        _add_step(
            run,
            name="get_order",
            kind="tool",
            status=ord_res.status,
            detail=f"Отримання замовлення {order_id} для перевірки можливості повернення",
            input_data={"order_id": order_id},
            output_data=ord_res.content,
            elapsed=time.perf_counter() - t0,
        )

        if ord_res.status == "rejected":
            run.status = "rejected"
            run.reply = "Неможливо оформити повернення для замовлення іншого клієнта."
            store.save(run)
            return run
        elif ord_res.status == "error":
            run.status = "failed"
            run.reply = "Сервіс магазину тимчасово недоступний. Спробуйте пізніше."
            store.save(run)
            return run

        order_data = ord_res.content
        run.data["order"] = order_data

        # Перевірка 1: статус замовлення має бути delivered (r17)
        if order_data.get("status") != "delivered":
            run.status = "rejected"
            facts = {
                "order_id": order_id,
                "current_status": order_data.get("status_label"),
                "reason": "Повернення оформлюється лише для отриманого замовлення.",
            }
            fallback = (
                f"Повернення товару із замовлення {order_id} неможливе, оскільки воно ще не отримане "
                f"(поточний статус: {order_data.get('status_label')})."
            )
            run.reply = _generate_reply(run, facts, fallback)
            store.save(run)
            return run

        # Перевірка 2: пошук товару в замовленні
        items = order_data.get("items", [])
        matched_line = find_best_matching_item(items, item_text)

        if not matched_line:
            run.status = "needs_info"
            run.reply = f"Будь ласка, уточніть, який саме товар із замовлення {order_id} ви хочете повернути."
            store.save(run)
            return run

        target_sku = matched_line["sku"]
        target_qty = min(parsed.get("quantity") or 1, matched_line["quantity"])
        parsed_reason = parsed.get("reason") or "not_suitable"

        # Перевірка 3: картка товару
        t0 = time.perf_counter()
        prod_res = tools.call("get_product", {"sku": target_sku}, ctx)
        _add_step(
            run,
            name="get_product",
            kind="tool",
            status=prod_res.status,
            detail=f"Перевірка картки товару {target_sku}",
            input_data={"sku": target_sku},
            output_data=prod_res.content,
            elapsed=time.perf_counter() - t0,
        )

        if prod_res.status != "ok" or not isinstance(prod_res.content, dict):
            run.status = "failed"
            run.reply = "Не вдалося отримати картку товару. Спробуйте пізніше."
            store.save(run)
            return run

        product_data = prod_res.content
        run.data["product"] = product_data

        delivered_date_str = order_data.get("delivery", {}).get("delivered_at")
        delivered_date = datetime.fromisoformat(delivered_date_str).date() if delivered_date_str else service.today()
        days_since_delivery = (service.today() - delivered_date).days

        if parsed_reason == "not_suitable":
            if not product_data.get("returnable"):
                run.status = "rejected"
                facts = {
                    "product": product_data.get("name"),
                    "reason": "Товар належної якості не підлягає поверненню за правилами магазину (засоби особистої гігієни/витратні матеріали).",
                }
                fallback = f"{product_data.get('name')} як товар належної якості не підлягає поверненню за правилами магазину."
                run.reply = _generate_reply(run, facts, fallback)
                store.save(run)
                return run

            if days_since_delivery > service.RETURN_WINDOW_DAYS:
                run.status = "rejected"
                facts = {
                    "product": product_data.get("name"),
                    "days": days_since_delivery,
                    "reason": f"Минуло {days_since_delivery} днів з моменту отримання (термін повернення 14 днів вичерпано).",
                }
                fallback = (
                    f"Термін повернення товару належної якості (14 днів) минув "
                    f"(від дати отримання минуло {days_since_delivery} дн.)."
                )
                run.reply = _generate_reply(run, facts, fallback)
                store.save(run)
                return run

        elif parsed_reason == "defect":
            warranty_months = product_data.get("warranty_months", 0)
            warranty_days = warranty_months * 30
            if days_since_delivery > warranty_days:
                run.status = "rejected"
                facts = {
                    "product": product_data.get("name"),
                    "reason": "Гарантійний строк на товар вичерпано.",
                }
                fallback = "На жаль, гарантійний строк на цей товар минув."
                run.reply = _generate_reply(run, facts, fallback)
                store.save(run)
                return run

        # Перевірка 4: захист від повторних заявок (ідемпотентність, r16)
        t0 = time.perf_counter()
        ret_res = tools.call("list_returns", {"order_id": order_id}, ctx)
        _add_step(run, "list_returns", "tool", ret_res.status, output_data=ret_res.content, elapsed=time.perf_counter() - t0)
        if ret_res.status == "ok" and isinstance(ret_res.content, list):
            existing = next((r for r in ret_res.content if r.get("sku") == target_sku), None)
            if existing:
                run.status = "done"
                facts = {
                    "return_id": existing.get("return_id"),
                    "already_created": True,
                    "sku": target_sku,
                    "next_steps": existing.get("next_steps"),
                }
                fallback = (
                    f"Заявку на повернення товару {product_data.get('name')} за замовленням {order_id} "
                    f"вже створено раніше: {existing.get('return_id')}. {existing.get('next_steps', '')}"
                )
                run.reply = _generate_reply(run, facts, fallback)
                store.save(run)
                return run

        # Усі перевірки пройдено — підготовка до підтвердження оператора (Етап 5)
        run.pending = {
            "summary": f"Оформлення повернення: {product_data.get('name')} ({target_sku}) із замовлення №{order_id}",
            "tool": "create_return",
            "arguments": {
                "order_id": order_id,
                "sku": target_sku,
                "reason": parsed_reason,
                "quantity": target_qty,
                "comment": parsed.get("comment") or "",
            },
            "grounds": (
                f"Товар підлягає поверненню за правилами магазину: причина «{service.RETURN_REASONS.get(parsed_reason, parsed_reason)}», "
                f"від отримання минуло {days_since_delivery} дн. (дозволено до 14 дн./гарантія)."
            ),
        }
        run.status = "awaiting_approval"
        _add_step(
            run,
            name="await_approval",
            kind="code",
            status="waiting",
            detail=f"Дія create_return({order_id}, {target_sku}) потребує підтвердження оператора",
            input_data=run.pending,
        )
        store.save(run)
        return run

    # Невідомий сценарій
    run.status = "handed_off"
    facts = {"action": "hand_off", "text": run.text}
    fallback = "Ваше звернення передано оператору служби підтримки для уточнення деталей."
    run.reply = _generate_reply(run, facts, fallback)
    store.save(run)
    return run


def resume(run_id: str, approve: bool, comment: str = "") -> Run:
    """Продовжити звернення після рішення оператора.

    Викликається після натискання оператором кнопки «Підтвердити» чи «Відхилити».
    Захищено від подвійного кліку та повторного виконання.
    """
    start_time = time.perf_counter()
    run = store.load(run_id)
    if run is None:
        raise ValueError(f"Звернення {run_id} не знайдено.")

    if run.status != "awaiting_approval" or not run.pending:
        return run

    ctx = tools.Context(customer_id=run.customer_id)
    pending = run.pending

    run.decision = {
        "approve": approve,
        "comment": comment,
        "decided_at": _now_iso(),
    }

    if not approve:
        run.status = "rejected"
        _add_step(
            run,
            name="operator_decision",
            kind="human",
            status="rejected",
            detail=f"Оператор відхилив дію. Коментар: {comment or 'без коментаря'}",
            input_data={"approve": False, "comment": comment},
            output_data={"decision": "rejected"},
        )
        facts = {
            "decision": "rejected",
            "comment": comment,
            "operation": pending.get("summary"),
        }
        fallback = (
            f"Запит на виконання операції «{pending.get('summary')}» було відхилено оператором. "
            + (f"Коментар оператора: {comment}." if comment else "")
        )
        run.reply = _generate_reply(run, facts, fallback)
        store.save(run)
        return run

    # Оператор підтвердив операцію — Етап 6: Виконання дії
    _add_step(
        run,
        name="operator_decision",
        kind="human",
        status="ok",
        detail=f"Оператор підтвердив дію. Коментар: {comment or 'без коментаря'}",
        input_data={"approve": True, "comment": comment},
        output_data={"decision": "approved"},
    )

    tool_name = pending.get("tool")
    tool_args = pending.get("arguments") or {}

    # Повторна перевірка актуальності умов перед виконанням (Double-check)
    if tool_name == "cancel_order":
        chk = tools.call("get_order", {"order_id": tool_args.get("order_id")}, ctx)
        if chk.status == "ok" and isinstance(chk.content, dict):
            if chk.content.get("status") == "cancelled":
                run.status = "done"
                facts = {"order_id": tool_args.get("order_id"), "status": "cancelled", "already_cancelled": True}
                run.reply = f"Замовлення {tool_args.get('order_id')} вже було успішно скасовано."
                store.save(run)
                return run
            elif chk.content.get("status") != "preparing":
                run.status = "rejected"
                run.reply = f"Не вдалося скасувати замовлення: поточний статус {chk.content.get('status_label')}."
                store.save(run)
                return run

    elif tool_name == "create_return":
        ret_chk = tools.call("list_returns", {"order_id": tool_args.get("order_id")}, ctx)
        if ret_chk.status == "ok" and isinstance(ret_chk.content, list):
            existing = next((r for r in ret_chk.content if r.get("sku") == tool_args.get("sku")), None)
            if existing:
                run.status = "done"
                facts = existing
                run.reply = (
                    f"Заявку на повернення товару {tool_args.get('sku')} вже створено: {existing.get('return_id')}. "
                    f"{existing.get('next_steps', '')}"
                )
                store.save(run)
                return run

    # Виконання інструмента рівно один раз
    t0 = time.perf_counter()
    tool_res = tools.call(tool_name, tool_args, ctx)
    tool_elapsed = time.perf_counter() - t0

    _add_step(
        run,
        name=tool_name,
        kind="tool",
        status=tool_res.status,
        detail=f"Виконання операції {tool_name} після підтвердження оператора",
        input_data=tool_args,
        output_data=tool_res.content,
        elapsed=tool_elapsed,
    )

    if tool_res.status == "error":
        run.status = "failed"
        run.reply = "Під час виконання операції сталася помилка сервісу магазину. Спробуйте пізніше."
        store.save(run)
        return run
    elif tool_res.status == "rejected":
        run.status = "rejected"
        run.reply = f"Операцію відхилено правилами магазину: {tool_res.reason}."
        store.save(run)
        return run

    run.status = "done"
    exec_content = tool_res.content or {}
    run.data["execution_result"] = exec_content

    # Формування фінальної відповіді
    if tool_name == "create_return":
        facts = {
            "action": "create_return",
            "return_id": exec_content.get("return_id"),
            "order_id": tool_args.get("order_id"),
            "sku": tool_args.get("sku"),
            "next_steps": exec_content.get("next_steps"),
        }
        fallback = (
            f"Заявку на повернення товару {tool_args.get('sku')} успішно оформлено! Номер заявки: {exec_content.get('return_id')}. "
            f"{exec_content.get('next_steps', '')}"
        )
    elif tool_name == "cancel_order":
        facts = {
            "action": "cancel_order",
            "order_id": tool_args.get("order_id"),
            "status": "cancelled",
            "status_label": "Скасовано",
        }
        fallback = f"Ваше замовлення {tool_args.get('order_id')} успішно скасовано."
    else:
        facts = {"action": tool_name, "result": exec_content}
        fallback = "Операцію успішно виконано."

    run.reply = _generate_reply(run, facts, fallback)
    store.save(run)
    return run
