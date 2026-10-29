"""Конвеєр обробки документа: від байтів файлу до рішення, що з ним робити.

Єдине місце, яке знає всі кроки послідовно: підготувати зображення
(`images.prepare`), вилучити поля (`llm.extract`), перевірити їх
правилами (`rules.check`) і вирішити долю документа.
"""

import time
from dataclasses import dataclass, field

from .images import ImageError, prepare
from . import llm
from .llm import LLMError
from .rules import Issue, check

# -------------------------------------------------------------------------
# Набір правил, що блокують AUTO
# -------------------------------------------------------------------------
# Якщо хоча б одна з цих `rule`-назв є серед помилок (severity=error),
# документ ніколи не отримає AUTO.
BLOCKING_RULES = {
    "document_type",        # не рахунок
    "missing_required",     # відсутні обов'язкові поля
    "iban_checksum",        # хибна контрольна сума IBAN
    "iban_registry",        # IBAN відрізняється від довідника
    "edrpou_checksum",      # хибний код ЄДРПОУ
    "rnokpp_checksum",      # хибний код РНОКПП
    "code_format",          # некоректна довжина коду
    "arithmetic_line",      # арифметика рядка
    "arithmetic_subtotal",  # не сходяться підсумки
    "arithmetic_total",     # не сходиться разом
    "buyer",                # покупець — не наша компанія
    "supplier_registry",    # постачальника немає в довіднику або код не збігається
    "date_future",          # дата в майбутньому
    "date_valid_until",     # строк дії до дати рахунку
}


@dataclass
class Result:
    """Результат конвеєра — те, з чим працює веб-рівень.

    `decision` — `auto`, `review` або `reject`; `reasons` — чому саме так.
    `document` — поля, які повернула модель (або `None`). `issues` —
    проблеми, знайдені правилами. `image` — розміри до й після підготовки.
    `elapsed` — час за етапами. `usage` — токени.
    """

    decision: str
    reasons: list[str] = field(default_factory=list)
    document: dict | None = None
    issues: list[Issue] = field(default_factory=list)
    image: dict = field(default_factory=dict)
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


def decide(document: dict, issues: list[Issue]) -> tuple[str, list[str]]:
    """Вирішити долю документа за вилученими полями й знайденими проблемами.

    Повертає (decision, reasons).
    """
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]

    # 1. Перевіряємо тип документа — reject якщо явно не рахунок
    doc_type = (document.get("document_type") or "").lower()
    if doc_type and ("накладна" in doc_type or "видаткова" in doc_type):
        return "reject", [
            f"Документ є «{document.get('document_type')}», а не рахунком на оплату."
        ]

    reasons = []

    # Збираємо всі blocking-помилки
    blocking_errors = [e for e in errors if e.rule in BLOCKING_RULES]
    non_blocking_errors = [e for e in errors if e.rule not in BLOCKING_RULES]

    if blocking_errors or non_blocking_errors:
        for issue in errors:
            reasons.append(issue.message)
        for issue in warnings:
            reasons.append(f"Увага: {issue.message}")
        return "review", reasons

    # Якщо є тільки попередження — review
    if warnings:
        for issue in warnings:
            reasons.append(f"Увага: {issue.message}")
        return "review", reasons

    # AUTO — всі перевірки пройдено
    return "auto", ["Усі перевірки пройдено. Документ може бути автоматично зареєстрований."]


def process(content: bytes, max_side: int | None = None) -> Result:
    """Обробити файл: підготувати → вилучити → перевірити → вирішити."""
    elapsed = {}

    # -----------------------------------------------------------------------
    # Крок 1. Підготовка зображення
    # -----------------------------------------------------------------------
    try:
        prepared = prepare(content, max_side=max_side)
        elapsed["prepare"] = prepared.prepare_time
        image_info = {
            "original": prepared.original,
            "sent": prepared.sent,
        }
    except ImageError as e:
        return Result(
            decision="reject",
            reasons=[str(e)],
            image={},
            elapsed=elapsed,
        )
    except Exception as e:
        return Result(
            decision="reject",
            reasons=["Не вдалося обробити зображення. Спробуйте завантажити інший файл."],
            image={},
            elapsed=elapsed,
        )

    # -----------------------------------------------------------------------
    # Крок 2. Вилучення даних моделлю
    # -----------------------------------------------------------------------
    try:
        t0 = time.monotonic()
        llm_result = llm.extract(prepared)
        elapsed["extraction"] = llm_result.get("elapsed_extraction", time.monotonic() - t0)
    except LLMError as e:
        # Технічна помилка ≠ поганий документ
        return Result(
            decision="review",
            reasons=[
                f"Не вдалося отримати відповідь від моделі: {e}. "
                "Перевірте документ вручну."
            ],
            image=image_info,
            elapsed=elapsed,
        )
    except Exception as e:
        return Result(
            decision="review",
            reasons=["Внутрішня помилка вилучення даних. Перевірте документ вручну."],
            image=image_info,
            elapsed=elapsed,
        )

    document = llm_result.get("document", {})
    model_name = llm_result.get("model")
    usage = llm_result.get("usage")
    finish_reason = llm_result.get("finish_reason", "")

    # Попередження про обрізану відповідь
    extra_issues = []
    if finish_reason and finish_reason not in ("stop", "end_turn", None):
        extra_issues.append(Issue(
            field="",
            rule="truncated_response",
            message=(
                f"Відповідь моделі була обрізана (finish_reason={finish_reason}). "
                "Дані можуть бути неповними."
            ),
            severity="warning"
        ))

    # -----------------------------------------------------------------------
    # Крок 3. Перевірка правилами
    # -----------------------------------------------------------------------
    t0 = time.monotonic()
    try:
        issues = check(document)
    except Exception:
        issues = []
        extra_issues.append(Issue(
            field="",
            rule="check_error",
            message="Внутрішня помилка перевірки правил. Перевірте документ вручну.",
            severity="error"
        ))
    elapsed["checks"] = time.monotonic() - t0

    issues = extra_issues + issues

    # -----------------------------------------------------------------------
    # Крок 4. Рішення
    # -----------------------------------------------------------------------
    decision, reasons = decide(document, issues)

    return Result(
        decision=decision,
        reasons=reasons,
        document=document,
        issues=issues,
        image=image_info,
        model=model_name,
        elapsed=elapsed,
        usage=usage,
    )
