"""Вартість виклику: токени → гроші.

Провайдер тарифікує токени окремо на вході й на виході; кешовані ним
вхідні токени — дешевше. Ціни — з `.env`, за мільйон токенів; значення
зі зразка орієнтовні, чинні — на сторінці цін провайдера. Безкоштовний
рівень не скасовує розрахунку: рахуємо так, ніби платимо, інакше нема
чого обмежувати й порівнювати.

Модуль знає лише ціни та `usage`; про провайдерів, чернетки й журнал
він не знає. Шлюз викликає `estimate` після кожного звертання і кладе
результат у `Result.cost` та в журнал; ліміт (`budget.py`) додає його
до витрат дня.
"""

import os

from dotenv import load_dotenv

load_dotenv()

# Ціни за мільйон токенів, у доларах.
PRICE_INPUT = float(os.getenv("LLM_PRICE_INPUT_PER_1M", "0.30"))
PRICE_OUTPUT = float(os.getenv("LLM_PRICE_OUTPUT_PER_1M", "2.50"))
_cached = os.getenv("LLM_PRICE_CACHED_INPUT_PER_1M", "")
PRICE_CACHED_INPUT = float(_cached) if _cached else PRICE_INPUT * 0.25
MOCK_BILLABLE = os.getenv("MOCK_BILLABLE", "1") == "1"


def settings() -> dict:
    """Чинні ціни — для сторінки й журналу."""
    return {
        "input_per_1m": PRICE_INPUT,
        "output_per_1m": PRICE_OUTPUT,
        "cached_input_per_1m": PRICE_CACHED_INPUT,
        "mock_billable": MOCK_BILLABLE,
        "currency": "USD",
    }


def estimate(usage: dict | None, *, provider: str | None = None, model: str | None = None,
             text: str | None = None) -> tuple[float | None, bool]:
    """Оцінити вартість одного звертання до провайдера.

    `usage` — те, що повернув провайдер (може бути `None`); `provider` і
    `model` — хто відповідав; `text` — текст відповіді, якщо доведеться
    оцінювати без `usage`.

    Повертає пару: вартість у доларах (або `None`, якщо її не визначити)
    і ознаку «це оцінка, а не розрахунок за фактичним `usage`».
    """
    if provider == "mock" and not MOCK_BILLABLE:
        return 0.0, False

    if usage and isinstance(usage, dict):
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        completion_tokens = int(usage.get("completion_tokens") or 0)
        
        details = usage.get("prompt_tokens_details") or {}
        cached_tokens = 0
        if isinstance(details, dict):
            cached_tokens = int(details.get("cached_tokens") or 0)
        
        uncached_prompt_tokens = max(0, prompt_tokens - cached_tokens)
        
        cost = (
            (uncached_prompt_tokens / 1_000_000.0) * PRICE_INPUT +
            (cached_tokens / 1_000_000.0) * PRICE_CACHED_INPUT +
            (completion_tokens / 1_000_000.0) * PRICE_OUTPUT
        )
        return round(cost, 6), False

    # Якщо usage немає (обрив звʼязку, скасування, старий провайдер без stream_options)
    # Оцінюємо за довжиною тексту (приблизно 3.5 символи на токен для укр./англ.)
    # Стандартний вхідний prompt для чернетки складає близько 180 токенів.
    est_prompt_tokens = 180
    est_completion_tokens = max(10, int(len(text or "") / 3.5)) if text else 0

    est_cost = (
        (est_prompt_tokens / 1_000_000.0) * PRICE_INPUT +
        (est_completion_tokens / 1_000_000.0) * PRICE_OUTPUT
    )
    return round(est_cost, 6), True
