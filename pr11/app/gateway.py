"""Шлюз до моделі: єдиний шлях, яким застосунок звертається до провайдера.

Відповідає за:
- вибір провайдера та запасного варіанту (fallback);
- таймаути (на спробу, на бездіяльність у streaming, на весь запит);
- повторні спроби (exponential backoff з jitter);
- обмеження одночасності (MAX_CONCURRENT);
- перевірку кешу та збереження відповідей;
- контроль та облік денного бюджету витрат;
- структуроване логування викликів у JSON lines.
"""

import logging
import os
import queue
import random
import threading
import time
from typing import Callable

from dotenv import load_dotenv

from . import budget, cache, costs, telemetry
from .providers.base import Provider, ProviderError, Request, Result
from .providers.mock import MockProvider
from .providers.openai_compat import OpenAICompatProvider

load_dotenv()

log = logging.getLogger("app.gateway")

PROVIDER = os.getenv("LLM_PROVIDER", "openai")
FALLBACK = os.getenv("LLM_FALLBACK", "none")

TIMEOUT = float(os.getenv("LLM_TIMEOUT", "20"))
IDLE_TIMEOUT = float(os.getenv("LLM_IDLE_TIMEOUT", "8"))
TOTAL_TIMEOUT = float(os.getenv("LLM_TOTAL_TIMEOUT", "45"))
MAX_ATTEMPTS = int(os.getenv("LLM_MAX_ATTEMPTS", "3"))
BACKOFF_BASE = float(os.getenv("LLM_BACKOFF_BASE", "0.5"))
BACKOFF_MAX = float(os.getenv("LLM_BACKOFF_MAX", "8"))
MAX_CONCURRENT = int(os.getenv("LLM_MAX_CONCURRENT", "2"))

MODEL = os.getenv("LLM_MODEL")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "600"))

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

_semaphore = threading.Semaphore(MAX_CONCURRENT)


def settings() -> dict:
    """Чинні налаштування шлюзу — для сторінки й журналу."""
    return {
        "provider": PROVIDER,
        "fallback": FALLBACK,
        "model": MODEL,
        "timeout": TIMEOUT,
        "idle_timeout": IDLE_TIMEOUT,
        "total_timeout": TOTAL_TIMEOUT,
        "max_attempts": MAX_ATTEMPTS,
        "max_concurrent": MAX_CONCURRENT,
    }


def make_provider(name: str) -> Provider:
    """Зібрати провайдера за назвою: `openai`, `mock` або резервна модель."""
    if name == "mock":
        return MockProvider()
    elif name == "openai":
        return OpenAICompatProvider(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            model=MODEL or "gemini-3.8-flash",
            timeout=TIMEOUT,
        )
    elif name != "none":
        # Запасна модель через той самий OpenAI-сумісний ендпоінт
        return OpenAICompatProvider(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            model=name,
            timeout=TIMEOUT,
        )
    raise ValueError(f"Невідомий провайдер: {name}")


def _stream_with_idle_timeout(
    provider: Provider,
    request: Request,
    on_delta: Callable[[str], None],
    is_cancelled: Callable[[], bool],
    idle_timeout: float,
) -> tuple[str, dict | None, float]:
    """Вичитування потоку чанків із чергою та тайм-аутом бездіяльності (IDLE_TIMEOUT).
    Повертає (accumulated_text, usage_dict, first_token_duration).
    """
    chunk_queue: queue.Queue = queue.Queue(maxsize=100)
    stop_event = threading.Event()
    error_holder: list[Exception] = []

    def producer():
        try:
            for chunk in provider.stream(request):
                if stop_event.is_set():
                    break
                chunk_queue.put(chunk)
            chunk_queue.put(None)  # Кінець потоку
        except Exception as e:
            error_holder.append(e)
            chunk_queue.put(None)

    worker = threading.Thread(target=producer, daemon=True)
    worker.start()

    accumulated: list[str] = []
    first_token_time: float | None = None
    stream_started = time.perf_counter()
    usage = None

    while True:
        if is_cancelled():
            stop_event.set()
            raise ProviderError("cancelled", "Генерацію скасовано користувачем", retryable=False)

        try:
            item = chunk_queue.get(timeout=idle_timeout)
        except queue.Empty:
            stop_event.set()
            raise ProviderError("timeout", f"Таймаут бездіяльності у streaming ({idle_timeout} с)", retryable=True)

        if item is None:
            # Завершення або помилка
            if error_holder:
                exc = error_holder[0]
                if isinstance(exc, ProviderError):
                    raise exc
                raise ProviderError("unknown", str(exc), detail=str(exc))
            break

        # Отримано черговий фрагмент
        if first_token_time is None and item.text:
            first_token_time = time.perf_counter() - stream_started

        if item.text:
            accumulated.append(item.text)
            on_delta(item.text)

        if item.usage:
            usage = item.usage

    stop_event.set()
    full_text = "".join(accumulated)
    first_dur = first_token_time if first_token_time is not None else (time.perf_counter() - stream_started)
    return full_text, usage, first_dur


def generate(
    request: Request,
    on_delta: Callable[[str], None],
    on_status: Callable[[dict], None],
    is_cancelled: Callable[[], bool],
) -> Result:
    """Згенерувати відповідь, дотримуючись таймаутів, спроб, обмеження
    одночасності, кешу, ліміту бюджету й запасного варіанту.
    """
    call_started = time.perf_counter()
    deadline = call_started + TOTAL_TIMEOUT

    metadata = request.metadata or {}
    req_id = metadata.get("request_id", "")
    draft_id = metadata.get("draft_id", "")
    client_id = metadata.get("client", "anonymous")
    prompt_version = metadata.get("prompt_version", "drafts-v1")
    fresh = bool(metadata.get("fresh", False))

    cache_key = cache.make_key(request, prompt_version)

    # 1. Перевірка кешу (якщо не fresh)
    if not fresh:
        cached_result, inflight_ev = cache.get_or_wait_inflight(cache_key)
        if cached_result is not None:
            log.info("llm.attempt", extra={
                "request_id": req_id,
                "draft_id": draft_id,
                "client": client_id,
                "attempt": 1,
                "provider": cached_result.provider,
                "model": cached_result.model,
                "prompt_version": prompt_version,
                "status": "success",
                "kind": None,
                "queue_ms": 0.0,
                "first_token_ms": 0.0,
                "duration_ms": 0.0,
                "fallback": False,
                "tokens_in": (cached_result.usage or {}).get("prompt_tokens", 0) if cached_result.usage else 0,
                "tokens_out": (cached_result.usage or {}).get("completion_tokens", 0) if cached_result.usage else 0,
                "tokens_cached": 0,
                "cost": 0.0,
                "cost_estimated": False,
                "cached": True,
            })
            on_status({"cached": True, "provider": cached_result.provider, "model": cached_result.model})
            on_delta(cached_result.text)
            return cached_result
    else:
        cache.record_bypassed()
        inflight_ev = None

    # 2. Перевірка бюджету перед платним викликом
    try:
        budget.check()
    except budget.BudgetExceeded as b_err:
        cache.release_inflight(cache_key, inflight_ev)
        err = ProviderError("budget", str(b_err), retryable=False, detail=f"spent={b_err.spent}, limit={b_err.limit}")
        log.error("llm.attempt", extra={
            "request_id": req_id,
            "draft_id": draft_id,
            "client": client_id,
            "attempt": 1,
            "provider": PROVIDER,
            "model": MODEL or "",
            "prompt_version": prompt_version,
            "status": "error",
            "kind": "budget",
            "queue_ms": 0.0,
            "first_token_ms": 0.0,
            "duration_ms": round((time.perf_counter() - call_started) * 1000, 1),
            "fallback": False,
            "cached": False,
        })
        raise err

    # 3. Очікування в черзі семафора одночасності
    q_start = time.perf_counter()
    _semaphore.acquire()
    queue_wait_ms = round((time.perf_counter() - q_start) * 1000, 1)

    # Список провайдерів для послідовного виконання: основний -> запасний (fallback)
    plan = [PROVIDER]
    if FALLBACK and FALLBACK != "none":
        plan.append(FALLBACK)

    last_error: ProviderError | None = None
    attempt_counter = 0

    try:
        for plan_idx, current_provider_name in enumerate(plan):
            is_fallback = (plan_idx > 0)
            try:
                active_provider = make_provider(current_provider_name)
            except Exception as e:
                log.error("llm.provider_init_failed", extra={"provider": current_provider_name, "error": str(e)})
                continue

            # Для fallback виконуємо 1 спробу, для основного — до MAX_ATTEMPTS
            max_provider_attempts = 1 if is_fallback else MAX_ATTEMPTS

            for attempt in range(1, max_provider_attempts + 1):
                attempt_counter += 1
                if is_cancelled():
                    raise ProviderError("cancelled", "Запит скасовано", retryable=False)

                # Перевірка загального таймауту
                remaining_total = deadline - time.perf_counter()
                if remaining_total <= 0:
                    raise ProviderError("timeout", f"Вичерпано загальний ліміт часу ({TOTAL_TIMEOUT} с)", retryable=False)

                on_status({
                    "status": "running",
                    "attempt": attempt_counter,
                    "provider": current_provider_name,
                    "fallback": is_fallback,
                })

                attempt_started = time.perf_counter()
                try:
                    full_text, usage, first_token_dur = _stream_with_idle_timeout(
                        provider=active_provider,
                        request=request,
                        on_delta=on_delta,
                        is_cancelled=is_cancelled,
                        idle_timeout=IDLE_TIMEOUT,
                    )
                    attempt_dur = time.perf_counter() - attempt_started

                    # Розрахунок вартості
                    actual_model = getattr(active_provider, "model", MODEL or "unknown")
                    cost, is_est = costs.estimate(
                        usage,
                        provider=current_provider_name,
                        model=actual_model,
                        text=full_text,
                    )

                    # Облік у бюджеті
                    budget.record(cost)

                    p_tokens = (usage or {}).get("prompt_tokens", 0) if usage else 0
                    c_tokens = (usage or {}).get("completion_tokens", 0) if usage else 0
                    cached_tokens = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens", 0) if usage else 0

                    result = Result(
                        text=full_text,
                        provider=current_provider_name,
                        model=actual_model,
                        usage=usage,
                        finish_reason="stop",
                        elapsed=round(attempt_dur, 2),
                        cost=cost,
                        cost_estimated=is_est,
                        cached=False,
                    )

                    # Запис у кеш
                    cache.put(cache_key, result)

                    # Журнал успішної спроби
                    log.info("llm.attempt", extra={
                        "request_id": req_id,
                        "draft_id": draft_id,
                        "client": client_id,
                        "attempt": attempt_counter,
                        "provider": current_provider_name,
                        "model": actual_model,
                        "prompt_version": prompt_version,
                        "status": "success",
                        "kind": None,
                        "queue_ms": queue_wait_ms,
                        "first_token_ms": round(first_token_dur * 1000, 1),
                        "duration_ms": round(attempt_dur * 1000, 1),
                        "fallback": is_fallback,
                        "tokens_in": p_tokens,
                        "tokens_out": c_tokens,
                        "tokens_cached": cached_tokens,
                        "cost": cost,
                        "cost_estimated": is_est,
                        "cached": False,
                    })

                    return result

                except ProviderError as err:
                    attempt_dur = time.perf_counter() - attempt_started
                    last_error = err

                    # Обчислення вартості для невдалої спроби (якщо є текст або провайдер)
                    fail_cost, is_fail_est = costs.estimate(None, provider=current_provider_name, text="")
                    if fail_cost and fail_cost > 0:
                        budget.record(fail_cost)

                    # Логування невдалої спроби
                    level = logging.WARNING if (err.retryable and attempt < max_provider_attempts) else logging.ERROR
                    log.log(level, "llm.attempt", extra={
                        "request_id": req_id,
                        "draft_id": draft_id,
                        "client": client_id,
                        "attempt": attempt_counter,
                        "provider": current_provider_name,
                        "model": getattr(active_provider, "model", MODEL or ""),
                        "prompt_version": prompt_version,
                        "status": "error",
                        "kind": err.kind,
                        "queue_ms": queue_wait_ms,
                        "first_token_ms": None,
                        "duration_ms": round(attempt_dur * 1000, 1),
                        "fallback": is_fallback,
                        "cost": fail_cost,
                        "cost_estimated": is_fail_est,
                        "cached": False,
                        "detail": err.detail or err.message,
                    })

                    if not err.retryable or err.kind == "cancelled":
                        raise err

                    # Якщо збій retryable і є спроби — робимо паузу з backoff
                    if attempt < max_provider_attempts:
                        backoff = min(BACKOFF_MAX, BACKOFF_BASE * (2 ** (attempt - 1))) + random.uniform(0.05, 0.2)
                        if err.retry_after:
                            backoff = max(backoff, err.retry_after)

                        if time.perf_counter() + backoff > deadline:
                            raise ProviderError("timeout", f"Вичерпано загальний ліміт часу перед повтором ({TOTAL_TIMEOUT} с)", retryable=False)

                        time.sleep(backoff)

        # Якщо всі спроби й запасні варіанти вичерпано
        if last_error:
            raise last_error
        raise ProviderError("unavailable", "Не вдалося отримати відповідь від жодного провайдера", retryable=False)

    finally:
        _semaphore.release()
        cache.release_inflight(cache_key, inflight_ev)
