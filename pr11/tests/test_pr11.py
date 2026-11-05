"""Комплексний набір тестів для перевірки компонентів ПР10 та ПР11.
"""

import json
import os
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import budget, cache, costs, drafting, gateway, jobs, limits, main, samples, telemetry
from app.providers.base import ProviderError, Request, Result


@pytest.fixture(autouse=True)
def clean_environment(tmp_path, monkeypatch):
    """Ізоляція тестового середовища: чисті файли кешу та бюджету, швидкий mock-провайдер."""
    test_cache = str(tmp_path / "cache.json")
    test_budget = str(tmp_path / "budget.json")

    orig_cache_file = cache.CACHE_FILE
    orig_budget_file = budget.BUDGET_FILE

    cache.CACHE_FILE = test_cache
    budget.BUDGET_FILE = test_budget

    cache.reset(clear_file=True)
    budget.reset(clear_file=True)
    limits.reset()

    # У тестах використовуємо швидкий mock-провайдер
    monkeypatch.setattr(gateway, "PROVIDER", "mock")
    monkeypatch.setattr("app.providers.mock.FIRST_DELAY_MS", 10)
    monkeypatch.setattr("app.providers.mock.DELAY_MS", 2)

    yield

    cache.CACHE_FILE = orig_cache_file
    budget.BUDGET_FILE = orig_budget_file
    cache.reset(clear_file=True)
    budget.reset(clear_file=True)


# ------------------------------------------------------------ 1. Вартість (costs.py)


def test_costs_estimate_with_usage():
    """Перевірка точного розрахунку вартості за usage."""
    usage = {
        "prompt_tokens": 1000,
        "completion_tokens": 500,
        "total_tokens": 1500,
        "prompt_tokens_details": {"cached_tokens": 200},
    }
    cost, is_est = costs.estimate(usage, provider="openai")
    assert not is_est
    assert cost == pytest.approx(0.001505, rel=1e-4)


def test_costs_estimate_without_usage():
    """Перевірка оцінки вартості без usage (наприклад, при обриві)."""
    cost, is_est = costs.estimate(None, provider="openai", text="Тестова відповідь українською")
    assert is_est
    assert cost is not None
    assert cost > 0.0


def test_costs_mock_unbillable(monkeypatch):
    """Підставний провайдер з MOCK_BILLABLE=0 безкоштовний."""
    monkeypatch.setattr(costs, "MOCK_BILLABLE", False)
    cost, is_est = costs.estimate({"prompt_tokens": 500, "completion_tokens": 200}, provider="mock")
    assert not is_est
    assert cost == 0.0


# ------------------------------------------------------------ 2. Кешування (cache.py)


def test_cache_key_generation():
    """Формування ключа кешу: залежить від моделі, промпту, повідомлень, але не від id."""
    req1 = Request(
        messages=[{"role": "user", "content": "Привіт"}],
        max_tokens=100,
        temperature=0.3,
        model="test-model",
        metadata={"request_id": "req-1", "draft_id": "draft-1"},
    )
    req2 = Request(
        messages=[{"role": "user", "content": "Привіт"}],
        max_tokens=100,
        temperature=0.3,
        model="test-model",
        metadata={"request_id": "req-2", "draft_id": "draft-2"},
    )
    key1 = cache.make_key(req1, "drafts-v1")
    key2 = cache.make_key(req2, "drafts-v1")
    assert key1 == key2

    # Зміна версії prompt змінює ключ
    key_v2 = cache.make_key(req1, "drafts-v2")
    assert key1 != key_v2


def test_cache_put_get_and_ttl(monkeypatch):
    """Збереження, отримання та закінчення терміну дії TTL."""
    monkeypatch.setattr(cache, "TTL_SECONDS", 0.5)
    res = Result(text="Готовий текст", provider="mock", model="mock-1", elapsed=1.2)
    cache.put("k1", res)

    cached = cache.get("k1")
    assert cached is not None
    assert cached.text == "Готовий текст"
    assert cached.cached is True

    # Очікуємо закінчення TTL
    time.sleep(0.6)
    assert cache.get("k1") is None


def test_cache_disk_persistence(tmp_path):
    """Збереження кешу на диск та відновлення після перезапуску."""
    cache_f = str(tmp_path / "persisted_cache.json")
    cache.CACHE_FILE = cache_f

    res = Result(text="Персистентний текст", provider="mock", model="mock-1")
    cache.put("k_persist", res)

    # Імітуємо перезапуск модуля
    cache._store.clear()
    cache._load_from_disk()

    cached = cache.get("k_persist")
    assert cached is not None
    assert cached.text == "Персистентний текст"


# ------------------------------------------------------------ 3. Бюджет (budget.py)


def test_budget_check_and_record():
    """Облік витрат і перевірка ліміту."""
    budget.reset()
    assert budget.status()["spent"] == 0.0

    budget.record(0.10)
    assert budget.status()["spent"] == 0.10
    budget.check()  # Не має підіймати помилку


def test_budget_exceeded():
    """Вичерпання ліміту блокує подальші виклики."""
    budget.reset()
    budget.record(budget.DAILY_USD + 0.01)

    with pytest.raises(budget.BudgetExceeded) as exc_info:
        budget.check()

    assert exc_info.value.spent >= budget.DAILY_USD
    assert budget.status()["exceeded"] is True


def test_budget_warning_threshold():
    """Досягнення порогу WARN_AT виставляє прапорець warning."""
    budget.reset()
    budget.record(budget.DAILY_USD * 0.85)
    st = budget.status()
    assert st["warning"] is True
    assert not st["exceeded"]


def test_budget_disk_persistence(tmp_path):
    """Збереження стану бюджету на диску при перезапуску."""
    budget_f = str(tmp_path / "persisted_budget.json")
    budget.BUDGET_FILE = budget_f

    budget.reset()
    budget.record(0.45)

    # Імітуємо перезапуск
    budget._spent = 0.0
    budget._load_from_disk()

    assert budget.status()["spent"] == 0.45


# ------------------------------------------------------------ 4. Обмеження частоти (limits.py)


def test_rate_limiting():
    """Перевірка блокування при перевищенні RATE_LIMIT_PER_MINUTE."""
    limits.reset()
    client = "test-client-1"

    for _ in range(limits.PER_MINUTE):
        limits.acquire(client)

    with pytest.raises(limits.RateLimited) as exc_info:
        limits.acquire(client)

    assert exc_info.value.retry_after > 0


# ------------------------------------------------------------ 5. Безпека журналу (telemetry)


def test_json_formatter_structure():
    """Перевірка форматування записів: безпека та відсутність секретів."""
    import logging
    formatter = telemetry.JsonFormatter()
    record = logging.LogRecord("app.gateway", logging.INFO, "gateway.py", 42, "llm.attempt", (), None)
    record.request_id = "req-12345"
    record.draft_id = "d-67890"
    record.attempt = 1
    record.duration_ms = 450.5

    formatted = formatter.format(record)
    parsed = json.loads(formatted)

    assert parsed["event"] == "llm.attempt"
    assert parsed["request_id"] == "req-12345"
    assert parsed["draft_id"] == "d-67890"
    assert parsed["attempt"] == 1
    assert parsed["duration_ms"] == 450.5
    # Жодних секретних полів
    assert "api_key" not in parsed
    assert "prompt" not in parsed


# ------------------------------------------------------------ 6. Інтеграція FastAPI веб-рівня


def test_fastapi_endpoints_and_request_id():
    """Перевірка X-Request-ID, заголовків та відповідей API."""
    client = TestClient(main.app)

    # Health endpoint
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert "X-Request-ID" in resp.headers
    data = resp.json()
    assert "gateway" in data
    assert "cache" in data
    assert "budget" in data
    assert data["prompt_version"] == "drafts-v1"

    # Usage endpoint
    resp_usage = client.get("/api/usage")
    assert resp_usage.status_code == 200
    u_data = resp_usage.json()
    assert "budget" in u_data
    assert "cache" in u_data

    # Create draft
    resp_create = client.post("/api/drafts", json={"request_id": "d01", "client": "t_client"})
    assert resp_create.status_code == 202
    assert "X-Request-ID" in resp_create.headers
    draft = resp_create.json()
    assert draft["status"] in ("queued", "running", "done")
    assert "draft_id" in draft


def test_fastapi_budget_exceeded_error():
    """Коли бюджет вичерпано, API повертає HTTP 429 з request_id і resets_at."""
    client = TestClient(main.app)
    budget.reset()
    budget.record(budget.DAILY_USD + 0.1)

    resp = client.post("/api/drafts", json={"request_id": "d02", "client": "c1", "fresh": True})
    assert resp.status_code == 429
    assert "X-Request-ID" in resp.headers
    body = resp.json()
    assert "error" in body
    assert body["error"]["kind"] == "budget"
    assert "request_id" in body["error"]
    assert "resets_at" in body["error"]


# ------------------------------------------------------------ 7. Single-Flight та Аналізатор


def test_single_flight_coalescing():
    """Перевірка обʼєднання однакових запитів під час одночасної генерації."""
    req = Request(
        messages=[{"role": "user", "content": "Одночасне питання"}],
        max_tokens=50,
        temperature=0.0,
        model="mock-1",
        metadata={"provider": "mock"},
    )
    key = cache.make_key(req, "drafts-v1")

    # Перший запит реєструє подію
    cached_res1, ev1 = cache.get_or_wait_inflight(key)
    assert cached_res1 is None
    assert ev1 is not None

    # Другий запит у паралельному потоці має отримати той самий результат після release
    def complete_first():
        time.sleep(0.1)
        res = Result(text="Результат від 1-го", provider="mock", model="mock-1")
        cache.put(key, res)
        cache.release_inflight(key, ev1)

    import threading
    t = threading.Thread(target=complete_first)
    t.start()

    cached_res2, ev2 = cache.get_or_wait_inflight(key, timeout=2.0)
    assert cached_res2 is not None
    assert cached_res2.text == "Результат від 1-го"
    t.join()


def test_analyze_script_functions(capsys):
    """Перевірка коректності роботи функцій eval/analyze.py."""
    from eval import analyze

    sample_records = [
        {
            "ts": "2026-10-24T12:00:00Z",
            "level": "INFO",
            "event": "llm.attempt",
            "request_id": "r1",
            "draft_id": "d1",
            "attempt": 1,
            "provider": "mock",
            "model": "mock-1",
            "status": "success",
            "duration_ms": 250.0,
            "first_token_ms": 100.0,
            "queue_ms": 5.0,
            "cost": 0.001,
            "cached": False,
            "tokens_in": 150,
            "tokens_out": 80,
        },
        {
            "ts": "2026-10-24T12:01:00Z",
            "level": "INFO",
            "event": "llm.attempt",
            "request_id": "r2",
            "draft_id": "d2",
            "attempt": 1,
            "provider": "mock",
            "model": "mock-1",
            "status": "success",
            "duration_ms": 0.0,
            "cost": 0.0,
            "cached": True,
        },
        {
            "ts": "2026-10-24T12:02:00Z",
            "level": "ERROR",
            "event": "llm.attempt",
            "request_id": "r3",
            "draft_id": "d3",
            "attempt": 1,
            "provider": "mock",
            "model": "mock-1",
            "status": "error",
            "kind": "timeout",
            "duration_ms": 5000.0,
            "cost": 0.0005,
            "cached": False,
        },
    ]

    analyze.summary(sample_records)
    analyze.slow(sample_records)
    analyze.failed(sample_records)
    analyze.repeated(sample_records)
    analyze.spend(sample_records)
    analyze.cache_report(sample_records)

    captured = capsys.readouterr()
    assert "записів: 3" in captured.out
    assert "timeout" in captured.out
    assert "влучань у кеш (hits):        1" in captured.out
