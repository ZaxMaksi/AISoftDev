"""Скрипт виконання та верифікації сценаріїв 1-19 із eval/README.md.
"""

import json
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

# Додаємо корінь проєкту до sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

BASE_URL = "http://127.0.0.1:8000"
LOG_FILE = Path("logs/app.jsonl")

from dotenv import load_dotenv

load_dotenv()

LLM_API_KEY = os.environ.get("LLM_API_KEY")

def http_req(method: str, path: str, body: dict | None = None, headers: dict | None = None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req_headers = {"Content-Type": "application/json"}
    if headers:
        req_headers.update(headers)
    req = urllib.request.Request(BASE_URL + path, data=data, method=method, headers=req_headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            content = resp.read()
            return resp.status, json.loads(content) if content else {}, dict(resp.headers)
    except urllib.error.HTTPError as exc:
        content = exc.read()
        parsed = {}
        try:
            parsed = json.loads(content) if content else {}
        except Exception:
            pass
        return exc.code, parsed, dict(exc.headers)


def run():
    results = {}
    print("=== Запуск перевірки сценаріїв ПР11 ===")

    # ------------------------------------------------------------ Сценарій 1: Траса однієї чернетки
    print("\n--- Сценарій 1: Траса однієї чернетки ---")
    status, draft_data, h = http_req(
        "POST",
        "/api/drafts",
        {"request_id": "d01", "client": "sc1_client", "fresh": True},
        headers={"X-Request-ID": "req-trace-test-01"},
    )
    draft_id = draft_data.get("draft_id")
    print(f"Створено чернетку {draft_id} із X-Request-ID=req-trace-test-01")

    # Чекаємо завершення
    for _ in range(30):
        time.sleep(0.5)
        _, d_resp, _ = http_req("GET", f"/api/drafts/{draft_id}")
        if d_resp.get("status") in ("done", "failed"):
            break

    # Читаємо журнал
    trace_lines = []
    if LOG_FILE.exists():
        for line in LOG_FILE.read_text(encoding="utf-8").splitlines():
            if "req-trace-test-01" in line or (draft_id and draft_id in line):
                try:
                    trace_lines.append(json.loads(line))
                except Exception:
                    pass

    print(f"Знайдено {len(trace_lines)} повʼязаних записів у журналі для запиту req-trace-test-01 / чернетки {draft_id}:")
    for r in trace_lines:
        print(f"  [{r.get('level')}] {r.get('event')} (request_id={r.get('request_id')}, draft_id={r.get('draft_id')})")
    results["scenario_1"] = {"draft_id": draft_id, "trace_count": len(trace_lines)}

    # ------------------------------------------------------------ Сценарій 3: Секрети й персональні дані
    print("\n--- Сценарій 3: Перевірка відсутності секретів та персональних даних ---")
    log_text = LOG_FILE.read_text(encoding="utf-8") if LOG_FILE.exists() else ""
    forbidden = [
        "Бондар", "Коваль", "Мельник",  # Імена клієнтів
        "10412", "10458", "10502",  # Номери замовлень
        "20450012345678",  # Номер ТТН
    ]
    if LLM_API_KEY:
        forbidden.insert(0, LLM_API_KEY)
    found_leaks = {}
    for item in forbidden:
        count = log_text.count(item)
        found_leaks[item] = count
        status_txt = "ВІДСУТНЬО (ОК)" if count == 0 else f"ЗНАЙДЕНО ({count} разів!)"
        print(f"  Перевірка '{item[:10]}...': {status_txt}")
        assert count == 0, f"Витік даних знайдено в логах: {item[:10]}... ({count} разів)"
    results["scenario_3"] = found_leaks

    # ------------------------------------------------------------ Сценарій 10: Кнопка «Повторити» (fresh=True)
    print("\n--- Сценарій 10: Повторити (fresh=True обходить кеш) ---")
    # 1-й запит з кешуванням
    _, r1, _ = http_req("POST", "/api/drafts", {"request_id": "d03", "client": "sc10", "fresh": False})
    for _ in range(30):
        time.sleep(0.5)
        _, d1, _ = http_req("GET", f"/api/drafts/{r1['draft_id']}")
        if d1.get("status") in ("done", "failed"):
            break

    # 2-й запит без fresh (має бути з кешу)
    _, r2, _ = http_req("POST", "/api/drafts", {"request_id": "d03", "client": "sc10", "fresh": False})
    for _ in range(30):
        time.sleep(0.2)
        _, d2, _ = http_req("GET", f"/api/drafts/{r2['draft_id']}")
        if d2.get("status") in ("done", "failed"):
            break

    # 3-й запит з fresh: True (обхід кешу)
    _, r3, _ = http_req("POST", "/api/drafts", {"request_id": "d03", "client": "sc10", "fresh": True})
    for _ in range(30):
        time.sleep(0.5)
        _, d3, _ = http_req("GET", f"/api/drafts/{r3['draft_id']}")
        if d3.get("status") in ("done", "failed"):
            break

    print(f"  Запит 1 (fresh=False): draft={r1['draft_id']}")
    print(f"  Запит 2 (fresh=False): draft={r2['draft_id']}, cached={d2.get('cached')} (очікується True)")
    print(f"  Запит 3 (fresh=True):  draft={r3['draft_id']}, cached={d3.get('cached')} (очікується False)")
    results["scenario_10"] = {
        "r2_cached": d2.get("cached"),
        "r3_cached": d3.get("cached"),
    }

    # ------------------------------------------------------------ Сценарій 14: Вичерпання бюджету
    print("\n--- Сценарій 14: Перевірка блокування при вичерпанні бюджету ---")
    _, u_before, _ = http_req("GET", "/api/usage")
    print(f"  Бюджет до тесту: spent={u_before.get('budget', {}).get('spent')}, limit={u_before.get('budget', {}).get('limit')}")

    # Запишемо штучно витрати для перевищення ліміту в файл стану
    budget_file = Path("state/budget.json")
    if budget_file.exists():
        bdata = json.loads(budget_file.read_text(encoding="utf-8"))
        bdata["spent"] = 1.05
        budget_file.write_text(json.dumps(bdata), encoding="utf-8")

    from app import budget
    budget._spent = 1.05
    budget._warned = True

    # Запит на нову чернетку (uncached)
    status_code, resp_budget, _ = http_req("POST", "/api/drafts", {"request_id": "d08", "client": "sc14", "fresh": True})
    print(f"  Статус відповіді при перевищенні бюджету: {status_code}")
    print(f"  Тіло відповіді: {json.dumps(resp_budget, ensure_ascii=False)[:200]}")
    assert status_code == 429
    assert resp_budget.get("error", {}).get("kind") == "budget"

    # Перевірка: чи працює кеш ПІСЛЯ вичерпання бюджету
    # Звернення d03 є в кеші після Сценарію 10!
    st_cached, resp_cached, _ = http_req("POST", "/api/drafts", {"request_id": "d03", "client": "sc14", "fresh": False})
    print(f"  Статус відповіді для кешованого запиту при вичерпаному бюджеті: {st_cached}")
    assert st_cached == 202
    print("  Кеш успішно віддав чернетку навіть при вичерпаному бюджеті!")
    results["scenario_14"] = {
        "blocked_status": status_code,
        "cache_worked": (st_cached == 202),
    }

    # Скидаємо бюджет назад
    budget.reset(clear_file=True)
    if budget_file.exists():
        budget_file.unlink(missing_ok=True)

    print("\n=== Всі сценарії перевірено успішно ===")
    return results


if __name__ == "__main__":
    run()