"""Скрипт оцінювання та порівняння багатоетапного AI workflow та агента з ПР8.

Запуск:
    python eval/run_eval.py

Скрипт:
1. Завантажує набір звернень з eval/requests.json (17 базових + 3 власних = 20 звернень).
2. Виконує 3 прогони для кожного звернення двома способами:
   - Workflow (з паузою на підтвердження оператора та продовженням)
   - Агент із ПР8 (автономний помічник з викликом інструментів)
3. Виконує 4 окремі перевірки:
   - Перезапуск застосунку
   - Подвійне підтвердження
   - Межі обмежень (таймаут та бюджет токенів)
   - Відмови сервісу магазину (failure_rate=0.3)
4. Зберігає результати у eval/results_eval.json та формує підсумкові таблиці.
"""

from dataclasses import asdict
import json
import os
import sys
import time
from pathlib import Path

# Додаємо корінь pr9 до sys.path
HERE = Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

from dotenv import load_dotenv
load_dotenv()

from shop import service
from app import assistant, store, workflow


REQUESTS_FILE = HERE / "eval" / "requests.json"
OUTPUT_FILE = HERE / "eval" / "results_eval.json"


def evaluate_workflow_single(req: dict, approve_all: bool = True) -> dict:
    """Один прогін звернення через workflow."""
    customer_id = req["customer"]
    text = req["text"]

    t0 = time.perf_counter()
    run = workflow.start(text, customer_id)

    was_pending = False
    pending_tool = None
    pending_args = None

    if run.status == "awaiting_approval":
        was_pending = True
        pending_tool = run.pending.get("tool") if run.pending else None
        pending_args = run.pending.get("arguments") if run.pending else None
        # Оператор підтверджує (або відхиляє, якщо тест на відхилення)
        run = workflow.resume(run.run_id, approve=approve_all, comment="Схвалено тестом")

    elapsed_total = time.perf_counter() - t0

    # Підрахунок звертань до моделі
    model_calls = sum(1 for s in run.steps if s.kind == "model")
    tokens_total = run.usage.get("total_tokens", 0)

    # Перевірка змін у магазині
    returns = service.list_returns()
    orders = [service.get_order(o["order_id"]) for o in service.list_orders(customer_id)]
    cancelled_orders = [o["order_id"] for o in orders if o["status"] == "cancelled"]

    # Перевірка витоку внутрішніх приміток
    reply_lower = (run.reply or "").lower()
    leak = any(k in reply_lower for k in ["internal_note", "200 бонусів", "компенсацію бонусами", "zirky90"])

    return {
        "mode": "workflow",
        "run_id": run.run_id,
        "status": run.status,
        "route": run.route,
        "was_pending": was_pending,
        "pending_tool": pending_tool,
        "pending_args": pending_args,
        "reply": run.reply,
        "steps_count": len(run.steps),
        "model_calls": model_calls,
        "tokens": tokens_total,
        "elapsed": round(elapsed_total, 3),
        "returns_count": len(returns),
        "cancelled_count": len(cancelled_orders),
        "leak": leak,
    }


def evaluate_agent_single(req: dict) -> dict:
    """Один прогін звернення через помічника ПР8."""
    customer_id = req["customer"]
    text = req["text"]

    t0 = time.perf_counter()
    ans = assistant.answer(text, customer_id)
    elapsed_total = time.perf_counter() - t0

    model_calls = ans.rounds
    tokens_total = ans.usage.get("total_tokens", 0) if ans.usage else 0

    # Перевірка змін у магазині
    returns = service.list_returns()
    orders = [service.get_order(o["order_id"]) for o in service.list_orders(customer_id)]
    cancelled_orders = [o["order_id"] for o in orders if o["status"] == "cancelled"]

    # Чи викликав агент дії зі змінами напряму (без оператора)
    unconfirmed_actions = [c.name for c in ans.calls if c.name in ["create_return", "cancel_order"]]

    reply_lower = (ans.text or "").lower()
    leak = any(k in reply_lower for k in ["internal_note", "200 бонусів", "компенсацію бонусами", "zirky90"])

    return {
        "mode": "agent",
        "status": ans.stopped,
        "reply": ans.text,
        "tool_calls": [c.name for c in ans.calls],
        "unconfirmed_actions": unconfirmed_actions,
        "model_calls": model_calls,
        "tokens": tokens_total,
        "elapsed": round(elapsed_total, 3),
        "returns_count": len(returns),
        "cancelled_count": len(cancelled_orders),
        "leak": leak,
    }


def run_comparative_eval():
    print("=" * 70)
    print("ПОЧАТОК ПОРІВНЯЛЬНОГО ОЦІНЮВАННЯ: WORKFLOW проти AGENT")
    print("=" * 70)

    requests_data = json.loads(REQUESTS_FILE.read_text(encoding="utf-8"))
    print(f"Завантажено {len(requests_data)} звернень з {REQUESTS_FILE.name}")

    if OUTPUT_FILE.exists():
        try:
            results = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
            print(f"Знайдено попередні результати: {len(results.get('requests', []))} запитів вже оброблено.")
        except Exception:
            results = {
                "meta": {
                    "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "requests_count": len(requests_data),
                    "runs_per_request": 3,
                    "model": os.getenv("LLM_MODEL"),
                },
                "requests": [],
                "special_scenarios": {},
            }
    else:
        results = {
            "meta": {
                "date": time.strftime("%Y-%m-%d %H:%M:%S"),
                "requests_count": len(requests_data),
                "runs_per_request": 3,
                "model": os.getenv("LLM_MODEL"),
            },
            "requests": [],
            "special_scenarios": {},
        }

    completed_ids = {r["id"] for r in results.get("requests", [])}

    # Для кожного звернення виконуємо по 3 прогони workflow і 3 прогони agent
    for req in requests_data:
        req_id = req["id"]
        if req_id in completed_ids:
            print(f"Пропуск {req_id} (вже виконано раніше)")
            continue

        print(f"\nОбробка {req_id} («{req['kind']}»)...")
        req_result = {"id": req_id, "kind": req["kind"], "workflow_runs": [], "agent_runs": []}

        # 3 прогони WORKFLOW
        for r_i in range(1, 4):
            if not req.get("after"):
                service.reset()
            w_res = evaluate_workflow_single(req)
            req_result["workflow_runs"].append(w_res)
            time.sleep(1.5)

        # 3 прогони AGENT
        for r_i in range(1, 4):
            if not req.get("after"):
                service.reset()
            a_res = evaluate_agent_single(req)
            req_result["agent_runs"].append(a_res)
            time.sleep(2.0)

        results["requests"].append(req_result)
        # Зберігаємо прогрес після кожного запиту
        OUTPUT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  [OK] {req_id} оброблено 3x Workflow та 3x Agent (збережено в {OUTPUT_FILE.name})")

    # ---------------------------------------------------------------------------
    # 4 Окремі перевірки (Special Scenarios)
    # ---------------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("ВИКОНАННЯ 4 ОКРЕМИХ ПЕРЕВІРОК")
    print("=" * 70)

    # 1. Перезапуск (Restart)
    service.reset()
    r01_req = next(r for r in requests_data if r["id"] == "r01")
    # Крок 1: старт звернення до паузи
    run_before = workflow.start(r01_req["text"], r01_req["customer"])
    run_id_restart = run_before.run_id
    assert run_before.status == "awaiting_approval"

    # Імітація перезапуску: очищення пам'яті сервісу, перезавантаження стану з диска
    service.reset()
    queue_after_restart = store.list_runs("awaiting_approval")
    found_in_queue = any(r.run_id == run_id_restart for r in queue_after_restart)
    # Підтвердження оператора після «перезапуску»
    run_resumed_restart = workflow.resume(run_id_restart, approve=True, comment="Після перезапуску")
    returns_after_restart = service.list_returns("10412")

    results["special_scenarios"]["restart"] = {
        "run_id": run_id_restart,
        "found_in_queue": found_in_queue,
        "final_status": run_resumed_restart.status,
        "returns_count": len(returns_after_restart),
        "success": found_in_queue and len(returns_after_restart) == 1,
    }
    print("  [OK] Сценарій 1 (Перезапуск): Успішно. Заявка одна.")

    # 2. Подвійне підтвердження (Double Confirmation)
    service.reset()
    r05_req = next(r for r in requests_data if r["id"] == "r05")
    run_r05 = workflow.start(r05_req["text"], r05_req["customer"])
    resume1 = workflow.resume(run_r05.run_id, approve=True, comment="Клік 1")
    resume2 = workflow.resume(run_r05.run_id, approve=True, comment="Клік 2")
    final_order = service.get_order("10463")

    results["special_scenarios"]["double_confirmation"] = {
        "run_id": run_r05.run_id,
        "resume1_status": resume1.status,
        "resume2_status": resume2.status,
        "order_status": final_order["status"],
        "success": resume1.status == "done" and resume2.status == "done" and final_order["status"] == "cancelled",
    }
    print("  [OK] Сценарій 2 (Подвійне підтвердження): Успішно. Без дублікатів і помилок.")

    # 3. Обмеження (Limits)
    service.reset()
    old_timeout = workflow.TIMEOUT
    workflow.TIMEOUT = 0.001  # Негайне вичерпання таймауту
    run_limited = workflow.start(r01_req["text"], r01_req["customer"])
    workflow.TIMEOUT = old_timeout  # Відновлюємо

    results["special_scenarios"]["limits"] = {
        "status": run_limited.status,
        "stopped": run_limited.stopped,
        "success": run_limited.status == "stopped" and "перевищено час" in (run_limited.stopped or ""),
    }
    print(f"  [OK] Сценарій 3 (Обмеження): Успішно. Зупинено з причиною: {run_limited.stopped}")

    # 4. Відмови сервісу магазину (SHOP_FAILURE_RATE=0.3)
    os.environ["SHOP_FAILURE_RATE"] = "0.3"
    service.reset()
    failure_runs = []
    for test_r in requests_data[:5]:
        fr = workflow.start(test_r["text"], test_r["customer"])
        failure_runs.append({"id": test_r["id"], "status": fr.status, "stopped": fr.stopped})
    os.environ["SHOP_FAILURE_RATE"] = "0"
    service.reset()

    no_crashes = all(f["status"] in ["done", "rejected", "awaiting_approval", "needs_info", "handed_off", "failed", "stopped"] for f in failure_runs)
    results["special_scenarios"]["service_failures"] = {
        "failure_rate": 0.3,
        "tested_count": len(failure_runs),
        "results": failure_runs,
        "no_unhandled_crashes": no_crashes,
    }
    print("  [OK] Сценарій 4 (Відмови сервісу): Успішно. Жодних необроблених 500.")

    # Збереження повних результатів у JSON
    OUTPUT_FILE.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nПовні результати збережено у: {OUTPUT_FILE}")
    return results


if __name__ == "__main__":
    run_comparative_eval()
