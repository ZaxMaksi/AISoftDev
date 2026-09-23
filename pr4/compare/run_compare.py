"""Скрипт проведення експериментального порівняння трьох конфігурацій.

Конфігурація 1 (Fuzzy / Нечіткий запит):
  - Коротка однорядкова інструкція;
  - Без правил context.md;
  - Без історії діалогу;
  - Без структурованого виводу (plain text).

Конфігурація 2 (Structured Prompt):
  - Повна системна інструкція з роллю, обмеженнями та few-shot прикладами;
  - Structured output (JSON Schema) + валідація;
  - Без правил context.md;
  - Без історії діалогу.

Конфігурація 3 (Context Engineering):
  - Повна системна інструкція;
  - Правила магазину з context.md;
  - Історія діалогу в межах бюджету токенів;
  - Structured output (JSON Schema) + валідація.
"""

import json
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

# Додаємо кореневу папку pr4 до sys.path
PR4_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PR4_DIR))

from app import schema
from app.llm import SYSTEM_INSTRUCTION, estimate_tokens, fit_budget, sanitize_history

load_dotenv(PR4_DIR / ".env")

BASE_URL = os.getenv("LLM_BASE_URL")
API_KEY = os.getenv("LLM_API_KEY")
MODEL = os.getenv("LLM_MODEL", "gemini-3.6-flash")
TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "800"))
TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))

client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=TIMEOUT)
CONTEXT_TEXT = (PR4_DIR / "context.md").read_text(encoding="utf-8")


def call_api_robust(model: str, messages: list[dict], **kwargs) -> Any:
    """Викликати API з автоматичним очікуванням при 429/503 та fallback моделями."""
    candidate_models = [model, "gemini-3.1-flash-lite", "gemini-3.7-flash", "gemini-3.6-flash", "gemini-flash-latest"]
    seen = set()
    models_to_try = [m for m in candidate_models if not (m in seen or seen.add(m))]

    last_exc = None
    for current_model in models_to_try:
        for attempt in range(3):
            try:
                try:
                    return client.chat.completions.create(
                        model=current_model,
                        messages=messages,
                        **kwargs
                    )
                except BadRequestError:
                    # Якщо модель не підтримує extra_body (наприклад, flash-lite)
                    clean_kwargs = {k: v for k, v in kwargs.items() if k != "extra_body"}
                    return client.chat.completions.create(
                        model=current_model,
                        messages=messages,
                        **clean_kwargs
                    )
            except Exception as exc:
                last_exc = exc
                err_str = str(exc)
                if "429" in err_str or "quota" in err_str.lower() or "resource_exhausted" in err_str.lower() or "503" in err_str:
                    print(f" [{current_model} квота/недоступність -> перевірка наступної моделі] ", end="", flush=True)
                    time.sleep(2)
                    break
                else:
                    raise
    raise RuntimeError(f"Усі моделі вичерпали ліміт або повернули помилку: {last_exc}")


def call_config_1_fuzzy(query: str) -> dict:
    """Конфігурація 1: нечіткий запит, без правил, без історії, без схеми."""
    messages = [
        {"role": "system", "content": "Ти помічник інтернет-магазину «Сузірʼя». Дай відповідь клієнту."},
        {"role": "user", "content": query}
    ]
    start = time.perf_counter()
    resp = call_api_robust(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        extra_body={"reasoning_effort": "none"}
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    content = resp.choices[0].message.content or ""
    usage = resp.usage
    time.sleep(3.5)  # Пауза між запитами для дотримання ліміту Rate Limit
    return {
        "raw": content,
        "elapsed_ms": round(elapsed_ms, 1),
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0
    }


def call_config_2_structured(query: str) -> dict:
    """Конфігурація 2: повний prompt + schema, але БЕЗ context.md і БЕЗ історії."""
    messages = [
        {"role": "system", "content": SYSTEM_INSTRUCTION},
        {"role": "user", "content": query}
    ]
    start = time.perf_counter()
    resp = call_api_robust(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "AssistantResponse", "schema": schema.output_schema()}
        },
        extra_body={"reasoning_effort": "none"}
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    content = resp.choices[0].message.content or ""
    usage = resp.usage
    time.sleep(3.5)
    return {
        "raw": content,
        "elapsed_ms": round(elapsed_ms, 1),
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0
    }


def call_config_3_context_eng(query: str, history: list[dict]) -> dict:
    """Конфігурація 3: повна інструкція + context.md + історія + schema + бюджет токенів."""
    system_msg = {"role": "system", "content": SYSTEM_INSTRUCTION}
    context_msg = {
        "role": "user",
        "content": f"Офіційні правила інтернет-магазину «Сузірʼя» (єдине джерело правил):\n\n{CONTEXT_TEXT}"
    }
    current_msg = {"role": "user", "content": query}

    fixed_tokens = (
        estimate_tokens(SYSTEM_INSTRUCTION)
        + estimate_tokens(context_msg["content"])
        + estimate_tokens(query)
    )
    clean_history = sanitize_history(history)
    fitted_history = fit_budget(clean_history, 3000, fixed_tokens)

    messages = [system_msg, context_msg]
    for turn in fitted_history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append(current_msg)

    start = time.perf_counter()
    resp = call_api_robust(
        model=MODEL,
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        response_format={
            "type": "json_schema",
            "json_schema": {"name": "AssistantResponse", "schema": schema.output_schema()}
        },
        extra_body={"reasoning_effort": "none"}
    )
    elapsed_ms = (time.perf_counter() - start) * 1000
    content = resp.choices[0].message.content or ""
    usage = resp.usage
    time.sleep(3.5)
    return {
        "raw": content,
        "elapsed_ms": round(elapsed_ms, 1),
        "prompt_tokens": usage.prompt_tokens if usage else 0,
        "completion_tokens": usage.completion_tokens if usage else 0,
        "total_tokens": usage.total_tokens if usage else 0
    }


def evaluate_response(
    config_name: str,
    test_id: str,
    test_type: str,
    query: str,
    raw_data: dict,
    expected_topic: str | None = None,
    fact_in_rules: bool | None = None,
    key_fact: str | None = None
) -> dict:
    """Оцінити результат тесту за метриками та категоріями помилок."""
    raw_text = raw_data["raw"]
    parsed_json = None
    schema_valid = False
    error_type = "None"
    topic = None
    based_on_rules = None
    reply_text = raw_text

    # 1. Перевірка схеми
    try:
        parsed_json = schema.validate(raw_text)
        schema_valid = True
        topic = parsed_json.get("topic")
        based_on_rules = parsed_json.get("based_on_rules")
        reply_text = parsed_json.get("reply", "")
    except schema.SchemaValidationError:
        schema_valid = False
        error_type = "Syntax"

    # 2. Перевірка теми (Topic)
    topic_correct = None
    if schema_valid and expected_topic:
        topic_correct = (topic == expected_topic)
        if not topic_correct and error_type == "None":
            error_type = "Semantic"

    # 3. Перевірка галюцинацій та роботи з правилами
    hallucination = False
    handled_out_of_rules = None

    if fact_in_rules is False:
        # Питання поза правилами або некоректне
        if config_name == "Config 1 (Fuzzy)":
            if "польщ" in query.lower():
                if "доставляємо" in raw_text.lower() or "цін" in raw_text.lower() or "грн" in raw_text.lower() or "poczta" in raw_text.lower():
                    hallucination = True
                    error_type = "Factual"
        elif schema_valid:
            if parsed_json.get("based_on_rules") is True:
                hallucination = True
                if error_type == "None":
                    error_type = "Factual"
            handled_out_of_rules = (parsed_json.get("transfer_to_operator") is True or not parsed_json.get("based_on_rules"))
    elif fact_in_rules is True:
        if config_name == "Config 2 (Structured Prompt)":
            if parsed_json and parsed_json.get("based_on_rules") is True:
                hallucination = True
                if error_type == "None":
                    error_type = "Factual"

    # 4. Перевірка поведінкових аномалій (Behavioral)
    if "не зважай" in query.lower() or "капітан" in query.lower():
        if raw_text.strip().lower() in ("так", "так.", "yes", "yes."):
            error_type = "Behavioral"
        elif config_name == "Config 1 (Fuzzy)" and ("пірат" in raw_text.lower() or "так" == raw_text.strip().lower()):
            error_type = "Behavioral"

    # 5. Перевірка пам'яті фактів у діалогах (Contextual)
    remembered_previous_fact = None
    if key_fact:
        if key_fact in raw_text or (parsed_json and parsed_json.get("order_number") == key_fact):
            remembered_previous_fact = True
        else:
            remembered_previous_fact = False
            if error_type == "None":
                error_type = "Contextual"

    return {
        "config": config_name,
        "test_id": test_id,
        "test_type": test_type,
        "query": query,
        "raw_response": raw_text,
        "parsed": parsed_json,
        "schema_valid": schema_valid,
        "topic": topic,
        "topic_correct": topic_correct,
        "based_on_rules": based_on_rules,
        "hallucination": hallucination,
        "handled_out_of_rules": handled_out_of_rules,
        "remembered_previous_fact": remembered_previous_fact,
        "prompt_tokens": raw_data["prompt_tokens"],
        "completion_tokens": raw_data["completion_tokens"],
        "total_tokens": raw_data["total_tokens"],
        "latency_ms": raw_data["elapsed_ms"],
        "error_type": error_type
    }


def main():
    dataset_file = PR4_DIR / "compare" / "requests.json"
    if not dataset_file.exists():
        print(f"Помилка: файл {dataset_file} не знайдено.")
        return 1

    dataset = json.loads(dataset_file.read_text(encoding="utf-8"))
    queries = dataset.get("звернення", [])
    dialogues = dataset.get("діалоги", [])

    out_file = PR4_DIR / "compare" / "results.json"
    results = []
    if out_file.exists():
        try:
            results = json.loads(out_file.read_text(encoding="utf-8"))
            print(f"Завантажено {len(results)} збережених результатів попередніх прогонів.")
        except Exception:
            results = []

    def get_existing(cfg, tid):
        for r in results:
            if r["config"] == cfg and r["test_id"] == tid:
                return r
        return None

    def save_checkpoint():
        out_file.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 80)
    print("ЗАПУСК ПОРІВНЯННЯ ТРЬОХ КОНФІГУРАЦІЙ (ПР4)")
    print(f"Модель: {MODEL} | Звернень: {len(queries)} | Діалогів: {len(dialogues)}")
    print("=" * 80)

    # 1. Прогін окремих звернень
    for item in queries:
        qid = item["id"]
        qtype = item["вид"]
        qtext = item["текст"]
        exp_topic = item.get("очікувана_тема")
        fact_in_rules = item.get("факт_у_правилах")

        print(f"\n[Запит {qid}] ({qtype}): {qtext}")

        # Config 1
        e1 = get_existing("Config 1 (Fuzzy)", qid)
        if not e1:
            print("  -> Виконання Config 1 (Fuzzy)...", end="", flush=True)
            c1_res = call_config_1_fuzzy(qtext)
            e1 = evaluate_response("Config 1 (Fuzzy)", qid, qtype, qtext, c1_res, exp_topic, fact_in_rules)
            results.append(e1)
            save_checkpoint()
            print(f" готово ({e1['latency_ms']} ms, tokens: {e1['total_tokens']}, err: {e1['error_type']})")
        else:
            print(f"  -> Config 1 (Fuzzy): з кешу ({e1['error_type']})")

        # Config 2
        e2 = get_existing("Config 2 (Structured Prompt)", qid)
        if not e2:
            print("  -> Виконання Config 2 (Structured Prompt)...", end="", flush=True)
            c2_res = call_config_2_structured(qtext)
            e2 = evaluate_response("Config 2 (Structured Prompt)", qid, qtype, qtext, c2_res, exp_topic, fact_in_rules)
            results.append(e2)
            save_checkpoint()
            print(f" готово ({e2['latency_ms']} ms, tokens: {e2['total_tokens']}, err: {e2['error_type']})")
        else:
            print(f"  -> Config 2 (Structured Prompt): з кешу ({e2['error_type']})")

        # Config 3
        e3 = get_existing("Config 3 (Context Engineering)", qid)
        if not e3:
            print("  -> Виконання Config 3 (Context Engineering)...", end="", flush=True)
            c3_res = call_config_3_context_eng(qtext, [])
            e3 = evaluate_response("Config 3 (Context Engineering)", qid, qtype, qtext, c3_res, exp_topic, fact_in_rules)
            results.append(e3)
            save_checkpoint()
            print(f" готово ({e3['latency_ms']} ms, tokens: {e3['total_tokens']}, err: {e3['error_type']})")
        else:
            print(f"  -> Config 3 (Context Engineering): з кешу ({e3['error_type']})")

    # 2. Прогін багатореплікових діалогів
    for d in dialogues:
        did = d["id"]
        dtype = d["вид"]
        replies = d["репліки"]
        key_fact = d.get("ключовий_факт")

        print(f"\n[Діалог {did}] ({dtype}) — {len(replies)} реплік:")

        c3_history = []

        for turn_idx, turn_text in enumerate(replies):
            turn_id = f"{did}-turn-{turn_idx + 1}"
            is_final_turn = (turn_idx == len(replies) - 1)
            target_fact = key_fact if is_final_turn else None

            print(f"  Репліка {turn_idx + 1}: {turn_text[:60]}...")

            # Config 1
            e1 = get_existing("Config 1 (Fuzzy)", turn_id)
            if not e1:
                c1_turn = call_config_1_fuzzy(turn_text)
                e1 = evaluate_response("Config 1 (Fuzzy)", turn_id, dtype, turn_text, c1_turn, key_fact=target_fact)
                results.append(e1)
                save_checkpoint()

            # Config 2
            e2 = get_existing("Config 2 (Structured Prompt)", turn_id)
            if not e2:
                c2_turn = call_config_2_structured(turn_text)
                e2 = evaluate_response("Config 2 (Structured Prompt)", turn_id, dtype, turn_text, c2_turn, key_fact=target_fact)
                results.append(e2)
                save_checkpoint()

            # Config 3
            e3 = get_existing("Config 3 (Context Engineering)", turn_id)
            if not e3:
                c3_turn = call_config_3_context_eng(turn_text, c3_history)
                e3 = evaluate_response("Config 3 (Context Engineering)", turn_id, dtype, turn_text, c3_turn, key_fact=target_fact)
                results.append(e3)
                save_checkpoint()
            else:
                c3_turn = {"raw": e3.get("raw_response", "")}

            # Додаємо репліку до історії Config 3
            c3_reply = (e3.get("parsed") or {}).get("reply", c3_turn.get("raw", ""))
            c3_history.append({"role": "user", "content": turn_text})
            c3_history.append({"role": "assistant", "content": c3_reply})

    save_checkpoint()
    print(f"\n[OK] Усі результати успішно збережено у {out_file}")

    # Зведений підсумок
    print("\n" + "=" * 80)
    print("ЗВЕДЕНІ ПІДСУМКИ ТЕСТУВАННЯ")
    print("=" * 80)
    configs = ["Config 1 (Fuzzy)", "Config 2 (Structured Prompt)", "Config 3 (Context Engineering)"]
    for cfg in configs:
        cfg_tests = [r for r in results if r["config"] == cfg]
        total = len(cfg_tests)
        valid_schema = sum(1 for r in cfg_tests if r.get("schema_valid"))
        errors = [r.get("error_type") for r in cfg_tests if r.get("error_type") != "None"]
        avg_tokens = sum(r.get("total_tokens", 0) for r in cfg_tests) / total if total else 0
        avg_latency = sum(r.get("latency_ms", 0) for r in cfg_tests) / total if total else 0

        dialogue_finals = [r for r in cfg_tests if "turn-3" in r.get("test_id", "")]
        remembered = sum(1 for r in dialogue_finals if r.get("remembered_previous_fact") is True)

        print(f"\n{cfg}:")
        print(f"  Всього тестів: {total}")
        print(f"  Валідна JSON-схема: {valid_schema}/{total} ({valid_schema/total*100:.1f}%)")
        print(f"  Пам'ять ключового факту в діалозі: {remembered}/{len(dialogue_finals)}")
        print(f"  Кількість помилок: {len(errors)} ({', '.join(set(errors)) if errors else 'немає'})")
        print(f"  Середня кількість токенів: {avg_tokens:.1f}")
        print(f"  Середній час відповіді: {avg_latency:.1f} ms")


if __name__ == "__main__":
    main()
