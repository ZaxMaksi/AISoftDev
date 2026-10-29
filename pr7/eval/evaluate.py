"""Скрипт оцінювання якості вилучення даних та перевірки програмних правил.

Проганяє 27 зразків (clean та degraded) та власні зразки (own),
порівнює вилучені поля з expected.json після нормалізації,
фіксує категорії: правильно / помилка / пропуск / вигадка,
аналізує критичні поля, матрицю рішень та підраховує тихі помилки.
"""

import argparse
import json
import os
import sys
import time
import unicodedata
import re
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

# Додаємо pr7 до sys.path для імпорту app
PR7_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PR7_DIR))

from app import extraction
from app.extraction import Result
from app.rules import Issue

SAMPLES_DIR = PR7_DIR / "samples"
EXPECTED_FILE = SAMPLES_DIR / "expected.json"
OWN_EXPECTED_FILE = SAMPLES_DIR / "own_expected.json"
EVAL_DIR = PR7_DIR / "eval"


# ---------------------------------------------------------------------------
# Функції нормалізації значень для коректного порівняння
# ---------------------------------------------------------------------------

def normalize_text(val) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    s = unicodedata.normalize("NFC", s)
    # Єдиний вигляд апострофа
    s = re.sub(r"[\u2018\u2019\u02BC\u0027\u0060]", "\u2019", s)
    # Зведення зайвих пробілів
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def normalize_code(val) -> str:
    if val is None:
        return ""
    return re.sub(r"\D", "", str(val))


def normalize_iban(val) -> str:
    if val is None:
        return ""
    return re.sub(r"\s+", "", str(val)).upper()


def normalize_decimal(val) -> str | None:
    if val is None:
        return None
    s = str(val).replace(",", ".").replace(" ", "").replace("\xa0", "")
    try:
        d = Decimal(s)
        return f"{d:.2f}"
    except (InvalidOperation, ValueError):
        return str(val).strip()


def normalize_field(path: str, val):
    """Нормалізувати значення залежно від призначення поля."""
    if val is None:
        return None

    p_lower = path.lower()
    if "iban" in p_lower:
        return normalize_iban(val)
    if "code" in p_lower:
        return normalize_code(val)
    if any(k in p_lower for k in ("price", "amount", "total", "vat")):
        norm = normalize_decimal(val)
        return norm if norm is not None else normalize_text(val)
    if "quantity" in p_lower:
        try:
            return f"{Decimal(str(val)):g}"
        except Exception:
            return normalize_text(val)
    if "date" in p_lower or "valid_until" in p_lower:
        return normalize_text(val)

    return normalize_text(val).lower()


# ---------------------------------------------------------------------------
# Розгортання вкладеного словника у плоский список полів
# ---------------------------------------------------------------------------

def flatten_dict(d: dict | None, prefix: str = "") -> dict:
    if not d or not isinstance(d, dict):
        return {}
    res = {}
    for k, v in d.items():
        curr_path = f"{prefix}.{k}" if prefix else k
        if isinstance(v, dict):
            res.update(flatten_dict(v, curr_path))
        elif isinstance(v, list):
            for idx, item in enumerate(v):
                if isinstance(item, dict):
                    res.update(flatten_dict(item, f"{curr_path}.{idx}"))
                else:
                    res[f"{curr_path}.{idx}"] = item
        else:
            res[curr_path] = v
    return res


# ---------------------------------------------------------------------------
# Порівняння вилученого документа з еталоном
# ---------------------------------------------------------------------------

def compare_document(expected_fields: dict, actual_doc: dict | None):
    """Порівняти документ поле за полем.
    
    Повертає:
      field_results: dict[path -> {'status': ..., 'expected': ..., 'actual': ...}]
      counts: {'correct': int, 'error': int, 'miss': int, 'hallucination': int}
    """
    flat_exp = flatten_dict(expected_fields)
    flat_act = flatten_dict(actual_doc) if actual_doc else {}

    all_keys = sorted(set(flat_exp.keys()) | set(flat_act.keys()))
    results = {}
    counts = {"correct": 0, "error": 0, "miss": 0, "hallucination": 0}

    for path in all_keys:
        exp_raw = flat_exp.get(path)
        act_raw = flat_act.get(path)

        norm_exp = normalize_field(path, exp_raw)
        norm_act = normalize_field(path, act_raw)

        if norm_exp is None and norm_act is None:
            # Обидва відсутні (наприклад, valid_until == null) — правильне визначення відсутності
            status = "correct"
            counts["correct"] += 1
        elif norm_exp is not None and norm_act is None:
            status = "miss"  # Пропуск: значення було, модель повернула null / не вилучила
            counts["miss"] += 1
        elif norm_exp is None and norm_act is not None:
            status = "hallucination"  # Вигадка: значення не було, модель підставила своє
            counts["hallucination"] += 1
        else:
            if norm_exp == norm_act:
                status = "correct"
                counts["correct"] += 1
            else:
                status = "error"  # Помилка: отримано інше значення
                counts["error"] += 1

        results[path] = {
            "status": status,
            "expected": exp_raw,
            "actual": act_raw,
            "norm_expected": norm_exp,
            "norm_actual": norm_act,
        }

    return results, counts


# ---------------------------------------------------------------------------
# Головна логіка запуску та кешування
# ---------------------------------------------------------------------------

def run_evaluation(max_side: int, cache_file: Path, force: bool = False):
    # 1. Завантажуємо очікувані дані
    expected_data = json.loads(EXPECTED_FILE.read_text(encoding="utf-8"))
    if OWN_EXPECTED_FILE.exists():
        own_data = json.loads(OWN_EXPECTED_FILE.read_text(encoding="utf-8"))
        expected_data.update(own_data)

    # 2. Перевіряємо наявний кеш
    cache = {}
    if cache_file.exists() and not force:
        try:
            cache = json.loads(cache_file.read_text(encoding="utf-8"))
            print(f"Завантажено {len(cache)} результатів із кешу {cache_file.name}")
        except Exception as e:
            print(f"Не вдалося прочитати кеш: {e}, починаємо наново.")

    results_by_sample = {}

    for sample_key, sample_info in expected_data.items():
        sample_path = SAMPLES_DIR / sample_key
        if not sample_path.exists():
            print(f"[!] Файл не знайдено: {sample_path}")
            continue

        print(f"Обробка {sample_key} (max_side={max_side})...", end=" ", flush=True)

        if sample_key in cache and not force:
            raw_res = cache[sample_key]
            print("[КЕШ]")
        else:
            content = sample_path.read_bytes()
            t0 = time.time()
            res = extraction.process(content, max_side=max_side)
            t_total = time.time() - t0

            raw_res = {
                "decision": res.decision,
                "reasons": res.reasons,
                "document": res.document,
                "issues": [asdict(i) for i in res.issues],
                "image": res.image,
                "model": res.model,
                "elapsed": res.elapsed,
                "usage": res.usage,
                "process_total_time": t_total,
            }
            cache[sample_key] = raw_res
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"[{raw_res['decision']}] за {t_total:.2f}с")

        # Порівнюємо
        exp_fields = sample_info.get("fields", {})
        act_doc = raw_res.get("document")
        field_eval, counts = compare_document(exp_fields, act_doc)

        # Критичні поля: supplier.iban, supplier.code, total
        critical_correct = 0
        critical_total = 0
        for cp in ("supplier.iban", "supplier.code", "total"):
            if cp in field_eval:
                critical_total += 1
                if field_eval[cp]["status"] == "correct":
                    critical_correct += 1

        # Перевірка спрацювання правил на помилках
        issues = raw_res.get("issues", [])
        issue_rules = [i["rule"] for i in issues]
        issue_fields = [i["field"] for i in issues]

        # Підрахунок тихих помилок:
        # Неправильні значення (error / hallucination), які пройшли в auto
        is_auto = raw_res["decision"] == "auto"
        silent_errors = 0
        if is_auto:
            silent_errors = counts["error"] + counts["hallucination"]

        decision_matched = raw_res["decision"] == sample_info.get("expected_decision")

        results_by_sample[sample_key] = {
            "source": sample_info.get("source"),
            "variant": sample_info.get("variant"),
            "expected_decision": sample_info.get("expected_decision"),
            "actual_decision": raw_res["decision"],
            "decision_matched": decision_matched,
            "counts": counts,
            "critical_correct": critical_correct,
            "critical_total": critical_total,
            "silent_errors": silent_errors,
            "tokens": (raw_res.get("usage") or {}).get("total_tokens", 0),
            "prompt_tokens": (raw_res.get("usage") or {}).get("prompt_tokens", 0),
            "elapsed_extraction": (raw_res.get("elapsed") or {}).get("extraction", 0),
            "reasons": raw_res.get("reasons", []),
            "issues": issues,
            "field_eval": field_eval,
            "note": sample_info.get("note", ""),
        }

    return results_by_sample


# ---------------------------------------------------------------------------
# Агрегація та генерація звіту
# ---------------------------------------------------------------------------

def aggregate_and_print(results: dict, title: str) -> str:
    # Групування за варіантами: clean, blur, lowres, jpeg, faded, photo, crop, own*
    variants = {}
    total_counts = {"correct": 0, "error": 0, "miss": 0, "hallucination": 0}
    total_files = 0
    total_silent = 0
    total_decisions_matched = 0
    total_tokens = 0
    total_prompt_tokens = 0
    total_critical_correct = 0
    total_critical_all = 0

    confusion = {
        "auto_expected": {"auto": 0, "review": 0, "reject": 0},
        "review_expected": {"auto": 0, "review": 0, "reject": 0},
        "reject_expected": {"auto": 0, "review": 0, "reject": 0},
    }

    for key, data in results.items():
        v = data["variant"]
        if v not in variants:
            variants[v] = {
                "files": 0,
                "correct": 0,
                "error": 0,
                "miss": 0,
                "hallucination": 0,
                "silent": 0,
                "decision_matched": 0,
                "tokens": 0,
                "prompt_tokens": 0,
                "critical_correct": 0,
                "critical_total": 0,
            }
        grp = variants[v]
        grp["files"] += 1
        c = data["counts"]
        grp["correct"] += c["correct"]
        grp["error"] += c["error"]
        grp["miss"] += c["miss"]
        grp["hallucination"] += c["hallucination"]
        grp["silent"] += data["silent_errors"]
        if data["decision_matched"]:
            grp["decision_matched"] += 1
        grp["tokens"] += data["tokens"]
        grp["prompt_tokens"] += data["prompt_tokens"]
        grp["critical_correct"] += data["critical_correct"]
        grp["critical_total"] += data["critical_total"]

        total_files += 1
        total_counts["correct"] += c["correct"]
        total_counts["error"] += c["error"]
        total_counts["miss"] += c["miss"]
        total_counts["hallucination"] += c["hallucination"]
        total_silent += data["silent_errors"]
        if data["decision_matched"]:
            total_decisions_matched += 1
        total_tokens += data["tokens"]
        total_prompt_tokens += data["prompt_tokens"]
        total_critical_correct += data["critical_correct"]
        total_critical_all += data["critical_total"]

        # Матриця рішень
        exp_d = data["expected_decision"]
        act_d = data["actual_decision"]
        matrix_row = f"{exp_d}_expected"
        if matrix_row in confusion and act_d in confusion[matrix_row]:
            confusion[matrix_row][act_d] += 1

    lines = [f"### Результати оцінювання: {title}\n"]
    lines.append("| Варіант | Файлів | Правильно | Помилок | Пропусків | Вигадок | Спіймано правилами | Тихих помилок | Рішення збіглося | Токени запиту |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")

    for v, g in sorted(variants.items()):
        total_errs = g["error"] + g["hallucination"]
        caught = total_errs - g["silent"]
        prompt_avg = f"{g['prompt_tokens'] // g['files']}" if g["files"] else "0"
        lines.append(
            f"| `{v}` | {g['files']} | {g['correct']} | {g['error']} | {g['miss']} | {g['hallucination']} | "
            f"{caught} | **{g['silent']}** | {g['decision_matched']}/{g['files']} | {prompt_avg} |"
        )

    all_errs = total_counts["error"] + total_counts["hallucination"]
    all_caught = all_errs - total_silent
    all_prompt_avg = f"{total_prompt_tokens // total_files}" if total_files else "0"
    lines.append(
        f"| **РАЗОМ** | **{total_files}** | **{total_counts['correct']}** | **{total_counts['error']}** | "
        f"**{total_counts['miss']}** | **{total_counts['hallucination']}** | **{all_caught}** | "
        f"**{total_silent}** | **{total_decisions_matched}/{total_files}** | **{all_prompt_avg}** |"
    )

    lines.append("\n#### Критичні поля (IBAN, ЄДРПОУ/РНОКПП, Разом до сплати)")
    crit_pct = (total_critical_correct / total_critical_all * 100) if total_critical_all else 0
    lines.append(f"- Правильно вилучено: **{total_critical_correct} / {total_critical_all}** ({crit_pct:.1f}%)\n")

    lines.append("#### Матриця рішень (Очікуване vs Фактичне)")
    lines.append("| Очікуване рішення | Отримано AUTO | Отримано REVIEW | Отримано REJECT |")
    lines.append("|---|---|---|---|")
    lines.append(f"| **auto** | {confusion['auto_expected']['auto']} | {confusion['auto_expected']['review']} (зайвий ручний перегляд) | {confusion['auto_expected']['reject']} |")
    lines.append(f"| **review** | **{confusion['review_expected']['auto']}** (критичний пропуск!) | {confusion['review_expected']['review']} | {confusion['review_expected']['reject']} |")
    lines.append(f"| **reject** | **{confusion['reject_expected']['auto']}** (неприпустимо!) | {confusion['reject_expected']['review']} | {confusion['reject_expected']['reject']} |")

    report_str = "\n".join(lines)
    print(report_str)
    return report_str


def main():
    parser = argparse.ArgumentParser(description="Оцінювання конвеєра на зразках.")
    parser.add_argument("--max-side", type=int, default=1600, help="Максимальна сторона зображення (за замовчуванням 1600)")
    parser.add_argument("--compare", action="store_true", help="Запустити порівняння (1600 проти 800)")
    parser.add_argument("--force", action="store_true", help="Ігнорувати кеш і викликати модель наново")
    args = parser.parse_args()

    if args.compare:
        print("\n=== ПРОГІН 1: IMAGE_MAX_SIDE = 1600 ===")
        cache_1600 = EVAL_DIR / "cache_1600.json"
        res_1600 = run_evaluation(1600, cache_1600, force=args.force)
        rep_1600 = aggregate_and_print(res_1600, "Базова конфігурація (IMAGE_MAX_SIDE = 1600)")

        print("\n=== ПРОГІН 2: IMAGE_MAX_SIDE = 800 ===")
        cache_800 = EVAL_DIR / "cache_800.json"
        res_800 = run_evaluation(800, cache_800, force=args.force)
        rep_800 = aggregate_and_print(res_800, "Порівняльна конфігурація (IMAGE_MAX_SIDE = 800)")

        return

    cache_file = EVAL_DIR / f"cache_{args.max_side}.json"
    res = run_evaluation(args.max_side, cache_file, force=args.force)
    aggregate_and_print(res, f"Конфігурація IMAGE_MAX_SIDE = {args.max_side}")


if __name__ == "__main__":
    main()
