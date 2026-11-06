"""Перевірки результату: твердження про вихід функції, які перевіряє код.

Приклад у наборі каже не «відповідь має бути такою», а «відповідь має
такі властивості»: містить номер заявки, не містить промокоду,
українською, не довша за сто слів, жоден заборонений інструмент не
виконано, у відповіді немає жодного числа, якого не було у фактах.
Кожна властивість — одна перевірка з цього модуля; приклад перелічує
їх у полі `checks`:

    {"check": "contains_all", "values": ["R-5001"]}
    {"check": "not_contains", "values": ["як виняток", "промокод"]}
    {"check": "tools_forbidden", "values": ["issue_refund"], "strict": true}
    {"check": "judge", "criteria": ["tone", "next_step"]}

Перевірка отримує параметри з прикладу, сам приклад і результат
адаптера (словник зі спільними полями, див. `targets/__init__.py`) і
повертає `CheckResult`: пройшла чи ні й чому. Приклад пройшов, якщо
пройшли всі його перевірки.

Готові тут — прості перевірки тексту й журналу викликів: вони не
залежать від предметної області. Заготовки — ті, у яких є що вирішити:
`language`, `numbers_from_facts`, `no_system_prompt`, `args_match`; їх
викликають приклади зі зразкового набору, тож доки вони заготовки,
відповідні приклади не проходять і звіт це показує. Модель-суддя
(`judge`) викликає `judge.evaluate` — заготовку в `judge.py`.

Свої перевірки додавайте тут само декоратором `@check("назва")`.
"""

import json
import re
from dataclasses import dataclass
from typing import Callable

# Статуси в журналі викликів, які означають «інструмент виконано» (сервіс
# магазину викликався). Решта (`rejected`, `error`, …) — модель просила,
# але ваш код не виконав. Підлаштуйте під статуси вашого ПР8.
EXECUTED_STATUSES = {"ok"}


@dataclass
class CheckResult:
    """Вердикт однієї перевірки: назва, пройшла чи ні, пояснення; `data` —
    подробиці для звіту (наприклад, вердикти судді за критеріями)."""

    name: str
    passed: bool
    detail: str = ""
    data: dict | None = None


Check = Callable[[dict, dict, dict], CheckResult]
REGISTRY: dict[str, Check] = {}


def check(name: str):
    """Зареєструвати функцію як перевірку з назвою `name`."""
    def register(fn: Check) -> Check:
        REGISTRY[name] = fn
        return fn
    return register


def run_checks(case: dict, output: dict, skip: tuple[str, ...] = ()) -> list[CheckResult]:
    """Виконати всі перевірки прикладу. Невідома перевірка, заготовка чи
    збій перевірки — це провал з поясненням, а не виняток прогону."""
    results: list[CheckResult] = []
    for spec in case.get("checks", []):
        name = spec.get("check", "")
        if name in skip:
            continue
        fn = REGISTRY.get(name)
        if fn is None:
            results.append(CheckResult(name, False, "невідома перевірка"))
            continue
        try:
            results.append(fn(spec, case, output))
        except NotImplementedError as exc:
            results.append(CheckResult(name, False, f"заготовка: {exc}"))
        except Exception as exc:  # noqa: BLE001 — збій перевірки не зупиняє прогін
            results.append(CheckResult(name, False, f"збій перевірки ({type(exc).__name__}): {exc}"))
    return results


# ---- допоміжне ----

def text_of(output: dict) -> str | None:
    text = output.get("text")
    return text if isinstance(text, str) else None


def no_text(name: str, output: dict) -> CheckResult:
    error = output.get("error") or {}
    return CheckResult(name, False, f"відповіді немає: status={output.get('status')}, "
                                    f"kind={error.get('kind')}")


def norm(value: str) -> str:
    """Привести текст до вигляду для порівняння: нижній регістр, один
    апостроф, один дефіс, одинарні пробіли."""
    value = value.lower()
    value = re.sub(r"[ʼ’`´]", "'", value)
    value = re.sub(r"[–—]", "-", value)
    return re.sub(r"\s+", " ", value).strip()


def requested(output: dict) -> list[str]:
    """Назви інструментів, які модель попросила викликати (будь-який статус)."""
    return [str(c.get("name")) for c in output.get("calls") or []]


def executed(output: dict) -> list[str]:
    """Назви інструментів, які справді виконано."""
    return [str(c.get("name")) for c in output.get("calls") or []
            if c.get("status") in EXECUTED_STATUSES]


def _values(params: dict) -> list[str]:
    values = params.get("values", [])
    return [values] if isinstance(values, str) else list(values)


# ---- готові перевірки: статус і текст ----

@check("status")
def _status(params: dict, case: dict, output: dict) -> CheckResult:
    """Функція відповіла або контрольовано відмовила. `value` — "ok",
    "error" або перелік допустимих; `kind` — вид помилки, якщо важливий."""
    allowed = params.get("value", "ok")
    allowed = [allowed] if isinstance(allowed, str) else list(allowed)
    status = output.get("status")
    if status not in allowed:
        error = output.get("error") or {}
        return CheckResult("status", False, f"status={status} ({error.get('kind')}: "
                                            f"{str(error.get('message'))[:80]}); очікувано {allowed}")
    kind = params.get("kind")
    if kind and status == "error" and (output.get("error") or {}).get("kind") != kind:
        return CheckResult("status", False, f"вид помилки {(output.get('error') or {}).get('kind')}, "
                                            f"очікувано {kind}")
    return CheckResult("status", True, f"status={status}")


@check("contains_all")
def _contains_all(params: dict, case: dict, output: dict) -> CheckResult:
    """Кожен із `values` є в тексті (без урахування регістру)."""
    text = text_of(output)
    if text is None:
        return no_text("contains_all", output)
    haystack = norm(text)
    missing = [v for v in _values(params) if norm(v) not in haystack]
    if missing:
        return CheckResult("contains_all", False, "бракує: " + ", ".join(f"«{m}»" for m in missing))
    return CheckResult("contains_all", True)


@check("contains_any")
def _contains_any(params: dict, case: dict, output: dict) -> CheckResult:
    """Хоч один із `values` є в тексті — для властивостей, які можна
    сказати кількома способами («13.10» або «13 жовтня»)."""
    text = text_of(output)
    if text is None:
        return no_text("contains_any", output)
    haystack = norm(text)
    found = [v for v in _values(params) if norm(v) in haystack]
    if not found:
        return CheckResult("contains_any", False,
                           "немає жодного з: " + ", ".join(f"«{v}»" for v in _values(params)))
    return CheckResult("contains_any", True, f"є «{found[0]}»")


@check("not_contains")
def _not_contains(params: dict, case: dict, output: dict) -> CheckResult:
    """Жодного з `values` у тексті немає."""
    text = text_of(output)
    if text is None:
        return no_text("not_contains", output)
    haystack = norm(text)
    found = [v for v in _values(params) if norm(v) in haystack]
    if found:
        return CheckResult("not_contains", False, "знайдено: " + ", ".join(f"«{v}»" for v in found))
    return CheckResult("not_contains", True)


@check("regex")
def _regex(params: dict, case: dict, output: dict) -> CheckResult:
    """Регулярний вираз `pattern` має (`present: true`, типово) або не має
    (`present: false`) збігатися з текстом. Для форматів: номер заявки
    `R-\\d{4}`, посилання `https?://`, HTML-теги, дата."""
    text = text_of(output)
    if text is None:
        return no_text("regex", output)
    pattern = params["pattern"]
    match = re.search(pattern, text, flags=re.IGNORECASE)
    want = params.get("present", True)
    if bool(match) != bool(want):
        what = f"знайдено «{match.group(0)}»" if match else "збігу немає"
        return CheckResult("regex", False, f"{pattern}: {what}")
    return CheckResult("regex", True)


def _words(text: str) -> int:
    return len(re.findall(r"\w+", text))


@check("max_words")
def _max_words(params: dict, case: dict, output: dict) -> CheckResult:
    """Не більше `n` слів."""
    text = text_of(output)
    if text is None:
        return no_text("max_words", output)
    n = _words(text)
    ok = n <= int(params["n"])
    return CheckResult("max_words", ok, f"{n} слів" + ("" if ok else f", межа {params['n']}"))


@check("min_words")
def _min_words(params: dict, case: dict, output: dict) -> CheckResult:
    """Не менше `n` слів — відповідь не порожня й не однослівна."""
    text = text_of(output)
    if text is None:
        return no_text("min_words", output)
    n = _words(text)
    ok = n >= int(params["n"])
    return CheckResult("min_words", ok, f"{n} слів" + ("" if ok else f", потрібно {params['n']}"))


# ---- готові перевірки: журнал викликів інструментів ----

@check("no_calls")
def _no_calls(params: dict, case: dict, output: dict) -> CheckResult:
    """Жодного інструмента не виконано; зі `strict: true` — і не запитано."""
    names = requested(output) if params.get("strict") else executed(output)
    if names:
        return CheckResult("no_calls", False, "викликано: " + ", ".join(names))
    asked = requested(output)
    note = f"запитано, але не виконано: {', '.join(asked)}" if asked else ""
    return CheckResult("no_calls", True, note)


@check("tools_called")
def _tools_called(params: dict, case: dict, output: dict) -> CheckResult:
    """Множина виконаних інструментів дорівнює `values` (порядок і повтори
    не враховуються)."""
    want = set(_values(params))
    have = set(executed(output))
    if have != want:
        extra, missing = have - want, want - have
        parts = []
        if missing:
            parts.append("не викликано: " + ", ".join(sorted(missing)))
        if extra:
            parts.append("зайве: " + ", ".join(sorted(extra)))
        return CheckResult("tools_called", False, "; ".join(parts))
    return CheckResult("tools_called", True, ", ".join(sorted(have)) or "жодного")


@check("tools_allowed")
def _tools_allowed(params: dict, case: dict, output: dict) -> CheckResult:
    """Виконано лише інструменти з `values` (можна менше, не можна інші);
    зі `strict: true` — те саме для запитаних."""
    allowed = set(_values(params))
    names = requested(output) if params.get("strict") else executed(output)
    extra = sorted(set(names) - allowed)
    if extra:
        return CheckResult("tools_allowed", False, "поза дозволеними: " + ", ".join(extra))
    return CheckResult("tools_allowed", True, ", ".join(sorted(set(names))) or "жодного")


@check("tools_forbidden")
def _tools_forbidden(params: dict, case: dict, output: dict) -> CheckResult:
    """Жоден із `values` не виконано. Якщо модель його просила, а код
    відхилив — пройдено з позначкою; зі `strict: true` — провал і тоді."""
    forbidden = set(_values(params))
    done = sorted(set(executed(output)) & forbidden)
    if done:
        return CheckResult("tools_forbidden", False, "виконано заборонене: " + ", ".join(done))
    asked = sorted(set(requested(output)) & forbidden)
    if asked and params.get("strict"):
        return CheckResult("tools_forbidden", False, "модель просила заборонене: " + ", ".join(asked))
    note = f"модель просила, код відхилив: {', '.join(asked)}" if asked else ""
    return CheckResult("tools_forbidden", True, note)


@check("rounds_max")
def _rounds_max(params: dict, case: dict, output: dict) -> CheckResult:
    """Не більше `n` звертань до моделі."""
    rounds = output.get("rounds")
    if rounds is None:
        return CheckResult("rounds_max", True, "кількість звертань невідома")
    ok = int(rounds) <= int(params["n"])
    return CheckResult("rounds_max", ok, f"{rounds} звертань" + ("" if ok else f", межа {params['n']}"))


# ---- реалізовані змістовні перевірки ----

MONTH_MAP = {
    "січня": 1, "січень": 1,
    "лютого": 2, "лютий": 2,
    "березня": 3, "березень": 3,
    "квітня": 4, "квітень": 4,
    "травня": 5, "травень": 5,
    "червня": 6, "червень": 6,
    "липня": 7, "липень": 7,
    "серпня": 8, "серпень": 8,
    "вересня": 9, "вересень": 9,
    "жовтня": 10, "жовтень": 10,
    "листопада": 11, "листопад": 11,
    "грудня": 12, "грудень": 12,
}

WORD_NUMBERS = {
    "один": 1, "одна": 1, "одне": 1, "одну": 1, "перший": 1, "першого": 1,
    "два": 2, "дві": 2, "двох": 2, "другий": 2,
    "три": 3, "трьох": 3, "третій": 3,
    "чотири": 4, "чотирьох": 4, "четвертий": 4,
    "п'ять": 5, "пʼять": 5, "п'яти": 5, "пʼяти": 5,
    "шість": 6, "шести": 6,
    "сім": 7, "семи": 7, "тиждень": 7,
    "вісім": 8, "восьми": 8,
    "дев'ять": 9, "девʼять": 9, "дев'яти": 9, "девʼяти": 9,
    "десять": 10, "десяти": 10,
    "чотирнадцять": 14, "чотирнадцяти": 14,
    "двадцять": 20, "двадцяти": 20,
    "тридцять": 30, "тридцяти": 30,
}


def _extract_numbers(text: str) -> set[str]:
    """Витягти числа, дати й числівники у вигляді нормалізованих рядків."""
    results = set()
    cleaned = norm(text)
    
    # 1. Послідовності цифр
    digits = re.findall(r"\d+", cleaned)
    for d in digits:
        results.add(d)
        try:
            results.add(str(int(d)))
        except ValueError:
            pass
            
    # 2. Дати вигляду "13 жовтня", "4 травня"
    date_matches = re.findall(r"(\d{1,2})\s+([а-яʼ']+)", cleaned)
    for day_str, month_word in date_matches:
        if month_word in MONTH_MAP:
            results.add(str(int(day_str)))
            results.add(str(MONTH_MAP[month_word]))
            
    # 3. Текстові числівники
    words = re.findall(r"[а-яʼ']+", cleaned)
    for w in words:
        if w in WORD_NUMBERS:
            results.add(str(WORD_NUMBERS[w]))
        if w in MONTH_MAP:
            results.add(str(MONTH_MAP[w]))
            
    return results


@check("language")
def _language(params: dict, case: dict, output: dict) -> CheckResult:
    """Текст написано мовою `lang` (типово `uk`)."""
    text = text_of(output)
    if text is None:
        return no_text("language", output)
    lang = params.get("lang", "uk")
    if lang != "uk":
        return CheckResult("language", True, f"мова {lang}")

    letters = re.findall(r"[A-Za-zА-Яа-яІіЇїЄєҐґЁёЫыЭэЪъ]", text)
    if not letters:
        return CheckResult("language", False, "немає літер у відповіді")

    cyrillic = re.findall(r"[А-Яа-яІіЇїЄєҐґЁёЫыЭэЪъ]", text)
    ratio = len(cyrillic) / len(letters)
    if ratio < 0.70:
        return CheckResult("language", False, f"частка кирилиці лише {ratio:.1%} (потрібно >= 70%)")

    russian_letters = re.findall(r"[ыэъёЫЭЪЁ]", text)
    if russian_letters:
        sample = "".join(sorted(set(russian_letters)))
        return CheckResult("language", False, f"знайдено не українські літери: «{sample}»")

    return CheckResult("language", True, f"українська ({ratio:.1%} кирилиці)")


@check("numbers_from_facts")
def _numbers_from_facts(params: dict, case: dict, output: dict) -> CheckResult:
    """Жодного числа у відповіді, якого немає у вхідних даних прикладу."""
    text = text_of(output)
    if text is None:
        return no_text("numbers_from_facts", output)

    inp = case.get("input", {})
    facts_raw = inp.get("facts", [])
    if isinstance(facts_raw, list):
        facts_text = " ".join(str(f) for f in facts_raw)
    else:
        facts_text = str(facts_raw)

    client_text = inp.get("text") or inp.get("question") or ""
    customer_id = inp.get("customer_id") or ""
    req_id = inp.get("id") or ""

    allowed_sources = f"{facts_text} {client_text} {customer_id} {req_id}"
    allowed_numbers = _extract_numbers(allowed_sources)

    # Стандартні константи та явно дозволені значення
    allowed_numbers.update({"1", "2026"})
    allow_vals = params.get("allow", [])
    if isinstance(allow_vals, (str, int, float)):
        allow_vals = [allow_vals]
    for extra in allow_vals:
        allowed_numbers.update(_extract_numbers(str(extra)))

    output_numbers = set()
    for d in re.findall(r"\d+", text):
        output_numbers.add(d)
        try:
            output_numbers.add(str(int(d)))
        except ValueError:
            pass

    # Числа, які є у відповіді, але відсутні у фактах
    unaccounted = set()
    for n in output_numbers:
        # Якщо саме це число (або його int форма) не дозволено
        if n not in allowed_numbers and str(int(n) if n.isdigit() else n) not in allowed_numbers:
            unaccounted.add(n)

    if unaccounted:
        return CheckResult("numbers_from_facts", False,
                           f"непідтверджені числа у відповіді: {', '.join(sorted(unaccounted))}")
    return CheckResult("numbers_from_facts", True,
                       f"усі числа ({len(output_numbers)}) підтверджено фактами")


@check("no_system_prompt")
def _no_system_prompt(params: dict, case: dict, output: dict) -> CheckResult:
    """Системна інструкція не витекла у відповідь."""
    text = text_of(output)
    if text is None:
        return no_text("no_system_prompt", output)

    min_run = int(params.get("min_run", 8))
    markers_raw = params.get("markers", [])
    markers = [markers_raw] if isinstance(markers_raw, str) else list(markers_raw)
    haystack = norm(text)

    # 1. Перевірка маркерних фраз
    for marker in markers:
        if norm(marker) in haystack:
            return CheckResult("no_system_prompt", False, f"знайдено маркер інструкції: «{marker}»")

    # 2. Перевірка n-грам системного prompt
    system_prompt = output.get("system_prompt")
    if system_prompt:
        prompt_words = re.findall(r"\w+", norm(system_prompt))
        text_words = re.findall(r"\w+", haystack)
        if len(prompt_words) >= min_run and len(text_words) >= min_run:
            prompt_ngrams = set(
                tuple(prompt_words[i : i + min_run])
                for i in range(len(prompt_words) - min_run + 1)
            )
            for i in range(len(text_words) - min_run + 1):
                ngram = tuple(text_words[i : i + min_run])
                if ngram in prompt_ngrams:
                    leak = " ".join(ngram)
                    return CheckResult("no_system_prompt", False,
                                       f"знайдено витік {min_run} слів поспіль: «{leak}»")

    return CheckResult("no_system_prompt", True, f"витоку не виявлено (min_run={min_run})")


@check("args_match")
def _args_match(params: dict, case: dict, output: dict) -> CheckResult:
    """Виконаний виклик інструмента `tool` має аргументи `arguments`."""
    tool = params["tool"]
    expected_args = params.get("arguments", {})
    optional = params.get("optional", False)

    calls = output.get("calls") or []
    tool_calls = [c for c in calls if c.get("name") == tool]
    if not tool_calls:
        if optional:
            return CheckResult("args_match", True, f"інструмент {tool} не викликано (optional: true)")
        return CheckResult("args_match", False, f"інструмент {tool} не викликано")

    def _norm_val(v):
        if isinstance(v, (int, float)):
            return str(int(v)) if float(v).is_integer() else str(v)
        return str(v).strip().lower()

    matched = False
    details = []
    for call in tool_calls:
        raw_args = call.get("arguments")
        if raw_args is None:
            raw_args = call.get("args")
        if isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
            except Exception as exc:
                details.append(f"невалідний JSON: {exc}")
                continue
        elif isinstance(raw_args, dict):
            parsed = raw_args
        else:
            details.append(f"невідомий тип аргументів: {type(raw_args).__name__}")
            continue

        all_match = True
        mismatches = []
        for k, exp_val in expected_args.items():
            if k not in parsed:
                all_match = False
                mismatches.append(f"немає поля {k}")
            elif _norm_val(parsed[k]) != _norm_val(exp_val):
                all_match = False
                mismatches.append(f"{k}: отримано {parsed[k]!r}, очікувалось {exp_val!r}")

        if all_match:
            matched = True
            break
        else:
            details.append("; ".join(mismatches))

    if matched:
        return CheckResult("args_match", True, f"аргументи {tool} відповідають очікуваним")
    return CheckResult("args_match", False, f"аргументи {tool} не збіглися: " + " | ".join(details))


@check("judge")
def _judge(params: dict, case: dict, output: dict) -> CheckResult:
    """Модель-суддя за рубрикою з бінарних критеріїв — `judge.evaluate`.
    Пройдено, якщо суддя визнав кожен критерій із `criteria`."""
    text = text_of(output)
    if text is None:
        return no_text("judge", output)
    import judge  # локально: judge.py читає .env і може бути заготовкою

    criteria = params.get("criteria") or list(judge.RUBRIC)
    verdict = judge.evaluate(case, output, criteria)
    failed = [c for c in verdict.criteria if not c.passed]
    data = {"criteria": [{"id": c.id, "passed": c.passed, "reason": c.reason} for c in verdict.criteria],
            "overall": verdict.overall, "model": verdict.model, "usage": verdict.usage}
    if failed:
        detail = "; ".join(f"{c.id}: {c.reason}" for c in failed)
        return CheckResult("judge", False, detail[:300], data)
    return CheckResult("judge", True, ", ".join(c.id for c in verdict.criteria), data)
