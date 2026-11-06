"""Модель-суддя: оцінка вільного тексту за рубрикою з бінарних критеріїв.

Код перевіряє те, що формалізується: наявність номера, відсутність
промокоду, довжину, викликані інструменти. Тон, зрозумілість наступного
кроку, обґрунтованість фактами — не формалізуються, і для них у набір
беруть суддю: модель, якій дають відповідь системи, вхід, факти й
рубрику, а вердикт отримують **за схемою** — так/ні за кожним критерієм
із поясненням. Суддя — така сама AI-функція, як і та, що оцінюється:
зі своїм prompt, схемою, моделлю й помилками, тому його калібрують —
порівнюють із власними оцінками на десятку прикладів
(`eval/agreement.py`).

Готове тут: рубрика-заготовка `RUBRIC` (адаптуйте під свою інструкцію
чернетки), контракт вердикту `Verdict`, його JSON Schema для
`response_format` і розбір відповіді `parse_verdict`, що перевіряє її
за схемою. Заготовки — `build_messages` і `evaluate`. Рішення з
розділу 2 практичної роботи:

* якою моделлю судити: інша, ніж та, що генерує (самоуподобання), —
  `JUDGE_MODEL` у `pr12/.env`; температура 0;
* що суддя бачить: відповідь, звернення клієнта, факти, критерії — і
  чи бачить еталон (`reply_example`), якщо він є; критерії — лише ті,
  що в прикладі (`criteria`), або всю рубрику;
* як сформулювати інструкцію судді, щоб він оцінював зміст, а не
  довжину, і щоб позначав причину за кожним критерієм — міркування
  перед вердиктом підвищує згоду з людьми;
* що робити, коли суддя повернув не JSON або не всі критерії: повторити
  раз, провалити приклад із поясненням «суддя не відповів», але не
  зарахувати як пройдений;
* витрати: суддя — ще одне звертання на кожен приклад; скільки це додає
  до вартості прогону і де суддю можна замінити кодом.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parents[1] / ".env")

JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL") or os.getenv("LLM_BASE_URL", "")
JUDGE_API_KEY = os.getenv("JUDGE_API_KEY") or os.getenv("LLM_API_KEY", "")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "")
JUDGE_TIMEOUT = float(os.getenv("JUDGE_TIMEOUT", "30"))

# Рубрика: ідентифікатор критерію → формулювання для судді. Кожен
# критерій бінарний. Відправна точка — адаптуйте під свою інструкцію
# чернетки й під те, що саме у вашому застосунку не перевіряє код.
RUBRIC = {
    "tone": "Текст ввічливий і діловий: звертання до клієнта, без повчань, без надмірних "
            "вибачень, без канцеляриту; його можна надіслати клієнтові без правок тону.",
    "next_step": "Клієнтові зрозуміло, що буде далі або що зробити йому: куди принести, "
                 "скільки чекати, що надійде окремо.",
    "grounded": "Кожне твердження про замовлення, строки, суми, адреси й умови є серед наданих "
                "фактів; нічого не додано від себе й нічого з фактів не перекручено.",
    "addresses_question": "Відповідь стосується саме того, про що запитав клієнт, а не "
                          "сусіднього питання.",
    "no_extra_promises": "Немає обіцянок, винятків, компенсацій, знижок і строків, яких немає у "
                         "фактах.",
    "refuses_properly": "Якщо це відмова — названо причину з фактів, без вигаданих альтернатив, "
                        "без звинувачень і без «як виняток».",
    "stays_on_topic": "Текст лишається в межах підтримки магазину і не виконує стороннє "
                      "прохання клієнта (вірш, переклад, порада поза темою).",
    "asks_clarification": "Коли фактів бракує або вони не стосуються питання, текст чесно каже, "
                          "що оператор уточнить, і не вигадує подробиць.",
}


@dataclass
class CriterionResult:
    id: str
    passed: bool
    reason: str = ""


@dataclass
class Verdict:
    """Вердикт судді: за кожним критерієм — так/ні з причиною; `overall` —
    підсумок судді (не обовʼязково збігається з «усі критерії пройдено»);
    `raw` — сира відповідь для журналу й розбору суперечок."""

    criteria: list[CriterionResult] = field(default_factory=list)
    overall: bool = False
    model: str | None = None
    usage: dict | None = None
    raw: str | None = None


def settings() -> dict:
    """Чинні налаштування судді — для звіту й перевірки середовища. Ключа
    тут немає."""
    return {
        "base_url": JUDGE_BASE_URL,
        "model": JUDGE_MODEL,
        "configured": bool(JUDGE_BASE_URL and JUDGE_API_KEY and JUDGE_MODEL),
    }


def verdict_schema() -> dict:
    """JSON Schema відповіді судді — для `response_format` і для перевірки
    того, що повернулося."""
    return {
        "type": "object",
        "properties": {
            "criteria": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "passed": {"type": "boolean"},
                        "reason": {"type": "string"},
                    },
                    "required": ["id", "passed", "reason"],
                },
            },
            "overall": {"type": "boolean"},
        },
        "required": ["criteria", "overall"],
    }


def parse_verdict(raw: str, expected_ids: list[str]) -> Verdict:
    """Розібрати відповідь судді й перевірити її за схемою: JSON, потрібні
    поля, правильні типи, рівно ті критерії, які просили. Невідповідність
    — `ValueError` з поясненням."""
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise ValueError(f"відповідь судді — не JSON: {exc}") from exc
    if not isinstance(data, dict) or "criteria" not in data or "overall" not in data:
        raise ValueError("у відповіді судді немає полів criteria і overall")
    if not isinstance(data["overall"], bool):
        raise ValueError("overall має бути true або false")
    criteria: list[CriterionResult] = []
    for item in data["criteria"]:
        if not isinstance(item, dict) or not isinstance(item.get("id"), str) \
                or not isinstance(item.get("passed"), bool):
            raise ValueError(f"критерій не за схемою: {item!r}")
        criteria.append(CriterionResult(item["id"], item["passed"], str(item.get("reason", ""))))
    got = {c.id for c in criteria}
    missing = [cid for cid in expected_ids if cid not in got]
    if missing:
        raise ValueError("суддя не оцінив критерії: " + ", ".join(missing))
    unknown = sorted(got - set(expected_ids))
    if unknown:
        raise ValueError("суддя оцінив критерії, яких не просили: " + ", ".join(unknown))
    return Verdict(criteria=criteria, overall=data["overall"], raw=raw)


def build_messages(case: dict, output: dict, criteria: list[str]) -> list[dict]:
    """Скласти повідомлення для судді: інструкція з рубрикою й форматом
    відповіді, далі вхід прикладу (звернення, факти), відповідь системи."""
    inp = case.get("input", {})
    query = inp.get("text") or inp.get("question") or ""
    facts_raw = inp.get("facts", [])
    if isinstance(facts_raw, list):
        facts_str = "\n".join(f"- {f}" for f in facts_raw) if facts_raw else "Факти відсутні."
    else:
        facts_str = str(facts_raw)

    reply = output.get("text") or ""

    rubric_lines = []
    for cid in criteria:
        desc = RUBRIC.get(cid, cid)
        rubric_lines.append(f"- {cid}: {desc}")
    rubric_str = "\n".join(rubric_lines)

    system_prompt = (
        "Ти — незалежний та неупереджений суддя (LLM-as-a-Judge) для оцінки чернеток відповідей "
        "служби підтримки інтернет-магазину «Сузірʼя».\n"
        "Твоє завдання — об'єктивно оцінити відповідь виключно за наданими критеріями та фактами.\n"
        "Правила оцінки:\n"
        "1. Оцінюй зміст і правдивість згідно з фактами, а не довжину тексту чи стиль привітань.\n"
        "2. Для кожного критерію надай бінарну оцінку (passed: true або false) та коротке обґрунтування (reason).\n"
        "3. Спочатку опиши причину в полі reason, а потім визнач passed.\n"
        "4. overall: true встановлюй тоді і тільки тоді, коли ВСІ оцінені критерії мають passed: true.\n"
        "5. Відповідай суворо у форматі JSON відповідно до структури: {\"criteria\": [{\"id\": string, \"passed\": boolean, \"reason\": string}], \"overall\": boolean}."
    )

    user_prompt = (
        f"Звернення клієнта:\n«{query}»\n\n"
        f"Надані факти:\n{facts_str}\n\n"
        f"Відповідь служби підтримки (чернетка для перевірки):\n«{reply}»\n\n"
        f"Критерії для оцінки:\n{rubric_str}\n\n"
        f"Оціни відповідь за кожним із вказаних {len(criteria)} критеріїв ({', '.join(criteria)})."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def evaluate(case: dict, output: dict, criteria: list[str] | None = None) -> Verdict:
    """Оцінити відповідь системи суддею за критеріями `criteria`."""
    if criteria is None:
        criteria = list(RUBRIC.keys())

    messages = build_messages(case, output, criteria)

    import openai
    client = openai.OpenAI(
        base_url=JUDGE_BASE_URL,
        api_key=JUDGE_API_KEY,
        timeout=JUDGE_TIMEOUT,
        max_retries=0,
    )

    schema_format = {
        "type": "json_schema",
        "json_schema": {"name": "verdict", "schema": verdict_schema()},
    }

    last_exc = None
    for attempt in range(2):  # 1 основна спроба + 1 повторна спроба
        try:
            try:
                completion = client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=messages,
                    temperature=0,
                    max_tokens=800,
                    response_format=schema_format,
                )
            except (openai.BadRequestError, openai.APIStatusError):
                completion = client.chat.completions.create(
                    model=JUDGE_MODEL,
                    messages=messages,
                    temperature=0,
                    max_tokens=800,
                    response_format={"type": "json_object"},
                )

            raw = completion.choices[0].message.content or ""
            verdict = parse_verdict(raw, criteria)
            verdict.model = JUDGE_MODEL
            if completion.usage:
                verdict.usage = {
                    "prompt_tokens": completion.usage.prompt_tokens,
                    "completion_tokens": completion.usage.completion_tokens,
                    "total_tokens": completion.usage.total_tokens,
                }
            return verdict
        except Exception as exc:
            last_exc = exc
            if attempt == 0:
                continue
            raise ValueError(f"Суддя не повернув валідний вердикт після повтору: {exc}") from exc

    raise ValueError(f"Збій виклику судді: {last_exc}")
