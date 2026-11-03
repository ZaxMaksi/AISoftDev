"""Цикл помічника: від питання клієнта до відповіді, спертої на дані магазину.

Перенесено з ПР8 для порівняльного аналізу в ПР9.
Послідовність: спитати модель з описами інструментів (`llm.chat`),
виконати виклики, які вона запропонувала (`tools.call`), повернути їй результати
і спитати ще раз — доки вона не відповість текстом або доки не вичерпано обмеження.
"""

import json
import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

from . import llm, tools

load_dotenv()

MAX_ROUNDS = int(os.getenv("TOOL_MAX_ROUNDS", "3"))


@dataclass
class ToolTrace:
    """Запис про один виклик інструмента — для сторінки й для перевірки.

    `round` — на якому звертанні до моделі його запропоновано (з 1).
    `name` і `arguments` — як їх повернула модель, до будь-яких перевірок.
    `status` і `reason` — що з викликом зробив ваш код.
    `result` — що повернулося моделі.
    `elapsed` — час виконання, секунди.
    """

    round: int
    name: str
    arguments: str
    status: str
    reason: str | None = None
    result: dict | list | str | None = None
    elapsed: float | None = None


@dataclass
class Answer:
    """Результат циклу — те, з чим працює веб-рівень.

    `text` — відповідь клієнтові.
    `calls` — журнал викликів інструментів у порядку виконання.
    `rounds` — скільки разів звертались до моделі.
    `stopped` — чому цикл завершився: `answer` або `limit`.
    `model` — назва моделі.
    `elapsed` — час за видами: `{"model": ..., "tools": ...}`.
    `usage` — токени запиту й відповіді, сумарно за всі звертання.
    """

    text: str
    calls: list[ToolTrace] = field(default_factory=list)
    rounds: int = 0
    stopped: str = "answer"
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


def answer(question: str, customer_id: str) -> Answer:
    """Відповісти клієнтові, за потреби викликаючи інструменти.

    `customer_id` — клієнт, який увійшов; його передає веб-рівень із сесії,
    а не модель з аргументів.
    """
    max_rounds = int(os.getenv("TOOL_MAX_ROUNDS", "3"))
    if not question or not str(question).strip():
        return Answer(
            text="Будь ласка, введіть ваше запитання.",
            rounds=0,
            stopped="answer",
            elapsed={"model": 0.0, "tools": 0.0},
            usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        )

    ctx = tools.Context(customer_id=str(customer_id).strip())
    messages = llm.build_messages(question)
    tool_specs = tools.specs()

    all_calls: list[ToolTrace] = []
    total_model_time = 0.0
    total_tools_time = 0.0
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    rounds = 0
    model_name = None
    stopped = "answer"
    final_text = ""

    while rounds < max_rounds:
        rounds += 1

        response = llm.chat(messages, tools=tool_specs, tool_choice="auto")
        total_model_time += response["elapsed"]
        model_name = response["model"]

        usage = response.get("usage") or {}
        total_usage["prompt_tokens"] += usage.get("prompt_tokens", 0)
        total_usage["completion_tokens"] += usage.get("completion_tokens", 0)
        total_usage["total_tokens"] += usage.get("total_tokens", 0)

        model_msg = response["message"]
        messages.append(model_msg)

        tool_calls = getattr(model_msg, "tool_calls", None) or []

        if not tool_calls:
            final_text = (model_msg.content or "").strip()
            stopped = "answer"
            break

        for call_item in tool_calls:
            fname = call_item.function.name
            fargs = call_item.function.arguments

            t0 = time.perf_counter()
            tool_res = tools.call(name=fname, raw_arguments=fargs, ctx=ctx)
            t_elapsed = time.perf_counter() - t0
            total_tools_time += t_elapsed

            trace = ToolTrace(
                round=rounds,
                name=fname,
                arguments=fargs if isinstance(fargs, str) else json.dumps(fargs, ensure_ascii=False),
                status=tool_res.status,
                reason=tool_res.reason,
                result=tool_res.content,
                elapsed=round(t_elapsed, 4),
            )
            all_calls.append(trace)

            if isinstance(tool_res.content, (dict, list)):
                content_str = json.dumps(tool_res.content, ensure_ascii=False)
            else:
                content_str = str(tool_res.content) if tool_res.content is not None else "{}"

            messages.append({
                "role": "tool",
                "tool_call_id": call_item.id,
                "content": content_str,
            })

    if rounds >= max_rounds and not final_text:
        stopped = "limit"
        try:
            final_resp = llm.chat(messages, tools=None)
            total_model_time += final_resp["elapsed"]
            u = final_resp.get("usage") or {}
            total_usage["prompt_tokens"] += u.get("prompt_tokens", 0)
            total_usage["completion_tokens"] += u.get("completion_tokens", 0)
            total_usage["total_tokens"] += u.get("total_tokens", 0)
            final_text = (final_resp["message"].content or "").strip()
        except Exception:
            final_text = (
                "Досягнуто ліміту звертань до сервісу. Будь ласка, сформулюйте запит більш конкретно."
            )

    return Answer(
        text=final_text,
        calls=all_calls,
        rounds=rounds,
        stopped=stopped,
        model=model_name,
        elapsed={
            "model": round(total_model_time, 3),
            "tools": round(total_tools_time, 4),
        },
        usage=total_usage,
    )
