"""Підставний провайдер: канонічні тексти, затримки, збої на вимогу.

Працює без мережі й ключа, дозволяє імітувати затримки та збої
для тестування надійності, бюджету й кешування.
"""

import json
import os
import random
import time
from pathlib import Path
from typing import Iterator

from dotenv import load_dotenv

from .base import Chunk, ProviderError, Request, Result

load_dotenv()

FIRST_DELAY_MS = int(os.getenv("MOCK_FIRST_DELAY_MS", "600"))
DELAY_MS = int(os.getenv("MOCK_DELAY_MS", "40"))
FAILURE_RATE = float(os.getenv("MOCK_FAILURE_RATE", "0"))
FAILURE_KIND = os.getenv("MOCK_FAILURE_KIND", "unavailable")

# Кешуємо зразки відповідей для правдоподібних текстів
_SAMPLES_FILE = Path(__file__).resolve().parent.parent.parent / "samples" / "requests.json"
_REPLIES: dict[str, str] = {}
if _SAMPLES_FILE.exists():
    try:
        data = json.loads(_SAMPLES_FILE.read_text(encoding="utf-8"))
        for item in data:
            _REPLIES[item["id"]] = item.get("reply_example", "")
    except Exception:
        pass


class MockProvider:
    """Провайдер без моделі: канонічні тексти, затримки, збої на вимогу."""

    name = "mock"
    model = "mock-1"

    def complete(self, request: Request) -> Result:
        """Повернути відповідь цілком після імітації затримок."""
        chunks = list(self.stream(request))
        text = "".join(c.text for c in chunks)
        last_chunk = chunks[-1] if chunks else Chunk(text="")
        return Result(
            text=text,
            provider=self.name,
            model=self.model,
            usage=last_chunk.usage,
            finish_reason="stop",
            elapsed=0.0,
        )

    def stream(self, request: Request) -> Iterator[Chunk]:
        """Віддавати текст фрагментами з затримкою `DELAY_MS` між ними."""
        first_delay = FIRST_DELAY_MS / 1000.0
        step_delay = DELAY_MS / 1000.0

        if first_delay > 0:
            time.sleep(first_delay)

        # Перевірка на симуляцію збою
        should_fail = random.random() < FAILURE_RATE
        # Можливість форсувати успіх або збій через metadata (для детермінованих тестів)
        if "mock_fail" in request.metadata:
            should_fail = bool(request.metadata["mock_fail"])

        fail_kind = request.metadata.get("mock_failure_kind", FAILURE_KIND)

        if should_fail and fail_kind != "mid_stream":
            if fail_kind == "timeout":
                raise ProviderError("timeout", "Таймаут очікування відповіді провайдера", retryable=True)
            elif fail_kind == "rate_limit":
                raise ProviderError("rate_limit", "Перевищено ліміт запитів провайдера (429)", retryable=True, retry_after=2.0)
            elif fail_kind == "auth":
                raise ProviderError("auth", "Помилка автентифікації API ключа (401)", retryable=False)
            elif fail_kind == "bad_request":
                raise ProviderError("bad_request", "Некоректний запит до моделі (400)", retryable=False)
            else:
                raise ProviderError("unavailable", "Провайдер тимчасово недоступний (503)", retryable=True)

        req_id = request.metadata.get("request_id", "")
        reply_text = _REPLIES.get(req_id) or "Доброго дня! Ваше звернення опрацьовано. Інформацію перевірено за наявною базою даних магазину «Сузірʼя»."

        words = reply_text.split(" ")
        num_words = len(words)

        prompt_len = sum(len(m.get("content", "")) for m in request.messages)
        prompt_tokens = max(50, int(prompt_len / 3.5))
        completion_tokens = max(20, int(len(reply_text) / 3.5))

        for idx, word in enumerate(words):
            if step_delay > 0:
                time.sleep(step_delay)

            # Перевірка збою посеред потоку
            if should_fail and fail_kind == "mid_stream" and idx >= 3:
                raise ProviderError("unavailable", "Обрив зʼєднання під час потокової передачі (mid_stream)", retryable=True)

            piece = word + (" " if idx < num_words - 1 else "")
            
            # В останньому фрагменті віддаємо usage
            if idx == num_words - 1:
                usage = {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                    "prompt_tokens_details": {"cached_tokens": 0},
                }
                yield Chunk(text=piece, usage=usage, finish_reason="stop")
            else:
                yield Chunk(text=piece)
