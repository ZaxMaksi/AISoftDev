"""Обмеження частоти звертань до власного API.

Ліміт провайдера — спільний на весь застосунок: якщо один користувач
сторінки натисне «скласти чернетку» двадцять разів за хвилину, 429 від
провайдера отримають усі. Тому веб-рівень обмежує кожного клієнта ще до
того, як запит дійде до шлюзу. Це інше обмеження, ніж `MAX_CONCURRENT`
у шлюзі: там — скільки звертань до провайдера виконується одночасно на
всіх, тут — скільки запитів за вікно часу дозволено одному.
"""

import os
import threading
import time
from collections import defaultdict

from dotenv import load_dotenv

load_dotenv()

PER_MINUTE = int(os.getenv("RATE_LIMIT_PER_MINUTE", "10"))


class RateLimited(Exception):
    """Ліміт вичерпано. `retry_after` — через скільки секунд пробувати."""

    def __init__(self, retry_after: float) -> None:
        super().__init__(f"ліміт вичерпано, повторіть через {retry_after:.0f} с")
        self.retry_after = max(1.0, retry_after)


# Зберігаємо часові мітки запитів клієнтів для ковзного вікна
_requests: dict[str, list[float]] = defaultdict(list)
_lock = threading.Lock()


def settings() -> dict:
    """Чинні обмеження — для сторінки."""
    return {"per_minute": PER_MINUTE}


def acquire(client: str) -> None:
    """Зарахувати одне звертання клієнта або підняти `RateLimited`.

    Викликається веб-рівнем перед створенням кожної чернетки.
    Використовує алгоритм ковзного вікна в 60 секунд.
    """
    if PER_MINUTE <= 0:
        return

    now = time.monotonic()
    window = 60.0

    with _lock:
        timestamps = _requests[client]
        # Очищаємо застарілі запити, які вийшли за межі вікна
        cutoff = now - window
        while timestamps and timestamps[0] <= cutoff:
            timestamps.pop(0)

        if len(timestamps) >= PER_MINUTE:
            oldest = timestamps[0]
            retry_after = max(1.0, window - (now - oldest))
            raise RateLimited(retry_after=retry_after)

        timestamps.append(now)


def reset() -> None:
    """Очистити лічильники (для тестів)."""
    with _lock:
        _requests.clear()
