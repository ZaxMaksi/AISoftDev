"""Кеш відповідей: однакове питання — одна оплата.

Повтори беруться звідусіль: «Повторити» без правок, пʼять натискань
поспіль, двоє операторів над одним зверненням. Кеш повертає готовий
результат, не звертаючись до провайдера, — і це видно операторові,
журналу й бюджету.
"""

import copy
import hashlib
import json
import os
import tempfile
import threading
import time
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from .providers.base import Request, Result

load_dotenv()

ENABLED = os.getenv("CACHE_ENABLED", "1") == "1"
TTL_SECONDS = float(os.getenv("CACHE_TTL_SECONDS", "3600"))
MAX_ENTRIES = int(os.getenv("CACHE_MAX_ENTRIES", "500"))
# Порожнє значення — кеш лише в памʼяті процесу.
CACHE_FILE = os.getenv("CACHE_FILE", "state/cache.json")

_lock = threading.RLock()
# _store: key -> {"result": dict, "created_at": float}
_store: dict[str, dict] = {}
_stats = {"hits": 0, "misses": 0, "bypassed": 0}

# Single-flight: відстеження однакових запитів, що виконуються зараз
_inflight_events: dict[str, threading.Event] = {}


def settings() -> dict:
    """Чинні налаштування — для сторінки й журналу."""
    return {
        "enabled": ENABLED,
        "ttl_seconds": TTL_SECONDS,
        "max_entries": MAX_ENTRIES,
        "file": CACHE_FILE or None,
    }


def _load_from_disk() -> None:
    """Завантажити стан кешу з диска."""
    global _store
    if not CACHE_FILE:
        return
    path = Path(CACHE_FILE)
    if not path.exists():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        now = time.time()
        # Залишаємо лише непрострочені записи
        _store = {
            k: v for k, v in data.get("entries", {}).items()
            if now - v.get("created_at", 0) <= TTL_SECONDS
        }
        _stats["hits"] = data.get("stats", {}).get("hits", 0)
        _stats["misses"] = data.get("stats", {}).get("misses", 0)
        _stats["bypassed"] = data.get("stats", {}).get("bypassed", 0)
    except Exception:
        _store = {}


def _save_to_disk() -> None:
    """Атомарно зберегти стан кешу на диск."""
    if not CACHE_FILE:
        return
    path = Path(CACHE_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    payload = {
        "stats": _stats,
        "entries": _store,
    }
    raw = json.dumps(payload, ensure_ascii=False, indent=2)
    
    # Атомарний запис через тимчасовий файл
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(raw, encoding="utf-8")
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


# Ініціалізація сховища
with _lock:
    _load_from_disk()


def make_key(request: Request, prompt_version: str) -> str:
    """Ключ кешу для запиту: однакові запити — однаковий ключ, будь-яка
    відмінність у тому, що впливає на відповідь, — інший."""
    canonical = {
        "provider": request.metadata.get("provider", ""),
        "model": request.model or "",
        "prompt_version": prompt_version,
        "messages": request.messages,
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
    }
    raw = json.dumps(canonical, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get(key: str) -> Result | None:
    """Результат за ключем, якщо він є й не прострочений; інакше `None`.
    Коли кеш вимкнено — завжди `None`."""
    if not ENABLED:
        return None

    now = time.time()
    with _lock:
        entry = _store.get(key)
        if entry is None:
            _stats["misses"] += 1
            return None

        # Перевірка TTL
        if now - entry["created_at"] > TTL_SECONDS:
            del _store[key]
            _save_to_disk()
            _stats["misses"] += 1
            return None

        _stats["hits"] += 1
        r_dict = entry["result"]
        res = Result(
            text=r_dict["text"],
            provider=r_dict["provider"],
            model=r_dict["model"],
            usage=r_dict.get("usage"),
            finish_reason=r_dict.get("finish_reason"),
            elapsed=r_dict.get("elapsed", 0.0),
            cost=0.0,  # Кешований запит безкоштовний
            cost_estimated=False,
            cached=True,
        )
        return res


def record_bypassed() -> None:
    """Зафіксувати обхід кешу (fresh=True)."""
    with _lock:
        _stats["bypassed"] += 1


def put(key: str, result: Result) -> None:
    """Зберегти результат. Коли кеш вимкнено — нічого не робити."""
    if not ENABLED:
        return

    # Не кешуємо порожні або зі збоями (неповні результати)
    if not result.text:
        return

    with _lock:
        # Перевірка ліміту записів
        if len(_store) >= MAX_ENTRIES and key not in _store:
            # Видаляємо найстаріший запис
            oldest_key = min(_store.keys(), key=lambda k: _store[k].get("created_at", 0))
            del _store[oldest_key]

        _store[key] = {
            "created_at": time.time(),
            "result": {
                "text": result.text,
                "provider": result.provider,
                "model": result.model,
                "usage": result.usage,
                "finish_reason": result.finish_reason,
                "elapsed": result.elapsed,
            }
        }
        _save_to_disk()


def stats() -> dict:
    """Лічильники для `GET /api/usage` і сторінки: наприклад,
    `{"hits": 3, "misses": 7, "bypassed": 1, "entries": 6}`."""
    with _lock:
        return {
            "hits": _stats["hits"],
            "misses": _stats["misses"],
            "bypassed": _stats["bypassed"],
            "entries": len(_store),
        }


def reset(clear_file: bool = False) -> None:
    """Скинути кеш і лічильники (для тестів та ізольованих прогонів)."""
    global _store
    with _lock:
        _store.clear()
        _stats["hits"] = 0
        _stats["misses"] = 0
        _stats["bypassed"] = 0
        _inflight_events.clear()
        if clear_file and CACHE_FILE:
            p = Path(CACHE_FILE)
            if p.exists():
                p.unlink(missing_ok=True)


# Single-Flight механізм
def get_or_wait_inflight(key: str, timeout: float = 30.0) -> tuple[Result | None, threading.Event | None]:
    """Допоміжна функція single-flight:
    Якщо такий ключ уже генерується іншим потоком — чекає на завершення або повертає кеш.
    Якщо запит перший — реєструє подію і повертає (None, event).
    """
    if not ENABLED:
        return None, None

    with _lock:
        # Спочатку перевіримо, можливо вже є готовий кеш
        cached = get(key)
        if cached is not None:
            return cached, None

        ev = _inflight_events.get(key)
        if ev is None:
            # Ми перші
            my_event = threading.Event()
            _inflight_events[key] = my_event
            return None, my_event

    # Якщо запит уже в польоті, чекаємо
    ev.wait(timeout=timeout)
    return get(key), None


def release_inflight(key: str, ev: threading.Event | None) -> None:
    """Позначити завершення генерації запиту в single-flight."""
    if not ev:
        return
    with _lock:
        if _inflight_events.get(key) == ev:
            del _inflight_events[key]
    ev.set()
