"""Денний ліміт витрат: скільки застосунок може витратити за добу.

Обмеження частоти з ПР10 рахує запити за хвилину і не завадить
витратити місячний бюджет рівномірним потоком. Тут рахуються гроші за
день на весь застосунок. Ліній дві: мʼяка — попередження на частці
`WARN_AT` (оператор бачить, журнал отримує `WARNING`), жорстка —
відмова після вичерпання.
"""

import datetime
import json
import logging
import os
import threading
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DAILY_USD = float(os.getenv("BUDGET_DAILY_USD", "1.00"))
WARN_AT = float(os.getenv("BUDGET_WARN_AT", "0.8"))
BUDGET_FILE = os.getenv("BUDGET_FILE", "state/budget.json")

log = logging.getLogger("app.budget")


class BudgetExceeded(Exception):
    """Денний ліміт вичерпано. `spent`, `limit` — у доларах; `resets_at` —
    коли ліміт відновиться, у форматі ISO."""

    def __init__(self, spent: float, limit: float, resets_at: str) -> None:
        super().__init__(f"денний ліміт витрат {limit:.2f} $ вичерпано (витрачено {spent:.4f} $); "
                         f"відновиться {resets_at}")
        self.spent = spent
        self.limit = limit
        self.resets_at = resets_at


_lock = threading.Lock()
_spent: float = 0.0
_current_day: str = ""
_warned: bool = False


def _today_str() -> str:
    return datetime.date.today().isoformat()


def _next_midnight_iso() -> str:
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    # Початок наступної доби
    midnight = datetime.datetime.combine(tomorrow, datetime.time.min)
    # З урахуванням локального часового поясу
    local_tz = datetime.datetime.now().astimezone().tzinfo
    return midnight.replace(tzinfo=local_tz).isoformat()


def _load_from_disk() -> None:
    global _spent, _current_day, _warned
    today = _today_str()
    if not BUDGET_FILE:
        _current_day = today
        return

    path = Path(BUDGET_FILE)
    if not path.exists():
        _current_day = today
        _spent = 0.0
        return

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("day") == today:
            _current_day = today
            _spent = float(data.get("spent", 0.0))
            _warned = _spent >= (DAILY_USD * WARN_AT)
        else:
            # Настав новий день — лічильник скидається
            _current_day = today
            _spent = 0.0
            _warned = False
            _save_to_disk()
    except Exception:
        _current_day = today
        _spent = 0.0


def _save_to_disk() -> None:
    if not BUDGET_FILE:
        return
    path = Path(BUDGET_FILE)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "day": _current_day,
        "spent": round(_spent, 6),
        "resets_at": _next_midnight_iso(),
    }
    tmp = path.with_suffix(".tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except Exception:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


with _lock:
    _load_from_disk()


def settings() -> dict:
    """Чинні налаштування — для сторінки."""
    return {"daily_usd": DAILY_USD, "warn_at": WARN_AT, "file": BUDGET_FILE}


def _check_rollover_locked() -> None:
    global _spent, _current_day, _warned
    today = _today_str()
    if _current_day != today:
        _current_day = today
        _spent = 0.0
        _warned = False
        _save_to_disk()
    elif BUDGET_FILE and Path(BUDGET_FILE).exists():
        try:
            data = json.loads(Path(BUDGET_FILE).read_text(encoding="utf-8"))
            if data.get("day") == today:
                disk_spent = float(data.get("spent", 0.0))
                if disk_spent != _spent:
                    _spent = disk_spent
                    _warned = _spent >= (DAILY_USD * WARN_AT)
        except Exception:
            pass
    elif BUDGET_FILE and not Path(BUDGET_FILE).exists():
        _spent = 0.0
        _warned = False


def check(expected_cost: float | None = None) -> None:
    """Переконатися, що платний виклик ще дозволений; інакше підняти
    `BudgetExceeded`. `expected_cost` — очікувана вартість, якщо ви
    вирішите резервувати її наперед."""
    with _lock:
        _check_rollover_locked()
        total_projected = _spent + (expected_cost or 0.0)
        if total_projected >= DAILY_USD:
            resets_at = _next_midnight_iso()
            raise BudgetExceeded(spent=_spent, limit=DAILY_USD, resets_at=resets_at)


def record(cost: float | None) -> None:
    """Зарахувати вартість виконаного звертання до витрат дня."""
    if cost is None or cost <= 0:
        return

    global _spent, _warned
    with _lock:
        _check_rollover_locked()
        _spent += cost
        resets_at = _next_midnight_iso()
        
        # Перевірка порогу попередження
        if not _warned and _spent >= (DAILY_USD * WARN_AT):
            _warned = True
            log.warning("budget.warning", extra={
                "spent": round(_spent, 6),
                "limit": DAILY_USD,
                "warn_at": WARN_AT,
                "percentage": round((_spent / DAILY_USD) * 100, 1),
                "resets_at": resets_at,
            })
            
        _save_to_disk()


def status() -> dict:
    """Стан для `GET /api/usage` і сторінки: наприклад,
    `{"day": "2026-10-23", "spent": 0.1234, "limit": 1.0, "remaining": 0.8766,
    "warning": False, "exceeded": False, "resets_at": "2026-10-24T00:00:00+03:00"}`."""
    with _lock:
        _check_rollover_locked()
        spent = round(_spent, 6)
        remaining = max(0.0, round(DAILY_USD - spent, 6))
        return {
            "day": _current_day,
            "spent": spent,
            "limit": DAILY_USD,
            "remaining": remaining,
            "warning": spent >= (DAILY_USD * WARN_AT),
            "exceeded": spent >= DAILY_USD,
            "resets_at": _next_midnight_iso(),
        }


def reset(clear_file: bool = False) -> None:
    """Скинути стан лічильника (для тестів)."""
    global _spent, _warned, _current_day
    with _lock:
        _spent = 0.0
        _warned = False
        _current_day = _today_str()
        if clear_file and BUDGET_FILE:
            p = Path(BUDGET_FILE)
            if p.exists():
                p.unlink(missing_ok=True)
