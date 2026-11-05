"""Фонові задачі: генерація чернетки у фоновому потоці.

Забезпечує:
- передачу ідентифікатора запиту (request_id) у фоновий потік;
- потік server-sent events для сторінки (status, delta, done, error, cancelled);
- облік кешу, вартості та токенів у фінальному знімку чернетки;
- скасування роботи користувачем (cancel).
"""

import contextvars
import datetime
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Iterator

from . import drafting, gateway, telemetry
from .providers.base import ProviderError

log = logging.getLogger("app.jobs")

TERMINAL = {"done", "failed", "cancelled"}


@dataclass
class Draft:
    """Стан однієї чернетки."""

    draft_id: str
    request_id: str
    client: str
    status: str = "queued"
    text: str = ""
    attempts: int = 0
    provider: str | None = None
    model: str | None = None
    fallback: bool = False
    cached: bool = False
    cost: float | None = None
    cost_estimated: bool = False
    error: dict | None = None
    usage: dict | None = None
    elapsed: dict = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    events: list[dict] = field(default_factory=list)


_drafts: dict[str, Draft] = {}
_order: list[str] = []
_cancel_events: dict[str, threading.Event] = {}

_lock = threading.Lock()
_changed = threading.Condition(_lock)


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def new_draft(request_id: str, client: str) -> Draft:
    """Створити чернетку зі статусом `queued` і зареєструвати її."""
    draft = Draft(
        draft_id=uuid.uuid4().hex[:8],
        request_id=request_id,
        client=client,
        created_at=_now(),
        updated_at=_now(),
    )
    with _lock:
        _drafts[draft.draft_id] = draft
        _order.insert(0, draft.draft_id)
        _cancel_events[draft.draft_id] = threading.Event()
    return draft


def get(draft_id: str) -> Draft | None:
    with _lock:
        return _drafts.get(draft_id)


def list_drafts(limit: int = 50) -> list[Draft]:
    """Чернетки, нові першими."""
    with _lock:
        return [_drafts[i] for i in _order[:limit]]


def emit(draft: Draft, event: str, data: dict | None = None) -> None:
    """Додати подію до журналу чернетки й розбудити підписників."""
    with _changed:
        draft.events.append({"id": len(draft.events) + 1, "event": event, "data": data or {}})
        draft.updated_at = _now()
        _changed.notify_all()


def subscribe(draft_id: str, since: int = 0, heartbeat: float = 15.0) -> Iterator[dict | None]:
    """Віддавати події чернетки для SSE."""
    while True:
        with _changed:
            draft = _drafts.get(draft_id)
            if draft is None:
                return
            pending = draft.events[since:]
            if not pending:
                if draft.status in TERMINAL:
                    return
                _changed.wait(timeout=heartbeat)
                pending = draft.events[since:]
        if not pending:
            yield None
            continue
        for event in pending:
            yield event
        since += len(pending)


def _worker(draft: Draft, sample: dict, fresh: bool, request_id: str) -> None:
    """Виконання запиту в окремому потоці."""
    telemetry.request_id_var.set(request_id)
    cancel_event = _cancel_events.get(draft.draft_id)

    def is_cancelled() -> bool:
        return cancel_event.is_set() if cancel_event else False

    start_time = time.perf_counter()
    first_token_time: float | None = None

    def on_delta(delta: str) -> None:
        nonlocal first_token_time
        if first_token_time is None and delta:
            first_token_time = time.perf_counter() - start_time
        with _lock:
            draft.text += delta
            draft.status = "running"
        emit(draft, "delta", {"text": delta})

    def on_status(status_info: dict) -> None:
        with _lock:
            if "status" in status_info:
                draft.status = status_info["status"]
            if "attempt" in status_info:
                draft.attempts = status_info["attempt"]
            if "provider" in status_info:
                draft.provider = status_info["provider"]
            if "model" in status_info:
                draft.model = status_info["model"]
            if "fallback" in status_info:
                draft.fallback = status_info["fallback"]
            if "cached" in status_info:
                draft.cached = status_info["cached"]
        emit(draft, "status", status_info)

    emit(draft, "status", {"status": "running", "attempt": 1})

    try:
        # Будуємо запит
        req = drafting.build_request(sample)
        req.metadata["draft_id"] = draft.draft_id
        req.metadata["client"] = draft.client
        req.metadata["fresh"] = fresh
        req.metadata["request_id"] = sample.get("id", "")

        result = gateway.generate(
            request=req,
            on_delta=on_delta,
            on_status=on_status,
            is_cancelled=is_cancelled,
        )

        total_time = time.perf_counter() - start_time
        f_token = first_token_time if first_token_time is not None else total_time

        # Валідація тексту чернетки
        warnings = drafting.check(result.text, sample)

        with _lock:
            draft.status = "done"
            draft.text = result.text
            draft.provider = result.provider
            draft.model = result.model
            draft.cached = result.cached
            draft.cost = result.cost
            draft.cost_estimated = result.cost_estimated
            draft.usage = result.usage
            draft.warnings = warnings
            draft.elapsed = {
                "first_token": round(f_token, 2),
                "total": round(total_time, 2),
            }

        done_payload = {
            "text": draft.text,
            "provider": draft.provider,
            "model": draft.model,
            "attempts": draft.attempts or 1,
            "fallback": draft.fallback,
            "cached": draft.cached,
            "cost": draft.cost,
            "cost_estimated": draft.cost_estimated,
            "usage": draft.usage,
            "elapsed": draft.elapsed,
            "warnings": draft.warnings,
        }
        emit(draft, "done", done_payload)

        log.info("draft.done", extra={
            "draft_id": draft.draft_id,
            "sample_id": sample.get("id"),
            "client": draft.client,
            "provider": draft.provider,
            "model": draft.model,
            "cached": draft.cached,
            "cost": draft.cost,
            "cost_estimated": draft.cost_estimated,
            "duration_ms": round(total_time * 1000, 1),
        })

    except ProviderError as err:
        total_time = time.perf_counter() - start_time
        if err.kind == "cancelled":
            with _lock:
                draft.status = "cancelled"
            emit(draft, "cancelled", {})
            log.info("draft.cancelled", extra={
                "draft_id": draft.draft_id,
                "sample_id": sample.get("id"),
                "duration_ms": round(total_time * 1000, 1),
            })
        else:
            with _lock:
                draft.status = "failed"
                draft.error = {
                    "kind": err.kind,
                    "message": err.message,
                    "retryable": err.retryable,
                    "request_id": request_id,
                }
            emit(draft, "error", draft.error)
            log.error("draft.failed", extra={
                "draft_id": draft.draft_id,
                "sample_id": sample.get("id"),
                "kind": err.kind,
                "duration_ms": round(total_time * 1000, 1),
                "detail": err.detail or err.message,
            })
    except Exception as exc:
        total_time = time.perf_counter() - start_time
        with _lock:
            draft.status = "failed"
            draft.error = {
                "kind": "unknown",
                "message": f"Непередбачена помилка: {exc}",
                "retryable": False,
                "request_id": request_id,
            }
        emit(draft, "error", draft.error)
        log.exception("draft.exception", extra={
            "draft_id": draft.draft_id,
            "duration_ms": round(total_time * 1000, 1),
        })


def start(sample: dict, client: str, fresh: bool = False) -> Draft:
    """Створити чернетку для звернення `sample` і запустити генерацію у фоні."""
    req_id = telemetry.request_id_var.get() or telemetry.new_request_id()
    draft = new_draft(sample.get("id", ""), client)

    log.info("draft.created", extra={
        "draft_id": draft.draft_id,
        "sample_id": sample.get("id"),
        "client": client,
        "fresh": fresh,
    })

    # Запуск у фоновому потоці з копією контексту (для збереження request_id)
    ctx = contextvars.copy_context()
    thread = threading.Thread(
        target=ctx.run,
        args=(_worker, draft, sample, fresh, req_id),
        daemon=True,
    )
    thread.start()

    return draft


def cancel(draft_id: str) -> Draft:
    """Попросити фонову роботу зупинитися."""
    with _lock:
        draft = _drafts.get(draft_id)
        if not draft:
            raise KeyError(f"Чернетку {draft_id} не знайдено")
        ev = _cancel_events.get(draft_id)
        if ev:
            ev.set()
        if draft.status not in TERMINAL:
            draft.status = "cancelled"
            emit(draft, "cancelled", {})
        return draft
