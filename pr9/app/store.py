"""Сховище стану звернень: зберегти, прочитати, перелічити.

Стан живе поза памʼяттю процесу (у файловому каталозі RUNS_DIR як окремі JSON-файли).
Для запобігання пошкодженню файлів під час раптових перезапусків чи збоїв
використовується атомарний запис: спочатку запис у тимчасовий файл у тому ж каталозі,
а потім атомарне перейменування через os.replace.
"""

from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from dotenv import load_dotenv

from .state import Run, Step

load_dotenv()

RUNS_DIR = Path(os.getenv("RUNS_DIR", "runs"))


def _ensure_dir() -> Path:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    return RUNS_DIR


def _dict_to_step(d: dict[str, Any]) -> Step:
    return Step(
        n=d.get("n", 1),
        name=d.get("name", ""),
        kind=d.get("kind", "code"),
        status=d.get("status", "ok"),
        detail=d.get("detail"),
        input=d.get("input"),
        output=d.get("output"),
        elapsed=d.get("elapsed"),
        usage=d.get("usage"),
        at=d.get("at"),
    )


def _dict_to_run(d: dict[str, Any]) -> Run:
    raw_steps = d.get("steps") or []
    steps = [_dict_to_step(s) if isinstance(s, dict) else s for s in raw_steps]
    return Run(
        run_id=d["run_id"],
        customer_id=d["customer_id"],
        text=d["text"],
        status=d.get("status", "running"),
        route=d.get("route"),
        steps=steps,
        data=d.get("data") or {},
        pending=d.get("pending"),
        decision=d.get("decision"),
        reply=d.get("reply"),
        stopped=d.get("stopped"),
        usage=d.get("usage") or {},
        elapsed=d.get("elapsed") or {},
        created_at=d.get("created_at"),
        updated_at=d.get("updated_at"),
    )


def save(run: Run) -> None:
    """Зберегти стан звернення, атомарно замінивши попередній файл."""
    folder = _ensure_dir()
    target_path = folder / f"{run.run_id}.json"

    data = asdict(run)
    json_bytes = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")

    # Атомарний запис через тимчасовий файл в тій же файловій системі
    tmp_fd, tmp_path = tempfile.mkstemp(dir=str(folder), prefix=f".{run.run_id}_", suffix=".tmp")
    try:
        with os.fdopen(tmp_fd, "wb") as f:
            f.write(json_bytes)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target_path)
    except Exception:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass
        raise


def load(run_id: str) -> Run | None:
    """Прочитати стан звернення; `None`, якщо такого немає."""
    target_path = _ensure_dir() / f"{run_id}.json"
    if not target_path.exists() or not target_path.is_file():
        return None
    try:
        content = target_path.read_text(encoding="utf-8")
        data = json.loads(content)
        return _dict_to_run(data)
    except Exception:
        return None


def list_runs(status: str | None = None) -> list[Run]:
    """Усі звернення, за потреби — лише з певним статусом; нові першими."""
    folder = _ensure_dir()
    runs: list[Run] = []
    for p in folder.glob("*.json"):
        if p.name.startswith("."):
            continue
        run = load(p.stem)
        if run is not None:
            if status is None or run.status == status:
                runs.append(run)

    # Сортуємо новіші першими за updated_at або created_at
    def sort_key(r: Run) -> str:
        return r.updated_at or r.created_at or ""

    runs.sort(key=sort_key, reverse=True)
    return runs
