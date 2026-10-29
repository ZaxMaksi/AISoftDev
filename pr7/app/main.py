"""Веб-рівень застосунку: сторінка розбору рахунків і JSON-ендпоінти.

Цей файл не знає ані як готується зображення, ані якою моделлю й за
якою інструкцією вилучено поля, ані якими правилами їх перевірено — усе
це в `app/images.py`, `app/llm.py`, `app/schema.py`, `app/rules.py` і
поєднується в `app/extraction.py`. Тут вирішується інше: що застосунок
приймає від сторінки, що віддає їй і з яким HTTP-статусом.

Запуск із папки pr7:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

import logging
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import extraction
from .images import ImageError

logger = logging.getLogger(__name__)

app = FastAPI(title="Розбір рахунків — ПР7")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

# Зразки віддаються як статичні файли
app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")


def result_to_dict(result: extraction.Result) -> dict:
    """Перетворити результат конвеєра на те, що піде на сторінку."""
    return {
        "decision": result.decision,
        "reasons": result.reasons,
        "document": result.document,
        "issues": [asdict(issue) for issue in result.issues],
        "image": result.image,
        "model": result.model,
        "elapsed": result.elapsed,
        "usage": result.usage,
    }


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Віддати сторінку."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/samples")
def api_samples() -> dict:
    """Перелік зразків за папками: `clean`, `degraded` і, якщо є, `own`."""
    groups = {}
    for folder in sorted(p for p in SAMPLES_DIR.iterdir() if p.is_dir()):
        files = sorted(f.name for f in folder.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
        if files:
            groups[folder.name] = files
    return groups


@app.post("/api/extract")
async def api_extract(image: UploadFile = File(...)) -> JSONResponse:
    """Розібрати документ і повернути поля, проблеми й рішення."""

    # 1. Читаємо вміст
    try:
        content = await image.read()
    except Exception as e:
        logger.error("Помилка читання завантаженого файлу: %s", e)
        raise HTTPException(status_code=400, detail="Не вдалося прочитати файл.")

    # 2. Перевірка порожнього файлу
    if not content:
        raise HTTPException(status_code=400, detail="Файл порожній. Будь ласка, завантажте зображення.")

    # 3. Обробка через конвеєр
    try:
        result = extraction.process(content)
    except Exception as e:
        logger.error("Необроблена помилка конвеєра: %s", e, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Внутрішня помилка сервера. Зверніться до адміністратора."
        )

    data = result_to_dict(result)

    # 4. Статус відповіді залежить від рішення
    # reject через невалідний файл → 400; решта → 200 (сторінка розбирається сама)
    if result.decision == "reject" and not result.document:
        # reject через проблему з файлом або моделлю — не технічна помилка
        return JSONResponse(status_code=200, content=data)

    return JSONResponse(status_code=200, content=data)
