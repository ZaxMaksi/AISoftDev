"""Веб-рівень застосунку: сторінка із завантаженням файлу і JSON-ендпоінт.

Цей файл не має знати про `ultralytics`, ваги моделі й формат її «сирого»
виводу — усе це лишається в `app/detector.py`. Тут вирішується інше: що
застосунок віддає клієнтові та з яким HTTP-статусом.

Запуск із папки pr2:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

from pathlib import Path
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import HTMLResponse
from . import detector

app = FastAPI(title="Детекція обʼєктів — ПР2")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"

@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Віддати сторінку із завантаженням зображення."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.post("/api/detect")
async def api_detect(image: UploadFile = File(...), confidence: float = Query(detector.DEFAULT_CONFIDENCE, ge=0.0, le=1.0)):
    """
    Повернути знайдені на зображенні обʼєкти у форматі JSON.
    
    Обробляє помилки детекції (некоректний файл, порожній файл тощо)
    і повертає відповідні HTTP-статуси.
    """
    try:
        content = await image.read()
        result = detector.detect(content, confidence=confidence)
        return result
    except detector.DetectionError as e:
        raise HTTPException(status_code=e.status_code, detail=e.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Внутрішня помилка сервера: {e}")

