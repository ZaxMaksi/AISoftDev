"""Веб-рівень застосунку: сторінка з формою і JSON-ендпоінт.

Цей файл не має знати про `requests`, адреси сервісів і коди їхніх
відповідей — усе це лишається в `app/weather.py`. Тут вирішується інше:
що застосунок віддає клієнтові та з яким HTTP-статусом.

Запуск із папки pr1:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse

from . import weather

app = FastAPI(title="Погода — ПР1")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Віддати сторінку з формою вводу міста."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/weather")
def api_weather(city: str):
    """Повернути поточну погоду в місті у форматі JSON.

    Обробляє специфічні помилки модуля інтеграції та повертає відповідні
    HTTP-статуси, щоб клієнт міг зрозуміло повідоти користувача.
    """
    try:
        return weather.get_current_weather(city)
    except weather.CityNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except weather.InvalidAPIRequestError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except weather.APIServiceUnavailableError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except weather.WeatherError as e:
        raise HTTPException(status_code=500, detail=f"Внутрішня помилка модуля погоди: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail="Сталася непередбачена помилка")

