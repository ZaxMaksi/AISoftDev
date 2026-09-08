"""Модуль inference: єдине місце застосунку, яке знає про модель.

Тут живуть ваги, поріг упевненості й формат «сирого» результату моделі.
Веб-рівень (`app/main.py`) отримує звідси готовий структурований список
знайдених обʼєктів і нічого не знає ані про `ultralytics`, ані про те,
у якому вигляді модель віддає рамки.

Функції нижче — заготовки. Реалізуйте їх самі, ухваливши по дорозі
рішення з розділу 2 практичної роботи:

* де саме завантажувати ваги, щоб це сталося **один раз**, а не на кожен запит;
* яким узяти поріг упевненості й чи дозволяти змінювати його ззовні;
* у якому вигляді віддавати результат: які поля, які одиниці координат;
* як виміряти час inference і що саме до нього зараховувати;
* як повестися, коли надійшов не той файл — не зображення або порожній.

Довідка про модель: https://docs.ultralytics.com/
"""

import time
from io import BytesIO
from typing import List, Dict, Any

from PIL import Image
from ultralytics import YOLO

WEIGHTS = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.25

_model = None


class DetectionError(Exception):
    """Помилка детекції, яку ми показуємо користувачеві."""
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(self.message)


def load_model() -> YOLO:
    """Повертає завантажену модель (singleton)."""
    global _model
    if _model is None:
        try:
            _model = YOLO(WEIGHTS)
        except Exception as e:
            raise DetectionError(f"Не вдалося завантажити модель: {e}", status_code=500)
    return _model


def detect(image_bytes: bytes, confidence: float = DEFAULT_CONFIDENCE) -> Dict[str, Any]:
    """
    Знайти обʼєкти на зображенні.
    
    Повертає:
    {
        "detections": [{"box": [x1, y1, x2, y2], "conf": 0.9, "class": "cat"}, ...],
        "count": 2,
        "inference_time": 0.123
    }
    """
    if not image_bytes:
        raise DetectionError("Файл порожній або не отриманий")

    model = load_model()

    try:
        img = Image.open(BytesIO(image_bytes))
        img.verify()
        img = Image.open(BytesIO(image_bytes))
    except Exception:
        raise DetectionError("Файл не є коректним зображенням")

    start_time = time.perf_counter()
    results = model.predict(img, conf=confidence, verbose=False)
    elapsed = time.perf_counter() - start_time
    detections = []
    result = results[0]
    
    for box in result.boxes:
        coords = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        cls_name = model.names[cls_id]
        detections.append({
            "box": [round(c, 1) for c in coords],
            "conf": round(conf, 3),
            "class": cls_name
        })

    return {
        "detections": detections,
        "count": len(detections),
        "inference_time": round(elapsed, 4)
    }

