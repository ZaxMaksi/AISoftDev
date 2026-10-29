"""Підготовка зображення до надсилання моделі.

Файл, який надіслав користувач, — недовірені байти: це може бути не
зображення, зображення завбільшки 40 мегапікселів, фото, повернуте на
бік (орієнтація записана в EXIF, а не в пікселях), або знімок 300 × 200,
на якому цифр не розібрати. Модуль вирішує, що з цього можна віддати
моделі і в якому вигляді. Про модель він не знає: повертає підготовлені
байти й відомості про них, а як вкласти їх у запит — справа `app/llm.py`.
"""

import io
import os
import time
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()

UPLOAD_MAX_BYTES = int(float(os.getenv("UPLOAD_MAX_MB", "10")) * 1024 * 1024)
IMAGE_MAX_SIDE = int(os.getenv("IMAGE_MAX_SIDE", "1600"))

# Мінімальний розмір зображення в пікселях по коротшій стороні
IMAGE_MIN_SIDE = 100

# Підтримувані MIME-типи зображень (формати Pillow)
SUPPORTED_FORMATS = {"PNG", "JPEG", "WEBP", "BMP", "TIFF"}

# Максимальна кількість мегапікселів (захист від bomb-зображень)
MAX_MEGAPIXELS = 40


class ImageError(Exception):
    """Файл не можна віддати моделі. Повідомлення — для користувача."""


@dataclass
class PreparedImage:
    """Зображення, готове до надсилання.

    `data` і `mime` — те, що піде моделі. `original` і `sent` — розміри
    до й після підготовки, наприклад `{"width": 1240, "height": 1754,
    "bytes": 179639}`: сторінка показує їх поруч із токенами, щоб було
    видно, скільки коштує роздільність.
    `prepare_time` — час підготовки в секундах.
    """

    data: bytes
    mime: str
    original: dict
    sent: dict
    prepare_time: float = 0.0


def prepare(content: bytes, max_side: int | None = None) -> PreparedImage:
    """Перевірити байти, що надійшли, і підготувати зображення для моделі.

    Непридатний файл — `ImageError` з поясненням, що не так.
    """
    t0 = time.monotonic()

    # 1. Порожній файл
    if not content:
        raise ImageError("Файл порожній. Будь ласка, завантажте зображення.")

    # 2. Розмір файлу
    if len(content) > UPLOAD_MAX_BYTES:
        max_mb = UPLOAD_MAX_BYTES / (1024 * 1024)
        actual_mb = len(content) / (1024 * 1024)
        raise ImageError(
            f"Файл завеликий: {actual_mb:.1f} МБ. "
            f"Максимальний розмір — {max_mb:.0f} МБ."
        )

    # 3. Перевірка формату через Pillow (не довіряємо розширенню)
    try:
        from PIL import Image, ImageOps
    except ImportError:
        raise ImageError("Pillow не встановлено. Запустіть: pip install pillow")

    try:
        img = Image.open(io.BytesIO(content))
        # Перевірка, що файл читається (lazy load → треба getdata або load)
        img.load()
    except Exception:
        raise ImageError(
            "Файл не є підтримуваним зображенням або пошкоджений. "
            "Підтримуються: PNG, JPEG, WEBP."
        )

    # 4. Перевірка формату
    fmt = (img.format or "").upper()
    if fmt not in SUPPORTED_FORMATS:
        raise ImageError(
            f"Непідтримуваний формат зображення: {fmt or 'невідомий'}. "
            f"Підтримуються: PNG, JPEG, WEBP."
        )

    # 5. Захист від bomb-зображень
    w, h = img.size
    megapixels = (w * h) / 1_000_000
    if megapixels > MAX_MEGAPIXELS:
        raise ImageError(
            f"Зображення надто велике: {w}×{h} пікс. ({megapixels:.1f} МП). "
            f"Максимум — {MAX_MEGAPIXELS} МП."
        )

    # 6. Мінімальний розмір
    if min(w, h) < IMAGE_MIN_SIDE:
        raise ImageError(
            f"Зображення надто мале: {w}×{h} пікс. "
            f"Мінімальний розмір по коротшій стороні — {IMAGE_MIN_SIDE} пікс."
        )

    original_size = {"width": w, "height": h, "bytes": len(content)}

    # 7. EXIF-орієнтація (фото з телефона)
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass  # якщо EXIF немає або некоректний — ігноруємо

    # 8. Конвертація в RGB (JPEG не підтримує RGBA/P)
    if img.mode not in ("RGB", "L"):
        try:
            img = img.convert("RGB")
        except Exception as e:
            raise ImageError(f"Не вдалося конвертувати зображення: {e}")
    elif img.mode == "L":
        img = img.convert("RGB")

    # 9. Зменшення до максимальної сторони зі збереженням пропорцій
    target_max_side = max_side if max_side is not None else IMAGE_MAX_SIDE
    if max(img.size) > target_max_side:
        img.thumbnail((target_max_side, target_max_side), Image.LANCZOS)

    sent_w, sent_h = img.size

    # 10. Кодування для передачі моделі — JPEG (менший розмір при схожій якості)
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=92)
    data = out.getvalue()

    sent_size = {"width": sent_w, "height": sent_h, "bytes": len(data)}
    prepare_time = time.monotonic() - t0

    return PreparedImage(
        data=data,
        mime="image/jpeg",
        original=original_size,
        sent=sent_size,
        prepare_time=prepare_time,
    )
