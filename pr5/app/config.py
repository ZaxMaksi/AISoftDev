"""Конфігурація застосунку та пошуку.

Усі параметри читаються зі змінних середовища (.env) із безпечними значеннями
за замовчуванням.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Завантажуємо .env з кореня pr5
load_dotenv(BASE_DIR / ".env")

# Шляхи до папок
DOCS_DIR = BASE_DIR / os.getenv("DOCS_DIR", "docs")
INDEX_DIR = BASE_DIR / os.getenv("INDEX_DIR", "index")

# Назва локальної моделі ембедінгів
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

# Параметри пошуку
TOP_K = int(os.getenv("TOP_K") or os.getenv("SEARCH_TOP_K") or "5")

_thresh_raw = os.getenv("SEARCH_THRESHOLD") or os.getenv("SIMILARITY_THRESHOLD")
SEARCH_THRESHOLD: float | None = float(_thresh_raw.strip()) if _thresh_raw and _thresh_raw.strip() else 0.70

# Параметри розбиття на фрагменти (chunking)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Дозволені поля фільтрів
ALLOWED_FILTERS = {"category", "product", "audience", "status", "date", "updated", "section", "heading"}
