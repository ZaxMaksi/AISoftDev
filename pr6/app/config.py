"""Конфігурація застосунку, пошуку та RAG-помічника (ПР6).

Усі параметри читаються зі змінних середовища (.env) із безпечними значеннями
за замовчуванням.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# Завантажуємо .env з кореня pr6
load_dotenv(BASE_DIR / ".env")

# Шляхи до папок
DOCS_DIR = BASE_DIR / os.getenv("DOCS_DIR", "docs")
INDEX_DIR = BASE_DIR / os.getenv("INDEX_DIR", "index")

# Назва локальної моделі ембедінгів
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

# Параметри розбиття на фрагменти (chunking)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# Параметри пошуку
TOP_K = int(os.getenv("SEARCH_TOP_K") or os.getenv("TOP_K") or "8")

_thresh_raw = os.getenv("SIMILARITY_THRESHOLD") or os.getenv("SEARCH_THRESHOLD")
SIMILARITY_THRESHOLD: float | None = float(_thresh_raw.strip()) if _thresh_raw and _thresh_raw.strip() else 0.83

# Параметри відбору та збирання контексту
RAG_CONTEXT_CHUNKS = int(os.getenv("RAG_CONTEXT_CHUNKS", "4"))
RAG_CONTEXT_BUDGET = int(os.getenv("RAG_CONTEXT_BUDGET", "1500"))

# Налаштування мовної моделі
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
LLM_MODEL = os.getenv("LLM_MODEL", "gemini-3.1-flash-lite")
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "700"))
LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30.0"))

# Дозволені поля фільтрів для користувача
ALLOWED_FILTERS = {"product", "category", "heading", "section"}
