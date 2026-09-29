"""Модуль ембедінгів: єдине місце застосунку, яке знає, яка модель
перетворює текст на вектор і як її викликати.

Індекс (`app/index.py`) і веб-рівень (`app/main.py`) отримують звідси
готові вектори й не знають внутрішнього влаштування SentenceTransformer.
"""

from __future__ import annotations
import threading
from typing import Optional
import numpy as np
from sentence_transformers import SentenceTransformer
from . import config

MODEL_NAME = config.EMBEDDING_MODEL

class EmbeddingService:
    """Сервіс для локальної побудови ембедінгів через sentence-transformers.

    Підтримує:
    - Кешування єдиного екземпляра моделі (singleton/lazy loading).
    - Префікси для сімейства e5 (passage: / query:).
    - Нормалізацію векторів (unit length) для швидкого обчислення cosine similarity через dot product.
    - Пакетне кодування (batch encoding) для високої швидкодії.
    """

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or MODEL_NAME
        self._model: Optional[SentenceTransformer] = None
        self._lock = threading.Lock()

        # Визначаємо префікси за карткою моделі
        if "e5" in self.model_name.lower():
            self.passage_prefix = "passage: "
            self.query_prefix = "query: "
        else:
            self.passage_prefix = ""
            self.query_prefix = ""

    @property
    def model(self) -> SentenceTransformer:
        """Повернути або ліниво завантажити модель один раз."""
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = SentenceTransformer(self.model_name)
        return self._model

    def embed_passages(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Перетворити список текстів фрагментів на нормалізовані вектори."""
        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        prefixed = [f"{self.passage_prefix}{text}" for text in texts]
        vectors = self.model.encode(
            prefixed,
            normalize_embeddings=True,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        """Перетворити запит користувача на нормалізований вектор тієї самої розмірності."""
        prefixed = f"{self.query_prefix}{text}"
        vectors = self.model.encode(
            [prefixed],
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(vectors[0], dtype=np.float32)


# Глобальний екземпляр сервісу (singleton)
_embedding_service: EmbeddingService | None = None
_service_lock = threading.Lock()


def get_service(model_name: str | None = None) -> EmbeddingService:
    """Отримати глобальний екземпляр EmbeddingService."""
    global _embedding_service
    if _embedding_service is None or (model_name and _embedding_service.model_name != model_name):
        with _service_lock:
            if _embedding_service is None or (model_name and _embedding_service.model_name != model_name):
                _embedding_service = EmbeddingService(model_name=model_name or MODEL_NAME)
    return _embedding_service


def get_model() -> SentenceTransformer:
    """Повернути готову до роботи модель із глобального сервісу."""
    return get_service().model


def embed_passages(texts: list[str], batch_size: int = 32) -> np.ndarray:
    """Перетворити тексти фрагментів на нормалізовані вектори."""
    return get_service().embed_passages(texts, batch_size=batch_size)


def embed_query(text: str) -> np.ndarray:
    """Перетворити запит користувача на нормалізований вектор."""
    return get_service().embed_query(text)
