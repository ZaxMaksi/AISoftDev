"""Скрипт побудови та збереження пошукового індексу.

Запуск:
    python scripts/build_index.py
або
    python ingest.py

Кроки:
1. Знаходить усі документи в docs/.
2. Розбирає метадані front-matter.
3. Розбиває документи на фрагменти (chunks) із контекстом розділів.
4. Обчислює вектори через sentence-transformers.
5. Нормалізує вектори (L2 норма).
6. Зберігає вектори у vectors.npy та метадані/тексти у chunks.json.
7. Зберігає назву моделі та параметри розбиття.
8. Виводить підсумкову статистику.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path
import time

# Забезпечуємо коректне кодування UTF-8 у консолі Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Додаємо кореневу папку pr5 до sys.path
PR5_ROOT = Path(__file__).resolve().parent.parent
if str(PR5_ROOT) not in sys.path:
    sys.path.insert(0, str(PR5_ROOT))

from app import config, documents, embeddings, index


def build_index() -> int:
    """Виконати повний цикл побудови та збереження індексу."""
    started = time.perf_counter()
    print("=" * 60)
    print("ПОБУДОВА ПОШУКОВОГО ІНДЕКСУ БАЗИ ЗНАНЬ «СУЗІР'Я»")
    print("=" * 60)

    # 1. Завантаження документів
    docs = documents.load_documents(config.DOCS_DIR)
    print(f"Документів знайдено: {len(docs)}")
    if not docs:
        print("Помилка: не знайдено документів для індексації.")
        return 1

    # 2. Розбиття на фрагменти
    chunks = documents.load_chunks(
        docs_dir=config.DOCS_DIR,
        chunk_size=config.CHUNK_SIZE,
        overlap=config.CHUNK_OVERLAP,
    )
    print(f"Фрагментів створено: {len(chunks)}")
    lengths = [len(c.text) for c in chunks]
    if lengths:
        print(f"Довжина фрагментів (символів): min {min(lengths)}, "
              f"avg {sum(lengths) // len(lengths)}, max {max(lengths)}")

    per_doc = Counter(c.source for c in chunks)
    print("Фрагментів на документ:")
    for source, count in sorted(per_doc.items()):
        print(f"  {count:2d}  {source}")

    # 3. Обчислення ембедінгів
    model_name = config.EMBEDDING_MODEL
    print(f"\nМодель ембедінгів: {model_name}")
    print(f"Параметри chunking: CHUNK_SIZE={config.CHUNK_SIZE}, CHUNK_OVERLAP={config.CHUNK_OVERLAP}")
    print("Обчислення ембедінгів для фрагментів...")

    t_emb = time.perf_counter()
    texts_to_embed = [c.get_embedding_text() for c in chunks]
    vectors = embeddings.embed_passages(texts_to_embed)
    emb_time = time.perf_counter() - t_emb

    dimensions = vectors.shape[1]
    print(f"Обчислено векторів: {vectors.shape[0]} × {dimensions} за {emb_time:.2f} с")

    # 4. Складання та збереження індексу
    extra_params = {
        "chunk_size": config.CHUNK_SIZE,
        "overlap": config.CHUNK_OVERLAP,
        "dimensions": dimensions,
    }
    built_idx = index.build(chunks, vectors, model_name=model_name, extra=extra_params)
    index.save(built_idx, path=config.INDEX_DIR)

    total_time = time.perf_counter() - started
    print("-" * 60)
    print(f"Documents: {len(docs)}")
    print(f"Chunks: {len(chunks)}")
    print(f"Embedding model: {model_name}")
    print(f"Dimensions: {dimensions}")
    print(f"Index directory: {config.INDEX_DIR}")
    print(f"Index successfully built in {total_time:.2f}s.")
    print("=" * 60)
    return 0


def main() -> int:
    return build_index()


if __name__ == "__main__":
    sys.exit(main())
