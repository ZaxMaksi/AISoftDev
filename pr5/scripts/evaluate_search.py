"""Скрипт оцінювання пошуку (Evaluation): порівняння Semantic Search та BM25.

Запуск:
    python scripts/evaluate_search.py
    або
    python scripts/evaluate_search.py --queries compare/queries.json

Для кожного тестового запиту визначаються:
- Ранг очікуваного документа (1, 2, 3, або >top_k).
- Оцінка першого результату (top score).
- Час виконання пошуку.
- Розрахунок Hit@1 та Hit@3 для обох методів.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

# Забезпечуємо коректне кодування UTF-8 у консолі Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

PR5_ROOT = Path(__file__).resolve().parent.parent
if str(PR5_ROOT) not in sys.path:
    sys.path.insert(0, str(PR5_ROOT))

from app import config, embeddings, index, keyword


def evaluate(
    queries_file: Path = PR5_ROOT / "compare" / "queries.json",
    top_k: int = 10,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Запустити оцінювання набору запитів."""
    if not queries_file.exists():
        print(f"Помилка: файл із запитами не знайдено: {queries_file}")
        sys.exit(1)

    with open(queries_file, "r", encoding="utf-8") as f:
        data = json.load(f)
    queries = data.get("queries", [])
    if not queries:
        print("Помилка: набір запитів порожній.")
        sys.exit(1)

    # Завантажуємо індекси
    try:
        search_idx = index.load(config.INDEX_DIR)
    except Exception as exc:
        print(f"Помилка завантаження індексу: {exc}")
        print("Спочатку збудуйте індекс: python scripts/build_index.py")
        sys.exit(1)

    kw_idx = keyword.build(search_idx.chunks)

    # Прогрів моделі ембедінгів, щоб час першого запиту не спотворював статистику
    _ = embeddings.embed_query("тестовий прогрів")

    results: list[dict[str, Any]] = []

    sem_hit1 = 0
    sem_hit3 = 0
    kw_hit1 = 0
    kw_hit3 = 0
    eval_count = 0

    print("=" * 110)
    print(f"{'#':<3} | {'Запит':<35} | {'Очікуваний документ':<28} | {'Sem Rank':<8} | {'Sem Sc':<6} | {'BM25 R':<6} | {'BM25 Sc':<7}")
    print("-" * 110)

    for item in queries:
        qid = item.get("id")
        q_text = item.get("query", "").strip()
        expected = item.get("expected_document", "").strip()
        q_filters = item.get("filter")

        # 1. Semantic search
        t0 = time.perf_counter()
        q_vec = embeddings.embed_query(q_text)
        sem_hits = index.search(
            search_idx,
            query_vector=q_vec,
            top_k=top_k,
            filters=q_filters,
            threshold=threshold,
        )
        sem_time = (time.perf_counter() - t0) * 1000

        sem_rank = None
        sem_top_score = sem_hits[0].score if sem_hits else 0.0
        for rank, hit in enumerate(sem_hits, start=1):
            if expected and hit.chunk.source == expected:
                sem_rank = rank
                break

        # 2. BM25 keyword search
        t0 = time.perf_counter()
        kw_hits = keyword.search(
            kw_idx,
            query=q_text,
            top_k=top_k,
            filters=q_filters,
        )
        kw_time = (time.perf_counter() - t0) * 1000

        kw_rank = None
        kw_top_score = kw_hits[0].score if kw_hits else 0.0
        for rank, hit in enumerate(kw_hits, start=1):
            if expected and hit.chunk.source == expected:
                kw_rank = rank
                break

        # Метрики Hit@1 та Hit@3 рахуються для запитів, що мають очікувану відповідь
        if expected:
            eval_count += 1
            if sem_rank == 1:
                sem_hit1 += 1
            if sem_rank and sem_rank <= 3:
                sem_hit3 += 1

            if kw_rank == 1:
                kw_hit1 += 1
            if kw_rank and kw_rank <= 3:
                kw_hit3 += 1

        sem_rank_str = str(sem_rank) if sem_rank else "-"
        kw_rank_str = str(kw_rank) if kw_rank else "-"

        # Друк рядка таблиці
        q_disp = (q_text[:32] + "...") if len(q_text) > 35 else q_text
        exp_disp = (expected[:25] + "...") if len(expected) > 28 else (expected or "[поза колекцією]")
        print(f"{qid:<3} | {q_disp:<35} | {exp_disp:<28} | {sem_rank_str:<8} | {sem_top_score:<6.3f} | {kw_rank_str:<6} | {kw_top_score:<7.2f}")

        results.append({
            "id": qid,
            "type": item.get("type", ""),
            "query": q_text,
            "expected_document": expected,
            "filter": q_filters,
            "semantic_rank": sem_rank,
            "semantic_top_score": round(sem_top_score, 4),
            "semantic_search_time_ms": round(sem_time, 2),
            "bm25_rank": kw_rank,
            "bm25_top_score": round(kw_top_score, 4),
            "bm25_search_time_ms": round(kw_time, 2),
        })

    print("=" * 110)
    print("\n========================")
    print("SEMANTIC SEARCH")
    print("========================")
    sem_h1_pct = (sem_hit1 / eval_count * 100) if eval_count else 0.0
    sem_h3_pct = (sem_hit3 / eval_count * 100) if eval_count else 0.0
    print(f"Hit@1: {sem_hit1}/{eval_count} = {sem_h1_pct:.1f}%")
    print(f"Hit@3: {sem_hit3}/{eval_count} = {sem_h3_pct:.1f}%")

    print("\n========================")
    print("BM25")
    print("========================")
    kw_h1_pct = (kw_hit1 / eval_count * 100) if eval_count else 0.0
    kw_h3_pct = (kw_hit3 / eval_count * 100) if eval_count else 0.0
    print(f"Hit@1: {kw_hit1}/{eval_count} = {kw_h1_pct:.1f}%")
    print(f"Hit@3: {kw_hit3}/{eval_count} = {kw_h3_pct:.1f}%")

    # Перевірка запиту поза колекцією
    out_of_coll = [r for r in results if not r["expected_document"]]
    if out_of_coll:
        print("\n========================")
        print("ЗАПИТ ПОЗА КОЛЕКЦІЄЮ")
        print("========================")
        for r in out_of_coll:
            print(f"Запит: '{r['query']}'")
            print(f"Найкраща оцінка Semantic: {r['semantic_top_score']:.3f}")
            print(f"Найкраща оцінка BM25: {r['bm25_top_score']:.2f}")
            thresh_val = config.SEARCH_THRESHOLD or 0.70
            rejected = r["semantic_top_score"] < thresh_val
            print(f"Поріг {thresh_val:.2f} відсікає цей результат: {'ТАК (успішно)' if rejected else 'НІ (хибне спрацювання)'}")

    return {
        "results": results,
        "eval_count": eval_count,
        "semantic": {
            "hit1": sem_hit1,
            "hit3": sem_hit3,
            "hit1_pct": sem_h1_pct,
            "hit3_pct": sem_h3_pct,
        },
        "bm25": {
            "hit1": kw_hit1,
            "hit3": kw_hit3,
            "hit1_pct": kw_h1_pct,
            "hit3_pct": kw_h3_pct,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Оцінювання якості пошуку: Semantic vs BM25")
    parser.add_argument(
        "--queries",
        type=Path,
        default=PR5_ROOT / "compare" / "queries.json",
        help="Шлях до файлу queries.json",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="Глибина видачі top_k для обчислення рангів",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Поріг схожості (за замовчуванням без порога для перевірки rank)",
    )
    args = parser.parse_args()
    evaluate(queries_file=args.queries, top_k=args.top_k, threshold=args.threshold)


if __name__ == "__main__":
    main()
