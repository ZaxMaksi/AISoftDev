"""Аналіз журналу: повільні, невдалі й повторні виклики, витрати, кеш.

Запуск із папки pr11:

    python eval/analyze.py                       # logs/app.jsonl, усі розділи
    python eval/analyze.py logs/app.jsonl --since 2026-10-23 --section slow
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load(path: Path, since: str | None = None) -> list[dict]:
    """Прочитати JSON lines. Рядки, які не розбираються, рахуються й
    пропускаються; `since` — дата або час у форматі ISO, від якого брати
    записи (порівнюється з полем `ts` як рядок)."""
    records: list[dict] = []
    broken = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError:
                broken += 1
                continue
            if since and str(record.get("ts", "")) < since:
                continue
            records.append(record)
    if broken:
        print(f"(пропущено рядків, що не розбираються: {broken})", file=sys.stderr)
    return records


def percentile(values: list[float], p: float) -> float | None:
    """p-й процентиль (0 ≤ p ≤ 1) за найближчим рангом; `None` для
    порожнього списку."""
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))
    return ordered[index]


def group(records: list[dict], key: str) -> dict:
    """Згрупувати записи за значенням поля."""
    groups: dict = defaultdict(list)
    for record in records:
        groups[record.get(key)].append(record)
    return groups


def summary(records: list[dict]) -> None:
    """Що взагалі є в журналі: скільки записів, за який час, яких подій і
    рівнів скільки."""
    if not records:
        print("записів немає")
        return
    stamps = [r.get("ts", "") for r in records if "ts" in r]
    if stamps:
        print(f"записів: {len(records)}; від {min(stamps)} до {max(stamps)}")
    else:
        print(f"записів: {len(records)}")
    print("рівні:", dict(Counter(r.get("level") for r in records)))
    print("події:")
    for event, count in Counter(r.get("event") for r in records).most_common():
        print(f"  {count:>6}  {event}")


def slow(records: list[dict]) -> None:
    """Які виклики найповільніші й що таке «повільно» в цих числах?"""
    # Аналізуємо спроби звертання до LLM (виключаючи взяті з кешу)
    calls = [r for r in records if r.get("event") == "llm.attempt" and not r.get("cached")]
    if not calls:
        print("немає звертань до провайдера для аналізу")
        return

    durations = [float(r["duration_ms"]) for r in calls if "duration_ms" in r]
    first_tokens = [float(r["first_token_ms"]) for r in calls if r.get("first_token_ms") is not None]
    queues = [float(r["queue_ms"]) for r in calls if r.get("queue_ms") is not None]

    print(f"всього викликів провайдера: {len(calls)}")
    print(f"загальна тривалість (мс): медіана {percentile(durations, 0.5):.1f}, "
          f"p95 {percentile(durations, 0.95):.1f}, max {max(durations, default=0):.1f}")
    if first_tokens:
        print(f"до першого фрагмента (мс): медіана {percentile(first_tokens, 0.5):.1f}, "
              f"p95 {percentile(first_tokens, 0.95):.1f}, max {max(first_tokens, default=0):.1f}")
    if queues:
        print(f"очікування в черзі (мс): медіана {percentile(queues, 0.5):.1f}, "
              f"p95 {percentile(queues, 0.95):.1f}, max {max(queues, default=0):.1f}")

    # За провайдерами та моделями
    by_model = defaultdict(list)
    for r in calls:
        key = f"{r.get('provider')}/{r.get('model')}"
        by_model[key].append(float(r.get("duration_ms", 0)))

    print("\nза провайдером/моделлю:")
    for key, vals in by_model.items():
        print(f"  {key:<30} викликів: {len(vals):>3}, p50: {percentile(vals, 0.5):>6.1f} мс, p95: {percentile(vals, 0.95):>6.1f} мс")

    # Топ найповільніших викликів
    slowest = sorted(calls, key=lambda r: float(r.get("duration_ms", 0)), reverse=True)[:10]
    print(f"\nнайповільніші виклики (топ {len(slowest)}):")
    print(f"{'#':>2} {'request_id':<12} {'draft_id':<9} {'пров./модель':<25} {'спроба':>6} {'черга':>8} {'1-й токен':>10} {'разом':>9} {'статус':<7}")
    for idx, r in enumerate(slowest, 1):
        pm = f"{r.get('provider')}/{r.get('model')}"[:24]
        q = f"{r.get('queue_ms', 0):.0f}мс" if r.get('queue_ms') is not None else "—"
        f = f"{r.get('first_token_ms', 0):.0f}мс" if r.get('first_token_ms') is not None else "—"
        tot = f"{r.get('duration_ms', 0):.0f}мс"
        print(f"{idx:>2} {str(r.get('request_id')):<12} {str(r.get('draft_id')):<9} {pm:<25} {r.get('attempt', 1):>6} {q:>8} {f:>10} {tot:>9} {r.get('status'):<7}")


def failed(records: list[dict]) -> None:
    """Що й чому не вдалося?"""
    attempts = [r for r in records if r.get("event") == "llm.attempt"]
    errors = [r for r in attempts if r.get("status") == "error"]
    failed_drafts = [r for r in records if r.get("event") == "draft.failed"]
    done_drafts = [r for r in records if r.get("event") == "draft.done"]

    print(f"всього спроб звертань: {len(attempts)}")
    print(f"невдалих спроб: {len(errors)} ({len(errors)/len(attempts)*100:.1f} %)" if attempts else "спроб немає")
    print(f"остаточних збоїв чернеток: {len(failed_drafts)}")

    if errors:
        print("\nзбої за видами (kind):")
        for kind, count in Counter(r.get("kind") or "unknown" for r in errors).most_common():
            print(f"  {count:>4}  {kind}")

    # Аналіз ефективності повторів (retries)
    # Згрупуємо спроби за draft_id
    by_draft = defaultdict(list)
    for r in attempts:
        d_id = r.get("draft_id")
        if d_id:
            by_draft[d_id].append(r)

    drafts_with_error_on_1st = [
        d_id for d_id, atts in by_draft.items()
        if len(atts) > 1 and atts[0].get("status") == "error"
    ]
    recovered = [
        d_id for d_id in drafts_with_error_on_1st
        if any(a.get("status") == "success" for a in by_draft[d_id])
    ]

    print(f"\nефективність повторів (retries):")
    print(f"  чернеток із невдалою першою спробою: {len(drafts_with_error_on_1st)}")
    if drafts_with_error_on_1st:
        print(f"  з них врятовано повторними спробами: {len(recovered)} ({len(recovered)/len(drafts_with_error_on_1st)*100:.1f} %)")

    # Вартість невдалих спроб
    failed_cost = sum(float(r.get("cost") or 0.0) for r in errors)
    print(f"\nвартість невдалих спроб: {failed_cost:.6f} $")


def repeated(records: list[dict]) -> None:
    """Що ми робили двічі?"""
    created_drafts = [r for r in records if r.get("event") == "draft.created"]
    sample_counts = Counter(r.get("sample_id") for r in created_drafts if r.get("sample_id"))

    print(f"всього створено чернеток: {len(created_drafts)}")
    print(f"унікальних звернень у чернетках: {len(sample_counts)}")

    repeats = {sid: cnt for sid, cnt in sample_counts.items() if cnt > 1}
    print(f"звернень, замовлених більше 1 разу: {len(repeats)}")
    for sid, cnt in repeats.items():
        print(f"  звернення {sid}: замовлено {cnt} разів")

    # Чернетки з кількома спробами
    attempts = [r for r in records if r.get("event") == "llm.attempt"]
    draft_attempts = Counter(r.get("draft_id") for r in attempts if r.get("draft_id"))
    multi_attempts = {did: cnt for did, cnt in draft_attempts.items() if cnt > 1}
    print(f"\nчернеток із кількома спробами звернення до моделі: {len(multi_attempts)}")
    for did, cnt in multi_attempts.items():
        print(f"  чернетка {did}: {cnt} спроб")


def spend(records: list[dict]) -> None:
    """Скільки це коштувало?"""
    attempts = [r for r in records if r.get("event") == "llm.attempt"]
    if not attempts:
        print("немає даних про витрати")
        return

    costs = [float(r.get("cost") or 0.0) for r in attempts]
    total_cost = sum(costs)
    tokens_in = sum(int(r.get("tokens_in") or 0) for r in attempts)
    tokens_out = sum(int(r.get("tokens_out") or 0) for r in attempts)
    tokens_cached = sum(int(r.get("tokens_cached") or 0) for r in attempts)

    estimated_calls = [r for r in attempts if r.get("cost_estimated")]
    actual_calls = [r for r in attempts if not r.get("cost_estimated") and not r.get("cached")]
    cached_calls = [r for r in attempts if r.get("cached")]

    print(f"загальна вартість усіх викликів: {total_cost:.6f} $")
    print(f"токенів: вхід {tokens_in:,}, вихід {tokens_out:,}, кешовано провайдером {tokens_cached:,}")
    print(f"структура звертань:")
    print(f"  розраховано за фактичним usage: {len(actual_calls)} викликів ({sum(float(r.get('cost') or 0.0) for r in actual_calls):.6f} $)")
    print(f"  оцінено без usage: {len(estimated_calls)} викликів ({sum(float(r.get('cost') or 0.0) for r in estimated_calls):.6f} $)")
    print(f"  обслуговано з кешу (0.00 $): {len(cached_calls)} викликів")

    # Розподіл за моделлю
    by_model = defaultdict(float)
    for r in attempts:
        m = f"{r.get('provider')}/{r.get('model')}"
        by_model[m] += float(r.get("cost") or 0.0)

    print("\nвитрати за моделлю:")
    for m, c in sorted(by_model.items(), key=lambda x: x[1], reverse=True):
        print(f"  {m:<30} {c:.6f} $")


def cache_report(records: list[dict]) -> None:
    """Що дав кеш?"""
    attempts = [r for r in records if r.get("event") == "llm.attempt"]
    created = [r for r in records if r.get("event") == "draft.created"]

    hits = sum(1 for r in attempts if r.get("cached"))
    misses = sum(1 for r in attempts if not r.get("cached"))
    bypassed = sum(1 for r in created if r.get("fresh"))
    total = hits + misses

    print(f"влучань у кеш (hits):     {hits:>4}")
    print(f"промахів повз кеш (misses): {misses:>4}")
    print(f"обходів кешу (bypassed):   {bypassed:>4}")
    if total > 0:
        print(f"частка влучань (hit rate): {hits / total * 100:.1f} %")

    # Економія ресурсів
    uncached_attempts = [r for r in attempts if not r.get("cached") and r.get("status") == "success"]
    if uncached_attempts and hits > 0:
        avg_cost = sum(float(r.get("cost") or 0.0) for r in uncached_attempts) / len(uncached_attempts)
        avg_tokens = sum(int(r.get("tokens_in") or 0) + int(r.get("tokens_out") or 0) for r in uncached_attempts) / len(uncached_attempts)
        avg_time = sum(float(r.get("duration_ms") or 0.0) for r in uncached_attempts) / len(uncached_attempts)

        saved_cost = hits * avg_cost
        saved_tokens = hits * avg_tokens
        saved_time_sec = (hits * avg_time) / 1000.0

        print(f"\nоцінка заощаджених ресурсів завдяки кешуванню:")
        print(f"  заощаджено звертань до моделі: {hits}")
        print(f"  заощаджено коштів: ~{saved_cost:.6f} $")
        print(f"  заощаджено токенів: ~{int(saved_tokens):,}")
        print(f"  заощаджено часу генерації: ~{saved_time_sec:.2f} с")


SECTIONS = {
    "summary": summary,
    "slow": slow,
    "failed": failed,
    "repeated": repeated,
    "spend": spend,
    "cache": cache_report,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log", nargs="?", default="logs/app.jsonl")
    parser.add_argument("--since", help="брати записи від цієї дати/часу (ISO), наприклад 2026-10-23")
    parser.add_argument("--section", choices=["all", *SECTIONS], default="all")
    args = parser.parse_args()

    path = Path(args.log)
    if not path.exists():
        print(f"журналу немає: {path}", file=sys.stderr)
        return 1
    records = load(path, args.since)

    names = list(SECTIONS) if args.section == "all" else [args.section]
    for name in names:
        print(f"\n=== {name}")
        try:
            SECTIONS[name](records)
        except Exception as exc:
            print(f"(помилка виконання {name}: {exc})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
