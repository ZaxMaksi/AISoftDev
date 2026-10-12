"""Конвеєр відповіді за базою знань: від питання до відповіді з джерелами.

Повна послідовність кроків:
1. Пошук та відбір фрагментів за допомогою retrieval.retrieve (з жорсткими фільтрами безпеки
   та порогом схожості).
2. Замір часу пошуку (retrieval time).
3. Механізм 1 (відмова до моделі): якщо після фільтрації та порогу не знайдено жодного фрагмента,
   повертається детермінована безпечна відповідь без звернення до LLM.
4. Збирання контексту з нумерацією [1..N] та дотриманням бюджету токенів (retrieval.build_context).
5. Виклик мовної моделі через llm.ask та замір часу генерації (generation time).
6. Програмна перевірка (валідація змісту та посилань):
   - Чи всі номери джерел з відповіді існують серед реально переданих моделі [1..N];
   - Якщо found=True, чи вказано хоча б одне валідне джерело;
   - Якщо модель заявила found=True, але не надала валідних джерел (або всі номери фантомні),
     поле found скидається у False, а відповідь коригується;
   - Формування списку джерел Answer.sources (лише підтверджуючі джерела);
   - Збереження повного списку переданих фрагментів в Answer.retrieved для діагностики помилок;
   - Перевірка на відсутність витоку внутрішніх документів.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from . import llm, retrieval
from .index import Hit, SearchIndex
from .keyword import KeywordIndex
from .retrieval import Source

logger = logging.getLogger(__name__)

NO_INFO_FALLBACK = "На жаль, у базі знань магазину «Сузірʼя» немає інформації для відповіді на це питання."


@dataclass
class Answer:
    """Результат роботи RAG-конвеєра для відображення на веб-рівні та оцінювання."""

    text: str
    found: bool
    sources: list[Source] = field(default_factory=list)
    retrieved: list[Hit] = field(default_factory=list)
    model: str | None = None
    elapsed: dict[str, float] = field(default_factory=dict)
    usage: dict[str, int] | None = None


def validate_citations(
    model_found: bool,
    model_sources: list[int],
    available_sources: list[Source],
    answer_text: str,
) -> tuple[bool, list[Source], str]:
    """Програмна перевірка обґрунтованості відповіді та коректності посилань.

    Правила:
    1. Жодне джерело не може мати номер поза діапазоном доступних [1..len(available_sources)].
    2. Якщо found=True, обов'язково має бути хоча б одне дійсне підтверджуюче джерело.
    3. Якщо названо неіснуючі номери — вони відкидаються.
    4. Якщо після фільтрації неіснуючих номерів список джерел спорожнів, але модель стверджувала found=True:
       відповідь вважається необґрунтованою (галюцинація цитування), found скидається на False.
    """
    valid_source_map = {s.ref: s for s in available_sources}
    valid_selected_sources: list[Source] = []

    # Фільтрація фантомних посилань
    for ref_num in model_sources:
        if ref_num in valid_source_map:
            if valid_source_map[ref_num] not in valid_selected_sources:
                valid_selected_sources.append(valid_source_map[ref_num])
        else:
            logger.warning("Модель послалася на неіснуюче джерело [%d]", ref_num)

    final_found = model_found

    # Перевірка: якщо стверджується found=True, але джерел немає
    if final_found and not valid_selected_sources:
        # Якщо в тексті є прямі слова про відсутність інформації
        lower_ans = answer_text.lower()
        if "немає інформації" in lower_ans or "не знайдено" in lower_ans or "відсутня" in lower_ans:
            final_found = False
        else:
            logger.warning(
                "Відповідь позначена як знайдена, але не має валідних посилань на надані джерела. "
                "Скидання found -> False для захисту від галюцинацій."
            )
            final_found = False

    # Якщо found=False, список джерел очищується
    if not final_found:
        valid_selected_sources = []

    return final_found, valid_selected_sources, answer_text


def answer(
    question: str,
    index: SearchIndex,
    keyword_index: KeywordIndex | None = None,
    filters: dict | None = None,
) -> Answer:
    """Виконати повний цикл RAG-конвеєра: знайти -> зібрати -> згенерувати -> перевірити."""
    q_clean = question.strip() if question else ""
    if not q_clean:
        return Answer(
            text="Будь ласка, введіть ваше запитання.",
            found=False,
            sources=[],
            retrieved=[],
            model=None,
            elapsed={"retrieval": 0.0, "generation": 0.0},
            usage=None,
        )

    # 1. Пошук та відбір фрагментів
    t_retrieval_start = time.perf_counter()
    hits = retrieval.retrieve(
        query=q_clean,
        index=index,
        keyword_index=keyword_index,
        filters=filters,
    )
    t_retrieval = time.perf_counter() - t_retrieval_start

    # МЕХАНІЗМ 1 (до виклику моделі):
    # Якщо пошук не дав жодного влучення вище за поріг схожості
    if not hits:
        return Answer(
            text=NO_INFO_FALLBACK,
            found=False,
            sources=[],
            retrieved=[],
            model=None,
            elapsed={"retrieval": round(t_retrieval, 4), "generation": 0.0},
            usage=None,
        )

    # 2. Збирання контексту
    context_text, available_sources = retrieval.build_context(hits)

    if not context_text or not available_sources:
        return Answer(
            text=NO_INFO_FALLBACK,
            found=False,
            sources=[],
            retrieved=hits,
            model=None,
            elapsed={"retrieval": round(t_retrieval, 4), "generation": 0.0},
            usage=None,
        )

    # 3. Виклик мовної моделі
    llm_result = llm.ask(question=q_clean, context=context_text)

    # 4. Програмна перевірка та валідація посилань
    verified_found, verified_sources, final_text = validate_citations(
        model_found=bool(llm_result["found"]),
        model_sources=llm_result["sources"],
        available_sources=available_sources,
        answer_text=llm_result["answer"],
    )

    return Answer(
        text=final_text,
        found=verified_found,
        sources=verified_sources,
        retrieved=hits,
        model=llm_result["model"],
        elapsed={
            "retrieval": round(t_retrieval, 4),
            "generation": round(llm_result["elapsed"], 4),
        },
        usage=llm_result["usage"],
    )
