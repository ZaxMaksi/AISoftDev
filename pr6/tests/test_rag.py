"""Автоматизовані тести компонентів RAG-помічника (ПР6).

Перевіряє:
- Валідацію схеми відповідей (app.schema);
- Безпекову фільтрацію та захист від витоку внутрішніх документів (app.retrieval);
- Програмну перевірку цитувань та скидання поля found при фантомних посиланнях (app.rag);
- Дворівневий механізм відмови;
- HTTP-ендпоінти FastAPI (/api/status, /api/ask) та обробку некоректного вводу.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from app import config, index, keyword, rag, retrieval, schema
from app.main import app


class TestRAGComponents(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = index.load(config.INDEX_DIR)
        cls.kw_index = keyword.build(cls.index.chunks)
        cls.client = TestClient(app)
        # Ініціалізуємо startup стан для тестів
        app.state.index = cls.index
        app.state.keyword_index = cls.kw_index

    def test_schema_valid_json(self):
        """Перевірка валідації коректного JSON-відповіді."""
        raw = '{"answer": "Товар можна повернути протягом 14 днів.", "found": true, "sources": [1]}'
        data = schema.validate(raw)
        self.assertEqual(data["answer"], "Товар можна повернути протягом 14 днів.")
        self.assertTrue(data["found"])
        self.assertEqual(data["sources"], [1])

    def test_schema_markdown_codeblock(self):
        """Перевірка коректного зняття markdown-огорож ```json."""
        raw = '```json\n{"answer": "Умови гарантії 12 місяців.", "found": true, "sources": [2]}\n```'
        data = schema.validate(raw)
        self.assertTrue(data["found"])
        self.assertEqual(data["sources"], [2])

    def test_schema_invalid_empty(self):
        """Перевірка відхилення порожнього рядка."""
        with self.assertRaises(schema.SchemaValidationError):
            schema.validate("")

    def test_schema_invalid_structure(self):
        """Перевірка відхилення JSON із відсутніми обов'язковими полями."""
        with self.assertRaises(schema.SchemaValidationError):
            schema.validate('{"answer": "Текст без поля found"}')

    def test_sanitize_filters_hard_security(self):
        """Серверні фільтри повинні примусово забороняти доступ до персоналу та архіву."""
        malicious_filters = {"audience": "персонал", "status": "архів", "product": "Вега S"}
        safe = retrieval.sanitize_filters(malicious_filters)
        self.assertEqual(safe["audience"], "клієнти")
        self.assertEqual(safe["status"], "чинний")
        self.assertEqual(safe["product"], "Вега S")

    def test_citation_validation_phantom_references(self):
        """Фантомні номери джерел повинні видалятися, а found=True без джерел скидатися у False."""
        mock_source = retrieval.Source(
            ref=1,
            chunk=index.Chunk(text="Текст", source="документ.md", metadata={"title": "Тест"}),
            score=0.9,
        )
        # Модель назвала [1, 99], де 99 не існує
        found, sources, text = rag.validate_citations(
            model_found=True,
            model_sources=[1, 99],
            available_sources=[mock_source],
            answer_text="Відповідь з посиланням.",
        )
        self.assertTrue(found)
        self.assertEqual([s.ref for s in sources], [1])

        # Модель назвала лише неіснуюче [99]
        found_bad, sources_bad, _ = rag.validate_citations(
            model_found=True,
            model_sources=[99],
            available_sources=[mock_source],
            answer_text="Відповідь з фантомним посиланням.",
        )
        self.assertFalse(found_bad)
        self.assertEqual(sources_bad, [])

    def test_mechanism_1_refusal_without_llm(self):
        """Механізм 1: Запит поза базою знань має відхилятися без виклику моделі."""
        ans = rag.answer("Чи можна оформити кредит на автомобіль?", self.index, self.kw_index)
        self.assertFalse(ans.found)
        self.assertIn("немає інформації", ans.text.lower())
        self.assertEqual(ans.sources, [])
        self.assertIsNone(ans.usage)
        self.assertEqual(ans.elapsed["generation"], 0.0)

    def test_api_status_endpoint(self):
        """Перевірка GET /api/status."""
        resp = self.client.get("/api/status")
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertTrue(data["ready"])
        self.assertGreater(data["chunks"], 0)

    def test_api_empty_question_validation(self):
        """Перевірка POST /api/ask з порожнім питанням повертає HTTP 400."""
        resp = self.client.post("/api/ask", json={"question": "   "})
        self.assertEqual(resp.status_code, 400)

    def test_api_unknown_filter_validation(self):
        """Перевірка POST /api/ask з невідомим фільтром повертає HTTP 400."""
        resp = self.client.post("/api/ask", json={"question": "Як повернути товар?", "filters": {"unknown": "test"}})
        self.assertEqual(resp.status_code, 400)


if __name__ == "__main__":
    unittest.main()
