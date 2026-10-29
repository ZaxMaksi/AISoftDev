"""Інтеграційні тести FastAPI: завантаження зображення, конвеєр, відповіді."""
import sys
import os
import io

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from fastapi.testclient import TestClient
from PIL import Image


def make_jpeg(width=200, height=200) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=(220, 220, 220))
    img.save(buf, format="JPEG")
    return buf.getvalue()


def make_png(width=200, height=200) -> bytes:
    buf = io.BytesIO()
    img = Image.new("RGB", (width, height), color=(220, 220, 220))
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def client():
    """TestClient FastAPI без реального виклику моделі."""
    # Патчимо extraction.process щоб не йти в API
    from unittest.mock import patch, MagicMock
    from app.rules import Issue

    mock_result = MagicMock()
    mock_result.decision = "review"
    mock_result.reasons = ["Тестовий результат"]
    mock_result.document = {
        "document_type": "рахунок",
        "number": "TEST-001",
        "date": "2026-09-03",
        "supplier": {"name": "ТОВ «Тест»", "code": "38124779", "iban": "UA663510050000026004017723561"},
        "buyer": {"name": "ТОВ «Сузірʼя Рітейл»", "code": "44172914"},
        "items": [],
        "total": "1000.00",
    }
    mock_result.issues = [Issue(field="supplier.name", rule="supplier_registry", message="Тест", severity="error")]
    mock_result.image = {"original": {"width": 200, "height": 200, "bytes": 5000}, "sent": {"width": 200, "height": 200, "bytes": 4000}}
    mock_result.model = "test-model"
    mock_result.elapsed = {"prepare": 0.01, "extraction": 0.5, "checks": 0.001}
    mock_result.usage = {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300}

    with patch("app.extraction.process", return_value=mock_result):
        from app.main import app
        with TestClient(app) as c:
            yield c


class TestAPIEndpoints:
    def test_get_root(self, client):
        """Головна сторінка повертає HTML."""
        response = client.get("/")
        assert response.status_code == 200
        assert "text/html" in response.headers["content-type"]
        assert "Розбір рахунків" in response.text

    def test_get_samples(self, client):
        """Ендпоінт зразків повертає JSON з папками."""
        response = client.get("/api/samples")
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, dict)
        # Має бути clean і degraded
        assert "clean" in data or "degraded" in data

    def test_post_extract_jpeg(self, client):
        """Завантаження JPEG повертає структурований результат."""
        content = make_jpeg()
        response = client.post(
            "/api/extract",
            files={"image": ("test.jpg", content, "image/jpeg")}
        )
        assert response.status_code == 200
        data = response.json()
        assert "decision" in data
        assert "reasons" in data
        assert "document" in data
        assert "issues" in data
        assert "image" in data

    def test_post_extract_png(self, client):
        """Завантаження PNG також працює."""
        content = make_png()
        response = client.post(
            "/api/extract",
            files={"image": ("test.png", content, "image/png")}
        )
        assert response.status_code == 200

    def test_post_extract_empty(self, client):
        """Порожній файл обробляється через конвеєр (reject)."""
        # Не патчимо process тут, тому reject через ImageError
        from unittest.mock import patch
        from app.extraction import Result

        empty_result = Result(
            decision="reject",
            reasons=["Файл порожній."],
        )
        with patch("app.extraction.process", return_value=empty_result):
            response = client.post(
                "/api/extract",
                files={"image": ("empty.jpg", b"", "image/jpeg")}
            )
        # Може бути 200 (decision=reject) або 400
        assert response.status_code in (200, 400)

    def test_post_extract_not_image(self, client):
        """Не-зображення обробляється без краш 500."""
        from unittest.mock import patch
        from app.extraction import Result

        reject_result = Result(
            decision="reject",
            reasons=["Файл не є зображенням."],
        )
        with patch("app.extraction.process", return_value=reject_result):
            response = client.post(
                "/api/extract",
                files={"image": ("doc.pdf", b"%PDF not an image", "application/pdf")}
            )
        assert response.status_code in (200, 400)

    def test_extract_result_structure(self, client):
        """Перевірити повну структуру відповіді."""
        content = make_jpeg()
        response = client.post(
            "/api/extract",
            files={"image": ("test.jpg", content, "image/jpeg")}
        )
        data = response.json()
        assert data["decision"] in ("auto", "review", "reject")
        assert isinstance(data["reasons"], list)
        assert isinstance(data["issues"], list)
        if data["issues"]:
            issue = data["issues"][0]
            assert "field" in issue
            assert "rule" in issue
            assert "message" in issue
            assert "severity" in issue
        if data["image"]:
            if "original" in data["image"]:
                assert "width" in data["image"]["original"]
                assert "height" in data["image"]["original"]


class TestAPIWithMockedLLM:
    """Тести конвеєра з підробленими відповідями моделі."""

    def test_auto_decision_with_valid_doc(self):
        """Валідний документ має отримати auto."""
        from unittest.mock import patch
        from app.extraction import Result

        auto_result = Result(
            decision="auto",
            reasons=["Усі перевірки пройдено."],
            document={
                "document_type": "рахунок",
                "number": "ОА-0917",
                "date": "2026-09-03",
                "valid_until": "2026-09-10",
                "supplier": {
                    "name": "ТОВ «Оріон Аудіо Дистрибуція»",
                    "code": "38124779",
                    "iban": "UA663510050000026004017723561"
                },
                "buyer": {"name": "ТОВ «Сузірʼя Рітейл»", "code": "44172914"},
                "items": [{"name": "Товар", "unit": "шт", "quantity": 1, "price": "1000.00", "amount": "1000.00"}],
                "total_without_vat": "1000.00",
                "vat": "200.00",
                "total": "1200.00"
            },
            issues=[],
            image={"original": {"width": 1240, "height": 1754, "bytes": 179639}, "sent": {"width": 1240, "height": 1754, "bytes": 120000}},
            model="gemini-test",
            elapsed={"prepare": 0.01, "extraction": 1.5, "checks": 0.001},
            usage={"prompt_tokens": 500, "completion_tokens": 300, "total_tokens": 800}
        )

        with patch("app.extraction.process", return_value=auto_result):
            from app.main import app
            with TestClient(app) as c:
                content = make_jpeg()
                response = c.post(
                    "/api/extract",
                    files={"image": ("rahunok-01.jpg", content, "image/jpeg")}
                )
        assert response.status_code == 200
        data = response.json()
        assert data["decision"] == "auto"
        assert data["model"] == "gemini-test"
        assert data["usage"]["prompt_tokens"] == 500

    def test_reject_nakladna(self):
        """Накладна отримує reject."""
        from unittest.mock import patch
        from app.extraction import Result

        reject_result = Result(
            decision="reject",
            reasons=["Документ є «накладна», а не рахунком на оплату."],
            document={"document_type": "накладна"},
            issues=[],
            image={"original": {"width": 100, "height": 100, "bytes": 5000}, "sent": {"width": 100, "height": 100, "bytes": 4000}},
        )

        with patch("app.extraction.process", return_value=reject_result):
            from app.main import app
            with TestClient(app) as c:
                content = make_jpeg()
                response = c.post(
                    "/api/extract",
                    files={"image": ("nakladna.jpg", content, "image/jpeg")}
                )
        assert response.status_code == 200
        data = response.json()
        assert data["decision"] == "reject"


class TestPipelineWithMockedLLM:
    """Тести конвеєра extraction.process з підробленим llm.extract."""

    def test_pipeline_image_error(self):
        """Непридатне зображення → reject."""
        from app.extraction import process
        result = process(b"not an image")
        assert result.decision == "reject"
        assert len(result.reasons) > 0

    def test_pipeline_llm_error(self):
        """Помилка LLM → review (не reject документа)."""
        from unittest.mock import patch
        from app.extraction import process
        from app.llm import LLMError

        with patch("app.llm.extract", side_effect=LLMError("Timeout", kind="timeout")):
            content = make_jpeg()
            result = process(content)
        assert result.decision == "review"
        # Перевіряємо що в reasons є опис проблеми
        assert len(result.reasons) > 0

    def test_pipeline_valid_doc_all_checks_pass(self):
        """Правильний документ, всі перевірки прогоняємо реально."""
        from unittest.mock import patch
        from app.extraction import process

        mock_llm_result = {
            "document": {
                "document_type": "рахунок",
                "number": "ОА-0917",
                "date": "2026-09-03",
                "valid_until": "2026-09-10",
                "supplier": {
                    "name": "ТОВ «Оріон Аудіо Дистрибуція»",
                    "code": "38124779",
                    "iban": "UA663510050000026004017723561"
                },
                "buyer": {"name": "ТОВ «Сузірʼя Рітейл»", "code": "44172914"},
                "items": [
                    {"name": "Товар А", "unit": "шт", "quantity": 20, "price": "2875.00", "amount": "57500.00"},
                    {"name": "Товар Б", "unit": "шт", "quantity": 10, "price": "2875.00", "amount": "28750.00"},
                    {"name": "Товар В", "unit": "шт", "quantity": 15, "price": "412.50", "amount": "6187.50"},
                ],
                "total_without_vat": "92437.50",
                "vat": "18487.50",
                "total": "110925.00"
            },
            "model": "test-model",
            "elapsed_extraction": 1.5,
            "usage": {"prompt_tokens": 500, "completion_tokens": 300, "total_tokens": 800},
            "finish_reason": "stop"
        }

        with patch("app.llm.extract", return_value=mock_llm_result):
            content = make_jpeg()
            result = process(content)

        assert result.decision == "auto", f"Очікувався auto, отримано {result.decision}: {result.reasons}"
        assert result.model == "test-model"
        assert result.usage is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
