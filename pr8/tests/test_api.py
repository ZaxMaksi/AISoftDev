"""Tests for FastAPI HTTP endpoints."""

from fastapi.testclient import TestClient
from unittest.mock import patch
from app.main import app
from app.llm import LLMError
from app import assistant

client = TestClient(app)


def test_get_index_html():
    response = client.get("/")
    assert response.status_code == 200
    assert "Помічник клієнта магазину «Сузірʼя»" in response.text
    assert "text/html" in response.headers["content-type"]


def test_get_customers_list():
    response = client.get("/api/customers")
    assert response.status_code == 200
    customers = response.json()
    assert isinstance(customers, list)
    assert len(customers) >= 3
    ids = {c["customer_id"] for c in customers}
    assert {"C-1001", "C-1002", "C-1003"}.issubset(ids)


def test_get_tools_specs():
    response = client.get("/api/tools")
    assert response.status_code == 200
    specs = response.json()
    assert isinstance(specs, list)
    names = {s["function"]["name"] for s in specs}
    assert "list_orders" in names
    assert "create_return" in names


def test_post_reset():
    response = client.post("/api/reset")
    assert response.status_code == 200
    assert response.json() == {"ok": True}


def test_api_ask_validation_empty_question():
    response = client.post("/api/ask", json={"customer_id": "C-1001", "question": ""})
    assert response.status_code == 400
    assert "порожнім" in response.json()["detail"]


def test_api_ask_validation_missing_customer():
    response = client.post("/api/ask", json={"customer_id": "", "question": "Де моє замовлення?"})
    assert response.status_code == 400
    assert "клієнта" in response.json()["detail"]


def test_api_ask_unknown_customer():
    response = client.post("/api/ask", json={"customer_id": "C-9999", "question": "Де моє замовлення?"})
    assert response.status_code == 404


def test_api_ask_llm_error_graceful_handling():
    """If LLM raises an LLMError, endpoint returns 200 with structured safe error message."""
    with patch("app.assistant.answer", side_effect=LLMError("API rate limit exceeded", kind="rate_limit")):
        response = client.post("/api/ask", json={"customer_id": "C-1001", "question": "Привіт"})
        assert response.status_code == 200
        data = response.json()
        assert "проблема зі зв'язком із мовною моделлю" in data["answer"]
        assert data["stopped"] == "error"
