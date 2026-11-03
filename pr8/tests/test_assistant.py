"""Unit tests for the assistant tool calling loop with mocked LLM."""

import json
from unittest.mock import MagicMock, patch
import pytest
from app import assistant, llm, tools


class FakeToolCallFunction:
    def __init__(self, name: str, arguments: str):
        self.name = name
        self.arguments = arguments


class FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.type = "function"
        self.function = FakeToolCallFunction(name, arguments)


class FakeMessage:
    def __init__(self, content: str | None = None, tool_calls: list | None = None):
        self.role = "assistant"
        self.content = content
        self.tool_calls = tool_calls or []


def test_assistant_direct_text_answer():
    """Assistant returns direct text without tool calls."""
    mock_resp = {
        "message": FakeMessage(content="Вітаю! Чим можу допомогти?"),
        "finish_reason": "stop",
        "model": "test-model",
        "elapsed": 0.5,
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }

    with patch("app.llm.chat", return_value=mock_resp):
        ans = assistant.answer("Привіт", "C-1001")
        assert ans.text == "Вітаю! Чим можу допомогти?"
        assert ans.rounds == 1
        assert ans.stopped == "answer"
        assert len(ans.calls) == 0
        assert ans.usage["total_tokens"] == 15
        assert ans.elapsed["model"] == 0.5


def test_assistant_single_tool_call_flow():
    """Assistant calls list_orders, receives result, and answers."""
    call1 = FakeToolCall("call_abc123", "list_orders", "{}")
    step1_resp = {
        "message": FakeMessage(content=None, tool_calls=[call1]),
        "finish_reason": "tool_calls",
        "model": "test-model",
        "elapsed": 0.4,
        "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
    }

    step2_resp = {
        "message": FakeMessage(content="У вас 3 замовлення: 10463, 10458, 10421."),
        "finish_reason": "stop",
        "model": "test-model",
        "elapsed": 0.6,
        "usage": {"prompt_tokens": 50, "completion_tokens": 15, "total_tokens": 65},
    }

    with patch("app.llm.chat", side_effect=[step1_resp, step2_resp]) as mock_chat:
        ans = assistant.answer("Які мої замовлення?", "C-1001")
        assert ans.rounds == 2
        assert ans.stopped == "answer"
        assert "У вас 3 замовлення" in ans.text
        assert len(ans.calls) == 1
        assert ans.calls[0].name == "list_orders"
        assert ans.calls[0].status == "ok"
        assert ans.usage["total_tokens"] == 90

        # Verify messages sent to chat in round 2 include tool message with tool_call_id
        second_call_messages = mock_chat.call_args_list[1][0][0]
        tool_msg = next((m for m in second_call_messages if isinstance(m, dict) and m.get("role") == "tool"), None)
        assert tool_msg is not None
        assert tool_msg["tool_call_id"] == "call_abc123"
        tool_content = json.loads(tool_msg["content"])
        assert isinstance(tool_content, list)
        assert len(tool_content) == 3


def test_assistant_multiple_tool_calls_in_single_turn():
    """Assistant calls search_products, get_stock, delivery_quote in one turn."""
    call1 = FakeToolCall("call_1", "get_stock", json.dumps({"sku": "ALT-AX3"}))
    call2 = FakeToolCall("call_2", "delivery_quote", json.dumps({
        "city": "Львів",
        "method": "branch",
        "items": [{"sku": "ALT-AX3", "quantity": 1}]
    }))

    step1_resp = {
        "message": FakeMessage(content=None, tool_calls=[call1, call2]),
        "finish_reason": "tool_calls",
        "model": "test-model",
        "elapsed": 0.8,
        "usage": {"prompt_tokens": 40, "completion_tokens": 10, "total_tokens": 50},
    }

    step2_resp = {
        "message": FakeMessage(content="Роутер є в кількості 7 шт, доставка у Львів безкоштовна."),
        "finish_reason": "stop",
        "model": "test-model",
        "elapsed": 0.5,
        "usage": {"prompt_tokens": 80, "completion_tokens": 20, "total_tokens": 100},
    }

    with patch("app.llm.chat", side_effect=[step1_resp, step2_resp]) as mock_chat:
        ans = assistant.answer("Чи є роутер і скільки доставка?", "C-1002")
        assert ans.rounds == 2
        assert len(ans.calls) == 2
        assert ans.calls[0].name == "get_stock"
        assert ans.calls[0].status == "ok"
        assert ans.calls[1].name == "delivery_quote"
        assert ans.calls[1].status == "ok"

        # Verify both tool results were appended to messages
        second_call_messages = mock_chat.call_args_list[1][0][0]
        tool_msgs = [m for m in second_call_messages if isinstance(m, dict) and m.get("role") == "tool"]
        assert len(tool_msgs) == 2
        assert tool_msgs[0]["tool_call_id"] == "call_1"
        assert tool_msgs[1]["tool_call_id"] == "call_2"


def test_assistant_max_rounds_exhaustion(monkeypatch):
    """When max rounds limit is reached, assistant stops with limit status."""
    monkeypatch.setenv("TOOL_MAX_ROUNDS", "2")

    infinite_call = FakeToolCall("call_inf", "list_orders", "{}")
    looping_resp = {
        "message": FakeMessage(content=None, tool_calls=[infinite_call]),
        "finish_reason": "tool_calls",
        "model": "test-model",
        "elapsed": 0.3,
        "usage": {"prompt_tokens": 20, "completion_tokens": 5, "total_tokens": 25},
    }

    fallback_summary_resp = {
        "message": FakeMessage(content="Ось що вдалося знайти до досягнення ліміту."),
        "finish_reason": "stop",
        "model": "test-model",
        "elapsed": 0.2,
        "usage": {"prompt_tokens": 30, "completion_tokens": 10, "total_tokens": 40},
    }

    with patch("app.llm.chat", side_effect=[looping_resp, looping_resp, fallback_summary_resp]):
        ans = assistant.answer("Що там?", "C-1001")
        assert ans.rounds == 2
        assert ans.stopped == "limit"
        assert len(ans.calls) == 2
