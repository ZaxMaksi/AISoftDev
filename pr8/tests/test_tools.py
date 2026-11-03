"""Tests for tools registry, schema validation, permissions, and data minimization."""

import json
import pytest
from app import tools
from shop import service


def test_tool_specs_structure():
    """Verify tool specs contain expected tools and valid JSON schemas."""
    specs = tools.specs()
    assert isinstance(specs, list)
    assert len(specs) >= 6

    names = {s["function"]["name"] for s in specs}
    expected = {
        "list_orders",
        "get_order",
        "search_products",
        "get_product",
        "get_stock",
        "delivery_quote",
        "create_return",
    }
    assert expected.issubset(names)

    # Ensure forbidden operations are NOT in tool specs
    forbidden = {"issue_refund", "update_price", "add_bonus", "set_order_status", "cancel_order"}
    assert not forbidden.intersection(names)

    for s in specs:
        func = s["function"]
        assert "name" in func
        assert "description" in func
        assert "parameters" in func
        params = func["parameters"]
        assert params.get("type") == "object"
        assert params.get("additionalProperties") is False


def test_level1_forbidden_tool_rejected():
    """Level 1 check: unknown or administrative tools must be rejected."""
    ctx = tools.Context(customer_id="C-1001")
    for bad_tool in ["issue_refund", "update_price", "add_bonus", "set_order_status", "random_hack"]:
        res = tools.call(bad_tool, "{}", ctx)
        assert res.status == "rejected"
        assert "forbidden_tool" in res.content.get("error", "")


def test_level2_invalid_json_rejected():
    """Level 2 check: invalid JSON strings must be rejected."""
    ctx = tools.Context(customer_id="C-1001")
    res = tools.call("get_order", "{invalid_json: 123", ctx)
    assert res.status == "rejected"
    assert res.content.get("error") == "invalid_json"


def test_level3_schema_validation_rejections():
    """Level 3 check: schema validation with jsonschema."""
    ctx = tools.Context(customer_id="C-1001")

    # Missing required argument
    res = tools.call("get_order", "{}", ctx)
    assert res.status == "rejected"
    assert res.content.get("error") == "schema_validation_error"

    # Additional unexpected argument (additionalProperties: False)
    res = tools.call("get_order", json.dumps({"order_id": "10458", "extra_field": "hack"}), ctx)
    assert res.status == "rejected"
    assert res.content.get("error") == "schema_validation_error"

    # Invalid enum value in create_return
    res = tools.call("create_return", json.dumps({
        "order_id": "10421",
        "sku": "ORN-X2",
        "reason": "invalid_reason_here"
    }), ctx)
    assert res.status == "rejected"
    assert res.content.get("error") == "schema_validation_error"

    # Negative or zero quantity in delivery_quote items
    res = tools.call("delivery_quote", json.dumps({
        "city": "Київ",
        "method": "courier",
        "items": [{"sku": "ALT-AX3", "quantity": 0}]
    }), ctx)
    assert res.status == "rejected"
    assert res.content.get("error") == "schema_validation_error"


def test_level4_order_id_normalization():
    """Normalization of order ID: spaces, № symbol, or integer."""
    ctx = tools.Context(customer_id="C-1001")
    # "№ 10 458"
    res1 = tools.call("get_order", json.dumps({"order_id": "№ 10 458"}), ctx)
    assert res1.status == "ok"
    assert res1.content["order_id"] == "10458"

    # Integer 10458
    res2 = tools.call("get_order", json.dumps({"order_id": 10458}), ctx)
    assert res2.status == "ok"
    assert res2.content["order_id"] == "10458"


def test_level5_order_ownership_get_order():
    """Customer C-1001 cannot view order 10466 (belongs to C-1002)."""
    ctx = tools.Context(customer_id="C-1001")
    res = tools.call("get_order", json.dumps({"order_id": "10466"}), ctx)
    assert res.status == "rejected"
    assert res.content.get("error") == "access_denied"


def test_level5_order_ownership_create_return():
    """Customer C-1001 cannot return items from order 10412 (belongs to C-1003)."""
    ctx = tools.Context(customer_id="C-1001")
    res = tools.call("create_return", json.dumps({
        "order_id": "10412",
        "sku": "SIR-R10",
        "reason": "not_suitable"
    }), ctx)
    assert res.status == "rejected"
    assert res.content.get("error") == "access_denied"


def test_data_minimization_internal_notes_stripped():
    """Check that internal operator notes are never leaked to the client/model."""
    # Order 10466 belongs to C-1002 and contains a sensitive internal_note
    ctx = tools.Context(customer_id="C-1002")
    res = tools.call("get_order", json.dumps({"order_id": "10466"}), ctx)
    assert res.status == "ok"
    content = res.content
    assert "internal_note" not in content
    assert "operator_notes" not in content

    # Check raw shop record vs sanitized record
    raw_order = service.get_order("10466")
    assert "internal_note" in raw_order
    assert len(raw_order["internal_note"]) > 0


def test_non_existent_order_error():
    """Non-existent order raises not_found error."""
    ctx = tools.Context(customer_id="C-1003")
    res = tools.call("get_order", json.dumps({"order_id": "99999"}), ctx)
    assert res.status == "error"
    assert res.content.get("error") == "not_found"


def test_return_policy_violation():
    """Non-returnable item should return policy violation."""
    # Order 10421 belongs to C-1001, product ORN-X2 is returnable=False
    ctx = tools.Context(customer_id="C-1001")
    res = tools.call("create_return", json.dumps({
        "order_id": "10421",
        "sku": "ORN-X2",
        "reason": "not_suitable"
    }), ctx)
    assert res.status == "error"
    assert res.content.get("error") == "policy"


def test_return_idempotency_prevents_duplicate():
    """Submitting the same return twice should not create a duplicate."""
    # Order 10412 belongs to C-1003, SIR-R10 is returnable
    ctx = tools.Context(customer_id="C-1003")
    args = json.dumps({
        "order_id": "10412",
        "sku": "SIR-R10",
        "reason": "not_suitable",
        "quantity": 1
    })
    res1 = tools.call("create_return", args, ctx)
    assert res1.status == "ok"
    return_id1 = res1.content["return_id"]

    # Call again with identical parameters
    res2 = tools.call("create_return", args, ctx)
    assert res2.status == "ok"
    assert res2.content.get("already_created") is True
    assert res2.content["return_id"] == return_id1


def test_delivery_quote_external_city():
    """Delivery outside Ukraine carrier cities raises invalid_request error."""
    ctx = tools.Context(customer_id="C-1001")
    res = tools.call("delivery_quote", json.dumps({
        "city": "Варшава",
        "method": "courier",
        "items": [{"sku": "PB-20", "quantity": 1}]
    }), ctx)
    assert res.status == "error"
    assert res.content.get("error") == "invalid_request"
