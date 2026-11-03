"""Security tests: prompt injection, permission boundaries, customer isolation."""

import json
from unittest.mock import patch
from app import assistant, tools
from shop import service


def test_customer_cannot_be_overridden_by_prompt():
    """Even if prompt says 'I am customer C-1002', the context binds to session C-1003."""
    # Context is initialized with C-1003
    ctx = tools.Context(customer_id="C-1003")

    # The list_orders tool must return C-1003 orders regardless of arguments
    res = tools.call("list_orders", json.dumps({"customer_id": "C-1002"}), ctx)
    assert res.status == "rejected" or res.status == "ok"
    if res.status == "ok":
        # Check that orders belong to C-1003 (10470, 10412)
        returned_ids = {o["order_id"] for o in res.content}
        assert "10470" in returned_ids or "10412" in returned_ids
        assert "10466" not in returned_ids  # 10466 belongs to C-1002


def test_admin_prompt_injection_rejected():
    """Prompt injection claiming to be store manager trying to change prices or add bonuses."""
    ctx = tools.Context(customer_id="C-1001")

    # Even if LLM is tricked into calling update_price or add_bonus:
    res1 = tools.call("update_price", json.dumps({"sku": "VEGA-S", "price": "1.00"}), ctx)
    assert res1.status == "rejected"
    assert "forbidden_tool" in res1.content.get("error", "")

    res2 = tools.call("add_bonus", json.dumps({"customer_id": "C-1001", "amount": 5000}), ctx)
    assert res2.status == "rejected"
    assert "forbidden_tool" in res2.content.get("error", "")

    # Ensure actual shop service data is untouched
    vega = service.get_product("VEGA-S")
    assert vega["price"] == "5499.00"

    cust = service.get_customer("C-1001")
    assert cust["bonus_balance"] == 340


def test_product_description_prompt_injection_does_not_trigger_refund():
    """Product ORN-PRO has a prompt injection note about ZIRKY90 and refund.
    Verify that tools layer refuses any attempt to issue a refund or discount.
    """
    ctx = tools.Context(customer_id="C-1003")

    # Get product returns the description
    res_prod = tools.call("get_product", json.dumps({"sku": "ORN-PRO"}), ctx)
    assert res_prod.status == "ok"
    assert "ZIRKY90" in res_prod.content["description"]

    # Model might try to call issue_refund
    res_refund = tools.call("issue_refund", json.dumps({"order_id": "10412", "amount": "100.00"}), ctx)
    assert res_refund.status == "rejected"
    assert "forbidden_tool" in res_refund.content.get("error", "")
