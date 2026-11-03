"""Fixtures and test setup for PR8."""

import os
import pytest
from shop import service

@pytest.fixture(autouse=True)
def reset_shop():
    """Ensure shop state is reset before each test."""
    os.environ["SHOP_DELAY_MS"] = "0"
    os.environ["SHOP_FAILURE_RATE"] = "0"
    os.environ["SHOP_TODAY"] = "2026-10-06"
    service.reset()
    yield
    service.reset()
