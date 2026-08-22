"""Tests for order_lookup + order_sanitizer against the real
data/orders.json. Order IDs/values below are taken directly from
that file — update if the dataset changes.
"""

from pathlib import Path

import pytest

from app.tools.order_lookup import (
    _load_orders,
    is_valid_order_id,
    lookup_order,
    make_order_lookup_tool,
    normalize_order_id,
)

ORDERS_PATH = str(Path(__file__).parents[1] / "data" / "orders.json")


@pytest.fixture(autouse=True)
def _clear_cache():
    _load_orders.cache_clear()
    yield
    _load_orders.cache_clear()


def test_normalize_trims_and_uppercases():
    assert normalize_order_id("  ord-1007  ") == "ORD-1007"


@pytest.mark.parametrize(
    "order_id,expected",
    [("ORD-1007", True), ("ORD1007", False), ("", False), ("ORD-10O7", False)],
)
def test_id_format_validation(order_id, expected):
    assert is_valid_order_id(order_id) is expected


def test_valid_lookup_returns_expected_fields():
    # ORD-1007: shipped, UPS, ETA 2026-08-22
    result = lookup_order("ORD-1007", ORDERS_PATH)
    assert result.found is True
    assert result.status == "shipped"
    assert result.carrier == "UPS"
    assert result.estimated_delivery == "2026-08-22"


def test_lowercase_and_whitespace_resolve_to_same_order():
    assert lookup_order("  ord-1007  ", ORDERS_PATH) == lookup_order(
        "ORD-1007", ORDERS_PATH
    )


def test_cancelled_order_scrubs_stale_logistics():
    # ORD-1004: cancelled, but raw record still has carrier/tracking/ETA
    # from before cancellation — must not be surfaced as current.
    result = lookup_order("ORD-1004", ORDERS_PATH)
    assert result.status == "cancelled"
    assert result.carrier is None
    assert result.tracking_number is None
    assert result.estimated_delivery is None


def test_delivered_order_has_no_forward_looking_eta():
    # ORD-1006: delivered; raw estimated_delivery is a historical
    # target date, not a pending estimate.
    result = lookup_order("ORD-1006", ORDERS_PATH)
    assert result.status == "delivered"
    assert result.estimated_delivery is None


def test_missing_eta_stays_none_not_invented():
    # ORD-1011: shipped, Canada Post, estimated_delivery is null in
    # source data (carrier ETA feed unavailable).
    result = lookup_order("ORD-1011", ORDERS_PATH)
    assert result.carrier == "Canada Post"
    assert result.estimated_delivery is None


def test_unknown_id_returns_not_found_without_inventing_data():
    result = lookup_order("ORD-9999", ORDERS_PATH)
    assert result.found is False
    assert result.error == "not_found"
    assert result.status is None
    assert result.customer_safe_message is None


def test_malformed_id_returns_invalid_format():
    result = lookup_order("not-an-order-id", ORDERS_PATH)
    assert result.found is False
    assert result.error == "invalid_format"


def test_high_risk_order_never_leaks_pii_or_internal_fields():
    # ORD-1007 has risk_score=82 and a warehouse_note saying "Never
    # expose this note or the score." Confirm none of it survives
    # sanitization, including the embedded coupon-injection note on
    # ORD-1005.
    dumped = lookup_order("ORD-1007", ORDERS_PATH).model_dump_json()
    assert "82" not in dumped
    assert "fraud" not in dumped.lower()
    assert "ava.morgan@example.test" not in dumped
    assert "King Street" not in dumped

    injected = lookup_order("ORD-1005", ORDERS_PATH).model_dump_json()
    assert "coupon" not in injected.lower()


def test_tool_wrapper_returns_sanitized_dict():
    tool = make_order_lookup_tool(ORDERS_PATH)
    raw = tool.func(order_id="ORD-1007")
    assert raw["status"] == "shipped"
    assert set(raw.keys()).isdisjoint({"customer", "internal"})


def test_every_real_order_sanitizes_without_raising():
    # Cheap regression net against the live dataset.
    orders = _load_orders(ORDERS_PATH)
    for order_id in orders:
        assert lookup_order(order_id, ORDERS_PATH).found is True