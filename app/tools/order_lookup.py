"""Order lookup tool: normalization, file-backed lookup, and the
LangChain StructuredTool wrapper the agent graph calls.

The model never sees orders.json. It only ever receives the
OrderLookupResult produced here.
"""
from __future__ import annotations

import json
import re
import threading
from functools import lru_cache
from pathlib import Path

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from app.schemas.order import OrderLookupResult, OrderRecord
from app.tools.order_sanitizer import not_found_result, sanitize_order


_ORDER_ID_PATTERN = re.compile(r"ORD-\d{4,}")
_load_lock = threading.Lock()


def normalize_order_id(raw: str) -> str:
    """Normalize harmless customer input differences."""
    return raw.strip().upper()


def is_valid_order_id(order_id: str) -> bool:
    """Return True only for valid order IDs such as ORD-1007."""
    return bool(_ORDER_ID_PATTERN.fullmatch(order_id))


@lru_cache(maxsize=1)
def _load_orders(orders_path: str) -> dict[str, OrderRecord]:
    """
    Load orders.json and cache records by normalized order ID.

    The raw orders file is only loaded here and is never passed
    directly to the model.
    """
    with _load_lock:
        data = json.loads(
            Path(orders_path).read_text(encoding="utf-8")
        )

        records = (
            OrderRecord.model_validate(row)
            for row in data["orders"]
        )

        return {
            normalize_order_id(record.order_id): record
            for record in records
        }


def lookup_order(
    order_id: str,
    orders_path: str,
) -> OrderLookupResult:
    """
    Safely look up one order.

    Returns only a sanitized OrderLookupResult.
    """

    if not isinstance(order_id, str):
        return not_found_result("", malformed=True)

    normalized = normalize_order_id(order_id)

    if not is_valid_order_id(normalized):
        return not_found_result(normalized, malformed=True)

    orders = _load_orders(orders_path)
    record = orders.get(normalized)

    if record is None:
        return not_found_result(normalized, malformed=False)

    return sanitize_order(record)


class OrderLookupInput(BaseModel):
    order_id: str = Field(
        description=(
            "Customer-provided order ID, for example 'ORD-1007'."
        )
    )


def make_order_lookup_tool(orders_path: str) -> StructuredTool:
    """
    Create the LangChain tool using the configured orders data path.
    """

    def _run(order_id: str) -> dict:
        result = lookup_order(
            order_id=order_id,
            orders_path=orders_path,
        )

        return result.model_dump(mode="json")

    return StructuredTool.from_function(
        func=_run,
        name="order_lookup",
        description=(
            "Look up the current status of a customer's order using an "
            "order ID. Returns only sanitized customer-safe information, "
            "including current status, carrier, tracking number, delivery "
            "estimate when available, and a customer-safe summary. "
            "Never returns customer PII or internal fields. Call this only "
            "when an order ID has been provided by the customer or is known "
            "from the current conversation session."
        ),
        args_schema=OrderLookupInput,
    )