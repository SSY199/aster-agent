"""Pydantic models for order data.

OrderRecord mirrors orders.json exactly (internal use only — never
serialized directly to the LLM or the customer).

OrderLookupResult is the sanitized, customer-safe shape that the tool
returns. Nothing outside this file should ever construct one by hand;
it must go through order_sanitizer.sanitize_order().
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

OrderStatus = Literal[
    "pending",
    "processing",
    "shipped",
    "delayed",
    "delivered",
    "returned",
    "cancelled",
    "exception",
]

# Statuses where any previously-set carrier/tracking/ETA data is stale
# and must not be surfaced to the customer as current information.
STALE_LOGISTICS_STATUSES: set[str] = {"cancelled"}

# Statuses where an *estimated_delivery* is meaningless (already
# resolved, one way or another) even if the field happens to be set.
RESOLVED_STATUSES: set[str] = {"delivered", "returned", "cancelled"}


class CustomerInfo(BaseModel):
    """Internal only. Never exposed to the model or the customer."""

    name: str
    email: str
    shipping_address: str


class OrderItem(BaseModel):
    sku: str
    name: str
    quantity: int
    final_sale: bool = False


class InternalInfo(BaseModel):
    """Internal only. Never exposed to the model or the customer."""

    risk_score: int
    warehouse_note: str
    support_tags: list[str] = Field(default_factory=list)


class OrderRecord(BaseModel):
    """Full internal representation of one row in orders.json."""

    order_id: str
    customer: CustomerInfo
    membership_tier: Literal["standard", "trailplus"]
    items: list[OrderItem]
    placed_at: datetime
    status: OrderStatus
    status_updated_at: datetime
    shipped_at: datetime | None = None
    delivered_at: datetime | None = None
    carrier: str | None = None
    tracking_number: str | None = None
    estimated_delivery: str | None = None  # date-only string in source data
    customer_safe_message: str
    internal: InternalInfo


class OrderLookupResult(BaseModel):
    """Sanitized result returned by the order lookup tool.

    This is the ONLY shape that may reach the LLM prompt or the
    customer-facing response. It contains no customer PII and no
    internal fields.
    """

    order_id: str
    found: bool
    error: Literal["not_found", "invalid_format"] | None = None

    status: OrderStatus | None = None
    carrier: str | None = None
    tracking_number: str | None = None
    estimated_delivery: str | None = None
    shipped_at: datetime | None = None
    delivered_at: datetime | None = None
    customer_safe_message: str | None = None