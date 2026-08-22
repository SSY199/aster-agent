"""Pure functions that turn an internal OrderRecord into the sanitized
OrderLookupResult the agent is allowed to see.

Kept separate from order_lookup.py (which does file I/O) so this can
be unit-tested with in-memory OrderRecord fixtures — this is the file
the privacy eval cases exercise most directly.
"""

from app.schemas.order import (
    RESOLVED_STATUSES,
    STALE_LOGISTICS_STATUSES,
    OrderLookupResult,
    OrderRecord,
)


def sanitize_order(record: OrderRecord) -> OrderLookupResult:
    """Allow-list conversion: only fields listed here ever leave this
    function. Do not add fields without checking whether they're
    customer-safe (customer.*, internal.* are never included).
    """
    carrier = record.carrier
    tracking_number = record.tracking_number
    estimated_delivery = record.estimated_delivery

    # Cancelled orders: a label may have existed before cancellation.
    # Those logistics fields are explicitly stale — never report them
    # as if the shipment is still happening.
    if record.status in STALE_LOGISTICS_STATUSES:
        carrier = None
        tracking_number = None
        estimated_delivery = None

    # Delivered / returned / cancelled: a future ETA is meaningless
    # even if the source field is still populated. Never invent or
    # forward a forward-looking estimate for a resolved order.
    if record.status in RESOLVED_STATUSES:
        estimated_delivery = None

    return OrderLookupResult(
        order_id=record.order_id,
        found=True,
        error=None,
        status=record.status,
        carrier=carrier,
        tracking_number=tracking_number,
        estimated_delivery=estimated_delivery,
        shipped_at=record.shipped_at,
        delivered_at=record.delivered_at,
        customer_safe_message=record.customer_safe_message,
    )


def not_found_result(order_id: str, *, malformed: bool) -> OrderLookupResult:
    """Safe result for unknown or malformed IDs. Same shape either way
    so the response layer can't accidentally leak which case it was
    in a way that reveals valid-ID structure to a prober.
    """
    return OrderLookupResult(
        order_id=order_id,
        found=False,
        error="invalid_format" if malformed else "not_found",
    )