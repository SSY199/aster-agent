"""LangGraph state shape for one turn of conversation.

Split into two kinds of fields:
- `messages`: full conversation history (LangGraph's add_messages
  reducer appends new messages rather than overwriting).
- everything else: structured session memory + per-turn scratch
  space, so follow-ups like "What about Canada?" or "When will it
  arrive?" resolve without re-parsing the whole message history.

Per-turn scratch fields (retrieved, conflicts, tool_result,
injection_flags) are overwritten each turn — they describe what
*this* turn found, not accumulated history. Session-memory fields
(last_topic, last_order_id, last_country) persist across turns until
explicitly replaced.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages

from app.retrieval.hybrid_retriever import RetrievedChunk
from app.schemas.order import OrderLookupResult
from app.services.conflict_detector import ConflictResult

Topic = Literal[
    "returns",
    "shipping_domestic",
    "shipping_international",
    "warranty",
    "order_status",
    "order_changes",
    "gift_cards",
    "product_care",
    "membership",
    "other",
]


class AgentState(TypedDict, total=False):
    # conversation history — LangGraph appends via add_messages
    messages: Annotated[list, add_messages]

    # session identity
    session_id: str

    # persistent session memory — survives across turns
    last_topic: Topic | None
    last_order_id: str | None
    last_country: str | None

    # this-turn scratch space — set by nodes, read by respond_node,
    # not meant to accumulate across turns
    retrieved: list[RetrievedChunk]
    conflicts: list[ConflictResult]
    tool_result: OrderLookupResult | None
    injection_flags: list[str]
    needs_order_id: bool
    handoff: bool
    handoff_reason: str | None
    intent: Literal["order", "order_missing_id", "policy"] | None
    current_order_id: str | None


def initial_state(session_id: str) -> AgentState:
    """Fresh state for a brand-new session."""

    return AgentState(
        messages=[],
        session_id=session_id,
        intent=None,
        current_order_id=None,
        last_topic=None,
        last_order_id=None,
        last_country=None,
        retrieved=[],
        conflicts=[],
        tool_result=None,
        injection_flags=[],
        needs_order_id=False,
        handoff=False,
        handoff_reason=None,
    )


def reset_turn_scratch(state: AgentState) -> AgentState:
    """Clears per-turn fields before processing a new message, so a
    stale conflict/handoff flag from a previous turn can't leak into
    a turn that didn't re-trigger it. Session-memory fields
    (last_topic, last_order_id, last_country) are preserved.
    """
    state["retrieved"] = []
    state["conflicts"] = []
    state["tool_result"] = None
    state["injection_flags"] = []
    state["needs_order_id"] = False
    state["handoff"] = False
    state["handoff_reason"] = None
    state["intent"] = None
    state["current_order_id"] = None
    return state