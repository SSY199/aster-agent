"""Wires the node functions into a runnable LangGraph StateGraph.

Flow:
    reset_scratch -> classify_intent -> (order_tool_node | retrieve_node)
                                              |                  |
                                        respond_node <- grounding_check

reset_scratch runs first, every turn, so a handoff/conflict flag set
on a previous turn can never leak into a turn that didn't re-trigger
it (see state.reset_turn_scratch's docstring for why this matters).
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.agent.nodes import (
    classify_intent,
    grounding_check,
    order_tool_node,
    respond_node,
    retrieve_node,
)
from app.agent.state import AgentState, reset_turn_scratch


def _reset_scratch(state: AgentState) -> dict:
    reset_turn_scratch(state)
    return {
        "retrieved": [],
        "conflicts": [],
        "tool_result": None,
        "injection_flags": [],
        "needs_order_id": False,
        "handoff": False,
        "handoff_reason": None,
        "intent": None,
        "current_order_id": None,
    }


def _route_after_classify(state: AgentState) -> str:
    if state["intent"] in ("order", "order_missing_id"):
        return "order_tool_node"
    return "retrieve_node"


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("reset_scratch", _reset_scratch)
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("order_tool_node", order_tool_node)
    graph.add_node("retrieve_node", retrieve_node)
    graph.add_node("grounding_check", grounding_check)
    graph.add_node("respond_node", respond_node)

    graph.set_entry_point("reset_scratch")
    graph.add_edge("reset_scratch", "classify_intent")

    graph.add_conditional_edges(
        "classify_intent",
        _route_after_classify,
        {
            "order_tool_node": "order_tool_node",
            "retrieve_node": "retrieve_node",
        },
    )

    graph.add_edge("order_tool_node", "respond_node")
    graph.add_edge("retrieve_node", "grounding_check")
    graph.add_edge("grounding_check", "respond_node")
    graph.add_edge("respond_node", END)

    return graph.compile()


_compiled_graph = None


def get_graph():
    """Cached compiled graph — compile once, reuse across requests."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph