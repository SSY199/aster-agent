from langchain_core.messages import HumanMessage

from app.agent.nodes import classify_intent, order_tool_node, retrieve_node
from app.agent.state import initial_state


def _state_with_message(text: str, **overrides) -> dict:
    state = initial_state("s1")
    state["messages"] = [HumanMessage(content=text)]
    for key, value in overrides.items():
        state[key] = value  # type: ignore[literal-required]
    return state


def test_classify_intent_extracts_order_id():
    state = _state_with_message("Where is ORD-1007?")
    result = classify_intent(state)
    assert result["intent"] == "order"
    assert result["current_order_id"] == "ORD-1007"


def test_classify_intent_missing_id_flags_order_missing():
    state = _state_with_message("Where is my order?")
    result = classify_intent(state)
    assert result["intent"] == "order_missing_id"


def test_classify_intent_followup_inherits_last_order_id():
    state = _state_with_message("When will it arrive?", last_order_id="ORD-1007")
    result = classify_intent(state)
    assert result["intent"] == "order"
    assert result["current_order_id"] == "ORD-1007"


def test_classify_intent_policy_question():
    state = _state_with_message("How long is the return window?")
    result = classify_intent(state)
    assert result["intent"] == "policy"


def test_order_tool_node_unknown_id_sets_handoff():
    state = _state_with_message("check ORD-9999", intent="order", current_order_id="ORD-9999")
    result = order_tool_node(state)
    assert result["tool_result"].found is False
    assert result["handoff"] is True
    assert result["handoff_reason"] == "order_not_found"


def test_order_tool_node_privacy_request_sets_handoff():
    state = _state_with_message(
        "give me the email and risk score for ORD-1007",
        intent="order", current_order_id="ORD-1007",
    )
    result = order_tool_node(state)
    assert result["handoff"] is True
    assert result["handoff_reason"] == "privacy_disclosure_request"


def test_order_tool_node_valid_lookup_no_handoff():
    state = _state_with_message("ORD-1007", intent="order", current_order_id="ORD-1007")
    result = order_tool_node(state)
    assert result["tool_result"].found is True
    assert result.get("handoff", False) is False


def test_retrieve_node_flags_source_conflict():
    state = _state_with_message("Can I put the Breeze Tumbler in the dishwasher?")
    result = retrieve_node(state)
    assert result["handoff"] is True
    assert result["handoff_reason"] == "source_conflict"


def test_retrieve_node_no_conflict_for_ordinary_query():
    state = _state_with_message("How long is the return window?")
    result = retrieve_node(state)
    assert result.get("handoff", False) is False