from app.agent.state import initial_state, reset_turn_scratch


def test_initial_state_has_empty_session_memory():
    state = initial_state("session-1")
    assert state.get("session_id") == "session-1"
    assert state.get("messages") == []
    assert state.get("last_topic") is None
    assert state.get("last_order_id") is None


def test_reset_turn_scratch_clears_per_turn_fields():
    state = initial_state("session-1")
    state["conflicts"] = ["placeholder"]
    state["handoff"] = True
    state["handoff_reason"] = "conflict"

    reset_turn_scratch(state)

    assert state.get("conflicts") == []
    assert state.get("handoff") is False
    assert state.get("handoff_reason") is None


def test_reset_turn_scratch_preserves_session_memory():
    state = initial_state("session-1")
    state["last_topic"] = "shipping_international"
    state["last_order_id"] = "ORD-1007"

    reset_turn_scratch(state)

    assert state.get("last_topic") == "shipping_international"
    assert state.get("last_order_id") == "ORD-1007"