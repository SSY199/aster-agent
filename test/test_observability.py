import json

from app.agent.state import initial_state
from app.observability.logging_config import log_turn


def test_log_turn_writes_valid_json_with_no_pii():
    state = initial_state("obs-test")
    state["intent"] = "policy"
    state["handoff"] = False

    entry = log_turn(state, user_message="How long is the return window?")

    # round-trips as valid JSON
    json.dumps(entry)

    assert entry["session_id"] == "obs-test"
    assert entry["user_message"] == "How long is the return window?"
    assert "email" not in json.dumps(entry).lower()
    assert "risk_score" not in json.dumps(entry).lower()


def test_log_turn_captures_handoff_reason():
    state = initial_state("obs-test-2")
    state["handoff"] = True
    state["handoff_reason"] = "source_conflict"

    entry = log_turn(state, user_message="Can I dishwasher the tumbler?")
    assert entry["handoff"] is True
    assert entry["handoff_reason"] == "source_conflict"


def test_log_turn_captures_error_when_provided():
    state = initial_state("obs-test-3")
    entry = log_turn(state, user_message="test", error="LLM timeout")
    assert entry["error"] == "LLM timeout"