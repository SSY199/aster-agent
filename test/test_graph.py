# test/test_graph.py
import pytest

from app.agent.graph import get_graph
from app.agent.state import initial_state
from app.config import get_settings

_settings = get_settings()
_missing = (
    not _settings.google_api_key
    or not _settings.faiss_index_path_resolved.exists()
)
pytestmark = pytest.mark.skipif(
    _missing, reason="requires GOOGLE_API_KEY and a built storage/ index"
)


def test_graph_answers_return_window_question():
    from langchain_core.messages import HumanMessage

    graph = get_graph()
    state = initial_state("test-session")
    state["messages"] = [HumanMessage(content="How long is the return window?")]

    result = graph.invoke(state)

    assert result["messages"][-1].content
    assert result["intent"] == "policy"
    assert result["handoff"] is False


def test_graph_flags_conflict_for_breeze_tumbler():
    from langchain_core.messages import HumanMessage

    graph = get_graph()
    state = initial_state("test-session-2")
    state["messages"] = [HumanMessage(content="Can I put the Breeze Tumbler in the dishwasher?")]

    result = graph.invoke(state)

    assert result["handoff"] is True
    assert result["handoff_reason"] == "source_conflict"


def test_graph_asks_for_order_id_when_missing():
    from langchain_core.messages import HumanMessage

    graph = get_graph()
    state = initial_state("test-session-3")
    state["messages"] = [HumanMessage(content="Where is my order?")]

    result = graph.invoke(state)

    assert result["needs_order_id"] is True
    assert "order" in result["messages"][-1].content.lower()
    
    
# add to test/test_graph.py

def test_multiturn_canada_followup_inherits_shipping_topic():
    from langchain_core.messages import HumanMessage

    graph = get_graph()
    state = initial_state("multiturn-1")

    state["messages"] = [HumanMessage(content="Do you ship internationally?")]
    state = graph.invoke(state)

    state["messages"] = state["messages"] + [HumanMessage(content="What about Canada, and how long does it take?")]
    state = graph.invoke(state)

    last_answer = state["messages"][-1].content.lower()
    assert "canada" in last_answer or "5" in last_answer  # 5-9 business days
    filenames = {c.filename for c in state["retrieved"]}
    assert "06-international-shipping.md" in filenames


def test_multiturn_order_followup_inherits_order_id():
    from langchain_core.messages import HumanMessage

    graph = get_graph()
    state = initial_state("multiturn-2")

    state["messages"] = [HumanMessage(content="Where is ORD-1007?")]
    state = graph.invoke(state)
    assert state["last_order_id"] == "ORD-1007"

    state["messages"] = state["messages"] + [HumanMessage(content="When will it arrive?")]
    state = graph.invoke(state)

    assert state["intent"] == "order"
    assert state["current_order_id"] == "ORD-1007"
    assert state["tool_result"].found is True


def test_multiturn_does_not_carry_unrelated_topic_forward():
    """A brand-new, clearly different question shouldn't be dragged
    toward the previous turn's topic just because last_topic is set.
    """
    from langchain_core.messages import HumanMessage

    graph = get_graph()
    state = initial_state("multiturn-3")

    state["messages"] = [HumanMessage(content="Do you ship internationally?")]
    state = graph.invoke(state)

    state["messages"] = state["messages"] + [
        HumanMessage(content="What is the warranty on your bags?")
    ]
    state = graph.invoke(state)

    filenames = {c.filename for c in state["retrieved"]}
    assert "07-warranty.md" in filenames