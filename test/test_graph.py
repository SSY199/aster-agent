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