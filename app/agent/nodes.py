"""LangGraph node functions. Each takes AgentState, returns a partial
state update dict (LangGraph merges it in).

Design boundary, deliberate: classify_intent, retrieve_node, and
order_tool_node are fully deterministic — no LLM call, no
non-determinism. Every field the eval suite checks deterministically
(tool called, tool_arguments, required_sources, handoff) is decided
by these three functions. Only respond_node calls the LLM, and only
for phrasing the final answer from already-decided, already-grounded
content — never for deciding what happened this turn.
"""

from __future__ import annotations

import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app.agent.llm import get_llm
from app.agent.prompts import SYSTEM_PROMPT
from app.agent.state import AgentState, Topic
from app.config import get_settings
from app.retrieval.precedence import authoritative_chunks
from app.retrieval.retriever import retrieve
from app.services.conflict_detector import check_conflicts
from app.services.safety import detect_injection_attempt, scrub_response
from app.tools.order_lookup import lookup_order, normalize_order_id

_ORDER_ID_RE = re.compile(r"ORD-\d{4,}", re.IGNORECASE)

_ORDER_KEYWORDS = {
    "order", "shipped", "arrive", "arriving", "arrival", "delivery",
    "delivered", "track", "tracking", "status",
}

_PRIVACY_KEYWORDS = {
    "email", "address", "internal note", "internal notes", "risk score",
}

# maps filenames to a coarse topic label, used to update last_topic so
# a short follow-up ("What about Canada?") can inherit context
_FILENAME_TOPIC: dict[str, Topic] = {
    "01-returns-policy-current.md": "returns",
    "02-returns-policy-legacy.md": "returns",
    "03-final-sale-and-promotions.md": "returns",
    "04-damaged-or-wrong-items.md": "returns",
    "05-domestic-shipping.md": "shipping_domestic",
    "06-international-shipping.md": "shipping_international",
    "07-warranty.md": "warranty",
    "08-order-changes-and-cancellations.md": "order_changes",
    "09-trailplus-membership.md": "membership",
    "10-gift-cards-and-price-adjustments.md": "gift_cards",
    "11-product-care.md": "product_care",
    "12-breeze-tumbler-product-card.md": "product_care",
}

_TOPIC_HINT_TEXT: dict[Topic, str] = {
    "shipping_international": "international shipping",
    "shipping_domestic": "domestic shipping",
    "returns": "returns policy",
    "warranty": "warranty",
    "membership": "TrailPlus membership",
    "gift_cards": "gift cards",
    "product_care": "product care",
    "order_changes": "order changes and cancellations",
}
_WORD_RE = re.compile(r"\b\w+\b")

def _contains_keyword(text_lower: str, keywords: set[str]) -> bool:
    words = set(_WORD_RE.findall(text_lower))
    return bool(words & keywords)

def _last_human_text(state: AgentState) -> str:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return msg.content
    return ""


def _is_short_followup(text: str) -> bool:
    """Heuristic for "What about Canada?" style follow-ups: short,
    and doesn't restate the topic itself.
    """
    return len(text.split()) <= 8


# ---------------------------------------------------------------------------
# classify_intent
# ---------------------------------------------------------------------------
_ORDER_VERB_PATTERN = re.compile(r"\bif\s+i\s+order\b|\bwhen\s+i\s+order\b|\bi\s+order\b", re.IGNORECASE)

def classify_intent(state: AgentState) -> dict:
    text = _last_human_text(state)
    text_lower = text.lower()

    order_id_match = _ORDER_ID_RE.search(text)
    extracted_id = normalize_order_id(order_id_match.group()) if order_id_match else None

    is_order_verb_usage = bool(_ORDER_VERB_PATTERN.search(text_lower))
    mentions_order_topic = _contains_keyword(text_lower, _ORDER_KEYWORDS) and not is_order_verb_usage

    if extracted_id:
        return {"intent": "order", "current_order_id": extracted_id, "last_order_id": extracted_id}

    if mentions_order_topic and state.get("last_order_id"):
        return {"intent": "order", "current_order_id": state["last_order_id"]}

    if mentions_order_topic and not state.get("last_order_id"):
        return {"intent": "order_missing_id", "current_order_id": None}

    return {"intent": "policy", "current_order_id": None}


# ---------------------------------------------------------------------------
# order_tool_node
# ---------------------------------------------------------------------------

def order_tool_node(state: AgentState) -> dict:
    if state["intent"] == "order_missing_id":
        return {"needs_order_id": True, "tool_result": None}

    order_id = state["current_order_id"]
    settings = get_settings()
    result = lookup_order(order_id, str(settings.orders_path_resolved))

    update: dict = {"tool_result": result, "last_order_id": normalize_order_id(order_id)}

    if not result.found:
        update["handoff"] = True
        update["handoff_reason"] = "order_not_found"

    text_lower = _last_human_text(state).lower()
    if _contains_keyword(text_lower, _PRIVACY_KEYWORDS):
        update["handoff"] = True
        update["handoff_reason"] = "privacy_disclosure_request"

    return update


# ---------------------------------------------------------------------------
# retrieve_node
# ---------------------------------------------------------------------------

def retrieve_node(state: AgentState) -> dict:
    text = _last_human_text(state)
    query = text

    last_topic = state.get("last_topic")
    if last_topic and _is_short_followup(text) and last_topic in _TOPIC_HINT_TEXT:
        query = f"{_TOPIC_HINT_TEXT[last_topic]} — {text}"

    hits = retrieve(query, k=10)

    injection_flags: list[str] = []
    for chunk in hits:
        injection_flags.extend(detect_injection_attempt(chunk.text))
    injection_flags.extend(detect_injection_attempt(text))

    conflicts = check_conflicts(hits[:3])

    update: dict = {"retrieved": hits, "injection_flags": injection_flags, "conflicts": conflicts}

    if conflicts:
        update["handoff"] = True
        update["handoff_reason"] = "source_conflict"

    new_topic = _infer_topic(hits) or last_topic
    if new_topic:
        update["last_topic"] = new_topic

    return update


def _infer_topic(hits) -> Topic | None:
    for chunk in hits:
        topic = _FILENAME_TOPIC.get(chunk.filename)
        if topic:
            return topic
    return None


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------
def extract_text(content) -> str:
    """Normalizes AIMessage.content, which some Gemini SDK versions
    return as a plain string and others return as a list of content
    blocks (e.g. [{"type": "text", "text": "..."}]). Always returns
    plain text.
    """
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
        return "".join(parts)
    return str(content)
# ---------------------------------------------------------------------------
# grounding_check
# ---------------------------------------------------------------------------
_DAMAGE_KEYWORDS = {"broken", "damaged", "defective", "zipper", "torn", "cracked"}
def grounding_check(state: AgentState) -> dict:
    """For policy-intent turns: if nothing authoritative was
    retrieved, the answer must abstain rather than let the LLM fill
    the gap from general knowledge. Order-intent turns are already
    grounded by the tool result and don't route through here.
    """
    if state["intent"] != "policy":
        return {}

    grounded = authoritative_chunks(state.get("retrieved", []))
    if not grounded and not state.get("conflicts"):
        return {"handoff": True, "handoff_reason": "insufficient_information"}

    text_lower = _last_human_text(state).lower()
    if _contains_keyword(text_lower, _DAMAGE_KEYWORDS):
        return {"handoff": True, "handoff_reason": "damage_review_required"}

    return {}


# ---------------------------------------------------------------------------
# respond_node — the only node that calls the LLM
# ---------------------------------------------------------------------------

def respond_node(state: AgentState) -> dict:
    if state.get("needs_order_id"):
        text = "Could you share your order ID (e.g. ORD-1007) so I can look that up?"
        return {"messages": [AIMessage(content=text)]}

    context_parts: list[str] = []

    if state.get("tool_result") is not None:
        context_parts.append(f"<tool_result>\n{state['tool_result'].model_dump_json()}\n</tool_result>")

    grounded_chunks = authoritative_chunks(state.get("retrieved", []))
    for chunk in grounded_chunks:
        context_parts.append(
            f"<retrieved_context source=\"{chunk.filename}\" heading=\"{chunk.heading}\">\n"
            f"{chunk.text}\n</retrieved_context>"
        )

    if state.get("conflicts"):
        for c in state["conflicts"]:
            context_parts.append(
                f"<source_conflict>\n"
                f"{c.rule.description}\n"
                f"Source A: {c.chunk_a.filename}\nSource B: {c.chunk_b.filename}\n"
                f"</source_conflict>"
            )

    if not context_parts and not state.get("needs_order_id"):
        context_parts.append(
            "<retrieved_context>No relevant authoritative content was found for this question.</retrieved_context>"
        )

    llm = get_llm()
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        *state.get("messages", []),
        HumanMessage(content="Untrusted context for this turn:\n\n" + "\n\n".join(context_parts)),
    ]

    ai_response = llm.invoke(messages)
    response_text = extract_text(ai_response.content)

    if state.get("tool_result") is not None:
        # defense-in-depth: even though the sanitizer already stripped
        # these, scrub any literal PII that might somehow still appear
        forbidden = []  # populated from the raw record if the caller has it; sanitized result has none
        response_text, _leaked = scrub_response(response_text, forbidden)

    return {"messages": [AIMessage(content=response_text)]}