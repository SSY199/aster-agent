"""Structured JSON logging: one line per turn, containing everything
the assignment's observability requirement asks for — user message,
retrieved passages+scores, tool calls+sanitized results, final
response, and any handoff/error. Never logs secrets: tool_result is
already the sanitized OrderLookupResult (no PII fields exist on it
to accidentally log), and the raw orders.json/OrderRecord is never
passed to this function.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.agent.state import AgentState

_LOG_DIR = Path(__file__).resolve().parents[2] / "logs"
_LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("aster_agent")
logger.setLevel(logging.INFO)

if not logger.handlers:
    file_handler = logging.FileHandler(_LOG_DIR / "turns.jsonl", encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(stream_handler)


def log_turn(state: AgentState, *, user_message: str, error: str | None = None) -> dict:
    """Builds and writes one structured JSON log line for a completed
    turn. Returns the log dict too, so callers (e.g. a /debug API
    route) can return the same structure without re-parsing the file.
    """
    tool_result = state.get("tool_result")

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "session_id": state.get("session_id"),
        "user_message": user_message,
        "intent": state.get("intent"),
        "retrieved": [
            {
                "filename": c.filename,
                "heading": c.heading,
                "score": round(c.score, 4),
                "status": c.status,
                "policy_authority": c.policy_authority,
            }
            for c in state.get("retrieved", [])
        ],
        "conflicts": [
            {
                "rule": c.rule.name,
                "sources": [c.chunk_a.filename, c.chunk_b.filename],
            }
            for c in state.get("conflicts", [])
        ],
        "tool_call": (
            {
                "name": "order_lookup",
                "order_id": tool_result.order_id,
                "found": tool_result.found,
                "status": tool_result.status,
            }
            if tool_result is not None
            else None
        ),
        "injection_flags": state.get("injection_flags", []),
        "handoff": state.get("handoff", False),
        "handoff_reason": state.get("handoff_reason"),
        "final_response": (
            state["messages"][-1].content if state.get("messages") else None
        ),
        "error": error,
    }

    logger.info(json.dumps(entry, ensure_ascii=False))
    return entry