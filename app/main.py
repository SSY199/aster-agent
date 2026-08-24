"""Minimal FastAPI wrapper around the agent graph. One endpoint:
POST /chat. Session state (message history) is kept in memory,
keyed by session_id — fine for this assignment's scope (no auth,
no persistence needed per the brief).
"""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from langchain_core.messages import HumanMessage

from app.agent.graph import run_turn
from app.agent.state import AgentState, initial_state
from app.retrieval.precedence import authoritative_chunks
from app.schemas.chat import ChatRequest, ChatResponse, SourceRef

app = FastAPI(title="Aster & Row Support Agent")

# In-memory session store: session_id -> AgentState.
# Not persisted across restarts — acceptable per assignment scope
# ("no production deployment infrastructure" required).
_sessions: dict[str, AgentState] = {}


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    if not req.message.strip():
        raise HTTPException(status_code=400, detail="message must not be empty")

    state = _sessions.get(req.session_id) or initial_state(req.session_id)
    state["messages"] = state.get("messages", []) + [HumanMessage(content=req.message)]

    try:
        state = run_turn(req.session_id, state, req.message)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"agent error: {e}") from e

    _sessions[req.session_id] = state

    answer = state["messages"][-1].content if state.get("messages") else ""
    sources = [
        SourceRef(filename=c.filename, heading=c.heading)
        for c in authoritative_chunks(state.get("retrieved", []))
    ]

    return ChatResponse(
        answer=answer,
        sources=sources,
        handoff=state.get("handoff", False),
        handoff_reason=state.get("handoff_reason"),
    )


@app.delete("/chat/{session_id}")
def reset_session(session_id: str) -> dict:
    _sessions.pop(session_id, None)
    return {"status": "cleared"}