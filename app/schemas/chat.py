"""Request/response shapes for the chat API."""

from __future__ import annotations

from pydantic import BaseModel


class ChatRequest(BaseModel):
    session_id: str
    message: str


class SourceRef(BaseModel):
    filename: str
    heading: str | None = None


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    handoff: bool
    handoff_reason: str | None = None