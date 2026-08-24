"""Gemini chat model for response generation. Isolated here so only
respond_node depends on it — every other node is deterministic and
needs no LLM call at all, which is what makes them cheaply
unit-testable.
"""

from __future__ import annotations

from functools import lru_cache
from pydantic import SecretStr

# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from app.config import get_settings

# @lru_cache
# def get_llm() -> ChatGoogleGenerativeAI:
#     settings = get_settings()
#     return ChatGoogleGenerativeAI(
#         model=settings.gemini_model,
#         google_api_key=settings.google_api_key,
#         temperature=0.1,  # low — this is a grounded support agent, not creative writing
#     )


@lru_cache
def get_llm() -> ChatGroq:
    settings = get_settings()
    return ChatGroq(
        model=settings.groq_model,
        api_key=SecretStr(settings.groq_api_key),
        temperature=0.1,
        timeout=30,
    )
