"""Gemini embedding client wrapper. Isolated in its own file so the
rest of retrieval never imports langchain_google_genai directly —
makes it easy to swap providers or mock in tests.
"""

from __future__ import annotations

from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import get_settings


def get_embeddings() -> GoogleGenerativeAIEmbeddings:
    settings = get_settings()
    return GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        api_key=settings.google_api_key,
    )