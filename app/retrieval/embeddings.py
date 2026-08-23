"""Local HuggingFace embedding model. Chosen over a hosted API
(Gemini embeddings) after hitting repeated model-availability/version
issues with Google's embedding endpoint — running locally removes
network calls and API quota as a point of failure for indexing and
for retrieval at query time.
"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

@lru_cache
def get_embeddings() -> HuggingFaceEmbeddings:
    # cached so the model is loaded into memory once per process,
    # not once per call (loading has real startup cost)
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")