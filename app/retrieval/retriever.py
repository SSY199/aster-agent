"""Single entrypoint the agent graph calls for policy retrieval.
Wraps: load indexes -> hybrid search -> authority ranking, so nodes
never touch FAISS/BM25/precedence directly.
"""

from __future__ import annotations

import pickle
from functools import lru_cache

from app.config import get_settings
from app.retrieval.bm25_store import BM25Store
from app.retrieval.chunker import Chunk
from app.retrieval.hybrid_retriever import HybridRetriever, RetrievedChunk
from app.retrieval.precedence import rank_by_authority
from app.retrieval.vector_store import load_faiss_index


@lru_cache
def _load_retriever() -> HybridRetriever:
    """Loads pre-built FAISS + BM25 indexes from storage/. Requires
    scripts/build_index.py to have been run first — raises a clear
    error otherwise rather than a confusing file-not-found deep in
    FAISS internals.
    """
    settings = get_settings()

    if not settings.faiss_index_path_resolved.exists():
        raise RuntimeError(
            "FAISS index not found. Run `uv run python -m scripts.build_index` first."
        )
    if not settings.bm25_index_path_resolved.exists():
        raise RuntimeError(
            "BM25 index not found. Run `uv run python -m scripts.build_index` first."
        )

    faiss_index = load_faiss_index(settings.faiss_index_path_resolved)

    with open(settings.bm25_index_path_resolved, "rb") as f:
        bm25_store: BM25Store = pickle.load(f)

    chunks_by_id: dict[str, Chunk] = {c.chunk_id: c for c in bm25_store.chunks}

    return HybridRetriever(faiss_index, bm25_store, chunks_by_id)


def retrieve(query: str, k: int = 5) -> list[RetrievedChunk]:
    """Returns chunks ranked by authority (active+official first),
    then by hybrid retrieval score within each tier. Superseded/
    non-authoritative chunks are included, not dropped, so callers
    (e.g. a node explaining "there's a legacy version too") can still
    see them — filtering to authoritative-only happens at the call
    site via precedence.authoritative_chunks when needed.
    """
    retriever = _load_retriever()
    hits = retriever.retrieve(query, k=k)
    return rank_by_authority(hits)