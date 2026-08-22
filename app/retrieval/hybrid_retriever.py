"""Combines FAISS (dense) + BM25 (sparse) results via Reciprocal Rank
Fusion, so a query benefits from both semantic and exact-term matching
without needing to tune a blend weight.
"""

from __future__ import annotations

from langchain_community.vectorstores import FAISS

from app.retrieval.bm25_store import BM25Store
from app.retrieval.chunker import Chunk
from app.retrieval.vector_store import vector_search

_RRF_K = 60  # standard RRF constant


class RetrievedChunk(Chunk):
    score: float = 0.0


class HybridRetriever:
    def __init__(self, faiss_index: FAISS, bm25_store: BM25Store, chunks_by_id: dict[str, Chunk]):
        self.faiss_index = faiss_index
        self.bm25_store = bm25_store
        self.chunks_by_id = chunks_by_id

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        dense_hits = vector_search(self.faiss_index, query, k=k * 2)
        sparse_hits = self.bm25_store.search(query, k=k * 2)

        rrf_scores: dict[str, float] = {}

        for rank, (doc, _score) in enumerate(dense_hits):
            chunk_id = doc.metadata["chunk_id"]
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)

        for rank, (chunk, _score) in enumerate(sparse_hits):
            rrf_scores[chunk.chunk_id] = rrf_scores.get(chunk.chunk_id, 0.0) + 1.0 / (_RRF_K + rank + 1)

        ranked_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]

        return [
            RetrievedChunk(**self.chunks_by_id[chunk_id].model_dump(), score=score)
            for chunk_id, score in ranked_ids
            if chunk_id in self.chunks_by_id
        ]