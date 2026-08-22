"""BM25 sparse retrieval over Chunks — catches exact-term matches
(e.g. 'ORD-1007', '30 calendar days') that embeddings can blur."""

from __future__ import annotations

from rank_bm25 import BM25Okapi

from app.retrieval.chunker import Chunk


class BM25Store:
    def __init__(self, chunks: list[Chunk]) -> None:
        self.chunks = chunks
        self._tokenized = [self._tokenize(c.text) for c in chunks]
        self._bm25 = BM25Okapi(self._tokenized)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    def search(self, query: str, k: int = 5) -> list[tuple[Chunk, float]]:
        scores = self._bm25.get_scores(self._tokenize(query))
        ranked = sorted(zip(self.chunks, scores), key=lambda x: x[1], reverse=True)
        return ranked[:k]