"""FAISS-backed dense retrieval over Chunks."""

from __future__ import annotations

from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from app.retrieval.chunker import Chunk
from app.retrieval.embeddings import get_embeddings


def _chunk_to_lc_document(chunk: Chunk) -> Document:
    # metadata carried so a raw FAISS hit can be mapped back to a
    # Chunk without a second lookup
    return Document(
        page_content=chunk.text,
        metadata={
            "chunk_id": chunk.chunk_id,
            "filename": chunk.filename,
            "heading": chunk.heading,
            "document_id": chunk.document_id,
            "status": chunk.status,
            "policy_authority": chunk.policy_authority,
            "supersedes": chunk.supersedes,
        },
    )


def build_faiss_index(chunks: list[Chunk]) -> FAISS:
    docs = [_chunk_to_lc_document(c) for c in chunks]
    return FAISS.from_documents(docs, get_embeddings())


def save_faiss_index(index: FAISS, path: str | Path) -> None:
    index.save_local(str(path))


def load_faiss_index(path: str | Path) -> FAISS:
    return FAISS.load_local(
        str(path), get_embeddings(), allow_dangerous_deserialization=True
    )


def vector_search(index: FAISS, query: str, k: int = 5) -> list[tuple[Document, float]]:
    """Returns (document, similarity_score) pairs, higher score = more similar."""
    # FAISS returns L2 distance by default; convert to a similarity-like
    # score so it's comparable/combinable with BM25 scores downstream.
    results = index.similarity_search_with_score(query, k=k)
    return [(doc, 1.0 / (1.0 + score)) for doc, score in results]