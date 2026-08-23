"""One-shot CLI: load KB -> chunk -> build FAISS + BM25 -> save to
storage/. Run once (or whenever knowledge-base/ changes) before
starting the API — the app loads pre-built indexes at startup rather
than re-embedding on every boot.

Usage:
    uv run python scripts/build_index.py
"""

from __future__ import annotations

import pickle

from app.config import get_settings
from app.retrieval.bm25_store import BM25Store
from app.retrieval.chunker import chunk_documents
from app.retrieval.loader import load_kb_directory
from app.retrieval.vector_store import build_faiss_index, save_faiss_index


def main() -> None:
    settings = get_settings()

    print(f"Loading knowledge base from {settings.kb_dir_path}")
    docs = load_kb_directory(settings.kb_dir_path)
    print(f"Loaded {len(docs)} documents")

    chunks = chunk_documents(docs)
    print(f"Produced {len(chunks)} chunks")

    print("Building FAISS index (calls HuggingFace embeddings)...")
    faiss_index = build_faiss_index(chunks)
    settings.faiss_index_path_resolved.parent.mkdir(parents=True, exist_ok=True)
    save_faiss_index(faiss_index, settings.faiss_index_path_resolved)
    print(f"Saved FAISS index to {settings.faiss_index_path_resolved}")

    print("Building BM25 index...")
    bm25_store = BM25Store(chunks)
    settings.bm25_index_path_resolved.parent.mkdir(parents=True, exist_ok=True)
    with open(settings.bm25_index_path_resolved, "wb") as f:
        pickle.dump(bm25_store, f)
    print(f"Saved BM25 index to {settings.bm25_index_path_resolved}")

    print("Done.")


if __name__ == "__main__":
    main()