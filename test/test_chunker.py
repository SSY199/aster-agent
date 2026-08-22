from app.retrieval.chunker import chunk_document, chunk_documents
from app.retrieval.loader import load_kb_directory, load_kb_document
from pathlib import Path

KB_DIR = Path(__file__).parents[1] / "knowledge-base"


def test_splits_on_headings():
    doc = load_kb_document(KB_DIR / "01-returns-policy-current.md")
    chunks = chunk_document(doc)
    headings = [c.heading for c in chunks]
    assert "Standard return window" in headings


def test_every_chunk_carries_source_metadata():
    doc = load_kb_document(KB_DIR / "01-returns-policy-current.md")
    chunks = chunk_document(doc)
    for c in chunks:
        assert c.filename == "01-returns-policy-current.md"
        assert c.status == "active"
        assert c.policy_authority == "official"


def test_chunk_id_is_unique_per_chunk():
    doc = load_kb_document(KB_DIR / "01-returns-policy-current.md")
    chunks = chunk_document(doc)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_text_contains_relevant_content():
    doc = load_kb_document(KB_DIR / "01-returns-policy-current.md")
    chunks = chunk_document(doc)
    standard_chunk = next(c for c in chunks if c.heading == "Standard return window")
    assert "30 calendar days" in standard_chunk.text


def test_chunk_documents_covers_all_kb_files():
    docs = load_kb_directory(KB_DIR)
    chunks = chunk_documents(docs)
    filenames = {c.filename for c in chunks}
    assert len(filenames) == 14