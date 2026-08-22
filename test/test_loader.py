from pathlib import Path

from app.retrieval.loader import load_kb_directory, load_kb_document

KB_DIR = Path(__file__).parents[1] / "knowledge-base"


def test_loads_all_kb_files():
    docs = load_kb_directory(KB_DIR)
    assert len(docs) == 14


def test_current_returns_policy_parses_front_matter():
    doc = load_kb_document(KB_DIR / "01-returns-policy-current.md")
    assert doc.status == "active"
    assert doc.policy_authority == "official"
    assert doc.supersedes == "RET-2024-01"
    assert "30 calendar days" in doc.content


def test_legacy_policy_marked_not_active():
    doc = load_kb_document(KB_DIR / "02-returns-policy-legacy.md")
    assert doc.status != "active"


def test_content_excludes_front_matter_block():
    doc = load_kb_document(KB_DIR / "01-returns-policy-current.md")
    assert "document_id:" not in doc.content
    assert "---" not in doc.content.split("\n")[0]


def test_filename_derived_from_path_not_metadata():
    doc = load_kb_document(KB_DIR / "01-returns-policy-current.md")
    assert doc.filename == "01-returns-policy-current.md"