# test/test_retriever.py
import pytest

from app.retrieval.retriever import retrieve

pytestmark = pytest.mark.skipif(
    False,  # flip to False once storage/ is built on your machine / CI
    reason="requires storage/ index built via scripts/build_index.py",
)


def test_standard_return_window_query_hits_current_policy():
    results = retrieve("How long is the return window?", k=5)
    filenames = {c.filename for c in results}
    assert "01-returns-policy-current.md" in filenames


def test_current_policy_ranked_above_legacy_for_same_query():
    results = retrieve("How long is the return window?", k=8)
    current_rank = next(
        i for i, c in enumerate(results) if c.filename == "01-returns-policy-current.md"
    )
    legacy_ranks = [
        i for i, c in enumerate(results) if c.filename == "02-returns-policy-legacy.md"
    ]
    if legacy_ranks:  # only assert ordering if legacy was retrieved at all
        assert current_rank < legacy_ranks[0]


def test_canada_shipping_query_hits_international_doc():
    results = retrieve("Do you ship to Canada?", k=5)
    filenames = {c.filename for c in results}
    assert "06-international-shipping.md" in filenames


def test_warranty_query_hits_warranty_doc():
    results = retrieve("What is the warranty on bags?", k=5)
    filenames = {c.filename for c in results}
    assert "07-warranty.md" in filenames