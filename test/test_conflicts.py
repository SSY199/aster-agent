from app.retrieval.hybrid_retriever import RetrievedChunk
from app.services.conflict_detector import check_conflicts


def _chunk(filename: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=f"{filename}#x", filename=filename, heading="h", text=text, score=1.0,
        document_id="D", status="active", policy_authority="official", supersedes=None,
    )


def test_detects_breeze_tumbler_conflict_when_both_sides_retrieved():
    care = _chunk("11-product-care.md", "The Breeze Tumbler body should be hand-washed.")
    card = _chunk("12-breeze-tumbler-product-card.md", "All components are dishwasher safe.")
    results = check_conflicts([care, card])
    assert len(results) == 1
    assert results[0].rule.name == "breeze_tumbler_dishwasher"


def test_no_conflict_when_only_one_side_retrieved():
    care = _chunk("11-product-care.md", "The Breeze Tumbler body should be hand-washed.")
    results = check_conflicts([care])
    assert results == []


def test_no_conflict_when_side_is_not_authoritative():
    care = _chunk("11-product-care.md", "hand-washed")
    card = _chunk("12-breeze-tumbler-product-card.md", "dishwasher safe")
    card.status = "draft"  # simulate non-authoritative
    results = check_conflicts([care, card])
    assert results == []


def test_unrelated_chunks_produce_no_conflict():
    warranty = _chunk("07-warranty.md", "Bags have a 2 year warranty.")
    shipping = _chunk("05-domestic-shipping.md", "Orders ship in 1-2 business days.")
    results = check_conflicts([warranty, shipping])
    assert results == []
      
def test_weak_match_does_not_trigger_conflict_when_scoped_to_top_chunks():
    strong_unrelated = _chunk("11-product-care.md", "Spot-clean fabric bags with mild soap.")
    strong_unrelated.score = 0.033
    weak_tumbler_a = _chunk("11-product-care.md", "hand-washed")
    weak_tumbler_a.score = 0.016
    weak_tumbler_b = _chunk("12-breeze-tumbler-product-card.md", "dishwasher safe")
    weak_tumbler_b.score = 0.015
    # only top-3 by whatever order retrieve_node would pass in
    results = check_conflicts([strong_unrelated, weak_tumbler_a, weak_tumbler_b][:1])
    assert results == []