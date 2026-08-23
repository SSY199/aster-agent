from app.retrieval.hybrid_retriever import RetrievedChunk
from app.retrieval.precedence import authoritative_chunks, is_authoritative, rank_by_authority


def _chunk(**kwargs) -> RetrievedChunk:
    defaults = dict(
        chunk_id="x", filename="x.md", heading="h", text="t", score=1.0,
        document_id=None, status=None, policy_authority=None, supersedes=None,
    )
    defaults.update(kwargs)
    return RetrievedChunk(**defaults)


def test_active_official_is_authoritative():
    c = _chunk(status="active", policy_authority="official")
    assert is_authoritative(c) is True


def test_missing_metadata_is_not_authoritative():
    c = _chunk()
    assert is_authoritative(c) is False


def test_superseded_status_is_not_authoritative():
    c = _chunk(status="superseded", policy_authority="official")
    assert is_authoritative(c) is False


def test_current_policy_ranks_above_legacy():
    current = _chunk(
        chunk_id="current", document_id="RET-2026-01",
        status="active", policy_authority="official",
    )
    legacy = _chunk(
        chunk_id="legacy", document_id="RET-2024-01",
        status="superseded", policy_authority="official",
        supersedes=None,
    )
    ranked = rank_by_authority([legacy, current])
    assert ranked[0].chunk_id == "current"


def test_explicit_supersedes_demotes_even_if_status_missing():
    current = _chunk(
        chunk_id="current", document_id="RET-2026-01",
        status="active", policy_authority="official",
        supersedes="RET-2024-01",
    )
    stale = _chunk(
        chunk_id="stale", document_id="RET-2024-01",
        status=None, policy_authority="official",  # ambiguous/missing status
    )
    ranked = rank_by_authority([stale, current])
    assert ranked[0].chunk_id == "current"
    assert ranked[-1].chunk_id == "stale"


def test_internal_migration_notes_excluded_from_authoritative():
    note = _chunk(status="active", policy_authority="internal")
    assert authoritative_chunks([note]) == []


def test_authoritative_chunks_excludes_superseded_document():
    current = _chunk(
        chunk_id="current", document_id="RET-2026-01",
        status="active", policy_authority="official", supersedes="RET-2024-01",
    )
    legacy = _chunk(
        chunk_id="legacy", document_id="RET-2024-01",
        status="active", policy_authority="official",  # even if mismarked active
    )
    result = authoritative_chunks([legacy, current])
    ids = [c.chunk_id for c in result]
    assert "current" in ids
    assert "legacy" not in ids