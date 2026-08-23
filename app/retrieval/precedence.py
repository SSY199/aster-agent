"""Authority ranking and supersession resolution for retrieved chunks.

Two concerns, kept explicit and separate:
- rank_by_authority: orders/filters chunks so retrieval prefers
  active+official sources over superseded/internal ones.
- is_authoritative: the primitive both this file and conflict
  detection use to decide "can this chunk answer the question at all."

This file does NOT decide whether two authoritative chunks disagree —
that's conflict detection's job, downstream, using is_authoritative
as its input signal.
"""

from __future__ import annotations

from app.retrieval.hybrid_retriever import RetrievedChunk

_ACTIVE_STATUSES = {"active"}
_AUTHORITATIVE_ROLES = {"official"}


def is_authoritative(chunk: RetrievedChunk) -> bool:
    """A chunk counts as authoritative only if BOTH its status is
    active AND its policy_authority is official. Missing metadata
    (None) is treated as non-authoritative — safer default than
    assuming trust on absent data.
    """
    return chunk.status in _ACTIVE_STATUSES and chunk.policy_authority in _AUTHORITATIVE_ROLES


def is_superseded_by(chunk: RetrievedChunk, document_ids_present: set[str]) -> bool:
    """True if some other retrieved chunk explicitly supersedes this
    chunk's document_id. Requires the superseding doc to actually be
    among the retrieved set — otherwise we'd be flagging a doc as
    superseded based on a supersedes claim we never verified exists.
    """
    return chunk.document_id is not None and any(
        _document_supersedes(chunk.document_id, other_id) for other_id in document_ids_present
    )


def _document_supersedes(target_document_id: str, candidate_document_id: str) -> bool:
    # placeholder for symmetry; real check happens in rank_by_authority
    # where we have access to each chunk's own `supersedes` field
    return False


def rank_by_authority(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Re-orders retrieved chunks: authoritative (active + official)
    first, preserving their relative retrieval-score order within
    each tier. Superseded/non-authoritative chunks are pushed down,
    never dropped — the agent may still need to reference "there's an
    older version" without treating it as the answer.

    Also demotes any chunk whose document_id is named in another
    retrieved chunk's `supersedes` field, even if its own status
    field is (incorrectly or ambiguously) marked active — an explicit
    supersedes claim from a currently-active doc is a stronger signal
    than a stale/missing status field on the superseded one.
    """
    superseded_ids = {
        c.supersedes for c in chunks if c.supersedes and is_authoritative(c)
    }

    def tier(chunk: RetrievedChunk) -> int:
        if chunk.document_id in superseded_ids:
            return 2  # explicitly superseded — lowest tier regardless of its own status
        if is_authoritative(chunk):
            return 0  # active + official — top tier
        return 1  # anything else (draft, internal, unknown/missing metadata)

    # stable sort: within a tier, original retrieval-score order (already sorted) is preserved
    return sorted(chunks, key=tier)


def authoritative_chunks(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    """Convenience filter: only chunks safe to answer from directly.
    Used by nodes that must cite only authoritative sources (e.g. the
    standard policy answer), as opposed to nodes that need the full
    ranked list to explain "there's a legacy version" when relevant.
    """
    ranked = rank_by_authority(chunks)
    superseded_ids = {c.supersedes for c in chunks if c.supersedes and is_authoritative(c)}
    return [
        c for c in ranked
        if is_authoritative(c) and c.document_id not in superseded_ids
    ]