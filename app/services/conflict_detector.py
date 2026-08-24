"""Detects known conflicts between currently-active, official chunks
that precedence.py cannot resolve by ranking (neither supersedes the
other). Deliberately NOT a general contradiction detector — that
would be non-deterministic and out of scope. Instead: a small
registry of known conflict pairs, checked by keyword presence in the
actually-retrieved chunks.

Known limitation (documented in README): only catches conflicts
registered here. A genuinely novel conflict the corpus doesn't yet
have would not be caught automatically.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.retrieval.hybrid_retriever import RetrievedChunk
from app.retrieval.precedence import is_authoritative


@dataclass(frozen=True)
class ConflictRule:
    name: str
    filename_a: str
    keyword_a: str
    filename_b: str
    keyword_b: str
    description: str


_KNOWN_CONFLICTS: list[ConflictRule] = [
    ConflictRule(
        name="breeze_tumbler_dishwasher",
        filename_a="11-product-care.md",
        keyword_a="hand-washed",
        filename_b="12-breeze-tumbler-product-card.md",
        keyword_b="dishwasher safe",
        description=(
            "Product Care Guide says the Breeze Tumbler body must be "
            "hand-washed; the product card says all components are "
            "dishwasher safe."
        ),
    ),
]


@dataclass(frozen=True)
class ConflictResult:
    rule: ConflictRule
    chunk_a: RetrievedChunk
    chunk_b: RetrievedChunk


def check_conflicts(chunks: list[RetrievedChunk]) -> list[ConflictResult]:
    """Returns any registered conflicts where BOTH sides are present
    among the retrieved chunks and both are currently authoritative.
    A conflict only fires if precedence can't already resolve it —
    if one side were superseded, precedence.rank_by_authority would
    have handled it, so we only need to check authoritative chunks.
    """
    relevant = [c for c in chunks if c.score >= 0.02]
    authoritative = [c for c in relevant if is_authoritative(c)]
    by_filename: dict[str, list[RetrievedChunk]] = {}
    for c in authoritative:
        by_filename.setdefault(c.filename, []).append(c)

    results: list[ConflictResult] = []
    for rule in _KNOWN_CONFLICTS:
        chunk_a = _find_matching(by_filename.get(rule.filename_a, []), rule.keyword_a)
        chunk_b = _find_matching(by_filename.get(rule.filename_b, []), rule.keyword_b)
        if chunk_a and chunk_b:
            results.append(ConflictResult(rule=rule, chunk_a=chunk_a, chunk_b=chunk_b))

    return results


def _find_matching(chunks: list[RetrievedChunk], keyword: str) -> RetrievedChunk | None:
    for c in chunks:
        if keyword.lower() in c.text.lower():
            return c
    return None