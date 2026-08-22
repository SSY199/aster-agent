"""Splits a KBDocument's markdown body into heading-scoped chunks.

Each chunk carries the document's full metadata plus its own heading
path, so downstream precedence/retrieval never needs to re-fetch the
source document — the chunk is self-describing.
"""

from __future__ import annotations

import re

from pydantic import BaseModel

from app.retrieval.loader import KBDocument

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)


class Chunk(BaseModel):
    chunk_id: str  # f"{filename}#{heading_slug}"
    filename: str
    heading: str | None  # nearest heading above this text, if any
    text: str

    # metadata carried from the parent document, needed for precedence
    document_id: str | None = None
    status: str | None = None
    policy_authority: str | None = None
    supersedes: str | None = None


def _slugify(heading: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")


def chunk_document(doc: KBDocument) -> list[Chunk]:
    """Split on ##/### headings. Text before the first heading (if any)
    becomes its own chunk with heading=None, since a top-level intro
    paragraph is still retrievable content.
    """
    matches = list(_HEADING_RE.finditer(doc.content))

    if not matches:
        text = doc.content.strip()
        if not text:
            return []
        return [_make_chunk(doc, heading=None, text=text)]

    chunks: list[Chunk] = []

    # leading text before the first heading
    leading = doc.content[: matches[0].start()].strip()
    if leading:
        chunks.append(_make_chunk(doc, heading=None, text=leading))

    for i, m in enumerate(matches):
        heading = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(doc.content)
        body = doc.content[start:end].strip()
        text = f"{heading}\n\n{body}" if body else heading
        chunks.append(_make_chunk(doc, heading=heading, text=text))

    return chunks


def _make_chunk(doc: KBDocument, *, heading: str | None, text: str) -> Chunk:
    slug = _slugify(heading) if heading else "intro"
    return Chunk(
        chunk_id=f"{doc.filename}#{slug}",
        filename=doc.filename,
        heading=heading,
        text=text,
        document_id=doc.document_id,
        status=doc.status,
        policy_authority=doc.policy_authority,
        supersedes=doc.supersedes,
    )


def chunk_documents(docs: list[KBDocument]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(doc))
    return chunks