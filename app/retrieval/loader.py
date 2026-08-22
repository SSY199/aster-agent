"""Loads knowledge-base/*.md files, splitting YAML front matter from
the markdown body. Downstream retrieval/precedence code depends on
every doc having consistent metadata — this is the single place that
parses it.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

_FRONT_MATTER_DELIM = "---"


class KBDocument(BaseModel):
    """One knowledge-base file: parsed front matter + markdown body."""

    filename: str
    document_id: str | None = None
    title: str | None = None
    status: str | None = None          # "active" | "superseded" | ...
    policy_authority: str | None = None  # "official" | "internal" | ...
    supersedes: str | None = None
    effective_date: str | None = None
    last_reviewed: str | None = None
    audience: str | None = None
    content: str  # markdown body, front matter stripped


def _split_front_matter(raw: str) -> tuple[dict, str]:
    """Return (metadata_dict, body). If no front matter block is
    present, metadata is {} and body is the raw text unchanged —
    loader must not crash on a malformed/missing block, just degrade.
    """
    lines = raw.splitlines()
    if not lines or lines[0].strip() != _FRONT_MATTER_DELIM:
        return {}, raw

    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == _FRONT_MATTER_DELIM:
            fm_text = "\n".join(lines[1:i])
            body = "\n".join(lines[i + 1 :]).lstrip("\n")
            try:
                metadata = yaml.safe_load(fm_text) or {}
                for key in ("effective_date", "last_reviewed"):
                    if key in metadata and hasattr(metadata[key], "isoformat"):
                        metadata[key] = metadata[key].isoformat()
            except yaml.YAMLError:
                metadata = {}
            return metadata, body

    # opening delimiter but no closing one — treat whole thing as body
    return {}, raw


def load_kb_document(path: Path) -> KBDocument:
    raw = path.read_text(encoding="utf-8")
    metadata, body = _split_front_matter(raw)
    return KBDocument(filename=path.name, content=body, **metadata)


def load_kb_directory(kb_dir: str | Path) -> list[KBDocument]:
    kb_dir = Path(kb_dir)
    paths = sorted(kb_dir.glob("*.md"))
    return [load_kb_document(p) for p in paths]