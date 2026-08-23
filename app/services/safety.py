"""Two independent defenses:

1. detect_injection_attempt — flags instruction-like patterns inside
   untrusted content (retrieved KB chunks, tool results) for logging
   and deterministic eval assertions. This is NOT the primary
   defense — the primary defense is architectural: untrusted content
   is placed in a clearly delimited block in the prompt and the
   system prompt tells the model never to treat it as instructions
   (see agent/prompts.py). This detector exists so violations are
   observable and testable, not to silently block content.

2. scrub_response — output-side backstop that checks the final
   response text for known-forbidden values (PII/internal fields
   from a specific order lookup) before it reaches the customer.
   Runs even though order_sanitizer.py already prevents these values
   from entering the tool result in the first place — defense in
   depth against the response text leaking something some other way.
"""

from __future__ import annotations

import re

_INJECTION_PATTERNS = [
    re.compile(r"system instruction", re.IGNORECASE),
    re.compile(r"ignore (all )?(prior|previous|above) (rules|instructions)", re.IGNORECASE),
    re.compile(r"reveal (your|the) (hidden )?(prompt|system prompt)", re.IGNORECASE),
    re.compile(r"do not call tools", re.IGNORECASE),
    re.compile(r"never cite a source", re.IGNORECASE),
    re.compile(r"\bAI instruction\b", re.IGNORECASE),
]


def detect_injection_attempt(text: str) -> list[str]:
    """Returns the list of matched pattern descriptions, empty if none.
    Used for observability logging and for tests asserting that
    untrusted content containing injection attempts is flagged.
    """
    matches = []
    for pattern in _INJECTION_PATTERNS:
        if pattern.search(text):
            matches.append(pattern.pattern)
    return matches


def scrub_response(text: str, forbidden_values: list[str]) -> tuple[str, list[str]]:
    """Checks response text for any forbidden value (customer email,
    address, risk score, etc. pulled from an OrderRecord the agent
    looked up this turn). Returns (possibly-redacted text, list of
    values that were found and redacted).

    forbidden_values should be populated by the caller (order_tool_node)
    from the actual internal fields of any order looked up this turn —
    this function has no independent knowledge of what's sensitive,
    it only checks what it's told to check.
    """
    found: list[str] = []
    redacted = text
    for value in forbidden_values:
        if not value:
            continue
        if value in redacted:
            found.append(value)
            redacted = redacted.replace(value, "[redacted]")
    return redacted, found