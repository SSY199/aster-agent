"""Deterministic evaluation runner. Replays each case in
visible-cases.json (and custom-cases.json) through the agent graph,
asserting against the case's `expect` block. No LLM-as-judge for
pass/fail — every check is a substring/keyword/field match against
what the graph actually produced.

Usage:
    uv run python -m evaluation.run_eval
    uv run python -m evaluation.run_eval --file evaluation/custom-cases.json
    uv run python -m evaluation.run_eval --out evaluation/results/final.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from langchain_core.messages import HumanMessage

from app.agent.graph import run_turn
from app.agent.state import initial_state
import time
import re


_EVAL_DIR = Path(__file__).parent

def _normalize(text: str) -> str:
    """Collapses hyphens/dashes to spaces and squashes whitespace, so
    '45-calendar-day' matches a check for '45 calendar days' and
    similar surface variations don't cause false failures.
    """
    text = re.sub(r"[-–—‑]", " ", text)  # added U+2011 (non-breaking hyphen)
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def _text_of(state: dict) -> str:
    if not state.get("messages"):
        return ""
    return state["messages"][-1].content or ""


def _tool_was_called(state: dict) -> bool:
    return state.get("tool_result") is not None


def _keyword_present(concept: str, text_lower: str) -> bool:
    """Coarse heuristic for must_include_concepts: checks that a
    meaningful fraction of the concept's significant words appear in
    the response. Deliberately conservative (not full NLI) — flagged
    as a known limitation in the README. Stopwords are stripped so
    short connector words don't inflate the match.
    """
    stopwords = {
        "the", "a", "an", "is", "are", "does", "not", "to", "of", "and",
        "or", "for", "in", "on", "with", "this", "that",
    }
    words = [w.strip(".,?!") for w in _normalize(concept).split()]
    significant = [w for w in words if w not in stopwords and len(w) > 2]
    if not significant:
        return True
    hits = sum(1 for w in significant if w in text_lower)
    return hits / len(significant) >= 0.4


def _run_case(case: dict) -> dict:
    case_id = case["id"]
    category = case.get("category", "uncategorized")
    expect = case.get("expect", {})

    session_id = f"eval-{case_id}"
    state = initial_state(session_id)

    failures: list[str] = []
    final_state = state

    for msg in case["messages"]:
        final_state["messages"] = final_state.get("messages", []) + [
            HumanMessage(content=msg["content"])
        ]
        try:
            final_state = run_turn(session_id, final_state, msg["content"])
            time.sleep(4)
        except Exception as e:
            failures.append(f"exception during turn: {e}")
            return {
                "id": case_id, "category": category, "passed": False,
                "failures": failures, "response": None,
            }

    response_text = _text_of(final_state)
    response_lower = _normalize(response_text)

    if "must_include" in expect:
        for phrase in expect["must_include"]:
            if _normalize(phrase) not in response_lower:
                failures.append(f"missing required phrase: {phrase!r}")

    if "must_not_include" in expect:
        for phrase in expect["must_not_include"]:
            if _normalize(phrase) in response_lower:
                failures.append(f"contains forbidden phrase: {phrase!r}")

    if "must_include_concepts" in expect:
        for concept in expect["must_include_concepts"]:
            if not _keyword_present(concept, response_lower):
                failures.append(f"concept not sufficiently present: {concept!r}")

    if "required_sources" in expect:
        cited_files = {c.filename for c in final_state.get("retrieved", [])}
        for src in expect["required_sources"]:
            if src not in cited_files and src not in response_text:
                failures.append(f"required source not retrieved/cited: {src}")

    if "forbidden_sources_as_authority" in expect:
        grounded_files = {
            c.filename for c in final_state.get("retrieved", [])
            if c.status == "active" and c.policy_authority == "official"
        }
        for src in expect["forbidden_sources_as_authority"]:
            if src in grounded_files:
                failures.append(f"forbidden source used as authority: {src}")

    tool_expect = expect.get("tool")
    tool_called = _tool_was_called(final_state)
    if tool_expect == "not_called" and tool_called:
        failures.append("tool was called but expected not_called")
    elif tool_expect == "order_lookup" and not tool_called:
        failures.append("expected order_lookup tool call, none occurred")
    elif tool_expect == "not_called_without_id":
        if final_state.get("intent") == "order" and not final_state.get("needs_order_id"):
            failures.append("order lookup proceeded without an order ID")

    if "tool_arguments" in expect and tool_called:
        expected_id = expect["tool_arguments"].get("order_id")
        actual_id = final_state["tool_result"].order_id
        if expected_id and expected_id != actual_id:
            failures.append(f"tool called with order_id={actual_id!r}, expected {expected_id!r}")

    if "must_ask_for" in expect:
        if not final_state.get("needs_order_id"):
            failures.append("expected a clarifying question (needs_order_id), got none")

    if "handoff" in expect:
        actual_handoff = final_state.get("handoff", False)
        if actual_handoff != expect["handoff"]:
            failures.append(f"handoff={actual_handoff}, expected {expect['handoff']}")

    return {
        "id": case_id,
        "category": category,
        "passed": len(failures) == 0,
        "failures": failures,
        "response": response_text,
    }


def run_eval(cases_path: Path) -> dict:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))["cases"]
    results = [_run_case(c) for c in cases]

    by_category: dict[str, dict] = {}
    for r in results:
        cat = r["category"]
        by_category.setdefault(cat, {"total": 0, "passed": 0})
        by_category[cat]["total"] += 1
        if r["passed"]:
            by_category[cat]["passed"] += 1

    total = len(results)
    passed = sum(1 for r in results if r["passed"])

    return {
        "summary": {"total": total, "passed": passed, "failed": total - passed},
        "by_category": by_category,
        "cases": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", default=str(_EVAL_DIR / "visible-cases.json"))
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    report = run_eval(Path(args.file))

    print(f"\n{'=' * 60}")
    print(f"EVAL RESULTS: {report['summary']['passed']}/{report['summary']['total']} passed")
    print(f"{'=' * 60}\n")

    for cat, stats in report["by_category"].items():
        print(f"  {cat:30s} {stats['passed']}/{stats['total']}")

    print()
    for case in report["cases"]:
        status = "PASS" if case["passed"] else "FAIL"
        print(f"[{status}] {case['id']} ({case['category']})")
        for f in case["failures"]:
            print(f"    - {f}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()