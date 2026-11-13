"""Stable source identities used by retrieval, context, grading and tracing."""


def evidence_id(result: dict) -> str:
    meta = result.get("metadata", result)
    return f"{meta['file']}::{meta['name']}@L{meta['start_line']}-{meta['end_line']}"


def merge_evidence(history: list[dict], new_results: list[dict]) -> list[dict]:
    merged = {evidence_id(r): r for r in history}
    merged.update({evidence_id(r): r for r in new_results})
    return list(merged.values())


def calculate_novelty(history: list[dict], new_results: list[dict]) -> dict:
    old = {evidence_id(r) for r in history}
    new = {evidence_id(r) for r in new_results}
    count = len(new - old)
    return {"retrieved_count": len(new), "new_count": count,
            "novelty_rate": count / len(new) if new else 0.0}
