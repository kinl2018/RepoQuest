"""Exact lookup precedes similarity ranking; IDs address the supplied corpus."""
from collections import defaultdict
import re


def build_symbol_index(documents) -> dict[str, list[int]]:
    index = defaultdict(list)
    for doc_id, doc in enumerate(documents):
        meta = doc.metadata
        if meta.get("kind") not in {"function", "method", "class"}:
            continue
        name = meta["name"]
        for alias in {name, name.rsplit(".", 1)[-1], f"{meta['file']}::{name}"}:
            index[alias].append(doc_id)
    return dict(index)


def extract_explicit_symbols(question: str, symbol_index: dict) -> list[str]:
    matches, occupied = [], []
    for name in sorted(symbol_index, key=lambda s: (-len(s), s)):
        pattern = rf"(?<![A-Za-z0-9_./:]){re.escape(name)}(?![A-Za-z0-9_.])"
        for match in re.finditer(pattern, question):
            if any(a <= match.start() and match.end() <= b for a, b in occupied):
                continue
            occupied.append(match.span())
            matches.append((match.start(), name))
    return list(dict.fromkeys(name for _, name in sorted(matches)))


def exact_symbol_search(question: str, documents, symbol_index: dict) -> list[dict]:
    results, seen = [], set()
    groups = [symbol_index[name] for name in extract_explicit_symbols(question, symbol_index)]
    # Round-robin split symbols so one large function cannot consume the whole Top-K.
    for offset in range(max(map(len, groups), default=0)):
        for group in groups:
            if offset >= len(group) or group[offset] in seen:
                continue
            doc_id = group[offset]
            seen.add(doc_id)
            doc = documents[doc_id]
            results.append({"doc_id": doc_id, "score": 1.0, "metadata": doc.metadata,
                            "content": doc.page_content, "retrieval_type": "exact_symbol"})
    return results


def merge_retrieval_results(exact_results: list[dict], hybrid_results: list[dict], k: int = 5) -> list[dict]:
    if k <= 0:
        raise ValueError("k 必须大于 0")
    merged, seen = [], set()
    for result in exact_results + hybrid_results:
        if result["doc_id"] not in seen:
            seen.add(result["doc_id"])
            merged.append(result)
        if len(merged) == k:
            break
    return merged
