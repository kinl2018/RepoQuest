"""Budgeted context with explicit provenance and observable protected-evidence loss."""
import re
from evidence_store import evidence_id, merge_evidence


def token_upper_bound(text: str) -> int:
    # One token cannot require less than one byte in byte-level tokenizers.
    # This deliberately conservative bound avoids a remote tokenizer dependency.
    return len(text.encode("utf-8"))


def render_evidence(result):
    m = result["metadata"]
    return (f"[证据 {evidence_id(result)}]\n文件：{m['file']}\n符号：{m['name']}\n"
            f"源码范围：L{m['start_line']}-L{m['end_line']}\n```python\n{result['content']}\n```")


def build_context(results, history=(), *, protected_ids=(), max_tokens=12000, tool_evidence=None):
    if max_tokens < 1:
        raise ValueError("Context budget 必须大于 0")
    all_results = merge_evidence(list(history), list(results))
    by_id = {evidence_id(r): r for r in all_results}
    observations = tool_evidence or {}
    order = list(dict.fromkeys([*protected_ids, *reversed(observations),
                               *(evidence_id(r) for r in results),
                               *(evidence_id(r) for r in reversed(history))]))
    sections, sources, selected_ids = [], [], set()
    omitted = []
    # Protected source and tool evidence share one priority order and one budget.
    for eid in order:
        if eid in observations:
            block = f"[证据 {eid}]\n{observations[eid]}"
            source = {"evidence_id": eid, "origin": "tool"}
        elif eid in by_id:
            result = by_id[eid]
            block = render_evidence(result)
            source = {**result["metadata"], "evidence_id": eid,
                      "origin": result.get("retrieval_type", "history")}
        else:
            continue
        if token_upper_bound("\n\n".join(sections + [block])) > max_tokens:
            omitted.append(eid)
            continue
        sections.append(block)
        sources.append(source)
        selected_ids.add(eid)
    return {"context": "\n\n".join(sections), "context_sources": sources,
            "context_stats": {"token_upper_bound": token_upper_bound("\n\n".join(sections)),
                              "budget": max_tokens, "omitted_ids": omitted,
                              "missing_protected_ids": [i for i in protected_ids if i not in selected_ids]}}


def build_multi_rag_context(root, results, documents, max_results=2, max_callers=2,
                            source_trace=None, max_tokens=12000):
    # Graph expansion happens once in Retriever, before budget accounting.
    pack = build_context(results[:max_results], max_tokens=max_tokens)
    if source_trace is not None:
        source_trace.extend(pack["context_sources"])
    return pack["context"]


def select_context_results(current_results, evidence_history, latest_k=2, history_k=2, protected_results=None):
    if latest_k < 0 or history_k < 0:
        raise ValueError("Context 数量限制不能为负数")
    ordered = merge_evidence(protected_results or [], current_results[:latest_k])
    return merge_evidence(ordered, evidence_history)[:latest_k + history_k]


def prioritize_explicit_symbol_hits(results, unresolved):
    text = "\n".join(item["question"] + "\n" + item.get("reason", "") for item in unresolved)
    def priority(result):
        name = result["metadata"]["name"].split(".")[-1]
        return not (name.isidentifier() and re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text))
    return sorted(results, key=priority)
