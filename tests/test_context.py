from conftest import make_doc
from context_expansion import build_context, token_upper_bound
from evidence_store import evidence_id, calculate_novelty


def result(doc):
    return {"metadata": doc.metadata, "content": doc.page_content, "score": 1}


def test_context_protects_previous_answer_under_budget():
    a = result(make_doc("a"))
    b = result(make_doc("b", start=5))
    large = result(make_doc("large", start=20, content="x"*5000))
    pack = build_context([large, b], [a], protected_ids=[evidence_id(a)], max_tokens=1000)
    assert evidence_id(a) in {s["evidence_id"] for s in pack["context_sources"]}
    assert "def b()" in pack["context"]
    assert evidence_id(large) in pack["context_stats"]["omitted_ids"]
    assert token_upper_bound(pack["context"]) <= 1000


def test_unretainable_protected_evidence_is_reported():
    a = result(make_doc("a", content="x"*2000))
    pack = build_context([a], protected_ids=[evidence_id(a)], max_tokens=200)
    assert pack["context_stats"]["missing_protected_ids"] == [evidence_id(a)]
    assert pack["context_sources"] == []


def test_tool_evidence_counts_towards_same_budget():
    pack = build_context([], tool_evidence={"index_metadata": "维度"*500}, max_tokens=500)
    assert pack["context"] == ""
    assert "index_metadata" in pack["context_stats"]["omitted_ids"]


def test_large_tool_observation_cannot_evict_protected_source():
    a = result(make_doc("a"))
    pack = build_context([], [a], protected_ids=[evidence_id(a)], max_tokens=500,
                         tool_evidence={"tool:new": "x"*450})
    assert evidence_id(a) in {s["evidence_id"] for s in pack["context_sources"]}
    assert "tool:new" in pack["context_stats"]["omitted_ids"]


def test_novelty_counts_unique_evidence():
    a = result(make_doc("a"))
    assert calculate_novelty([], [a, a]) == {"retrieved_count": 1, "new_count": 1, "novelty_rate": 1.0}
