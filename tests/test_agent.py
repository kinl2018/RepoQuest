import json
import numpy as np
import pytest
from evidence_store import evidence_id
from repo_agent import build_graph
from repo_service import run_investigation
from rag_index import save_index


def one_question(_):
    return [{"id": "q1", "question": "leaf 定义"}]


def source_grader(**kwargs):
    ids = sorted(kwargs["valid_evidence_ids"])
    valid = [i for i in ids if "leaf@" in i]
    return {"q1": {"sufficient": bool(valid), "reason": "有源码" if valid else "缺少源码",
                   "missing_evidence": [] if valid else ["source_code"], "supporting_evidence_ids": valid[:1]}}


def plan_leaf(_):
    return {"tool": "find_symbol", "args": {"name": "lib.py::leaf"}, "reason": "查定义"}


def test_agent_tool_context_grade_generate(retriever):
    graph = build_graph(retriever, planner=plan_leaf, decomposer=one_question, grader=source_grader,
                        answerer=lambda q,c: "定义见 [lib.py:L1-L2]")
    state = graph.invoke({"question": "leaf?"})
    assert state["stop_reason"] == "sufficient"
    assert state["coverage_stats"]["coverage"] == 1
    assert [e["node"] for e in state["trace"]] == ["initialize", "plan", "execute", "context", "grade", "generate"]


def test_metadata_is_validated_before_grading(retriever, tmp_path):
    index_dir = tmp_path / ".repoquest_index"
    save_index(retriever.documents, np.ones((len(retriever.documents), 3)), index_dir)
    def grader(**kwargs):
        assert "index_metadata" in kwargs["valid_evidence_ids"]
        return {"q1": {"sufficient": True, "reason": "真实元数据", "missing_evidence": [],
                       "supporting_evidence_ids": ["index_metadata"]}}
    state = build_graph(retriever, planner=lambda s: {"tool":"index_metadata", "args":{}, "reason":"读取维度"},
                        decomposer=one_question, grader=grader, answerer=lambda q,c:"3").invoke({"question":"维度?"})
    assert state["stop_reason"] == "sufficient"


def test_repeated_tool_call_terminates(retriever):
    graph = build_graph(retriever, planner=lambda s: {"tool":"find_symbol", "args":{"name":"absent"}, "reason":"查找"},
                        decomposer=one_question, max_steps=6)
    state = graph.invoke({"question":"unknown"})
    assert state["stop_reason"] == "repeated_tool_call"
    assert state["steps"] == 1


def test_bad_tool_arguments_do_not_become_supporting_evidence(retriever):
    state = build_graph(retriever, planner=lambda s: {"tool":"read_source", "args":{"file":"../outside.py","start_line":True,"end_line":2}, "reason":"read"},
                        decomposer=one_question, max_steps=1).invoke({"question":"read"})
    assert state["tool_evidence"] == {}
    assert state["stop_reason"] == "step_budget"
    assert "error" in state["last_observation"]["result"]


def test_checkpoint_can_resume_without_reexecuting_tool(retriever, tmp_path):
    options = dict(planner=plan_leaf, decomposer=one_question, grader=source_grader, answerer=lambda q,c:"done")
    path = tmp_path / "checkpoint.sqlite"
    first = run_investigation(retriever, "leaf?", checkpoint_path=path, thread_id="case", interrupt_after=["execute"], **options)
    assert first["run"]["pending_nodes"] == ["context"]
    second = run_investigation(retriever, resume=True, thread_id="case", checkpoint_path=path, **options)
    assert second["answer"] == "done"
    assert second["steps"] == 1
    assert sum(e["node"] == "execute" for e in second["trace"]) == 1
    with pytest.raises(ValueError, match="线程已存在"):
        run_investigation(retriever, "other", thread_id="case", checkpoint_path=path, **options)


def test_resume_rejects_changed_source(retriever, tmp_path):
    options = dict(planner=plan_leaf, decomposer=one_question, grader=source_grader, answerer=lambda q,c:"done")
    path = tmp_path / "checkpoint.sqlite"
    run_investigation(retriever, "leaf?", checkpoint_path=path, thread_id="changed", **options)
    (retriever.root / "new.py").write_text("def new(): pass")
    with pytest.raises(ValueError, match="变化"):
        run_investigation(retriever, resume=True, checkpoint_path=path, thread_id="changed", **options)


def test_grader_cannot_claim_unseen_evidence(retriever):
    def hallucinate(**kwargs):
        return {"q1": {"sufficient":True, "reason":"invented", "missing_evidence":[], "supporting_evidence_ids":["fake"]}}
    graph = build_graph(retriever, planner=plan_leaf, decomposer=one_question, grader=hallucinate)
    with pytest.raises(ValueError, match="不存在"):
        graph.invoke({"question":"leaf?"})


def test_multiround_keeps_earlier_support_and_completes_both_questions(retriever):
    def planner(state):
        name = "lib.py::leaf" if state["steps"] == 0 else "app.py::entry"
        return {"tool":"find_symbol", "args":{"name":name}, "reason":"补充缺失定义"}
    def grader(**kwargs):
        grades = {}
        for qid, symbol in [("q1", "leaf"), ("q2", "entry")]:
            ids = sorted(i for i in kwargs["valid_evidence_ids"] if f"::{symbol}@" in i)
            grades[qid] = {"sufficient":bool(ids), "reason":"有源码" if ids else "缺少定义",
                           "missing_evidence":[] if ids else ["source_code"], "supporting_evidence_ids":ids[:1]}
        return grades
    state = build_graph(retriever, planner=planner,
                        decomposer=lambda _: [{"id":"q1","question":"leaf?"},{"id":"q2","question":"entry?"}],
                        grader=grader, answerer=lambda q,c:"done").invoke({"question":"leaf 和 entry"})
    rounds = [e["coverage"]["coverage"] for e in state["trace"] if "coverage" in e]
    assert rounds == [0.5, 1.0]
    assert state["steps"] == 2
    assert not state["context_stats"]["missing_protected_ids"]
