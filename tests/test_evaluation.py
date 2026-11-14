import json
from pathlib import Path
import pytest
from conftest import make_doc, FixedModel
from evaluation import EvalCase, retrieval_metrics, audit_citations, evaluate_retrieval, load_cases, write_report, compare_agents
from retrieval import Retriever
from repo_service import open_repository


def test_recall_mrr_respect_rank_and_deduplicate_gold():
    hits = [{"metadata":{"file":"a.py","name":name}} for name in ["no","yes","yes"]]
    gold = [{"file":"a.py","name":"yes"},{"file":"a.py","name":"also"}]
    assert retrieval_metrics(hits,gold,3) == {"recall_at_k":0.5,"mrr":0.5,"hit_at_k":1.0}


def test_citations_require_real_lines_and_context(sample_repo):
    audit = audit_citations("[lib.py:L1-L2] [lib.py:L4-L5] [lib.py:L99] [../outside.py:L1]", sample_repo,
                           [{"file":"lib.py","start_line":1,"end_line":2}])
    assert audit["citation_accuracy"] == 0.5
    assert audit["citation_context_accuracy"] == 0.25
    assert audit_citations("没有引用",sample_repo,[])["citation_accuracy"] is None


def test_contiguous_context_intervals_support_one_citation(sample_repo):
    sources = [{"file":"lib.py","start_line":1,"end_line":2},{"file":"lib.py","start_line":3,"end_line":5}]
    assert audit_citations("[lib.py:L1-L5]",sample_repo,sources)["citation_context_accuracy"] == 1


def test_three_retrieval_strategies_report_and_progress(tmp_path):
    docs = [make_doc("first"),make_doc("second"),make_doc("third")]
    retriever = Retriever(docs,[[1,0],[0,1],[-1,0]],FixedModel())
    cases = [EvalCase(id="q",question="first",relevant=[{"file":"lib.py","name":"first"}])]
    updates = []
    report = evaluate_retrieval(retriever,cases,k=2,progress=lambda *args:updates.append(args))
    assert len(report["rows"]) == 3
    assert len(updates) == 3 and updates[-1][0:2] == (3,3)
    assert all(r["recall_at_k"] == 1 and r["citation_accuracy"] is None for r in report["rows"])
    path = write_report(report,tmp_path / "output")
    assert "N/A" in path.read_text(encoding="utf-8")
    assert json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))["case_count"] == 1


def test_thirty_case_labels_exist_in_current_corpus():
    root = Path(__file__).resolve().parents[1]
    assert len(load_cases(root / "eval_cases.jsonl",open_repository(root).documents)) == 30


def test_invalid_dataset_labels_are_rejected(tmp_path):
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps({"id":"q","question":"q","relevant":[{"file":"missing.py","name":"missing"}]}))
    with pytest.raises(ValueError,match="标签"):
        load_cases(path,[make_doc("first")])


def test_agent_evaluation_keeps_failed_cases_in_report(retriever):
    cases = [EvalCase(id="q",question="leaf?",relevant=[{"file":"lib.py","name":"leaf"}])]
    def fail(*args,**kwargs):
        raise RuntimeError("provider unavailable")
    report = compare_agents(retriever,cases,runner=fail)
    assert len(report["rows"]) == 2
    assert all(s["coverage"] == 0 and s["errors"] == 1 for s in report["summary"])
