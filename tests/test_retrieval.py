import numpy as np
import pytest
from conftest import make_doc, FixedModel
from retrieval import Retriever, reciprocal_rank_fusion
from symbol_index import build_symbol_index, exact_symbol_search
from scanner import scan_python_files
from code_chunker import chunk_repo, split_long_chunk


def test_exact_symbols_include_short_qualified_and_class_names():
    docs = [make_doc("Worker.leaf", "one.py", kind="method"), make_doc("Worker.leaf", "two.py", kind="method"), make_doc("Worker", kind="class")]
    index = build_symbol_index(docs)
    assert len(exact_symbol_search("leaf() 在哪里", docs, index)) == 2
    assert [r["doc_id"] for r in exact_symbol_search("one.py::Worker.leaf", docs, index)] == [0]
    assert len(exact_symbol_search("Worker 类", docs, index)) == 1
    assert not exact_symbol_search("leaflet 或 some_leaf", docs, index)


def test_exact_does_not_starve_second_split_symbol():
    docs = [make_doc("large", start=1), make_doc("large", start=60), make_doc("small", start=100)]
    results = Retriever(docs).search("large small", k=2, strategy="exact")
    assert [r["metadata"]["name"] for r in results] == ["large", "small"]


def test_filters_apply_to_old_corpus_and_vectors():
    docs = [make_doc("prod"), make_doc("test_prod", "tests/test_lib.py"), make_doc("demo", "retry_demo.py")]
    retriever = Retriever(docs, np.eye(3, dtype=np.float32), FixedModel())
    assert len(retriever.documents) == 1
    assert retriever.embeddings.shape == (1, 3)


def test_scanner_prunes_environment_and_chunk_policy(sample_repo):
    hidden = sample_repo / ".venv"
    hidden.mkdir()
    (hidden / "secret.py").write_text("def secret(): pass")
    assert all(".venv" not in f.parts for f in scan_python_files(sample_repo))
    chunks, _ = chunk_repo(sample_repo)
    assert not any(c["file"].startswith("tests/") for c in chunks)
    chunks, _ = chunk_repo(sample_repo, include_tests=True)
    assert any(c["file"].startswith("tests/") for c in chunks)


def test_split_chunk_preserves_source_line_ranges():
    lines = [f"line_{i}" for i in range(150)]
    parts = split_long_chunk({"content":"\n".join(lines), "start_line":10, "end_line":159}, max_lines=60, overlap=10)
    assert [(p["start_line"],p["end_line"]) for p in parts] == [(10,69),(60,119),(110,159)]
    assert parts[1]["content"].splitlines() == lines[50:110]


@pytest.mark.parametrize("backend", ["faiss", "numpy"])
def test_vector_similarity_and_ids(backend):
    docs = [make_doc("first"), make_doc("second"), make_doc("third")]
    r = Retriever(docs, [[0, 8], [4, 0], [-3, 0]], FixedModel(), vector_backend=backend)
    results = r.search("semantic", k=10, strategy="vector")
    assert [x["doc_id"] for x in results] == [1, 0, 2]
    assert results[0]["score"] == pytest.approx(1.0)


def test_vector_requires_model_and_matching_dimensions():
    with pytest.raises(ValueError, match="需要"):
        Retriever([make_doc("one")]).search("query", strategy="vector")
    r = Retriever([make_doc("one")], [[1, 2, 3]], FixedModel())
    with pytest.raises(ValueError, match="维度"):
        r.search("query", strategy="vector")


def test_hybrid_keeps_exact_hit_and_expands_callers(retriever):
    results = retriever.search("lib.py::leaf", k=3)
    assert results[0]["metadata"]["name"] == "leaf"
    assert any(r["metadata"]["name"] == "middle" for r in results)
    assert len({r["doc_id"] for r in results}) == len(results)


def test_rrf_duplicate_ids_only_vote_once():
    assert dict(reciprocal_rank_fusion([[1, 1, 2]]))[1] == pytest.approx(1/61)
    assert reciprocal_rank_fusion([[1], [2, 1]])[0][0] == 1


@pytest.mark.parametrize("kwargs", [{"k": 0}, {"candidate_k": 0}, {"strategy": "unknown"}])
def test_invalid_retrieval_configuration(retriever, kwargs):
    with pytest.raises(ValueError):
        retriever.search("leaf", **kwargs)


def test_windows_auto_backend_does_not_import_faiss(monkeypatch):
    import builtins
    import os
    if os.name != "nt":
        pytest.skip("Windows default backend")
    original = builtins.__import__
    def without_faiss(name, *args, **kwargs):
        if name == "faiss":
            raise AssertionError("Windows auto mode must not load FAISS")
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins,"__import__",without_faiss)
    r = Retriever([make_doc("leaf")],[[1,0]],FixedModel())
    assert r.vector_backend == "numpy"
    assert r.search("leaf",strategy="vector")[0]["doc_id"] == 0
