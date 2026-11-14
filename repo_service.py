"""Python service boundary used directly by Streamlit and CLI."""
from pathlib import Path
from time import perf_counter
from uuid import uuid4
import json
import hashlib
import os
from langgraph.checkpoint.sqlite import SqliteSaver
from code_chunker import chunk_repo
from rag_documents import chunk_to_document
from rag_index import load_index, assert_current_index, read_manifest, source_fingerprint, save_index
from retrieval import Retriever
from repo_agent import build_graph
from llm_runtime import begin_usage, end_usage
from rag_embeddings import embed_repo


def open_repository(root, *, use_vectors=False, include_tests=False, include_demos=False, model=None):
    root = Path(root).expanduser().resolve()
    if use_vectors:
        from dotenv import load_dotenv
        load_dotenv(Path(__file__).with_name(".env"))
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
        index_dir = root / ".repoquest_index"
        if model is None:
            from embedding_runtime import get_embedding_model
            model = get_embedding_model(model_name)
        documents, embeddings = ensure_vector_index(
            root, index_dir, model_name, model,
            include_tests=include_tests,
            include_demos=include_demos
        )
    else:
        chunks, errors = chunk_repo(root, include_tests=include_tests, include_demos=include_demos)
        documents, embeddings = [chunk_to_document(c) for c in chunks], None
    retriever = Retriever(documents, embeddings, model, root=root,
                          vector_backend=os.getenv("VECTOR_BACKEND", "auto"),
                          include_tests=include_tests, include_demos=include_demos)
    retriever.parse_errors = errors if not use_vectors else {}
    return retriever


def run_investigation(retriever, question=None, *, thread_id=None, resume=False,
                      checkpoint_path=None, max_steps=6, context_budget=12000,
                      retry_policy="coverage_aware", **graph_options):
    root = retriever.root
    if not question and not resume:
        raise ValueError("问题不能为空")
    directory = root / ".repoquest_runs"
    directory.mkdir(exist_ok=True)
    path = Path(checkpoint_path or directory / "checkpoints.sqlite")
    path.parent.mkdir(parents=True, exist_ok=True)
    thread_id = thread_id or uuid4().hex
    config = {"configurable": {"thread_id": thread_id}, "recursion_limit": max_steps * 5 + 10}
    fingerprint = source_fingerprint(root, include_tests=True, include_demos=True)
    corpus = json.dumps([{"metadata": d.metadata, "content": d.page_content} for d in retriever.documents], sort_keys=True, ensure_ascii=False)
    identity = {"root": str(root), "source_fingerprint": fingerprint, "max_steps": max_steps,
                "corpus_fingerprint": hashlib.sha256(corpus.encode()).hexdigest(), "vector_backend": retriever.vector_backend,
                "context_budget": context_budget, "retry_policy": retry_policy}
    # Identity lives alongside checkpoint threads, preventing accidental resume against another repository/config.
    import sqlite3
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS repoquest_threads (id TEXT PRIMARY KEY, identity TEXT NOT NULL)")
        row = conn.execute("SELECT identity FROM repoquest_threads WHERE id=?", (thread_id,)).fetchone()
        if resume:
            if not row or json.loads(row[0]) != identity:
                raise ValueError("无法恢复：线程不存在，或源码/仓库/运行配置已变化")
        elif row:
            raise ValueError("线程已存在；请使用 resume 或新的 thread_id")
        else:
            if not question or not question.strip():
                raise ValueError("问题不能为空")
            conn.execute("INSERT INTO repoquest_threads VALUES (?, ?)", (thread_id, json.dumps(identity)))
    usage_token = begin_usage()
    start = perf_counter()
    try:
        with SqliteSaver.from_conn_string(str(path)) as saver:
            graph = build_graph(retriever, max_steps=max_steps, context_budget=context_budget,
                                retry_policy=retry_policy, checkpointer=saver, **graph_options)
            if resume and not graph.get_state(config).values:
                raise ValueError("该线程没有可恢复的 checkpoint")
            result = graph.invoke(None if resume else {"question": question}, config)
            pending = list(graph.get_state(config).next)
    except Exception as exc:
        raise RuntimeError(f"调查中断，线程 ID：{thread_id}。可修复后使用相同配置恢复。原因：{exc}") from exc
    finally:
        usage = end_usage(usage_token)
    result["run"] = {"thread_id": thread_id, "latency_ms": (perf_counter()-start)*1000,
                     "usage": usage, "retry_policy": retry_policy, "pending_nodes": pending,
                     "parse_errors": retriever.parse_errors if hasattr(retriever, "parse_errors") else {}}
    (directory / f"run-{uuid4().hex}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def ensure_vector_index(root, index_dir, model_name, model, include_tests=False, include_demos=False):
    # 1. 尝试复用已有索引
    try:
        assert_current_index(root, index_dir, model_name)
        manifest = read_manifest(index_dir)
        scope_ok = (
            (not include_tests or manifest.get("include_tests", False))
            and (not include_demos or manifest.get("include_demos", False))
        )
        if scope_ok:
            return load_index(index_dir)
    except (ValueError, FileNotFoundError, KeyError):
        # 索引缺失、损坏或过期，转入重建流程
        pass
    # 2. 自动重建索引
    print("检测到索引缺失或过期，正在自动重建...")
    before = source_fingerprint(root, include_tests=include_tests, include_demos=include_demos)
    documents, embeddings, errors = embed_repo(root, model, include_tests=include_tests, include_demos=include_demos)
    if errors:
        raise ValueError(f"源码解析失败，保留旧索引：{errors}")
    # 防止构建过程中源码再次发生变化
    after = source_fingerprint(root, include_tests=include_tests, include_demos=include_demos)
    if before != after:
        raise ValueError("构建过程中源码发生变化，请重试")
    # 3. 保存新索引
    save_index(
        documents, embeddings, index_dir,
        root=root,
        model_name=model_name,
        include_tests=include_tests,
        include_demos=include_demos
    )
    return load_index(index_dir)
