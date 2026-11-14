"""RepoQuest command line: python -m repoquest --help."""
import argparse
import json
import os
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description="RepoQuest · RAG 与 LangGraph 代码理解 Agent")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("index", "search", "ask", "impact", "evaluate"):
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, default=Path.cwd())
        p.add_argument("--include-tests", action="store_true")
        p.add_argument("--include-demos", action="store_true")
        if name in {"search", "ask", "evaluate"}:
            p.add_argument("--vectors", action="store_true", help="使用与当前源码、模型一致的向量索引")
        if name in {"search", "ask"}:
            p.add_argument("question", nargs="?")
        if name == "search":
            p.add_argument("--strategy", choices=["exact", "bm25", "vector", "hybrid"], default="hybrid")
            p.add_argument("-k", type=int, default=5)
        if name == "ask":
            p.add_argument("--thread-id")
            p.add_argument("--resume", action="store_true")
            p.add_argument("--max-steps", type=int, default=6)
            p.add_argument("--context-budget", type=int, default=12000)
            p.add_argument("--policy", choices=["ordinary", "coverage_aware"], default="coverage_aware")
        if name == "impact":
            p.add_argument("file")
            p.add_argument("symbol")
            p.add_argument("--depth", type=int, default=3)
        if name == "evaluate":
            p.add_argument("--cases", type=Path, default=Path(__file__).parent / "eval_cases.jsonl")
            p.add_argument("--output", type=Path, default=Path("reports"))
            p.add_argument("--agents", action="store_true")
            p.add_argument("--judge", action="store_true")
            p.add_argument("--limit", type=int, help="仅运行前 N 题；默认运行全部问题")
            p.add_argument("-k", type=int, default=5)
    args = parser.parse_args(argv)
    root = args.root.expanduser().resolve()
    try:
        if args.command == "index":
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).with_name(".env"))
            from embedding_runtime import get_embedding_model
            from rag_embeddings import embed_repo
            from rag_index import save_index, source_fingerprint
            name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh-v1.5")
            model = get_embedding_model(name)
            before = source_fingerprint(root, include_tests=args.include_tests, include_demos=args.include_demos)
            documents, vectors, errors = embed_repo(root, model, include_tests=args.include_tests, include_demos=args.include_demos)
            if errors:
                raise ValueError(f"存在解析错误，保留旧索引：{errors}")
            if source_fingerprint(root, include_tests=args.include_tests, include_demos=args.include_demos) != before:
                raise ValueError("构建期间源码发生变化，保留旧索引，请重新运行")
            save_index(documents, vectors, root / ".repoquest_index", root=root, model_name=name,
                       include_tests=args.include_tests, include_demos=args.include_demos)
            print(f"索引已更新：{len(documents)} 个代码块，维度 {vectors.shape[1]}")
            return
        if args.command == "impact":
            from impact_analysis import analyze_impact
            result = analyze_impact(root, args.file, args.symbol, args.depth)
        else:
            from repo_service import open_repository, run_investigation
            retriever = open_repository(root, use_vectors=args.vectors, include_tests=args.include_tests, include_demos=args.include_demos)
            if args.command == "search":
                result = retriever.search(args.question or "", args.k, strategy=args.strategy)
            elif args.command == "ask":
                if args.resume and not args.thread_id:
                    raise ValueError("--resume 需要 --thread-id")
                result = run_investigation(retriever, args.question, thread_id=args.thread_id, resume=args.resume,
                                          max_steps=args.max_steps, context_budget=args.context_budget, retry_policy=args.policy)
            else:
                from evaluation import load_cases, evaluate_retrieval, compare_agents, write_report
                cases = load_cases(args.cases, retriever.documents)
                if args.limit is not None:
                    if args.limit < 1:
                        raise ValueError("--limit 必须大于 0")
                    cases = cases[:args.limit]
                report = compare_agents(retriever, cases, judge=args.judge) if args.agents else evaluate_retrieval(
                    retriever, cases, strategies=("bm25", "vector", "hybrid") if args.vectors else ("bm25", "hybrid"), k=args.k)
                print(write_report(report, args.output).resolve())
                return
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except (ValueError, OSError, ImportError, RuntimeError) as exc:
        parser.exit(1, f"RepoQuest: {exc}\n")


if __name__ == "__main__":
    main()
