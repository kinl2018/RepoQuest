from pathlib import Path
from sentence_transformers import SentenceTransformer
from rag_index import load_index
from rag_bm25 import build_bm25
from rag_hybrid import hybrid_search, fuse_query_results, deduplicate_results, diversify_results
from query_rewriter import rewrite_query
from context_expansion import build_multi_rag_context
from grounded_qa import answer_with_context

def retrieve_candidates(question, documents, embeddings, model, bm25, k=5, rewritten_question=None):
    if not question.strip():
        raise ValueError("问题不能为空")

    # 1. LLM 改写查询
    rewritten = rewritten_question if rewritten_question is not None else rewrite_query(question)
    # 2. 分别检索原始查询和改写查询
    candidate_k = 40
    search_k = candidate_k * 2
    original_results = hybrid_search(question, documents, embeddings, model, bm25, k=search_k, candidate_k=candidate_k)
    rewritten_results = hybrid_search(rewritten, documents, embeddings, model, bm25, k=search_k, candidate_k=candidate_k)
    # 3. Multi-Query RRF 融合
    fused = fuse_query_results(original_results, rewritten_results, documents, k=len(documents))
    # 4. 去重与多样化
    deduplicated = deduplicate_results(fused)
    results = diversify_results(deduplicated, k=k)

    return rewritten, results


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    index_dir = root / ".repoquest_index"

    # 加载已有索引
    documents, embeddings = load_index(index_dir)

    # 加载 Embedding 模型
    model = SentenceTransformer(
        "BAAI/bge-small-zh-v1.5",
        device="cpu"
    )

    bm25 = build_bm25(documents)

    question = input("请输入代码问题：").strip()

    rewritten, results = retrieve_candidates(
        question, documents, embeddings, model, bm25
    )

    print("\n===== Query Rewriting =====")
    print("原始问题：", question)
    print("改写问题：", rewritten)

    print("\n===== RAG Top-5 =====")
    for rank, result in enumerate(results, start=1):
        meta = result["metadata"]
        print(
            f"{rank}. {meta['file']}::{meta['name']}"
            f" | L{meta['start_line']}-{meta['end_line']}"
            f" | RRF={result['score']:.6f}"
        )

    # 构造多证据 RAG Context
    context = build_multi_rag_context(
        root=root,
        results=results,
        documents=documents,
        max_results=2,
        max_callers=2
    )

    print("\n===== Multi-Source RAG Context =====")
    print(context)

    # 生成最终回答
    answer = answer_with_context(question, context)

    print("\n===== End-to-End RAG QA =====")
    print("问题：", question)
    print("\n回答：")
    print(answer)

