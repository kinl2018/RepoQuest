from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer
from rag_embeddings import embed_repo
from rag_bm25 import build_bm25, tokenize
import sys
from rag_index import save_index, load_index
from query_rewriter import rewrite_query

def reciprocal_rank_fusion(rankings: list[list[int]], rrf_constant: int = 60) -> list[tuple[int, float]]:
    if rrf_constant <= 0:
        raise ValueError("rrf_constant 必须大于 0")
    scores = {}
    for ranking in rankings:
        seen = set()
        for rank, doc_id in enumerate(ranking, start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            scores[doc_id] = (scores.get(doc_id, 0) + 1 / (rrf_constant + rank))

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)

def hybrid_search(question, documents, embeddings, model, bm25, k=5, candidate_k=10):
    # 1. Embedding 检索
    instruction = "为这个句子生成表示以用于检索相关文章："
    query_vector = model.encode(instruction + question, normalize_embeddings=True)
    semantic_scores = embeddings @ query_vector
    semantic_ranking = np.argsort(semantic_scores)[::-1][:candidate_k].tolist()
    # 2. BM25 检索
    bm25_scores = bm25.get_scores(tokenize(question))
    bm25_ranking = [int(i) for i in np.argsort(bm25_scores)[::-1] if bm25_scores[i] > 0][:candidate_k]
    # 3. 使用 RRF 融合两路排名
    fused = reciprocal_rank_fusion([semantic_ranking, bm25_ranking])
    # 4. 根据文档索引取回源码
    results = []
    for doc_id, score in fused[:k]:
        doc = documents[doc_id]
        results.append({
            "doc_id": int(doc_id),
            "score": score,
            "metadata": doc.metadata,
            "content": doc.page_content
        })

    return results

def fuse_query_results(original_results: list[dict], rewritten_results: list[dict], documents: list, k: int = 5) -> list[dict]:
    # 1. 提取两组检索结果中的 Document ID
    original_ranking = [result["doc_id"] for result in original_results]
    rewritten_ranking = [result["doc_id"] for result in rewritten_results]
    # 2. 第二轮 RRF 排名融合
    fused = reciprocal_rank_fusion([original_ranking, rewritten_ranking])
    # 3. 按照 doc_id 取回 Document
    results = []
    for doc_id, score in fused[:k]:
        doc = documents[doc_id]
        results.append({
            "doc_id": int(doc_id),
            "score": score,
            "metadata": doc.metadata,
            "content": doc.page_content
        })

    return results

def calculate_overlap(meta_a: dict, meta_b: dict) -> float:
    # 1. 计算两个区间的交集
    start = max(meta_a["start_line"], meta_b["start_line"])
    end = min(meta_a["end_line"], meta_b["end_line"])
    # 2. 计算重叠行数
    overlap = max(0, end - start + 1)
    # 3. 计算两个 Chunk 的长度
    len_a = (meta_a["end_line"] - meta_a["start_line"] + 1)
    len_b = (meta_b["end_line"] - meta_b["start_line"] + 1)
    # 4. 计算相对于较短 Chunk 的重叠比例
    return overlap / min(len_a, len_b)

def deduplicate_results(results: list[dict], overlap_threshold: float = 0.8) -> list[dict]:
    # 确保按照 RRF 分数从高到低处理
    ordered = sorted(results, key=lambda x: x["score"], reverse=True)
    kept = []
    for candidate in ordered:
        duplicate = False
        for selected in kept:
            meta_a = candidate["metadata"]
            meta_b = selected["metadata"]
            # 只比较同一文件、同一符号的 Chunk
            same_symbol = (meta_a["file"] == meta_b["file"] and meta_a["name"] == meta_b["name"] and meta_a["kind"] == meta_b["kind"])
            if not same_symbol:
                continue
            # 计算源码区间重叠比例
            ratio = calculate_overlap(meta_a, meta_b)
            if ratio >= overlap_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(candidate)

    return kept


def diversify_results(results: list[dict], k: int = 5) -> list[dict]:
    selected = []
    seen = set()
    for result in results:
        meta = result["metadata"]
        # 用文件、符号名称、符号类型确定一个逻辑区域
        key = (meta["file"], meta["name"], meta["kind"])
        # 该逻辑区域已经展示过，就跳过
        if key in seen:
            continue
        seen.add(key)
        selected.append(result)
        # 收集够 k 个结果就结束
        if len(selected) >= k:
            break

    return selected


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    index_dir = root / ".repoquest_index"

    model = SentenceTransformer(
        "BAAI/bge-small-zh-v1.5",
        device="cpu"
    )

    # 判断是否强制重建
    rebuild = "--rebuild" in sys.argv

    # 检查索引文件是否完整存在
    index_exists = (
        (index_dir / "documents.json").is_file()
        and (index_dir / "embeddings.npy").is_file()
    )

    if index_exists and not rebuild:
        # 使用已有的索引
        documents, embeddings = load_index(index_dir)
        print("已从磁盘加载索引")

    else:
        # 重新生成 Embedding
        documents, embeddings, errors = embed_repo(root, model)

        if errors:
            raise RuntimeError(f"索引构建出现解析错误：{errors}")

        save_index(documents, embeddings, index_dir)
        print("已重新生成并保存索引")

    # BM25 仍然基于相同的 Document 构建
    bm25 = build_bm25(documents)

    print(f"已加载 {len(documents)} 个代码块")
    print(f"Embedding 矩阵形状：{embeddings.shape}")


    # 1. 获取原始问题
    question = input("\n请输入检索问题：").strip()

    # 2. 使用 LLM 改写问题
    rewritten_question = rewrite_query(question)

    print("\n===== Query Rewriter =====")
    print("原始问题：", question)
    print("改写结果：", rewritten_question)

    # 3. 两组实验使用相同的检索参数
    candidate_k = 40
    result_k = candidate_k * 2

    # 实验 A：原始查询
    original_results = hybrid_search(
        question,
        documents,
        embeddings,
        model,
        bm25,
        k=result_k,
        candidate_k=candidate_k
    )

    # 实验 B：LLM 改写查询
    rewritten_results = hybrid_search(
        rewritten_question,
        documents,
        embeddings,
        model,
        bm25,
        k=result_k,
        candidate_k=candidate_k
    )

    # 4. 查找目标函数排名
    def find_target_rank(results):
        for rank, result in enumerate(results, start=1):
            meta = result["metadata"]

            if (
                meta["file"] == "repo_tools.py"
                and meta["name"] == "get_callers"
            ):
                return rank

        return None

    rank_a = find_target_rank(original_results)
    rank_b = find_target_rank(rewritten_results)

    # 5. 输出实验结果
    print("\n===== A/B 实验结果 =====")
    print(f"原始查询排名：{rank_a}")
    print(f"LLM 改写查询排名：{rank_b}")

    # 6. 展示两组 Top-5
    for title, results in [
        ("原始查询", original_results),
        ("LLM 改写查询", rewritten_results)
    ]:
        print(f"\n===== {title} Top-5 =====")

        for rank, result in enumerate(results[:5], start=1):
            meta = result["metadata"]

            print(
                f"{rank}. {meta['file']}::{meta['name']}, RRF: {result['score']:.6f}, doc_id: {result['doc_id']}"
            )


    # 第三组实验：Multi-Query Fusion
    fused_results = fuse_query_results(
        original_results,
        rewritten_results,
        documents,
        k=len(documents)
    )

    # 先按高度重叠情况去重
    deduplicated_results = deduplicate_results(fused_results)

    # 再进行结果多样化
    diversified_results = diversify_results(
        deduplicated_results,
        k=5
    )

    print("\n===== Diversified Top-5 =====")

    for rank, result in enumerate(diversified_results, start=1):
        meta = result["metadata"]

        print(
            f"{rank}. {meta['file']}::{meta['name']}"
            f" | doc_id: {result['doc_id']}"
        )
