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
