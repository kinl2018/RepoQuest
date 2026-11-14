"""Reusable exact/BM25/vector/hybrid search with optional graph expansion."""
import numpy as np
import os
from importlib.util import find_spec
from pathlib import Path
from rag_bm25 import build_bm25, tokenize
from scanner import source_category
from symbol_index import build_symbol_index, exact_symbol_search, merge_retrieval_results


def reciprocal_rank_fusion(rankings, rrf_constant=60):
    if rrf_constant <= 0:
        raise ValueError("rrf_constant 必须大于 0")
    scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(dict.fromkeys(ranking), 1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1 / (rrf_constant + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))


class Retriever:
    def __init__(self, documents, embeddings=None, model=None, *, root=None,
                 include_tests=False, include_demos=False, vector_backend="auto", call_graph=None):
        allowed = [i for i, doc in enumerate(documents)
                   if (include_tests or source_category(doc.metadata["file"]) != "test")
                   and (include_demos or source_category(doc.metadata["file"]) != "demo")]
        self.documents = [documents[i] for i in allowed]
        if not self.documents:
            raise ValueError("索引中没有可检索源码")
        self.model, self.root, self.call_graph = model, Path(root).resolve() if root is not None else None, call_graph
        self.symbol_index = build_symbol_index(self.documents)
        self.bm25 = build_bm25(self.documents)
        self.embeddings, self.vector_index = None, None
        self.vector_backend = "unavailable"
        if embeddings is not None:
            if vector_backend == "auto":
                vector_backend = "numpy" if os.name == "nt" or find_spec("faiss") is None else "faiss"
            matrix = np.asarray(embeddings, dtype=np.float32)
            if matrix.ndim != 2 or len(matrix) != len(documents) or not np.isfinite(matrix).all():
                raise ValueError("向量数量、形状或数值无效")
            matrix = matrix[allowed].copy()
            matrix /= np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12)
            self.embeddings = matrix
            if vector_backend == "faiss":
                try:
                    import faiss
                except ImportError as exc:
                    raise ValueError("当前 Python 环境缺少 faiss；可将 VECTOR_BACKEND 设为 auto/numpy，或安装 faiss-cpu") from exc
                self.vector_index = faiss.IndexFlatIP(matrix.shape[1])
                self.vector_index.add(matrix)
            elif vector_backend != "numpy":
                raise ValueError("未知向量后端")
            self.vector_backend = vector_backend

    def _result(self, doc_id, score, origin):
        doc = self.documents[doc_id]
        return {"doc_id": int(doc_id), "score": float(score), "metadata": dict(doc.metadata),
                "content": doc.page_content, "retrieval_type": origin}

    def _vector(self, query, k):
        if self.model is None or self.embeddings is None:
            raise ValueError("向量检索需要索引和同一 Embedding 模型；可先使用 BM25")
        vector = np.asarray(self.model.encode("为这个句子生成表示以用于检索相关文章：" + query,
                                             normalize_embeddings=True), dtype=np.float32).reshape(1, -1)
        if vector.shape[1] != self.embeddings.shape[1] or not np.isfinite(vector).all():
            raise ValueError("查询向量与索引维度不一致或包含无效数值")
        vector /= max(float(np.linalg.norm(vector)), 1e-12)
        if self.vector_index is not None:
            scores, ids = self.vector_index.search(vector, min(k, len(self.documents)))
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        scores = self.embeddings @ vector[0]
        return [(int(i), float(scores[i])) for i in np.argsort(-scores, kind="stable")[:k]]

    def search(self, query, k=5, *, strategy="hybrid", candidate_k=40, rewritten=None, expand_graph=True):
        if not isinstance(query, str) or not query.strip() or k <= 0 or candidate_k <= 0:
            raise ValueError("问题不能为空，k 和 candidate_k 必须为正数")
        if strategy not in {"exact", "bm25", "vector", "hybrid"}:
            raise ValueError("未知检索策略")
        exact = exact_symbol_search(query, self.documents, self.symbol_index)
        if strategy == "exact":
            return exact[:k]
        rankings, values = [], {}
        for text in dict.fromkeys([query, rewritten or query]):
            if strategy in {"bm25", "hybrid"}:
                scores = self.bm25.get_scores(tokenize(text))
                ids = [int(i) for i in np.argsort(-scores, kind="stable") if scores[i] > 0][:candidate_k]
                rankings.append(ids)
                values.update({i: float(scores[i]) for i in ids})
            if strategy in {"vector", "hybrid"} and (strategy == "vector" or self.model is not None):
                pairs = self._vector(text, candidate_k)
                rankings.append([i for i, _ in pairs])
                values.update(dict(pairs))
        pairs = reciprocal_rank_fusion(rankings) if strategy == "hybrid" else [(i, values[i]) for i in rankings[0]]
        ranked = [self._result(i, score, strategy) for i, score in pairs]
        if strategy != "hybrid":
            return ranked[:k]
        # Exact candidates survive fusion; group only similarity candidates by symbol.
        diversified, seen = [], set()
        for result in ranked:
            m = result["metadata"]
            group = (m["file"], m["name"], m["kind"])
            if group not in seen:
                diversified.append(result)
                seen.add(group)
        merged = merge_retrieval_results(exact, diversified, k=max(k, candidate_k))
        if expand_graph and self.root is not None:
            related = self.related(merged[:2])
            # Reserve graph space without evicting exact hits.
            slots = min(2, len(related), max(0, k - len(exact)))
            return merge_retrieval_results(merged[:k-slots], related + merged, k=k)
        return merged[:k]

    def related(self, results):
        from call_analysis import build_call_index
        if self.call_graph is None:
            self.call_graph = build_call_index(self.root)
        related, seen = [], {r["doc_id"] for r in results}
        for result in results:
            m = result["metadata"]
            node = (self.root.resolve() / m["file"], m["name"])
            if node not in self.call_graph:
                continue
            neighbors = set(self.call_graph.predecessors(node)) | set(self.call_graph.successors(node))
            for file, name in sorted(neighbors):
                for i, doc in enumerate(self.documents):
                    if i not in seen and (doc.metadata["file"], doc.metadata["name"]) == (file.relative_to(self.root).as_posix(), name):
                        related.append(self._result(i, 0.0, "call_graph"))
                        seen.add(i)
        return related
