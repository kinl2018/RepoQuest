from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

from rag_embeddings import embed_repo


def semantic_search(question: str, documents: list, embeddings: np.ndarray, model: SentenceTransformer, k: int = 3) -> list[dict]:
    # 1. 将用户问题转换为向量
    instruction = "为这个句子生成表示以用于检索相关文章："
    query_vector = model.encode(instruction + question, normalize_embeddings=True)
    # 2. 与所有代码向量计算相似度
    scores = embeddings @ query_vector
    # 3. 取得相似度最高的 K 个索引
    top_indices = np.argsort(scores)[::-1][:k]
    # 4. 根据索引取回 Document
    results = []
    for i in top_indices:
        doc = documents[i]
        results.append({
            "score": float(scores[i]),
            "content": doc.page_content,
            "metadata": doc.metadata
        })
    return results


if __name__ == "__main__":
    root = Path(__file__).resolve().parent

    model = SentenceTransformer(
        "BAAI/bge-small-zh-v1.5",
        device="cpu"
    )

    documents, embeddings, errors = embed_repo(root, model)

    question = input("请输入要检索的问题：").strip()

    results = semantic_search(
        question, documents, embeddings, model, k=len(documents))

    # 输入扩展后的查询
    expanded_question = input(
        "\n请输入扩展后的检索问题："
    ).strip()

    # 使用相同的源码 Embedding 重新检索
    expanded_results = semantic_search(
        expanded_question,
        documents,
        embeddings,
        model,
        k=len(documents)
    )

    # 找到目标函数在扩展查询中的排名
    target_rank = next(
        (
            rank
            for rank, result in enumerate(
                expanded_results, start=1
            )
            if result["metadata"]["file"] == "repo_tools.py"
            and result["metadata"]["name"] == "get_callers"
        ),
        None
    )

    print("\n===== Query Expansion 实验 =====")
    print(f"原始问题排名：16")
    print(f"扩展查询排名：{target_rank}")

    print("\n扩展查询 Top-3：")

    for rank, result in enumerate(
        expanded_results[:3], start=1
    ):
        meta = result["metadata"]

        print(
            f"{rank}. {meta['file']}::{meta['name']}"
            f" | 相似度：{result['score']:.4f}"
        )

