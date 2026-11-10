import re
from pathlib import Path

import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from code_chunker import chunk_repo
from rag_documents import chunk_to_document


# 提取英文标识符、中文文本和数字
TOKEN_PATTERN = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*|[\u4e00-\u9fff]+|\d+"
)


def tokenize(text: str) -> list[str]:
    tokens = []

    for item in TOKEN_PATTERN.findall(text):
        if re.fullmatch(r"[\u4e00-\u9fff]+", item):
            # 中文分词
            tokens.extend(jieba.lcut(item))
        else:
            item = item.lower()
            # 保留完整标识符
            tokens.append(item)
            # 同时拆分下划线命名
            if "_" in item:
                tokens.extend(part for part in item.split("_") if part)

    return tokens


def build_bm25(documents):
    corpus = []

    for doc in documents:
        meta = doc.metadata
        # 将 Metadata 和源码一起用于关键词检索
        text = (f"{meta['file']} {meta['name']} {meta['kind']} {doc.page_content}")
        corpus.append(tokenize(text))

    return BM25Okapi(corpus)


def bm25_search(query, documents, bm25, k=3):
    query_tokens = tokenize(query)
    # 计算查询与所有文档的 BM25 分数
    scores = bm25.get_scores(query_tokens)
    # 按分数降序排列
    indices = np.argsort(scores)[::-1]
    results = []
    for i in indices:
        if scores[i] <= 0:
            break

        results.append({
            "score": float(scores[i]),
            "content": documents[i].page_content,
            "metadata": documents[i].metadata
        })

        if len(results) >= k:
            break

    return results


if __name__ == "__main__":
    root = Path(__file__).resolve().parent

    # 1. 构建 Document
    chunks, errors = chunk_repo(root)

    documents = [
        chunk_to_document(chunk)
        for chunk in chunks
    ]

    # 2. 构建 BM25 索引
    bm25 = build_bm25(documents)

    print("Document 数量：", len(documents))
    print("解析失败文件数：", len(errors))

    # 3. 输入查询
    query = input("\n请输入检索问题：")

    print("分词结果：", tokenize(query))

    # 4. 检索
    results = bm25_search(
        query, documents, bm25, k=3
    )

    # 5. 输出结果
    print("\n===== BM25 Top-3 =====")

    if not results:
        print("没有匹配到关键词")

    for rank, result in enumerate(results, start=1):
        meta = result["metadata"]

        print(f"\nTop {rank}")
        print(f"BM25 分数：{result['score']:.4f}")
        print(f"文件：{meta['file']}")
        print(f"函数：{meta['name']}")
        print(
            f"行号：L{meta['start_line']}-"
            f"{meta['end_line']}"
        )
