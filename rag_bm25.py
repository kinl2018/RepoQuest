import re

import jieba
import numpy as np
from rank_bm25 import BM25Okapi



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
    if not documents:
        raise ValueError("BM25 语料不能为空")
    corpus = []

    for doc in documents:
        meta = doc.metadata
        # 将 Metadata 和源码一起用于关键词检索
        text = (f"{meta['file']} {meta['name']} {meta['kind']} {doc.page_content}")
        corpus.append(tokenize(text) or ["<empty>"])

    return BM25Okapi(corpus)


def bm25_search(query, documents, bm25, k=3):
    if k <= 0:
        raise ValueError("k 必须大于 0")
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
