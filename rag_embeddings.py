from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

from code_chunker import chunk_repo
from rag_documents import chunk_to_document


def embed_repo(root: Path, model: SentenceTransformer):
    # 1. 扫描仓库并生成 CodeChunk
    chunks, errors = chunk_repo(root)
    # 2. 转换为 LangChain Document
    documents = [chunk_to_document(chunk) for chunk in chunks]
    if not documents:
        raise ValueError("仓库中没有可向量化的代码块")
    # 3. 提取需要向量化的源码文本
    texts = [doc.page_content for doc in documents]
    # 4. 批量生成 Embedding
    embeddings = model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True)
    return documents, embeddings, errors


if __name__ == "__main__":
    root = Path(__file__).resolve().parent

    model = SentenceTransformer(
        "BAAI/bge-small-zh-v1.5",
        device="cpu"
    )

    documents, embeddings, errors = embed_repo(root, model)

    print("Document 数量：", len(documents))
    print("Embedding 矩阵形状：", embeddings.shape)
    print("解析失败文件数：", len(errors))

    print("\n第一个 Document：")
    print(documents[0].metadata)

    print("\n对应 Embedding 前10维：")
    print(embeddings[0][:10])

    assert len(documents) == embeddings.shape[0]
    assert embeddings.shape[1] == 512
    assert np.allclose(
        np.linalg.norm(embeddings, axis=1),
        1.0
    )

    print("\n全部验证通过！")
