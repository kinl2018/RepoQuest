import json
from pathlib import Path

import numpy as np
from langchain_core.documents import Document


def save_index(documents, embeddings, index_dir: Path):
    index_dir.mkdir(parents=True, exist_ok=True)
    # 保存向量矩阵
    np.save(index_dir / "embeddings.npy", embeddings)
    # 将 Document 转换成 JSON 可序列化的数据
    data = [
        {
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }
        for doc in documents
    ]
    (index_dir / "documents.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index(index_dir: Path):
    # 加载向量矩阵
    embeddings = np.load(index_dir / "embeddings.npy", allow_pickle=False)
    # 加载 Document 数据
    data = json.loads((index_dir / "documents.json").read_text(encoding="utf-8"))
    documents = [Document(**item) for item in data]
    if len(documents) != len(embeddings):
        raise ValueError("Document 与 Embedding 数量不一致")

    return documents, embeddings


if __name__ == "__main__":
    from sentence_transformers import SentenceTransformer
    from rag_embeddings import embed_repo

    root = Path(__file__).resolve().parent
    index_dir = root / ".repoquest_index"

    model = SentenceTransformer(
        "BAAI/bge-small-zh-v1.5",
        device="cpu"
    )

    documents, embeddings, errors = embed_repo(root, model)

    # 保存
    save_index(documents, embeddings, index_dir)

    print("索引已保存：", index_dir)

    # 重新从磁盘读取
    loaded_docs, loaded_vectors = load_index(index_dir)

    # 验证数据没有丢失
    assert len(documents) == len(loaded_docs)

    assert np.array_equal(
        embeddings, loaded_vectors
    )

    for original, loaded in zip(documents, loaded_docs):
        assert original.page_content == loaded.page_content
        assert original.metadata == loaded.metadata

    print("原始 Document 数量：", len(documents))
    print("加载 Document 数量：", len(loaded_docs))
    print("加载向量矩阵形状：", loaded_vectors.shape)
    print("全部验证通过！")
