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
