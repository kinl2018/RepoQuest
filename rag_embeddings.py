from pathlib import Path
import numpy as np
from code_chunker import chunk_repo
from rag_documents import chunk_to_document

def embed_repo(root: Path, model, **chunk_options):
    # 1. 扫描仓库并生成 CodeChunk
    chunks, errors = chunk_repo(root, **chunk_options)
    # 2. 转换为 LangChain Document
    documents = [chunk_to_document(chunk) for chunk in chunks]
    if not documents:
        raise ValueError("仓库中没有可向量化的代码块")
    # 3. 提取需要向量化的源码文本
    texts = [doc.page_content for doc in documents]
    # 4. 批量生成 Embedding
    embeddings = model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=True)
    return documents, embeddings, errors
