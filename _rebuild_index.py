from pathlib import Path
from sentence_transformers import SentenceTransformer
from rag_embeddings import embed_repo
from rag_index import save_index, load_index

ROOT = Path(__file__).resolve().parent
INDEX_DIR = ROOT / ".repoquest_index"
# 1. 加载与原索引相同的 Embedding 模型
model = SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cpu")
# 2. 扫描当前本地源码并重新生成向量
print("===== Rebuilding RepoQuest Index =====")
documents, embeddings, errors = embed_repo(ROOT, model)
print("生成代码块数量：", len(documents))
print("Embedding Shape：", embeddings.shape)
print("解析失败文件：", errors)
if errors:
    raise RuntimeError("存在源码解析失败，暂不覆盖原索引")
# 4. 覆盖旧索引
save_index(documents, embeddings, INDEX_DIR)
# 5. 重新加载，验证文件保存成功
loaded_docs, loaded_vectors = load_index(INDEX_DIR)
assert len(loaded_docs) == len(documents)
assert loaded_vectors.shape == embeddings.shape
assert loaded_vectors.shape[1] == 512
print("\n索引保存成功！")
print("新索引代码块数量：", len(loaded_docs))
print("新索引向量形状：", loaded_vectors.shape)