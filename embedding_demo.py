import numpy as np
from sentence_transformers import SentenceTransformer

# 1. 加载本地 Embedding 模型
model = SentenceTransformer(
    "BAAI/bge-small-zh-v1.5",
    device="cpu"
)

# 2. 准备用户问题
question = "哪个函数负责解析 Python 源码并提取 AST？"

# 3. 准备两段候选代码
codes = [
    """def parse_file(path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    return visit(tree)
""",
    """def scan_python_files(root):
    return list(root.rglob("*.py"))
"""
]

# 4. 将文本转换为向量
instruction = "为这个句子生成表示以用于检索相关文章："

query_vector = model.encode(
    instruction + question,
    normalize_embeddings=True
)

code_vectors = model.encode(
    codes,
    normalize_embeddings=True
)

# 5. 计算相似度
scores = code_vectors @ query_vector

print("问题向量形状：", query_vector.shape)
print("代码向量形状：", code_vectors.shape)
print("问题向量前10维：", query_vector[:10])
print("问题向量长度：", np.linalg.norm(query_vector))

for code, score in zip(codes, scores):
    name = code.split("(")[0].replace("def ", "")
    print(f"{name}: 相似度 = {score:.4f}")

print("最相关的代码：", codes[np.argmax(scores)])
