from pathlib import Path
from context_expansion import find_caller_chunk, build_rag_context
from rag_index import load_index
from repo_tools import get_callers


# 1. 加载项目和 RAG 索引
root = Path.cwd().resolve()
index_dir = root / ".repoquest_index"

documents, _ = load_index(index_dir)


# 2. 从索引中查找一个函数
target = next(
    doc for doc in documents
    if doc.metadata["file"] == "call_analysis.py"
    and doc.metadata["name"] == "extract_local_calls"
)

print("===== RAG 检索结果 =====")
print("文件：", target.metadata["file"])
print("函数：", target.metadata["name"])


# 3. 从 Metadata 提取参数
target_file = target.metadata["file"]
target_function = target.metadata["name"]


# 4. 调用此前实现的 AST 分析工具

from context_expansion import expand_context

# 模拟一条 Hybrid Search 的结果
result = {
    "metadata": target.metadata,
    "content": target.page_content
}

expanded = expand_context(root, result)


print("\n===== 调用者源码扩展 =====")

for caller in expanded["callers"]:
    caller_doc = find_caller_chunk(documents, caller)

    if caller_doc is None:
        print(f"未找到 {caller['caller']} 对应的源码 Chunk")
        continue

    print(f"\n调用者：{caller['caller']}")
    print(f"调用位置：L{caller['line']}")
    print(
        f"源码范围："
        f"L{caller_doc.metadata['start_line']}"
        f"-L{caller_doc.metadata['end_line']}"
    )
    print("\n调用者源码：")
    print(caller_doc.page_content)

    print("\n===== 最终 RAG Context =====")

    context = build_rag_context(
        root=root,
        result=result,
        documents=documents,
        max_callers=2
    )

    print(context)
