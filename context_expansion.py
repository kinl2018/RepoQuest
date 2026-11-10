from pathlib import Path
from repo_tools import get_callers

def expand_context(root: Path, result: dict) -> dict:
    meta = result["metadata"]
    file = meta["file"]
    name = meta["name"]
    kind = meta["kind"]
    # 默认没有调用者
    callers = []
    # 只有函数、方法才查询调用者
    if kind in ("function", "method"):
        callers = get_callers(root, file, name)
    # 整合源码与调用关系
    return {
        "file": file,
        "name": name,
        "start_line": meta["start_line"],
        "end_line": meta["end_line"],
        "code": result["content"],
        "callers": callers
    }

def find_caller_chunk(documents: list, caller: dict):
    # 找到与调用者匹配的代码块
    for doc in documents:
        meta = doc.metadata
        if (
            meta["file"] == caller["file"]
            and meta["name"] == caller["caller"]
            and meta["start_line"] <= caller["line"] <= meta["end_line"]
        ):
            return doc
    return None

def build_rag_context(root: Path, result: dict, documents: list, max_callers: int = 2) -> str:
    # 1. 获取检索命中函数的信息
    expanded = expand_context(root, result)
    sections = [
        f"""[证据 1：检索命中的代码]
文件：{expanded["file"]}
符号：{expanded["name"]}
源码范围：L{expanded["start_line"]}-L{expanded["end_line"]}

源码：
```python
{expanded["code"]}
```"""]

    # 2. 获取调用者信息
    for index, caller in enumerate(expanded["callers"][:max_callers], start=2):
        caller_doc = find_caller_chunk(documents, caller)
        if caller_doc is None:
            continue
        meta = caller_doc.metadata
        sections.append(
            f"""[证据 {index}：调用者]
调用关系：{caller["caller"]} → {expanded["name"]}
调用位置：{caller["file"]} L{caller["line"]}
调用者源码范围：L{meta["start_line"]}-L{meta["end_line"]}

源码：
```python
{caller_doc.page_content}
```""")
    # 3. 将所有证据拼接为一个字符串
    return "\n\n".join(sections)
