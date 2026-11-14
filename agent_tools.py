"""One validated tool registry for the planner, execution and compatibility API."""
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field
from repo_tools import read_source, search_code, get_dependencies
from evidence_store import evidence_id


class Arguments(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class SearchArgs(Arguments):
    query: str = Field(min_length=1, max_length=4000)


class SymbolArgs(Arguments):
    name: str = Field(min_length=1, max_length=300)


class CallerArgs(Arguments):
    file: str = Field(min_length=1)
    symbol: str = Field(min_length=1)


class ReadArgs(Arguments):
    file: str = Field(min_length=1)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)


class FileArgs(Arguments):
    file: str = Field(min_length=1)


class EmptyArgs(Arguments):
    pass


class PlanAction(Arguments):
    tool: Literal["search_index", "find_symbol", "search_source", "read_source", "get_callers", "get_dependencies", "index_metadata"]
    args: dict
    reason: str = Field(min_length=1)


REGISTRY = {
    "search_index": (SearchArgs, "按自然语言混合检索相关源码；不知道实现位置时使用。"),
    "find_symbol": (SymbolArgs, "精确查找函数、类、方法；支持 file.py::Class.method。"),
    "search_source": (SearchArgs, "在真实源码中按字符串搜索，包含尚未索引的符号。"),
    "read_source": (ReadArgs, "按真实行号读取仓库内源码，每次最多 80 行。"),
    "get_callers": (CallerArgs, "使用 AST 调用图查询某函数的直接调用者及源码。"),
    "get_dependencies": (FileArgs, "查询 Python 文件的本地模块依赖。"),
    "index_metadata": (EmptyArgs, "读取真实向量索引数量、维度与模型信息。"),
}


def tool_schemas():
    return [{"type": "function", "function": {"name": name, "description": description,
             "parameters": schema.model_json_schema()}} for name, (schema, description) in REGISTRY.items()]


def execute_tool(action: PlanAction, retriever, index_dir):
    from rag_index import get_index_metadata
    args = REGISTRY[action.tool][0].model_validate(action.args).model_dump()
    root = retriever.root
    results, observation = [], None
    if action.tool in {"search_index", "find_symbol"}:
        results = retriever.search(args.get("query", args.get("name")), k=8,
                                   strategy="exact" if action.tool == "find_symbol" else "hybrid")
    elif action.tool == "search_source":
        hits = search_code(root, args["query"])
        for hit in hits[:8]:
            source = read_source(root, hit["file"], max(1, hit["line"] - 5), hit["line"] + 15)
            results.append(source_result(root, source))
    elif action.tool == "read_source":
        results = [source_result(root, read_source(root, **args))]
    elif action.tool == "get_callers":
        from call_analysis import build_call_index
        if retriever.call_graph is None:
            retriever.call_graph = build_call_index(root)
        graph = retriever.call_graph
        node = ((root / args["file"]).resolve(), args["symbol"])
        call_sites = []
        if node in graph:
            for caller in sorted(graph.predecessors(node))[:10]:
                for line in graph[caller][node]["lines"][:2]:
                    call_sites.append({"file": caller[0].relative_to(root).as_posix(), "caller": caller[1],
                                       "line": line, "callee": args["symbol"]})
                    source = read_source(root, caller[0].relative_to(root).as_posix(), max(1, line-8), line+15)
                    result = source_result(root, source)
                    result["metadata"]["name"] = caller[1]
                    result["doc_id"] = evidence_id(result)
                    results.append(result)
        observation = {"target": args, "resolved_call_sites": len(results),
                       "target_exists": node in graph, "call_sites": call_sites,
                       "total_callers": graph.in_degree(node) if node in graph else 0,
                       "note": "AST 静态调用结果；未命中不证明运行时没有调用者。"}
    elif action.tool == "get_dependencies":
        observation = get_dependencies(root, args["file"])
        if observation is None:
            raise ValueError("未找到指定文件")
    else:
        observation = get_index_metadata(index_dir)
    return results, observation


def source_result(root, source):
    # Read directly again without display line prefixes, preserving exact source bytes.
    lines = (root / source["file"]).read_text(encoding="utf-8").splitlines()
    meta = {"file": source["file"], "name": "<source>", "kind": "module",
            "start_line": source["start_line"], "end_line": source["end_line"]}
    result = {"metadata": meta, "content": "\n".join(lines[meta["start_line"]-1:meta["end_line"]]),
              "score": 1.0, "retrieval_type": "source_tool"}
    result["doc_id"] = evidence_id(result)
    return result
