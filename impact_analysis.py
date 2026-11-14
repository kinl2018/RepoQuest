"""Reverse-call reachability with call-site evidence and explicit analysis limits."""
from collections import deque
from pathlib import Path
from call_analysis import build_call_index
from dependency import build_dependency_edges
from scanner import scan_python_files


def analyze_impact(root: Path, file: str, symbol: str, max_depth: int = 3, *, graph=None) -> dict:
    root = Path(root).resolve()
    if not 1 <= max_depth <= 20:
        raise ValueError("max_depth 必须在 1 到 20 之间")
    target_file = (root / file).resolve()
    if not target_file.is_relative_to(root) or target_file not in scan_python_files(root):
        raise ValueError("目标必须是仓库内的 Python 文件")
    graph = build_call_index(root) if graph is None else graph
    target = (target_file, symbol)
    if target not in graph:
        raise ValueError(f"未找到函数或方法：{file}::{symbol}")
    queue, seen, affected, sites = deque([(target, 0, [target])]), {target}, [], []
    truncated = False
    while queue:
        node, depth, path = queue.popleft()
        predecessors = sorted(graph.predecessors(node))
        if depth >= max_depth:
            truncated |= any(p not in seen for p in predecessors)
            continue
        for caller in predecessors:
            for line in graph[caller][node]["lines"]:
                sites.append({"file": caller[0].relative_to(root).as_posix(), "caller": caller[1],
                              "callee_file": node[0].relative_to(root).as_posix(), "callee": node[1], "line": line})
            if caller in seen:
                continue
            seen.add(caller)
            next_path = [caller] + path
            affected.append({"file": caller[0].relative_to(root).as_posix(), "symbol": caller[1],
                             "depth": depth + 1, "direct": depth == 0,
                             "path": [f"{p.relative_to(root).as_posix()}::{n}" for p, n in next_path]})
            queue.append((caller, depth + 1, next_path))
    dependencies = build_dependency_edges(root, scan_python_files(root))
    importers = sorted({a.relative_to(root).as_posix() for a, b in dependencies if b == target_file})
    return {"target": f"{file}::{symbol}", "affected": affected, "call_sites": sites,
            "module_importers": importers, "max_depth": max_depth, "truncated": truncated,
            "parse_errors": graph.graph.get("errors", {}),
            "limitations": ["静态候选影响范围，不代表运行时一定受影响。",
                            "不解析动态分派、反射、猴子补丁、重导出和任意实例类型；模块导入仅表示潜在依赖。"]}
