from pathlib import Path
from scanner import scan_python_files
from ast_parser import parse_files
from dependency import find_local_dependencies
from call_analysis import collect_call_edges

def find_symbol(root: Path, name: str) -> list[dict]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"不是有效的目录: {root}")
    if not name.strip():
        return []
    files = scan_python_files(root)
    results, errors = parse_files(files)
    hits = []
    for file, data in results.items():
        for symbol in data["symbols"]:
            if symbol["name"] != name:
                continue
            lines = file.read_text(encoding="utf-8").splitlines()
            start = symbol["start_lineno"]
            end = symbol["end_lineno"]
            hits.append({
                "file": file.relative_to(root).as_posix(),
                "name": symbol["name"],
                "kind": symbol["kind"],
                "lineno": symbol["lineno"],
                "source": "\n".join(lines[start - 1:end]),
            })
    return hits

def get_dependencies(root: Path, name: str) -> dict | None:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"不是有效的目录: {root}")
    if not name.strip():
        return None
    target = (root / name).resolve()
    if target not in scan_python_files(root):
        return None
    dependencies = find_local_dependencies(root, target)
    return {"file": target.relative_to(root).as_posix(),
            "dependencies": sorted({dep.relative_to(root).as_posix() for dep in dependencies})
            }

def get_callers(root: Path, target_file: str, target_function: str) -> list[dict]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"不是有效的目录: {root}")
    if not target_file.strip():
        return []
    files = scan_python_files(root)
    target_path = (root / target_file).resolve()
    if target_path not in files:
        return []
    callers = []
    for file in files:
        try:
            edges = collect_call_edges(root, file)  # [((file, caller), (target_file, callee), lineno)]
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        for source, target, lineno in edges:
            if (target_path, target_function) == target:
                source_file, caller = source
                callers.append({"file": source_file.relative_to(root).as_posix(), "caller": caller, "line": lineno})
    return callers

def read_source(root, file, start_line, end_line, max_lines=80):
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"不是有效的目录: {root}")
    file_path = (root / file).resolve()
    if file_path not in scan_python_files(root):
        raise ValueError("文件不存在或不允许读取")
    if start_line < 1 or end_line < start_line:
        raise ValueError("行号范围不合法")
    lines = file_path.read_text(encoding="utf-8").splitlines()
    if start_line > len(lines):
        raise ValueError("起始行号超过文件总行数")
    actual_end = min(end_line, len(lines))
    limited_end = min(actual_end, start_line + max_lines - 1)
    truncated = limited_end < actual_end
    source = "\n".join(f"{i}: {line}" for i, line in enumerate(lines[start_line - 1:limited_end], start=start_line))  # 为每一行添加真实行号
    return {"file": file, "start_line": start_line, "end_line": limited_end, "source": source,
            "truncated": truncated, "next_start_line": limited_end + 1 if truncated else None}

def search_code(root: Path, keyword: str) -> list[dict]:
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"不是有效的目录: {root}")
    if not keyword.strip():
        return []
    hits = []
    for file in scan_python_files(root):
        try:
            lines = file.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(lines, start=1):
            if keyword.casefold() in line.casefold():
                hits.append({
                    "file": file.relative_to(root).as_posix(),
                    "line": lineno,
                    "text": line.strip()[:300]
                })
                # 避免一次返回太多结果给 LLM
                if len(hits) >= 20:
                    return hits

    return hits


def list_files(root: Path) -> dict:
    root = root.expanduser().resolve()

    if not root.is_dir():
        raise ValueError(f"不是有效的目录: {root}")

    files = scan_python_files(root)

    # 转成相对路径字符串，方便交给 LLM
    paths = [file.relative_to(root).as_posix() for file in files]

    return {
        "total": len(paths),
        "files": paths[:100],
        "truncated": len(paths) > 100
    }
