from pathlib import Path
from ast_parser import parse_file
import ast
from scanner import scan_python_files

def chunk_file(path: Path, root: Path) -> list[dict]:
    path = path.resolve()
    root = root.resolve()
    symbols, _, _ = parse_file(path)
    lines = path.read_text(encoding="utf-8").splitlines()
    chunks = []
    for symbol in symbols:
        if symbol["kind"] not in ("function", "method", "class"):
            continue
        start = symbol["start_lineno"]
        end = symbol["end_lineno"]
        content = "\n".join(lines[start - 1:end])
        chunks.append({
            "file": path.relative_to(root).as_posix(),
            "name": symbol["name"],
            "kind": symbol["kind"],
            "start_line": start,
            "end_line": end,
            "content": content
        })

    return chunks

def chunk_module(path: Path, root: Path) -> list[dict]:
    path = path.resolve()
    root = root.resolve()
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    lines = source.splitlines()
    chunks = []
    pending = []

    def flush():
        if not pending:
            return
        start = pending[0].lineno
        end = pending[-1].end_lineno
        chunks.append({
            "file": path.relative_to(root).as_posix(),
            "name": "<module>",
            "kind": "module",
            "start_line": start,
            "end_line": end,
            "content": "\n".join(lines[start - 1:end])
        })
        pending.clear()

    for node in tree.body:
        if isinstance(node, (
            ast.FunctionDef,
            ast.AsyncFunctionDef,
            ast.ClassDef
        )):
            flush()
        else:
            pending.append(node)

    flush()
    return chunks

def split_long_chunk(
    chunk: dict,
    max_lines: int = 60,
    overlap: int = 10
) -> list[dict]:

    if max_lines <= 0 or not 0 <= overlap < max_lines:
        raise ValueError("分块参数不合法")

    lines = chunk["content"].splitlines()

    # 不超过限制，直接保留原始 Chunk
    if len(lines) <= max_lines:
        return [chunk]

    parts = []
    start = 0

    while start < len(lines):
        end = min(start + max_lines, len(lines))

        part = chunk.copy()

        part["start_line"] = chunk["start_line"] + start
        part["end_line"] = chunk["start_line"] + end - 1
        part["content"] = "\n".join(lines[start:end])
        part["part"] = len(parts) + 1

        parts.append(part)

        if end == len(lines):
            break

        start = end - overlap

    return parts

def chunk_repo(
    root: Path,
    max_lines: int = 60,
    overlap: int = 10,
    *, include_tests: bool = False, include_demos: bool = False
) -> tuple[list[dict], dict]:

    root = root.expanduser().resolve()

    if not root.is_dir():
        raise ValueError("仓库路径不存在或不是目录")

    chunks = []
    errors = {}

    from scanner import source_category
    files = scan_python_files(root, include_tests=include_tests, include_demos=include_demos)

    for file in files:
        try:
            # 第一步：按 AST 结构提取原始代码块
            raw_chunks = (
                chunk_file(file, root)
                + chunk_module(file, root)
            )

            # 第二步：对超长代码块再次切分
            for chunk in raw_chunks:
                parts = split_long_chunk(
                    chunk,
                    max_lines=max_lines,
                    overlap=overlap
                )

                for part in parts:
                    part["category"] = source_category(part["file"])
                chunks.extend(parts)

        except (OSError, UnicodeError, SyntaxError) as e:
            relative = file.relative_to(root).as_posix()
            errors[relative] = str(e)

    return chunks, errors
