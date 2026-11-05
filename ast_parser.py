import ast
from pathlib import Path

def find_functions(node):
    functions = []
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        functions.append({"name": node.name, "lineno": node.lineno})
    for child in ast.iter_child_nodes(node):
        functions.extend(find_functions(child))
    return functions

def parse_file(path: Path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    return find_functions(tree)

def extract_symbols(node):
    return []


