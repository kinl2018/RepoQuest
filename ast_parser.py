import ast
from pathlib import Path

def parse_file(path: Path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    return visit(tree), extract_imports(tree), extract_import_details(tree)

def parse_files(files: list[Path]):
    results = {}
    errors = {}
    for file in files:
        try:
            symbols, imports, import_details = parse_file(file)
        except (OSError, UnicodeDecodeError, SyntaxError) as e:
            errors[file] = e
            continue
        results[file] = {"symbols": symbols, "imports": imports, "import_details": import_details}
    return results, errors

def visit(node, scope="", parent_kind="module"):
    symbols = []
    if isinstance(node, ast.ClassDef):
        kind = "class"
    elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        kind = "method" if parent_kind == "class" else "function"
    else:
        for child in ast.iter_child_nodes(node):
            symbols.extend(visit(child, scope, parent_kind))
        return symbols
    new_scope = f"{scope}.{node.name}" if scope else node.name
    symbols.append({"name": new_scope, "kind": kind, "lineno": node.lineno})
    for child in node.body:
        symbols.extend(visit(child, new_scope, kind))
    return symbols

def extract_imports(tree):
    imports = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module is not None:
                imports.append(node.module)
    return imports

def extract_import_details(tree):
    details = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            details.append({"module": node.module, "level": node.level, "names": [alias.name for alias in node.names]})
        elif isinstance(node, ast.Import):
            for alias in node.names:
                details.append({"module": alias.name, "level": 0, "names": []})
    return details

def find_functions(node):
    functions = []
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        functions.append({"name": node.name, "lineno": node.lineno})
    for child in ast.iter_child_nodes(node):
        functions.extend(find_functions(child))
    return functions

def extract_symbols(tree):
    symbols = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            symbols.append({"kind": "class", "name": node.name})
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    symbols.append({"kind": "method", "name": f"{node.name}.{child.name}"})
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols.append({"kind": "function", "name": node.name})
    return symbols

