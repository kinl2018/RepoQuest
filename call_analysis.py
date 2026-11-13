"""Conservative AST call graph. Edges are syntactically resolved, never executed."""
import ast
from pathlib import Path
import networkx as nx
from dependency import file_to_module, resolve_local_module, resolve_relative_import
from scanner import scan_python_files


def collect_functions(tree):
    return [(name, node) for name, node in collect_definitions(tree) if not isinstance(node, ast.ClassDef)]


def collect_definitions(tree):
    found = []
    def visit(node, scope=""):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            scope = f"{scope}.{node.name}" if scope else node.name
            found.append((scope, node))
        for child in ast.iter_child_nodes(node):
            visit(child, scope)
    visit(tree)
    return found


def walk_function_body(func):
    nodes = []
    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):
            return
        nodes.append(node)
        for child in ast.iter_child_nodes(node):
            visit(child)
    for stmt in func.body:
        visit(stmt)
    return nodes


def build_call_index(root: Path):
    root = Path(root).resolve()
    trees, errors, symbols = {}, {}, {}
    for file in scan_python_files(root):
        try:
            tree = ast.parse(file.read_text(encoding="utf-8"), filename=str(file))
        except (OSError, UnicodeError, SyntaxError) as exc:
            errors[file.relative_to(root).as_posix()] = str(exc)
            continue
        trees[file] = tree
        for name, node in collect_functions(tree):
            symbols[(file, name)] = node
    graph = nx.DiGraph()
    graph.add_nodes_from(symbols)
    for file, tree in trees.items():
        imports = {}
        for node in tree.body:
            if isinstance(node, ast.ImportFrom):
                module = node.module
                if node.level:
                    current = file_to_module(root, file)
                    # __init__ is a package, not a child module.
                    if file.name == "__init__.py":
                        current += ".__init__"
                    module = resolve_relative_import(current, module, node.level)
                target = resolve_local_module(root, module) if module else None
                if target:
                    for alias in node.names:
                        imports[alias.asname or alias.name] = (target.resolve(), alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    target = resolve_local_module(root, alias.name)
                    if target:
                        imports[alias.asname or alias.name] = (target.resolve(), "")
        class_scopes = {name for name, node in collect_definitions(tree) if isinstance(node, ast.ClassDef)}
        scopes = collect_functions(tree) + [("<module>", tree)]
        for caller, func in scopes:
            body = walk_function_body(func)
            # Parameters and assignments shadow imported or global function names.
            assigned = {n.id for n in body if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
            shadowed = set(assigned)
            if caller != "<module>":
                for n in body:
                    if isinstance(n, (ast.Import, ast.ImportFrom)):
                        shadowed.update(a.asname or a.name.split(".")[0] for a in n.names)
            if isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
                shadowed.update(a.arg for a in [*func.args.posonlyargs, *func.args.args, *func.args.kwonlyargs])
                shadowed.update(a.arg for a in [func.args.vararg, func.args.kwarg] if a)
            for node in body:
                if not isinstance(node, ast.Call):
                    continue
                target = None
                call = node.func
                if isinstance(call, ast.Name) and call.id not in shadowed:
                    # Closest lexical definition wins, then a top-level definition/import.
                    parts = caller.split(".")[:-1]
                    for size in range(len(parts), -1, -1):
                        if ".".join(parts[:size]) in class_scopes:
                            continue
                        candidate = (file, ".".join(parts[:size] + [call.id]))
                        if candidate in symbols:
                            target = candidate
                            break
                    if target is None:
                        target = imports.get(call.id)
                elif isinstance(call, ast.Attribute):
                    names, value = [], call
                    while isinstance(value, ast.Attribute):
                        names.insert(0, value.attr)
                        value = value.value
                    if isinstance(value, ast.Name):
                        base = value.id
                        parent = caller.rsplit(".", 1)[0]
                        if base in {"self", "cls"} and base not in assigned and parent in class_scopes and len(names) == 1:
                            target = (file, parent + "." + names[0])
                        elif base not in shadowed:
                            dotted = [base] + names
                            for size in range(len(dotted)-1, 0, -1):
                                binding = imports.get(".".join(dotted[:size]))
                                if binding:
                                    target_file, prefix = binding
                                    target = (target_file, ".".join(([prefix] if prefix else []) + dotted[size:]))
                                    break
                if target not in symbols:
                    continue
                source = (file, caller)
                if graph.has_edge(source, target):
                    graph[source][target]["lines"].append(node.lineno)
                else:
                    graph.add_edge(source, target, lines=[node.lineno])
    graph.graph["errors"] = errors
    return graph


def collect_call_edges(root: Path, file: Path):
    """Compatibility API; services build the graph once for the whole repository."""
    graph = build_call_index(root)
    return sorted((source, target, line) for source, target, data in graph.edges(data=True)
                  if source[0] == Path(file).resolve() for line in data["lines"])
