import ast
import sys
from pathlib import Path
from dependency import resolve_local_module

def extract_local_calls(path: Path):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    functions = {node.name for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}  # 记录当前文件中定义的顶层函数
    edges = set()
    for caller, func in collect_functions(tree):
        if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for node in walk_function_body(func):  # 在该函数中查找调用表达式
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Name):
                callee = node.func.id
                if callee in functions:
                    edges.add((caller, callee, node.lineno))
    return sorted(edges)

def walk_function_body(func):  # 访问顶层函数内部
    nodes = []

    def visit(node):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Lambda)):  # 跳过函数内的嵌套函数/类定义
            return
        nodes.append(node)
        for child in ast.iter_child_nodes(node):
            visit(child)

    for stmt in func.body:
        visit(stmt)
    return nodes

def collect_functions(tree):  # 收集顶层和嵌套函数
    results = []

    def visit_body(body, scope=""):
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                name = f"{scope}.{node.name}" if scope else node.name
                results.append((name, node))
                visit_body(node.body, name)

    visit_body(tree.body)
    return results

def extract_import_bindings(root: Path, file: Path):  # from demo_utils import greet as hello -> {"hello": (Path("demo_utils.py"), "greet")}
    source = file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file))
    bindings = {}
    for node in tree.body:
        if not isinstance(node, ast.ImportFrom):
            continue
        if node.level != 0 or node.module is None:
            continue
        target = resolve_local_module(root, node.module)
        if target is None:
            continue
        for alias in node.names:
            local_name = alias.asname or alias.name
            bindings[local_name] = (target, alias.name)
    return bindings

def extract_cross_file_calls(root: Path, file: Path):
    """
    demo_utils.py: def greet(): print("Hello!")
    demo_main.py: from demo_utils import greet as hello; def main(): hello()
    return [('main', Path_to_demo_utils.py, 'greet', line_no)]
    """
    source = file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file))
    bindings = extract_import_bindings(root, file)
    edges = set()
    for caller, func in collect_functions(tree):
        for node in walk_function_body(func):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name):
                continue
            local_name = node.func.id
            if local_name not in bindings:
                continue
            target_file, target_name = bindings[local_name]
            if not is_top_level_function(target_file, target_name):
                continue
            edges.add((caller, target_file, target_name, node.lineno))
    return sorted(edges)

def is_top_level_function(path: Path, name: str) -> bool:  # 检查文件中(所有顶层函数里)是否含有名为name的函数的定义
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return True
    return False

def extract_module_bindings(root: Path, file: Path):  # import demo_utils as u -> {"u": Path("demo_utils.py")}
    source = file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file))
    bindings = {}
    for node in tree.body:
        if not isinstance(node, ast.Import):
            continue
        for alias in node.names:
            if "." in alias.name:
                continue
            target = resolve_local_module(root, alias.name)
            if target is None:
                continue
            local_name = alias.asname or alias.name
            bindings[local_name] = target
    return bindings

def extract_attribute_calls(root: Path, file: Path):
    source = file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file))
    module_bindings = extract_module_bindings(root, file)
    edges = set()
    for caller, func in collect_functions(tree):
        for node in walk_function_body(func):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Attribute):
                continue
            # 只处理 u.greet() 这种形式
            if not isinstance(node.func.value, ast.Name):
                continue
            module_alias = node.func.value.id  # "u"
            function_name = node.func.attr  # "greet"
            if module_alias not in module_bindings:
                continue
            target_file = module_bindings[module_alias]
            if not is_top_level_function(target_file, function_name):
                continue
            edges.add((caller, target_file, function_name, node.lineno))
    return sorted(edges)

def collect_call_edges(root: Path, file: Path):
    root = root.resolve()
    file = file.resolve()
    edges = set()
    for caller, callee, lineno in extract_local_calls(file):  # 第一部分：同文件函数调用
        source = (file, caller)
        target = (file, callee)
        edges.add((source, target, lineno))
    for analyzer in (extract_cross_file_calls, extract_attribute_calls):  # 第二部分：两种跨文件函数调用
        for caller, target_file, target_name, lineno in analyzer(root, file):
            source = (file, caller)
            target = (target_file, target_name)
            edges.add((source, target, lineno))
    return sorted(edges)
