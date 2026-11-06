from pathlib import Path
import ast
from ast_parser import extract_import_details

def resolve_local_module(root: Path, module_name: str) -> Path | None:
    f_list = module_name.split(".")
    f_dir = root
    for f in f_list[:-1]:
        f_dir = f_dir / f
    f_dir1 = f_dir / f"{f_list[-1]}.py"
    if f_dir1.is_file():
        return f_dir1
    f_dir2 = f_dir / f_list[-1] / "__init__.py"
    return f_dir2 if f_dir2.is_file() else None

def resolve_relative_import(current_module: str, module: str | None, level: int) -> str | None:
    # resolve_relative_import("myapp.services.matcher", None, 1) -> "myapp.services"
    if level == 0:
        return module
    parts = current_module.split(".")
    if level >= len(parts):
        return None
    prefix = ".".join(parts[:-level])
    if module is None:
        return prefix
    return f"{prefix}.{module}"

def file_to_module(root: Path, file: Path) -> str:
    # services/matcher.py -> "services.matcher"
    relative = file.relative_to(root)
    relative = relative.with_suffix("")
    return ".".join(relative.parts)

def find_local_dependencies(root: Path, file: Path) -> list[Path]:
    source = file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file))
    import_details = extract_import_details(tree)
    return find_dependencies_from_details(root, file, import_details)

def find_dependencies_from_details(root: Path, file: Path, import_details: list[dict]) -> list[Path]:
    """
        input: root = tmp_path, file = tmp_path/services/matcher.py: from .utils import helper
        file_to_module: services/matcher.py -> current_module = "services.matcher"
        extract_import_details: -> import_details = {"module": "utils", "level": 1, "names": ["helper"]}
        resolve_relative_import: -> module_name = "services.utils"
        resolve_local_module: -> target = tmp_path/services/utils.py
        output: dependencies = [targets]
    """
    dependencies = []
    current_module = file_to_module(root, file)
    for import_detail in import_details:
        module_name = resolve_relative_import(current_module, import_detail["module"], import_detail["level"])
        if module_name is None:
            continue
        if import_detail["module"] is not None:
            target_file = resolve_local_module(root, module_name)
            if target_file is not None:
                dependencies.append(target_file)
        else:  # from . import config -> "services.config" -> services/config.py
            for name in import_detail["names"]:
                target_file = resolve_local_module(root, f"{module_name}.{name}")
                if target_file is not None:
                    dependencies.append(target_file)
    return dependencies

def build_dependency_edges_from_results(root: Path, results: dict) -> list[tuple[Path, Path]]:
    # output: edges = [(Path("main.py"), Path("services/utils.py"))], 即 [(source1, target1),]
    edges = []
    for file, result in results.items():
        dependencies = find_dependencies_from_details(root, file, import_details=result["import_details"])
        edges.extend((file, dependency) for dependency in dependencies)
    return sorted(set(edges))

def build_dependency_edges(root: Path, files: list[Path]) -> list[tuple[Path, Path]]:
    # output: edges = [(Path("main.py"), Path("services/utils.py"))], 即 [(source1, target1),]
    edges = []
    for file in files:
        if not file.is_file():
            continue
        try:
            dependencies = find_local_dependencies(root, file)
            edges.extend((file, dependency) for dependency in dependencies)
        except (OSError, UnicodeDecodeError, SyntaxError) as e:
            print(f"文件解析失败：{e}")
            continue
    return sorted(set(edges))