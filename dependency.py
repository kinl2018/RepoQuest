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
    """
    #input: root = tmp_path, file = tmp_path/services/matcher.py: from .utils import helper
    file_to_module: services/matcher.py -> current_module = "services.matcher"
    extract_import_details: -> imports_list = {"module": "utils", "level": 1, "names": ["helper"]}
    resolve_relative_import: -> module_name = "services.utils"
    resolve_local_module: -> target = tmp_path/services/utils.py
    """
    dependencies = []
    source = file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(file))
    current_module = file_to_module(root, file)
    imports_list = extract_import_details(tree)  # [{"module": node.module, "level": node.level, "names": [alias.name for alias in node.names]}]
    for import_detail in imports_list:
        if import_detail["level"] == 0 or import_detail["module"] is None:
            continue
        module_name = resolve_relative_import(current_module, import_detail["module"], import_detail["level"])
        if module_name is None:
            continue
        target = resolve_local_module(root, module_name)
        if target is not None:
            dependencies.append(target)
    return dependencies
