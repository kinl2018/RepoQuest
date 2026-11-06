from dependency import resolve_local_module


def test_resolve_existing_module(tmp_path):
    # 创建一个临时 Python 文件
    target = tmp_path / "ast_parser.py"
    target.write_text("", encoding="utf-8")

    result = resolve_local_module(tmp_path, "ast_parser")

    assert result == target


def test_resolve_missing_module(tmp_path):
    result = resolve_local_module(tmp_path, "not_exists")

    assert result is None


def test_resolve_nested_module(tmp_path):
    services = tmp_path / "services"
    services.mkdir()

    target = services / "matcher.py"
    target.write_text("", encoding="utf-8")

    result = resolve_local_module(tmp_path, "services.matcher")

    assert result == target

from dependency import resolve_relative_import
def test_resolve_package(tmp_path):
    services = tmp_path / "services"
    services.mkdir()

    init_file = services / "__init__.py"
    init_file.write_text("", encoding="utf-8")

    result = resolve_local_module(tmp_path, "services")

    assert result == init_file

def test_relative_import_same_package():
    result = resolve_relative_import(
        "myapp.services.matcher", "utils", 1
    )
    assert result == "myapp.services.utils"


def test_relative_import_parent_package():
    result = resolve_relative_import(
        "myapp.services.matcher", "common", 2
    )
    assert result == "myapp.common"


def test_relative_import_without_module():
    result = resolve_relative_import(
        "myapp.services.matcher", None, 1
    )
    assert result == "myapp.services"


def test_relative_import_beyond_top_level():
    result = resolve_relative_import(
        "myapp.services.matcher", "utils", 3
    )
    assert result is None


from dependency import file_to_module

def test_file_to_module(tmp_path):
    services = tmp_path / "services"
    services.mkdir()

    file = services / "matcher.py"
    file.write_text("", encoding="utf-8")

    result = file_to_module(tmp_path, file)

    assert result == "services.matcher"


from dependency import find_local_dependencies

def test_find_relative_dependency(tmp_path):
    services = tmp_path / "services"
    services.mkdir()

    (services / "__init__.py").write_text("")

    source = services / "matcher.py"
    source.write_text(
        "from .utils import helper",
        encoding="utf-8"
    )

    target = services / "utils.py"
    target.write_text(
        "def helper(): pass",
        encoding="utf-8"
    )

    result = find_local_dependencies(tmp_path, source)

    assert result == [target]

