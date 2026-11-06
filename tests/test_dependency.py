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

def test_find_absolute_dependency(tmp_path):
    # 1. 创建 services 目录
    services = tmp_path / "services"
    services.mkdir()
    # 2. 创建被导入的 utils.py
    target = services / "utils.py"
    target.write_text("def helper(): pass", encoding="utf-8")
    # 3. 创建 main.py，使用绝对导入
    source = tmp_path / "main.py"
    source.write_text("from services.utils import helper", encoding="utf-8")
    # 4. 检查是否找到依赖
    result = find_local_dependencies(tmp_path, source)
    assert result == [target]

def test_find_normal_import_dependency(tmp_path):
    services = tmp_path / "services"
    services.mkdir()

    target = services / "utils.py"
    target.write_text("", encoding="utf-8")

    source = tmp_path / "main.py"
    source.write_text(
        "import services.utils",
        encoding="utf-8"
    )

    result = find_local_dependencies(tmp_path, source)

    assert result == [target]

def test_find_relative_import_without_module(tmp_path):
    services = tmp_path / "services"
    services.mkdir()

    (services / "__init__.py").write_text("", encoding="utf-8")

    source = services / "matcher.py"
    source.write_text(
        "from . import config",
        encoding="utf-8"
    )

    target = services / "config.py"
    target.write_text(
        "DEBUG = True",
        encoding="utf-8"
    )

    result = find_local_dependencies(tmp_path, source)

    assert result == [target]


from dependency import build_dependency_edges

def test_build_dependency_edges(tmp_path):
    services = tmp_path / "services"
    services.mkdir()

    # 被依赖文件
    target = services / "utils.py"
    target.write_text(
        "def helper(): pass",
        encoding="utf-8"
    )

    # 第一个源文件：绝对导入
    main = tmp_path / "main.py"
    main.write_text(
        "import services.utils",
        encoding="utf-8"
    )

    # 第二个源文件：相对导入
    matcher = services / "matcher.py"
    matcher.write_text(
        "from .utils import helper",
        encoding="utf-8"
    )

    files = [main, matcher, target]

    result = build_dependency_edges(tmp_path, files)

    assert result == [
        (main, target),
        (matcher, target)
    ]


def test_build_dependency_edges_no_duplicates(tmp_path):
    target = tmp_path / "utils.py"
    target.write_text("", encoding="utf-8")

    source = tmp_path / "main.py"
    source.write_text(
        "import utils\n"
        "from utils import helper\n",
        encoding="utf-8"
    )

    files = [source, target]
    result = build_dependency_edges(tmp_path, files)

    assert result == [(source, target)]


def test_build_edges_with_broken_file(tmp_path):
    # 被依赖的正常文件
    target = tmp_path / "utils.py"
    target.write_text("", encoding="utf-8")

    # 正常文件
    good = tmp_path / "good.py"
    good.write_text("import utils", encoding="utf-8")

    # 存在语法错误的文件
    bad = tmp_path / "bad.py"
    bad.write_text("def broken(", encoding="utf-8")

    result = build_dependency_edges(
        tmp_path, [bad, good, target]
    )

    assert result == [(good, target)]


from dependency import find_dependencies_from_details

def test_find_dependencies_from_details(tmp_path):
    target = tmp_path / "utils.py"
    target.write_text("", encoding="utf-8")

    source = tmp_path / "main.py"
    source.write_text("import utils", encoding="utf-8")

    # 模拟 parse_files() 已经提取好的数据
    details = [
        {
            "module": "utils",
            "level": 0,
            "names": []
        }
    ]

    result = find_dependencies_from_details(
        tmp_path, source, details
    )

    assert result == [target]



from dependency import build_dependency_edges_from_results

def test_build_edges_from_results(tmp_path):
    target = tmp_path / "utils.py"
    target.write_text("", encoding="utf-8")

    source = tmp_path / "main.py"

    results = {
        source: {
            "import_details": [
                {"module": "utils", "level": 0, "names": []}
            ]
        }
    }

    edges = build_dependency_edges_from_results(
        tmp_path, results
    )

    assert edges == [(source, target)]
