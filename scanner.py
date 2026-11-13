"""Repository file policy shared by tools and indexing."""
from pathlib import Path

EXCLUDED = {".venv", "venv", "env", "__pycache__", ".git", ".idea", ".codex",
            ".pytest_cache", ".pytest_tmp", ".repoquest_index", ".repoquest_runs",
            "node_modules", "site-packages", "build", "dist"}


def source_category(file: str) -> str:
    path = Path(file)
    if set(path.parts) & {"tests", "test"} or path.name.startswith("test_") or path.name.endswith("_test.py"):
        return "test"
    if set(path.parts) & {"examples", "demos", "benchmarks"} or path.stem.endswith("_demo"):
        return "demo"
    return "source"


def scan_python_files(root: Path, *, include_tests: bool = True, include_demos: bool = True) -> list[Path]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"不是有效的仓库目录：{root}")
    files, pending = [], [root]
    while pending:
        directory = pending.pop()
        for entry in directory.iterdir():
            if entry.is_symlink() or entry.name in EXCLUDED:
                continue
            if entry.is_dir():
                pending.append(entry)
            elif entry.suffix == ".py":
                category = source_category(entry.relative_to(root).as_posix())
                if (include_tests or category != "test") and (include_demos or category != "demo"):
                    files.append(entry)
    return sorted(files)
