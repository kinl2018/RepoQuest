from pathlib import Path

excluded = {".venv", "venv", "__pycache__", ".git", ".pytest_tmp"}

def scan_python_files(root: Path) -> list[Path]:
    files = [file for file in root.rglob("*.py")
        if file.is_file()
        and not file.is_symlink()
        and not set(file.relative_to(root).parts) & excluded]
    return sorted(files)
