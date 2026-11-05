from pathlib import Path
import sys

if __name__ == '__main__':
    current_path = Path.cwd() if len(sys.argv) <= 1 else Path(sys.argv[1])
    if not current_path.exists():
        print('指定路径不存在')
        sys.exit(1)
    if not current_path.is_dir():
        print('指定路径不是目录')
        sys.exit(1)
    excluded = {".venv", "venv", "__pycache__", ".git"}
    print(f"项目路径: {current_path}")
    files = sorted([x for x in current_path.rglob("*.py") if not set(x.relative_to(current_path).parts) & excluded])
    for f in files:
        print(f.relative_to(current_path))
    print("Python文件总数：", end='')
    print(len(files))