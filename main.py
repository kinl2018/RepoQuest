from pathlib import Path
import sys
from ast_parser import parse_file

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
        print(f"文件：{f.relative_to(current_path)}")
        try:
            symbols = parse_file(f)
        except (OSError, UnicodeDecodeError, SyntaxError) as e:
            print(f"解析失败：{e}")
            continue
        if not symbols:
            print("暂无类或函数定义")
        for symbol in symbols:
            print(f"{symbol['kind']}: {symbol['name']} (第{symbol['lineno']}行)")
    print("Python文件总数：", end='')
    print(len(files))