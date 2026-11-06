from pathlib import Path
import sys
from ast_parser import parse_files
from dependency import  build_dependency_edges_from_results
from graph_analysis import build_graph, find_cycles, save_graph

if __name__ == '__main__':
    current_path = Path.cwd() if len(sys.argv) <= 1 else Path(sys.argv[1])
    if not current_path.exists():
        print('指定路径不存在')
        sys.exit(1)
    if not current_path.is_dir():
        print('指定路径不是目录')
        sys.exit(1)
    excluded = {".venv", "venv", "__pycache__", ".git", ".pytest_tmp"}
    print(f"项目路径: {current_path}")
    files = sorted([x for x in current_path.rglob("*.py") if not set(x.relative_to(current_path).parts) & excluded])
    print(f"Python文件总数：{len(files)}")
    results, errors = parse_files(files)

    for file in files:
        print(f"文件：{file.relative_to(current_path)}")
        if file in errors:
            print(f"解析失败：{errors[file]}")
            continue
        data = results[file]
        symbols = data["symbols"]
        imports = data["imports"]
        if not symbols:
            print("暂无类或函数定义")
        for symbol in symbols:
            print(f"{symbol['kind']}: {symbol['name']} (第{symbol['lineno']}行)")
        if not imports:
            print("暂无import")
        for module in imports:
            print(f"import: {module}")

    edges = build_dependency_edges_from_results(current_path, results)
    print("项目依赖关系：")
    for source, target in edges:
        print(f"{source.relative_to(current_path)} -> {target.relative_to(current_path)}")
    graph = build_graph(files, edges)
    cycles = find_cycles(graph)
    print("循环依赖检测：")
    if len(cycles) == 0:
        print("未发现循环依赖")
    else:
        print(f"发现{len(cycles)}个循环依赖")
        for cycle in cycles:
            paths = [str(file.relative_to(current_path)) for file in cycle]
            paths.append(paths[0])
            print("->".join(paths))
    output = current_path / "dependency_graph.png"
    save_graph(graph, output, current_path)
    print(f"依赖图已保存至：{output}")