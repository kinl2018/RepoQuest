from pathlib import Path
import streamlit as st
from scanner import scan_python_files
from ast_parser import parse_file, parse_files
from tempfile import TemporaryDirectory
from dependency import build_dependency_edges_from_results
from graph_analysis import build_graph, save_graph, build_call_graph, save_call_graph
from call_analysis import collect_call_edges

st.set_page_config(page_title="RepoQuest", layout="wide")
st.title("RepoQuest")
if "scan_result" not in st.session_state:
    st.session_state["scan_result"] = None
st.caption("Python 项目代码探索器")
path_text = st.text_input("请输入本地项目路径", value=str(Path.cwd()))

@st.cache_data
def parse_cached(path: str, mtime_ns: int):
    return parse_file(Path(path))

if st.button("扫描仓库", type="primary"):
    if not path_text.strip():
        st.error("请输入项目路径")
    else:
        root = Path(path_text).expanduser().resolve()
        if not root.is_dir():
            st.error("路径不存在或不是目录")
        else:
            files = scan_python_files(root)
            st.session_state["scan_result"] = {"root": root, "files": files}
            st.session_state.pop("dependency_image", None)
            st.session_state.pop("dependency_sub_image", None)
            st.session_state.pop("dependency_sub_node", None)
            st.session_state.pop("dependency_sub_direction", None)
            st.session_state.pop("call_graph_image", None)
            st.session_state.pop("call_graph", None)
            st.session_state.pop("call_sub_direction", None)

if st.session_state["scan_result"] is not None:
    data = st.session_state["scan_result"]
    root = data["root"]
    files = data["files"]
    st.success("扫描完成！")
    st.metric("Python 文件数量", len(files))
    st.subheader("文件列表")
    if files:
        st.dataframe(
            [{"文件路径": str(f.relative_to(root))}
             for f in files],
            hide_index=True
        )
        st.subheader("代码结构浏览器")
        selected = st.selectbox("选择一个 Python 文件", options=files, format_func=lambda f: str(f.relative_to(root)), key="selected_file_details")
        mtime_ns = selected.stat().st_mtime_ns
        try:
            symbols, imports, details = parse_cached(str(selected), mtime_ns)
        except (OSError, UnicodeDecodeError, SyntaxError) as e:
            st.error(f"解析失败：{e}")
        else:
            if symbols:
                st.dataframe(
                    [
                        {
                            "类型": s["kind"],
                            "名称": s["name"],
                            "行号": s["lineno"]
                        }
                        for s in symbols
                    ],
                    hide_index=True
                )

                st.subheader("源代码阅读器")
                selected_index = st.selectbox(
                    "选择一个函数或类",
                    options=range(len(symbols)),
                    format_func=lambda i: (
                        f"{symbols[i]['kind']}: {symbols[i]['name']}"
                    )
                )
                selected_symbol = symbols[selected_index]
                lines = selected.read_text(encoding="utf-8").splitlines()
                start = selected_symbol["start_lineno"] - 1
                end = selected_symbol["end_lineno"]
                source_code = "\n".join(lines[start:end])
                st.caption(f"第 {start + 1}–{end} 行")
                st.code(source_code, language="python")
            else:
                st.info("这个文件没有找到函数或类")

        st.divider()
        st.subheader("模块依赖图")
        if st.button("生成依赖图"):
            results, errors = parse_files(files)
            edges = build_dependency_edges_from_results(root, results)
            graph = build_graph(files, edges)
            with TemporaryDirectory() as tmp:
                output = Path(tmp) / "dependency_graph.svg"
                save_graph(graph, output, root)
                svg_text = output.read_text(encoding="utf-8")
                svg_text = svg_text[svg_text.index("<svg"):]
                st.session_state["dependency_image"] = svg_text
            st.success("依赖图生成成功！")
            st.caption(f"模块数：{graph.number_of_nodes()}，依赖数：{graph.number_of_edges()}")
        if "dependency_image" in st.session_state:
            st.image(st.session_state["dependency_image"], width="stretch")

        st.subheader("局部依赖图")
        selected_node = st.selectbox("选择一个 Python 文件", options=files,
                                     format_func=lambda f: str(f.relative_to(root)), key="selected_dependency_node")
        direction = st.radio("选择依赖方向", options=["正向依赖", "反向依赖"], horizontal=True)
        if st.button("生成局部依赖图"):
            results, errors = parse_files(files)
            edges = build_dependency_edges_from_results(root, results)
            graph = build_graph(files, edges)
            nodes = {selected_node}
            neighbors = graph.successors(selected_node) if direction == "正向依赖" else graph.predecessors(selected_node)
            nodes.update(neighbors)
            subgraph = graph.subgraph(nodes).copy()
            with TemporaryDirectory() as tmp:
                output = Path(tmp) / "dependency_subgraph.svg"
                save_graph(subgraph, output, root)
                sub_svg_text = output.read_text(encoding="utf-8")
                sub_svg_text = sub_svg_text[sub_svg_text.index("<svg"):]
                st.session_state["dependency_sub_image"] = sub_svg_text
                st.session_state["dependency_sub_node"] = selected_node
                st.session_state["dependency_sub_direction"] = direction
            st.success("局部依赖图生成成功！")
            st.caption(f"模块数：{subgraph.number_of_nodes()}，依赖数：{subgraph.number_of_edges()}")
        if "dependency_sub_image" in st.session_state:
            if st.session_state["dependency_sub_node"] == selected_node and st.session_state["dependency_sub_direction"] == direction:
                st.image(st.session_state["dependency_sub_image"], width="stretch")
            else:
                st.info("当前选择的文件已变化，请重新生成局部依赖图")

        st.divider()
        st.subheader("函数调用图")
        if st.button("生成函数调用图"):
            call_edges = []
            errors = []
            for file in files:
                try:
                    call_edges.extend(collect_call_edges(root, file))
                except (OSError, UnicodeDecodeError, SyntaxError) as e:
                    errors.append((file, str(e)))
            call_graph = build_call_graph(call_edges)
            st.session_state["call_graph"] = call_graph
            if call_graph.number_of_nodes() == 0:
                st.info("没有识别到函数调用关系")
                st.session_state.pop("call_graph_image", None)
            else:
                with TemporaryDirectory() as tmp:
                    output = Path(tmp) / "call_graph.svg"
                    save_call_graph(call_graph, output, root)
                    svg_text = output.read_text(encoding="utf-8")
                    svg_text = svg_text[svg_text.index("<svg"):]
                    st.session_state["call_graph_image"] = svg_text
                st.success("函数调用图生成成功！")
                st.caption(f"函数节点数：{call_graph.number_of_nodes()}，调用关系数：{call_graph.number_of_edges()}")
            if errors:
                st.warning(f"有 {len(errors)} 个文件分析失败")
        if "call_graph_image" in st.session_state:
            st.image(st.session_state["call_graph_image"], width="stretch")

        if "call_graph" in st.session_state:
            call_graph = st.session_state["call_graph"]
            if call_graph.number_of_nodes() > 0:
                st.subheader("局部函数调用图")
                selected_call_node = st.selectbox(
                    "选择一个函数",
                    options=sorted(
                        call_graph.nodes,
                        key=lambda node: (str(node[0]), node[1])
                    ),
                    format_func=lambda node: (f"{node[0].relative_to(root)} :: {node[1]}()")
                )
                call_direction = st.radio("选择调用方向", ["它调用了谁", "谁调用了它"], horizontal=True)
                st.session_state["call_sub_direction"] = call_direction
                if st.button("查看该函数的调用关系"):
                    nodes = {selected_call_node}
                    if call_direction == "它调用了谁":
                        neighbors = call_graph.successors(selected_call_node)
                    else:
                        neighbors = call_graph.predecessors(selected_call_node)
                    nodes.update(neighbors)
                    subgraph = call_graph.subgraph(nodes).copy()
                    with TemporaryDirectory() as tmp:
                        output = Path(tmp) / "call_subgraph.svg"
                        save_call_graph(subgraph, output, root)
                        svg_text = output.read_text(encoding="utf-8")
                        svg_text = svg_text[svg_text.index("<svg"):]
                        st.session_state["call_sub_image"] = svg_text
                        st.session_state["call_sub_node"] = selected_call_node
                    st.success("局部调用图生成成功！")
                    st.caption(f"该函数直接调用了{len(set(call_graph.successors(selected_call_node)))}个不同函数")
                if "call_sub_image" in st.session_state:
                    if (st.session_state["call_sub_node"] == selected_call_node
                            and st.session_state["call_sub_direction"] == call_direction):
                        st.image(st.session_state["call_sub_image"], width="stretch")
                    else:
                        st.info("选择已变化，请重新生成局部调用图")

    else:
        st.info("未找到 Python 文件")
