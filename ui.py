from pathlib import Path
import streamlit as st
from scanner import scan_python_files
from ast_parser import parse_file, parse_files
from tempfile import TemporaryDirectory
from dependency import build_dependency_edges_from_results
from graph_analysis import build_graph, save_graph, build_call_graph, save_call_graph
from call_analysis import collect_call_edges
from agent_core import ask_repo
from sentence_transformers import SentenceTransformer
from rag_index import load_index
from rag_bm25 import build_bm25
from rag_hybrid import hybrid_search

@st.cache_data
def parse_cached(path: str, mtime_ns: int):
    return parse_file(Path(path))

def show_tool_trace(trace):
    if not trace:
        return

    with st.expander(f"查看工具执行记录（{len(trace)} 次）"):
        for i, item in enumerate(trace, start=1):
            st.markdown(f"**第 {i} 次：{item['tool']}**")
            st.caption("调用参数")
            st.json(item["args"])
            st.caption("执行结果")
            result = item["result"]

            if isinstance(result, dict) and "error" in result:
                st.error(result["error"])

            elif item["tool"] == "find_symbol" and isinstance(result, list):
                for symbol in result:
                    st.caption(f"{symbol['file']}:{symbol['lineno']} · {symbol['name']}")
                    st.code(symbol["source"], language="python")

            elif item["tool"] == "read_source" and isinstance(result, dict):
                st.caption(f"{result['file']} 第 {result['start_line']}~{result['end_line']} 行")
                st.code(result["source"], language="python")

            else:
                st.json(result)

            st.divider()

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("BAAI/bge-small-zh-v1.5", device="cpu")

st.set_page_config(page_title="RepoQuest", layout="wide")
st.title("RepoQuest")
st.caption("Python 项目代码探索器")

# 初始化扫描状态
if "scan_result" not in st.session_state:
    st.session_state["scan_result"] = None
data = None
# ===== 左侧工作区 =====
with st.sidebar:
    st.header("工作区")

    path_text = st.text_input(
        "请输入本地项目路径",
        value=str(Path.cwd())
    )

    if st.button("扫描仓库", type="primary", use_container_width=True):
        if not path_text.strip():
            st.error("项目路径不能为空")
        else:
            root = Path(path_text).expanduser().resolve()

            if not root.is_dir():
                st.error("路径不存在或不是目录")
            else:
                old_data = st.session_state["scan_result"]
                files = scan_python_files(root)

                st.session_state["scan_result"] = {
                    "root": root,
                    "files": files
                }

                # 清除旧图表缓存
                for key in (
                    "dependency_image",
                    "dependency_sub_image",
                    "dependency_sub_node",
                    "dependency_sub_direction",
                    "call_graph_image",
                    "call_graph",
                    "call_sub_image",
                    "call_sub_node",
                    "call_sub_direction",
                    "dependency_stats"
                ):
                    st.session_state.pop(key, None)

                # 切换仓库时清空旧聊天，避免混淆
                if old_data is None or old_data["root"] != root:
                    st.session_state.pop("chat_history", None)

                st.success("扫描完成！")

    # 显示当前仓库状态
    data = st.session_state["scan_result"]
    if data is not None:
        st.divider()
        st.caption(f"当前仓库：{data['root'].name}")
        st.metric("Python 文件数量", len(data["files"]))

if data is not None:
    root = data["root"]
    files = data["files"]
tab_files, tab_deps, tab_calls, tab_ai, tab_rag = st.tabs([
    "文件浏览",
    "模块依赖",
    "函数调用",
    "AI 代码导游",
    "RAG 检索"
])

with tab_files:
    st.subheader("文件浏览")

    if data is None:
        st.info("请先在左侧扫描仓库")
    elif not files:
        st.info("当前仓库没有 Python 文件")
    else:
        # 左侧1份，右侧2份
        left, right = st.columns([1, 2], gap="large")

        with left:
            st.markdown("#### 文件导航")

            selected = st.selectbox(
                "选择 Python 文件",
                options=files,
                format_func=lambda f: str(f.relative_to(root))
            )

            try:
                symbols, imports, details = parse_cached(
                    str(selected),
                    selected.stat().st_mtime_ns
                )
            except (OSError, UnicodeDecodeError, SyntaxError) as e:
                st.error(f"解析失败：{e}")
                symbols = None

            selected_symbol = None

            if symbols is not None:
                st.caption(f"发现 {len(symbols)} 个符号")

                if symbols:
                    selected_index = st.selectbox(
                        "选择函数或类",
                        options=range(len(symbols)),
                        format_func=lambda i: (
                            f"{symbols[i]['kind']}: {symbols[i]['name']}"
                        )
                    )
                    selected_symbol = symbols[selected_index]
                else:
                    st.info("这个文件没有函数或类定义")

        with right:
            st.markdown("#### 源代码阅读器")

            if selected_symbol is not None:
                lines = selected.read_text(
                    encoding="utf-8"
                ).splitlines()

                start = selected_symbol["start_lineno"] - 1
                end = selected_symbol["end_lineno"]

                source_code = "\n".join(lines[start:end])

                st.caption(
                    f"{selected.relative_to(root)} "
                    f"· 第 {start + 1}~{end} 行"
                )
                st.code(source_code, language="python")

            else:
                st.info("请选择一个函数或类来查看源码")

with tab_deps:
    st.subheader("模块依赖分析")
    if data is None:
        st.info("请先在左侧扫描仓库")
    elif not files:
        st.info("当前仓库没有 Python 文件")
    else:

        view = st.radio(
            "依赖分析模式",
            ["完整依赖", "局部依赖"],
            horizontal=True
        )

        if view == "完整依赖":
            left, right = st.columns([1, 2], gap="large")

            with left:
                with st.container(border=True):
                    st.markdown("#### 分析控制台")
                    st.caption("分析当前仓库中 Python 模块之间的导入关系")

                    if st.button("生成完整依赖图", type="primary"):
                        results, errors = parse_files(files)
                        edges = build_dependency_edges_from_results(root, results)
                        graph = build_graph(files, edges)

                        with TemporaryDirectory() as tmp:
                            output = Path(tmp) / "dependency_graph.svg"
                            save_graph(graph, output, root)

                            svg_text = output.read_text(encoding="utf-8")
                            svg_text = svg_text[svg_text.index("<svg"):]

                            st.session_state["dependency_image"] = svg_text

                        # 保存统计信息
                        st.session_state["dependency_stats"] = {
                            "nodes": graph.number_of_nodes(),
                            "edges": graph.number_of_edges()
                        }

                    # 生成后展示统计信息
                    if "dependency_stats" in st.session_state:
                        stats = st.session_state["dependency_stats"]
                        st.metric("模块数量", stats["nodes"])
                        st.metric("依赖关系数量", stats["edges"])

            with right:
                with st.container(border=True):
                    st.markdown("#### 完整依赖图")

                    if "dependency_image" in st.session_state:
                        st.image(
                            st.session_state["dependency_image"],
                            width="stretch"
                        )
                    else:
                        st.info("点击左侧按钮生成模块依赖图")

        else:
            left, right = st.columns([1, 2], gap="large")

            with left:
                with st.container(border=True):
                    st.markdown("#### 局部依赖控制台")
                    selected_node = st.selectbox("选择一个 Python 文件", options=files,
                                                 format_func=lambda f: str(f.relative_to(root)),
                                                 key="selected_dependency_node")
                    direction = st.radio("选择依赖方向", options=["正向依赖", "反向依赖"], horizontal=True)
                    if st.button("生成局部依赖图"):
                        results, errors = parse_files(files)
                        edges = build_dependency_edges_from_results(root, results)
                        graph = build_graph(files, edges)
                        nodes = {selected_node}
                        neighbors = graph.successors(selected_node) if direction == "正向依赖" else graph.predecessors(
                            selected_node)
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
                        st.caption(f"模块数：{subgraph.number_of_nodes()}，依赖数：{subgraph.number_of_edges()}")

            with right:
                with st.container(border=True):
                    st.markdown("#### 局部依赖图")
                    if "dependency_sub_image" in st.session_state:
                        if st.session_state["dependency_sub_node"] == selected_node and st.session_state["dependency_sub_direction"] == direction:
                            st.image(st.session_state["dependency_sub_image"], width="stretch")
                        else:
                            st.info("点击左侧按钮生局部依赖图")


with tab_calls:
    st.subheader("函数调用分析")
    if data is None:
        st.info("请先在左侧扫描仓库")
    elif not files:
        st.info("当前仓库没有 Python 文件")
    else:
        view = st.radio(
            "调用分析模式",
            ["完整调用", "局部调用"],
            horizontal=True
        )
        if view == "完整调用":
            left, right = st.columns([1, 2], gap="large")

            with left:
                with st.container(border=True):
                    st.markdown("#### 分析控制台")
                    st.caption("分析当前仓库中的函数调用关系")

                    if st.button("生成函数调用图", type="primary"):
                        for key in (
                                "call_sub_image",
                                "call_sub_node",
                                "call_sub_direction"
                        ):
                            st.session_state.pop(key, None)
                        call_edges = []
                        errors = []

                        for file in files:
                            try:
                                call_edges.extend(
                                    collect_call_edges(root, file)
                                )
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

                        if errors:
                            st.warning(f"有 {len(errors)} 个文件分析失败")

                    if "call_graph" in st.session_state:
                        graph = st.session_state["call_graph"]
                        st.metric("函数节点数", graph.number_of_nodes())
                        st.metric("调用关系数", graph.number_of_edges())

            with right:
                with st.container(border=True):
                    st.markdown("#### 完整函数调用图")

                    if "call_graph_image" in st.session_state:
                        st.image(
                            st.session_state["call_graph_image"],
                            width="stretch"
                        )
                    else:
                        st.info("请点击左侧按钮生成函数调用图")

        else:
            if "call_graph" not in st.session_state:
                st.info("请先切换到完整调用模式，生成函数调用图")
            else:
                call_graph = st.session_state["call_graph"]

                if call_graph.number_of_nodes() == 0:
                    st.info("当前没有识别到函数调用关系")
                else:
                    left, right = st.columns([1, 2], gap="large")

                    with left:
                        with st.container(border=True):
                            st.markdown("#### 局部调用控制台")

                            selected_call_node = st.selectbox(
                                "选择一个函数",
                                options=sorted(
                                    call_graph.nodes,
                                    key=lambda node: (str(node[0]), node[1])
                                ),
                                format_func=lambda node: (
                                    f"{node[0].relative_to(root)} :: {node[1]}()"
                                )
                            )

                            call_direction = st.radio(
                                "选择调用方向",
                                ["它调用了谁", "谁调用了它"],
                                horizontal=True
                            )

                            if st.button("生成局部调用图"):
                                if call_direction == "它调用了谁":
                                    neighbors = set(
                                        call_graph.successors(selected_call_node)
                                    )
                                else:
                                    neighbors = set(
                                        call_graph.predecessors(selected_call_node)
                                    )

                                nodes = {selected_call_node} | neighbors
                                subgraph = call_graph.subgraph(nodes).copy()

                                with TemporaryDirectory() as tmp:
                                    output = Path(tmp) / "call_subgraph.svg"
                                    save_call_graph(subgraph, output, root)

                                    svg_text = output.read_text(
                                        encoding="utf-8"
                                    )
                                    svg_text = svg_text[svg_text.index("<svg"):]

                                    st.session_state["call_sub_image"] = svg_text
                                    st.session_state["call_sub_node"] = selected_call_node
                                    st.session_state["call_sub_direction"] = call_direction

                                st.success(f"找到 {len(neighbors)} 个直接关联函数")

                    with right:
                        with st.container(border=True):
                            st.markdown("#### 局部函数调用图")

                            image = st.session_state.get("call_sub_image")
                            saved_node = st.session_state.get("call_sub_node")
                            saved_direction = st.session_state.get("call_sub_direction")

                            if (
                                    image is not None
                                    and saved_node == selected_call_node
                                    and saved_direction == call_direction
                            ):
                                st.image(image, width="stretch")
                            else:
                                st.info("请在左侧选择函数并生成局部调用图")

with tab_ai:
    if st.session_state["scan_result"] is None:
        st.info("请先扫描仓库")
    else:
        data = st.session_state["scan_result"]
        root = data["root"]
        files = data["files"]
        if files:
            st.subheader("AI 代码导游")

            # 初始化聊天历史
            if "chat_history" not in st.session_state:
                st.session_state["chat_history"] = []

            # 展示之前的聊天记录
            for message in st.session_state["chat_history"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
                    if message["role"] == "assistant":
                        show_tool_trace(message.get("trace", []))

            # 用户输入问题
            question = st.chat_input("向 RepoQuest 提问，例如：parse_file 是做什么的？")

            if question:
                # 显示用户的问题
                st.session_state["chat_history"].append({
                    "role": "user",
                    "content": question
                })

                with st.chat_message("user"):
                    st.markdown(question)

                # 调用 Agent
                with st.chat_message("assistant"):
                    with st.spinner("RepoQuest 正在分析代码..."):
                        try:
                            history = [
                                {"role": m["role"], "content": m["content"]}
                                for m in st.session_state["chat_history"][:-1]
                            ]

                            answer, trace = ask_repo(question, root, history=history)

                        except Exception as e:
                            answer = f"分析失败：{e}"
                            trace = []

                    st.markdown(answer)
                    show_tool_trace(trace)

                # 保存回答
                st.session_state["chat_history"].append({
                    "role": "assistant",
                    "content": answer,
                    "trace": trace
                })

        else:
            st.info("未找到 Python 文件")


with tab_rag:
    st.subheader("RAG 代码检索")

    if data is None:
        st.info("请先扫描仓库")
    else:
        root = data["root"]
        index_dir = root / ".repoquest_index"

        index_exists = (
            (index_dir / "documents.json").is_file()
            and (index_dir / "embeddings.npy").is_file()
        )

        if not index_exists:
            st.warning("当前仓库尚未建立 RAG 索引")
        else:
            with st.form("rag_search_form"):
                question = st.text_input(
                    "请输入检索问题",
                    placeholder="例如：get_callers"
                )
                submitted = st.form_submit_button("搜索源码")

            if submitted:
                if not question.strip():
                    st.warning("请输入问题")
                else:
                    try:
                        with st.spinner("正在检索相关源码..."):
                            documents, embeddings = load_index(
                                index_dir
                            )

                            model = load_embedding_model()
                            bm25 = build_bm25(documents)

                            results = hybrid_search(
                                question,
                                documents,
                                embeddings,
                                model,
                                bm25,
                                k=5
                            )

                            st.session_state["rag_results"] = {
                                "root": str(root),
                                "question": question,
                                "results": results
                            }

                    except Exception as e:
                        st.error(f"检索失败：{e}")

            saved = st.session_state.get("rag_results")

            if saved and saved["root"] == str(root):
                st.caption(f"检索问题：{saved['question']}")

                for rank, result in enumerate(
                    saved["results"], start=1
                ):
                    meta = result["metadata"]

                    with st.expander(
                        f"Top {rank} · "
                        f"{meta['file']}::{meta['name']}",
                        expanded=(rank == 1)
                    ):
                        st.caption(
                            f"L{meta['start_line']}-"
                            f"{meta['end_line']} | "
                            f"RRF: {result['score']:.6f}"
                        )

                        st.code(
                            result["content"],
                            language="python"
                        )
