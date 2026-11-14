"""Streamlit panels for the shared Python services."""
import json
from pathlib import Path
import streamlit as st
from rag_index import source_fingerprint
from repo_service import open_repository, run_investigation


@st.cache_resource
def repository_cached(root, fingerprint, use_vectors, include_tests, include_demos, manifest_stamp):
    return open_repository(Path(root), use_vectors=use_vectors, include_tests=include_tests, include_demos=include_demos)


def get_repository(root, use_vectors, include_tests=False, include_demos=False):
    manifest = root / ".repoquest_index" / "manifest.json"
    return repository_cached(str(root), source_fingerprint(root, include_tests=True, include_demos=True),
                             use_vectors, include_tests, include_demos,
                             manifest.stat().st_mtime_ns if manifest.is_file() else 0)


def show_run(result):
    st.markdown(result.get("answer") or "调查已保存，可以使用线程 ID 恢复。")
    run = result["run"]
    st.caption(f"线程 ID：{run['thread_id']} · 停止原因：{result.get('stop_reason') or 'checkpoint'}")
    cols = st.columns(4)
    cols[0].metric("Coverage", f"{result.get('coverage_stats', {}).get('coverage', 0):.0%}")
    cols[1].metric("工具步骤", result.get("steps", 0))
    cols[2].metric("耗时", f"{run['latency_ms']/1000:.2f}s")
    usage = run["usage"]
    cols[3].metric("LLM Tokens", usage["prompt_tokens"]+usage["completion_tokens"] if usage["usage_complete"] else "未知")
    with st.expander("查看执行轨迹和证据"):
        coverage = [e["coverage"]["coverage"] for e in result["trace"] if "coverage" in e]
        if coverage:
            st.line_chart({"Coverage": coverage})
        st.json(result["trace"])
        st.json(result.get("context_sources", []))
        st.json(result.get("context_stats", {}))
    st.download_button("下载调查记录", json.dumps(result, ensure_ascii=False, indent=2),
                       file_name=f"repoquest-{run['thread_id']}.json", mime="application/json")


def agent_panel(root):
    if root is None:
        st.info("请先扫描仓库")
        return
    st.subheader("Agent 自主调查")
    vectors = st.checkbox("使用向量索引", key="agent_vectors")
    st.caption("未启用向量时使用精确符号、BM25 与 AST 工具；首次构建向量索引可运行 python -m repoquest index。")
    with st.form("agent_form"):
        question = st.text_area("代码问题", placeholder="build_multi_rag_context 和 merge_evidence 如何协作？")
        a, b = st.columns(2)
        steps = a.number_input("最大工具步骤", min_value=1, max_value=30, value=6)
        budget = b.number_input("Context 预算（Token 保守上界）", min_value=500, max_value=100000, value=12000, step=500)
        thread = st.text_input("恢复线程 ID（新调查留空）")
        submitted = st.form_submit_button("开始 / 恢复调查")
    if submitted:
        try:
            with st.spinner("正在规划、调用工具并核验证据…"):
                retriever = get_repository(root, vectors)
                result = run_investigation(retriever, question, thread_id=thread or None, resume=bool(thread),
                                          max_steps=int(steps), context_budget=int(budget))
                st.session_state["agent_run"] = {"root": str(root), "result": result}
        except Exception as exc:
            st.error(f"调查失败：{exc}")
    saved = st.session_state.get("agent_run")
    if saved and saved["root"] == str(root):
        show_run(saved["result"])


def retrieval_panel(root):
    if root is None:
        st.info("请先扫描仓库")
        return
    st.subheader("代码检索")
    with st.form("search_form"):
        question = st.text_input("检索问题")
        vectors = st.checkbox("使用向量索引", key="search_vectors")
        tests = st.checkbox("包含测试代码")
        demos = st.checkbox("包含演示代码")
        strategy = st.selectbox("检索策略", ["hybrid", "exact", "bm25", "vector"])
        submitted = st.form_submit_button("搜索源码")
    if submitted:
        try:
            retriever = get_repository(root, vectors, tests, demos)
            results = retriever.search(question, strategy=strategy)
            st.session_state["rag_results"] = {"root": str(root), "results": results, "backend": retriever.vector_backend}
        except Exception as exc:
            st.error(f"检索失败：{exc}")
    saved = st.session_state.get("rag_results")
    if saved and saved["root"] == str(root):
        st.caption(f"向量后端：{saved.get('backend', 'unavailable')}")
        for rank, result in enumerate(saved["results"], 1):
            m = result["metadata"]
            with st.expander(f"Top {rank} · {m['file']}::{m['name']}", expanded=rank == 1):
                st.caption(f"L{m['start_line']}-L{m['end_line']} · {result['retrieval_type']}")
                st.code(result["content"], language="python")


def impact_panel(root):
    if root is None:
        st.info("请先扫描仓库")
        return
    from impact_analysis import analyze_impact
    st.subheader("代码修改影响分析")
    with st.form("impact_form"):
        file = st.text_input("目标文件", value="ast_parser.py")
        symbol = st.text_input("目标函数 / 方法", value="parse_file")
        depth = st.slider("追溯深度", 1, 10, 3)
        submitted = st.form_submit_button("分析影响范围")
    if submitted:
        try:
            result = analyze_impact(root, file, symbol, depth)
            st.session_state["impact_result"] = {"root": str(root), "result": result}
        except (ValueError, OSError) as exc:
            st.error(str(exc))
    saved = st.session_state.get("impact_result")
    if saved and saved["root"] == str(root):
        result = saved["result"]
        st.dataframe(result["affected"], use_container_width=True)
        st.caption(" · ".join(result["limitations"]))
        if result["truncated"]:
            st.info("已达到追溯深度，仍有上游调用者。")
        with st.expander("调用位置、模块依赖与解析错误"):
            st.json(result)
        st.download_button("下载影响分析", json.dumps(result, ensure_ascii=False, indent=2), "impact.json", "application/json")


def evaluation_panel(root):
    if root is None:
        st.info("请先扫描仓库")
        return
    from evaluation import load_cases, evaluate_retrieval, compare_agents, write_report
    st.subheader("检索与 Agent 评测")
    st.caption("用固定问题集检查代码召回、排名和耗时。默认先跑 5 题；只做检索对照不会调用回答用的 LLM。")
    with st.form("eval_form"):
        cases_path = st.text_input("JSONL 评测集路径", value=str(Path(__file__).with_name("eval_cases.jsonl")))
        count = st.number_input("本次评测题数", min_value=1, max_value=1000, value=5)
        vectors = st.checkbox("比较 BM25 / Vector / Hybrid（需要向量索引）")
        agents = st.checkbox("比较普通与 Coverage-Aware Agent（调用配置的 LLM）")
        judge = st.checkbox("额外使用 LLM judge 评估 Groundedness")
        submitted = st.form_submit_button("运行评测并生成报告")
    if submitted:
        bar = st.progress(0.0, text="正在准备源码和索引…")
        try:
            retriever = get_repository(root, vectors)
            cases = load_cases(cases_path, retriever.documents)[:int(count)]
            def progress(done, total, label):
                bar.progress(done/total, text=f"{done}/{total} · {label}")
            report = compare_agents(retriever, cases, judge=judge, progress=progress) if agents else evaluate_retrieval(
                retriever, cases, strategies=("bm25", "vector", "hybrid") if vectors else ("bm25", "hybrid"), progress=progress)
            write_report(report, root / "reports")
            st.session_state["evaluation"] = {"root": str(root), "report": report}
            bar.progress(1.0, text=f"评测完成：{len(cases)} 题")
        except Exception as exc:
            bar.empty()
            st.error(f"评测失败：{exc}")
    saved = st.session_state.get("evaluation")
    if saved and saved["root"] == str(root):
        st.dataframe(saved["report"]["summary"], use_container_width=True)
        st.download_button("下载评测报告 JSON", json.dumps(saved["report"], ensure_ascii=False, indent=2),
                           "report.json", "application/json")
