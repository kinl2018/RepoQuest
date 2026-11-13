from grounded_qa import answer_with_context
from pathlib import Path
from typing import TypedDict, NotRequired
from context_expansion import build_multi_rag_context
from langgraph.graph import StateGraph, START, END
from sentence_transformers import SentenceTransformer
from evidence_grader import grade_evidence
from query_rewriter import rewrite_query, rewrite_retry_query
from rag_index import load_index, get_index_metadata
from rag_bm25 import build_bm25
from rag_pipeline import retrieve_candidates
from evidence_store import merge_evidence, calculate_novelty
from context_expansion import select_context_results, prioritize_explicit_symbol_hits
from question_decomposer import decompose_question
from subquestion_grader import grade_subquestions, aggregate_subquestion_grades
from coverage_metrics import calculate_coverage_gain
from retry_focus import get_unresolved_subquestions

# 1. 定义共享 State
class RepoState(TypedDict):
    question: str
    rewritten: NotRequired[str]
    results: NotRequired[list[dict]]
    context: NotRequired[str]
    answer: NotRequired[str]
    evidence_grade: NotRequired[dict]
    retry_count: NotRequired[int]
    evidence_history: NotRequired[list[dict]]
    metadata_attempted: NotRequired[bool]
    index_metadata: NotRequired[dict]
    metadata_evidence: NotRequired[str]
    retrieval_novelty: NotRequired[dict]
    subquestions: NotRequired[list[dict]]
    subquestion_grades: NotRequired[dict]
    coverage_status: NotRequired[dict[str, bool]]
    coverage_stats: NotRequired[dict]
    retrieval_question: NotRequired[str]

# 2. Query Rewrite Node
def rewrite_node(state: RepoState) -> dict:
    rewritten = rewrite_query(state["question"])
    return {"rewritten": rewritten}

def route_after_retrieve(state: RepoState) -> str:
    if state.get("results"):
        return "context"
    return "no_evidence"

def route_after_grade(state: RepoState) -> str:
    grade = state.get("evidence_grade", {})
    if grade.get("sufficient") is True:
        return "generate"
    return "insufficient"

def route_after_grade_with_retry(state: RepoState, max_retries=2) -> str:
    grade = state.get("evidence_grade", {})
    # 1. 证据充分，直接生成答案
    if grade.get("sufficient") is True:
        return "generate"
    # 2. 证据不足，检查还有没有重试机会
    retry_count = state.get("retry_count", 0)
    if retry_count < max_retries:
        return "retry"
    return "insufficient"


def retry_node(state: RepoState) -> dict:
    # 1. 提取尚未解决的子问题
    unresolved = get_unresolved_subquestions(subquestions=state["subquestions"], grades=state["subquestion_grades"])
    if not unresolved:
        raise ValueError("Retry Node 没有找到未解决的子问题")
    # 2. 只保留未解决的问题
    focused_question = "\n".join(
        f"{item['id']}: {item['question']}"
        for item in unresolved
    )
    # 3. 收集这些问题缺少的证据
    focused_reason = "\n".join(
        f"{item['id']}: {item['reason']}"
        for item in unresolved
    )
    # 4. 针对未解决问题重新生成查询
    new_query = rewrite_retry_query(
        question=focused_question,
        previous_query=state["rewritten"],
        reason=focused_reason
    )
    # 5. 更新查询和重试次数
    return {
        "retrieval_question": focused_question,
        "rewritten": new_query,
        "retry_count": state.get("retry_count", 0) + 1
    }


def route_after_grade_with_tools(state: RepoState, max_retries=2) -> str:
    grade = state.get("evidence_grade", {})
    # 1. 证据充分，进入生成阶段
    if grade.get("sufficient") is True:
        return "generate"
    missing = grade.get("missing_evidence", [])
    # 2. 缺少索引元数据，且尚未调用工具
    if "index_metadata" in missing and not state.get("metadata_attempted", False):
        return "index_metadata"
    # 3. 其他情况，尝试重新检索
    retry_count = state.get("retry_count", 0)
    if retry_count < max_retries:
        return "retry"
    # 4. 没有可用的调查手段
    return "insufficient"

# 3. 创建 LangGraph
def build_graph(documents, embeddings, model, bm25):

    # Retrieve Node
    def retrieve_node(state: RepoState) -> dict:
        _, results = retrieve_candidates(
            question=state.get("retrieval_question", state["question"]),
            documents=documents,
            embeddings=embeddings,
            model=model,
            bm25=bm25,
            k=5,
            rewritten_question=state["rewritten"]
        )

        # 1. 获取此前积累的证据
        history = state.get("evidence_history", [])
        # 2. 统计本轮新证据
        novelty = calculate_novelty(history=history, new_results=results)
        # 3. 合并历史证据
        updated_history = merge_evidence(history=history, new_results=results)
        # 4. 更新 State
        return {
            "results": results,
            "evidence_history": updated_history,
            "retrieval_novelty": novelty
        }

    # Context Expansion Node
    def context_node(state: RepoState) -> dict:
        # 1. 获取本轮结果和历史证据
        current_results = state["results"]
        history = state.get("evidence_history", [])
        # 1. 获取上一轮未解决的子问题
        unresolved = []
        if state.get("subquestion_grades"):
            unresolved = get_unresolved_subquestions(subquestions=state["subquestions"], grades=state["subquestion_grades"])
        # 2. 如果存在未解决问题，优先选择明确需要的函数
        if unresolved:
            current_results = prioritize_explicit_symbol_hits(results=current_results, unresolved=unresolved)
        # 3. 执行Context Selection
        selected = select_context_results(
            current_results=current_results,
            evidence_history=history,
            latest_k=2,
            history_k=2
        )
        # 3. 构建包含 AST 关系的 Context
        context = build_multi_rag_context(
            root=Path(__file__).resolve().parent,
            results=selected,
            documents=documents,
            max_results=len(selected),
            max_callers=2
        )
        # 保留此前工具获得的元数据证据
        metadata_evidence = state.get("metadata_evidence", "")
        if metadata_evidence:
            context = f"{context}\n\n{metadata_evidence}".strip()

        return {"context": context}

    # Generate Answer Node
    def generate_node(state: RepoState) -> dict:
        question = state["question"]
        context = state["context"]
        answer = answer_with_context(question=question, context=context)
        return {"answer": answer}

    def no_evidence_node(state: RepoState) -> dict:
        return {
            "answer": (
                "未检索到可用的源码证据，"
                "目前无法回答该问题。"
            )
        }

    # Evidence Grader Node


    def grade_node(state: RepoState) -> dict:
        question = state["question"]
        context = state["context"]

        # 1. 使用 LLM 逐子问题评分
        subquestion_grades = grade_subquestions(
            question=question,
            subquestions=state["subquestions"],
            context=context
        )

        # 2. 使用 Python 汇总整体评分
        grade = aggregate_subquestion_grades(subquestion_grades)
        # 3. 获取当前子问题解决状态
        current_status = {
            qid: result["sufficient"]
            for qid, result in subquestion_grades.items()
        }
        # 4. 获取上一轮解决状态
        previous_status = state.get("coverage_status")
        if previous_status is None:
            previous_status = {qid: False for qid in current_status}
        # 5. 计算 Coverage Gain
        coverage_stats = calculate_coverage_gain(previous=previous_status, current=current_status)
        # 6. 保存所有评分结果
        return {
            "evidence_grade": grade,
            "subquestion_grades": subquestion_grades,
            "coverage_status": current_status,
            "coverage_stats": coverage_stats
        }

    def insufficient_node(state: RepoState) -> dict:
        reason = state["evidence_grade"]["reason"]
        return {
            "answer": (
                "当前源码证据不足，暂时无法可靠回答。\n"
                f"原因：{reason}"
            )
        }

    def index_metadata_node(state: RepoState) -> dict:
        index_dir = (Path(__file__).resolve().parent / ".repoquest_index")
        # 1. 执行工具
        metadata = get_index_metadata(index_dir)
        # 2. 构造证据
        evidence = (
            "[索引元数据证据]\n"
            "来源：.repoquest_index/embeddings.npy\n"
            f"Embedding 数量：{metadata['embedding_count']}\n"
            f"Embedding 维度：{metadata['embedding_dimension']}"
        )
        # 3. 把工具证据加入当前 Context
        context = state.get("context", "")
        updated_context = (f"{context}\n\n{evidence}").strip()
        return {
            "index_metadata": metadata,
            "metadata_evidence": evidence,
            "metadata_attempted": True,
            "context": updated_context
        }

    def decompose_node(state: RepoState) -> dict:
        question = state["question"]
        subquestions = decompose_question(question)
        return {"subquestions": subquestions}


    builder = StateGraph(RepoState)

    builder.add_node("rewrite", rewrite_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("context", context_node)
    builder.add_node("generate", generate_node)
    builder.add_node("no_evidence", no_evidence_node)
    builder.add_edge("rewrite", "retrieve")
    builder.add_node("grade", grade_node)
    builder.add_node("insufficient", insufficient_node)
    builder.add_node("retry", retry_node)
    builder.add_node("index_metadata", index_metadata_node)
    builder.add_node("decompose", decompose_node)

    builder.add_edge(START, "decompose")
    builder.add_edge("decompose", "rewrite")
    builder.add_conditional_edges(
        "retrieve",
        route_after_retrieve,
        {
            "context": "context",
            "no_evidence": "no_evidence"
        }
    )


    builder.add_edge("context", "grade")


    builder.add_conditional_edges(
        "grade",
        route_after_grade_with_tools,
        {
            "generate": "generate",
            "index_metadata": "index_metadata",
            "retry": "retry",
            "insufficient": "insufficient"
        }
    )

    # 工具执行后重新评估证据
    builder.add_edge("index_metadata", "grade")

    builder.add_edge("retry", "retrieve")
    builder.add_edge("insufficient", END)

    builder.add_edge("generate", END)
    builder.add_edge("no_evidence", END)

    return builder.compile()



if __name__ == "__main__":

    root = Path(__file__).resolve().parent
    index_dir = root / ".repoquest_index"

    # 加载代码索引
    documents, embeddings = load_index(index_dir)

    # 加载 Embedding 模型
    model = SentenceTransformer(
        "BAAI/bge-small-zh-v1.5",
        device="cpu"
    )

    # 创建 BM25
    bm25 = build_bm25(documents)

    # 编译工作流
    app = build_graph(
        documents, embeddings, model, bm25
    )

    print("\n===== Tool Routing Test =====")

    cases = [
        # 证据充分，直接回答
        (True, [], False, 0, "generate"),

        # 缺少索引元数据，优先调用工具
        (False, ["index_metadata"], False, 0, "index_metadata"),

        # 工具已调用，不应再次调用
        (False, ["index_metadata"], True, 0, "retry"),

        # 缺少源码，重新检索
        (False, ["source_code"], False, 0, "retry"),

        # 达到最大重试次数
        (False, ["source_code"], False, 2, "insufficient"),

        # 即使检索重试达到上限，仍允许调用尚未使用的工具
        (False, ["index_metadata"], False, 2, "index_metadata"),
    ]

    for sufficient, missing, attempted, count, expected in cases:
        state = {
            "evidence_grade": {
                "sufficient": sufficient,
                "missing_evidence": missing
            },
            "metadata_attempted": attempted,
            "retry_count": count
        }

        actual = route_after_grade_with_tools(state)

        print(
            f"missing={missing}, "
            f"attempted={attempted}, "
            f"retry={count} → {actual}"
        )

        assert actual == expected

    print("Tool Routing 测试通过！")
