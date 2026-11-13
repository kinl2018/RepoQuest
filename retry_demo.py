
from pathlib import Path

from sentence_transformers import SentenceTransformer

from rag_index import load_index
from rag_bm25 import build_bm25
from langgraph_demo import build_graph


# 1. 初始化已有组件
root = Path(__file__).resolve().parent

documents, embeddings = load_index(
    root / ".repoquest_index"
)

model = SentenceTransformer(
    "BAAI/bge-small-zh-v1.5",
    device="cpu"
)

bm25 = build_bm25(documents)

app = build_graph(
    documents, embeddings, model, bm25
)


# 2. 准备一个跨模块问题
question = (
    "RepoQuest 如何融合 BM25 和向量检索？"
    "如何扩展函数调用者上下文？"
    "以及在多轮重试中如何保存历史证据？"
)


# 3. 实际执行 Agent
print("===== Real Agent Investigation =====")
print("问题：", question)

round_count = 0
current_retrieval_question = question
for update in app.stream(
    {
        "question": question,
        "retry_count": 0,
        "evidence_history": []
    },
    stream_mode="updates"
):
    for node, data in update.items():

        print(f"\n执行节点：{node}")

        if node == "decompose":
            print("===== Subquestions =====")

            for item in data["subquestions"]:
                print(
                    f"{item['id']}: {item['question']}"
                )

        elif node == "rewrite":
            print("改写查询：", data["rewritten"])

        elif node == "retrieve":
            round_count += 1

            print("检索轮次：", round_count)
            print("本轮结果：", len(data["results"]))
            print(
                "累计证据：",
                len(data["evidence_history"])
            )

            for item in data["results"][:3]:
                meta = item["metadata"]
                print(
                    f"  {meta['file']}::{meta['name']}"
                )

            print("当前检索目标：", current_retrieval_question)

            novelty = data["retrieval_novelty"]

            print(
                f"新增证据：{novelty['new_count']} 条，"
                f"Novelty Rate：{novelty['novelty_rate']:.1%}"
            )

        elif node == "context":
            print(
                "Context 长度：",
                len(data["context"]),
                "字符"
            )



        elif node == "grade":

            grade = data["evidence_grade"]

            print("整体证据充分：", grade["sufficient"])

            print("判断理由：", grade["reason"])

            print(

                "缺失证据类型：",

                grade.get("missing_evidence", [])

            )

            # 逐子问题评分

            print("\n===== Subquestion Grades =====")

            for qid, result in data["subquestion_grades"].items():
                print(

                    f"{qid}: "

                    f"sufficient={result['sufficient']}"

                )

                print("  原因：", result["reason"])

            # Coverage 统计

            stats = data["coverage_stats"]

            print("\n===== Coverage Tracking =====")

            print(

                "当前覆盖率：",

                f"{stats['current_coverage']:.1%}"

            )

            print(

                "Coverage Gain：",

                f"{stats['coverage_gain']:+.1%}"

            )

            print(

                "新增解决：",

                stats["newly_solved"]

            )


        elif node == "retry":
            print("新查询：", data["rewritten"])
            print("重试次数：", data["retry_count"])
            current_retrieval_question = data["retrieval_question"]

            print("新的检索目标：", current_retrieval_question)
            print("LLM 改写查询：", data["rewritten"])

        elif node in (
            "generate", "insufficient", "no_evidence"
        ):
            print("\n===== Final Answer =====")
            print(data["answer"])

        elif node == "index_metadata":
            print("调用 Index Metadata Tool")
            print("工具结果：", data["index_metadata"])
            print("工具证据：")
            print(data["metadata_evidence"])

