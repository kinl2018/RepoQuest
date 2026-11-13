
from unittest.mock import patch
import langgraph_demo as lg


# 1. 创建两条模拟源码证据
def make_result(file, marker):
    return {
        "metadata": {
            "file": file,
            "name": "<module>",
            "kind": "module",
            "start_line": 1,
            "end_line": 1
        },
        "content": f"# {marker}"
    }


a = make_result("a.py", "A_MARKER")
b = make_result("b.py", "B_MARKER")


# 2. 模拟两轮检索
fake_searches = [
    ("搜索 A", [a]),
    ("搜索 B", [a, b])
]

# 3. 模拟证据充分性判断
def fake_grade(question, context):
    has_a = "A_MARKER" in context
    has_b = "B_MARKER" in context

    return {
        "sufficient": has_a and has_b,
        "reason": "两条证据均已找到" if has_a and has_b
                  else "缺少部分证据"
    }


# 4. 创建 Graph
# 检索器会被 Mock，因此不需要加载模型和真实索引
app = lg.build_graph(
    documents=[],
    embeddings=None,
    model=None,
    bm25=None
)


# 5. 运行模拟实验
print("===== Cross-Round Evidence Test =====")

contexts = []
steps = []
novelty_rates = []
with (
    patch.object(lg, "rewrite_query", return_value="搜索 A"),
    patch.object(
        lg, "retrieve_candidates",
        side_effect=fake_searches
    ),
    patch.object(
        lg, "rewrite_retry_query",
        return_value="搜索 B"
    ),
    patch.object(
        lg, "grade_evidence",
        side_effect=fake_grade
    ),
    patch.object(
        lg, "answer_with_context",
        return_value="测试回答：A 和 B 已找到"
    ) as mock_generate
):

    for update in app.stream(
        {
            "question": "A 和 B 的代码分别是什么？",
            "retry_count": 0
        },
        stream_mode="updates"
    ):
        for node, data in update.items():
            steps.append(node)

            if node == "retrieve":
                history = data["evidence_history"]
                print(
                    "累计证据：",
                    [x["metadata"]["file"] for x in history]
                )

                stats = data["retrieval_novelty"]

                novelty_rates.append(
                    stats["novelty_rate"]
                )

                print(
                    f"本轮检索：{stats['retrieved_count']} 条，"
                    f"新增证据：{stats['new_count']} 条，"
                    f"Novelty Rate：{stats['novelty_rate']:.1%}"
                )


            if node == "context":
                contexts.append(data["context"])

            if node == "grade":
                print(
                    "证据充分：",
                    data["evidence_grade"]["sufficient"]
                )

            if node == "generate":
                print("最终回答：", data["answer"])

    # 6. 验证跨轮证据确实被使用
    assert len(contexts) == 2

    # 第一轮只有 A
    assert "A_MARKER" in contexts[0]
    assert "B_MARKER" not in contexts[0]

    # 第二轮同时包含 A 和 B
    assert "A_MARKER" in contexts[1]
    assert "B_MARKER" in contexts[1]

    assert steps.count("retry") == 1
    assert steps[-1] == "generate"
    mock_generate.assert_called_once()

assert novelty_rates == [1.0, 0.5]
print("LangGraph Retrieval Novelty 测试通过！")
