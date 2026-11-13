from agent_core import client, MODEL


SYSTEM_PROMPT = """
你是 RepoQuest 的代码检索查询优化器。

RepoQuest 是一个 Python 代码仓库分析工具，
主要通过 AST 静态分析理解源码，包括：
- Python 文件、函数和类的定义
- 模块之间的依赖关系
- 函数之间的调用关系
- 函数的调用者（Caller）和被调用者（Callee）

你的任务是将用户的自然语言问题改写为更适合检索
Python 源码的查询语句。

【重要：理解代码分析语境】

1. 默认情况下，用户询问某个函数的调用者，
   指的是静态分析：哪些其他函数调用了这个函数。

2. 用户询问函数调用关系、谁调用了某个函数、
   某个函数调用了哪些函数时，
   应优先考虑 AST、Call Graph、Caller、Callee、
   Call Sites 等静态代码分析术语。

3. 不要将静态函数调用关系误解为运行时调用栈。
   不要无故引入 inspect、traceback、sys._getframe、
   currentframe、f_back 等运行时栈追踪技术。

【查询改写要求】

1. 保留用户原始问题的核心意图。
2. 补充相关中文同义词和英文技术术语。
3. 可以补充相关 Python 技术概念。
4. 如果用户明确提供了函数名、类名或变量名，
   必须原样保留。
5. 不要凭空编造仓库中存在的具体函数名。
6. 不要引入与用户问题无关的技术概念。
7. 不要回答用户的问题。
8. 优先使用中文，适当补充英文标识符和术语。
9. 只输出一条适合代码检索的查询语句，
   不要输出解释或 Markdown。

【示例】

用户问题：
哪个函数负责寻找指定函数的调用者？

改写：
查找哪些函数调用了目标函数，获取函数调用者列表，
分析函数调用关系，静态调用分析，AST Call Graph，
Caller Callee Call Sites。

用户问题：
项目是如何解析 Python 文件并提取函数定义的？

改写：
Python 源码解析，AST 语法树，提取函数定义，
FunctionDef AsyncFunctionDef，AST 遍历，函数索引。

用户问题：
get_callers 函数在哪里定义？

改写：
get_callers 函数定义位置，源码搜索，函数声明，
Python Function Definition。
"""


def rewrite_query(question: str) -> str:
    if not question.strip():
        raise ValueError("检索问题不能为空")

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    rewritten = response.choices[0].message.content
    if not rewritten or not rewritten.strip():
        raise RuntimeError("LLM 没有返回有效的改写结果")

    return rewritten.strip()

def rewrite_retry_query(question: str, previous_query: str, reason: str) -> str:
    system_prompt = """
你是 RepoQuest 的代码检索优化助手。

此前的代码检索未能找到足够的证据。
请根据 Evidence Grader 提供的证据不足原因，
重新制定一条更有针对性的检索查询。

要求：
1. 保留用户原始问题的核心目标。
2. 重点搜索之前缺失的信息。
3. 避免简单重复上一轮查询。
4. 如果缺少函数实现，优先搜索函数定义和内部逻辑。
5. 保留用户提供的函数名、类名等标识符。
6. 不要编造仓库中存在的函数或文件。
7. 不要直接回答问题。
8. 只输出一条检索查询，不要附加解释。
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": (
                    f"原始问题：\n{question}\n\n"
                    f"上一轮查询：\n{previous_query}\n\n"
                    f"证据不足原因：\n{reason}"
                )
            }
        ],
        timeout=120.0
    )

    query = response.choices[0].message.content
    if not query or not query.strip():
        raise RuntimeError("Retry Query Rewriter 返回空结果")
    return query.strip()


if __name__ == "__main__":
    question = input("请输入原始问题：").strip()

    rewritten = rewrite_query(question)

    print("\n===== Query Rewriter =====")
    print("原始问题：", question)
    print("改写结果：", rewritten)
