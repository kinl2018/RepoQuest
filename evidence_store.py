def merge_evidence(history: list[dict], new_results: list[dict]) -> list[dict]:
    merged = []
    seen = set()
    for result in history + new_results:
        meta = result["metadata"]
        # 用源码位置标识一个 CodeChunk
        key = (
            meta["file"],
            meta["name"],
            meta["kind"],
            meta["start_line"],
            meta["end_line"]
        )
        if key in seen:
            continue
        seen.add(key)
        merged.append(result)
    return merged

def calculate_novelty(history: list[dict], new_results: list[dict]) -> dict:
    # 先对历史证据去重
    unique_history = merge_evidence([], history)
    # 合并本轮检索结果
    merged = merge_evidence(unique_history, new_results)
    # 计算本轮新增证据数量
    new_count = len(merged) - len(unique_history)
    retrieved_count = len(new_results)
    novelty_rate = new_count / retrieved_count if retrieved_count > 0 else 0.0
    
    return {
        "retrieved_count": retrieved_count,
        "new_count": new_count,
        "novelty_rate": novelty_rate
    }


if __name__ == "__main__":

    def make_result(name, start, end):
        return {
            "metadata": {
                "file": "call_analysis.py",
                "name": name,
                "kind": "function",
                "start_line": start,
                "end_line": end
            },
            "content": f"def {name}(): ...",
            "score": 0.03
        }

    # 第一轮检索
    round1 = [
        make_result("extract_local_calls", 6, 21),
        make_result("collect_functions", 37, 48)
    ]

    # 第二轮检索
    round2 = [
        make_result("collect_functions", 37, 48),
        make_result("collect_call_edges", 142, 156)
    ]

    history = merge_evidence([], round1)
    history = merge_evidence(history, round2)

    print("===== Evidence Accumulation Test =====")

    print("第一轮结果数量：", len(round1))
    print("第二轮结果数量：", len(round2))
    print("合并去重后数量：", len(history))

    for item in history:
        print(item["metadata"]["name"])

    assert len(history) == 3
    assert len(round1) == 2
    assert len(round2) == 2

    assert [
        item["metadata"]["name"]
        for item in history
    ] == [
        "extract_local_calls",
        "collect_functions",
        "collect_call_edges"
    ]

    print("Evidence Accumulation 测试通过！")
