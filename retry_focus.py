def get_unresolved_subquestions(subquestions: list[dict], grades: dict) -> list[dict]:
    # 检查子问题 ID 是否与评分结果一致
    expected_ids = [item["id"] for item in subquestions]
    if (
        not expected_ids
        or len(set(expected_ids)) != len(expected_ids)
        or set(expected_ids) != set(grades)
    ):
        raise ValueError("子问题与评分结果不一致")
    unresolved = []
    for item in subquestions:
        qid = item["id"]
        grade = grades[qid]
        if not grade["sufficient"]:
            unresolved.append({
                "id": qid,
                "question": item["question"],
                "reason": grade["reason"],
                "missing_evidence": grade["missing_evidence"]
            })
    return unresolved
