
def calculate_coverage_gain(previous: dict[str, bool], current: dict[str, bool]) -> dict:
    # 每轮必须评估同一组子问题
    if not previous or previous.keys() != current.keys():
        raise ValueError("前后两轮的子问题必须一致且非空")
    # 每个子问题只能标记 True / False
    if any(type(v) is not bool for v in list(previous.values()) + list(current.values())):
        raise ValueError("子问题状态必须是布尔值")
    total = len(previous)
    previous_solved = sum(previous.values())
    current_solved = sum(current.values())
    previous_coverage = previous_solved / total
    current_coverage = current_solved / total
    # 本轮新解决了哪些问题？
    newly_solved = [name for name in previous if not previous[name] and current[name]]
    return {
        "total": total,
        "previous_coverage": previous_coverage,
        "current_coverage": current_coverage,
        "coverage_gain": current_coverage - previous_coverage,
        "newly_solved": newly_solved
    }
