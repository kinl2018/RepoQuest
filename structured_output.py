"""Strict JSON object parsing shared by decomposition and evidence grading."""
import json


def parse_grader_json(raw: str) -> dict:
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("模型返回了空内容")
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) < 3 or lines[0].strip().lower() not in ("```json", "```") or lines[-1].strip() != "```":
            raise ValueError("JSON 代码块格式不正确")
        text = "\n".join(lines[1:-1]).strip()
    try:
        result = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError("模型没有返回合法 JSON") from exc
    if not isinstance(result, dict):
        raise ValueError("模型必须返回 JSON 对象")
    return result
