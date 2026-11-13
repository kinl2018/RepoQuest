"""Small compatibility facade; all investigation uses the same LangGraph agent."""
from repo_service import open_repository, run_investigation


def ask_repo(question, root, history=None, max_rounds=6):
    # Earlier conversation is advisory, never a replacement for source evidence.
    if history:
        previous = "\n".join(m["content"] for m in history[-4:] if m["role"] == "user")
        question = f"先前用户问题（用于消解指代）：\n{previous}\n当前问题：{question}"
    result = run_investigation(open_repository(root), question, max_steps=max_rounds)
    return result["answer"], result["trace"]
