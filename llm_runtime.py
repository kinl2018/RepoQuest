"""Lazy configuration and per-request usage. Imports never need an API key."""
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
import os

_usage = ContextVar("repoquest_usage", default=None)


@lru_cache(maxsize=1)
def get_client():
    from dotenv import load_dotenv
    from openai import OpenAI
    load_dotenv(Path(__file__).with_name(".env"))
    key = os.getenv("LLM_API_KEY")
    if not key or not os.getenv("LLM_MODEL"):
        raise ValueError("请在 .env 配置 LLM_API_KEY 和 LLM_MODEL")
    return OpenAI(api_key=key, base_url=os.getenv("LLM_BASE_URL") or None, timeout=60, max_retries=1)


def complete(**kwargs):
    client = get_client()
    response = client.chat.completions.create(model=os.environ["LLM_MODEL"], **kwargs)
    usage = _usage.get()
    if usage is not None:
        usage["calls"] += 1
        data = response.usage
        if data is None:
            usage["usage_complete"] = False
        else:
            usage["prompt_tokens"] += data.prompt_tokens
            usage["completion_tokens"] += data.completion_tokens
    return response


def begin_usage():
    return _usage.set({"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "usage_complete": True})


def end_usage(token):
    usage = dict(_usage.get())
    _usage.reset(token)
    prices = [os.getenv("LLM_INPUT_PRICE_PER_MILLION"), os.getenv("LLM_OUTPUT_PRICE_PER_MILLION")]
    usage["cost_usd"] = None
    if all(prices) and usage["usage_complete"]:
        usage["cost_usd"] = (usage["prompt_tokens"] * float(prices[0]) +
                             usage["completion_tokens"] * float(prices[1])) / 1_000_000
    return usage
