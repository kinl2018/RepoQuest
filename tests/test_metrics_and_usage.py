from types import SimpleNamespace
import pytest
import llm_runtime
from coverage_metrics import calculate_coverage_gain


def test_coverage_gain_can_be_negative():
    stats = calculate_coverage_gain({"q1":True,"q2":False}, {"q1":False,"q2":False})
    assert stats["coverage_gain"] == -0.5


def test_usage_accumulates_real_api_counts_and_cost(monkeypatch):
    response = SimpleNamespace(usage=SimpleNamespace(prompt_tokens=100, completion_tokens=20))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_:response)))
    monkeypatch.setattr(llm_runtime, "get_client", lambda:client)
    monkeypatch.setenv("LLM_MODEL", "test")
    monkeypatch.setenv("LLM_INPUT_PRICE_PER_MILLION", "2")
    monkeypatch.setenv("LLM_OUTPUT_PRICE_PER_MILLION", "4")
    token = llm_runtime.begin_usage()
    llm_runtime.complete(messages=[])
    llm_runtime.complete(messages=[])
    usage = llm_runtime.end_usage(token)
    assert usage["prompt_tokens"] == 200
    assert usage["completion_tokens"] == 40
    assert usage["cost_usd"] == pytest.approx(0.00056)


def test_missing_usage_never_claims_zero_cost(monkeypatch):
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_:SimpleNamespace(usage=None))))
    monkeypatch.setattr(llm_runtime, "get_client", lambda:client)
    monkeypatch.setenv("LLM_MODEL", "test")
    token = llm_runtime.begin_usage()
    llm_runtime.complete(messages=[])
    usage = llm_runtime.end_usage(token)
    assert not usage["usage_complete"]
    assert usage["cost_usd"] is None
