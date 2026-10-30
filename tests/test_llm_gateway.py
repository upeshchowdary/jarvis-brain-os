"""Provider contract tests for the unified LLM gateway (HTTP mocked with httpx.MockTransport)."""

import json

import httpx
import pytest

from core import connectivity
from core.config import settings
from core.llm import LLMUnavailable, llm, resolve_model


@pytest.fixture
def gw(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "gem-key")
    monkeypatch.setattr(settings, "GROQ_API_KEY", "groq-key")
    monkeypatch.setattr(settings, "ANTHROPIC_API_KEY", "ant-key")
    monkeypatch.setattr(settings, "DEFAULT_MODEL", "gemini/gemini-test")
    monkeypatch.setattr(settings, "LLM_FALLBACKS", ["groq/llama-test", "ollama/qwen-test"])
    monkeypatch.setattr(settings, "MAX_RETRIES", 1)
    connectivity.set_state_for_tests(True)
    seen = []

    def use(handler):
        def wrapped(request: httpx.Request):
            seen.append(request)
            return handler(request)

        llm.set_transport(httpx.MockTransport(wrapped))
        return seen

    yield use
    llm.set_transport(None)
    connectivity.set_state_for_tests(None)


def ok(content="hi", tool_calls=None, model="m"):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return httpx.Response(200, json={"model": model, "choices": [{"message": msg, "finish_reason": "stop"}],
                                     "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}})


def test_resolve_model():
    assert resolve_model("groq/llama-3.3-70b") == ("groq", "llama-3.3-70b")
    assert resolve_model("gemini-2.5-flash") == ("gemini", "gemini-2.5-flash")
    assert resolve_model("gpt-4o-mini") == ("openai", "gpt-4o-mini")
    assert resolve_model("claude-sonnet-4-5") == ("anthropic", "claude-sonnet-4-5")
    assert resolve_model("lmstudio/local-model") == ("lmstudio", "local-model")
    assert resolve_model("qwen3.5:latest") == ("ollama", "qwen3.5:latest")


async def test_primary_provider_and_usage(gw):
    seen = gw(lambda r: ok("Hello!", model="gemini-test"))
    res = await llm.chat([{"role": "user", "content": "hi"}])
    assert res.content == "Hello!" and res.provider == "gemini" and res.usage["total_tokens"] == 5
    assert str(seen[0].url) == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    assert seen[0].headers["authorization"] == "Bearer gem-key"


async def test_tool_calls_are_parsed(gw):
    tc = [{"id": "c1", "type": "function", "function": {"name": "set_volume", "arguments": json.dumps({"action": "up"})}}]
    gw(lambda r: ok("", tool_calls=tc))
    tools = [{"type": "function", "function": {"name": "set_volume", "parameters": {"type": "object"}}}]
    res = await llm.chat([{"role": "user", "content": "louder"}], tools=tools)
    assert res.tool_calls[0].name == "set_volume" and res.tool_calls[0].arguments == {"action": "up"}
    assert res.assistant_message()["tool_calls"][0]["function"]["name"] == "set_volume"


async def test_retry_on_429_then_success(gw):
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        return httpx.Response(429, text="slow down") if state["n"] == 1 else ok("after retry")

    gw(handler)
    res = await llm.chat([{"role": "user", "content": "x"}])
    assert res.content == "after retry" and state["n"] == 2 and res.provider == "gemini"


async def test_fallback_chain_on_errors(gw):
    def handler(request):
        if "googleapis" in str(request.url):
            return httpx.Response(500, text="boom")
        return ok("from groq")

    gw(handler)
    res = await llm.chat([{"role": "user", "content": "x"}])
    assert res.provider == "groq" and res.fallback_used is True


async def test_all_fail_raises(gw):
    gw(lambda r: httpx.Response(401, text="bad key"))
    with pytest.raises(LLMUnavailable):
        await llm.chat([{"role": "user", "content": "x"}])


async def test_local_only_uses_local_provider(gw):
    seen = gw(lambda r: ok("local answer"))
    res = await llm.chat([{"role": "user", "content": "secret"}], local_only=True)
    assert res.provider == "ollama" and "127.0.0.1:11434/v1/chat/completions" in str(seen[0].url)
    assert "authorization" not in seen[0].headers


async def test_anthropic_headers(gw, monkeypatch):
    seen = gw(lambda r: ok("claude"))
    await llm.chat([{"role": "user", "content": "x"}], model="anthropic/claude-test")
    assert seen[0].headers["x-api-key"] == "ant-key" and "anthropic-version" in seen[0].headers


async def test_streaming_text_and_tool_calls(gw):
    chunks = [
        {"choices": [{"delta": {"content": "Hel"}}]},
        {"choices": [{"delta": {"content": "lo"}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c9", "function": {"name": "get_time_date", "arguments": "{"}}]}}]},
        {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "}"}}]}, "finish_reason": "tool_calls"}]},
    ]
    body = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
    gw(lambda r: httpx.Response(200, text=body, headers={"content-type": "text/event-stream"}))
    events = [e async for e in llm.stream_events([{"role": "user", "content": "x"}])]
    texts = [v for k, v in events if k == "text"]
    final = events[-1][1]
    assert "".join(texts) == "Hello" and final.tool_calls[0].name == "get_time_date" and final.tool_calls[0].arguments == {}


async def test_offline_prefers_local(gw):
    connectivity.set_state_for_tests(False)
    seen = gw(lambda r: ok("offline"))
    res = await llm.chat([{"role": "user", "content": "x"}])
    assert res.provider == "ollama" and "11434" in str(seen[0].url)
