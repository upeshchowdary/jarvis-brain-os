"""Provider contract tests for the unified LLM gateway (HTTP mocked with httpx.MockTransport)."""

import json

import httpx
import pytest

from core import connectivity
from core.config import settings
from core.llm import LLMUnavailable, llm, resolve_model
from core.llm import gateway as gateway_module


@pytest.fixture
def gw(monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "gem-key")
    monkeypatch.setattr(settings, "GEMINI_API_KEYS", [])
    monkeypatch.setattr(gateway_module, "_KEY_CURSOR", {})
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


async def test_gemini_thought_signature_is_sent_back(gw):
    """Gemini 3 rejects a follow-up turn unless each tool call's thought_signature is echoed back."""
    extra = {"google": {"thought_signature": "sig-123"}}
    tc = [{"id": "c1", "type": "function", "extra_content": extra, "function": {"name": "get_time_date", "arguments": "{}"}}]
    gw(lambda r: ok("", tool_calls=tc))
    res = await llm.chat([{"role": "user", "content": "time?"}], tools=[{"type": "function", "function": {"name": "get_time_date"}}])
    assert res.assistant_message()["tool_calls"][0]["extra_content"] == extra


async def test_streamed_thought_signature_is_sent_back(gw):
    extra = {"google": {"thought_signature": "sig-456"}}
    chunk = {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c2", "extra_content": extra,
                                                    "function": {"name": "get_time_date", "arguments": "{}"}}]}}]}
    body = f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n"
    gw(lambda r: httpx.Response(200, text=body, headers={"content-type": "text/event-stream"}))
    events = [e async for e in llm.stream_events([{"role": "user", "content": "x"}])]
    assert events[-1][1].assistant_message()["tool_calls"][0]["extra_content"] == extra


async def test_unsigned_tool_calls_get_placeholder_signature_for_gemini_only(gw):
    """Tool calls JARVIS injected itself (fast path) or got from another provider carry no signature."""
    signed = {"google": {"thought_signature": "real"}}
    history = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "fast_0", "type": "function", "function": {"name": "screen_look", "arguments": "{}"}},
            {"id": "c1", "type": "function", "extra_content": signed, "function": {"name": "web_search", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "fast_0", "content": "screen"},
        {"role": "tool", "tool_call_id": "c1", "content": "results"},
    ]
    seen = gw(lambda r: ok("answer"))
    await llm.chat(history)
    sent = json.loads(seen[0].content)["messages"][1]["tool_calls"]
    assert sent[0]["extra_content"]["google"]["thought_signature"] == "skip_thought_signature_validator"
    assert sent[1]["extra_content"] == signed
    assert "extra_content" not in history[1]["tool_calls"][0]  # caller's messages are not mutated
    seen.clear()
    await llm.chat(history, model="groq/llama-test")
    assert "extra_content" not in json.loads(seen[0].content)["messages"][1]["tool_calls"][0]


async def test_gemini_rotates_to_next_key_on_quota(gw, monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEYS", ["gem-key", "gem-2", "gem-3"])

    def handler(request):
        if request.headers["authorization"] == "Bearer gem-3":
            return ok("third key works")
        return httpx.Response(429, text="quota exceeded")

    seen = gw(handler)
    res = await llm.chat([{"role": "user", "content": "x"}])
    assert res.content == "third key works" and res.provider == "gemini" and res.fallback_used is False
    assert [r.headers["authorization"] for r in seen] == ["Bearer gem-key", "Bearer gem-2", "Bearer gem-3"]
    # The working key sticks for the next call.
    seen.clear()
    await llm.chat([{"role": "user", "content": "y"}])
    assert seen[0].headers["authorization"] == "Bearer gem-3"


async def test_all_gemini_keys_exhausted_falls_back(gw, monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEYS", ["gem-2"])

    def handler(request):
        if "googleapis" in str(request.url):
            return httpx.Response(429, text="quota exceeded")
        return ok("from groq")

    gw(handler)
    res = await llm.chat([{"role": "user", "content": "x"}])
    assert res.provider == "groq" and res.fallback_used is True


async def test_streaming_rotates_key(gw, monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEYS", ["gem-2"])
    body = f"data: {json.dumps({'choices': [{'delta': {'content': 'ok'}}]})}\n\ndata: [DONE]\n\n"

    def handler(request):
        if request.headers["authorization"] == "Bearer gem-key":
            return httpx.Response(403, text="key rejected")
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    seen = gw(handler)
    events = [e async for e in llm.stream_events([{"role": "user", "content": "x"}])]
    assert events[-1][1].content == "ok" and events[-1][1].provider == "gemini" and len(seen) == 2


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


async def test_reasoning_effort_sent_per_model(gw, monkeypatch):
    monkeypatch.setattr(settings, "REASONING_EFFORT", {"gemini-test": "minimal", "llama-test": "low"})
    seen = gw(lambda r: ok("x"))
    await llm.chat([{"role": "user", "content": "x"}])
    assert json.loads(seen[0].content)["reasoning_effort"] == "minimal"
    seen.clear()
    await llm.chat([{"role": "user", "content": "x"}], model="ollama/qwen-test")
    assert "reasoning_effort" not in json.loads(seen[0].content)


async def test_reasoning_effort_never_sent_to_local_provider(gw, monkeypatch):
    monkeypatch.setattr(settings, "REASONING_EFFORT", {"qwen-test": "low"})
    seen = gw(lambda r: ok("x"))
    await llm.chat([{"role": "user", "content": "x"}], model="lmstudio/qwen-test")
    assert "reasoning_effort" not in json.loads(seen[0].content)


def test_reasoning_effort_parses_json_from_env(monkeypatch):
    from core.config import Settings

    monkeypatch.setenv("REASONING_EFFORT", '{"m1": "low"}')
    assert Settings(_env_file=None).REASONING_EFFORT == {"m1": "low"}


async def test_response_format_passthrough(gw):
    seen = gw(lambda r: ok("{}"))
    fmt = {"type": "json_schema", "json_schema": {"name": "intent", "schema": {"type": "object"}}}
    await llm.chat([{"role": "user", "content": "x"}], response_format=fmt)
    assert json.loads(seen[0].content)["response_format"] == fmt


async def test_fallbacks_false_calls_only_the_requested_model(gw):
    seen = gw(lambda r: httpx.Response(400, text="bad request"))
    with pytest.raises(LLMUnavailable):
        await llm.chat([{"role": "user", "content": "x"}], model="groq/llama-test", fallbacks=False)
    assert len(seen) == 1 and str(seen[0].url).startswith("https://api.groq.com")
    assert await llm.candidates("groq/llama-test", fallbacks=False) == [("groq", "llama-test")]
    assert len(await llm.candidates("gemini/gemini-test")) == 3  # default: the full fallback chain


async def test_fallbacks_false_with_unconfigured_provider_makes_no_request(gw):
    seen = gw(lambda r: ok("x"))
    with pytest.raises(LLMUnavailable):
        await llm.chat([{"role": "user", "content": "x"}], model="openai/gpt-x", fallbacks=False)
    assert seen == []


async def test_retries_zero_fails_fast_without_sleeping(gw, monkeypatch):
    slept = []

    async def fake_sleep(delay):
        slept.append(delay)

    monkeypatch.setattr(gateway_module.asyncio, "sleep", fake_sleep)
    seen = gw(lambda r: httpx.Response(429, text="slow down", headers={"retry-after": "5"}))
    with pytest.raises(LLMUnavailable):
        await llm.chat([{"role": "user", "content": "x"}], model="groq/llama-test", fallbacks=False, retries=0)
    assert len(seen) == 1 and slept == []


async def test_retries_default_unchanged(gw, monkeypatch):
    slept = []

    async def fake_sleep(delay):
        slept.append(delay)

    monkeypatch.setattr(gateway_module.asyncio, "sleep", fake_sleep)
    seen = gw(lambda r: httpx.Response(429, text="slow down"))
    with pytest.raises(LLMUnavailable):
        await llm.chat([{"role": "user", "content": "x"}], model="groq/llama-test", fallbacks=False)
    assert len(seen) == settings.MAX_RETRIES + 1 and len(slept) == settings.MAX_RETRIES


async def test_warm_up_hits_each_configured_cloud_provider_once(gw, monkeypatch):
    monkeypatch.setattr(settings, "INTENT_MODEL", "groq/llama-test")
    monkeypatch.setattr(settings, "ACTION_MODEL", "gemini/gemini-test")
    seen = gw(lambda r: ok("x"))
    await llm.warm_up()
    hosts = sorted({r.url.host for r in seen})
    assert hosts == ["api.groq.com", "generativelanguage.googleapis.com"]
    assert len(seen) == 2
    # I2: a models listing (no completion quota), never a chat completion.
    urls = sorted(str(r.url) for r in seen)
    assert urls == ["https://api.groq.com/openai/v1/models",
                    "https://generativelanguage.googleapis.com/v1beta/openai/models"]
    assert all(r.method == "GET" and not r.content for r in seen)
    assert all(r.headers.get("authorization", "").startswith("Bearer ") for r in seen)


async def test_warm_up_skips_empty_local_and_unconfigured(gw, monkeypatch):
    monkeypatch.setattr(settings, "INTENT_MODEL", "")
    monkeypatch.setattr(settings, "ACTION_MODEL", "ollama/qwen-test")
    monkeypatch.setattr(settings, "GROQ_API_KEY", None)
    monkeypatch.setattr(settings, "DEFAULT_MODEL", "groq/llama-test")
    seen = gw(lambda r: ok("x"))
    await llm.warm_up()
    assert seen == []


async def test_warm_up_swallows_errors(gw):
    gw(lambda r: httpx.Response(500, text="down"))
    await llm.warm_up()  # must not raise


async def test_keepwarm_is_noop_when_disabled(gw, monkeypatch):
    monkeypatch.setattr(settings, "LLM_KEEPWARM", False)
    llm.start_keepwarm(0.01)
    assert llm._keepwarm_task is None


async def test_keepwarm_warms_when_idle_and_stops(gw, monkeypatch):
    import asyncio
    import time

    monkeypatch.setattr(settings, "LLM_KEEPWARM", True)
    seen = gw(lambda r: ok("x"))
    monkeypatch.setattr(llm, "_last_request", time.monotonic())  # a real request just happened
    llm.start_keepwarm(0.02)
    await asyncio.sleep(0.15)
    llm.stop_keepwarm()
    assert llm._keepwarm_task is None
    assert seen  # idle -> warmed at least once


async def test_warm_up_is_single_shot_and_bounded(gw, monkeypatch):
    import asyncio
    import time

    monkeypatch.setattr(settings, "MAX_RETRIES", 5)
    monkeypatch.setattr(settings, "INTENT_MODEL", "")
    monkeypatch.setattr(settings, "ACTION_MODEL", "")
    monkeypatch.setattr(gateway_module, "_WARM_TIMEOUT", 0.2)
    seen = gw(lambda r: httpx.Response(500, text="down"))
    await llm.warm_up()
    assert len(seen) == 1  # no retries on failure

    async def slow(request):
        await asyncio.sleep(5)
        return ok("x")

    llm.set_transport(httpx.MockTransport(slow))
    started = time.perf_counter()
    await llm.warm_up()
    assert time.perf_counter() - started < 1.5


async def test_warm_up_survives_bad_model_spec(gw, monkeypatch):
    monkeypatch.setattr(settings, "INTENT_MODEL", None)
    gw(lambda r: ok("x"))
    await llm.warm_up()


async def test_keepwarm_only_within_window_after_a_real_request(gw, monkeypatch):
    """I2: no pings before the first real request or > KEEPWARM_WINDOW_SECONDS after the last one."""
    import time

    monkeypatch.setattr(settings, "PRIVACY_MODE", "cloud")
    monkeypatch.setattr(llm, "_last_request", None)
    assert not llm._keepwarm_due(50.0)
    now = time.monotonic()
    monkeypatch.setattr(llm, "_last_request", now - 10)
    assert not llm._keepwarm_due(50.0)  # busy: a real request was just made
    monkeypatch.setattr(llm, "_last_request", now - 60)
    assert llm._keepwarm_due(50.0)
    monkeypatch.setattr(llm, "_last_request", now - gateway_module.KEEPWARM_WINDOW_SECONDS - 5)
    assert not llm._keepwarm_due(50.0)  # idle for a long time: let the connection go
    assert gateway_module.KEEPWARM_WINDOW_SECONDS <= 15 * 60


async def test_keepwarm_skips_offline_and_local_privacy(gw, monkeypatch):
    import time

    monkeypatch.setattr(llm, "_last_request", time.monotonic() - 60)
    monkeypatch.setattr(settings, "PRIVACY_MODE", "cloud")
    connectivity.set_state_for_tests(False)
    assert not llm._keepwarm_due(50.0)
    connectivity.set_state_for_tests(True)
    monkeypatch.setattr(settings, "PRIVACY_MODE", "local")
    assert not llm._keepwarm_due(50.0)


async def test_keepwarm_loop_does_not_ping_without_a_recent_request(gw, monkeypatch):
    import asyncio

    monkeypatch.setattr(settings, "LLM_KEEPWARM", True)
    monkeypatch.setattr(llm, "_last_request", None)
    seen = gw(lambda r: ok("x"))
    llm.start_keepwarm(0.02)
    await asyncio.sleep(0.12)
    llm.stop_keepwarm()
    assert seen == []
