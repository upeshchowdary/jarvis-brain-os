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
    monkeypatch.setattr(settings, "OLLAMA_BASE_URL", "http://localhost:11434")  # HTTP is mocked here
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


# ── I2: the raised error carries the HTTP status / headers (the intent step's cooldown reads them) ──

async def test_unavailable_exposes_status_and_headers(gw):
    gw(lambda r: httpx.Response(429, text="rate limited",
                                headers={"retry-after": "7", "x-ratelimit-reset-tokens": "1m2s"}))
    with pytest.raises(LLMUnavailable) as info:
        await llm.chat([{"role": "user", "content": "x"}], model="groq/llama-test", fallbacks=False, retries=0)
    exc = info.value
    assert exc.status == 429 and not exc.timed_out
    assert exc.headers["retry-after"] == "7" and exc.headers["x-ratelimit-reset-tokens"] == "1m2s"


async def test_unavailable_exposes_non_retryable_status(gw):
    gw(lambda r: httpx.Response(400, text="bad request"))
    with pytest.raises(LLMUnavailable) as info:
        await llm.chat([{"role": "user", "content": "x"}], model="groq/llama-test", fallbacks=False, retries=0)
    assert info.value.status == 400


async def test_unavailable_marks_timeouts(gw):
    def handler(request):
        raise httpx.ReadTimeout("slow", request=request)

    gw(handler)
    with pytest.raises(LLMUnavailable) as info:
        await llm.chat([{"role": "user", "content": "x"}], model="groq/llama-test", fallbacks=False, retries=0)
    assert info.value.timed_out and info.value.status is None


async def test_extra_content_is_stripped_for_non_gemini_providers(gw):
    """M1: Gemini thought signatures are Gemini-only; other providers get plain tool calls (and so do streams)."""
    signed = {"google": {"thought_signature": "real"}}
    history = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "c1", "type": "function", "extra_content": signed, "function": {"name": "web_search", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "c1", "content": "results"},
    ]
    seen = gw(lambda r: ok("answer"))
    await llm.chat(history, model="groq/llama-test")
    assert "extra_content" not in json.loads(seen[0].content)["messages"][1]["tool_calls"][0]
    assert history[1]["tool_calls"][0]["extra_content"] == signed  # caller's messages are not mutated
    seen.clear()
    await llm.chat(history, model="gemini/gemini-test")
    assert json.loads(seen[0].content)["messages"][1]["tool_calls"][0]["extra_content"] == signed
    body = "data: " + json.dumps({"choices": [{"delta": {"content": "ok"}}]}) + "\n\ndata: [DONE]\n\n"
    seen.clear()
    gw(lambda r: httpx.Response(200, text=body, headers={"content-type": "text/event-stream"}))
    _ = [e async for e in llm.stream_events(history, model="groq/llama-test")]
    assert "extra_content" not in json.loads(seen[0].content)["messages"][1]["tool_calls"][0]


# ── G2: Groq key rotation (keys from separate Groq organisations add capacity) ──

def test_groq_keys_primary_first_and_deduplicated(monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEY", "groq-key")
    monkeypatch.setattr(settings, "GROQ_API_KEYS", ["groq-2", "groq-key", "groq-3"])
    assert settings.api_keys_for("groq") == ["groq-key", "groq-2", "groq-3"]
    from core.config import Settings

    assert Settings(GROQ_API_KEYS="a, b").GROQ_API_KEYS == ["a", "b"]  # comma list from the environment


async def test_groq_rotates_on_429_even_with_retries_off(gw, monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEYS", ["groq-2"])

    def handler(request):
        if request.headers["authorization"] == "Bearer groq-key":
            return httpx.Response(429, text="rate limited", headers={"retry-after": "20"})
        return ok("second key works")

    seen = gw(handler)
    res = await llm.chat([{"role": "user", "content": "x"}], model="groq/llama-test", fallbacks=False, retries=0)
    assert res.content == "second key works" and res.provider == "groq"
    assert [r.headers["authorization"] for r in seen] == ["Bearer groq-key", "Bearer groq-2"]


async def test_groq_streaming_rotates_on_401(gw, monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEYS", ["groq-2"])
    body = f"data: {json.dumps({'choices': [{'delta': {'content': 'ok'}}]})}\n\ndata: [DONE]\n\n"

    def handler(request):
        if request.headers["authorization"] == "Bearer groq-key":
            return httpx.Response(401, text="invalid key")
        return httpx.Response(200, text=body, headers={"content-type": "text/event-stream"})

    seen = gw(handler)
    events = [e async for e in llm.stream_events([{"role": "user", "content": "x"}], model="groq/llama-test")]
    assert events[-1][1].content == "ok" and events[-1][1].provider == "groq" and len(seen) == 2


async def test_all_groq_keys_exhausted_reports_429(gw, monkeypatch):
    monkeypatch.setattr(settings, "GROQ_API_KEYS", ["groq-2", "groq-3"])
    seen = gw(lambda r: httpx.Response(429, text="rate limited", headers={"retry-after": "9"}))
    with pytest.raises(LLMUnavailable) as info:
        await llm.chat([{"role": "user", "content": "x"}], model="groq/llama-test", fallbacks=False, retries=0)
    assert len(seen) == 3 and info.value.status == 429 and info.value.headers["retry-after"] == "9"



# ── P1 item 2: hedged calls for tail latency ─────────────────────────────

import asyncio  # noqa: E402

from core.metrics import metrics  # noqa: E402


def _counter(name):
    return metrics.snapshot()["counters"].get(name, 0)


@pytest.fixture
def hedge(gw, monkeypatch):
    """Hedge after 50 ms; Gemini has two keys. ``slow`` maps an Authorization header / host to a delay."""
    monkeypatch.setattr(settings, "LLM_HEDGE_AFTER_SECONDS", 0.05)
    monkeypatch.setattr(settings, "GEMINI_API_KEYS", ["gem-2"])
    state = {"cancelled": [], "slow": {}, "reply": {}}

    def use(slow=None, reply=None):
        state["slow"], state["reply"] = dict(slow or {}), dict(reply or {})

        async def handler(request):
            who = request.headers.get("authorization") or request.url.host
            delay = state["slow"].get(who, 0.0)
            try:
                if delay:
                    await asyncio.sleep(delay)
            except asyncio.CancelledError:
                state["cancelled"].append(who)
                raise
            make = state["reply"].get(who)
            return make(request) if make else ok(f"from {who}")

        return gw(handler)

    use.state = state
    return use


def _sse(*parts):
    body = "".join(f"data: {json.dumps({'choices': [{'delta': {'content': p}}]})}\n\n" for p in parts)
    return body + "data: [DONE]\n\n"


def _stream(*parts):
    return lambda request: httpx.Response(200, text=_sse(*parts), headers={"content-type": "text/event-stream"})


async def test_slow_primary_is_hedged_on_the_next_key(hedge):
    fired, won = _counter("llm.hedge.fired"), _counter("llm.hedge.won")
    seen = hedge(slow={"Bearer gem-key": 5.0})
    started = asyncio.get_running_loop().time()
    res = await llm.chat([{"role": "user", "content": "x"}])
    assert asyncio.get_running_loop().time() - started < 2.0
    assert res.content == "from Bearer gem-2" and res.provider == "gemini" and res.fallback_used is False
    assert [r.headers["authorization"] for r in seen] == ["Bearer gem-key", "Bearer gem-2"]
    assert hedge.state["cancelled"] == ["Bearer gem-key"]  # the loser was cancelled, not left running
    assert _counter("llm.hedge.fired") == fired + 1 and _counter("llm.hedge.won") == won + 1
    # A slow key is not an exhausted key: the next call still starts on the first one.
    seen.clear()
    hedge.state["slow"] = {}
    await llm.chat([{"role": "user", "content": "y"}])
    assert [r.headers["authorization"] for r in seen] == ["Bearer gem-key"]


async def test_single_key_hedges_on_the_next_provider(hedge, monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEYS", [])
    seen = hedge(slow={"Bearer gem-key": 5.0})
    res = await llm.chat([{"role": "user", "content": "x"}])
    assert res.provider == "groq" and res.fallback_used is True and res.content == "from Bearer groq-key"
    assert len(seen) == 2 and hedge.state["cancelled"] == ["Bearer gem-key"]


async def test_fast_primary_is_not_hedged(hedge):
    fired = _counter("llm.hedge.fired")
    seen = hedge()
    res = await llm.chat([{"role": "user", "content": "x"}])
    assert res.content == "from Bearer gem-key" and len(seen) == 1 and _counter("llm.hedge.fired") == fired


async def test_primary_that_answers_after_the_hedge_fired_can_still_win(hedge):
    fired, won = _counter("llm.hedge.fired"), _counter("llm.hedge.won")
    seen = hedge(slow={"Bearer gem-key": 0.15, "Bearer gem-2": 5.0})
    res = await llm.chat([{"role": "user", "content": "x"}])
    assert res.content == "from Bearer gem-key" and len(seen) == 2
    assert hedge.state["cancelled"] == ["Bearer gem-2"]
    assert _counter("llm.hedge.fired") == fired + 1 and _counter("llm.hedge.won") == won


async def test_failed_hedge_waits_for_the_primary(hedge):
    seen = hedge(slow={"Bearer gem-key": 0.15},
                 reply={"Bearer gem-2": lambda r: httpx.Response(400, text="bad request")})
    res = await llm.chat([{"role": "user", "content": "x"}])
    assert res.content == "from Bearer gem-key" and len(seen) == 2


async def test_hedge_disabled_at_zero(hedge, monkeypatch):
    monkeypatch.setattr(settings, "LLM_HEDGE_AFTER_SECONDS", 0.0)
    seen = hedge(slow={"Bearer gem-key": 0.2})
    res = await llm.chat([{"role": "user", "content": "x"}])
    assert res.content == "from Bearer gem-key" and len(seen) == 1


async def test_intent_calls_are_never_hedged(hedge):
    seen = hedge(slow={"Bearer gem-key": 0.2})
    res = await llm.chat([{"role": "user", "content": "x"}], model="gemini/gemini-test", fallbacks=False,
                         retries=0, purpose="intent")
    assert res.content == "from Bearer gem-key" and len(seen) == 1


async def test_local_providers_are_never_hedged(hedge):
    seen = hedge(slow={"127.0.0.1": 0.2})
    res = await llm.chat([{"role": "user", "content": "x"}], model="ollama/qwen-test")
    assert res.provider == "ollama" and len(seen) == 1
    seen.clear()
    await llm.chat([{"role": "user", "content": "x"}], local_only=True)
    assert len(seen) == 1


async def test_hedge_request_keeps_gemini_signature_handling(hedge, monkeypatch):
    history = [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "", "tool_calls": [
            {"id": "fast_0", "type": "function", "function": {"name": "screen_look", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "fast_0", "content": "screen"},
    ]
    seen = hedge(slow={"Bearer gem-key": 5.0})
    await llm.chat(history)
    primary, hedged = (json.loads(r.content)["messages"] for r in seen)
    assert primary == hedged  # the same messages, on the second key
    assert hedged[1]["tool_calls"][0]["extra_content"]["google"]["thought_signature"] == "skip_thought_signature_validator"
    # Hedging on another provider strips the Gemini-only field as usual.
    monkeypatch.setattr(settings, "GEMINI_API_KEYS", [])
    seen.clear()
    await llm.chat(history)
    assert "extra_content" not in json.loads(seen[1].content)["messages"][1]["tool_calls"][0]
    assert "extra_content" not in history[1]["tool_calls"][0]


async def test_streaming_slow_primary_is_hedged_without_interleaving(hedge):
    fired, won = _counter("llm.hedge.fired"), _counter("llm.hedge.won")
    seen = hedge(slow={"Bearer gem-key": 5.0},
                 reply={"Bearer gem-key": _stream("slow ", "primary"), "Bearer gem-2": _stream("fast ", "hedge")})
    events = [e async for e in llm.stream_events([{"role": "user", "content": "x"}])]
    assert [v for k, v in events if k == "text"] == ["fast ", "hedge"]
    assert events[-1][0] == "done" and events[-1][1].content == "fast hedge" and events[-1][1].provider == "gemini"
    assert len(seen) == 2 and hedge.state["cancelled"] == ["Bearer gem-key"]
    assert _counter("llm.hedge.fired") == fired + 1 and _counter("llm.hedge.won") == won + 1


async def test_streaming_headers_without_output_still_get_hedged(hedge):
    """'Started' means first output, not just response headers: a stalled body is hedged too."""
    closed = []

    async def stalled_body():
        try:
            await asyncio.sleep(5)
            yield b""
        finally:
            closed.append(True)

    seen = hedge(reply={
        "Bearer gem-key": lambda r: httpx.Response(200, content=stalled_body(),
                                                   headers={"content-type": "text/event-stream"}),
        "Bearer gem-2": _stream("hedged"),
    })
    events = [e async for e in llm.stream_events([{"role": "user", "content": "x"}])]
    assert [v for k, v in events if k == "text"] == ["hedged"] and len(seen) == 2
    assert closed == [True]  # the primary's stream was closed


async def test_streaming_primary_that_started_is_not_hedged(hedge):
    async def slow_tail():
        yield _sse("first ").replace("data: [DONE]\n\n", "").encode()
        await asyncio.sleep(0.2)
        yield _sse("second").encode()

    seen = hedge(reply={"Bearer gem-key": lambda r: httpx.Response(200, content=slow_tail(),
                                                                   headers={"content-type": "text/event-stream"})})
    events = [e async for e in llm.stream_events([{"role": "user", "content": "x"}])]
    assert [v for k, v in events if k == "text"] == ["first ", "second"] and len(seen) == 1


async def test_streaming_hedge_disabled_at_zero(hedge, monkeypatch):
    monkeypatch.setattr(settings, "LLM_HEDGE_AFTER_SECONDS", 0.0)
    seen = hedge(slow={"Bearer gem-key": 0.2}, reply={"Bearer gem-key": _stream("primary")})
    events = [e async for e in llm.stream_events([{"role": "user", "content": "x"}])]
    assert events[-1][1].content == "primary" and len(seen) == 1


async def test_streaming_consumer_that_stops_early_cancels_both_requests(hedge):
    seen = hedge(slow={"Bearer gem-key": 5.0},
                 reply={"Bearer gem-key": _stream("a"), "Bearer gem-2": _stream("b ", "c")})
    gen = llm.stream_events([{"role": "user", "content": "x"}])
    first = await gen.__anext__()
    await gen.aclose()
    assert first == ("text", "b ") and len(seen) == 2 and hedge.state["cancelled"] == ["Bearer gem-key"]


def test_hedge_setting_default():
    from core.config import Settings

    assert Settings(_env_file=None).LLM_HEDGE_AFTER_SECONDS == 3.0  # the default, not whatever .env says


# ── P1 fix round 1: hedges fail fast, never rotate, never reuse a key the primary used ──

def _auths(seen, prefix="Bearer "):
    return [r.headers.get("authorization") for r in seen if (r.headers.get("authorization") or "").startswith(prefix)]


async def test_key_hedge_is_a_single_attempt(hedge, monkeypatch):
    seen = hedge(slow={"Bearer gem-key": 1.0},
                 reply={"Bearer gem-2": lambda r: httpx.Response(429, text="quota", headers={"retry-after": "0"})})
    res = await llm.chat([{"role": "user", "content": "x"}])
    assert res.content == "from Bearer gem-key"
    assert _auths(seen).count("Bearer gem-2") == 1  # no retry after the 429
    assert gateway_module._KEY_CURSOR.get("gemini", 0) == 0


async def test_provider_hedge_is_a_single_attempt_and_never_rotates(hedge, monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEYS", [])
    monkeypatch.setattr(settings, "GROQ_API_KEYS", ["groq-2"])
    seen = hedge(slow={"Bearer gem-key": 1.0},
                 reply={"Bearer groq-key": lambda r: httpx.Response(429, text="rate limited")})
    res = await llm.chat([{"role": "user", "content": "x"}])
    assert res.content == "from Bearer gem-key"
    groq = [a for a in _auths(seen) if "groq" in a]
    assert groq == ["Bearer groq-key"]  # one attempt, no rotation to groq-2, no retry
    assert gateway_module._KEY_CURSOR.get("groq", 0) == 0  # the intent step's Groq key choice is untouched


async def test_streaming_provider_hedge_never_rotates(hedge, monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEYS", [])
    monkeypatch.setattr(settings, "GROQ_API_KEYS", ["groq-2"])
    seen = hedge(slow={"Bearer gem-key": 0.3}, reply={
        "Bearer gem-key": _stream("primary"), "Bearer groq-key": lambda r: httpx.Response(401, text="invalid key")})
    events = [e async for e in llm.stream_events([{"role": "user", "content": "x"}])]
    assert events[-1][1].content == "primary"
    assert [a for a in _auths(seen) if "groq" in a] == ["Bearer groq-key"]
    assert gateway_module._KEY_CURSOR.get("groq", 0) == 0


async def test_no_provider_hedge_onto_the_intent_providers_only_key(hedge, monkeypatch):
    monkeypatch.setattr(settings, "GEMINI_API_KEYS", [])
    monkeypatch.setattr(settings, "INTENT_MODEL", "groq/llama-test")
    seen = hedge(slow={"Bearer gem-key": 0.2})
    res = await llm.chat([{"role": "user", "content": "x"}])
    assert res.content == "from Bearer gem-key" and len(seen) == 1
    # With a second Groq key the hedge is allowed again.
    monkeypatch.setattr(settings, "GROQ_API_KEYS", ["groq-2"])
    seen.clear()
    res = await llm.chat([{"role": "user", "content": "x"}])
    assert len(seen) == 2


async def test_hedge_skips_keys_the_primary_already_used(hedge, monkeypatch):
    """gem-key is exhausted, the primary moved on to gem-2 (slow): the hedge must use gem-3, not gem-key."""
    monkeypatch.setattr(settings, "GEMINI_API_KEYS", ["gem-2", "gem-3"])
    seen = hedge(slow={"Bearer gem-2": 5.0}, reply={"Bearer gem-key": lambda r: httpx.Response(429, text="quota")})
    res = await llm.chat([{"role": "user", "content": "x"}])
    assert res.content == "from Bearer gem-3"
    assert _auths(seen) == ["Bearer gem-key", "Bearer gem-2", "Bearer gem-3"]


async def test_no_key_hedge_when_the_primary_used_every_other_key(hedge):
    seen = hedge(slow={"Bearer gem-2": 0.3}, reply={"Bearer gem-key": lambda r: httpx.Response(429, text="quota")})
    res = await llm.chat([{"role": "user", "content": "x"}])
    assert res.content == "from Bearer gem-2"
    assert _auths(seen) == ["Bearer gem-key", "Bearer gem-2"]  # no hedge back onto the exhausted key


async def test_stream_failure_after_output_is_never_restarted_on_another_provider(gw, monkeypatch):
    """stream_events itself guards partial output, whatever a single-provider stream does after it."""
    from core.llm import LLMResponse

    tried = []

    async def fake_stream_one(prov_name, mod, payload, timeout, fallback_used, key=None):
        tried.append(prov_name)
        if prov_name == "gemini":
            yield "text", "partial "
            raise RuntimeError("connection dropped")
        yield "text", "other provider"
        yield "done", LLMResponse("other provider", prov_name, mod)

    monkeypatch.setattr(settings, "LLM_HEDGE_AFTER_SECONDS", 0.0)
    monkeypatch.setattr(llm, "_stream_one", fake_stream_one)
    events = [e async for e in llm.stream_events([{"role": "user", "content": "x"}])]
    assert tried == ["gemini"]
    assert [v for k, v in events if k == "text"] == ["partial "]
    assert events[-1][0] == "done" and events[-1][1].finish_reason == "error" and events[-1][1].content == "partial"
