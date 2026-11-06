"""Intent step: standalone rewrite + classification, retries, fallback model, legacy routing."""

import asyncio
import json
import time

import pytest

from brain import intent as intent_mod
from brain.intent import INTENT_SCHEMA, Intent, analyze, legacy_intent
from brain.task_state import PlanItem, Task, TaskStep, Turn
from core.config import settings
from core.llm import LLMUnavailable
from tests.fakes import FakeLLM, text
from tools.selection import GROUPS

INTENT_MODEL = "groq/openai/gpt-oss-120b"
FALLBACK_MODEL = "gemini/gemini-3.5-flash"  # plan B (G2)

NEXT_DO = {
    "request": "Send a WhatsApp message to Nithesh saying hi",
    "kind": "action",
    "continues_task": True,
    "single_step": False,
    "tool_groups": ["messaging"],
    "needs_screen": False,
}


@pytest.fixture(autouse=True)
def intent_enabled(monkeypatch):
    """conftest disables the intent step for the rest of the suite; enable it here."""
    monkeypatch.setattr(settings, "INTENT_MODEL", INTENT_MODEL)
    monkeypatch.setattr(settings, "INTENT_FALLBACK_MODEL", FALLBACK_MODEL)


def use(monkeypatch, script):
    fake = FakeLLM(script)
    monkeypatch.setattr(intent_mod, "llm", fake)
    return fake


def whatsapp_task() -> Task:
    now = time.time()
    return Task(
        id="t1", goal="Open WhatsApp and message Nithesh 'hi'", status="active", channel="voice",
        plan=[PlanItem("Open WhatsApp", "done"), PlanItem("Send 'hi' to Nithesh", "todo")],
        steps=[TaskStep("open_app", "name=whatsapp", "Opened WhatsApp", True, now)],
        created_at=now, updated_at=now,
    )


def whatsapp_turns():
    now = time.time()
    return [
        Turn("voice", "s1", "what time is it", "It's 4:05 PM.", now - 60),
        Turn("voice", "s1", "open whatsapp and msg nithesh say hi", "Opened WhatsApp. [did: open_app(whatsapp) ✓]", now - 30),
    ]


def raising(*_a, **_k):
    raise LLMUnavailable(["groq/openai/gpt-oss-120b: HTTP 429: rate limited"])


# ── Brief tests ─────────────────────────────────────────────────────────

async def test_parses_valid_json(monkeypatch):
    use(monkeypatch, [text(json.dumps(NEXT_DO))])
    got = await analyze("next do", whatsapp_turns(), whatsapp_task())
    assert isinstance(got, Intent)
    assert got.request == "Send a WhatsApp message to Nithesh saying hi"
    assert got.kind == "action"
    assert got.continues_task is True and got.single_step is False
    assert got.tool_groups == ["messaging"] and got.needs_screen is False
    assert got.source == "llm"


async def test_prompt_contains_task_and_turns(monkeypatch):
    fake = use(monkeypatch, [text(json.dumps(NEXT_DO))])
    await analyze("next do", whatsapp_turns(), whatsapp_task())
    msgs = fake.calls[0]["messages"]
    assert msgs[0]["role"] == "system"
    everything = "\n".join(m["content"] for m in msgs)
    assert "Open WhatsApp and message Nithesh 'hi'" in everything  # task goal
    assert "what time is it" in everything and "It's 4:05 PM." in everything  # 1st turn
    assert "open whatsapp and msg nithesh say hi" in everything and "[did: open_app(whatsapp) ✓]" in everything
    assert "next do" in msgs[-1]["content"]
    system = msgs[0]["content"]
    for turn_text in ("what time is it", "It's 4:05 PM.", "open whatsapp and msg nithesh say hi", "Opened WhatsApp. [did"):
        assert turn_text not in system
    # Turns appear oldest first, the new message after them.
    dynamic = msgs[-1]["content"]
    assert dynamic.index("what time is it") < dynamic.index("open whatsapp and msg") < dynamic.rindex("next do")


async def test_retry_then_fallback_model(monkeypatch):
    fake = use(monkeypatch, [text("garbage"), text("not json either"), text(json.dumps(NEXT_DO))])
    got = await analyze("next do", whatsapp_turns(), whatsapp_task())
    assert fake.calls[0]["model"] == settings.INTENT_MODEL
    assert fake.calls[1]["model"] == settings.INTENT_MODEL
    assert fake.calls[2]["model"] == settings.INTENT_FALLBACK_MODEL
    assert got.source == "llm" and got.request == NEXT_DO["request"]


async def test_total_failure_returns_legacy(monkeypatch):
    """Review Focus 4: intent model rate-limited, fallback failing too → legacy routing, no exception."""
    fake = use(monkeypatch, raising)
    got = await analyze("open chrome", [], None)
    assert got.source == "legacy" and got.kind == "action" and got.request == "open chrome"
    got = await analyze("how do I restart my router?", [], None)
    assert got.source == "legacy" and got.kind == "question"
    assert {c["model"] for c in fake.calls} == {INTENT_MODEL, FALLBACK_MODEL}


async def test_unknown_tool_groups_dropped(monkeypatch):
    use(monkeypatch, [text(json.dumps({**NEXT_DO, "tool_groups": ["messaging", "bogus"]}))])
    got = await analyze("next do", whatsapp_turns(), whatsapp_task())
    assert got.tool_groups == ["messaging"] and got.source == "llm"


# ── Controller rulings ──────────────────────────────────────────────────

async def test_empty_intent_model_disables_step(monkeypatch):
    """R3: INTENT_MODEL='' → legacy result, no LLM call."""
    monkeypatch.setattr(settings, "INTENT_MODEL", "")
    fake = use(monkeypatch, [text(json.dumps(NEXT_DO))])
    got = await analyze("open chrome", whatsapp_turns(), whatsapp_task())
    assert fake.calls == []
    assert got == legacy_intent("open chrome")


async def test_empty_fallback_model_is_skipped(monkeypatch):
    """R3: INTENT_FALLBACK_MODEL='' → after the intent model's two attempts, straight to legacy."""
    monkeypatch.setattr(settings, "INTENT_FALLBACK_MODEL", "")
    fake = use(monkeypatch, [text("garbage"), text("garbage"), text(json.dumps(NEXT_DO))])
    got = await analyze("next do", [], None)
    assert len(fake.calls) == 2 and got.source == "legacy"


def test_legacy_single_step_follows_kind():
    """R4: legacy actions keep today's single-step behaviour; questions don't."""
    act = legacy_intent("open chrome")
    assert act.kind == "action" and act.single_step is True
    q = legacy_intent("how do I restart my router?")
    assert q.kind == "question" and q.single_step is False
    assert act.continues_task is False and act.tool_groups == [] and act.needs_screen is False
    assert act.source == "legacy"


def test_schema_tool_groups_enum():
    """R2: every selection group plus keyboard/history, sorted, no duplicates."""
    enum = INTENT_SCHEMA["properties"]["tool_groups"]["items"]["enum"]
    assert enum == sorted(set(GROUPS) | {"keyboard", "history"})
    assert len(enum) == len(set(enum))
    assert set(INTENT_SCHEMA["required"]) == {"request", "kind", "continues_task", "single_step", "tool_groups",
                                              "needs_screen"}
    assert "source" not in INTENT_SCHEMA["properties"]


# ── Robustness ──────────────────────────────────────────────────────────

async def test_call_parameters(monkeypatch):
    fake = use(monkeypatch, [text(json.dumps(NEXT_DO))])
    await analyze("next do", [], None)
    kw = fake.calls[0]["kwargs"]
    assert kw["temperature"] == 0 and kw["max_tokens"] == intent_mod.MAX_TOKENS
    assert kw["fallbacks"] is False  # never wander into the general fallback chain
    fmt = kw["response_format"]
    assert fmt["type"] == "json_schema" and fmt["json_schema"]["schema"] == INTENT_SCHEMA


async def test_system_prompt_is_identical_across_calls(monkeypatch):
    """The static part must not vary per call, so provider prompt caching applies."""
    fake = use(monkeypatch, [text(json.dumps(NEXT_DO)), text(json.dumps(NEXT_DO))])
    await analyze("next do", whatsapp_turns(), whatsapp_task())
    await analyze("set volume to 30", [], None)
    assert fake.calls[0]["messages"][0] == fake.calls[1]["messages"][0]


async def test_code_fenced_json_is_accepted(monkeypatch):
    use(monkeypatch, [text("```json\n" + json.dumps(NEXT_DO) + "\n```")])
    got = await analyze("next do", whatsapp_turns(), whatsapp_task())
    assert got.source == "llm" and got.kind == "action"


@pytest.mark.parametrize("bad", [
    {**NEXT_DO, "kind": "command"},
    {**NEXT_DO, "continues_task": "yes"},
    {**NEXT_DO, "request": "   "},
    {**NEXT_DO, "tool_groups": "messaging"},
    {k: v for k, v in NEXT_DO.items() if k != "needs_screen"},
    ["not", "an", "object"],
])
async def test_invalid_output_is_retried(monkeypatch, bad):
    fake = use(monkeypatch, [text(json.dumps(bad)), text(json.dumps(NEXT_DO))])
    got = await analyze("next do", whatsapp_turns(), whatsapp_task())
    assert len(fake.calls) == 2 and fake.calls[1]["model"] == INTENT_MODEL
    assert got.source == "llm" and got.request == NEXT_DO["request"]


async def test_provider_failure_goes_straight_to_fallback(monkeypatch):
    """The gateway already retried transport errors; an unavailable model is not asked again."""
    def script(messages, tools):
        if len(fake.calls) == 1:
            raise LLMUnavailable(["groq: HTTP 429"])
        return text(json.dumps(NEXT_DO))

    fake = use(monkeypatch, script)
    got = await analyze("next do", whatsapp_turns(), whatsapp_task())
    assert [c["model"] for c in fake.calls] == [INTENT_MODEL, FALLBACK_MODEL]
    assert got.source == "llm"


async def test_unexpected_exception_never_raises(monkeypatch):
    def boom(*_a, **_k):
        raise ValueError("unexpected")

    use(monkeypatch, boom)
    got = await analyze("open chrome", [], None)
    assert got.source == "legacy" and got.kind == "action"


async def test_legacy_fallback_is_logged_at_warning(monkeypatch):
    use(monkeypatch, raising)
    seen = []
    sink = intent_mod.logger.add(lambda m: seen.append(m.record), level="WARNING")
    try:
        await analyze("open chrome", [], None)
    finally:
        intent_mod.logger.remove(sink)
    assert any("legacy" in r["message"] and r["level"].name == "WARNING" for r in seen)


# ── Fix round 1: latency bound, action notes, length handling, prompt layout ──

class SlowLLM:
    """Never answers in time."""

    def __init__(self):
        self.calls = []

    async def chat(self, messages, *, model=None, **kw):
        self.calls.append(model)
        await asyncio.sleep(5)
        return text(json.dumps(NEXT_DO))


async def test_deadline_returns_legacy_within_bound(monkeypatch):
    monkeypatch.setattr(intent_mod, "DEADLINE_SECONDS", 0.3)
    slow = SlowLLM()
    monkeypatch.setattr(intent_mod, "llm", slow)
    started = time.perf_counter()
    got = await analyze("open chrome", [], None)
    elapsed = time.perf_counter() - started
    assert got.source == "legacy" and got.kind == "action"
    assert elapsed < 0.3 + 0.5
    assert slow.calls == [INTENT_MODEL]


def test_deadline_and_per_call_timeout_values():
    """G2: Groq primary (~0.5-0.7 s) capped at 1.5 s; 3.5 s overall; 2.0 s for the Gemini Flash fallback (~1.6 s)."""
    assert intent_mod.TIMEOUT_SECONDS == 1.5
    assert intent_mod.DEADLINE_SECONDS == 3.5
    assert intent_mod.FALLBACK_RESERVE_SECONDS == intent_mod.DEADLINE_SECONDS - intent_mod.TIMEOUT_SECONDS == 2.0


async def test_429_on_primary_goes_to_fallback_once_without_sleeping(monkeypatch):
    """Real gateway, mocked HTTP: Groq 429 → no in-gateway retry/sleep → Gemini answers."""
    import httpx

    from core import connectivity
    from core.llm import gateway as gateway_module
    from core.llm import llm as real_llm

    monkeypatch.setattr(settings, "GROQ_API_KEY", "groq-key")
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "gem-key")
    monkeypatch.setattr(settings, "GEMINI_API_KEYS", [])
    monkeypatch.setattr(gateway_module, "_KEY_CURSOR", {})
    slept = []

    async def fake_sleep(delay):
        slept.append(delay)

    monkeypatch.setattr(gateway_module.asyncio, "sleep", fake_sleep)
    seen = []

    def handler(request):
        seen.append(request.url.host)
        if request.url.host == "api.groq.com":
            return httpx.Response(429, text="rate limited", headers={"retry-after": "5"})
        body = {"model": "gemini-3.5-flash", "choices": [{"message": {"role": "assistant",
                "content": json.dumps(NEXT_DO)}, "finish_reason": "stop"}]}
        return httpx.Response(200, json=body)

    connectivity.set_state_for_tests(True)
    real_llm.set_transport(httpx.MockTransport(handler))
    try:
        got = await analyze("next do", whatsapp_turns(), whatsapp_task())
    finally:
        real_llm.set_transport(None)
        connectivity.set_state_for_tests(None)
    assert seen == ["api.groq.com", "generativelanguage.googleapis.com"]
    assert slept == []
    assert got.source == "llm" and got.request == NEXT_DO["request"]


async def test_intent_calls_disable_gateway_retries(monkeypatch):
    fake = use(monkeypatch, [text(json.dumps(NEXT_DO))])
    await analyze("next do", [], None)
    kw = fake.calls[0]["kwargs"]
    assert kw["retries"] == 0 and kw["timeout"] == intent_mod.TIMEOUT_SECONDS


async def test_length_finish_skips_same_model_retry(monkeypatch):
    from core.llm import LLMResponse

    truncated = LLMResponse(content='{"request": "Send a Wha', provider="fake", model="fake-1", finish_reason="length")
    fake = use(monkeypatch, [truncated, text(json.dumps(NEXT_DO))])
    got = await analyze("next do", whatsapp_turns(), whatsapp_task())
    assert [c["model"] for c in fake.calls] == [INTENT_MODEL, FALLBACK_MODEL]
    assert got.source == "llm"


async def test_long_reply_keeps_action_notes(monkeypatch):
    fake = use(monkeypatch, [text(json.dumps(NEXT_DO))])
    notes = "[did: open_app(whatsapp) ✓, send_message(to=Nithesh, text=hi) ✗]"
    reply = "Here is a long explanation. " * 30 + notes
    assert len(reply) > 600
    turns = [Turn("voice", "s1", "open whatsapp and msg nithesh say hi", reply, time.time())]
    await analyze("next do", turns, whatsapp_task())
    dynamic = fake.calls[0]["messages"][-1]["content"]
    assert notes in dynamic
    jarvis_line = next(line for line in dynamic.splitlines() if "JARVIS:" in line)
    assert jarvis_line.rstrip().endswith(notes)
    assert len(jarvis_line) < len(reply)  # the prose before the notes was clipped


def test_examples_use_the_real_context_layout():
    system = intent_mod.SYSTEM_PROMPT
    assert "Examples (illustrative only — not part of this conversation)" in system
    # Same layout as build_messages: multi-line Task.summary() and the conversation heading.
    assert "Recent conversation (oldest first):" in system
    assert "\nPlan:\n" in system and "\nRecent steps:\n" in system and "\nTask: " in system
    real = intent_mod.build_messages("next do", whatsapp_turns(), whatsapp_task())[1]["content"]
    for heading in ("Current task:", "Recent conversation (oldest first):", "New message: "):
        assert heading in real and heading in system


def test_correction_example_present():
    system = intent_mod.SYSTEM_PROMPT
    assert "not windows but in the chrome browser tabs" in system
    assert "Switch to the Instagram tab in Google Chrome" in system


# ── I2: free-tier quota: cooldown after a 429 / timeout, primary timeout leaves the fallback time ──

@pytest.fixture(autouse=True)
def no_cooldown(monkeypatch):
    monkeypatch.setattr(intent_mod, "_COOLDOWN", {})
    monkeypatch.setattr(intent_mod, "_TIMEOUTS", {})


def rate_limited(headers=None):
    exc = LLMUnavailable(["groq/openai/gpt-oss-120b: HTTP 429: rate limited"])
    exc.status, exc.headers, exc.timed_out = 429, dict(headers or {}), False
    return exc


def primary_fails_with(monkeypatch, exc):
    def script(messages, tools):
        if fake.calls[-1]["model"] == INTENT_MODEL:
            raise exc
        return text(json.dumps(NEXT_DO))

    fake = use(monkeypatch, script)
    return fake


@pytest.mark.parametrize("headers, seconds", [
    ({"retry-after": "12"}, 12.0),
    ({"x-ratelimit-reset-tokens": "1m2.5s"}, 62.5),
    ({"x-ratelimit-reset-tokens": "750ms"}, 0.75),
    ({}, 30.0),
])
async def test_primary_429_skips_primary_until_reset(monkeypatch, headers, seconds):
    now = [1000.0]
    monkeypatch.setattr(intent_mod, "_clock", lambda: now[0])
    fake = primary_fails_with(monkeypatch, rate_limited(headers))
    got = await analyze("next do", [], None)
    assert got.source == "llm" and [c["model"] for c in fake.calls] == [INTENT_MODEL, FALLBACK_MODEL]
    now[0] += seconds - 0.01
    await analyze("next do", [], None)
    assert [c["model"] for c in fake.calls][2:] == [FALLBACK_MODEL]  # primary skipped during the cooldown
    now[0] += 0.02
    await analyze("next do", [], None)
    assert [c["model"] for c in fake.calls][3:] == [INTENT_MODEL, FALLBACK_MODEL]  # tried again after it


def _timeout_exc():
    exc = LLMUnavailable(["gemini: ReadTimeout"])
    exc.status, exc.headers, exc.timed_out = None, {}, True
    return exc


async def test_one_slow_primary_call_does_not_bench_it(monkeypatch):
    """G1: a single timeout must not push the next turns to the fallback."""
    now = [50.0]
    monkeypatch.setattr(intent_mod, "_clock", lambda: now[0])
    fake = primary_fails_with(monkeypatch, _timeout_exc())
    await analyze("next do", [], None)
    now[0] += 1.0
    await analyze("next do", [], None)
    assert [c["model"] for c in fake.calls] == [INTENT_MODEL, FALLBACK_MODEL, INTENT_MODEL, FALLBACK_MODEL]


async def test_two_consecutive_primary_timeouts_start_a_short_cooldown(monkeypatch):
    now = [50.0]
    monkeypatch.setattr(intent_mod, "_clock", lambda: now[0])
    fake = primary_fails_with(monkeypatch, _timeout_exc())
    await analyze("next do", [], None)
    await analyze("next do", [], None)  # second timeout in a row → short cooldown
    assert 5.0 <= intent_mod.TIMEOUT_COOLDOWN_SECONDS <= 10.0
    now[0] += intent_mod.TIMEOUT_COOLDOWN_SECONDS - 0.5
    await analyze("next do", [], None)
    now[0] += 1.0
    await analyze("next do", [], None)
    assert [c["model"] for c in fake.calls] == [INTENT_MODEL, FALLBACK_MODEL, INTENT_MODEL, FALLBACK_MODEL,
                                                FALLBACK_MODEL, INTENT_MODEL, FALLBACK_MODEL]


async def test_a_primary_success_resets_the_timeout_count(monkeypatch):
    now = [50.0]
    monkeypatch.setattr(intent_mod, "_clock", lambda: now[0])
    outcomes = iter(["timeout", "ok", "timeout", "ok"])

    def script(messages, tools):
        if fake.calls[-1]["model"] == INTENT_MODEL and next(outcomes) == "timeout":
            raise _timeout_exc()
        return text(json.dumps(NEXT_DO))

    fake = use(monkeypatch, script)
    for _ in range(3):
        await analyze("next do", [], None)
    assert [c["model"] for c in fake.calls] == [INTENT_MODEL, FALLBACK_MODEL, INTENT_MODEL, INTENT_MODEL,
                                                FALLBACK_MODEL]
    assert not intent_mod._COOLDOWN.get(INTENT_MODEL)


async def test_other_primary_errors_do_not_cool_down(monkeypatch):
    exc = LLMUnavailable(["groq: HTTP 400"])
    exc.status, exc.headers, exc.timed_out = 400, {}, False
    fake = primary_fails_with(monkeypatch, exc)
    await analyze("next do", [], None)
    await analyze("next do", [], None)
    assert [c["model"] for c in fake.calls] == [INTENT_MODEL, FALLBACK_MODEL] * 2


def test_primary_timeout_leaves_the_fallback_at_least_2_s():
    """Plan B: Gemini Flash (~1.6 s measured) needs the full 2.0 s reserve."""
    assert intent_mod.DEADLINE_SECONDS - intent_mod.FALLBACK_RESERVE_SECONDS >= intent_mod.TIMEOUT_SECONDS
    assert intent_mod.FALLBACK_RESERVE_SECONDS >= 2.0


class HangingPrimary:
    """The primary never answers; the fallback answers at once."""

    def __init__(self):
        self.calls = []

    async def chat(self, messages, *, model=None, timeout=None, **kw):
        self.calls.append({"model": model, "timeout": timeout})
        if model == INTENT_MODEL:
            await asyncio.sleep(10)
        return text(json.dumps(NEXT_DO))


async def test_hanging_primary_still_lets_the_fallback_answer(monkeypatch):
    # Same proportions as the real plan-B values (1.5 s cap, 3.5 s deadline, 2.0 s reserve), scaled down 5x.
    monkeypatch.setattr(intent_mod, "TIMEOUT_SECONDS", 0.3)
    monkeypatch.setattr(intent_mod, "DEADLINE_SECONDS", 0.7)
    monkeypatch.setattr(intent_mod, "FALLBACK_RESERVE_SECONDS", 0.4)
    monkeypatch.setattr(intent_mod, "_MIN_CALL_SECONDS", 0.05)
    fake = HangingPrimary()
    monkeypatch.setattr(intent_mod, "llm", fake)
    started = time.perf_counter()
    got = await analyze("next do", [], None)
    elapsed = time.perf_counter() - started
    assert got.source == "llm" and [c["model"] for c in fake.calls] == [INTENT_MODEL, FALLBACK_MODEL]
    assert elapsed < intent_mod.TIMEOUT_SECONDS + 0.5
    assert fake.calls[1]["timeout"] >= intent_mod.FALLBACK_RESERVE_SECONDS - 0.05
    assert not intent_mod._COOLDOWN.get(INTENT_MODEL)  # one slow call does not bench the primary


# ── I2c: token budget (Groq free tier is 8k tokens/minute) ──────────────

def _tokens(chars: int) -> int:
    return round(chars / 4)  # rough estimate; no tokenizer for gpt-oss is installed


_SCHEMA_CHARS = len(json.dumps(intent_mod.RESPONSE_FORMAT, separators=(",", ":")))


def test_static_prompt_is_small():
    static = len(intent_mod.SYSTEM_PROMPT) + _SCHEMA_CHARS
    assert _tokens(static) <= 1150  # system prompt + JSON schema (was ~1.8k)


def test_typical_call_stays_trimmed():
    now = time.time()
    turns = [Turn("voice", "s1", "open whatsapp and message nithesh saying ee em chesadu bro",
                  "Opened WhatsApp. Shall I send the message to Nithesh now? [did: open_app(whatsapp) ✓]", now - 90 + i)
             for i in range(settings.INTENT_HISTORY_TURNS)]
    messages = intent_mod.build_messages("next do", turns, whatsapp_task())
    total = sum(len(m["content"]) for m in messages) + _SCHEMA_CHARS
    assert _tokens(total) <= 1400  # 4 turns + a task (was ~2.2k with 8 turns and the long prompt)


def test_history_defaults_and_clips():
    from core.config import Settings

    assert Settings.model_fields["INTENT_HISTORY_TURNS"].default == 4
    notes = "[did: open_app(whatsapp) ✓]"
    turn = Turn("voice", "s1", "u" * 500, "a" * 500 + " " + notes, time.time())
    lines = intent_mod.render_context("m", [turn], None).splitlines()
    user = next(ln for ln in lines if "User:" in ln)
    jarvis = next(ln for ln in lines if "JARVIS:" in ln)
    assert len(user.split("User: ", 1)[1]) <= 200
    assert jarvis.endswith(notes) and len(jarvis.split("JARVIS: ", 1)[1]) <= 200 + 1 + len(notes)


def test_task_summary_clips_plan_subgoals():
    task = whatsapp_task()
    task.plan.append(PlanItem("x" * 300, "todo"))
    line = next(ln for ln in task.summary().splitlines() if "xxx" in ln)
    assert len(line) <= 80 + 8


# ── G2: the 429 cooldown benches the primary only once every Groq key is exhausted ──

def _real_gateway(monkeypatch, handler):
    import httpx

    from core import connectivity
    from core.llm import gateway as gateway_module
    from core.llm import llm as real_llm

    monkeypatch.setattr(settings, "GROQ_API_KEY", "groq-key")
    monkeypatch.setattr(settings, "GROQ_API_KEYS", ["groq-2"])
    monkeypatch.setattr(settings, "GEMINI_API_KEY", "gem-key")
    monkeypatch.setattr(settings, "GEMINI_API_KEYS", [])
    monkeypatch.setattr(gateway_module, "_KEY_CURSOR", {})
    monkeypatch.setattr(intent_mod, "llm", real_llm)
    connectivity.set_state_for_tests(True)
    real_llm.set_transport(httpx.MockTransport(handler))


def _intent_reply():
    import httpx

    body = {"model": "m", "choices": [{"message": {"role": "assistant", "content": json.dumps(NEXT_DO)},
                                       "finish_reason": "stop"}]}
    return httpx.Response(200, json=body)


async def test_groq_second_key_answers_and_no_cooldown(monkeypatch):
    import httpx

    from core import connectivity
    from core.llm import llm as real_llm

    seen = []

    def handler(request):
        seen.append((request.url.host, request.headers["authorization"]))
        if request.headers["authorization"] == "Bearer groq-key":
            return httpx.Response(429, text="rate limited", headers={"retry-after": "30"})
        return _intent_reply()

    _real_gateway(monkeypatch, handler)
    try:
        got = await analyze("next do", whatsapp_turns(), whatsapp_task())
    finally:
        real_llm.set_transport(None)
        connectivity.set_state_for_tests(None)
    assert got.source == "llm"
    assert seen == [("api.groq.com", "Bearer groq-key"), ("api.groq.com", "Bearer groq-2")]  # answered by groq
    assert not intent_mod._COOLDOWN


async def test_every_groq_key_exhausted_benches_the_primary(monkeypatch):
    import httpx

    from core import connectivity
    from core.llm import llm as real_llm

    def handler(request):
        if request.url.host == "api.groq.com":
            return httpx.Response(429, text="rate limited", headers={"retry-after": "30"})
        return _intent_reply()

    _real_gateway(monkeypatch, handler)
    try:
        got = await analyze("next do", whatsapp_turns(), whatsapp_task())
    finally:
        real_llm.set_transport(None)
        connectivity.set_state_for_tests(None)
    assert got.source == "llm" and intent_mod._COOLDOWN.get(INTENT_MODEL)
