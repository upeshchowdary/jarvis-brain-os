"""Intent step: standalone rewrite + classification, retries, fallback model, legacy routing."""

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
FALLBACK_MODEL = "gemini/gemini-3.5-flash-lite"

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
