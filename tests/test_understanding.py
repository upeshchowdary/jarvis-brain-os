"""Task 6: intent-driven routing, the current task and the run-to-completion agent loop.

``brain.pipeline.analyze`` is patched to return scripted :class:`Intent` objects; the agent runs on a
``FakeLLM``; every tool that would touch the OS has its ``run`` replaced by an ``AsyncMock``.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

import brain.pipeline as pipeline
from automation.gateway import gateway
from brain import router, task_state
from brain.conversation_manager import conversation_manager
from brain.intent import Intent, legacy_intent
from brain.pipeline import brain
from brain.prompts import AGENT_RULES, build_system_prompt
from core.config import settings
from tests.fakes import FakeLLM, call, install, text
from tools.base import ToolResult
from tools.registry import load_builtin_tools, registry
from tools.selection import CORE, GROUPS, select_tools


@pytest.fixture(autouse=True)
def fresh_state(monkeypatch):
    """A private task tracker / turn buffer per test (the pipeline and task_plan share it)."""
    tracker, turns = task_state.TaskTracker(), task_state.RecentTurns()
    monkeypatch.setattr(task_state, "tasks", tracker)
    monkeypatch.setattr(task_state, "recent_turns", turns)
    monkeypatch.setattr(pipeline, "tasks", tracker)
    monkeypatch.setattr(pipeline, "recent_turns", turns)
    gateway.resume()
    yield tracker
    gateway.resume()


@pytest.fixture
def tools(monkeypatch):
    """Replace the OS-touching tools with mocks that report success."""
    load_builtin_tools()
    mocks = {}
    for name, msg in (("open_app", "Opened WhatsApp."), ("send_message", "Message sent to Nithesh. It may take a moment."),
                      ("media_control", "Skipped to the next track."), ("open_website", "Opened the site.")):
        mocks[name] = AsyncMock(return_value=ToolResult.ok(msg))
        monkeypatch.setattr(registry.get(name), "run", mocks[name])
    return mocks


def intent(request, kind="action", continues=False, single=False, groups=None):
    return Intent(request=request, kind=kind, continues_task=continues, single_step=single,
                  tool_groups=list(groups or []), needs_screen=False, source="llm")


def patch_intent(monkeypatch, value):
    mock = AsyncMock(return_value=value) if isinstance(value, Intent) else AsyncMock(side_effect=value)
    monkeypatch.setattr(pipeline, "analyze", mock)
    return mock


def side_effect_names(names):
    reg = load_builtin_tools()
    return [n for n in names if reg.get(n) is not None and reg.get(n).side_effect]


# ── Agent loop ──────────────────────────────────────────────────────────

async def test_multi_step_continues_after_final_tool(monkeypatch, tools):
    patch_intent(monkeypatch, intent("Open WhatsApp and send Nithesh 'hi'", groups=["messaging", "system"]))
    fake = install(monkeypatch, FakeLLM([
        call("open_app", name="whatsapp"),
        call("send_message", app="whatsapp", contact="Nithesh", text="hi"),
        text("Sent."),
    ]))
    res = await brain.handle("open whatsapp and msg nithesh hi", "u-multi")
    tools["open_app"].assert_awaited_once()
    tools["send_message"].assert_awaited_once()
    assert res.response == "Sent." and len(fake.calls) == 3


async def test_single_step_keeps_shortcut(monkeypatch, tools):
    patch_intent(monkeypatch, intent("Launch WhatsApp from the Start menu", single=True, groups=["system"]))
    fake = install(monkeypatch, FakeLLM([call("open_app", name="whatsapp"), text("should not be reached")]))
    res = await brain.handle("whatsapp kholo yaar", "u-single")
    assert len(fake.calls) == 1 and res.response == "Opened WhatsApp."


async def test_action_step_budget(monkeypatch, tools):
    monkeypatch.setattr(settings, "AGENT_MAX_STEPS_ACTION", 7)
    patch_intent(monkeypatch, intent("Open WhatsApp again and again", groups=["system"]))
    fake = install(monkeypatch, FakeLLM(lambda m, t: call("open_app", name="whatsapp")))
    await brain.handle("keep opening whatsapp", "u-budget")
    assert len(fake.calls) == 7


async def test_followup_uses_rewritten_request(monkeypatch, tools):
    patch_intent(monkeypatch, intent("Send a WhatsApp message to Nithesh saying hi", continues=True, groups=["messaging"]))
    fake = install(monkeypatch, FakeLLM([text("Sent.")]))
    await brain.handle("next do", "u-followup")
    user = [m for m in fake.calls[0]["messages"] if m["role"] == "user"][-1]["content"]
    assert "Send a WhatsApp message to Nithesh saying hi" in user
    assert "(user's exact words: next do)" in user


async def test_unchanged_request_has_no_exact_words_suffix(monkeypatch):
    patch_intent(monkeypatch, intent("tell me a joke about owls", kind="chat"))
    fake = install(monkeypatch, FakeLLM([text("Hoo knows.")]))
    await brain.handle("tell me a joke about owls", "u-same")
    user = [m for m in fake.calls[0]["messages"] if m["role"] == "user"][-1]["content"]
    assert user == "tell me a joke about owls"


async def test_yes_open_gets_action_tools(monkeypatch, tools):
    patch_intent(monkeypatch, intent("Open Instagram in the web browser", continues=True, single=True, groups=["web"]))
    fake = install(monkeypatch, FakeLLM([call("open_website", url="https://instagram.com")]))
    await brain.handle("yes open", "u-yes")
    assert "open_app" in fake.offered(0) and "open_website" in fake.offered(0)
    tools["open_website"].assert_awaited_once()


async def test_question_gets_read_only_tools(monkeypatch):
    patch_intent(monkeypatch, intent("How do I restart my router?", kind="question"))
    fake = install(monkeypatch, FakeLLM([text("Unplug it for 30 seconds.")]))
    await brain.handle("how do i restart my router", "u-question")
    assert side_effect_names(fake.offered(0)) == []
    assert "task_plan" not in fake.offered(0)


async def test_groups_select_tools_without_keywords(monkeypatch, tools):
    """'next do' has no keyword; the intent's groups decide which tools are offered."""
    patch_intent(monkeypatch, intent("Send a WhatsApp message to Nithesh saying hi", continues=True, groups=["messaging"]))
    fake = install(monkeypatch, FakeLLM([text("ok")]))
    await brain.handle("next do", "u-groups")
    offered = fake.offered(0)
    assert "send_message" in offered and "task_plan" in offered and "chat_history" in offered


# ── Fast path vs. context ───────────────────────────────────────────────

async def test_next_mid_task_not_media_key(monkeypatch, tools, fresh_state):
    fresh_state.start("Open WhatsApp and message Nithesh 'hi'", "voice")
    analyze = patch_intent(monkeypatch, intent("Send a WhatsApp message to Nithesh saying hi", continues=True,
                                               groups=["messaging"]))
    install(monkeypatch, FakeLLM([text("Done.")]))
    res = await brain.handle("next", "u-next-task")
    tools["media_control"].assert_not_awaited()
    analyze.assert_awaited_once()
    assert res.route == "agent"


async def test_next_without_task_is_media_key(monkeypatch, tools):
    analyze = patch_intent(monkeypatch, intent("unused"))
    fake = install(monkeypatch, FakeLLM([]))
    res = await brain.handle("next", "u-next-free")
    tools["media_control"].assert_awaited_once()
    analyze.assert_not_awaited()
    assert res.route == "fast_path" and fake.calls == []


async def test_stop_during_task_is_emergency_stop(monkeypatch, fresh_state):
    fresh_state.start("Open WhatsApp and message Nithesh 'hi'", "voice")
    analyze = patch_intent(monkeypatch, intent("unused"))
    fake = install(monkeypatch, FakeLLM([]))
    res = await brain.handle("stop", "u-stop")
    assert res.route == "fast_path" and gateway.is_stopped
    analyze.assert_not_awaited()
    assert fake.calls == []


async def test_rewritten_request_hits_fast_path(monkeypatch, tools):
    patch_intent(monkeypatch, intent("open whatsapp", continues=True, single=True))
    fake = install(monkeypatch, FakeLLM([]))
    res = await brain.handle("yes do that one", "u-rewrite-fast")
    tools["open_app"].assert_awaited_once()
    assert res.route == "fast_path" and fake.calls == []


async def test_question_rewrite_never_runs_a_side_effect_command(monkeypatch, tools):
    patch_intent(monkeypatch, intent("open whatsapp", kind="question"))
    fake = install(monkeypatch, FakeLLM([text("WhatsApp is a messaging app.")]))
    res = await brain.handle("what does that whatsapp thing do", "u-q-rewrite")
    tools["open_app"].assert_not_awaited()
    assert res.route == "agent" and side_effect_names(fake.offered(0)) == []


# ── Current task ────────────────────────────────────────────────────────

async def test_question_mid_task_keeps_task(monkeypatch, fresh_state):
    task = fresh_state.start("Send WhatsApp 'hi' to Nithesh", "voice")
    fresh_state.record_step("open_app", {"name": "whatsapp"}, "Opened WhatsApp.", True)
    patch_intent(monkeypatch, intent("What time is it in London right now?", kind="question"))
    fake = install(monkeypatch, FakeLLM([call("get_time_date"), text("It's 3 pm in London.")]))
    await brain.handle("what time is it in london right now?", "u-q-mid")
    cur = fresh_state.current()
    assert cur is task and cur.goal == "Send WhatsApp 'hi' to Nithesh" and cur.status == "active"
    assert [s.tool for s in cur.steps] == ["open_app"]  # the question's tool is not a task step
    assert all(side_effect_names(c["tools"]) == [] and "task_plan" not in c["tools"] for c in fake.calls)


async def test_fast_path_question_mid_task_keeps_task(monkeypatch, fresh_state):
    task = fresh_state.start("Send WhatsApp 'hi' to Nithesh", "voice")
    analyze = patch_intent(monkeypatch, intent("unused"))
    res = await brain.handle("what time is it?", "u-q-fast")
    assert res.route == "fast_path"
    analyze.assert_not_awaited()
    assert fresh_state.current() is task and task.status == "active" and task.steps == []


async def test_new_action_starts_task_and_single_step_finishes_it(monkeypatch, tools, fresh_state):
    patch_intent(monkeypatch, intent("Launch WhatsApp from the Start menu", single=True))
    install(monkeypatch, FakeLLM([call("open_app", name="whatsapp")]))
    await brain.handle("whatsapp kholo", "u-task-done")
    assert fresh_state.current() is None
    last = fresh_state._last
    assert last.goal == "Launch WhatsApp from the Start menu" and last.status == "done" and [s.tool for s in last.steps] == ["open_app"]


async def test_failed_last_step_marks_task_failed(monkeypatch, tools, fresh_state):
    tools["send_message"].return_value = ToolResult.error("Couldn't find Nithesh in WhatsApp.")
    patch_intent(monkeypatch, intent("Send Nithesh 'hi' on WhatsApp", groups=["messaging"]))
    install(monkeypatch, FakeLLM([call("send_message", app="whatsapp", contact="Nithesh", text="hi"),
                                  text("I couldn't find Nithesh in WhatsApp.")]))
    await brain.handle("msg nithesh hi", "u-task-failed")
    assert fresh_state.current() is None and fresh_state._last.status == "failed"


async def test_task_with_open_plan_stays_active(monkeypatch, tools, fresh_state):
    patch_intent(monkeypatch, intent("Open WhatsApp and send Nithesh 'hi'", groups=["messaging"]))
    install(monkeypatch, FakeLLM([
        call("task_plan", action="set", subgoals=["Open WhatsApp", "Send 'hi' to Nithesh"]),
        call("open_app", name="whatsapp"),
        text("WhatsApp is open; which chat?"),
    ]))
    await brain.handle("open whatsapp and msg nithesh hi", "u-task-open")
    cur = fresh_state.current()
    assert cur is not None and cur.status == "active" and len(cur.plan) == 2


async def test_continuation_revives_shelved_task(monkeypatch, tools, fresh_state):
    old = fresh_state.start("Send WhatsApp 'hi' to Nithesh", "voice")
    fresh_state.set_status("shelved")
    patch_intent(monkeypatch, intent("Send a WhatsApp message to Nithesh saying hi", continues=True, groups=["messaging"]))
    install(monkeypatch, FakeLLM([text("On it.")]))
    await brain.handle("continue what we were doing", "u-revive")
    assert fresh_state.current() is old and old.status == "active"


async def test_task_summary_in_system_prompt(monkeypatch, tools, fresh_state):
    fresh_state.start("Send WhatsApp 'hi' to Nithesh", "voice")
    patch_intent(monkeypatch, intent("Send a WhatsApp message to Nithesh saying hi", continues=True, groups=["messaging"]))
    fake = install(monkeypatch, FakeLLM([text("ok")]))
    await brain.handle("next do", "u-summary")
    assert "Task: Send WhatsApp 'hi' to Nithesh [active]" in fake.calls[0]["messages"][0]["content"]


# ── History, progress, models ───────────────────────────────────────────

async def test_action_notes_in_history(monkeypatch, tools, fresh_state):
    patch_intent(monkeypatch, intent("Launch WhatsApp from the Start menu", single=True))
    install(monkeypatch, FakeLLM([call("open_app", name="whatsapp")]))
    sid = "u-notes"
    res = await brain.handle("whatsapp kholo", sid)
    content = conversation_manager.get_history(sid)[-1]["content"]
    assert "[did: open_app(" in content and content.endswith("[did: open_app(whatsapp) ✓]")
    assert "[did:" not in res.response
    turn = pipeline.recent_turns.recent(5, 600)[-1]
    assert turn.user == "whatsapp kholo" and turn.assistant.endswith("[did: open_app(whatsapp) ✓]")


def test_action_notes_format_and_limit():
    from tools.base import ToolCallRecord

    recs = [ToolCallRecord("open_app", {"name": "whatsapp"}, ToolResult.ok("ok")),
            ToolCallRecord("send_message", {"app": "whatsapp", "contact": "Nithesh"}, ToolResult.error("no")),
            ToolCallRecord("list_windows", {}, ToolResult.ok("ok"))]
    assert pipeline._action_notes(recs) == " [did: open_app(whatsapp) ✓, send_message(whatsapp) ✗, list_windows() ✓]"
    many = [ToolCallRecord("type_text", {"text": "x" * 300}, ToolResult.ok("ok")) for _ in range(20)]
    note = pipeline._action_notes(many)
    assert len(note) <= 200 and note.startswith(" [did: ") and note.endswith("]") and note.count("[did:") == 1
    assert pipeline._action_notes([]) == ""


async def test_progress_events_for_multi_step(monkeypatch, tools):
    patch_intent(monkeypatch, intent("Open WhatsApp and send Nithesh 'hi'", groups=["messaging"]))
    install(monkeypatch, FakeLLM([call("open_app", name="whatsapp"),
                                  call("send_message", app="whatsapp", contact="Nithesh", text="hi"), text("Sent.")]))
    events = [e async for e in brain.stream("open whatsapp and msg nithesh hi", "u-progress", stream_tokens=False)]
    progress = [e["text"] for e in events if e["type"] == "progress"]
    assert progress == ["On it.", "Opened WhatsApp.", "Message sent to Nithesh."]
    assert all(len(p) <= 60 for p in progress)


async def test_no_progress_for_single_step(monkeypatch, tools):
    patch_intent(monkeypatch, intent("Launch WhatsApp from the Start menu", single=True))
    install(monkeypatch, FakeLLM([call("open_app", name="whatsapp")]))
    events = [e async for e in brain.stream("whatsapp kholo", "u-noprogress", stream_tokens=False)]
    assert not any(e["type"] == "progress" for e in events)


async def test_action_uses_action_model(monkeypatch, tools):
    fake = install(monkeypatch, FakeLLM([call("open_app", name="whatsapp"), text("Hi!")]))
    patch_intent(monkeypatch, intent("Launch WhatsApp from the Start menu", single=True))
    await brain.handle("whatsapp kholo", "u-model-action")
    patch_intent(monkeypatch, intent("hello there", kind="chat"))
    await brain.handle("hello there", "u-model-chat")
    assert fake.calls[0]["model"] == settings.ACTION_MODEL
    assert fake.calls[1]["model"] == settings.DEFAULT_MODEL


async def test_intent_failure_falls_back_to_legacy(monkeypatch, tools):
    patch_intent(monkeypatch, lambda message, turns, task: legacy_intent(message))
    fake = install(monkeypatch, FakeLLM([call("open_app", name="chrome")]))
    res = await brain.handle("open chrome and then go to github", "u-legacy")
    assert "open_app" in fake.offered(0) and side_effect_names(fake.offered(0))
    tools["open_app"].assert_awaited_once()
    assert len(fake.calls) == 1 and res.status == "ok"  # legacy actions keep the single-step shortcut


async def test_intent_and_memory_run_concurrently(monkeypatch):
    """analyze waits for the memory lookup to have started: a sequential pipeline would time out."""
    started = asyncio.Event()

    async def memory(self, query, is_command, session_id=None):
        started.set()
        await asyncio.sleep(0)
        return {}, []

    async def slow_analyze(message, turns, task):
        await asyncio.wait_for(started.wait(), 2)
        return intent(message, kind="chat")

    monkeypatch.setattr(pipeline.Brain, "_memory_context", memory)
    monkeypatch.setattr(pipeline, "analyze", slow_analyze)
    install(monkeypatch, FakeLLM([text("hi")]))
    res = await brain.handle("good evening jarvis how are you", "u-concurrent")
    assert res.response == "hi"


async def test_intent_sees_recent_turns_with_notes(monkeypatch, tools):
    patch_intent(monkeypatch, intent("Launch WhatsApp from the Start menu", single=True))
    install(monkeypatch, FakeLLM([call("open_app", name="whatsapp"), text("ok")]))
    await brain.handle("whatsapp kholo", "u-turns-1")
    analyze = patch_intent(monkeypatch, intent("Send hi to Nithesh on WhatsApp", continues=True, groups=["messaging"]))
    await brain.handle("next do", "u-turns-2")
    turns = analyze.await_args.args[1]
    assert turns and turns[-1].assistant.endswith("[did: open_app(whatsapp) ✓]")


# ── Router / selection / prompt units ───────────────────────────────────

def test_context_words_skip_fast_path_only_mid_task():
    assert router.fast_route("next").tool == "media_control"
    for word in ("next", "previous", "again", "back", "continue", "do it", "Next.", "jarvis, next"):
        assert router.fast_route(word, task_active=True) is None, word
    assert router.fast_route("volume up", task_active=True).tool == "set_volume"
    assert "go on" in router.CONTEXT_WORDS and "next" in router.CONTEXT_WORDS


@pytest.mark.parametrize("utterance,rule_name", [
    ("stop", "stop"), ("emergency stop", "stop"), ("resume automation", "resume"),
    ("enable dry run", "dry_run_on"), ("disable dry run", "dry_run_off"), ("automation status", "automation_status"),
])
def test_safety_rules_win_mid_task(utterance, rule_name):
    d = router.fast_route(utterance, task_active=True)
    assert d is not None and d.rule == rule_name


def test_select_tools_with_groups_skips_keywords():
    reg = load_builtin_tools()
    names = [t.name for t in select_tools("next do", reg.all(), groups=["messaging"])]
    assert "send_message" in names and "task_plan" in names and "chat_history" in names
    assert set(n for n in CORE if reg.get(n)) <= set(names)
    assert "set_volume" not in names  # system group not requested, no keyword matching
    assert all(n in names for n in GROUPS["messaging"]["tools"] if reg.get(n))
    # None → keyword behaviour unchanged
    legacy = [t.name for t in select_tools("turn the volume up", reg.all())]
    assert "set_volume" in legacy


def test_select_tools_groups_respect_read_only_pool():
    reg = load_builtin_tools()
    names = [t.name for t in select_tools("send hi", reg.read_only(), groups=["messaging"])]
    assert "send_message" not in names


def test_agent_rules_act_dont_ask():
    assert "If you are not sure" not in AGENT_RULES
    assert "task_plan" in AGENT_RULES and "last resort" in AGENT_RULES
    assert "<untrusted_data>" in AGENT_RULES and "Never claim" in AGENT_RULES
    prompt = build_system_prompt(personality=None, channel="text", facts={}, memories=[], model_label="x",
                                 task_summary="Task: Send hi [active]")
    assert "CURRENT TASK" in prompt and "Task: Send hi [active]" in prompt
    assert "CURRENT TASK" not in build_system_prompt(personality=None, channel="text", facts={}, memories=[],
                                                     model_label="x")
