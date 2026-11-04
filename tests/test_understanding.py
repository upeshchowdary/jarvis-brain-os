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
from tests.fakes import FakeLLM, call, calls, install, text
from tools.base import ToolResult
from tools.registry import load_builtin_tools, registry
from tools.selection import CORE, GROUP_CORE, GROUPS, select_tools


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
    # P1: once both steps succeeded the turn ends without a closing round; the last result is the reply.
    assert res.response == "Message sent to Nithesh. It may take a moment." and len(fake.calls) == 2


async def test_single_step_keeps_shortcut(monkeypatch, tools):
    patch_intent(monkeypatch, intent("Launch WhatsApp from the Start menu", single=True, groups=["system"]))
    fake = install(monkeypatch, FakeLLM([call("open_app", name="whatsapp"), text("should not be reached")]))
    res = await brain.handle("whatsapp kholo yaar", "u-single")
    assert len(fake.calls) == 1 and res.response == "Opened WhatsApp."


async def test_action_step_budget(monkeypatch, tools):
    monkeypatch.setattr(settings, "AGENT_MAX_STEPS_ACTION", 7)
    tools["open_app"].return_value = ToolResult.error("Couldn't open WhatsApp.")  # failing steps keep the loop going
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
    assert "send_message" in offered and "task_plan" in offered
    assert "chat_history" not in offered  # P1: only with the history group


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

    async def memory(self, query, is_command, session_id=None, remote=False):
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
    assert "send_message" in names
    assert set(n for n in GROUP_CORE if reg.get(n)) <= set(names)  # P1: a slim core, not all of CORE
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


# ── Fix round 1 ─────────────────────────────────────────────────────────

async def test_clarify_ends_multi_step_turn_and_waits(monkeypatch, tools, fresh_state):
    patch_intent(monkeypatch, intent("Send Nithesh a WhatsApp message", groups=["messaging"]))
    fake = install(monkeypatch, FakeLLM([call("clarify", question="What should the message say?"),
                                         text("should not be reached")]))
    res = await brain.handle("msg nithesh on whatsapp", "u-clarify")
    assert res.response == "What should the message say?" and len(fake.calls) == 1
    task = fresh_state.current()
    assert task is not None and task.status == "waiting_for_user" and task.steps == []

    # The answer continues the same task, which then completes.
    patch_intent(monkeypatch, intent("Send Nithesh 'hi' on WhatsApp", continues=True, groups=["messaging"]))
    install(monkeypatch, FakeLLM([call("send_message", app="whatsapp", contact="Nithesh", text="hi"), text("Sent.")]))
    await brain.handle("say hi", "u-clarify")
    tools["send_message"].assert_awaited_once()
    assert fresh_state.current() is None and fresh_state._last is task
    assert task.status == "done" and [s.tool for s in task.steps] == ["send_message"]


async def test_clarify_ends_question_turn(monkeypatch):
    patch_intent(monkeypatch, intent("Which city's weather?", kind="question"))
    fake = install(monkeypatch, FakeLLM([call("clarify", question="Which city?"), text("never")]))
    res = await brain.handle("how is it out there", "u-clarify-q")
    assert res.response == "Which city?" and len(fake.calls) == 1


def test_clarify_description_is_last_resort():
    assert "last resort" in load_builtin_tools().get("clarify").description.lower()


@pytest.fixture
def confirmable(monkeypatch, tools):
    """send_message queues a confirmation; the gateway's pending/confirm/cancel are faked."""
    from types import SimpleNamespace

    from automation.gateway import GatewayResult

    box = {"pending": None}
    queued = ToolResult(True, "This needs your confirmation: send 'hi' to Nithesh.",
                        status="needs_confirmation", pending_id="pid-1")

    async def run(ctx, **kw):
        box["pending"] = SimpleNamespace(pending_id="pid-1", actions=[])
        return queued

    tools["send_message"].side_effect = run
    monkeypatch.setattr(gateway, "pending_for", lambda sid: box["pending"])
    confirm = AsyncMock(return_value=GatewayResult("executed", "Message sent to Nithesh."))

    async def do_confirm(pid, sid=None):
        box["pending"] = None
        return await confirm(pid, sid)

    monkeypatch.setattr(gateway, "confirm", do_confirm)
    monkeypatch.setattr(gateway, "cancel", lambda pid, reason="": box.update(pending=None))
    return confirm


async def test_confirmed_action_is_booked_and_closes_task(monkeypatch, confirmable, fresh_state):
    patch_intent(monkeypatch, intent("Send Nithesh 'hi' on WhatsApp", groups=["messaging"]))
    install(monkeypatch, FakeLLM([call("send_message", app="whatsapp", contact="Nithesh", text="hi")]))
    sid = "u-confirm"
    first = await brain.handle("msg nithesh hi", sid)
    assert first.status == "needs_confirmation"
    assert conversation_manager.get_history(sid)[-1]["content"].endswith("[did: send_message(whatsapp) …]")
    task = fresh_state.current()
    assert task.status == "active" and task.steps[-1].mark == "…" and "… send_message(" in task.summary()

    second = await brain.handle("yes", sid)
    assert second.route == "confirmation" and second.status == "ok"
    confirmable.assert_awaited_once()
    assert conversation_manager.get_history(sid)[-1]["content"].endswith("[did: send_message(whatsapp) ✓]")
    assert fresh_state.current() is None and task.status == "done" and task.steps[-1].mark == "✓"


async def test_cancelled_confirmation_is_recorded(monkeypatch, confirmable, fresh_state):
    patch_intent(monkeypatch, intent("Send Nithesh 'hi' on WhatsApp", groups=["messaging"]))
    install(monkeypatch, FakeLLM([call("send_message", app="whatsapp", contact="Nithesh", text="hi")]))
    sid = "u-cancel"
    await brain.handle("msg nithesh hi", sid)
    task = fresh_state.current()
    res = await brain.handle("no", sid)
    assert res.route == "confirmation" and "cancelled" in res.response.lower()
    confirmable.assert_not_awaited()
    assert conversation_manager.get_history(sid)[-1]["content"].endswith("[did: send_message(whatsapp) (cancelled)]")
    assert fresh_state.current() is task and task.steps[-1].mark == "(cancelled)"


def test_note_marks_for_pending_dry_run_cancelled():
    from tools.base import ToolCallRecord

    recs = [ToolCallRecord("power_control", {"action": "restart"}, ToolResult(True, "x", status="needs_confirmation")),
            ToolCallRecord("file_delete", {"path": "a.txt"}, ToolResult(True, "x", status="dry_run")),
            ToolCallRecord("send_message", {"app": "whatsapp"}, ToolResult(False, "x", status="cancelled")),
            ToolCallRecord("kill_process", {"target": "x"}, ToolResult(False, "x", status="blocked"))]
    assert pipeline._action_notes(recs) == (" [did: power_control(restart) …, file_delete(a.txt) (dry run), "
                                            "send_message(whatsapp) (cancelled), kill_process(x) ✗]")
    assert "awaiting the user's confirmation" in AGENT_RULES and "(dry run)" in AGENT_RULES


async def test_legacy_context_word_continues_current_task(monkeypatch, tools, fresh_state):
    task = fresh_state.start("Open WhatsApp and message Nithesh 'hi'", "voice")
    patch_intent(monkeypatch, lambda message, turns, t: legacy_intent(message))
    fake = install(monkeypatch, FakeLLM([text("Sending it now.")]))
    await brain.handle("next", "u-legacy-next")
    tools["media_control"].assert_not_awaited()
    assert fresh_state.current() is task and task.goal == "Open WhatsApp and message Nithesh 'hi'"
    assert "Task: Open WhatsApp and message Nithesh 'hi'" in fake.calls[0]["messages"][0]["content"]


async def test_legacy_message_reactivates_waiting_task(monkeypatch, tools, fresh_state):
    task = fresh_state.start("Send Nithesh a WhatsApp message", "text")
    fresh_state.set_status("waiting_for_user")
    patch_intent(monkeypatch, lambda message, turns, t: legacy_intent(message))
    install(monkeypatch, FakeLLM([text("ok")]))
    await brain.handle("send hi", "u-legacy-wait")
    assert fresh_state.current() is task and task.status == "active"


async def test_emergency_stop_breaks_multi_step_loop(monkeypatch, tools, fresh_state):
    async def open_and_stop(ctx, **kw):
        gateway.emergency_stop("test")
        return ToolResult.ok("Opened WhatsApp.")

    tools["open_app"].side_effect = open_and_stop
    patch_intent(monkeypatch, intent("Open WhatsApp and send Nithesh 'hi'", groups=["messaging"]))
    fake = install(monkeypatch, FakeLLM([call("open_app", name="whatsapp"),
                                         call("send_message", app="whatsapp", contact="Nithesh", text="hi"),
                                         text("Sent.")]))
    res = await brain.handle("open whatsapp and msg nithesh hi", "u-stop-loop")
    assert len(fake.calls) == 1 and res.response == "Stopped."
    tools["send_message"].assert_not_awaited()
    assert fresh_state.current() is None and fresh_state._last.status == "shelved"


async def test_action_while_stopped_does_not_call_llm_but_questions_still_work(monkeypatch, tools):
    gateway.emergency_stop("test")
    patch_intent(monkeypatch, intent("Open WhatsApp and send Nithesh 'hi'", groups=["messaging"]))
    fake = install(monkeypatch, FakeLLM([text("Paris.")]))
    res = await brain.handle("open whatsapp and msg nithesh hi", "u-stopped-before")
    assert fake.calls == [] and "resume" in res.response
    patch_intent(monkeypatch, intent("What is the capital of France?", kind="question"))
    res = await brain.handle("what is the capital of france", "u-stopped-q")
    assert res.response == "Paris."


def test_task_plan_survives_tool_cap():
    from tools.selection import MAX_TOOLS

    reg = load_builtin_tools()
    names = [t.name for t in select_tools("x", reg.all(), groups=list(GROUPS), include=("task_plan",))]
    assert len(names) == MAX_TOOLS and "task_plan" in names
    every_keyword = " ".join(k for g in GROUPS.values() for k in g["keywords"])
    names = [t.name for t in select_tools(every_keyword, reg.all(), include=("task_plan",))]
    assert len(names) == MAX_TOOLS and "task_plan" in names


async def test_task_plan_offered_for_every_action(monkeypatch, tools):
    patch_intent(monkeypatch, intent("Do everything", groups=list(GROUPS)))
    fake = install(monkeypatch, FakeLLM([text("ok")]))
    await brain.handle("do everything", "u-plan-groups")
    assert "task_plan" in fake.offered(0)
    patch_intent(monkeypatch, lambda message, turns, t: legacy_intent(message))
    fake = install(monkeypatch, FakeLLM([text("ok")]))
    await brain.handle("open chrome and then go to github and send the file and play music", "u-plan-legacy")
    assert "task_plan" in fake.offered(0)


# ── Fix round 2 ─────────────────────────────────────────────────────────

async def test_prefilled_fast_path_does_not_close_unrelated_task(monkeypatch, fresh_state):
    """'run the tests' (fast path, non-final) mid-WhatsApp-task must not finish the WhatsApp task."""
    task = fresh_state.start("Open WhatsApp and message Nithesh 'hi'", "voice")
    monkeypatch.setattr(load_builtin_tools().get("run_tests"), "run", AsyncMock(return_value=ToolResult.ok("3 passed.")))
    analyze = patch_intent(monkeypatch, intent("unused"))
    fake = install(monkeypatch, FakeLLM([text("All 3 tests passed.")]))
    res = await brain.handle("run the tests", "u-prefilled")
    assert res.response == "All 3 tests passed." and len(fake.calls) == 1
    analyze.assert_not_awaited()
    assert fresh_state.current() is task and task.status == "active"


async def test_fallback_success_does_not_mark_task_done(monkeypatch, tools, fresh_state):
    task = fresh_state.start("Open WhatsApp and message Nithesh 'hi'", "voice")
    patch_intent(monkeypatch, lambda message, turns, t: legacy_intent(message))
    install(monkeypatch, FakeLLM([call("open_app", name="chrome")]))
    await brain.handle("open chrome and then go to github", "u-fallback")
    tools["open_app"].assert_awaited_once()
    assert fresh_state.current() is task and task.status == "active"
    assert [s.tool for s in task.steps] == ["open_app"]  # booked, but the task is not closed


async def test_stop_on_rewritten_fast_path_shelves_task(monkeypatch, fresh_state):
    gateway.emergency_stop("test")
    patch_intent(monkeypatch, intent("open whatsapp", single=True))
    fake = install(monkeypatch, FakeLLM([]))
    res = await brain.handle("yes do that one", "u-stop-rewrite")
    assert res.route == "fast_path" and res.status != "ok" and fake.calls == []
    assert fresh_state.current() is None and fresh_state._last.goal == "open whatsapp"
    assert fresh_state._last.status == "shelved"


async def test_fast_path_stop_shelves_active_task(monkeypatch, fresh_state):
    task = fresh_state.start("Open WhatsApp and message Nithesh 'hi'", "voice")
    patch_intent(monkeypatch, intent("unused"))
    await brain.handle("stop", "u-stop-shelve")
    assert gateway.is_stopped and fresh_state.current() is None and task.status == "shelved"
    gateway.resume()
    assert fresh_state.revive_last() is task


async def test_timing_breakdown_has_intent_and_memory(monkeypatch, tools):
    patch_intent(monkeypatch, intent("hello there", kind="chat"))
    install(monkeypatch, FakeLLM([text("hi")]))
    res = await brain.handle("hello there", "u-timing")
    timing = res.metadata["timing_breakdown_ms"]
    for key in ("intent_ms", "memory_ms", "llm_ms"):
        assert isinstance(timing[key], (int, float)), key
    assert "turn.intent" in pipeline.metrics.snapshot()["latency"]


async def test_interleaved_turns_do_not_corrupt_each_others_task(monkeypatch, tools, fresh_state):
    """I3: turn A's later steps / close must not land on task B that a concurrent turn started meanwhile."""
    a_waiting, b_started, a_done = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def fake_analyze(query, turns, task):
        if "alpha" in query:
            return intent("Alpha: open WhatsApp and send A 'hi'", groups=["messaging", "system"])
        return intent("Beta: open the site", groups=["web"])

    class GatedLLM(FakeLLM):
        async def chat(self, messages, *, model=None, tools=None, local_only=False, **kw):
            self.calls.append({"messages": messages, "tools": [], "model": model, "local_only": local_only, "kwargs": kw})
            user = [m for m in messages if m["role"] == "user"][-1]["content"]
            done = sum(1 for m in messages if m["role"] == "tool")
            if user.startswith("Alpha"):
                if done == 0:
                    return call("open_app", name="whatsapp")
                if done == 1:
                    a_waiting.set()
                    await b_started.wait()  # turn B starts its own task now
                    return call("task_plan", action="set", subgoals=["Open WhatsApp", "Send"])
                if done == 2:
                    return call("send_message", app="whatsapp", contact="A", text="hi")
                return text("A done.")
            b_started.set()
            await a_done.wait()  # B is still running while A finishes
            return text("B is thinking.")

    monkeypatch.setattr(pipeline, "analyze", fake_analyze)
    install(monkeypatch, GatedLLM([]))

    async def run_a():
        try:
            return await brain.handle("alpha go", "u-a")
        finally:
            a_done.set()

    async def run_b():
        await a_waiting.wait()  # A has begun its task and booked a step
        return await brain.handle("beta go", "u-b")

    await asyncio.gather(run_a(), run_b())
    current = fresh_state.current()
    assert current is not None and current.goal.startswith("Beta")
    assert current.steps == [] and current.plan == [] and current.status == "active"


# ── P1 item 1: fewer agent rounds ───────────────────────────────────────

@pytest.fixture
def typing(monkeypatch, tools):
    tools["open_app"].return_value = ToolResult.ok("Opened Notepad.")
    tools["type_text"] = AsyncMock(return_value=ToolResult.ok("Typed 'testing speed'."))
    monkeypatch.setattr(registry.get("type_text"), "run", tools["type_text"])
    return tools


NOTEPAD = intent("Open Notepad and type 'testing speed'", groups=["keyboard"])


async def test_notepad_plan_and_act_then_finish_takes_two_rounds(monkeypatch, typing, fresh_state):
    patch_intent(monkeypatch, NOTEPAD)
    fake = install(monkeypatch, FakeLLM([
        calls(("task_plan", {"action": "set", "subgoals": ["Open Notepad", "Type 'testing speed'"]}),
              ("open_app", {"name": "notepad"})),
        calls(("task_plan", {"action": "done", "index": 0}), ("type_text", {"text": "testing speed"}),
              ("task_plan", {"action": "done", "index": 1})),
        text("should not be reached"),
    ]))
    res = await brain.handle("open notepad and type testing speed", "u-p1-plan")
    assert len(fake.calls) == 2, [c["tools"] for c in fake.calls]
    typing["open_app"].assert_awaited_once()
    typing["type_text"].assert_awaited_once()
    assert res.response == "Typed 'testing speed'." and res.status == "ok"
    assert fresh_state.current() is None and fresh_state._last.status == "done"


async def test_all_actions_in_one_response_take_one_round(monkeypatch, typing, fresh_state):
    patch_intent(monkeypatch, NOTEPAD)
    fake = install(monkeypatch, FakeLLM([
        calls(("open_app", {"name": "notepad"}), ("type_text", {"text": "testing speed"})),
        text("should not be reached"),
    ]))
    res = await brain.handle("open notepad and type testing speed", "u-p1-batch")
    assert len(fake.calls) == 1
    typing["open_app"].assert_awaited_once()
    typing["type_text"].assert_awaited_once()
    assert res.response == "Opened Notepad.\nTyped 'testing speed'."
    assert fresh_state._last.status == "done"


async def test_two_steps_one_per_round_without_plan_take_two_rounds(monkeypatch, typing):
    patch_intent(monkeypatch, NOTEPAD)
    fake = install(monkeypatch, FakeLLM([call("open_app", name="notepad"), call("type_text", text="testing speed"),
                                         text("should not be reached")]))
    res = await brain.handle("open notepad and type testing speed", "u-p1-two")
    assert len(fake.calls) == 2 and res.response == "Typed 'testing speed'."


async def test_first_step_of_multi_step_request_without_plan_continues(monkeypatch, typing):
    """One step of a request the intent step called multi-step is not the end (no plan says otherwise)."""
    patch_intent(monkeypatch, NOTEPAD)
    fake = install(monkeypatch, FakeLLM([call("open_app", name="notepad"), text("Notepad is open; what text?")]))
    res = await brain.handle("open notepad and type testing speed", "u-p1-first")
    assert len(fake.calls) == 2 and res.response == "Notepad is open; what text?"


async def test_failed_step_still_continues(monkeypatch, typing):
    typing["type_text"].return_value = ToolResult.error("No window to type into.")
    patch_intent(monkeypatch, NOTEPAD)
    fake = install(monkeypatch, FakeLLM([
        calls(("open_app", {"name": "notepad"}), ("type_text", {"text": "testing speed"})),
        text("Notepad opened, but typing failed: no window to type into."),
    ]))
    res = await brain.handle("open notepad and type testing speed", "u-p1-fail")
    assert len(fake.calls) == 2 and res.response.startswith("Notepad opened, but typing failed")


async def test_open_plan_items_keep_the_turn_going(monkeypatch, typing, fresh_state):
    patch_intent(monkeypatch, intent("Open Notepad, type 'a', then save it", groups=["keyboard"]))
    fake = install(monkeypatch, FakeLLM([
        calls(("task_plan", {"action": "set", "subgoals": ["Open Notepad", "Type 'a'", "Save"]}),
              ("open_app", {"name": "notepad"})),
        calls(("task_plan", {"action": "done", "index": 0}), ("type_text", {"text": "a"}),
              ("task_plan", {"action": "done", "index": 1})),
        text("Typed it; saving needs a file name."),
    ]))
    res = await brain.handle("open notepad type a and save it", "u-p1-open-plan")
    assert len(fake.calls) == 3 and res.response == "Typed it; saving needs a file name."
    assert fresh_state.current() is not None and fresh_state.current().status == "active"


async def test_read_tool_in_action_turn_gets_synthesis_round(monkeypatch, tools):
    search = AsyncMock(return_value=ToolResult(True, "3 results.", content_for_llm="1. MrBeast - YouTube"))
    monkeypatch.setattr(registry.get("web_search"), "run", search)
    patch_intent(monkeypatch, intent("Find MrBeast's channel and open it", groups=["web"]))
    fake = install(monkeypatch, FakeLLM([
        calls(("web_search", {"query": "mrbeast youtube channel"}), ("open_website", {"url": "youtube.com"})),
        text("Opened MrBeast's channel."),
    ]))
    res = await brain.handle("find mr beast channel and open it", "u-p1-read")
    assert len(fake.calls) == 2 and res.response == "Opened MrBeast's channel."


async def test_question_with_history_tool_gets_synthesis_round(monkeypatch):
    history = AsyncMock(return_value=ToolResult(True, "2 turns today.", content_for_llm="09:00 open notepad"))
    monkeypatch.setattr(load_builtin_tools().get("chat_history"), "run", history)
    patch_intent(monkeypatch, intent("Read my chats with JARVIS from today and tell me what I asked",
                                     kind="question", groups=["history"]))
    fake = install(monkeypatch, FakeLLM([call("chat_history", period="today"), text("You asked me to open Notepad.")]))
    res = await brain.handle("what did I ask you today?", "u-p1-history")
    assert len(fake.calls) == 2 and res.response == "You asked me to open Notepad."


def test_agent_rules_plan_only_for_three_or_more_steps():
    assert "3 or more" in AGENT_RULES and "task_plan" in AGENT_RULES
    assert "several tools in one response" in AGENT_RULES
    assert all(line.startswith("- ") for line in AGENT_RULES.splitlines()[1:])  # one rule per line


# ── P1 item 3: smaller per-round prompt ─────────────────────────────────

def _names(groups, pool=None, include=()):
    reg = load_builtin_tools()
    return [t.name for t in select_tools("x", pool if pool is not None else reg.all(), groups=groups, include=include)]


def test_groups_offer_slim_core_plus_groups():
    reg = load_builtin_tools()
    names = _names(["keyboard"])
    expected = {n for n in GROUP_CORE if reg.get(n)} | set(GROUPS["keyboard"]["tools"])
    assert set(n for n in names if not n.startswith("mcp_")) == expected
    for dropped in ("memory_recall", "memory_remember", "system_status", "automation_status", "search_in_browser",
                    "chat_history", "task_plan"):
        assert dropped not in names, dropped
    assert {"clarify", "open_app"} <= set(names)


def test_core_tools_come_back_with_their_group():
    assert {"system_status", "automation_status"} <= set(_names(["system"]))
    assert {"web_search", "search_in_browser", "open_website"} <= set(_names(["web"]))
    assert {"memory_recall", "memory_remember"} <= set(_names(["memory"]))
    assert "chat_history" in _names(["history"])
    assert "task_plan" in _names(["keyboard"], include=("task_plan",))


def test_system_and_keyboard_groups_both_fit():
    """Before P1 the full CORE + extras + 17 system tools pushed the keyboard group past MAX_TOOLS."""
    names = _names(["system", "keyboard"], include=("task_plan",))
    assert {"type_text", "press_keys", "focus_window", "open_app", "task_plan"} <= set(names)


def test_keyword_path_keeps_full_core():
    reg = load_builtin_tools()
    names = [t.name for t in select_tools("open notepad and type hi", reg.all())]
    assert set(n for n in CORE if reg.get(n)) <= set(names) and "type_text" in names


async def test_single_step_action_with_groups_gets_no_task_plan(monkeypatch, tools):
    patch_intent(monkeypatch, intent("Launch WhatsApp from the Start menu", single=True, groups=["system"]))
    fake = install(monkeypatch, FakeLLM([call("open_app", name="whatsapp")]))
    await brain.handle("whatsapp kholo", "u-p1-noplan")
    assert "task_plan" not in fake.offered(0) and "open_app" in fake.offered(0)


async def test_multi_step_action_with_groups_gets_task_plan(monkeypatch, typing):
    patch_intent(monkeypatch, intent("Open Notepad and type 'testing speed'", groups=["system", "keyboard"]))
    fake = install(monkeypatch, FakeLLM([text("ok")]))
    await brain.handle("open notepad and type testing speed", "u-p1-plan-offered")
    offered = fake.offered(0)
    assert "task_plan" in offered and "type_text" in offered and "open_app" in offered
    assert "memory_remember" not in offered and "chat_history" not in offered


async def test_legacy_single_step_action_still_gets_task_plan(monkeypatch, tools):
    patch_intent(monkeypatch, lambda message, turns, t: legacy_intent(message))
    fake = install(monkeypatch, FakeLLM([text("ok")]))
    await brain.handle("open chrome and then go to github", "u-p1-legacy-plan")
    assert "task_plan" in fake.offered(0) and set(n for n in CORE if registry.get(n)) <= set(fake.offered(0))
