"""PC-wide current task, cross-channel recent turns, and the task_plan tool."""

import asyncio
import time

import pytest

from app.database.connection import db_manager
from brain import task_state
from brain.task_state import RecentTurns, TaskTracker
from core.config import settings
from tools.base import ToolContext
from tools.registry import registry


def test_start_shelves_previous_task():
    t = TaskTracker()
    a = t.start("A", "voice")
    t.start("B", "text")
    assert t.current().goal == "B"
    assert a.status == "shelved"


def test_idle_task_is_shelved(monkeypatch):
    t = TaskTracker()
    now = time.time()
    monkeypatch.setattr(time, "time", lambda: now)
    t.start("A", "voice")
    monkeypatch.setattr(time, "time", lambda: now + settings.TASK_IDLE_SECONDS + 1)
    assert t.current() is None
    revived = t.revive_last()
    assert revived.goal == "A"
    assert revived.status == "active"
    assert t.current().goal == "A"


def test_record_step_and_summary():
    t = TaskTracker()
    t.start("send a message", "voice")
    t.record_step("app_open", {"name": "WhatsApp"}, "Opened WhatsApp", True)
    t.record_step("type_text", {"text": "hi"}, "Typed", True)
    s = t.current().summary()
    assert "send a message" in s
    assert "app_open" in s and "type_text" in s
    assert "✓" in s


def test_plan_next_unfinished():
    t = TaskTracker()
    t.start("message Nithesh", "voice")
    t.set_plan(["open WhatsApp", "find Nithesh", "send"])
    t.mark_plan(0, "done")
    lines = t.current().summary().splitlines()
    assert any("✓" in ln and "open WhatsApp" in ln for ln in lines)
    assert any("→" in ln and "find Nithesh" in ln for ln in lines)
    assert not any("→" in ln and "send" in ln for ln in lines)


async def test_task_persists_and_reloads():
    await db_manager.init_db()
    t = TaskTracker()
    t.start("persist me", "voice")
    t.record_step("app_open", {"name": "x"}, "ok", True)
    await asyncio.sleep(0.2)  # let the fire-and-forget save run
    t2 = TaskTracker()
    await t2.load()
    cur = t2.current()
    assert cur.goal == "persist me"
    assert len(cur.steps) == 1


def test_mutation_without_event_loop_does_not_raise():
    t = TaskTracker()
    t.start("sync", "voice")  # no running loop here: save is skipped silently
    t.record_step("x", {}, "ok", True)


def test_recent_turns_cross_channel_window():
    r = RecentTurns()
    now = time.time()
    r.add("text", "old", "ancient", "reply")
    r._turns[-1].at = now - 4000
    r.add("voice", "s1", "hello", "hi")
    r._turns[-1].at = now
    r.add("text", "s2", "next", "ok")
    r._turns[-1].at = now + 10
    got = r.recent(8, 1800)
    assert [x.channel for x in got] == ["voice", "text"]
    assert [x.user for x in got] == ["hello", "next"]


async def test_task_plan_tool_sets_and_marks(monkeypatch):
    monkeypatch.setattr(task_state, "tasks", TaskTracker())
    task_state.tasks.start("goal", "text")
    tool = registry.get("task_plan")
    assert tool is not None and not tool.side_effect and not tool.final
    ctx = ToolContext()
    res = await tool.run(ctx, action="set", subgoals=["one", "two"])
    assert res.success and "one" in res.message
    await tool.run(ctx, action="done", index=0)
    assert task_state.tasks.current().plan[0].status == "done"
