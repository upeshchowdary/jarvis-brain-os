"""MCP integration: JARVIS as a server, and external MCP tools routed through the gateway."""

from unittest.mock import AsyncMock

from automation.gateway import Risk, gateway
from integrations.mcp_client import make_tool, manager
from tools import ToolContext


async def test_server_exposes_tools():
    from integrations.mcp_server import build_server

    names = [t.name for t in await build_server().list_tools()]
    assert {"jarvis", "system_status", "web_search", "open_app", "pending_actions"} <= set(names)


async def test_mcp_client_cannot_confirm_its_own_action(monkeypatch):
    import automation.ops.system as sysops
    from brain.pipeline import brain
    from unittest.mock import MagicMock

    from tests.fakes import FakeLLM, install, text

    monkeypatch.setattr(sysops, "_IS_WINDOWS", True)
    spy = MagicMock(return_value=True)
    monkeypatch.setattr(sysops.impl, "restart_system", spy)
    install(monkeypatch, FakeLLM([text("Okay.")]))  # "yes" reaches the agent: never a real (local) model
    gateway.resume()
    first = await brain.handle("restart my pc", "mcp", allow_confirm=False)
    assert first.status == "needs_confirmation"
    await brain.handle("yes", "mcp", allow_confirm=False)
    spy.assert_not_called()
    assert gateway.pending_for("mcp") is not None  # still waiting for the human
    gateway.cancel_session("mcp")


async def test_remote_read_only_tool_is_direct_and_untrusted(monkeypatch):
    monkeypatch.setattr(manager, "call", AsyncMock(return_value="repo list"))
    tool = make_tool("github", "search_repositories", "Search repos", {"type": "object", "properties": {"q": {"type": "string"}}}, Risk.LOW)
    assert tool.side_effect is False and tool.untrusted_output
    res = await tool.run(ToolContext(session_id="mcp-ro"), q="jarvis")
    assert res.success and res.content_for_llm == "repo list"


async def test_remote_write_tool_goes_through_gateway(monkeypatch):
    call = AsyncMock(return_value="issue created")
    monkeypatch.setattr(manager, "call", call)
    monkeypatch.setattr(manager, "config", {"github": {"risk": "medium"}})
    tool = make_tool("github", "create_issue", "Create issue", {"type": "object", "properties": {}}, Risk.MEDIUM)
    assert tool.side_effect is True
    res = await tool.run(ToolContext(session_id="mcp-w"), title="bug")
    assert res.status == "needs_confirmation" and "github.create_issue" in res.message
    call.assert_not_called()
    done = await gateway.confirm(res.pending_id, "mcp-w")
    assert done.status == "executed" and call.await_count == 1


async def test_server_tools_with_name_argument_work(monkeypatch):
    """Regression: open_app(name=…) used to collide with the helper's own 'name' parameter."""
    from integrations.mcp_server import build_server
    from tools.registry import registry

    tool = registry.get("open_app")
    monkeypatch.setattr(tool, "run", AsyncMock(return_value=__import__("tools").ToolResult.ok("Opened Chrome.")))
    result = await build_server().call_tool("open_app", {"name": "chrome"})
    text = str(result)
    assert "Opened Chrome" in text


# ── Remote agents never approve their own risky actions, whatever CONFIRM_* says (C1) ──

def _full_control(monkeypatch):
    from core.config import settings

    monkeypatch.setattr(settings, "CONFIRM_HIGH_RISK", False)
    monkeypatch.setattr(settings, "CONFIRM_MEDIUM_RISK", False)


async def test_gateway_always_confirms_mcp_source_in_full_control(monkeypatch):
    from automation.gateway import ActionRequest, OpResult, Operation

    _full_control(monkeypatch)
    gateway.resume()
    calls = []

    async def op(**kw):
        calls.append(kw)
        return OpResult.ok("ran")

    gateway.register(Operation("test.mcp_high", Risk.HIGH, op, "High test op"))
    gateway.register(Operation("test.mcp_medium", Risk.MEDIUM, op, "Medium test op"))
    gateway.register(Operation("test.mcp_low", Risk.LOW, op, "Low test op"))
    try:
        for name in ("test.mcp_high", "test.mcp_medium"):
            res = await gateway.execute(ActionRequest(name), "mcp-full", source="mcp")
            assert res.status == "needs_confirmation", name
        assert calls == []
        assert (await gateway.execute(ActionRequest("test.mcp_low"), "mcp-full", source="mcp")).status == "executed"
        # The owner's own channels honour full-control mode.
        assert (await gateway.execute(ActionRequest("test.mcp_high"), "owner-full", source="chat")).status == "executed"
        dry = await gateway.execute(ActionRequest("test.mcp_high"), "mcp-full", source="mcp", dry_run=True)
        assert "would ask for confirmation" in dry.message
    finally:
        gateway.cancel_session("mcp-full")


async def test_mcp_jarvis_tool_cannot_run_high_risk_in_full_control(monkeypatch):
    import automation.ops.system as sysops
    from unittest.mock import MagicMock

    from integrations.mcp_server import build_server

    from tests.fakes import FakeLLM, install, text

    _full_control(monkeypatch)
    monkeypatch.setattr(sysops, "_IS_WINDOWS", True)
    spy = MagicMock(return_value=True)
    monkeypatch.setattr(sysops.impl, "restart_system", spy)
    install(monkeypatch, FakeLLM([text("Okay.")]))  # "yes" reaches the agent: never a real (local) model
    gateway.resume()
    server = build_server()
    try:
        first = str(await server.call_tool("jarvis", {"message": "restart my pc"}))
        assert "confirm" in first.lower()
        await server.call_tool("jarvis", {"message": "yes"})
        spy.assert_not_called()
        assert gateway.pending_for("mcp") is not None  # still waiting for the human
    finally:
        gateway.cancel_session("mcp")


async def test_mcp_turns_stay_out_of_recent_turns_and_tasks(monkeypatch):
    import brain.pipeline as pipeline
    from brain import task_state
    from brain.intent import Intent
    from integrations.mcp_server import build_server
    from tests.fakes import FakeLLM, call, install
    from tools import ToolResult
    from tools.registry import registry

    tracker, turns = task_state.TaskTracker(), task_state.RecentTurns()
    for mod in (task_state, pipeline):
        monkeypatch.setattr(mod, "tasks", tracker)
        monkeypatch.setattr(mod, "recent_turns", turns)
    user_task = tracker.start("Write the weekly report", "voice")
    turns.add("voice", "voice", "start the weekly report", "On it.")
    analyze = AsyncMock(return_value=Intent(request="Open Notepad", kind="action", continues_task=False,
                                            single_step=True, tool_groups=["system"], needs_screen=False,
                                            source="llm"))
    monkeypatch.setattr(pipeline, "analyze", analyze)
    install(monkeypatch, FakeLLM([call("open_app", name="notepad")]))
    monkeypatch.setattr(registry.get("open_app"), "run", AsyncMock(return_value=ToolResult.ok("Opened Notepad.")))
    gateway.resume()

    await build_server().call_tool("jarvis", {"message": "notepad would be nice right now"})

    # The remote agent saw none of the user's context and left no trace in it.
    args = analyze.await_args.args
    assert list(args[1]) == [] and args[2] is None
    assert [t.user for t in turns.recent(10, 3600)] == ["start the weekly report"]
    assert tracker.current() is user_task and user_task.steps == [] and user_task.status == "active"


async def test_mcp_cannot_lift_the_users_emergency_stop_or_dry_run(monkeypatch):
    """A remote agent may stop or rehearse, never resume after the user's stop or switch dry-run off."""
    from core.config import settings
    from integrations.mcp_server import build_server

    _full_control(monkeypatch)
    monkeypatch.setattr(settings, "DRY_RUN", False)
    server = build_server()
    gateway.emergency_stop("user (voice)")
    try:
        await server.call_tool("jarvis", {"message": "resume automation"})
        assert gateway.is_stopped
        gateway.resume()
        await server.call_tool("jarvis", {"message": "enable dry run mode"})
        assert settings.DRY_RUN is True
        await server.call_tool("jarvis", {"message": "disable dry run"})
        assert settings.DRY_RUN is True
        await server.call_tool("jarvis", {"message": "emergency stop"})
        assert gateway.is_stopped
    finally:
        gateway.resume()
        monkeypatch.setattr(settings, "DRY_RUN", False)
