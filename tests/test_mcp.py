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

    monkeypatch.setattr(sysops, "_IS_WINDOWS", True)
    spy = MagicMock(return_value=True)
    monkeypatch.setattr(sysops.impl, "restart_system", spy)
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
