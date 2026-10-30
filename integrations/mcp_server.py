"""Expose JARVIS as an MCP server (official MCP Python SDK, MIT).

Run:  python -m integrations.mcp_server            (stdio, for Claude Desktop / Cursor / VS Code)

Safety: MCP clients are other programs/agents, so they get the same gateway as
everyone else *and* can never confirm a pending action — MEDIUM/HIGH actions they
request wait for the human to approve them in the dashboard, chat or API.
"""

from __future__ import annotations

from typing import Any, Dict

from automation.gateway import gateway
from brain.pipeline import brain
from tools import ToolContext, load_builtin_tools

SESSION = "mcp"


def build_server():
    from mcp.server.mcpserver import MCPServer

    registry = load_builtin_tools()
    server = MCPServer(
        name="jarvis",
        instructions=(
            "JARVIS controls the user's Windows PC. Actions that change anything are previewed and must be approved "
            "by the user in JARVIS; you cannot approve them yourself."
        ),
    )
    ctx = ToolContext(session_id=SESSION, channel="mcp", source="mcp")

    async def _tool(tool_name: str, **kwargs: Any) -> str:
        tool = registry.get(tool_name)
        assert tool is not None
        res = await tool.run(ctx, **kwargs)
        out = res.message
        if res.content_for_llm:
            out += "\n\n" + res.content_for_llm
        if res.status == "needs_confirmation":
            out += "\n\n(Waiting for the user to approve this in JARVIS.)"
        return out

    @server.tool(description="Ask JARVIS anything or give it a command in plain language.")
    async def jarvis(message: str) -> str:
        result = await brain.handle(message, SESSION, channel="api", allow_confirm=False)
        return result.response

    @server.tool(description="Current CPU, RAM, disk, battery and temperature of the PC.")
    async def system_status(focus: str = "") -> str:
        return await _tool("system_status", focus=focus)

    @server.tool(description="Search the web (DuckDuckGo).")
    async def web_search(query: str) -> str:
        return await _tool("web_search", query=query)

    @server.tool(description="List files in a folder (e.g. 'desktop', 'downloads').")
    async def list_files(location: str) -> str:
        return await _tool("files_list", location=location)

    @server.tool(description="Read a text/PDF/DOCX file.")
    async def read_file(location: str) -> str:
        return await _tool("files_read", location=location)

    @server.tool(description="What JARVIS remembers about the user.")
    async def recall_memory(topic: str = "") -> str:
        return await _tool("memory_recall", topic=topic)

    @server.tool(description="Open an installed application.")
    async def open_app(name: str) -> str:
        return await _tool("open_app", name=name)

    @server.tool(description="Open a website in the browser.")
    async def open_website(url: str) -> str:
        return await _tool("open_website", url=url)

    @server.tool(description="Set the system volume (0-100).")
    async def set_volume(level: int) -> str:
        return await _tool("set_volume", action="set", level=level)

    @server.tool(description="List actions waiting for the user's approval.")
    async def pending_actions() -> Dict[str, Any]:
        return {"pending": [p.to_dict() for p in gateway.list_pending()]}

    return server


def main() -> None:
    build_server().run("stdio")


if __name__ == "__main__":
    main()
