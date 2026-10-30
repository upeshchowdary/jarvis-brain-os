"""Consume external MCP servers (GitHub, filesystem, Google Calendar/Gmail …) as JARVIS tools.

Configure ``data/mcp_servers.json``::

    {"servers": {
        "github": {"command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"],
                   "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "…"},
                   "risk": "medium",
                   "read_only_tools": ["search_repositories", "get_file_contents"]}
    }}

Each remote tool becomes ``mcp_<server>_<tool>``. Tools listed in ``read_only_tools``
are called directly (output treated as untrusted); every other tool goes through the
ActionGateway op ``mcp.call`` with the configured risk (default MEDIUM → confirmation).
"""

from __future__ import annotations

import asyncio
import json
import re
from contextlib import AsyncExitStack
from typing import Any, Dict, List, Optional

from automation.gateway import ActionRequest, Risk
from core.logging import logger
from core.paths import data_path
from tools.base import Tool, ToolContext, ToolResult, via_gateway


def load_config() -> Dict[str, Any]:
    try:
        return json.loads(data_path("mcp_servers.json").read_text(encoding="utf-8")).get("servers", {})
    except (OSError, ValueError):
        return {}


def result_text(result: Any) -> str:
    parts = []
    for item in getattr(result, "content", None) or []:
        text = getattr(item, "text", None)
        parts.append(text if text is not None else str(item))
    out = "\n".join(parts).strip()
    if getattr(result, "isError", False):
        return f"Error from MCP tool: {out}"
    return out or "(no output)"


class MCPClientManager:
    def __init__(self) -> None:
        self._stack: Optional[AsyncExitStack] = None
        self.sessions: Dict[str, Any] = {}
        self.config: Dict[str, Any] = {}

    def risk_for(self, server: str, tool: str) -> Risk:
        cfg = self.config.get(server, {})
        if tool in cfg.get("read_only_tools", []):
            return Risk.LOW
        try:
            return Risk[str(cfg.get("risk", "medium")).upper()]
        except KeyError:
            return Risk.MEDIUM

    async def start(self) -> int:
        self.config = load_config()
        if not self.config:
            return 0
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError:
            logger.warning("[MCP] 'mcp' package not installed; skipping MCP servers")
            return 0
        from tools.registry import registry

        self._stack = AsyncExitStack()
        count = 0
        for name, cfg in self.config.items():
            try:
                params = StdioServerParameters(command=cfg["command"], args=cfg.get("args", []), env=cfg.get("env"))
                read, write = await self._stack.enter_async_context(stdio_client(params))
                session = await self._stack.enter_async_context(ClientSession(read, write))
                await asyncio.wait_for(session.initialize(), timeout=30)
                listed = await session.list_tools()
                self.sessions[name] = session
                for t in listed.tools:
                    registry.register(make_tool(name, t.name, t.description or "", t.inputSchema or {}, self.risk_for(name, t.name)))
                    count += 1
                logger.info(f"[MCP] {name}: {len(listed.tools)} tools")
            except Exception as exc:
                logger.warning(f"[MCP] could not start server '{name}': {exc}")
        return count

    async def call(self, server: str, tool: str, arguments: Dict[str, Any]) -> str:
        session = self.sessions.get(server)
        if session is None:
            raise RuntimeError(f"MCP server '{server}' is not connected")
        return result_text(await session.call_tool(tool, arguments or {}))

    async def stop(self) -> None:
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self.sessions.clear()


manager = MCPClientManager()


def make_tool(server: str, tool_name: str, description: str, schema: Dict[str, Any], risk: Risk) -> Tool:
    safe = re.sub(r"[^a-zA-Z0-9_]", "_", f"mcp_{server}_{tool_name}")[:64]
    read_only = risk == Risk.LOW

    class RemoteTool(Tool):
        name = safe
        parameters = schema if schema.get("type") == "object" else {"type": "object", "properties": {}}
        side_effect = not read_only
        untrusted_output = True
        final = False

        async def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
            if read_only:
                try:
                    out = await manager.call(server, tool_name, kwargs)
                except Exception as exc:
                    return ToolResult.error(str(exc))
                return ToolResult.ok(f"{server}.{tool_name} returned data.", content_for_llm=out)
            res = await via_gateway(ctx, ActionRequest("mcp.call", {"server": server, "tool": tool_name, "arguments": kwargs}))
            if res.success and res.status == "ok":
                res.content_for_llm = res.message
            return res

    RemoteTool.description = f"[{server}] {description}"[:1000]
    RemoteTool.risk = risk
    return RemoteTool()


__all__ = ["MCPClientManager", "make_tool", "manager", "result_text", "List"]
