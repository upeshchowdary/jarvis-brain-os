"""Gateway operation for calling tools on external MCP servers."""

from __future__ import annotations

import json
from typing import Any, Dict

from automation.gateway import OpResult, Preview, Risk, gateway


def _risk(args: Dict[str, Any]) -> Risk:
    from integrations.mcp_client import manager

    return manager.risk_for(str(args.get("server", "")), str(args.get("tool", "")))


def _preview(args: Dict[str, Any]) -> Preview:
    arg_text = json.dumps(args.get("arguments") or {}, ensure_ascii=False)[:400]
    return Preview(f"Call {args.get('server')}.{args.get('tool')} with {arg_text}")


@gateway.operation("mcp.call", _risk, "Call a tool on an external MCP server", preview=_preview)
async def mcp_call(server: str, tool: str, arguments: Dict[str, Any] | None = None) -> OpResult:
    from integrations.mcp_client import manager

    try:
        out = await manager.call(server, tool, arguments or {})
    except Exception as exc:
        return OpResult.fail(f"{server}.{tool} failed: {exc}")
    return OpResult(not out.startswith("Error from MCP tool"), out[:4000])
