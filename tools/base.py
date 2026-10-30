"""Tool framework: one tool per capability, each with a JSON schema and a risk level.

Tools are what the router / LLM can call. Read-only tools may run directly; every
tool with ``side_effect=True`` must perform its action through
``automation.gateway`` (use :func:`via_gateway`), so the risk policy, dry-run,
confirmation and audit log always apply.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from automation.gateway import ActionRequest, GatewayResult, Risk, gateway


@dataclass
class ToolContext:
    session_id: str = "default"
    channel: str = "text"  # text | voice | api
    dry_run: Optional[bool] = None
    source: str = "chat"


@dataclass
class ToolResult:
    success: bool
    message: str = ""  # user-facing summary
    data: Any = None
    status: str = "ok"  # ok | needs_confirmation | dry_run | blocked | error
    pending_id: Optional[str] = None
    content_for_llm: Optional[str] = None  # raw content (web page, file text, screen) for synthesis

    @classmethod
    def ok(cls, message: str = "", data: Any = None, content_for_llm: Optional[str] = None) -> "ToolResult":
        return cls(True, message, data, "ok", None, content_for_llm)

    @classmethod
    def error(cls, message: str, data: Any = None) -> "ToolResult":
        return cls(False, message, data, "error")

    def to_dict(self) -> Dict[str, Any]:
        return {"success": self.success, "status": self.status, "message": self.message, "pending_id": self.pending_id}


class Tool:
    name: str = ""
    description: str = ""
    parameters: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}
    risk: Risk = Risk.LOW
    side_effect: bool = False
    # Output contains external/untrusted text (web pages, files, screen OCR).
    untrusted_output: bool = False
    # Output contains the user's private content (files, screen) → local-only LLM in PRIVACY_MODE=local.
    sensitive_output: bool = False
    # The result message is a complete answer; no LLM synthesis step is needed.
    final: bool = True

    async def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:  # pragma: no cover - interface
        raise NotImplementedError

    def schema(self) -> Dict[str, Any]:
        return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}

    def info(self) -> Dict[str, Any]:
        return {"name": self.name, "description": self.description, "risk": self.risk.label, "side_effect": self.side_effect}


def obj(properties: Dict[str, Any], required: Optional[List[str]] = None) -> Dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required or []}


def from_gateway(res: GatewayResult) -> ToolResult:
    status = {"executed": "ok", "dry_run": "dry_run", "needs_confirmation": "needs_confirmation"}.get(res.status, "error" if res.status in ("failed", "expired") else res.status)
    success = res.status in ("executed", "dry_run", "needs_confirmation")
    return ToolResult(success, res.message, {"steps": res.steps, "risk": res.risk}, status, res.pending_id)


async def via_gateway(ctx: ToolContext, action: ActionRequest | List[ActionRequest]) -> ToolResult:
    res = await gateway.execute(action, ctx.session_id, dry_run=ctx.dry_run, source=ctx.source)
    return from_gateway(res)


@dataclass
class ToolCallRecord:
    name: str
    arguments: Dict[str, Any]
    result: ToolResult
    elapsed_ms: float = 0.0
    origin: str = "llm"  # fast | llm

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "arguments": self.arguments, "origin": self.origin, "elapsed_ms": round(self.elapsed_ms, 1), **self.result.to_dict()}


@dataclass
class ToolSet:
    tools: List[Tool] = field(default_factory=list)

    def names(self) -> List[str]:
        return [t.name for t in self.tools]

    def schemas(self) -> List[Dict[str, Any]]:
        return [t.schema() for t in self.tools]
