"""Deterministic stand-ins for the LLM (recorded / scripted responses)."""

from __future__ import annotations

from typing import Any, Callable, Dict, List

from core.llm import LLMResponse, ToolCall


def text(content: str) -> LLMResponse:
    return LLMResponse(content=content, provider="fake", model="fake-1")


def call(name: str, **arguments: Any) -> LLMResponse:
    return LLMResponse(content="", provider="fake", model="fake-1", tool_calls=[ToolCall(f"call_{name}", name, arguments)])


class FakeLLM:
    """Replays a script. ``script`` is a list of responses or a function(messages, tools) -> response."""

    def __init__(self, script: List[LLMResponse] | Callable[[List[Dict[str, Any]], List[Dict[str, Any]]], LLMResponse]):
        self.script = script
        self.calls: List[Dict[str, Any]] = []

    def _next(self, messages, tools) -> LLMResponse:
        if callable(self.script):
            return self.script(messages, tools or [])
        if not self.script:
            return text("(no more scripted replies)")
        return self.script.pop(0)

    async def chat(self, messages, *, model=None, tools=None, local_only=False, **kw) -> LLMResponse:
        self.calls.append({"messages": [dict(m) for m in messages], "tools": [t["function"]["name"] for t in (tools or [])],
                           "model": model, "local_only": local_only})
        return self._next(messages, tools)

    async def stream_events(self, messages, *, model=None, tools=None, local_only=False, **kw):
        resp = await self.chat(messages, model=model, tools=tools, local_only=local_only)
        if resp.content and not resp.tool_calls:
            for word in resp.content.split(" "):
                yield "text", word + " "
        yield "done", resp

    def offered(self, i: int = -1) -> List[str]:
        return self.calls[i]["tools"]


def install(monkeypatch, fake: FakeLLM) -> FakeLLM:
    import brain.pipeline as pipeline

    monkeypatch.setattr(pipeline, "llm", fake)
    return fake
