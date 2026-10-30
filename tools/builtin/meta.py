"""Conversation-control tools: clarification questions and per-session model switching."""

from __future__ import annotations

from typing import Any

from core.llm import PROVIDERS, resolve_model
from core.session import sessions
from tools.base import Tool, ToolContext, ToolResult, obj
from tools.registry import register


@register
class Clarify(Tool):
    name = "clarify"
    description = "Ask the user ONE short clarifying question when you are not sure whether they want an action performed or what exactly to do."
    parameters = obj({"question": {"type": "string"}}, ["question"])

    async def run(self, ctx: ToolContext, question: str = "", **_: Any) -> ToolResult:
        return ToolResult.ok(question.strip() or "Could you clarify what you'd like me to do?")


MODEL_ALIASES = {
    "gemini": "gemini/gemini-3.5-flash-lite",
    "groq": "groq/llama-3.3-70b-versatile",
    "llama": "groq/llama-3.3-70b-versatile",
    "openai": "openai/gpt-4o-mini",
    "gpt": "openai/gpt-4o-mini",
    "claude": "anthropic/claude-sonnet-4-5",
    "anthropic": "anthropic/claude-sonnet-4-5",
    "deepseek": "deepseek/deepseek-chat",
    "openrouter": "openrouter/openrouter/auto",
    "ollama": None,  # filled from settings at runtime
    "local": None,
    "lm studio": "lmstudio/local-model",
    "lmstudio": "lmstudio/local-model",
}


def resolve_alias(name: str) -> str:
    from core.config import settings

    key = name.strip().lower()
    if key in ("ollama", "local", "offline"):
        return f"ollama/{settings.OLLAMA_MODEL}"
    if key in ("vision", "ollama vision", "qwen vision", "qwen 2b", "qwen2b", "ollama qwen 2b"):
        return f"ollama/{settings.OLLAMA_VISION_MODEL}"
    if key in ("default", "reset"):
        return ""
    if key in MODEL_ALIASES and MODEL_ALIASES[key]:
        return MODEL_ALIASES[key]
    prov, model = resolve_model(name.strip())
    return f"{prov}/{model}"


@register
class SwitchModel(Tool):
    name = "switch_model"
    description = "Switch the language model for this conversation only (e.g. gemini, groq, openai, claude, ollama, or provider/model)."
    parameters = obj({"model": {"type": "string"}}, ["model"])
    side_effect = True  # changes session config; never triggered by untrusted content

    async def run(self, ctx: ToolContext, model: str = "", **_: Any) -> ToolResult:
        target = resolve_alias(model)
        state = sessions.get(ctx.session_id)
        if not target:
            state.model_override = None
            return ToolResult.ok("Back to the default model for this conversation.")
        prov = target.split("/", 1)[0]
        if not PROVIDERS[prov].configured():
            return ToolResult.error(f"{prov} isn't configured — add its API key to .env first.")
        state.model_override = target
        return ToolResult.ok(f"Switched this conversation to {target}.")
