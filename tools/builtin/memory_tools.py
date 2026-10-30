"""Memory tools: remember / recall / forget facts about the user."""

from __future__ import annotations

from typing import Any

from memory.long_term import long_term_memory
from tools.base import Tool, ToolContext, ToolResult, obj
from tools.registry import register


def _pretty_key(key: str) -> str:
    if key.startswith(("note_", "pref_")):
        return ""
    return key.replace("_", " ").capitalize() + ": "


@register
class Remember(Tool):
    name = "memory_remember"
    description = "Save a fact or note about the user for future conversations (only when the user asks you to remember something)."
    parameters = obj({"fact": {"type": "string"}}, ["fact"])
    side_effect = True  # writes JARVIS's own memory; blocked when untrusted content is in context

    async def run(self, ctx: ToolContext, fact: str = "", **_: Any) -> ToolResult:
        fact = fact.strip()
        if not fact:
            return ToolResult.error("What should I remember?")
        stored = await long_term_memory.auto_extract_facts(fact)
        if not any(s.startswith("note_") for s in stored):
            await long_term_memory.remember(fact)
        return ToolResult.ok(f"Got it — I'll remember that {fact.rstrip('.')}.")


@register
class Recall(Tool):
    name = "memory_recall"
    description = "Look up what you know about the user (all facts, or facts related to a topic)."
    parameters = obj({"topic": {"type": "string", "description": "Optional topic; empty = everything"}})

    async def run(self, ctx: ToolContext, topic: str = "", **_: Any) -> ToolResult:
        facts = await long_term_memory.list_facts()
        if topic:
            t = topic.lower()
            facts = [f for f in facts if t in f["fact_key"].lower() or t in f["fact_value"].lower()] or facts
        if not facts:
            return ToolResult.ok("I don't have anything saved about you yet. Tell me things like “remember that my exam is on Friday”.")
        lines = [f"• {_pretty_key(f['fact_key'])}{f['fact_value']}" for f in facts[:30]]
        return ToolResult.ok("Here's what I know about you:\n" + "\n".join(lines), data=facts)


@register
class Forget(Tool):
    name = "memory_forget"
    description = "Delete saved facts that match a topic (or 'everything')."
    parameters = obj({"topic": {"type": "string"}}, ["topic"])
    side_effect = True

    async def run(self, ctx: ToolContext, topic: str = "", **_: Any) -> ToolResult:
        removed = await long_term_memory.forget(topic)
        if not removed:
            return ToolResult.ok(f"I didn't have anything saved about “{topic}”.")
        return ToolResult.ok(f"Forgotten {len(removed)} item(s): " + "; ".join(removed[:5]))
