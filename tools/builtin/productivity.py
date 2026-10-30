"""Productivity tools: reminders, notes, to-dos, clipboard history, daily briefing, weather."""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Optional

import httpx

from automation.gateway import ActionRequest
from services.store import store
from services.timeparse import parse_when, strip_time_phrase
from tools.base import Tool, ToolContext, ToolResult, obj, via_gateway
from tools.registry import register


def _fmt_ts(ts: float) -> str:
    dt = datetime.fromtimestamp(ts)
    day = "today" if dt.date() == datetime.now().date() else dt.strftime("%a %d %b")
    return f"{day} {dt.strftime('%I:%M %p').lstrip('0')}"


class InternalWriteTool(Tool):
    """Writes JARVIS's own local data (not the OS) — side_effect so untrusted content can't trigger it."""

    side_effect = True


@register
class SetReminder(InternalWriteTool):
    name = "set_reminder"
    description = (
        "Set a reminder or recurring schedule. 'when' is natural language: 'in 10 minutes', 'at 6 pm', 'tomorrow at 9', "
        "'every weekday at 9', 'every 30 minutes'. Optional 'run' = a routine name or command to run at that time."
    )
    parameters = obj({"text": {"type": "string"}, "when": {"type": "string"}, "run": {"type": "string"}}, ["text", "when"])

    async def run(self, ctx: ToolContext, text: str = "", when: str = "", run: str = "", **_: Any) -> ToolResult:
        parsed = parse_when(when or text)
        if parsed is None:
            return ToolResult.error(f"I couldn't understand the time “{when}”. Try 'at 6 pm' or 'in 20 minutes'.")
        label = strip_time_phrase(text, parsed) or text or "Reminder"
        action = None
        if run:
            from services.routines import find

            action = f"routine:{run}" if find(run) else f"command:{run}"
        rid = await store.add_reminder(label, parsed.due.timestamp(), parsed.recurrence, action)
        if label.startswith("⏲"):
            return ToolResult.ok(f"⏲ Timer set — I'll let you know {parsed.describe()}.", data={"id": rid, "due": parsed.due.isoformat()})
        what = f"run “{run}”" if run else f"remind you to {label}" if not label.lower().startswith("remind") else label
        return ToolResult.ok(f"⏰ Okay — I'll {what} {parsed.describe()}.", data={"id": rid, "due": parsed.due.isoformat()})


@register
class ListReminders(Tool):
    name = "list_reminders"
    description = "List upcoming reminders and scheduled tasks."
    parameters = obj({})

    async def run(self, ctx: ToolContext, **_: Any) -> ToolResult:
        items = await store.reminders()
        if not items:
            return ToolResult.ok("You have no upcoming reminders.")
        lines = [f"• #{r['id']} {r['text']} — {_fmt_ts(r['due_ts'])}{' (repeats)' if r['recurrence'] else ''}" for r in items[:20]]
        return ToolResult.ok("Upcoming reminders:\n" + "\n".join(lines), data=items)


@register
class CancelReminder(InternalWriteTool):
    name = "cancel_reminder"
    description = "Cancel reminders by number or matching text (or 'all')."
    parameters = obj({"query": {"type": "string"}}, ["query"])

    async def run(self, ctx: ToolContext, query: str = "", **_: Any) -> ToolResult:
        hits = await store.cancel_reminders(query)
        if not hits:
            return ToolResult.ok(f"No reminder matched “{query}”.")
        return ToolResult.ok(f"Cancelled {len(hits)} reminder(s): " + "; ".join(h["text"] for h in hits[:5]))


@register
class AddNote(InternalWriteTool):
    name = "note_add"
    description = "Save a quick note."
    parameters = obj({"text": {"type": "string"}}, ["text"])

    async def run(self, ctx: ToolContext, text: str = "", **_: Any) -> ToolResult:
        await store.add_note(text.strip())
        return ToolResult.ok("📝 Noted.")


@register
class ListNotes(Tool):
    name = "notes_list"
    description = "Show saved notes (optionally matching a search term)."
    parameters = obj({"search": {"type": "string"}})

    async def run(self, ctx: ToolContext, search: str = "", **_: Any) -> ToolResult:
        notes = await store.notes(search)
        if not notes:
            return ToolResult.ok("No notes yet." if not search else f"No notes about “{search}”.")
        return ToolResult.ok("Your notes:\n" + "\n".join(f"• {n['text']} ({_fmt_ts(n['created_ts'])})" for n in notes[:20]), data=notes)


@register
class AddTodo(InternalWriteTool):
    name = "todo_add"
    description = "Add an item to the to-do list."
    parameters = obj({"text": {"type": "string"}}, ["text"])

    async def run(self, ctx: ToolContext, text: str = "", **_: Any) -> ToolResult:
        await store.add_todo(text.strip())
        return ToolResult.ok(f"✅ Added “{text.strip()}” to your to-do list.")


@register
class ListTodos(Tool):
    name = "todo_list"
    description = "Show the open to-do items."
    parameters = obj({})

    async def run(self, ctx: ToolContext, **_: Any) -> ToolResult:
        items = await store.todos()
        if not items:
            return ToolResult.ok("Your to-do list is empty. 🎉")
        return ToolResult.ok("To-do:\n" + "\n".join(f"• {t['text']}" for t in items), data=items)


@register
class CompleteTodo(InternalWriteTool):
    name = "todo_done"
    description = "Mark a to-do item as done (or remove it)."
    parameters = obj({"query": {"type": "string"}, "remove": {"type": "boolean"}}, ["query"])

    async def run(self, ctx: ToolContext, query: str = "", remove: bool = False, **_: Any) -> ToolResult:
        hits = await store.complete_todo(query, remove)
        if not hits:
            return ToolResult.ok(f"No to-do matched “{query}”.")
        return ToolResult.ok(f"{'Removed' if remove else 'Done'}: " + "; ".join(h["text"] for h in hits))


@register
class ClipboardHistory(Tool):
    name = "clipboard_history"
    description = "Show recently copied text (clipboard history)."
    parameters = obj({"limit": {"type": "integer", "default": 10}})
    sensitive_output = True

    async def run(self, ctx: ToolContext, limit: int = 10, **_: Any) -> ToolResult:
        clips = await store.clips(int(limit))
        if not clips:
            return ToolResult.ok("Clipboard history is empty (it fills while JARVIS is running).")
        lines = [f"{i}. {c['text'][:80].replace(chr(10), ' ')}" for i, c in enumerate(clips, 1)]
        return ToolResult.ok("Recently copied:\n" + "\n".join(lines), data=clips)


@register
class PasteFromHistory(Tool):
    name = "paste_from_history"
    description = "Put an earlier clipboard item back on the clipboard and paste it. index 1 = most recent, 3 = three items ago."
    parameters = obj({"index": {"type": "integer"}}, ["index"])
    side_effect = True

    async def run(self, ctx: ToolContext, index: int = 1, **_: Any) -> ToolResult:
        clips = await store.clips(max(int(index), 1))
        if len(clips) < int(index):
            return ToolResult.error(f"I only have {len(clips)} item(s) in clipboard history.")
        text = clips[int(index) - 1]["text"]
        plan = [
            ActionRequest("clipboard.set", {"text": text}, f"Copy “{text[:60]}” to the clipboard"),
            ActionRequest("computer.run_plan", {"goal": "Paste", "steps": [{"action": "hotkey", "value": "ctrl+v", "target": ""}]}, "Paste (Ctrl+V)"),
        ]
        return await via_gateway(ctx, plan)


async def fetch_weather(location: str = "") -> Optional[str]:
    """Current weather from wttr.in (free, no key)."""
    url = f"https://wttr.in/{location.replace(' ', '+')}?format=j1"
    try:
        async with httpx.AsyncClient(timeout=6.0, headers={"User-Agent": "curl/8"}) as client:
            data = (await client.get(url)).json()
        cur = data["current_condition"][0]
        area = data.get("nearest_area", [{}])[0].get("areaName", [{}])[0].get("value", location or "your area")
        today = data.get("weather", [{}])[0]
        return (f"{area}: {cur['weatherDesc'][0]['value'].lower()}, {cur['temp_C']}°C (feels {cur['FeelsLikeC']}°C), "
                f"high {today.get('maxtempC', '?')}° / low {today.get('mintempC', '?')}°")
    except Exception:
        return None


async def build_briefing() -> str:
    from memory.long_term import long_term_memory
    from tools.builtin.info import collect_system_status, format_status

    now = datetime.now()
    name = await long_term_memory.get_fact("user_name")
    location = await long_term_memory.get_fact("home_location") or ""
    greeting = "Good morning" if now.hour < 12 else "Good afternoon" if now.hour < 17 else "Good evening"
    weather, reminders, todos, status = await asyncio.gather(
        fetch_weather(location), store.reminders(), store.todos(), asyncio.to_thread(collect_system_status)
    )
    end_of_day = now.replace(hour=23, minute=59).timestamp()
    today = [r for r in reminders if r["due_ts"] <= end_of_day]
    parts = [f"{greeting}{', ' + name if name else ''}! It's {now.strftime('%A, %d %B')}."]
    if weather:
        parts.append(f"🌤 Weather — {weather}.")
    parts.append(f"⏰ {len(today)} reminder(s) today" + (": " + "; ".join(f"{r['text']} at {_fmt_ts(r['due_ts']).split(' ', 1)[1]}" for r in today[:5]) if today else "."))
    parts.append(f"✅ {len(todos)} open to-do(s)" + (": " + "; ".join(t["text"] for t in todos[:5]) if todos else "."))
    parts.append("💻 " + format_status(status, ""))
    return "\n".join(parts)


@register
class DailyBriefing(Tool):
    name = "daily_briefing"
    description = "Daily briefing: weather, today's reminders, to-dos and PC health."
    parameters = obj({})

    async def run(self, ctx: ToolContext, **_: Any) -> ToolResult:
        return ToolResult.ok(await build_briefing())


@register
class Weather(Tool):
    name = "weather"
    description = "Current weather and today's high/low for a place (default: the user's saved home location)."
    parameters = obj({"location": {"type": "string"}})

    async def run(self, ctx: ToolContext, location: str = "", **_: Any) -> ToolResult:
        from memory.long_term import long_term_memory

        loc = location or await long_term_memory.get_fact("home_location") or ""
        report = await fetch_weather(loc)
        return ToolResult.ok(report) if report else ToolResult.error("I couldn't reach the weather service.")
