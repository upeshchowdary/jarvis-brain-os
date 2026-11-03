"""chat_history: read JARVIS's own chat log ("what did I ask you today?", "what did we do about WhatsApp?")."""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta, timezone, tzinfo
from typing import Any, List, Optional, Tuple

from automation.gateway import REMOTE_SOURCES
from brain.prompts import wrap_untrusted
from memory import search as memory_search
from tools.base import Tool, ToolContext, ToolResult, obj
from tools.registry import register

DEFAULT_LIMIT = 20
MAX_LIMIT = 50
CLIP = 160
SINCE_HINT = "Use since = today, yesterday, this week, last week, or a date like 2026-10-29."
_TURN = re.compile(r"^User: (.*?) \| JARVIS: (.*)$", re.DOTALL)


def _local_tz() -> tzinfo:
    """The machine's local timezone (monkeypatched in tests)."""
    return datetime.now().astimezone().tzinfo or timezone.utc


def _now() -> datetime:
    """Current time as an aware datetime in the local timezone (monkeypatched in tests)."""
    return datetime.now(_local_tz())


def _midnight(day: date, tz: tzinfo) -> datetime:
    return datetime(day.year, day.month, day.day, tzinfo=tz)


def resolve_since(since: str, now: datetime) -> Tuple[Optional[datetime], Optional[datetime]]:
    """Local-day boundaries for ``since`` as aware datetimes (UTC-convertible). Raises ValueError if unknown."""
    key = " ".join((since or "").lower().replace("_", " ").split())
    tz = now.tzinfo or timezone.utc
    today = now.date()
    if not key:
        return None, None
    if key == "today":
        return _midnight(today, tz), None
    if key == "yesterday":
        return _midnight(today - timedelta(days=1), tz), _midnight(today, tz)
    monday = today - timedelta(days=today.weekday())
    if key == "this week":
        return _midnight(monday, tz), None
    if key == "last week":
        return _midnight(monday - timedelta(days=7), tz), _midnight(monday, tz)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", key):
        day = date.fromisoformat(key)  # ValueError on impossible dates
        return _midnight(day, tz), _midnight(day + timedelta(days=1), tz)
    raise ValueError(since)


def _clip(text: str, n: int = CLIP) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def _local(created_at: str, tz: tzinfo) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(created_at)
    except (TypeError, ValueError):
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(tz)


@register
class ChatHistory(Tool):
    name = "chat_history"
    description = ("Read your own past conversations with the user: what they asked you and what you answered. Use for "
                   "'what did I ask you today/yesterday', 'what did we do about X', 'what was that earlier'. "
                   "Optional query (topic), since (today|yesterday|this week|last week|YYYY-MM-DD), channel "
                   "(text|voice|api|mcp), limit (default 20, max 50).")
    parameters = obj({
        "query": {"type": "string", "description": "Optional topic to search for; empty = latest chats"},
        "since": {"type": "string", "description": "today | yesterday | this week | last week | YYYY-MM-DD"},
        "channel": {"type": "string", "description": "Optional: text | voice | api | mcp (MCP turns only when asked)"},
        "limit": {"type": "integer", "description": "How many turns (default 20, max 50)"},
    })
    final = False

    async def run(self, ctx: ToolContext, query: str = "", since: str = "", channel: str = "", limit: Any = DEFAULT_LIMIT,
                  **_: Any) -> ToolResult:
        if limit is None or limit == "":
            limit = DEFAULT_LIMIT
        if isinstance(limit, float) and limit.is_integer():  # models often send 5.0
            limit = int(limit)
        if isinstance(limit, bool) or not (isinstance(limit, int) or (isinstance(limit, str) and limit.strip().isdigit())):
            return ToolResult.error("limit must be a whole number between 1 and 50.")
        limit = max(1, min(int(limit), MAX_LIMIT))

        tz = _local_tz()
        now = _now()
        try:
            start, end = resolve_since(since or "", now)
        except ValueError:
            return ToolResult.error(f"I don't understand since=“{since}”. {SINCE_HINT}")
        start_utc = start.astimezone(timezone.utc) if start else None
        end_utc = end.astimezone(timezone.utc) if end else None
        channel = (channel or "").strip().lower() or None
        if ctx.source in REMOTE_SOURCES:
            channel = "mcp"  # a remote agent only reads its own turns, never the owner's conversations
        query = (query or "").strip()

        if query:
            hits = await memory_search.search(query, limit * 3, start_utc, end_utc, channel)
            # top-`limit` by relevance, shown chronologically like every other result
            chat_hits = [h for h in hits if h.chat_id is not None][:limit]  # chat turns only
            ordered = sorted(chat_hits, key=lambda h: _local(h.created_at, tz) or now)
        else:
            hits = await memory_search.recent(limit, start_utc, end_utc, channel)
            ordered = list(reversed(hits))  # newest last

        if not ordered:
            return ToolResult.ok("No chats found for that period.", data=[], content_for_llm="No chats found for that period.")

        stamps = [_local(h.created_at, tz) for h in ordered]
        multi_day = len({s.date() for s in stamps if s}) > 1
        lines: List[str] = []
        for hit, stamp in zip(ordered, stamps):
            if stamp is None:
                when = "?"
            else:
                when = stamp.strftime("%b %d %H:%M") if multi_day else stamp.strftime("%H:%M")
            m = _TURN.match(hit.text)
            if m:
                lines.append(f"[{when}] You: {_clip(m.group(1))} → JARVIS: {_clip(m.group(2))}")
            else:
                lines.append(f"[{when}] Memory: {_clip(hit.text)}")
        text = "\n".join(lines)
        # Past replies can quote web pages / files: the model gets them as untrusted data. The tool is deliberately
        # not untrusted_output (that would withdraw every action for the rest of the turn, e.g. "send it again").
        return ToolResult.ok(f"Found {len(lines)} chat turn{'s' if len(lines) != 1 else ''}.",
                             data=[{"chat_id": h.chat_id, "created_at": h.created_at} for h in ordered],
                             content_for_llm=wrap_untrusted(self.name, text))
