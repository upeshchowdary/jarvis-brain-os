"""Reminder / schedule engine (persisted in SQLite, survives restarts).

A reminder can carry an *action*: a routine name (``routine:study mode``) or a command
(``command:open my class links``). Actions run through the normal pipeline in the
``scheduler`` session, so the ActionGateway still applies — LOW-risk steps run on
their own, MEDIUM/HIGH ones wait for approval on the dashboard.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import Any, Dict, Optional

from core.logging import logger
from services.notify import notify
from services.store import store
from services.timeparse import next_occurrence


class Scheduler:
    def __init__(self, interval: float = 1.0) -> None:
        self.interval = interval
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.get_running_loop().create_task(self._loop())

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._task = None

    async def _loop(self) -> None:
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning(f"[Scheduler] tick failed: {exc}")
            await asyncio.sleep(self.interval)

    async def tick(self, now: Optional[float] = None) -> int:
        now = now or time.time()
        fired = 0
        for r in await store.due_reminders(now):
            fired += 1
            late = now - r["due_ts"]
            await self.fire(r, late)
            if r.get("recurrence"):
                nxt = next_occurrence(r["recurrence"], datetime.fromtimestamp(r["due_ts"]), datetime.fromtimestamp(now))
                if nxt:
                    await store.reschedule(r["id"], nxt.timestamp())
                    continue
            await store.mark_reminder(r["id"], "done")
        return fired

    async def fire(self, reminder: Dict[str, Any], late_seconds: float = 0.0) -> None:
        note = f" (delivered {int(late_seconds // 60)} min late — JARVIS was off)" if late_seconds > 120 else ""
        action = reminder.get("action") or ""
        if action.startswith(("routine:", "command:")):
            await self._run_action(action, reminder["text"])
        else:
            await notify("⏰ Reminder", f"{reminder['text']}{note}", kind="reminder")

    async def _run_action(self, action: str, label: str) -> None:
        from brain.pipeline import brain

        kind, _, value = action.partition(":")
        command = f"run routine {value}" if kind == "routine" else value
        result = await brain.handle(command, "scheduler", channel="api")
        title = "⏰ Scheduled task" if result.status != "needs_confirmation" else "⏰ Scheduled task needs approval"
        await notify(title, f"{label}: {result.response[:200]}", kind="scheduled_task")


scheduler = Scheduler()
