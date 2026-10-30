"""Local productivity store (SQLite at data/jarvis_productivity.db): reminders, notes, to-dos, clipboard."""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import aiosqlite

from core.paths import data_path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS reminders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    due_ts REAL NOT NULL,
    recurrence TEXT,
    action TEXT,
    status TEXT NOT NULL DEFAULT 'scheduled',
    created_ts REAL NOT NULL,
    fired_ts REAL
);
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    created_ts REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS todos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    done INTEGER NOT NULL DEFAULT 0,
    created_ts REAL NOT NULL,
    done_ts REAL
);
CREATE TABLE IF NOT EXISTS clipboard (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    text TEXT NOT NULL,
    created_ts REAL NOT NULL
);
"""


class Store:
    def __init__(self) -> None:
        self._ready_for: Optional[str] = None
        self._lock = asyncio.Lock()

    @property
    def path(self) -> str:
        return str(data_path("jarvis_productivity.db"))

    async def _db(self) -> aiosqlite.Connection:
        if self._ready_for != self.path:
            async with self._lock:
                if self._ready_for != self.path:
                    async with aiosqlite.connect(self.path) as db:
                        await db.executescript(_SCHEMA)
                        await db.commit()
                    self._ready_for = self.path
        db = await aiosqlite.connect(self.path)
        db.row_factory = aiosqlite.Row
        return db

    async def execute(self, sql: str, params: tuple = ()) -> int:
        db = await self._db()
        try:
            cur = await db.execute(sql, params)
            await db.commit()
            return cur.lastrowid or cur.rowcount
        finally:
            await db.close()

    async def query(self, sql: str, params: tuple = ()) -> List[Dict[str, Any]]:
        db = await self._db()
        try:
            async with db.execute(sql, params) as cur:
                return [dict(r) for r in await cur.fetchall()]
        finally:
            await db.close()

    # ── Reminders ───────────────────────────────────────────────────────
    async def add_reminder(self, text: str, due_ts: float, recurrence: Optional[str] = None, action: Optional[str] = None) -> int:
        return await self.execute(
            "INSERT INTO reminders (text, due_ts, recurrence, action, created_ts) VALUES (?, ?, ?, ?, ?)",
            (text, due_ts, recurrence, action, time.time()),
        )

    async def reminders(self, include_done: bool = False) -> List[Dict[str, Any]]:
        where = "" if include_done else "WHERE status = 'scheduled'"
        return await self.query(f"SELECT * FROM reminders {where} ORDER BY due_ts")

    async def due_reminders(self, now: float) -> List[Dict[str, Any]]:
        return await self.query("SELECT * FROM reminders WHERE status = 'scheduled' AND due_ts <= ? ORDER BY due_ts", (now,))

    async def reschedule(self, rid: int, due_ts: float) -> None:
        await self.execute("UPDATE reminders SET due_ts = ?, fired_ts = ? WHERE id = ?", (due_ts, time.time(), rid))

    async def mark_reminder(self, rid: int, status: str) -> int:
        return await self.execute("UPDATE reminders SET status = ?, fired_ts = ? WHERE id = ?", (status, time.time(), rid))

    async def cancel_reminders(self, query: str) -> List[Dict[str, Any]]:
        q = query.strip().lower()
        rows = await self.reminders()
        hits = [r for r in rows if (q.isdigit() and r["id"] == int(q)) or (q and q in r["text"].lower()) or q in ("all", "everything")]
        for r in hits:
            await self.mark_reminder(r["id"], "cancelled")
        return hits

    # ── Notes & to-dos ──────────────────────────────────────────────────
    async def add_note(self, text: str) -> int:
        return await self.execute("INSERT INTO notes (text, created_ts) VALUES (?, ?)", (text, time.time()))

    async def notes(self, search: str = "", limit: int = 50) -> List[Dict[str, Any]]:
        if search:
            return await self.query("SELECT * FROM notes WHERE lower(text) LIKE ? ORDER BY created_ts DESC LIMIT ?", (f"%{search.lower()}%", limit))
        return await self.query("SELECT * FROM notes ORDER BY created_ts DESC LIMIT ?", (limit,))

    async def add_todo(self, text: str) -> int:
        return await self.execute("INSERT INTO todos (text, created_ts) VALUES (?, ?)", (text, time.time()))

    async def todos(self, include_done: bool = False) -> List[Dict[str, Any]]:
        where = "" if include_done else "WHERE done = 0"
        return await self.query(f"SELECT * FROM todos {where} ORDER BY created_ts")

    async def complete_todo(self, query: str, remove: bool = False) -> List[Dict[str, Any]]:
        q = query.strip().lower()
        items = await self.todos()
        hits = [t for t in items if (q.isdigit() and t["id"] == int(q)) or (q and q in t["text"].lower())]
        for t in hits:
            if remove:
                await self.execute("DELETE FROM todos WHERE id = ?", (t["id"],))
            else:
                await self.execute("UPDATE todos SET done = 1, done_ts = ? WHERE id = ?", (time.time(), t["id"]))
        return hits

    # ── Clipboard history ───────────────────────────────────────────────
    async def add_clip(self, text: str, keep: int = 50) -> None:
        last = await self.query("SELECT text FROM clipboard ORDER BY id DESC LIMIT 1")
        if last and last[0]["text"] == text:
            return
        await self.execute("INSERT INTO clipboard (text, created_ts) VALUES (?, ?)", (text[:20000], time.time()))
        await self.execute("DELETE FROM clipboard WHERE id NOT IN (SELECT id FROM clipboard ORDER BY id DESC LIMIT ?)", (keep,))

    async def clips(self, limit: int = 20) -> List[Dict[str, Any]]:
        return await self.query("SELECT * FROM clipboard ORDER BY id DESC LIMIT ?", (limit,))


store = Store()
