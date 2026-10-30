"""Long-term memory: durable facts about the user, stored in SQLite (data/jarvis_memory.db).

Lifecycle: candidate → importance/confidence threshold → conflict detection → store.
Keys are stable across restarts (sha1-based, never Python's randomized ``hash``).
Facts are extracted only from explicit, unambiguous statements ("my name is …",
"remember that …"); an optional LLM pass decides whether other statements are
durable facts. "I am tired" never becomes a name.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import aiosqlite
from pydantic import BaseModel, Field

from core.logging import logger
from core.paths import data_path


class MemoryCategory(str, Enum):
    USER_PREFERENCE = "USER_PREFERENCE"
    USER_PROFILE = "USER_PROFILE"
    PROJECT = "PROJECT"
    TASK = "TASK"
    GOAL = "GOAL"
    FACT = "FACT"
    CONVERSATION = "CONVERSATION"
    EPISODIC_EVENT = "EPISODIC_EVENT"
    VISUAL_EVENT = "VISUAL_EVENT"


class MemoryCandidate(BaseModel):
    key: str
    value: str
    category: MemoryCategory = MemoryCategory.FACT
    importance: float = Field(default=0.8, ge=0.0, le=1.0)
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    source: str = "dialogue"


def stable_key(prefix: str, text: str) -> str:
    norm = re.sub(r"\s+", " ", text.strip().lower())
    return f"{prefix}_{hashlib.sha1(norm.encode('utf-8')).hexdigest()[:10]}"


# Words that can follow "my name is"/"call me" but are not names.
_NOT_NAMES = {
    "not", "no", "just", "really", "very", "so", "the", "a", "an", "your", "his", "her", "their", "what",
    "tired", "hungry", "fine", "ok", "okay", "here", "back", "busy", "sorry", "sure", "anything", "whatever",
}
_NAME_RE = re.compile(r"\b(?:my name is|my name's|call me|i am called|i'm called|people call me)\s+([A-Za-z][A-Za-z'\-]{1,24}(?:\s+[A-Za-z][A-Za-z'\-]{1,24}){0,2})", re.IGNORECASE)
_PROJECT_RE = re.compile(r"\b(?:i am|i'm|im)\s+(?:currently\s+)?working on\s+(?:a |an |my |the )?(.{3,80}?)(?:[.!?,;]|$)", re.IGNORECASE)
_MY_PROJECT_RE = re.compile(r"\bmy (?:current )?project is\s+(.{3,80}?)(?:[.!?,;]|$)", re.IGNORECASE)
_FAVORITE_RE = re.compile(r"\bmy fav(?:ou?rite)? ([a-z ]{2,30}?) is\s+(.{1,60}?)(?:[.!?,;]|$)", re.IGNORECASE)
_PREFER_RE = re.compile(r"\bi (?:prefer|always use|usually use)\s+(.{2,80}?)(?:[.!?;]|$)", re.IGNORECASE)
_REMEMBER_RE = re.compile(r"^\s*(?:jarvis[, ]+)?(?:please\s+)?remember\s+(?:that\s+)?(.{3,300})$", re.IGNORECASE)
_LIVE_RE = re.compile(r"\bi (?:live in|am from|'m from)\s+([A-Za-z][A-Za-z .'-]{1,40}?)(?:[.!?,;]|$)", re.IGNORECASE)
_BIRTHDAY_RE = re.compile(r"\bmy birthday is\s+(?:on\s+)?(.{3,30}?)(?:[.!?,;]|$)", re.IGNORECASE)


def extract_explicit_facts(text: str) -> List[MemoryCandidate]:
    """Rule-based extraction of unambiguous self-statements (no LLM)."""
    t = text.strip()
    out: List[MemoryCandidate] = []
    m = _NAME_RE.search(t)
    if m:
        words = m.group(1).split()
        # Stop at the first word that clearly isn't part of a name ("call me Rahul please")
        name_words = []
        for w in words:
            if w.lower() in _NOT_NAMES or w.lower() in ("please", "from", "and", "but", "now", "today"):
                break
            name_words.append(w)
        if name_words and name_words[0].lower() not in _NOT_NAMES:
            name = " ".join(x.capitalize() if x.islower() else x for x in name_words)
            out.append(MemoryCandidate(key="user_name", value=name, category=MemoryCategory.USER_PROFILE, importance=0.95, confidence=0.95))
    for rx in (_PROJECT_RE, _MY_PROJECT_RE):
        m = rx.search(t)
        if m:
            out.append(MemoryCandidate(key="current_project", value=m.group(1).strip(), category=MemoryCategory.PROJECT, importance=0.85, confidence=0.85))
            break
    m = _FAVORITE_RE.search(t)
    if m:
        thing = re.sub(r"\s+", "_", m.group(1).strip().lower())
        out.append(MemoryCandidate(key=f"favorite_{thing}", value=m.group(2).strip(), category=MemoryCategory.USER_PREFERENCE, importance=0.8, confidence=0.9))
    m = _PREFER_RE.search(t)
    if m and len(m.group(1).split()) <= 10:
        pref = m.group(1).strip()
        out.append(MemoryCandidate(key=stable_key("pref", pref), value=pref, category=MemoryCategory.USER_PREFERENCE, importance=0.75, confidence=0.8))
    m = _LIVE_RE.search(t)
    if m:
        out.append(MemoryCandidate(key="home_location", value=m.group(1).strip(), category=MemoryCategory.USER_PROFILE, importance=0.85, confidence=0.85))
    m = _BIRTHDAY_RE.search(t)
    if m:
        out.append(MemoryCandidate(key="birthday", value=m.group(1).strip(), category=MemoryCategory.USER_PROFILE, importance=0.85, confidence=0.9))
    m = _REMEMBER_RE.match(t)
    if m:
        note = m.group(1).strip().rstrip(".")
        out.append(MemoryCandidate(key=stable_key("note", note), value=note, category=MemoryCategory.FACT, importance=0.9, confidence=0.95, source="explicit"))
    return out


class LongTermMemoryManager:
    """Stores persistent user facts in SQLite."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._explicit_path = db_path
        self._ready_for: Optional[str] = None
        self._lock = asyncio.Lock()

    @property
    def db_path(self) -> Path:
        return self._explicit_path or data_path("jarvis_memory.db")

    async def init_db(self) -> None:
        """Create tables once per database file (not on every call)."""
        path = str(self.db_path)
        if self._ready_for == path:
            return
        async with self._lock:
            if self._ready_for == path:
                return
            async with aiosqlite.connect(path) as db:
                await db.execute(
                    """CREATE TABLE IF NOT EXISTS user_facts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        fact_key TEXT UNIQUE NOT NULL,
                        fact_value TEXT NOT NULL,
                        category TEXT DEFAULT 'FACT',
                        importance REAL DEFAULT 0.8,
                        confidence REAL DEFAULT 0.9,
                        updated_at TEXT NOT NULL
                    )"""
                )
                cols = {row[1] async for row in await db.execute("PRAGMA table_info(user_facts)")}
                for col, ddl in (("importance", "REAL DEFAULT 0.8"), ("confidence", "REAL DEFAULT 0.9"), ("category", "TEXT DEFAULT 'FACT'")):
                    if col not in cols:
                        await db.execute(f"ALTER TABLE user_facts ADD COLUMN {col} {ddl}")
                await db.commit()
            self._ready_for = path
            logger.info(f"LongTermMemory ready at {path}")

    async def store_candidate(self, candidate: MemoryCandidate) -> bool:
        if candidate.importance < 0.3 or candidate.confidence < 0.4:
            return False
        await self.init_db()
        key = candidate.key.lower().strip()
        existing = await self.get_fact(key)
        if existing and existing != candidate.value:
            logger.info(f"Memory update for '{key}': '{existing}' -> '{candidate.value}'")
        async with aiosqlite.connect(str(self.db_path)) as db:
            await db.execute(
                """INSERT INTO user_facts (fact_key, fact_value, category, importance, confidence, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(fact_key) DO UPDATE SET fact_value=excluded.fact_value, category=excluded.category,
                     importance=excluded.importance, confidence=excluded.confidence, updated_at=excluded.updated_at""",
                (key, candidate.value, candidate.category.value, candidate.importance, candidate.confidence,
                 datetime.now(timezone.utc).isoformat()),
            )
            await db.commit()
        return True

    async def set_fact(self, key: str, value: str, category: str = "FACT") -> bool:
        try:
            cat = MemoryCategory(category.upper())
        except ValueError:
            cat = MemoryCategory.FACT
        return await self.store_candidate(MemoryCandidate(key=key, value=value, category=cat))

    async def remember(self, note: str) -> str:
        """Explicitly remember a note; returns the stored key."""
        key = stable_key("note", note)
        await self.store_candidate(MemoryCandidate(key=key, value=note.strip(), category=MemoryCategory.FACT, importance=0.9, confidence=0.95, source="explicit"))
        return key

    async def get_fact(self, key: str) -> Optional[str]:
        await self.init_db()
        async with aiosqlite.connect(str(self.db_path)) as db:
            async with db.execute("SELECT fact_value FROM user_facts WHERE fact_key = ?", (key.lower().strip(),)) as cur:
                row = await cur.fetchone()
                return row[0] if row else None

    async def get_all_facts(self) -> Dict[str, str]:
        await self.init_db()
        async with aiosqlite.connect(str(self.db_path)) as db:
            async with db.execute("SELECT fact_key, fact_value FROM user_facts ORDER BY importance DESC, updated_at DESC") as cur:
                return {r[0]: r[1] for r in await cur.fetchall()}

    async def list_facts(self) -> List[Dict[str, Any]]:
        await self.init_db()
        async with aiosqlite.connect(str(self.db_path)) as db:
            db.row_factory = aiosqlite.Row
            async with db.execute("SELECT fact_key, fact_value, category, importance, updated_at FROM user_facts ORDER BY importance DESC, updated_at DESC") as cur:
                return [dict(r) for r in await cur.fetchall()]

    async def relevant_facts(self, query: str, limit: int = 12) -> Dict[str, str]:
        """Profile facts first, then facts sharing words with the query, capped at ``limit``."""
        facts = await self.list_facts()
        words = {w for w in re.findall(r"[a-z0-9]{3,}", query.lower())}

        def score(f: Dict[str, Any]) -> float:
            text = f"{f['fact_key']} {f['fact_value']}".lower()
            overlap = sum(1 for w in words if w in text)
            profile = 1.0 if f["category"] == MemoryCategory.USER_PROFILE.value else 0.0
            return profile * 2 + overlap + float(f.get("importance") or 0)

        ranked = sorted(facts, key=score, reverse=True)[:limit]
        return {f["fact_key"]: f["fact_value"] for f in ranked}

    async def forget(self, query: str) -> List[str]:
        """Delete facts whose key or value matches the query (case-insensitive). Returns removed values."""
        await self.init_db()
        q = query.strip().lower()
        if not q:
            return []
        if q in ("everything", "all", "all facts", "everything about me"):
            pattern = "%"
        else:
            pattern = f"%{q}%"
        async with aiosqlite.connect(str(self.db_path)) as db:
            async with db.execute("SELECT fact_key, fact_value FROM user_facts WHERE lower(fact_key) LIKE ? OR lower(fact_value) LIKE ?", (pattern, pattern)) as cur:
                rows = await cur.fetchall()
            if rows:
                await db.executemany("DELETE FROM user_facts WHERE fact_key = ?", [(r[0],) for r in rows])
                await db.commit()
        return [r[1] for r in rows]

    async def auto_extract_facts(self, text: str) -> List[str]:
        """Store unambiguous self-statements. Returns 'key: value' strings that were stored."""
        stored = []
        for cand in extract_explicit_facts(text):
            if await self.store_candidate(cand):
                stored.append(f"{cand.key}: {cand.value}")
        return stored

    async def llm_extract_facts(self, text: str) -> List[str]:
        """Ask the fast LLM whether a first-person statement contains a durable fact about the user."""
        if not re.search(r"\b(i|i'm|im|my|me)\b", text.lower()) or len(text) > 600:
            return []
        from brain.utils import extract_json
        from core.llm import llm

        prompt = (
            "Does this message state a DURABLE fact about the user worth remembering for months "
            "(name, job, studies, projects, preferences, family, location, goals)? Temporary states "
            "(tired, hungry, busy today) and questions are NOT durable.\n"
            f'Message: "{text}"\n'
            'Reply JSON only: {"facts": [{"key": "snake_case_topic", "value": "short value"}]} or {"facts": []}'
        )
        try:
            res = await llm.chat([{"role": "user", "content": prompt}], temperature=0.0, max_tokens=150, purpose="fast")
            data = extract_json(res.content) or {}
        except Exception:
            return []
        stored = []
        for f in (data.get("facts") or [])[:3] if isinstance(data, dict) else []:
            key = re.sub(r"[^a-z0-9_]", "_", str(f.get("key", "")).lower())[:40].strip("_")
            value = str(f.get("value", "")).strip()[:200]
            if key and value and key != "user_name":  # names only from explicit statements
                if await self.store_candidate(MemoryCandidate(key=key, value=value, importance=0.7, confidence=0.7, source="llm")):
                    stored.append(f"{key}: {value}")
        return stored


long_term_memory = LongTermMemoryManager()
