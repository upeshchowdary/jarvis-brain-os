"""Long-Term Memory Manager for persisting user facts, profiles, and preferences into SQLite.

Implements structured memory lifecycle:
    candidate → importance → confidence → conflict detection → store
Supported categories:
    USER_PREFERENCE, USER_PROFILE, PROJECT, TASK, GOAL, FACT, CONVERSATION, EPISODIC_EVENT, VISUAL_EVENT
"""

import aiosqlite
import re
from enum import Enum
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from brain.brain_config import brain_config
from brain.logger import logger


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


class LongTermMemoryManager:
    """Stores persistent user facts, key-value preferences, and system memories in SQLite."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or (brain_config.base_dir / "data" / "jarvis_memory.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def init_db(self) -> None:
        """Initialize memory SQLite tables if they do not exist."""
        schema = """
        CREATE TABLE IF NOT EXISTS user_facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            fact_key TEXT UNIQUE NOT NULL,
            fact_value TEXT NOT NULL,
            category TEXT DEFAULT 'FACT',
            importance REAL DEFAULT 0.8,
            confidence REAL DEFAULT 0.9,
            updated_at TEXT NOT NULL
        );
        """
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute(schema)
                # Auto-migrate missing columns for existing SQLite DBs
                try:
                    await db.execute("ALTER TABLE user_facts ADD COLUMN importance REAL DEFAULT 0.8;")
                except Exception:
                    pass
                try:
                    await db.execute("ALTER TABLE user_facts ADD COLUMN confidence REAL DEFAULT 0.9;")
                except Exception:
                    pass
                await db.commit()
            logger.info(f"LongTermMemoryManager SQLite initialized at {self.db_path}")
        except Exception as exc:
            logger.error(f"Error initializing LongTermMemory database: {exc}")

    async def store_candidate(self, candidate: MemoryCandidate) -> bool:
        """Process candidate through importance/confidence threshold and conflict resolution before storing."""
        if candidate.importance < 0.3 or candidate.confidence < 0.4:
            logger.debug(f"Candidate memory '{candidate.key}' rejected due to low importance/confidence.")
            return False

        await self.init_db()
        now_iso = datetime.now(timezone.utc).isoformat()
        clean_key = candidate.key.lower().strip()

        # Check existing value for conflict detection
        existing_val = await self.get_fact(clean_key)
        if existing_val and existing_val != candidate.value:
            logger.info(
                f"Memory conflict detected for key '{clean_key}': "
                f"updating '{existing_val}' -> '{candidate.value}'"
            )

        query = """
        INSERT INTO user_facts (fact_key, fact_value, category, importance, confidence, updated_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(fact_key) DO UPDATE SET
            fact_value = excluded.fact_value,
            category = excluded.category,
            importance = excluded.importance,
            confidence = excluded.confidence,
            updated_at = excluded.updated_at;
        """
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute(
                    query,
                    (clean_key, candidate.value, candidate.category.value, candidate.importance, candidate.confidence, now_iso),
                )
                await db.commit()
            logger.info(f"LongTermMemory stored fact [{candidate.category.value}]: '{clean_key}' = '{candidate.value}'")
            return True
        except Exception as exc:
            logger.error(f"Error saving long term fact '{clean_key}': {exc}")
            return False

    async def set_fact(self, key: str, value: str, category: str = "FACT") -> bool:
        """Save or update user fact directly."""
        cat_enum = MemoryCategory.FACT
        try:
            cat_enum = MemoryCategory(category.upper())
        except ValueError:
            pass

        candidate = MemoryCandidate(
            key=key,
            value=value,
            category=cat_enum,
            importance=0.8,
            confidence=0.9,
        )
        return await self.store_candidate(candidate)

    async def get_fact(self, key: str) -> Optional[str]:
        """Retrieve a specific user fact by key."""
        await self.init_db()
        query = "SELECT fact_value FROM user_facts WHERE fact_key = ?;"
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                async with db.execute(query, (key.lower().strip(),)) as cursor:
                    row = await cursor.fetchone()
                    return row[0] if row else None
        except Exception as exc:
            logger.error(f"Error reading long term fact '{key}': {exc}")
            return None

    async def get_all_facts(self) -> Dict[str, str]:
        """Retrieve all stored user facts as a dictionary."""
        await self.init_db()
        query = "SELECT fact_key, fact_value FROM user_facts;"
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                async with db.execute(query) as cursor:
                    rows = await cursor.fetchall()
                    return {row[0]: row[1] for row in rows}
        except Exception as exc:
            logger.error(f"Error retrieving all long term facts: {exc}")
            return {}

    async def auto_extract_facts(self, text: str) -> List[str]:
        """Automatically detect and extract key user candidates from dialogue text."""
        extracted = []
        t_clean = text.strip()

        # 1. Name extraction (USER_PROFILE)
        name_match = re.search(r"\b(?:my name is|call me)\s+([a-zA-Z0-9_]{2,20})|\bi am\s+([a-zA-Z0-9_]{2,20})\b", t_clean, re.IGNORECASE)
        if name_match:
            name = (name_match.group(1) or name_match.group(2) or "").strip()
            if name and not any(w in name.lower() for w in ["a", "the", "looking", "trying", "here", "asking", "thinking", "testing"]):
                cand = MemoryCandidate(
                    key="user_name",
                    value=name,
                    category=MemoryCategory.USER_PROFILE,
                    importance=0.95,
                    confidence=0.95,
                )
                if await self.store_candidate(cand):
                    extracted.append(f"user_name: {name}")

        # 2. Project / Work extraction (PROJECT)
        proj_match = re.search(r"\b(?:i am working on|my project is)\s+([a-zA-Z0-9_\-\s]{2,50})", t_clean, re.IGNORECASE)
        if proj_match:
            proj = proj_match.group(1).strip()
            cand = MemoryCandidate(
                key="current_project",
                value=proj,
                category=MemoryCategory.PROJECT,
                importance=0.90,
                confidence=0.90,
            )
            if await self.store_candidate(cand):
                extracted.append(f"current_project: {proj}")

        # 3. Explicit preference ("i prefer ...", "my preference is ...")
        pref_match = re.search(r"\b(?:i prefer|my preference is|i like)\s+(.+)", t_clean, re.IGNORECASE)
        if pref_match:
            pref = pref_match.group(1).strip()
            pref_key = f"pref_{hash(pref) % 10000}"
            cand = MemoryCandidate(
                key=pref_key,
                value=pref,
                category=MemoryCategory.USER_PREFERENCE,
                importance=0.85,
                confidence=0.85,
            )
            if await self.store_candidate(cand):
                extracted.append(f"preference: {pref}")

        # 4. Explicit "remember that ..." or "remember ..." (FACT / EPISODIC_EVENT)
        rem_match = re.search(r"\bremember\s+(?:that\s+)?(.+)", t_clean, re.IGNORECASE)
        if rem_match:
            note = rem_match.group(1).strip()
            fact_key = f"note_{hash(note) % 10000}"
            cand = MemoryCandidate(
                key=fact_key,
                value=note,
                category=MemoryCategory.FACT,
                importance=0.88,
                confidence=0.90,
            )
            if await self.store_candidate(cand):
                extracted.append(f"note: {note}")

        return extracted


# Global long term memory singleton
long_term_memory = LongTermMemoryManager()
