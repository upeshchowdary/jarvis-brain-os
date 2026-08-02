"""Long-Term Memory Manager for persisting user facts and preferences into SQLite."""

import aiosqlite
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
from brain.brain_config import brain_config
from brain.logger import logger


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
            category TEXT DEFAULT 'general',
            updated_at TEXT NOT NULL
        );
        """
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute(schema)
                await db.commit()
            logger.info(f"LongTermMemoryManager SQLite initialized at {self.db_path}")
        except Exception as exc:
            logger.error(f"Error initializing LongTermMemory database: {exc}")

    async def set_fact(self, key: str, value: str, category: str = "general") -> bool:
        """Save or update user fact."""
        await self.init_db()
        now_iso = datetime.now(timezone.utc).isoformat()
        query = """
        INSERT INTO user_facts (fact_key, fact_value, category, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(fact_key) DO UPDATE SET
            fact_value = excluded.fact_value,
            category = excluded.category,
            updated_at = excluded.updated_at;
        """
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute(query, (key.lower().strip(), value, category, now_iso))
                await db.commit()
            logger.info(f"LongTermMemory saved fact: '{key}' = '{value}'")
            return True
        except Exception as exc:
            logger.error(f"Error saving long term fact '{key}': {exc}")
            return False

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


# Global long term memory singleton
long_term_memory = LongTermMemoryManager()
