"""Async SQLite Database Connection and Telemetry Persistence Manager."""

import sqlite3
import aiosqlite
from pathlib import Path
from typing import Optional, List, Dict, Any
from app.config.settings import settings
from app.database.models import ChatLogRecord
from app.utils.logger import logger
from app.utils.exceptions import DatabaseError


class DatabaseManager:
    """Async SQLite Database Connection Manager for telemetry and metrics storage."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or (settings.base_dir / "data" / "jarvis.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    async def init_db(self) -> None:
        """Initialize SQLite database tables if they do not exist."""
        table_schema = """
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            user_query TEXT NOT NULL,
            intent TEXT NOT NULL,
            confidence REAL NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            latency_ms REAL NOT NULL,
            prompt_tokens INTEGER DEFAULT 0,
            completion_tokens INTEGER DEFAULT 0,
            total_tokens INTEGER DEFAULT 0,
            response_text TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        """
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute(table_schema)
                await db.commit()
            logger.info(f"SQLite database initialized successfully at: {self.db_path}")
        except Exception as exc:
            logger.error(f"Failed to initialize SQLite database: {exc}")
            raise DatabaseError(f"Database initialization error: {exc}") from exc

    async def log_chat_telemetry(self, record: ChatLogRecord) -> bool:
        """Insert execution log record into SQLite table."""
        insert_query = """
        INSERT INTO chat_logs (
            session_id, user_query, intent, confidence, provider, model,
            latency_ms, prompt_tokens, completion_tokens, total_tokens,
            response_text, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute(
                    insert_query,
                    (
                        record.session_id,
                        record.user_query,
                        record.intent,
                        record.confidence,
                        record.provider,
                        record.model,
                        record.latency_ms,
                        record.prompt_tokens,
                        record.completion_tokens,
                        record.total_tokens,
                        record.response_text,
                        record.created_at,
                    ),
                )
                await db.commit()
            logger.debug(f"Chat telemetry persisted to database for provider: {record.provider}")
            return True
        except Exception as exc:
            logger.error(f"Error persisting chat telemetry: {exc}")
            return False

    async def get_recent_logs(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Fetch recent execution logs for system status check."""
        query = "SELECT id, session_id, user_query, intent, provider, model, latency_ms, total_tokens, created_at FROM chat_logs ORDER BY id DESC LIMIT ?;"
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(query, (limit,)) as cursor:
                    rows = await cursor.fetchall()
                    return [dict(row) for row in rows]
        except Exception as exc:
            logger.error(f"Error reading recent logs: {exc}")
            return []


# Global database manager instance
db_manager = DatabaseManager()
