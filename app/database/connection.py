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

        CREATE TABLE IF NOT EXISTS session_buffers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS vector_documents (
            doc_id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            metadata TEXT DEFAULT '{}',
            updated_at TEXT NOT NULL
        );
        """
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.executescript(table_schema)
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

    async def get_session_chat_history(self, session_id: str, limit: int = 20) -> List[Dict[str, str]]:
        """Retrieve historical conversation messages for a session_id from SQLite database."""
        query = "SELECT user_query, response_text FROM chat_logs WHERE session_id = ? ORDER BY id DESC LIMIT ?;"
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(query, (session_id, limit)) as cursor:
                    rows = await cursor.fetchall()
                    history: List[Dict[str, str]] = []
                    for row in reversed(rows):
                        history.append({"role": "user", "content": row["user_query"]})
                        history.append({"role": "assistant", "content": row["response_text"]})
                    return history
        except Exception as exc:
            logger.error(f"Error fetching session chat history from database: {exc}")
            return []

    async def search_past_conversations(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Search all historical chat logs (days, weeks, or months ago) matching query keywords across all sessions."""
        if not query or not query.strip():
            return []

        import re
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "what", "how", "why", "who", "where",
            "can", "you", "tell", "me", "my", "your", "this", "that", "it", "to", "in", "on",
            "for", "with", "do", "did", "does", "i", "we", "about", "have", "has", "had", "switch", "change", "set", "use"
        }
        words = [w.lower().strip() for w in re.findall(r"\w+", query) if len(w) > 2 and w.lower() not in stop_words]

        if not words:
            fallback_query = "SELECT session_id, user_query, response_text, created_at FROM chat_logs ORDER BY id DESC LIMIT ?;"
            try:
                async with aiosqlite.connect(str(self.db_path)) as db:
                    db.row_factory = aiosqlite.Row
                    async with db.execute(fallback_query, (limit,)) as cursor:
                        rows = await cursor.fetchall()
                        return [dict(row) for row in rows]
            except Exception:
                return []

        conditions = []
        params = []
        for word in words:
            conditions.append("(LOWER(user_query) LIKE ? OR LOWER(response_text) LIKE ?)")
            term_pattern = f"%{word}%"
            params.extend([term_pattern, term_pattern])

        where_clause = " OR ".join(conditions)
        sql = f"""
        SELECT session_id, user_query, response_text, created_at
        FROM chat_logs
        WHERE {where_clause}
        ORDER BY id DESC
        LIMIT ?;
        """
        params.append(limit)

        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(sql, tuple(params)) as cursor:
                    rows = await cursor.fetchall()
                    return [dict(row) for row in rows]
        except Exception as exc:
            logger.error(f"Error searching past conversations in SQLite: {exc}")
            return []

    async def save_session_buffer_message(self, session_id: str, role: str, content: str) -> bool:
        """Persist a single message entry into session_buffers table."""
        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()
        query = "INSERT INTO session_buffers (session_id, role, content, created_at) VALUES (?, ?, ?, ?);"
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute(query, (session_id, role, content, now_iso))
                await db.commit()
            return True
        except Exception as exc:
            logger.error(f"Error persisting session buffer message: {exc}")
            return False

    async def get_session_buffer_messages(self, session_id: str, limit: int = 50) -> List[Dict[str, str]]:
        """Retrieve stored session messages from session_buffers table, with fallback to chat_logs."""
        query = "SELECT role, content FROM session_buffers WHERE session_id = ? ORDER BY id ASC LIMIT ?;"
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(query, (session_id, limit)) as cursor:
                    rows = await cursor.fetchall()
                    if rows:
                        return [{"role": row["role"], "content": row["content"]} for row in rows]

            # Fallback to chat_logs if session_buffers table is empty for this session_id
            return await self.get_session_chat_history(session_id=session_id, limit=limit)
        except Exception as exc:
            logger.error(f"Error fetching session buffer messages: {exc}")
            return []

    async def clear_session_buffer(self, session_id: str) -> bool:
        """Clear stored session messages from session_buffers table."""
        query = "DELETE FROM session_buffers WHERE session_id = ?;"
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute(query, (session_id,))
                await db.commit()
            return True
        except Exception as exc:
            logger.error(f"Error clearing session buffer: {exc}")
            return False

    async def save_vector_document(self, doc_id: str, content: str, metadata: Dict[str, Any]) -> bool:
        """Insert or update a vector document record in SQLite."""
        import json
        from datetime import datetime, timezone
        now_iso = datetime.now(timezone.utc).isoformat()
        meta_json = json.dumps(metadata)
        query = """
        INSERT INTO vector_documents (doc_id, content, metadata, updated_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(doc_id) DO UPDATE SET
            content = excluded.content,
            metadata = excluded.metadata,
            updated_at = excluded.updated_at;
        """
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute(query, (doc_id, content, meta_json, now_iso))
                await db.commit()
            return True
        except Exception as exc:
            logger.error(f"Error saving vector document '{doc_id}': {exc}")
            return False

    async def get_all_vector_documents(self) -> List[Dict[str, Any]]:
        """Fetch all stored vector documents from SQLite."""
        import json
        query = "SELECT doc_id, content, metadata FROM vector_documents;"
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(query) as cursor:
                    rows = await cursor.fetchall()
                    docs = []
                    for row in rows:
                        try:
                            meta = json.loads(row["metadata"]) if row["metadata"] else {}
                        except Exception:
                            meta = {}
                        docs.append({
                            "doc_id": row["doc_id"],
                            "content": row["content"],
                            "metadata": meta,
                        })
                    return docs
        except Exception as exc:
            logger.error(f"Error reading vector documents from SQLite: {exc}")
            return []

    async def clear_vector_documents(self) -> bool:
        """Clear all stored vector documents from SQLite."""
        query = "DELETE FROM vector_documents;"
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute(query)
                await db.commit()
            return True
        except Exception as exc:
            logger.error(f"Error clearing vector documents: {exc}")
            return False



# Global database manager instance
db_manager = DatabaseManager()
