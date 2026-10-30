"""Async SQLite Database Connection and Telemetry Persistence Manager."""

import aiosqlite
from pathlib import Path
from typing import Optional, List, Dict, Any
from app.database.models import ChatLogRecord
from app.utils.logger import logger
from core.paths import data_path
from app.utils.exceptions import DatabaseError


class DatabaseManager:
    """Async SQLite Database Connection Manager for telemetry and metrics storage."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self.db_path = db_path or data_path("jarvis.db")
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
            created_at TEXT NOT NULL,
            channel TEXT DEFAULT 'text'
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

        CREATE TABLE IF NOT EXISTS tasks (
            id TEXT PRIMARY KEY,
            status TEXT,
            data TEXT,
            updated_at REAL
        );
        """
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.executescript(table_schema)
                async with db.execute("PRAGMA table_info(chat_logs);") as cursor:
                    columns = {row[1] for row in await cursor.fetchall()}
                if "channel" not in columns:
                    await db.execute("ALTER TABLE chat_logs ADD COLUMN channel TEXT DEFAULT 'text';")
                await self._init_chat_fts(db)
                await db.commit()
            logger.info(f"SQLite database initialized successfully at: {self.db_path}")
        except Exception as exc:
            logger.error(f"Failed to initialize SQLite database: {exc}")
            raise DatabaseError(f"Database initialization error: {exc}") from exc

    @staticmethod
    async def _init_chat_fts(db: aiosqlite.Connection) -> None:
        """FTS5 keyword index over chat_logs (query + response), kept current by an insert trigger."""
        async with db.execute("SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'chat_fts';") as cursor:
            is_new = await cursor.fetchone() is None
        await db.executescript("""
        CREATE VIRTUAL TABLE IF NOT EXISTS chat_fts USING fts5(
            user_query, response_text, content='chat_logs', content_rowid='id'
        );
        CREATE TRIGGER IF NOT EXISTS chat_logs_fts_insert AFTER INSERT ON chat_logs BEGIN
            INSERT INTO chat_fts(rowid, user_query, response_text)
            VALUES (new.id, new.user_query, new.response_text);
        END;
        """)
        if is_new:  # one-time backfill of the turns logged before the index existed
            await db.execute("INSERT INTO chat_fts(chat_fts) VALUES('rebuild');")

    async def log_chat_telemetry(self, record: ChatLogRecord) -> Optional[int]:
        """Insert execution log record into SQLite table. Returns the new chat id (None on failure)."""
        insert_query = """
        INSERT INTO chat_logs (
            session_id, user_query, intent, confidence, provider, model,
            latency_ms, prompt_tokens, completion_tokens, total_tokens,
            response_text, created_at, channel
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                cursor = await db.execute(
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
                        record.channel,
                    ),
                )
                await db.commit()
            logger.debug(f"Chat telemetry persisted to database for provider: {record.provider}")
            return cursor.lastrowid
        except Exception as exc:
            logger.error(f"Error persisting chat telemetry: {exc}")
            return None

    async def search_chat_fts(self, terms: List[str], limit: int = 20, since_iso: Optional[str] = None,
                              until_iso: Optional[str] = None) -> List[Dict[str, Any]]:
        """Chat turns matching any of ``terms`` (FTS5), best bm25 first, optionally within [since, until)."""
        if not terms:
            return []
        match = " OR ".join('"' + term.replace('"', '""') + '"' for term in terms)
        clauses, params = ["chat_fts MATCH ?"], [match]
        if since_iso:
            clauses.append("c.created_at >= ?")
            params.append(since_iso)
        if until_iso:
            clauses.append("c.created_at < ?")
            params.append(until_iso)
        sql = f"""
        SELECT c.id, c.user_query, c.response_text, c.created_at, c.channel, bm25(chat_fts) AS score
        FROM chat_fts JOIN chat_logs c ON c.id = chat_fts.rowid
        WHERE {" AND ".join(clauses)}
        ORDER BY score LIMIT ?;
        """
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(sql, (*params, limit)) as cursor:
                    return [dict(row) for row in await cursor.fetchall()]
        except Exception as exc:
            logger.error(f"Error searching chat_fts: {exc}")
            return []

    async def recent_chats(self, limit: int, since_iso: Optional[str] = None, until_iso: Optional[str] = None,
                           channel: Optional[str] = None) -> List[Dict[str, Any]]:
        """Newest chat turns first, optionally within [since, until) and on one channel."""
        clauses, params = [], []
        if since_iso:
            clauses.append("created_at >= ?")
            params.append(since_iso)
        if until_iso:
            clauses.append("created_at < ?")
            params.append(until_iso)
        if channel:
            clauses.append("channel = ?")
            params.append(channel)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
        SELECT id, user_query, response_text, created_at, channel FROM chat_logs {where}
        ORDER BY created_at DESC, id DESC LIMIT ?;
        """
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                db.row_factory = aiosqlite.Row
                async with db.execute(sql, (*params, limit)) as cursor:
                    return [dict(row) for row in await cursor.fetchall()]
        except Exception as exc:
            logger.error(f"Error reading recent chats: {exc}")
            return []

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
                    for row in reversed(list(rows)):
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

    async def save_task(self, task: Dict[str, Any]) -> bool:
        """Insert or update a task (the full task dict is stored as JSON)."""
        import json
        query = """
        INSERT INTO tasks (id, status, data, updated_at) VALUES (?, ?, ?, ?)
        ON CONFLICT(id) DO UPDATE SET status = excluded.status, data = excluded.data, updated_at = excluded.updated_at;
        """
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                await db.execute(query, (task["id"], task.get("status"), json.dumps(task), task.get("updated_at", 0.0)))
                await db.commit()
            return True
        except Exception as exc:
            logger.error(f"Error saving task: {exc}")
            return False

    async def load_recent_tasks(self, limit: int = 5) -> List[Dict[str, Any]]:
        """Most recently updated tasks first."""
        import json
        query = "SELECT data FROM tasks ORDER BY updated_at DESC LIMIT ?;"
        try:
            async with aiosqlite.connect(str(self.db_path)) as db:
                async with db.execute(query, (limit,)) as cursor:
                    return [json.loads(row[0]) for row in await cursor.fetchall()]
        except Exception as exc:
            logger.error(f"Error loading tasks: {exc}")
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
