import asyncio
from typing import List, Dict, Any, Optional
from brain.brain_config import brain_config
from brain.utils import estimate_token_count
from brain.logger import logger
from app.database.connection import db_manager


class ShortTermMemoryManager:
    """Manages active conversation session memory buffers with sliding-window token bounds and SQLite persistence."""

    def __init__(self, max_tokens: int = 4096, max_history: int = 20) -> None:
        self.max_tokens = max_tokens
        self.max_history = max_history
        self._buffers: Dict[str, List[Dict[str, str]]] = {}

    def append(self, session_id: str, role: str, content: str) -> None:
        """Append message to session buffer and trigger persistence."""
        if session_id not in self._buffers:
            self._buffers[session_id] = []
        self._buffers[session_id].append({"role": role, "content": content})
        self._trim_buffer(session_id)

        # Trigger async persistence if event loop is active
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.create_task(db_manager.save_session_buffer_message(session_id, role, content))
        except RuntimeError:
            pass

    async def append_async(self, session_id: str, role: str, content: str) -> None:
        """Append message to session buffer and await persistence to SQLite."""
        self.append(session_id, role, content)
        await db_manager.save_session_buffer_message(session_id, role, content)

    async def load_from_db(self, session_id: str) -> List[Dict[str, str]]:
        """Restore short-term session memory buffer from SQLite database."""
        db_history = await db_manager.get_session_buffer_messages(session_id, limit=self.max_history)
        self._buffers[session_id] = db_history.copy()
        self._trim_buffer(session_id)
        return self._buffers[session_id].copy()

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Retrieve sliding history for session from memory buffer."""
        return self._buffers.get(session_id, []).copy()

    async def get_history_async(self, session_id: str) -> List[Dict[str, str]]:
        """Retrieve sliding history for session, loading from SQLite if buffer is not loaded."""
        if session_id not in self._buffers or not self._buffers[session_id]:
            await self.load_from_db(session_id)
        return self._buffers.get(session_id, []).copy()

    def clear(self, session_id: str) -> None:
        """Clear memory buffer for session."""
        if session_id in self._buffers:
            del self._buffers[session_id]
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.create_task(db_manager.clear_session_buffer(session_id))
        except RuntimeError:
            pass

    async def clear_async(self, session_id: str) -> None:
        """Clear memory buffer for session and delete from SQLite."""
        if session_id in self._buffers:
            del self._buffers[session_id]
        await db_manager.clear_session_buffer(session_id)

    def _trim_buffer(self, session_id: str) -> None:
        """Trim oldest messages if total token count or message count exceeds budget."""
        buf = self._buffers.get(session_id, [])
        while len(buf) > self.max_history:
            buf.pop(0)

        total_tokens = sum(estimate_token_count(m["content"]) for m in buf)
        while total_tokens > self.max_tokens and len(buf) > 2:
            buf.pop(0)
            total_tokens = sum(estimate_token_count(m["content"]) for m in buf)


# Global short term memory singleton
short_term_memory = ShortTermMemoryManager()

