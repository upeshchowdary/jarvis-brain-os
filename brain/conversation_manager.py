import uuid
from typing import Dict, List, Any, Optional
from brain.brain_config import brain_config
from brain.utils import estimate_token_count
from brain.logger import logger
from memory.short_term import short_term_memory
from app.database.connection import db_manager


class ConversationSession:
    """Represents a single conversation thread with history and token metrics."""

    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.messages: List[Dict[str, str]] = []
        self.summary: Optional[str] = None
        self.total_tokens: int = 0

    def add_message(self, role: str, content: str) -> None:
        """Append message and update estimated token count."""
        msg = {"role": role, "content": content}
        self.messages.append(msg)
        self.total_tokens += estimate_token_count(content)

    def truncate_history_if_needed(self, max_tokens: int = 4096, max_messages: int = 20) -> None:
        """Trim message history if token count or message count exceeds budget."""
        if len(self.messages) > max_messages or self.total_tokens > max_tokens:
            logger.info(f"Truncating conversation session '{self.session_id}' (Messages: {len(self.messages)}, Tokens: {self.total_tokens})")
            # Keep newest messages
            self.messages = self.messages[-max_messages:]
            self.total_tokens = sum(estimate_token_count(m["content"]) for m in self.messages)


class ConversationManager:
    """Manages active conversation sessions, token bounds, session lifecycles, and SQLite persistence."""

    def __init__(self) -> None:
        self._sessions: Dict[str, ConversationSession] = {}

    def get_or_create_session(self, session_id: Optional[str] = None) -> ConversationSession:
        """Retrieve existing session or spawn new session in memory."""
        sid = session_id or f"session_{uuid.uuid4().hex[:8]}"
        if sid not in self._sessions:
            logger.info(f"Creating new conversation session object: '{sid}'")
            self._sessions[sid] = ConversationSession(sid)
        return self._sessions[sid]

    async def get_or_create_session_async(self, session_id: Optional[str] = None) -> ConversationSession:
        """Retrieve existing session or spawn new session, restoring history from SQLite if needed."""
        session = self.get_or_create_session(session_id)
        if not session.messages:
            await self.restore_session_from_db(session.session_id)
        return session

    async def restore_session_from_db(self, session_id: str) -> ConversationSession:
        """Restore conversation history for a session from SQLite database."""
        session = self.get_or_create_session(session_id)
        db_history = await db_manager.get_session_buffer_messages(session_id, limit=brain_config.MAX_HISTORY_MESSAGES)
        if db_history:
            session.messages.clear()
            session.total_tokens = 0
            for msg in db_history:
                session.add_message(msg["role"], msg["content"])
            logger.info(f"Restored {len(session.messages)} messages for session '{session_id}' from SQLite database.")
        return session

    def add_user_message(self, session_id: str, content: str) -> None:
        """Add user message to session and short-term memory buffer."""
        session = self.get_or_create_session(session_id)
        session.add_message("user", content)
        session.truncate_history_if_needed(
            max_tokens=brain_config.MAX_CONVERSATION_TOKENS,
            max_messages=brain_config.MAX_HISTORY_MESSAGES,
        )
        short_term_memory.append(session_id, "user", content)

    def add_assistant_message(self, session_id: str, content: str) -> None:
        """Add assistant response to session and short-term memory buffer."""
        session = self.get_or_create_session(session_id)
        session.add_message("assistant", content)
        short_term_memory.append(session_id, "assistant", content)

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Fetch formatted message history for a session."""
        session = self.get_or_create_session(session_id)
        return session.messages.copy()

    def clear_session(self, session_id: str) -> bool:
        """Clear conversation history for a session in memory and SQLite."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            short_term_memory.clear(session_id)
            logger.info(f"Cleared session: '{session_id}'")
            return True
        return False


# Global conversation manager singleton
conversation_manager = ConversationManager()

