"""Short-Term Memory Manager for active session dialogue buffer."""

from typing import List, Dict, Any, Optional
from brain.brain_config import brain_config
from brain.utils import estimate_token_count
from brain.logger import logger


class ShortTermMemoryManager:
    """Manages active conversation session memory buffers with sliding-window token bounds."""

    def __init__(self, max_tokens: int = 4096, max_history: int = 20) -> None:
        self.max_tokens = max_tokens
        self.max_history = max_history
        self._buffers: Dict[str, List[Dict[str, str]]] = {}

    def append(self, session_id: str, role: str, content: str) -> None:
        """Append message to session buffer."""
        if session_id not in self._buffers:
            self._buffers[session_id] = []
        self._buffers[session_id].append({"role": role, "content": content})
        self._trim_buffer(session_id)

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        """Retrieve sliding history for session."""
        return self._buffers.get(session_id, []).copy()

    def clear(self, session_id: str) -> None:
        """Clear memory buffer for session."""
        if session_id in self._buffers:
            del self._buffers[session_id]

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
