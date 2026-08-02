"""Abstract Interface for Memory Subsystems (Short-term, Long-term, Vector)."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class BaseMemory(ABC):
    """Abstract protocol for future conversation, user, and vector memory integration."""

    @abstractmethod
    async def store(self, key: str, value: Any, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Store an entry in memory."""
        pass

    @abstractmethod
    async def retrieve(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve relevant context or memories matching query."""
        pass

    @abstractmethod
    async def clear(self, session_id: Optional[str] = None) -> bool:
        """Clear memory buffer or session context."""
        pass
