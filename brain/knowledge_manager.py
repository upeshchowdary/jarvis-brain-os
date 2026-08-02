"""Knowledge Manager interface for future Vector DB, RAG, PDF, and Document indexing integrations."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field


class DocumentChunk(BaseModel):
    id: str
    content: str
    source: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: Optional[float] = None


class BaseKnowledgeProvider(ABC):
    """Abstract interface contract for future Vector DBs (ChromaDB), RAG, and document indexing."""

    @abstractmethod
    async def index_document(self, document_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Index a document or note into vector storage."""
        pass

    @abstractmethod
    async def query_knowledge(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """Perform semantic search against vector memory."""
        pass

    @abstractmethod
    async def clear_knowledge_base() -> bool:
        """Clear indexed vector store."""
        pass


class KnowledgeManager:
    """Central gateway interface for future RAG and knowledge base integrations."""

    def __init__(self, provider: Optional[BaseKnowledgeProvider] = None) -> None:
        self.provider = provider

    def set_provider(self, provider: BaseKnowledgeProvider) -> None:
        """Inject vector database provider (e.g. ChromaDBProvider)."""
        self.provider = provider

    async def search(self, query: str, top_k: int = 5) -> List[DocumentChunk]:
        """Search knowledge base if provider is active."""
        if not self.provider:
            # Return empty list until vector memory module is attached
            return []
        return await self.provider.query_knowledge(query, top_k=top_k)


# Global knowledge manager singleton
knowledge_manager = KnowledgeManager()
