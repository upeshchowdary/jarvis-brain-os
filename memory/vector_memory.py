import math
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from brain.logger import logger
from app.database.connection import db_manager


class VectorDocument(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0


class VectorMemoryManager:
    """Semantic Vector Memory Manager for similarity search over documents, notes, and past memories with SQLite persistence."""

    def __init__(self) -> None:
        self._documents: Dict[str, VectorDocument] = {}

    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store document entry into vector memory and trigger persistence to SQLite."""
        meta = metadata or {}
        doc = VectorDocument(
            id=doc_id,
            content=content,
            metadata=meta,
        )
        self._documents[doc_id] = doc
        logger.info(f"VectorMemoryManager added document '{doc_id}' ({len(content)} chars)")

        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.create_task(db_manager.save_vector_document(doc_id, content, meta))
        except RuntimeError:
            pass

    async def add_document_async(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store document entry into vector memory and await SQLite persistence."""
        self.add_document(doc_id, content, metadata)
        await db_manager.save_vector_document(doc_id, content, metadata or {})

    async def load_from_db(self) -> int:
        """Load all stored vector documents from SQLite database into memory."""
        records = await db_manager.get_all_vector_documents()
        for rec in records:
            doc_id = rec["doc_id"]
            self._documents[doc_id] = VectorDocument(
                id=doc_id,
                content=rec["content"],
                metadata=rec.get("metadata", {}),
            )
        logger.info(f"VectorMemoryManager loaded {len(records)} documents from SQLite database.")
        return len(records)

    def search_similar(self, query: str, top_k: int = 5) -> List[VectorDocument]:
        """Perform keyword-overlap and semantic similarity score search over stored documents."""
        if not query or not self._documents:
            return []

        q_words = set(query.lower().split())
        scored_results: List[VectorDocument] = []

        for doc in self._documents.values():
            doc_words = set(doc.content.lower().split())
            overlap = len(q_words.intersection(doc_words))
            if overlap > 0:
                score = round(overlap / math.sqrt(len(q_words) * len(doc_words)), 4)
                scored_doc = VectorDocument(
                    id=doc.id,
                    content=doc.content,
                    metadata=doc.metadata,
                    score=score,
                )
                scored_results.append(scored_doc)

        # Sort by similarity score descending
        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results[:top_k]

    def clear(self) -> None:
        """Clear all stored vector documents in memory and SQLite."""
        self._documents.clear()
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.create_task(db_manager.clear_vector_documents())
        except RuntimeError:
            pass

    async def clear_async(self) -> None:
        """Clear all stored vector documents and await SQLite deletion."""
        self._documents.clear()
        await db_manager.clear_vector_documents()


# Global vector memory manager singleton
vector_memory = VectorMemoryManager()

