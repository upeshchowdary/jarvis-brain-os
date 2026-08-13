"""Semantic Vector Memory Manager for similarity search, ranking, and memory type categorization."""

import math
import time
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from brain.logger import logger
from app.database.connection import db_manager


class VectorDocument(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    memory_type: str = Field(default="semantic", description="semantic | episodic | conversation | project | visual_event")
    importance: float = Field(default=0.8, ge=0.0, le=1.0)
    created_at: float = Field(default_factory=time.time)
    score: float = 0.0


class VectorMemoryManager:
    """Semantic Vector Memory Manager for similarity search over documents, notes, and past memories with ranking."""

    def __init__(self) -> None:
        self._documents: Dict[str, VectorDocument] = {}

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        memory_type: str = "semantic",
        importance: float = 0.8,
    ) -> None:
        """Store document entry into vector memory and trigger persistence to SQLite."""
        meta = metadata or {}
        doc = VectorDocument(
            id=doc_id,
            content=content,
            metadata=meta,
            memory_type=memory_type,
            importance=importance,
            created_at=time.time(),
        )
        self._documents[doc_id] = doc
        logger.info(f"VectorMemoryManager added document '{doc_id}' (type={memory_type}, {len(content)} chars)")

        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.create_task(db_manager.save_vector_document(doc_id, content, meta))
        except RuntimeError:
            pass

    async def add_document_async(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        memory_type: str = "semantic",
        importance: float = 0.8,
    ) -> None:
        """Store document entry into vector memory and await SQLite persistence."""
        self.add_document(doc_id, content, metadata, memory_type=memory_type, importance=importance)
        await db_manager.save_vector_document(doc_id, content, metadata or {})

    async def load_from_db(self) -> int:
        """Load all stored vector documents from SQLite database into memory."""
        records = await db_manager.get_all_vector_documents()
        for rec in records:
            doc_id = rec["doc_id"]
            meta = rec.get("metadata", {})
            self._documents[doc_id] = VectorDocument(
                id=doc_id,
                content=rec["content"],
                metadata=meta,
                memory_type=meta.get("memory_type", "semantic"),
                importance=float(meta.get("importance", 0.8)),
                created_at=float(meta.get("created_at", time.time())),
            )
        logger.info(f"VectorMemoryManager loaded {len(records)} documents from SQLite database.")
        return len(records)

    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        memory_type_filter: Optional[str] = None,
    ) -> List[VectorDocument]:
        """Perform similarity, recency, importance, and memory-type ranked search."""
        if not query or not self._documents:
            return []

        q_words = set(query.lower().split())
        now = time.time()
        scored_results: List[VectorDocument] = []

        for doc in self._documents.values():
            if memory_type_filter and doc.memory_type != memory_type_filter:
                continue

            doc_words = set(doc.content.lower().split())
            overlap = len(q_words.intersection(doc_words))
            if overlap > 0:
                # 1. Base keyword similarity
                base_sim = overlap / math.sqrt(len(q_words) * len(doc_words))
                
                # 2. Recency decay (half-life of ~7 days)
                age_days = (now - doc.created_at) / 86400.0
                recency_score = math.exp(-0.1 * age_days)

                # 3. Composite rank score = similarity * 0.6 + recency * 0.2 + importance * 0.2
                final_score = round(base_sim * 0.6 + recency_score * 0.2 + doc.importance * 0.2, 4)

                scored_doc = VectorDocument(
                    id=doc.id,
                    content=doc.content,
                    metadata=doc.metadata,
                    memory_type=doc.memory_type,
                    importance=doc.importance,
                    created_at=doc.created_at,
                    score=final_score,
                )
                scored_results.append(scored_doc)

        # Sort by final score descending
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
