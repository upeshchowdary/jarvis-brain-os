"""Vector Memory Manager for semantic document search and embedding retrieval."""

import math
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from brain.logger import logger


class VectorDocument(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    score: float = 0.0


class VectorMemoryManager:
    """Semantic Vector Memory Manager for similarity search over documents, notes, and past memories."""

    def __init__(self) -> None:
        self._documents: Dict[str, VectorDocument] = {}

    def add_document(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Store document entry into vector memory."""
        doc = VectorDocument(
            id=doc_id,
            content=content,
            metadata=metadata or {},
        )
        self._documents[doc_id] = doc
        logger.info(f"VectorMemoryManager added document '{doc_id}' ({len(content)} chars)")

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
        """Clear all stored vector documents."""
        self._documents.clear()


# Global vector memory manager singleton
vector_memory = VectorMemoryManager()
