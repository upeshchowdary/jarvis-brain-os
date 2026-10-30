import math
import time
import asyncio
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field

from core.logging import logger
from core.paths import data_path
from app.database.connection import db_manager

try:
    import chromadb
    HAS_CHROMADB = True
except ImportError:
    HAS_CHROMADB = False


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

    def __init__(self, persist_dir: Optional[str] = None) -> None:
        self._documents: Dict[str, VectorDocument] = {}
        self._chroma_client = None
        self._collection = None

        if HAS_CHROMADB:
            try:
                db_path = persist_dir or str(data_path("chroma_db"))
                self._chroma_client = chromadb.PersistentClient(path=db_path)
                self._collection = self._chroma_client.get_or_create_collection(
                    name="jarvis_semantic_memory",
                    metadata={"description": "JARVIS long-term semantic memory embeddings"}
                )
                logger.info(f"[VectorMemory] ChromaDB vector store initialized at {db_path}")
            except Exception as e:
                logger.warning(f"[VectorMemory] ChromaDB init failed ({e}), falling back to in-memory index")
                self._chroma_client = None
                self._collection = None

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        memory_type: str = "semantic",
        importance: float = 0.8,
    ) -> None:
        """Store document entry into vector memory and trigger persistence to SQLite and ChromaDB."""
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

        # Sync to ChromaDB if active
        if self._collection is not None:
            try:
                chroma_meta = {
                    "memory_type": str(memory_type),
                    "importance": float(importance),
                    "created_at": float(doc.created_at),
                }
                for k, v in meta.items():
                    if isinstance(v, (str, int, float, bool)):
                        chroma_meta[k] = v
                self._collection.upsert(
                    ids=[doc_id],
                    documents=[content],
                    metadatas=[chroma_meta]
                )
            except Exception as e:
                logger.debug(f"[VectorMemory] ChromaDB upsert error: {e}")

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
        """Load all stored vector documents from SQLite database into memory and ChromaDB."""
        records = await db_manager.get_all_vector_documents()
        for rec in records:
            doc_id = rec["doc_id"]
            meta = rec.get("metadata", {})
            m_type = meta.get("memory_type", "semantic")
            imp = float(meta.get("importance", 0.8))
            c_at = float(meta.get("created_at", time.time()))
            
            self._documents[doc_id] = VectorDocument(
                id=doc_id,
                content=rec["content"],
                metadata=meta,
                memory_type=m_type,
                importance=imp,
                created_at=c_at,
            )
            if self._collection is not None:
                try:
                    chroma_meta = {"memory_type": str(m_type), "importance": imp, "created_at": c_at}
                    self._collection.upsert(
                        ids=[doc_id],
                        documents=[rec["content"]],
                        metadatas=[chroma_meta]
                    )
                except Exception:
                    pass

        logger.info(f"VectorMemoryManager loaded {len(records)} documents from SQLite database.")
        return len(records)

    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        memory_type_filter: Optional[str] = None,
    ) -> List[VectorDocument]:
        """Perform semantic similarity search using ChromaDB when available, falling back to composite ranking."""
        if not query:
            return []

        # Strategy 1: ChromaDB Semantic Embedding Search
        if self._collection is not None and len(self._documents) > 0:
            try:
                where_clause = {"memory_type": memory_type_filter} if memory_type_filter else None
                res = self._collection.query(
                    query_texts=[query],
                    n_results=min(top_k, len(self._documents)),
                    where=where_clause
                )
                if res and res.get("ids") and res["ids"][0]:
                    results: List[VectorDocument] = []
                    for idx, doc_id in enumerate(res["ids"][0]):
                        doc = self._documents.get(doc_id)
                        if doc:
                            dist = res["distances"][0][idx] if "distances" in res and res["distances"] else 0.5
                            # Convert cosine distance (0 to 2) to similarity (0 to 1)
                            sim_score = max(0.0, min(1.0, 1.0 - (dist / 2.0)))
                            results.append(
                                VectorDocument(
                                    id=doc.id,
                                    content=doc.content,
                                    metadata=doc.metadata,
                                    memory_type=doc.memory_type,
                                    importance=doc.importance,
                                    created_at=doc.created_at,
                                    score=round(sim_score, 4),
                                )
                            )
                    if results:
                        return results
            except Exception as e:
                logger.debug(f"[VectorMemory] ChromaDB query failed ({e}), using keyword fallback")

        # Strategy 2: Heuristic keyword & recency ranked fallback
        if not self._documents:
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
                base_sim = overlap / math.sqrt(len(q_words) * len(doc_words))
                age_days = (now - doc.created_at) / 86400.0
                recency_score = math.exp(-0.1 * age_days)
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

        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results[:top_k]

    def clear(self) -> None:
        """Clear all stored vector documents in memory, SQLite, and ChromaDB."""
        self._documents.clear()
        if self._chroma_client is not None:
            try:
                self._chroma_client.delete_collection("jarvis_semantic_memory")
                self._collection = self._chroma_client.get_or_create_collection(
                    name="jarvis_semantic_memory",
                    metadata={"description": "JARVIS long-term semantic memory embeddings"}
                )
            except Exception as e:
                logger.debug(f"[VectorMemory] ChromaDB clear failed: {e}")

        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                loop.create_task(db_manager.clear_vector_documents())
        except RuntimeError:
            pass

    async def clear_async(self) -> None:
        """Clear all stored vector documents and await SQLite deletion."""
        self.clear()
        await db_manager.clear_vector_documents()


# Global vector memory manager singleton
vector_memory = VectorMemoryManager()

