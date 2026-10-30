"""Semantic memory: ChromaDB (embedded, in-process) with local fastembed embeddings, plus a keyword fallback.

Nothing here blocks the event loop: chromadb/fastembed are imported and the embedding model is loaded by
``warm_up()`` in a worker thread, and async callers reach Chroma/the model through ``asyncio.to_thread``.
Until embeddings are ready (or if they never are, e.g. offline first run), documents are kept in memory and
SQLite, and ``warm_up()`` indexes whatever Chroma is missing once the model has loaded.
"""

import asyncio
import math
import threading
import time
from collections import OrderedDict
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, Field

from app.database.connection import db_manager
from core.config import settings
from core.logging import logger
from core.paths import data_path

EmbeddingFunction = Callable[[List[str]], List[List[float]]]

COLLECTION_NAME = "jarvis_memory"
QUERY_CACHE_SIZE = 256
INDEX_BATCH = 64


class VectorDocument(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    memory_type: str = Field(default="semantic", description="semantic | episodic | conversation | project | visual_event")
    importance: float = Field(default=0.8, ge=0.0, le=1.0)
    created_at: float = Field(default_factory=time.time)
    score: float = 0.0


def _epoch(value: Any) -> Optional[float]:
    """Epoch seconds from a number or an ISO timestamp (naive = UTC)."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            dt = datetime.fromisoformat(value)
        except ValueError:
            return None
        return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).timestamp()
    return None


class VectorMemoryManager:
    """Semantic vector memory over chat turns, facts and notes, with a keyword-overlap fallback."""

    def __init__(self, persist_dir: Optional[str] = None) -> None:
        self._documents: Dict[str, VectorDocument] = {}
        self._persist_dir = persist_dir
        self._client = None
        self._collection = None
        self._collection_lock = threading.Lock()
        self._embed_fn: Optional[EmbeddingFunction] = None
        self._embed_error: Optional[str] = None
        self._query_cache: "OrderedDict[str, List[float]]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._warm_task: Optional[asyncio.Task] = None
        self._tasks: set = set()

    # ── Embeddings ───────────────────────────────────────────────────────
    def set_embedding_function(self, fn: Optional[EmbeddingFunction]) -> None:
        """Use ``fn`` instead of the fastembed model (tests install a fake here; None = load the model)."""
        self._embed_fn = fn
        self._embed_error = None
        with self._cache_lock:
            self._query_cache.clear()

    @property
    def embeddings_ready(self) -> bool:
        return self._embed_fn is not None

    def _load_model(self) -> EmbeddingFunction:
        """Load the local fastembed model (downloads it once into data/models). Blocking: run in a thread."""
        from fastembed import TextEmbedding

        model = TextEmbedding(settings.EMBEDDING_MODEL, cache_dir=str(data_path("models", "fastembed")))
        return lambda texts: [vec.tolist() for vec in model.embed(texts)]

    def _embed_query(self, fn: EmbeddingFunction, query: str) -> List[float]:
        with self._cache_lock:
            if query in self._query_cache:
                self._query_cache.move_to_end(query)
                return self._query_cache[query]
        vector = list(fn([query])[0])
        with self._cache_lock:
            self._query_cache[query] = vector
            while len(self._query_cache) > QUERY_CACHE_SIZE:
                self._query_cache.popitem(last=False)
        return vector

    async def warm_up(self) -> None:
        """Open Chroma, load the embedding model and run a dummy query off the event loop, then index any
        documents Chroma is missing. On failure semantic search stays off and keyword search is used."""
        try:
            await asyncio.to_thread(self._ensure_collection)
            fn = self._embed_fn
            if fn is None:
                fn = await asyncio.to_thread(self._load_model)
            await asyncio.to_thread(fn, ["warm-up"])
            if self._embed_fn is None:
                self._embed_fn = fn
            indexed = await asyncio.to_thread(self._index_missing_sync)
            logger.info(f"[VectorMemory] semantic search ready ({settings.EMBEDDING_MODEL}); indexed {indexed} documents")
        except Exception as exc:
            self._embed_error = str(exc) or type(exc).__name__
            logger.warning(f"[VectorMemory] semantic search unavailable ({self._embed_error}); keyword search only")

    def start_warm_up(self) -> Optional[asyncio.Task]:
        """Run ``warm_up()`` as a background task (at most one at a time). Needs a running loop."""
        if self._warm_task is None or self._warm_task.done():
            self._warm_task = asyncio.get_running_loop().create_task(self.warm_up())
        return self._warm_task

    # ── Chroma ───────────────────────────────────────────────────────────
    def _ensure_collection(self):
        """Open the persistent Chroma collection. Blocking (imports chromadb): run in a thread."""
        if self._collection is not None:
            return self._collection
        with self._collection_lock:
            if self._collection is None:
                import chromadb
                from chromadb.config import Settings as ChromaSettings

                path = self._persist_dir or str(data_path("chroma_db"))
                self._client = chromadb.PersistentClient(path=path, settings=ChromaSettings(anonymized_telemetry=False))
                self._collection = self._client.get_or_create_collection(
                    COLLECTION_NAME, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None,
                )
                logger.info(f"[VectorMemory] ChromaDB collection '{COLLECTION_NAME}' opened at {path}")
        return self._collection

    @staticmethod
    def _chroma_metadata(doc: VectorDocument) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"memory_type": doc.memory_type, "importance": float(doc.importance)}
        for key, value in doc.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                meta[key] = value
        meta["ts"] = float(doc.created_at)  # numeric, so time windows can filter on it
        return meta

    def _upsert_sync(self, docs: List[VectorDocument]) -> None:
        fn = self._embed_fn
        if fn is None or not docs:
            return
        collection = self._ensure_collection()
        collection.upsert(
            ids=[d.id for d in docs],
            documents=[d.content for d in docs],
            embeddings=fn([d.content for d in docs]),
            metadatas=[self._chroma_metadata(d) for d in docs],
        )

    def _index_missing_sync(self) -> int:
        collection = self._ensure_collection()
        present = set(collection.get(include=[])["ids"])
        missing = [d for d in list(self._documents.values()) if d.id not in present]
        for start in range(0, len(missing), INDEX_BATCH):
            self._upsert_sync(missing[start:start + INDEX_BATCH])
        return len(missing)

    async def _index(self, docs: List[VectorDocument]) -> None:
        if self._embed_fn is None:
            return  # warm_up() indexes it once the model is ready
        try:
            await asyncio.to_thread(self._upsert_sync, docs)
        except Exception as exc:
            logger.debug(f"[VectorMemory] indexing skipped: {exc}")

    def _query_sync(self, query: str, limit: int, where: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
        fn = self._embed_fn
        if fn is None:
            return []
        collection = self._ensure_collection()
        res = collection.query(
            query_embeddings=[self._embed_query(fn, query)], n_results=limit, where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = (res.get("ids") or [[]])[0]
        return [
            {"id": doc_id, "text": res["documents"][0][i], "metadata": res["metadatas"][0][i] or {},
             "distance": res["distances"][0][i]}
            for i, doc_id in enumerate(ids)
        ]

    async def query(self, query: str, limit: int = 10, since_ts: Optional[float] = None,
                    until_ts: Optional[float] = None) -> List[Dict[str, Any]]:
        """Nearest documents to ``query`` (closest first) as {id, text, metadata, distance}, within
        [since_ts, until_ts) when given. Empty while embeddings are unavailable — never raises."""
        if not query:
            return []
        if self._embed_fn is None:
            if self._embed_error is None:
                self.start_warm_up()
            return []
        bounds = []
        if since_ts is not None:
            bounds.append({"ts": {"$gte": since_ts}})
        if until_ts is not None:
            bounds.append({"ts": {"$lt": until_ts}})
        where = bounds[0] if len(bounds) == 1 else ({"$and": bounds} if bounds else None)
        try:
            return await asyncio.to_thread(self._query_sync, query, limit, where)
        except Exception as exc:
            logger.debug(f"[VectorMemory] vector query failed: {exc}")
            return []

    # ── Documents ────────────────────────────────────────────────────────
    def _background(self, coro) -> None:
        task = asyncio.get_running_loop().create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def _store(self, doc_id: str, content: str, metadata: Optional[Dict[str, Any]], memory_type: str,
               importance: float) -> VectorDocument:
        meta = metadata or {}
        doc = VectorDocument(
            id=doc_id, content=content, metadata=meta, memory_type=memory_type, importance=importance,
            created_at=_epoch(meta.get("created_at")) or time.time(),
        )
        self._documents[doc_id] = doc
        logger.info(f"VectorMemoryManager added document '{doc_id}' (type={memory_type}, {len(content)} chars)")
        return doc

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        memory_type: str = "semantic",
        importance: float = 0.8,
    ) -> None:
        """Store a document; indexing and SQLite persistence run in the background when a loop is running."""
        doc = self._store(doc_id, content, metadata, memory_type, importance)
        try:
            asyncio.get_running_loop()
        except RuntimeError:  # plain sync caller: nothing to block, index inline
            try:
                self._upsert_sync([doc])
            except Exception as exc:
                logger.debug(f"[VectorMemory] indexing skipped: {exc}")
            return
        self._background(db_manager.save_vector_document(doc_id, content, metadata or {}))
        self._background(self._index([doc]))

    async def add_document_async(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        memory_type: str = "semantic",
        importance: float = 0.8,
    ) -> None:
        """Store a document, await its indexing (when embeddings are ready) and SQLite persistence."""
        doc = self._store(doc_id, content, metadata, memory_type, importance)
        await self._index([doc])
        await db_manager.save_vector_document(doc_id, content, metadata or {})

    async def load_from_db(self) -> int:
        """Load stored documents from SQLite into memory (fast; ``warm_up()`` indexes them into Chroma)."""
        records = await db_manager.get_all_vector_documents()
        for rec in records:
            meta = rec.get("metadata", {})
            self._documents[rec["doc_id"]] = VectorDocument(
                id=rec["doc_id"],
                content=rec["content"],
                metadata=meta,
                memory_type=meta.get("memory_type", "semantic"),
                importance=float(meta.get("importance", 0.8)),
                created_at=_epoch(meta.get("created_at")) or time.time(),
            )
        logger.info(f"VectorMemoryManager loaded {len(records)} documents from SQLite database.")
        return len(records)

    def search_similar(
        self,
        query: str,
        top_k: int = 5,
        memory_type_filter: Optional[str] = None,
    ) -> List[VectorDocument]:
        """Semantic similarity search when Chroma is ready, otherwise composite keyword ranking.
        Blocking when semantic: async callers should run it via ``asyncio.to_thread``."""
        if not query:
            return []

        # Strategy 1: Chroma (only once warm: this sync path never opens Chroma or loads the model)
        if self._collection is not None and self._embed_fn is not None and self._documents:
            try:
                where = {"memory_type": memory_type_filter} if memory_type_filter else None
                results: List[VectorDocument] = []
                for hit in self._query_sync(query, min(top_k, len(self._documents)), where):
                    doc = self._documents.get(hit["id"])
                    if doc:
                        # cosine distance (0..2) -> similarity (0..1)
                        sim = max(0.0, min(1.0, 1.0 - (hit["distance"] / 2.0)))
                        results.append(doc.model_copy(update={"score": round(sim, 4)}))
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

        for doc in list(self._documents.values()):  # may run in a worker thread
            if memory_type_filter and doc.memory_type != memory_type_filter:
                continue

            doc_words = set(doc.content.lower().split())
            overlap = len(q_words.intersection(doc_words))
            if overlap > 0:
                base_sim = overlap / math.sqrt(len(q_words) * len(doc_words))
                age_days = (now - doc.created_at) / 86400.0
                recency_score = math.exp(-0.1 * age_days)
                final_score = round(base_sim * 0.6 + recency_score * 0.2 + doc.importance * 0.2, 4)
                scored_results.append(doc.model_copy(update={"score": final_score}))

        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results[:top_k]

    def _clear_index_sync(self) -> None:
        collection = self._ensure_collection()
        ids = collection.get(include=[])["ids"]
        if ids:
            collection.delete(ids=ids)

    def clear(self) -> None:
        """Clear all stored vector documents in memory, SQLite, and ChromaDB."""
        self._documents.clear()
        try:
            self._clear_index_sync()
        except Exception as e:
            logger.debug(f"[VectorMemory] ChromaDB clear failed: {e}")

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return
        self._background(db_manager.clear_vector_documents())

    async def clear_async(self) -> None:
        """Clear all stored vector documents and await Chroma and SQLite deletion."""
        self._documents.clear()
        try:
            await asyncio.to_thread(self._clear_index_sync)
        except Exception as e:
            logger.debug(f"[VectorMemory] ChromaDB clear failed: {e}")
        await db_manager.clear_vector_documents()


# Global vector memory manager singleton
vector_memory = VectorMemoryManager()
