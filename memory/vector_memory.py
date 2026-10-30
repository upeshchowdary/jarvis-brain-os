"""Semantic memory: ChromaDB (embedded, in-process) with local fastembed embeddings, plus a keyword fallback.

Nothing here blocks the event loop: chromadb/fastembed are imported and the embedding model is loaded by
``warm_up()`` in a worker thread, and async callers reach Chroma/the model through ``asyncio.to_thread``.
Until embeddings are ready (or if they never are, e.g. offline first run), documents are kept in memory and
SQLite, and ``warm_up()`` indexes whatever Chroma is missing once the model has loaded.
"""

import asyncio
import re
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

QUERY_CACHE_SIZE = 256
INDEX_BATCH = 64
BACKFILL_LIMIT = 5000  # most recent chat turns embedded by warm_up() if Chroma lacks them

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "what", "how", "why", "who", "where", "when", "which",
    "can", "could", "you", "me", "my", "your", "this", "that", "it", "to", "in", "on", "of", "for", "with",
    "do", "did", "does", "i", "we", "about", "have", "has", "had", "and", "or", "at", "by", "from", "please",
}


def collection_name(model: str) -> str:
    """One Chroma collection per embedding model: switching EMBEDDING_MODEL starts a fresh index."""
    slug = re.sub(r"[^a-z0-9._-]+", "-", model.lower()).strip("-._")
    return f"jarvis_memory_{slug}"[:500]


def turn_text(query: str, response: str) -> str:
    """How a chat turn is indexed and shown."""
    return f"User: {query} | JARVIS: {response[:300]}"


def content_words(text: str) -> List[str]:
    return [w for w in re.findall(r"\w+", text.lower()) if len(w) > 1 and w not in STOP_WORDS]


class VectorDocument(BaseModel):
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    memory_type: str = Field(default="semantic", description="semantic | episodic | conversation | project | visual_event")
    importance: float = Field(default=0.8, ge=0.0, le=1.0)
    created_at: float = Field(default_factory=time.time)
    score: float = 0.0


def turn_document(chat_id: int, query: str, response: str, *, created_at: str, channel: str = "text",
                  session_id: Optional[str] = None) -> VectorDocument:
    """The vector document for one chat turn (id ``chat_{chat_id}``)."""
    meta: Dict[str, Any] = {"chat_id": chat_id, "created_at": created_at, "channel": channel}
    if session_id:
        meta["session_id"] = session_id
    return VectorDocument(id=f"chat_{chat_id}", content=turn_text(query, response), metadata=meta,
                          memory_type="conversation", created_at=_epoch(created_at) or time.time())


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
        self._runtime_warned = False
        self._query_cache: "OrderedDict[str, List[float]]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._warm_task: Optional[asyncio.Task] = None
        self._tasks: set = set()

    # ── Embeddings ───────────────────────────────────────────────────────
    def set_embedding_function(self, fn: Optional[EmbeddingFunction]) -> None:
        """Use ``fn`` instead of the fastembed model (tests install a fake here; None = load the model)."""
        self._embed_fn = fn
        self._embed_error = None
        self._runtime_warned = False
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
        """Open Chroma, load the embedding model and run a dummy query off the event loop, then index the
        documents and recent chat turns Chroma is missing. The model is published only once all of that
        succeeded; on any failure semantic search is off (logged once) and keyword search is used."""
        try:
            await asyncio.to_thread(self._ensure_collection)
            fn = self._embed_fn or await asyncio.to_thread(self._load_model)
            await asyncio.to_thread(fn, ["warm-up"])
            turns = await db_manager.recent_chats(BACKFILL_LIMIT)
            indexed = await asyncio.to_thread(self._index_missing_sync, fn, turns)
        except Exception as exc:
            self._embed_fn = None
            self._embed_error = str(exc) or type(exc).__name__
            logger.warning(f"[VectorMemory] semantic search unavailable ({self._embed_error}); keyword search only")
            return
        self._embed_fn = fn
        logger.info(f"[VectorMemory] semantic search ready ({settings.EMBEDDING_MODEL}); indexed {indexed} documents")

    def _runtime_failure(self, action: str, exc: Exception) -> None:
        """A Chroma/embedding error after warm-up: WARNING the first time, debug afterwards."""
        if self._runtime_warned:
            logger.debug(f"[VectorMemory] {action} failed: {exc}")
            return
        self._runtime_warned = True
        logger.warning(f"[VectorMemory] {action} failed ({exc}); semantic results may be missing, "
                       "keyword search still works (further failures logged at debug)")

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
                name = collection_name(settings.EMBEDDING_MODEL)
                self._client = chromadb.PersistentClient(path=path, settings=ChromaSettings(anonymized_telemetry=False))
                self._collection = self._client.get_or_create_collection(
                    name, configuration={"hnsw": {"space": "cosine"}}, embedding_function=None,
                )
                logger.info(f"[VectorMemory] ChromaDB collection '{name}' opened at {path}")
        return self._collection

    @staticmethod
    def _chroma_metadata(doc: VectorDocument) -> Dict[str, Any]:
        meta: Dict[str, Any] = {"memory_type": doc.memory_type, "importance": float(doc.importance)}
        for key, value in doc.metadata.items():
            if isinstance(value, (str, int, float, bool)):
                meta[key] = value
        meta["ts"] = float(doc.created_at)  # numeric, so time windows can filter on it
        return meta

    def _upsert_sync(self, docs: List[VectorDocument], fn: Optional[EmbeddingFunction] = None) -> None:
        fn = fn or self._embed_fn
        if fn is None or not docs:
            return
        collection = self._ensure_collection()
        collection.upsert(
            ids=[d.id for d in docs],
            documents=[d.content for d in docs],
            embeddings=fn([d.content for d in docs]),
            metadatas=[self._chroma_metadata(d) for d in docs],
        )

    def _index_missing_sync(self, fn: EmbeddingFunction, turns: List[Dict[str, Any]]) -> int:
        """Embed the in-memory documents and chat turns (chat_logs rows) that Chroma doesn't have yet."""
        collection = self._ensure_collection()
        present = set(collection.get(include=[])["ids"])
        missing = {d.id: d for d in list(self._documents.values()) if d.id not in present}
        for row in turns:
            doc = turn_document(row["id"], row["user_query"], row["response_text"], created_at=row["created_at"],
                                channel=row.get("channel") or "text", session_id=row.get("session_id"))
            if doc.id not in present and doc.id not in missing:
                missing[doc.id] = doc
        docs = list(missing.values())
        for start in range(0, len(docs), INDEX_BATCH):
            self._upsert_sync(docs[start:start + INDEX_BATCH], fn)
        return len(docs)

    async def _index(self, docs: List[VectorDocument]) -> None:
        if self._embed_fn is None:
            return  # warm_up() indexes it once the model is ready
        try:
            await asyncio.to_thread(self._upsert_sync, docs)
        except Exception as exc:
            self._runtime_failure("indexing", exc)

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
            self._runtime_failure("vector query", exc)
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
                self._runtime_failure("indexing", exc)
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
        exclude_session: Optional[str] = None,
    ) -> List[VectorDocument]:
        """Most similar documents, scored 0..1: cosine similarity when Chroma is ready, otherwise the share of
        the query's content words found in the document. ``exclude_session`` drops that session's documents.
        Blocking when semantic: async callers should run it via ``asyncio.to_thread``."""
        if not query:
            return []

        def wanted(meta: Dict[str, Any], memory_type: Optional[str]) -> bool:
            if memory_type_filter and memory_type != memory_type_filter:
                return False
            return not (exclude_session and meta.get("session_id") == exclude_session)

        # Strategy 1: Chroma (only once warm: this sync path never opens Chroma or loads the model)
        if self._collection is not None and self._embed_fn is not None:
            try:
                where = {"memory_type": memory_type_filter} if memory_type_filter else None
                fetch = top_k * 3 if exclude_session else top_k
                results: List[VectorDocument] = []
                for hit in self._query_sync(query, fetch, where):
                    meta = hit["metadata"]
                    if not wanted(meta, meta.get("memory_type")):
                        continue
                    doc = self._documents.get(hit["id"]) or VectorDocument(
                        id=hit["id"], content=hit["text"] or "", metadata=meta,
                        memory_type=str(meta.get("memory_type", "semantic")),
                        importance=float(meta.get("importance", 0.8)), created_at=float(meta.get("ts", time.time())),
                    )
                    sim = max(0.0, min(1.0, 1.0 - hit["distance"]))  # cosine distance -> cosine similarity
                    results.append(doc.model_copy(update={"score": round(sim, 4)}))
                return results[:top_k]
            except Exception as e:
                self._runtime_failure("vector search", e)

        # Strategy 2: keyword fallback — share of the query's content words present in the document
        q_words = set(content_words(query))
        if not q_words:
            return []
        scored_results: List[VectorDocument] = []
        for doc in list(self._documents.values()):  # may run in a worker thread
            if not wanted(doc.metadata, doc.memory_type):
                continue
            overlap = len(q_words.intersection(content_words(doc.content)))
            if overlap:
                scored_results.append(doc.model_copy(update={"score": round(overlap / len(q_words), 4)}))

        scored_results.sort(key=lambda x: (x.score, x.created_at), reverse=True)
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
