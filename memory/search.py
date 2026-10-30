"""Hybrid memory search: SQLite FTS5 keyword hits over chat turns + Chroma vector hits over turns, facts and
notes, fused with reciprocal rank fusion (k=60) and a recency tie-break.

If embeddings are unavailable (model not loaded yet, offline first run, fastembed missing) the vector side is
empty and search degrades to FTS5 only.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

from app.database.connection import db_manager
from memory.vector_memory import content_words, turn_document, turn_text, vector_memory

RRF_K = 60


@dataclass
class Hit:
    chat_id: Optional[int]
    text: str
    created_at: str  # UTC ISO
    score: float
    source: str  # fts | vector | fts+vector | recent


async def index_turn(chat_id: int, query: str, response: str, *, created_at: str, channel: str = "text",
                     session_id: Optional[str] = None) -> None:
    """Add one chat turn (already in chat_logs, hence in FTS) to the vector index."""
    doc = turn_document(chat_id, query, response, created_at=created_at, channel=channel, session_id=session_id)
    await vector_memory.add_document_async(doc.id, doc.content, doc.metadata, memory_type=doc.memory_type)


def _utc(dt: Optional[datetime]) -> Optional[datetime]:
    if dt is None:
        return None
    return (dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def _terms(query: str) -> List[str]:
    return list(dict.fromkeys(content_words(query)))


async def search(query: str, limit: int = 10, since_utc: Optional[datetime] = None,
                 until_utc: Optional[datetime] = None) -> List[Hit]:
    """Best matches for ``query`` across chat turns, facts and notes, within [since_utc, until_utc)."""
    query = (query or "").strip()
    if not query:
        return []
    since, until = _utc(since_utc), _utc(until_utc)
    pool = max(limit * 3, 20)
    fts_rows, vector_rows = await asyncio.gather(
        db_manager.search_chat_fts(_terms(query), pool, since.isoformat() if since else None,
                                   until.isoformat() if until else None),
        vector_memory.query(query, pool, since.timestamp() if since else None, until.timestamp() if until else None),
    )

    fused: Dict[str, Hit] = {}
    for rank, row in enumerate(fts_rows, 1):
        fused[f"chat:{row['id']}"] = Hit(row["id"], turn_text(row["user_query"], row["response_text"]),
                                         row["created_at"], 1.0 / (RRF_K + rank), "fts")
    for rank, row in enumerate(vector_rows, 1):
        meta = row["metadata"]
        chat_id = meta.get("chat_id")
        key = f"chat:{chat_id}" if chat_id is not None else f"doc:{row['id']}"
        if key in fused:
            fused[key].score += 1.0 / (RRF_K + rank)
            fused[key].source = "fts+vector"
            continue
        created = meta.get("created_at")
        if not isinstance(created, str):
            created = datetime.fromtimestamp(float(meta.get("ts", 0.0)), timezone.utc).isoformat()
        fused[key] = Hit(chat_id, row["text"], created, 1.0 / (RRF_K + rank), "vector")

    hits = sorted(fused.values(), key=lambda h: h.created_at, reverse=True)
    hits.sort(key=lambda h: h.score, reverse=True)  # stable: equal scores stay newest first
    return hits[:limit]


async def recent(limit: int, since_utc: Optional[datetime] = None, until_utc: Optional[datetime] = None,
                 channel: Optional[str] = None) -> List[Hit]:
    """Latest chat turns, newest first, within [since_utc, until_utc) and optionally on one channel."""
    since, until = _utc(since_utc), _utc(until_utc)
    rows = await db_manager.recent_chats(limit, since.isoformat() if since else None,
                                         until.isoformat() if until else None, channel)
    return [Hit(row["id"], turn_text(row["user_query"], row["response_text"]), row["created_at"], 0.0, "recent")
            for row in rows]
