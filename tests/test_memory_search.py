"""Hybrid memory search: SQLite FTS5 keyword hits + Chroma vector hits fused with reciprocal rank fusion.

Embeddings are always the deterministic fake from conftest (or a stand-in defined here): no model download.
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

import pytest

from app.database.connection import DatabaseManager
from app.database.models import ChatLogRecord
from memory import search as memory_search
from memory.vector_memory import vector_memory
from tests.conftest import fake_embedding


@pytest.fixture
async def db(tmp_path, monkeypatch):
    """A fresh chat database for search, and an empty vector collection."""
    fresh = DatabaseManager(db_path=tmp_path / "search.db")
    await fresh.init_db()
    monkeypatch.setattr(memory_search, "db_manager", fresh)
    await vector_memory.clear_async()
    yield fresh
    await vector_memory.clear_async()


async def log_turn(db: DatabaseManager, query: str, response: str, *, created_at: Optional[datetime] = None,
                   channel: str = "text", index: bool = True) -> int:
    when = (created_at or datetime.now(timezone.utc)).isoformat()
    chat_id = await db.log_chat_telemetry(ChatLogRecord(
        session_id="s1", user_query=query, intent="conversation", confidence=0.9, provider="test", model="test",
        latency_ms=1.0, response_text=response, created_at=when, channel=channel,
    ))
    assert chat_id
    if index:
        await memory_search.index_turn(chat_id, query, response, created_at=when, channel=channel)
    return chat_id


async def test_fts_finds_exact_name(db):
    await log_turn(db, "message Nithesh on WhatsApp", "Sent it to Nithesh.")
    await log_turn(db, "play lofi", "Playing lofi beats.")

    hits = await memory_search.search("Nithesh")

    assert hits
    assert "Nithesh" in hits[0].text
    assert hits[0].chat_id is not None


async def test_rrf_merges_both_sources(db):
    both = await log_turn(db, "what is the quantum telescope budget", "The quantum telescope budget is 4 million.")
    fts_only = await log_turn(db, "the old telescope in the attic", "Noted.", index=False)
    await vector_memory.add_document_async(
        doc_id="note_observatory", content="Observatory planning notes about quantum optics",
        metadata={"kind": "note"}, memory_type="semantic",
    )

    hits = await memory_search.search("quantum telescope", limit=5)

    assert hits[0].chat_id == both
    assert hits[0].source == "fts+vector"
    top5 = hits[:5]
    assert any(h.chat_id == fts_only and h.source == "fts" for h in top5)
    assert any(h.chat_id is None and "Observatory" in h.text and h.source == "vector" for h in top5)


async def test_time_window_filters(db):
    now = datetime.now(timezone.utc)
    await log_turn(db, "plan the holiday budget", "Old budget plan.", created_at=now - timedelta(days=3))
    recent_id = await log_turn(db, "update the holiday budget", "New budget plan.", created_at=now - timedelta(hours=1))

    hits = await memory_search.search("holiday budget", since_utc=now - timedelta(days=1))

    assert [h.chat_id for h in hits] == [recent_id]

    older = await memory_search.search("holiday budget", until_utc=now - timedelta(days=1))
    assert recent_id not in [h.chat_id for h in older]
    assert len(older) == 1


async def test_fts_only_when_embeddings_unavailable(db, monkeypatch):
    def broken(texts):
        raise RuntimeError("embedding model unavailable")

    vector_memory.set_embedding_function(broken)
    await log_turn(db, "message Nithesh on WhatsApp", "Sent it to Nithesh.")  # indexing fails quietly

    hits = await memory_search.search("Nithesh")

    assert hits and "Nithesh" in hits[0].text
    assert all(h.source == "fts" for h in hits)

    # Offline first run / fastembed import error: warm-up gives up quietly, search stays FTS-only.
    def no_fastembed():
        raise ImportError("No module named 'fastembed'")

    vector_memory.set_embedding_function(None)
    monkeypatch.setattr(vector_memory, "_load_model", no_fastembed)
    await vector_memory.warm_up()

    hits = await memory_search.search("Nithesh")
    assert hits and "Nithesh" in hits[0].text
    assert all(h.source == "fts" for h in hits)


async def test_warm_up_does_not_block(db):
    def slow(texts):
        time.sleep(0.5)
        return fake_embedding(texts)

    vector_memory.set_embedding_function(slow)
    task = asyncio.create_task(vector_memory.warm_up())
    await asyncio.sleep(0)  # let warm-up start

    await asyncio.wait_for(asyncio.sleep(0.05), 0.2)

    assert not task.done()
    await task


async def test_recent_filters_channel_newest_first(db):
    now = datetime.now(timezone.utc)
    first = await log_turn(db, "open notepad", "Opened Notepad.", created_at=now - timedelta(minutes=5), channel="voice")
    await log_turn(db, "what time is it", "It's 9 PM.", created_at=now - timedelta(minutes=3), channel="text")
    last = await log_turn(db, "close notepad", "Closed Notepad.", created_at=now - timedelta(minutes=1), channel="voice")

    voice = await memory_search.recent(10, channel="voice")
    assert [h.chat_id for h in voice] == [last, first]
    assert "close notepad" in voice[0].text

    windowed = await memory_search.recent(10, since_utc=now - timedelta(minutes=4))
    assert len(windowed) == 2


async def test_persist_indexes_every_turn_with_channel(db, monkeypatch):
    import app.database.connection as connection
    from brain.pipeline import TurnResult, brain

    monkeypatch.setattr(connection, "db_manager", memory_search.db_manager)
    result = TurnResult(query="open the Spotify app", response="Opened Spotify.", session_id="s9",
                        route="fast_path", intent="open_app")
    result.metadata["channel"] = "voice"

    await brain._persist(result)

    turns = await memory_search.recent(5, channel="voice")
    assert [h.text for h in turns] == ["User: open the Spotify app | JARVIS: Opened Spotify."]
    hits = await memory_search.search("Spotify")
    assert hits[0].chat_id == turns[0].chat_id and hits[0].source == "fts+vector"
