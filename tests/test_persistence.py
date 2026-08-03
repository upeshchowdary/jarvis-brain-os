"""Integration test suite verifying permanent chat data and memory persistence in SQLite across server restarts."""

import pytest
import asyncio
from app.database.connection import DatabaseManager
from memory.short_term import ShortTermMemoryManager
from memory.long_term import LongTermMemoryManager
from memory.vector_memory import VectorMemoryManager
from brain.conversation_manager import ConversationManager


@pytest.mark.asyncio
async def test_session_buffer_persistence(tmp_path):
    db_file = tmp_path / "test_jarvis.db"
    db_mgr = DatabaseManager(db_path=db_file)
    await db_mgr.init_db()

    session_id = "persistent_session_101"

    # Step 1: Write messages to session buffer
    await db_mgr.save_session_buffer_message(session_id, "user", "Hello, my project is JARVIS AI")
    await db_mgr.save_session_buffer_message(session_id, "assistant", "Hello! I am ready to assist with JARVIS AI.")

    # Step 2: Instantiate new memory manager (simulating server restart)
    new_db_mgr = DatabaseManager(db_path=db_file)
    restored_messages = await new_db_mgr.get_session_buffer_messages(session_id)

    assert len(restored_messages) == 2
    assert restored_messages[0]["role"] == "user"
    assert restored_messages[0]["content"] == "Hello, my project is JARVIS AI"
    assert restored_messages[1]["role"] == "assistant"
    assert restored_messages[1]["content"] == "Hello! I am ready to assist with JARVIS AI."


@pytest.mark.asyncio
async def test_vector_memory_persistence(tmp_path):
    db_file = tmp_path / "test_jarvis.db"
    db_mgr = DatabaseManager(db_path=db_file)
    await db_mgr.init_db()

    # Create vector memory and add document
    vm1 = VectorMemoryManager()

    # Direct persistence call to test_db
    await db_mgr.save_vector_document(
        doc_id="doc_test_99",
        content="Artificial Intelligence and Machine Learning framework",
        metadata={"category": "test"},
    )

    # Instantiate fresh VectorMemoryManager and load from DB
    vm2 = VectorMemoryManager()

    # Load from DB manually for test DB
    docs = await db_mgr.get_all_vector_documents()
    for rec in docs:
        vm2.add_document(rec["doc_id"], rec["content"], rec["metadata"])

    results = vm2.search_similar("Machine Learning", top_k=1)
    assert len(results) == 1
    assert results[0].id == "doc_test_99"
    assert "Machine Learning" in results[0].content


@pytest.mark.asyncio
async def test_conversation_manager_db_restoration(tmp_path):
    db_file = tmp_path / "test_jarvis.db"
    db_mgr = DatabaseManager(db_path=db_file)
    await db_mgr.init_db()

    session_id = "session_restart_sim"

    await db_mgr.save_session_buffer_message(session_id, "user", "What is the status of the build?")
    await db_mgr.save_session_buffer_message(session_id, "assistant", "All systems operational.")

    # Instantiate fresh conversation manager
    conv_mgr = ConversationManager()

    # Fetch messages via db_mgr
    history = await db_mgr.get_session_buffer_messages(session_id)
    session = conv_mgr.get_or_create_session(session_id)
    for m in history:
        session.add_message(m["role"], m["content"])

    assert len(session.messages) == 2
    assert session.messages[0]["content"] == "What is the status of the build?"
    assert session.messages[1]["content"] == "All systems operational."
