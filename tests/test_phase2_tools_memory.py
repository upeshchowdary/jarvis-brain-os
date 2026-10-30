"""Unit tests for Phase 2 Active Tools and Memory Engine."""

import pytest
from tools import ToolContext, load_builtin_tools
from memory.short_term import short_term_memory
from memory.long_term import long_term_memory
from memory.vector_memory import vector_memory

registry = load_builtin_tools()


@pytest.mark.network
async def test_internet_search_tool():
    res = await registry.get("web_search").run(ToolContext(), query="Python Programming Language")
    assert res.success is True
    assert isinstance(res.data, list)


async def test_system_status_tool():
    res = await registry.get("system_status").run(ToolContext())
    assert res.success is True
    assert "cpu_cores" in res.data and "os" in res.data


async def test_files_read_tool(tmp_path):
    f = tmp_path / "test_fs_tool.txt"
    f.write_text("JARVIS FS Test", encoding="utf-8")
    res = await registry.get("files_read").run(ToolContext(), location=str(f))
    assert res.success is True
    assert "JARVIS FS Test" in res.content_for_llm


def test_short_term_memory():
    short_term_memory.append("session_test", "user", "Hello JARVIS")
    history = short_term_memory.get_history("session_test")
    assert len(history) == 1
    short_term_memory.clear("session_test")


@pytest.mark.asyncio
async def test_long_term_memory():
    await long_term_memory.set_fact("user_name", "Upesh", category="user_info")
    name = await long_term_memory.get_fact("user_name")
    assert name == "Upesh"


def test_vector_memory():
    vector_memory.add_document("doc1", "Artificial Intelligence and Machine Learning")
    results = vector_memory.search_similar("Machine Learning", top_k=1)
    assert len(results) == 1
    assert results[0].id == "doc1"
    vector_memory.clear()
