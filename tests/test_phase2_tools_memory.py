"""Unit tests for Phase 2 Active Tools and Memory Engine."""

import pytest
from tools.search_tool import InternetSearchTool
from tools.scraper_tool import WebScraperTool
from tools.filesystem_tool import FileSystemTool
from tools.system_tool import SystemInfoTool
from memory.short_term import short_term_memory
from memory.long_term import long_term_memory
from memory.vector_memory import vector_memory


@pytest.mark.asyncio
async def test_internet_search_tool():
    tool = InternetSearchTool()
    res = await tool.execute(query="Python Programming Language", max_results=2)
    assert res["success"] is True
    assert isinstance(res["results"], list)


@pytest.mark.asyncio
async def test_system_info_tool():
    tool = SystemInfoTool()
    res = await tool.execute()
    assert res["success"] is True
    assert "os" in res
    assert "cpu_cores" in res


@pytest.mark.asyncio
async def test_filesystem_tool():
    tool = FileSystemTool()
    # Test file write and read
    write_res = await tool.execute(action="write", path="scratch/test_fs_tool.txt", content="JARVIS FS Test")
    assert write_res["success"] is True

    read_res = await tool.execute(action="read", path="scratch/test_fs_tool.txt")
    assert read_res["success"] is True
    assert read_res["content"] == "JARVIS FS Test"


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
