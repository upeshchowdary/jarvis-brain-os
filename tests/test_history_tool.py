"""chat_history tool: local-day boundaries, search, channel filter, formatting."""

from datetime import datetime, timedelta, timezone

import pytest

from app.database.connection import DatabaseManager
from memory import search as memory_search
from memory.vector_memory import vector_memory
from tests.test_memory_search import log_turn, vector_memory_module
from tools.base import ToolContext
from tools.builtin import history

IST = timezone(timedelta(hours=5, minutes=30))
NOW = datetime(2026, 10, 30, 10, 0, tzinfo=IST)


@pytest.fixture
async def db(tmp_path, monkeypatch):
    fresh = DatabaseManager(db_path=tmp_path / "history.db")
    await fresh.init_db()
    monkeypatch.setattr(memory_search, "db_manager", fresh)
    monkeypatch.setattr(vector_memory_module, "db_manager", fresh)
    monkeypatch.setattr(history, "_local_tz", lambda: IST)
    monkeypatch.setattr(history, "_now", lambda: NOW)
    await vector_memory.clear_async()
    yield fresh
    await vector_memory.clear_async()


def utc(s: str) -> datetime:
    return datetime.fromisoformat(s)


async def run(**kwargs):
    return await history.ChatHistory().run(ToolContext(), **kwargs)


async def test_today_filter_uses_local_midnight(db):
    await log_turn(db, "after midnight ist", "ok A", created_at=utc("2026-10-29T19:00:00+00:00"))
    await log_turn(db, "before midnight ist", "ok B", created_at=utc("2026-10-29T18:00:00+00:00"))
    res = await run(since="today")
    assert res.success
    assert "after midnight ist" in res.message
    assert "before midnight ist" not in res.message
    assert res.message.startswith("[00:30] You: after midnight ist → JARVIS: ok A")
    assert res.content_for_llm == res.message


async def test_query_uses_search(db):
    await log_turn(db, "send a whatsapp to Nithesh", "Sent on WhatsApp.", created_at=NOW - timedelta(hours=2))
    await log_turn(db, "what is the weather", "Sunny.", created_at=NOW - timedelta(hours=1))
    res = await run(query="whatsapp")
    assert res.success
    assert res.message.splitlines()[0].count("whatsapp") >= 1
    assert "weather" not in res.message.splitlines()[0]


async def test_empty_history_message(db):
    res = await run()
    assert res.success
    assert res.message == "No chats found for that period."


async def test_unknown_since_is_error(db):
    res = await run(since="last century")
    assert not res.success
    assert "today" in res.message and "YYYY" not in res.message.split("since")[0]


async def test_bad_limit_is_error_and_clamped(db):
    assert not (await run(limit="many")).success
    for i in range(3):
        await log_turn(db, f"turn {i}", "r", created_at=NOW - timedelta(minutes=10 - i))
    res = await run(limit=1000)
    assert res.success and len(res.message.splitlines()) == 3


async def test_channel_filter_with_and_without_query(db):
    await log_turn(db, "voice whatsapp thing", "r1", created_at=NOW - timedelta(hours=3), channel="voice")
    await log_turn(db, "typed whatsapp thing", "r2", created_at=NOW - timedelta(hours=2), channel="text")
    res = await run(channel="voice")
    assert "voice whatsapp" in res.message and "typed whatsapp" not in res.message
    res = await run(query="whatsapp", channel="text")
    assert "typed whatsapp" in res.message and "voice whatsapp" not in res.message


async def test_multi_day_range_shows_dates_and_newest_last(db):
    await log_turn(db, "old one", "r", created_at=utc("2026-10-29T18:00:00+00:00"))  # Oct 29 23:30 IST
    await log_turn(db, "new one", "r", created_at=utc("2026-10-30T02:00:00+00:00"))  # Oct 30 07:30 IST
    res = await run(since="this week")
    lines = res.message.splitlines()
    assert lines[0].startswith("[Oct 29 23:30]") and "old one" in lines[0]
    assert lines[1].startswith("[Oct 30 07:30]") and "new one" in lines[1]


async def test_single_day_has_no_date_and_text_is_clipped(db):
    await log_turn(db, "x" * 400, "y" * 400, created_at=NOW - timedelta(hours=1))
    res = await run(since="today")
    assert res.message.startswith("[09:00] You: ")
    assert len(res.message) < 400


async def test_yesterday_and_explicit_date(db):
    await log_turn(db, "yesterday turn", "r", created_at=utc("2026-10-29T06:00:00+00:00"))
    await log_turn(db, "today turn", "r", created_at=NOW - timedelta(hours=1))
    assert "yesterday turn" in (await run(since="yesterday")).message
    assert "today turn" not in (await run(since="yesterday")).message
    assert "yesterday turn" in (await run(since="2026-10-29")).message
    assert (await run(since="last week")).message == "No chats found for that period."
