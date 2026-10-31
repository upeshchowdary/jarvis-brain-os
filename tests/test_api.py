"""Integration tests for FastAPI REST API endpoints."""

def test_health_endpoint(sync_client):
    response = sync_client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "active_provider" in data
    assert "timestamp" in data


def test_config_endpoint(sync_client):
    response = sync_client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert "app_name" in data
    assert "default_llm_provider" in data


def test_status_endpoint(sync_client):
    response = sync_client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert "active_modules" in data
    assert "registered_tool_count" in data


async def test_shutdown_order(monkeypatch):
    """M3: voice + background services stop first, then persistence flushes, then keep-warm, then the LLM client."""
    import asyncio
    import sys
    import types
    from unittest.mock import AsyncMock, MagicMock

    import services
    from app import main as main_mod
    from brain.pipeline import brain
    from brain.task_state import tasks
    from core import connectivity
    from core.config import settings
    from core.llm import llm

    order = []

    def mark(name):
        return lambda *a, **k: order.append(name)

    monkeypatch.setattr(settings, "BACKGROUND_SERVICES", True)
    monkeypatch.setattr(services, "start_background_services", AsyncMock())
    monkeypatch.setattr(services, "stop_background_services", AsyncMock(side_effect=mark("services")))
    monkeypatch.setattr(connectivity, "start_background_refresh", lambda: None)
    monkeypatch.setattr(llm, "warm_up", AsyncMock())

    async def warm_loop():
        try:
            await asyncio.sleep(3600)
        finally:
            order.append("keepwarm_done")

    monkeypatch.setattr(llm, "start_keepwarm",
                        lambda interval=50.0: setattr(llm, "_keepwarm_task", asyncio.get_running_loop().create_task(warm_loop())))
    monkeypatch.setattr(llm, "aclose", AsyncMock(side_effect=mark("llm_aclose")))
    monkeypatch.setattr(brain, "flush_persistence", AsyncMock(side_effect=mark("flush_persistence")))
    monkeypatch.setattr(tasks, "flush", AsyncMock(side_effect=mark("tasks_flush")))
    voice = MagicMock(is_running=True)
    voice.stop = AsyncMock(side_effect=mark("voice"))
    monkeypatch.setitem(sys.modules, "voice.voice_manager", types.SimpleNamespace(voice_manager=voice))

    async with main_mod.lifespan(main_mod.app):
        await asyncio.sleep(0)
    assert order == ["voice", "services", "flush_persistence", "tasks_flush", "keepwarm_done", "llm_aclose"]
