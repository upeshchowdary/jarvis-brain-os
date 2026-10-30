"""API security: token auth everywhere except /health, localhost bind, CORS closed, dry-run honoured."""

from unittest.mock import MagicMock

import pytest
from fastapi.routing import APIRoute

from app.main import app
from automation.gateway import gateway
from core.config import Settings, settings

PUBLIC = {"/health", "/dashboard"}


def _walk(routes):
    for r in routes:
        if isinstance(r, APIRoute):
            yield r
        elif hasattr(r, "original_router"):  # FastAPI >= 0.16x wraps included routers
            yield from _walk(r.original_router.routes)
        elif hasattr(r, "routes"):
            yield from _walk(r.routes)


def _protected_routes():
    for r in _walk(app.routes):
        if r.path not in PUBLIC:
            for method in r.methods:
                yield method, r.path.replace("{pending_id}", "abc").replace("{name}", "x")


def test_route_discovery_found_routes():
    assert len(set(_protected_routes())) >= 20


@pytest.mark.parametrize("method,path", sorted(set(_protected_routes())))
def test_every_route_requires_token(anon_client, method, path):
    res = anon_client.request(method, path, json={})
    assert res.status_code == 401, (method, path, res.status_code)


def test_wrong_token_rejected(anon_client):
    assert anon_client.get("/config", headers={"Authorization": "Bearer nope"}).status_code == 401


def test_health_is_public(anon_client):
    res = anon_client.get("/health")
    assert res.status_code == 200 and "active_provider" in res.json()
    assert "test-token" not in res.text


def test_default_bind_is_localhost(monkeypatch):
    monkeypatch.delenv("HOST", raising=False)
    assert Settings(_env_file=None).HOST == "127.0.0.1"


def test_cors_rejects_unknown_origins(anon_client):
    res = anon_client.options("/chat", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"})
    assert "access-control-allow-origin" not in {k.lower() for k in res.headers}


def test_config_reports_real_provider(sync_client):
    data = sync_client.get("/config").json()
    assert data["default_llm_provider"] == settings.DEFAULT_MODEL.split("/")[0]
    assert data["default_model"] == settings.DEFAULT_MODEL.split("/", 1)[1]


def test_automation_execute_dry_run_never_executes(sync_client, monkeypatch):
    import automation.ops.system as sysops

    monkeypatch.setattr(sysops, "_IS_WINDOWS", True)
    spy = MagicMock(return_value=True)
    monkeypatch.setattr(sysops.impl, "restart_system", spy)
    calls = []
    real_execute = gateway.execute

    async def recording_execute(action, session_id="default", **kw):
        calls.append(kw.get("dry_run"))
        return await real_execute(action, session_id, **kw)

    monkeypatch.setattr(gateway, "execute", recording_execute)
    res = sync_client.post("/automation/execute", json={"command": "restart my pc", "dry_run": True})
    body = res.json()
    assert res.status_code == 200 and body["status"] == "dry_run" and body["dry_run"] is True
    assert calls == [True]
    spy.assert_not_called()


def test_confirm_endpoint_flow(sync_client, monkeypatch):
    import automation.ops.system as sysops

    monkeypatch.setattr(sysops, "_IS_WINDOWS", True)
    spy = MagicMock(return_value=True)
    monkeypatch.setattr(sysops.impl, "shutdown_system", spy)
    gateway.resume()
    first = sync_client.post("/automation/execute", json={"command": "shutdown the pc", "session_id": "api-confirm"}).json()
    assert first["status"] == "needs_confirmation"
    pid = first["pending"]["pending_id"]
    assert any(p["pending_id"] == pid for p in sync_client.get("/automation/pending").json()["pending"])
    spy.assert_not_called()
    done = sync_client.post(f"/automation/confirm/{pid}").json()
    assert done["status"] == "executed"
    spy.assert_called_once()
    assert sync_client.post(f"/automation/confirm/{pid}").status_code == 404


def test_rate_limit(sync_client, monkeypatch):
    from app.api.security import rate_limiter

    monkeypatch.setattr(settings, "RATE_LIMIT_PER_MINUTE", 3)
    rate_limiter._hits.clear()
    codes = [sync_client.get("/config").status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200] and 429 in codes[3:]
    rate_limiter._hits.clear()


def test_metrics_endpoint(sync_client):
    data = sync_client.get("/metrics").json()
    assert "latency" in data and "counters" in data
