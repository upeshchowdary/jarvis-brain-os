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
