"""Tests for the API endpoints."""

import pytest
from httpx import ASGITransport, AsyncClient

from src.api.app import app


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_check(self, client):
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.2.0"
        assert "model_status" in data
        assert data["model_status"] in ("untrained", "training", "trained", "degraded", "failed")


class TestDetectEndpoint:
    @pytest.mark.asyncio
    async def test_detect_returns_result(self, client):
        payload = {
            "node_id": "node-001",
            "timestamp": "2026-07-14T20:00:00Z",
            "metrics": {
                "cpu_usage": 45.0,
                "memory_usage": 62.0,
                "latency_ms": 15.0,
                "packet_loss": 0.01,
            },
        }
        resp = await client.post("/api/v1/detect", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "node_id" in data
        assert "is_anomaly" in data
        assert "severity" in data
        assert "model_status" in data


class TestStatsEndpoint:
    @pytest.mark.asyncio
    async def test_stats_returns_structure(self, client):
        resp = await client.get("/api/v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_nodes" in data
        assert "healing_success_rate" in data
        assert "model_status" in data
